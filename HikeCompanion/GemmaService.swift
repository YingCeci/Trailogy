// GemmaService.swift
// Wraps mlx-swift-lm's `ChatSession` over Gemma 4 E2B (INT4 quantized,
// ~2.8 GB on disk after audio-tower strip).
//
// DUAL-MODE LOADER (Phase 3b)
// ---------------------------
// We can load Gemma in two flavours:
//
//   .text  → MLXLLM.LLMModelFactory  (filters vision + audio at sanitize() →
//            ~2.5 GB MLX active, ~10–30 s cold load)
//   .vlm   → MLXVLM.VLMModelFactory  (filters audio only; keeps vision tower →
//            ~3.2 GB MLX active, ~13–37 s cold load)
//
// Each Ask calls `loadIfNeeded(kind)` with the kind appropriate for the
// turn (text-only vs. has-photo). If a different kind is currently
// loaded, we unload it first. After the response streams + Kokoro
// speaks, we unload to bring memory back to ~100 MB idle, same as the
// text-only flow we shipped earlier.
//
// LIFECYCLE
// ---------
//   • Lazy load on first Ask. App launches with only Kokoro resident.
//   • Unloaded after each generation completes (caller's responsibility).
//   • Conversation history is preserved as plain text across loads, so a
//     follow-up text Ask after an image Ask still has context — Gemma
//     just doesn't see the image bytes again, only the text it produced
//     last turn.
//   • `reset()` wipes history without unloading.
//
// COST
// ----
//   • Each Ask after the first pays a 10–37 s reload again. Multi-turn
//     coherence preserved; memory bounded between turns.
//
// HISTORY (capped at 20 messages = 10 turns)
//   • Every (user, assistant) pair appended after stream completion.
//   • Replay during prefill keeps the KV cache transient. At ~100
//     tokens/turn for Q&A, prefill is ~3–5 s on iPhone — small compared
//     to the load.

import CoreImage
import Foundation
import HuggingFace
import Hub
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers
import UIKit

@MainActor
final class GemmaService: ObservableObject {

    // MARK: - Loaded kind

    enum LoadedKind: String, Equatable {
        case text   // MLXLLM Gemma 4 (text-only path, ~2.5 GB MLX active)
        case vlm    // MLXVLM Gemma 4 (text + vision tower, ~3.2 GB MLX active)
    }

    // MARK: - Published state

    @Published private(set) var status: String = "Idle (Gemma loads on first Ask)"
    @Published private(set) var isLoaded: Bool = false
    @Published private(set) var historyTurnCount: Int = 0
    @Published private(set) var loadedKind: LoadedKind?

    // MARK: - Internals

    private var modelContainer: ModelContainer?
    private var conversationHistory: [Chat.Message] = []

    /// Cap on stored history. Each (user, assistant) pair is 2 messages,
    /// so 6 = 3 turns of context. Past this, oldest messages drop off.
    /// We deliberately keep this short — every replayed turn re-prefills
    /// through the model and inflates KV cache footprint. 3 turns is
    /// enough conversational coherence for a tour-companion app
    /// ("what about the bark on that tree?" → "the maple I just
    /// described") without paying a memory hit.
    private let maxHistoryMessages = 6

    /// VLM asks already carry a large image-token block (~196 tokens
    /// for a 14×14 patch grid). Replaying any conversation history on
    /// top of that pushes prefill activations dangerously close to the
    /// jetsam line. Drop history entirely for image asks — the photo
    /// is the context, not prior chat. (For continuation: bump back to
    /// 2 once we've validated the basic image path is stable.)
    private let maxImageHistoryMessages = 0

    /// Always-on persona instructions. Kept short — the bulk of the
    /// effective system prompt is the trail block composed below.
    private let baseInstructions = """
    You are a friendly outdoor companion who helps hikers understand what they \
    see — geology, plants, animals, weather, and climate change. Keep responses \
    brief and conversational: 2 to 4 short sentences. Speak as if narrating, \
    not as if writing a report. Remember earlier turns of this conversation \
    when answering follow-up questions.
    """

    /// Image-specific guidance, only injected on VLM turns. Encourages a
    /// confident best-guess with a visible cue, rather than the model's
    /// default "I'm not sure, it could be many things" hedge that we
    /// were getting on tree-ID asks.
    private let imageInstructions = """
    When the user shares a photo, identify the most likely subject and ground \
    your answer in what's visible. If you're not certain of a species, give \
    your best guess and one observable cue (leaf shape, bark pattern, color, \
    size). Don't refuse — even a confident "looks like a … because of …" is \
    more useful than vague hedging.
    """

    /// Composed once per `setActiveContext` call; injected into the
    /// system instructions for every Ask until cleared. Keeps the model
    /// grounded in the active trail rather than answering as a generic
    /// outdoor companion floating in the void.
    private var trailContextBlock: String = ""

    /// Per-turn framing prepended to each user prompt. Updates as the
    /// user advances between stops, so an answer to "what's that mossy
    /// stuff?" reflects where they are RIGHT NOW. Stripped from saved
    /// history (we don't want stale stop framings polluting future
    /// turns — only the user's actual question is preserved).
    private var stopContextBlock: String = ""

    /// Mobile memory budget for both text and VLM asks. Sized to fit
    /// what the worst real path actually needs, not a generous slop:
    ///
    ///   • VLM:  196 image + ~330 system + ~100 user + 160 gen ≈ 790
    ///   • Text: ~330 system + ~540 history (3 turns) + ~100 user
    ///           + 160 gen ≈ 1130
    ///
    /// 1280 covers both with ~150-token headroom. Bumping any higher
    /// trades real footprint (the KV cache is allocated by maxKVSize
    /// regardless of how full it actually is — going from 768 to 2048
    /// during the trail-context work cost ~1 GB peak and pushed VLM
    /// runs uncomfortably close to jetsam on iPhone).
    private let generationParameters = GenerateParameters(
        maxTokens: 160,
        maxKVSize: 1280,
        temperature: 0.7,
        prefillStepSize: 128
    )

    // MARK: - Lifecycle

    /// Load the model into memory in the requested mode. Idempotent if
    /// the requested kind matches what's already loaded; otherwise
    /// unloads the existing model first.
    func loadIfNeeded(_ kind: LoadedKind) async throws {
        if loadedKind == kind, modelContainer != nil { return }
        if modelContainer != nil { unload() }

        let modelDir = Bundle.main.bundleURL
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma")
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path
        ) else {
            throw GemmaError.modelMissing
        }

        switch kind {
        case .text:
            MemoryStats.log("gemma.load start (text)")
            status = "Loading Gemma 4 text (10–30 s)…"
            modelContainer = try await LLMModelFactory.shared.loadContainer(
                from: modelDir,
                using: #huggingFaceTokenizerLoader()
            )
        case .vlm:
            MemoryStats.log("gemma.load start (vlm)")
            status = "Loading Gemma 4 multimodal (13–37 s)…"
            modelContainer = try await VLMModelFactory.shared.loadContainer(
                from: modelDir,
                using: #huggingFaceTokenizerLoader()
            )
        }

        loadedKind = kind
        isLoaded = true
        status = "Gemma 4 loaded (\(kind.rawValue))"
        MemoryStats.log("gemma.load done (\(kind.rawValue))")
    }

    /// Drop the model from memory. Conversation history is preserved.
    func unload() {
        modelContainer = nil
        isLoaded = false
        loadedKind = nil
        Memory.clearCache()
        status = "Gemma unloaded (history kept; next Ask will reload)"
        MemoryStats.log("gemma.unload done")
    }

    /// Wipe the conversation history. Does not unload the model.
    func reset() {
        conversationHistory.removeAll()
        historyTurnCount = 0
        status = isLoaded ? "Gemma 4 loaded · history reset" : status
    }

    // MARK: - Active context (trail + current stop)

    /// Set the active trail and current-stop framing. Call once on tour
    /// start with `stopIdx: 0`, and again whenever the user advances to
    /// a new stop. Pass `trail: nil` to clear (e.g. on tour end / view
    /// disappear).
    ///
    /// The trail block goes into the system prompt (static across the
    /// conversation). The stop block is prepended to each user prompt
    /// as a `[Currently at Stop X of Y: ...]` framing — so answers
    /// shift as the user moves without us re-spending system-prompt
    /// tokens or invalidating the model's KV cache mid-tour.
    func setActiveContext(trail: Trail?, stopIdx: Int?) {
        guard let trail else {
            trailContextBlock = ""
            stopContextBlock = ""
            return
        }

        let highlights = trail.stops.map(\.name).joined(separator: ", ")
        let distance = trail.distanceMiles == floor(trail.distanceMiles)
            ? String(format: "%.0f", trail.distanceMiles)
            : String(format: "%.1f", trail.distanceMiles)

        trailContextBlock = """
        TODAY'S TRAIL: \(trail.name) at \(trail.parkLocation). \
        \(distance)-mile \(trail.difficulty.lowercased()) loop, about \
        \(trail.durationMinutes) minutes. Stops along the way: \(highlights).

        REGION: \(trail.regionalContext)
        """

        if let stopIdx, let stop = trail.stops[safe: stopIdx] {
            stopContextBlock = "[Currently at Stop \(stop.number) of \(trail.stops.count): \(stop.name). \(stop.spokenNarration)]"
        } else {
            stopContextBlock = ""
        }
    }

    /// System instructions for THIS turn = base + (image guidance if
    /// VLM) + trail context block. Composed fresh each Ask so it
    /// reflects the latest `setActiveContext` and per-turn modality.
    private func composedSystemInstructions(forImage hasImage: Bool) -> String {
        var parts: [String] = [baseInstructions]
        if hasImage { parts.append(imageInstructions) }
        if !trailContextBlock.isEmpty { parts.append(trailContextBlock) }
        return parts.joined(separator: "\n\n")
    }

    // MARK: - Inference

    /// Stream Gemma's response. If `image` is non-nil the caller must
    /// have called `loadIfNeeded(.vlm)` first (or this method will
    /// return nil). Conversation history is replayed each call.
    ///
    /// On stream completion, the (prompt, full response) pair is appended
    /// to history, capped at `maxHistoryMessages`.
    func streamResponse(prompt: String, image: UIImage? = nil)
        -> AsyncThrowingStream<String, Error>?
    {
        guard let container = modelContainer else { return nil }

        // Validate mode/kind/image alignment.
        if image != nil, loadedKind != .vlm {
            // Caller mismatched kinds — the text loader has no vision
            // tower so encoding the image would crash inside MLX.
            return nil
        }

        // Convert UIImage → UserInput.Image.ciImage for the VLM path.
        let imageInputs: [UserInput.Image]
        if let image, let ci = Self.ciImage(from: image) {
            imageInputs = [.ciImage(ci)]
        } else {
            imageInputs = []
        }

        // Snapshot history. Image asks have a much larger prompt because
        // Gemma4 expands each image into vision tokens, so use a smaller
        // replay window on that path.
        let historySnapshot = imageInputs.isEmpty
            ? conversationHistory
            : Array(conversationHistory.suffix(maxImageHistoryMessages))

        // Compose the system prompt once per turn so it reflects:
        //   • the active trail (set via setActiveContext)
        //   • whether this turn is text-only or VLM (different guidance)
        let composedInstructions = composedSystemInstructions(
            forImage: !imageInputs.isEmpty
        )

        // Sandwich the user's prompt with the current stop framing if
        // we have one. Saved history keeps just `prompt` (without the
        // stop framing) — see persist block below.
        let composedPrompt = stopContextBlock.isEmpty
            ? prompt
            : "\(stopContextBlock)\n\n\(prompt)"

        let session = ChatSession(
            container,
            instructions: composedInstructions,
            history: historySnapshot,
            generateParameters: generationParameters
        )

        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                var fullText = ""
                do {
                    let label = imageInputs.isEmpty ? "text" : "image"
                    MemoryStats.log("gemma.stream start (\(label))")
                    for try await chunk in session.streamResponse(
                        to: composedPrompt,
                        images: imageInputs,
                        videos: []
                    ) {
                        fullText += chunk
                        continuation.yield(chunk)
                    }
                    MemoryStats.log("gemma.stream done (\(label))")

                    // Persist the turn. Save the user's actual question
                    // (without the stop-framing prefix) so future turns
                    // aren't anchored to a stop the user has since left.
                    // The assistant's text response IS preserved in full
                    // and carries forward what the model "knew" when it
                    // answered.
                    self.conversationHistory.append(.init(role: .user, content: prompt))
                    self.conversationHistory.append(.init(role: .assistant, content: fullText))
                    while self.conversationHistory.count > self.maxHistoryMessages {
                        self.conversationHistory.removeFirst()
                    }
                    self.historyTurnCount = self.conversationHistory.count / 2
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - UIImage → CIImage helper

    /// Convert a `UIImage` into a `CIImage` suitable for
    /// `UserInput.Image.ciImage`. Falls back through the available
    /// representations: prefer the embedded CIImage if present, then
    /// the CGImage, otherwise nil. The captured photos in this app come
    /// from AVCaptureSession via UIImage(data: jpegData), so they
    /// always have a valid `cgImage` — the fallthrough is paranoia.
    private static func ciImage(from image: UIImage) -> CIImage? {
        if let ci = image.ciImage { return ci }
        if let cg = image.cgImage { return CIImage(cgImage: cg) }
        return nil
    }
}

enum GemmaError: LocalizedError {
    case modelMissing

    var errorDescription: String? {
        switch self {
        case .modelMissing:
            return "Gemma model missing — run scripts/fetch-gemma.sh, then bash scripts/generate-project.sh, then rebuild."
        }
    }
}
