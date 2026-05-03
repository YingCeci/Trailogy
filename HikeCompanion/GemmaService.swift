// GemmaService.swift
// Wraps mlx-swift-lm's `ChatSession` over Gemma 4 E2B (INT4 quantized,
// ~3.5 GB).
//
// LIFECYCLE: lazy load on first Ask, unload after generation completes.
// Reasons:
//   • At app launch we don't want Gemma's 3.5 GB resident — Kokoro alone
//     (700 MB) is enough to be useful for TTS-only flows.
//   • Loading Gemma in parallel with Kokoro at app start caused Kokoro's
//     own MLX inference to crash on first Speak Only tap (concurrent MLX
//     setup contention).
//   • Holding Gemma's weights resident while Kokoro starts TTS spiked
//     resident memory enough for iOS to jetsam the app.
//
// Trade-off: each Ask pays Gemma load time (~10–30 s on iPhone Pro) again
// — visible "Loading Gemma…" status. We accept this in exchange for
// bounded memory at the Gemma → Kokoro hand-off.

import Foundation
import HuggingFace
import Hub
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@MainActor
final class GemmaService: ObservableObject {

    // MARK: - Published state

    @Published private(set) var status: String = "Idle (Gemma loads on first Ask)"
    @Published private(set) var isLoaded: Bool = false

    // MARK: - Internals

    private var modelContainer: ModelContainer?

    private let systemInstructions = """
    You are a friendly outdoor companion who helps hikers understand what they \
    see — geology, plants, animals, weather, and climate change. Keep responses \
    brief and conversational: 2 to 4 short sentences. Speak as if narrating, \
    not as if writing a report.
    """

    // MARK: - Lifecycle

    /// Load the model into memory. Idempotent — no-op if already loaded.
    /// Throws if the bundled model files are missing.
    func loadIfNeeded() async throws {
        guard modelContainer == nil else { return }

        let modelDir = Bundle.main.bundleURL
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma")
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path
        ) else {
            throw GemmaError.modelMissing
        }

        status = "Loading Gemma 4 (10–30 s)…"
        modelContainer = try await loadModelContainer(
            from: modelDir,
            using: #huggingFaceTokenizerLoader()
        )
        isLoaded = true
        status = "Gemma 4 loaded"
    }

    /// Drop the model from memory and force MLX to release cached buffers.
    /// Call this between turns so Kokoro can run TTS without contending
    /// for ~3.5 GB of Gemma weights.
    func unload() {
        modelContainer = nil
        isLoaded = false
        Memory.clearCache()
        status = "Gemma unloaded (will reload on next Ask)"
    }

    // MARK: - Inference

    /// Stream Gemma's response token-by-token. Returns nil if not loaded;
    /// caller is expected to have run `loadIfNeeded()` first. Builds a
    /// fresh `ChatSession` per call so KV cache state doesn't accumulate.
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error>? {
        guard let container = modelContainer else { return nil }
        let session = ChatSession(
            container,
            instructions: systemInstructions,
            generateParameters: GenerateParameters(temperature: 0.7)
        )
        return session.streamResponse(to: prompt)
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
