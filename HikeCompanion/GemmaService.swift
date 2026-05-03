// GemmaService.swift
// Wraps mlx-swift-lm's `ChatSession` over Gemma 4 E2B (INT4 quantized,
// ~3.5 GB).
//
// LIFECYCLE:
//   • Lazy load on first Ask. App launches with only Kokoro resident.
//   • Unloaded after each generation completes (caller's responsibility
//     to call `unload()`) — keeping Gemma resident while Kokoro does
//     TTS crashed the app even on iPhone 17 Pro.
//   • Conversation history is persisted in this service across
//     unload/reload cycles, so it survives the per-turn drop. On the
//     next Ask, the full history is replayed into a fresh ChatSession
//     so Gemma can resolve "they", "that", etc. across turns.
//   • `reset()` wipes history without unloading.
//
// COST:
//   • Each Ask after the first pays a 10–30 s reload again. Multi-turn
//     coherence preserved; memory bounded between turns.
//
// HISTORY (uncapped for now):
//   • Every (user, assistant) pair appended after stream completion.
//   • A long conversation will grow KV cache during generation since
//     we replay the whole history. Add a turn cap if it becomes an
//     issue. For typical hike-Q&A turns this should be fine.

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
    @Published private(set) var historyTurnCount: Int = 0

    // MARK: - Internals

    private var modelContainer: ModelContainer?
    private var conversationHistory: [Chat.Message] = []

    /// Cap on stored history. Each (user, assistant) pair is 2 messages,
    /// so 20 = 10 turns of context. Past this, oldest messages drop off.
    ///
    /// Trade-off: each follow-up Ask pays a prefill cost proportional to
    /// total history length. At 10 turns of typical Q&A (~100 tokens
    /// each) that's ~3–5 s of prefill on Gemma 4 E2B over MLX on iPhone
    /// — small compared to the 10–30 s Gemma reload that already
    /// dominates each Ask.
    ///
    /// Memory: prefill KV cache for 1000 tokens ≈ 115 KB × 1000 = 115 MB
    /// transient peak during prefill, on top of the ~2.5 GB Gemma weights.
    /// Fits comfortably under iPhone 17 Pro's jetsam ceiling.
    ///
    /// Earlier history cap of 4 (2 turns) was set before the Kokoro
    /// lazy-load fix freed ~500 MB at peak; the headroom now allows
    /// longer memory.
    private let maxHistoryMessages = 20

    private let systemInstructions = """
    You are a friendly outdoor companion who helps hikers understand what they \
    see — geology, plants, animals, weather, and climate change. Keep responses \
    brief and conversational: 2 to 4 short sentences. Speak as if narrating, \
    not as if writing a report. Remember earlier turns of this conversation \
    when answering follow-up questions.
    """

    // MARK: - Lifecycle

    /// Load the model into memory. Idempotent.
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

    /// Drop the model from memory. Use sparingly — reload costs 10–30 s.
    /// Conversation history is preserved across unload/reload.
    func unload() {
        modelContainer = nil
        isLoaded = false
        Memory.clearCache()
        status = "Gemma unloaded (history kept; next Ask will reload)"
    }

    /// Wipe the conversation history. Does not unload the model.
    func reset() {
        conversationHistory.removeAll()
        historyTurnCount = 0
        status = isLoaded ? "Gemma 4 loaded · history reset" : status
    }

    // MARK: - Inference

    /// Stream Gemma's response, with conversation history replayed so
    /// follow-ups can reference prior turns. Appends the (prompt, full
    /// response) pair to history after the stream completes.
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error>? {
        guard let container = modelContainer else { return nil }

        // Snapshot current history (already capped to last `maxHistoryMessages`);
        // ChatSession will consume it.
        let historySnapshot = conversationHistory

        let session = ChatSession(
            container,
            instructions: systemInstructions,
            history: historySnapshot,
            generateParameters: GenerateParameters(temperature: 0.7)
        )

        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                var fullText = ""
                do {
                    for try await chunk in session.streamResponse(to: prompt) {
                        fullText += chunk
                        continuation.yield(chunk)
                    }
                    // Persist the turn, then trim the buffer.
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
