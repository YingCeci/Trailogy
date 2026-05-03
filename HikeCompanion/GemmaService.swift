// GemmaService.swift
// Wraps mlx-swift-lm's `ChatSession` over Gemma 4 E2B (INT4 quantized,
// ~3.5 GB). Loads from a bundled directory at `Bundle/Models/Gemma/`,
// exposes streaming + batch response APIs.
//
// Model files come from `mlx-community/gemma-4-e2b-it-4bit` (or unsloth's
// UD-MLX-4bit fallback) via `scripts/fetch-gemma.sh`.

import Foundation
import HuggingFace
import Hub
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@MainActor
final class GemmaService: ObservableObject {

    // MARK: - Published state

    @Published private(set) var status: String = "Idle"
    @Published private(set) var isReady: Bool = false
    @Published private(set) var loadProgress: Double = 0

    // MARK: - Internals

    private var session: ChatSession?

    private let systemInstructions = """
    You are a friendly outdoor companion who helps hikers understand what they \
    see — geology, plants, animals, weather, and climate change. Keep responses \
    brief and conversational: 2 to 4 short sentences. Speak as if narrating, \
    not as if writing a report.
    """

    // MARK: - Lifecycle

    init() {
        Task { await loadAsync() }
    }

    private func loadAsync() async {
        // xcodegen's `type: folder` + `buildPhase: resources` flattens the
        // contents of Resources/Models/ into the .app bundle ROOT — files
        // end up alongside HikeCompanion.app/Info.plist etc., not under a
        // Models/Gemma/ subdirectory. mlx-swift-lm's loadModelContainer(from:)
        // needs a directory containing config.json + model.safetensors +
        // tokenizer.json, so we point it at the bundle root.
        //
        // Kokoro's safetensors and voices.npz live at the same root with
        // different names — they don't collide with mlx-swift-lm's loader.
        let modelDir = Bundle.main.bundleURL
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path
        ) else {
            status = "Gemma model missing — run scripts/fetch-gemma.sh, then bash scripts/generate-project.sh, then rebuild."
            return
        }

        status = "Loading Gemma 4 (10–30 s)…"
        do {
            let container = try await loadModelContainer(
                from: modelDir,
                using: #huggingFaceTokenizerLoader()
            )
            session = ChatSession(
                container,
                instructions: systemInstructions,
                generateParameters: GenerateParameters(temperature: 0.7)
            )
            isReady = true
            status = "Gemma 4 ready"
        } catch {
            status = "Load error: \(error.localizedDescription)"
        }
    }

    // MARK: - Inference

    /// Stream Gemma's response token-by-token. Returns nil if session not ready.
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error>? {
        session?.streamResponse(to: prompt)
    }

    /// Block until full response is generated. Returns nil if session not ready.
    func respond(to prompt: String) async throws -> String? {
        guard let session else { return nil }
        return try await session.respond(to: prompt)
    }
}
