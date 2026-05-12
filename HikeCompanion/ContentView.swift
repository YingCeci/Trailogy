// ContentView.swift
// Root view. Owns the three @StateObject services
// (Gemma, Kokoro/ValidationRunner, SpeechRecognizer) and the AppRouter,
// and switches between the four main screens with a fade transition.
//
// All screen-specific UI lives in HikeCompanion/Views/*.swift.
// The original "develop everything in one file" UI (Form-based debug
// console with memory profiler, Speak-only TTS, voice/speed pickers,
// conversation reset) is preserved verbatim in `Views/DebugView.swift`
// and surfaced as a sheet from PickerView's bug icon.
//
// Mockup: design/mockups.html (Nature companion). The view-switch
// state machine is mirrored in `AppRouter`.

import SwiftUI

struct ContentView: View {
    @StateObject private var gemma = GemmaService()
    @StateObject private var tts = ValidationRunner()
    @StateObject private var speech = SpeechRecognizer()
    @StateObject private var rag = RAGService()
    @StateObject private var router = AppRouter()

    /// iOS scene state — used to gate the RAG preload below.
    @Environment(\.scenePhase) private var scenePhase

    /// One-shot latch so preload runs exactly once across the app
    /// lifetime, the FIRST time the scene reaches `.active`.
    @State private var didPreloadRAG = false

    var body: some View {
        ZStack {
            switch router.screen {
            case .picker:
                PickerView()
                    .transition(.opacity)
            case .detail:
                DetailView()
                    .transition(.opacity)
            case .walking:
                WalkingView()
                    .transition(.opacity)
            case .journal:
                JournalView()
                    .transition(.opacity)
            }
        }
        .preferredColorScheme(.dark)
        .environmentObject(router)
        .environmentObject(gemma)
        .environmentObject(tts)
        .environmentObject(speech)
        .environmentObject(rag)
        // Preload the bundled MiniLM embedder, but ONLY after the
        // scene reaches `.active`. Doing this in a bare `.task`
        // modifier at view-appear time races with iOS's "prewarming"
        // phase — the app exists but isn't yet active, and Metal
        // command buffer submissions from that state are rejected
        // with `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`,
        // which surfaces as a C++ std::runtime_error from MLX that
        // Swift cannot catch (process terminates).
        //
        // `.task(id: scenePhase)` re-runs the body whenever scenePhase
        // changes, INCLUDING the initial value at view appearance —
        // unlike `.onChange`, which only fires on transitions. The
        // body is gated on `.active` and uses `didPreloadRAG` as a
        // one-shot latch so the model loads exactly once across the
        // app lifetime (and stays loaded across foreground/background
        // cycles — it's only ~87 MB).
        .task(id: scenePhase) {
            guard scenePhase == .active, !didPreloadRAG else { return }
            didPreloadRAG = true
            do {
                try await rag.preload()
            } catch {
                print("[RAG] preload failed: \(error.localizedDescription) — retrieval will retry on first use")
            }
        }
        .sheet(isPresented: $router.debugVisible) {
            DebugView()
                .environmentObject(gemma)
                .environmentObject(tts)
                .environmentObject(speech)
                .environmentObject(rag)
        }
    }
}

#Preview {
    ContentView()
}
