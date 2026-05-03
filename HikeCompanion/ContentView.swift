// ContentView.swift
// Phase 1 flow: type a question → Gemma 4 streams the response →
// finished response goes through Kokoro → spoken aloud.

import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var gemma = GemmaService()
    @StateObject private var tts = ValidationRunner()

    @State private var question: String = "What's a hemlock tree?"
    @State private var streamingText: String = ""
    @State private var isAsking: Bool = false
    @State private var speed: Double = 1.0

    var body: some View {
        NavigationStack {
            Form {
                Section("Status") {
                    Text("Gemma: \(gemma.status)")
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                    Text("Kokoro: \(tts.status)")
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                }

                Section("Ask Gemma") {
                    TextField("Question", text: $question, axis: .vertical)
                        .lineLimit(1...4)
                        .textFieldStyle(.roundedBorder)

                    Picker("Voice", selection: $tts.selectedVoice) {
                        ForEach(tts.voiceNames, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .disabled(tts.voiceNames.isEmpty)

                    HStack {
                        Text("Speed")
                        Slider(value: $speed, in: 0.5...2.0, step: 0.05)
                        Text(String(format: "%.2f×", speed))
                            .font(.callout.monospaced())
                            .frame(width: 60, alignment: .trailing)
                    }

                    Button {
                        ask()
                    } label: {
                        HStack {
                            if isAsking {
                                ProgressView().padding(.trailing, 6)
                            }
                            Text(isAsking ? "Thinking…" : "Ask")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!gemma.isReady || !tts.isReady || isAsking || question.isEmpty)
                }

                if !streamingText.isEmpty {
                    Section("Gemma's response") {
                        Text(streamingText)
                            .font(.body)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if !tts.currentCaption.isEmpty {
                    Section("Spoken so far") {
                        Text(tts.currentCaption)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if let r = tts.lastResult {
                    Section("Last TTS run") {
                        Text(String(format: "RTF %.3f   audio %.2f s   %d chunks",
                                    r.rtf, r.audioDurationSec, r.chunkCount))
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                        Button("Replay") { tts.playLastAgain() }
                            .buttonStyle(.bordered)
                    }
                }
            }
            .navigationTitle("HikeCompanion")
        }
    }

    private func ask() {
        let prompt = question
        streamingText = ""
        isAsking = true

        Task {
            guard let stream = gemma.streamResponse(to: prompt) else {
                streamingText = "[error: Gemma session not ready]"
                isAsking = false
                return
            }
            do {
                var fullText = ""
                for try await chunk in stream {
                    fullText += chunk
                    streamingText = fullText
                }
                isAsking = false
                // Phase 1: speak the full response in one go via the existing
                // chunked-TTS pipeline. Phase 1.5 will pipe at sentence
                // granularity so audio starts before Gemma finishes.
                if !fullText.isEmpty {
                    tts.synthesize(text: fullText, speed: Float(speed))
                }
            } catch {
                streamingText += "\n\n[stream error: \(error.localizedDescription)]"
                isAsking = false
            }
        }
    }
}

#Preview {
    ContentView()
}
