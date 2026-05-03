// ContentView.swift
// Phase 1 flow: type a question → Gemma 4 streams the response →
// finished response goes through Kokoro → spoken aloud.
// Also has a separate "Speak only" text field that bypasses Gemma — used
// to verify Kokoro works in isolation when the full pipeline crashes.

import MLX
import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var gemma = GemmaService()
    @StateObject private var tts = ValidationRunner()
    @StateObject private var speech = SpeechRecognizer()

    @State private var question: String = "What's a hemlock tree?"
    @State private var streamingText: String = ""
    @State private var isAsking: Bool = false
    @State private var speed: Double = 1.0
    @State private var directSpeakText: String = "Listen carefully to the sounds around you."
    @State private var memorySnapshot: MemoryStats = .current()
    @State private var memoryEvents: [(label: String, stats: MemoryStats)] = []

    var body: some View {
        NavigationStack {
            Form {
                Section("Status") {
                    Text("Kokoro: \(tts.status)")
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                    Text("Gemma: \(gemma.status)")
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                }

                Section("Ask Gemma") {
                    TextField("Question", text: $question, axis: .vertical)
                        .lineLimit(1...4)
                        .textFieldStyle(.roundedBorder)

                    // Voice-input row: tap to start/stop on-device ASR.
                    // Recognized text replaces the question field on stop.
                    HStack {
                        Button {
                            toggleRecording()
                        } label: {
                            HStack {
                                Image(systemName: speech.isRecording ? "stop.circle.fill" : "mic.fill")
                                Text(speech.isRecording ? "Stop" : "Tap to speak")
                            }
                        }
                        .buttonStyle(.bordered)
                        .tint(speech.isRecording ? .red : .accentColor)
                        .disabled(!speech.isAuthorized || isAsking)

                        Text(speech.status)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    if speech.isRecording && !speech.transcript.isEmpty {
                        Text(speech.transcript)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

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
                    // Gemma is lazy-loaded on first tap; only require Kokoro
                    // to be ready (so we can speak the response) and that we
                    // have a question and aren't already running.
                    .disabled(!tts.isReady || isAsking || question.isEmpty)

                    HStack {
                        Text("Conversation: \(gemma.historyTurnCount) turn\(gemma.historyTurnCount == 1 ? "" : "s")")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Reset") { gemma.reset() }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(gemma.historyTurnCount == 0 || isAsking)
                    }
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

                // Debug helper: speak text directly (no Gemma in the loop).
                // Useful when the full Ask pipeline crashes — confirms Kokoro
                // is healthy in isolation.
                Section("Speak directly (debug)") {
                    TextField("Text", text: $directSpeakText, axis: .vertical)
                        .lineLimit(1...3)
                        .textFieldStyle(.roundedBorder)
                    Button("Speak only (no Gemma)") {
                        markMemoryEvent("Before Speak only")
                        tts.synthesize(text: directSpeakText, speed: Float(speed))
                    }
                    .buttonStyle(.bordered)
                    .disabled(!tts.isReady || tts.isRunning)
                }

                Section("Memory") {
                    Text(memorySnapshot.summary)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Refresh memory snapshot") {
                        memorySnapshot = .current()
                    }
                    .buttonStyle(.bordered)
                    if !memoryEvents.isEmpty {
                        ForEach(memoryEvents.indices, id: \.self) { i in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(memoryEvents[i].label)
                                    .font(.caption2.weight(.semibold))
                                Text(memoryEvents[i].stats.summary)
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Button("Clear events") {
                            memoryEvents.removeAll()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }
            .navigationTitle("HikeCompanion")
        }
    }

    /// Snapshot current memory and append a labeled entry to the events log.
    /// Also refreshes the headline `memorySnapshot` view.
    private func markMemoryEvent(_ label: String) {
        let stats = MemoryStats.current()
        memorySnapshot = stats
        memoryEvents.append((label: label, stats: stats))
    }

    /// Toggle voice recording. On stop, copy the transcript into the
    /// Question field. User reviews, then taps Ask.
    private func toggleRecording() {
        if speech.isRecording {
            speech.stopRecording()
            // Brief delay so the final partial result lands before we read it.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                let text = speech.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    self.question = text
                }
            }
        } else {
            do {
                try speech.startRecording()
            } catch {
                // Errors surface in speech.status via the recognizer itself
                // when possible; surface unexpected ones here too.
                print("Speech start error: \(error.localizedDescription)")
            }
        }
    }

    private func ask() {
        let prompt = question
        streamingText = ""
        isAsking = true
        markMemoryEvent("Ask: start")

        Task {
            do {
                // 1. Load Gemma into memory (lazy — first ask, or after
                //    a previous unload). UI shows "Loading Gemma 4…".
                try await gemma.loadIfNeeded()
                markMemoryEvent("Ask: after Gemma load")

                // 2. Stream the response.
                guard let stream = gemma.streamResponse(to: prompt) else {
                    streamingText = "[error: Gemma session not ready]"
                    isAsking = false
                    return
                }
                var fullText = ""
                for try await chunk in stream {
                    fullText += chunk
                    streamingText = fullText
                }
                markMemoryEvent("Ask: after generation")

                // 3. UNLOAD Gemma before Kokoro starts. Keeping it resident
                //    across turns OOM'd the device — even on iPhone 17 Pro,
                //    the combined Gemma weights + KV cache + Kokoro working
                //    set crossed the iOS jetsam line.
                //
                //    Conversation history lives in GemmaService and survives
                //    unload — it's replayed into a fresh ChatSession on the
                //    next Ask. Trade-off: each follow-up Ask pays the 10–30 s
                //    reload again. Multi-turn coherence preserved; memory
                //    bounded.
                gemma.unload()
                markMemoryEvent("Ask: after Gemma unload")

                // 4. Speak the response.
                if !fullText.isEmpty {
                    tts.synthesize(text: fullText, speed: Float(speed))
                }
                isAsking = false
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
