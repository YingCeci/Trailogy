// DebugView.swift
// The original ContentView UI, preserved verbatim as a debug screen.
// This is what we used to develop and verify Phases 1-2 (typed text →
// Gemma → Kokoro, and hold-to-speak voice input). Kept intact so we can
// fall back to it whenever the new UI hides something we need to inspect:
// raw status strings, memory snapshots, the standalone "speak only"
// debug TTS, voice/speed pickers, conversation reset.
//
// Wired to the same @StateObject services as the main UI via
// @EnvironmentObject. No service-layer behavior changes.

import MLX
import SwiftUI
import UIKit

struct DebugView: View {
    @EnvironmentObject var gemma: GemmaService
    @EnvironmentObject var tts: ValidationRunner
    @EnvironmentObject var speech: SpeechRecognizer
    @EnvironmentObject var router: AppRouter
    @EnvironmentObject var rag: RAGService

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

                    HStack {
                        HoldToSpeakButton(
                            isRecording: speech.isRecording,
                            isEnabled: speech.isAuthorized && !isAsking && gemma.status != "Loading Gemma 4 (10–30 s)…",
                            onPress: { startRecording() },
                            onRelease: { holdReleased() }
                        )

                        Spacer()

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

                ragSubjectsSection

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
            .navigationTitle("Debug")
        }
    }

    // MARK: - RAG subjects picker

    /// Override or fall back to per-trail defaults. Each toggle drives
    /// `router.ragSubjectsOverride` directly; `Reset to per-trail
    /// default` clears the override and the next tour reads
    /// `Trail.defaultRAGSubjects` instead. Toggling any subject from
    /// the default state automatically activates the override.
    private var ragSubjectsSection: some View {
        Section {
            ForEach(RAGService.Subject.allCases) { subject in
                Toggle(subject.displayName, isOn: bindingForSubject(subject))
            }

            HStack {
                Text(modeLabel)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Reset to per-trail default") {
                    router.ragSubjectsOverride = nil
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(router.ragSubjectsOverride == nil)
            }
        } header: {
            Text("RAG context")
        } footer: {
            Text("Picks which subject corpora to query during Ask. Default per trail: \(trailDefaultLabel). Toggling any switch activates the override for every tour until Reset.")
                .font(.caption2)
        }
    }

    /// Two-way binding the Toggle drives. Reading: yes if the subject
    /// is in the effective set (override if set, else trail default).
    /// Writing: copies the current effective set into the override (if
    /// not already) and adds/removes the subject.
    private func bindingForSubject(_ subject: RAGService.Subject) -> Binding<Bool> {
        Binding(
            get: { effectiveSubjects.contains(subject) },
            set: { isOn in
                var s = effectiveSubjects
                if isOn { s.insert(subject) } else { s.remove(subject) }
                router.ragSubjectsOverride = s
            }
        )
    }

    /// Subjects currently in effect for the picker's selected trail —
    /// override if the user has touched anything, otherwise the
    /// trail's curator-authored default.
    private var effectiveSubjects: Set<RAGService.Subject> {
        router.resolvedRAGSubjects(for: router.currentTrail)
    }

    private var modeLabel: String {
        router.ragSubjectsOverride == nil ? "Mode: trail default" : "Mode: override"
    }

    private var trailDefaultLabel: String {
        let defaults = router.currentTrail.defaultRAGSubjects
        return defaults.isEmpty ? "(none)" : defaults.joined(separator: " + ")
    }

    private func markMemoryEvent(_ label: String) {
        let stats = MemoryStats.current()
        memorySnapshot = stats
        memoryEvents.append((label: label, stats: stats))
    }

    private func startRecording() {
        do {
            try speech.startRecording()
        } catch {
            // Permission / availability errors already surface in
            // speech.status via SpeechRecognizer; nothing further to do.
        }
    }

    private func holdReleased() {
        speech.stopRecording()
        Task {
            try? await Task.sleep(for: .milliseconds(600))
            let text = speech.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }
            await MainActor.run {
                self.question = text
                self.ask()
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
                // DebugView is text-only — no image input here. The
                // image path lives in WalkingView (Phase 3b). We always
                // load the text loader so we don't inflate memory while
                // someone's just kicking the tires.
                try await gemma.loadIfNeeded(.text)
                markMemoryEvent("Ask: after Gemma load")

                guard let stream = gemma.streamResponse(prompt: prompt) else {
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

                gemma.unload()
                markMemoryEvent("Ask: after Gemma unload")

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

// MARK: - HoldToSpeakButton (Form-styled, used in DebugView)

/// Press-and-hold button that fires `onPress` when the finger lands and
/// `onRelease` when it lifts. Uses a zero-distance `DragGesture` because
/// SwiftUI's `Button` and `LongPressGesture` don't expose a clean
/// "press began" hook. Form-row version (small, inline).
struct HoldToSpeakButton: View {
    let isRecording: Bool
    let isEnabled: Bool
    let onPress: () -> Void
    let onRelease: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: isRecording ? "mic.fill" : "mic")
            Text(isRecording ? "Listening…" : "Hold to speak")
                .fontWeight(.semibold)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isRecording ? Color.red.opacity(0.18) : Color.gray.opacity(0.18))
        )
        .foregroundStyle(isRecording ? Color.red : (isEnabled ? Color.primary : Color.secondary))
        .opacity(isEnabled ? 1.0 : 0.5)
        .scaleEffect(isRecording ? 1.04 : 1.0)
        .animation(.easeInOut(duration: 0.12), value: isRecording)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard isEnabled, !isRecording else { return }
                    onPress()
                }
                .onEnded { _ in
                    guard isRecording else { return }
                    onRelease()
                }
        )
    }
}

#Preview {
    DebugView()
        .environmentObject(GemmaService())
        .environmentObject(ValidationRunner())
        .environmentObject(SpeechRecognizer())
}
