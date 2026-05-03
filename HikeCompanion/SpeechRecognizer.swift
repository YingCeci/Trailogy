// SpeechRecognizer.swift
// Apple on-device ASR via SFSpeechRecognizer + AVAudioEngine input tap.
//
// Why Apple's framework instead of Whisper or Gemma 4 audio input:
//   • Free, no model to bundle (~0 MB added)
//   • Built-in, mature, well-tested
//   • Runs fully on-device with `requiresOnDeviceRecognition = true`
//   • Streams partial results so the UI can show a live transcript
//
// Trade-off: not as accurate as Whisper Large or Gemma 4 audio for
// rare words or accents. Fine for typical hike Q&A.
//
// Audio session: this service sets `.record` while listening, then
// switches back to `.playback` so Kokoro TTS can resume. Each user
// of the audio session sets its own category just before use.

import AVFoundation
import Foundation
import Speech

@MainActor
final class SpeechRecognizer: ObservableObject {

    // MARK: - Published state

    @Published private(set) var transcript: String = ""
    @Published private(set) var isRecording: Bool = false
    @Published private(set) var status: String = "Idle"
    @Published private(set) var isAuthorized: Bool = false

    // MARK: - Internals

    private let recognizer: SFSpeechRecognizer?
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    // MARK: - Lifecycle

    init() {
        // US English by default — matches af_bella + the rest of the
        // hike companion's expected locale.
        recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        Task { await requestAuthorization() }
    }

    private func requestAuthorization() async {
        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        let micGranted = await AVAudioApplication.requestRecordPermission()

        let authorized = (speechStatus == .authorized) && micGranted

        self.isAuthorized = authorized
        if !authorized {
            switch speechStatus {
            case .denied: status = "Speech recognition denied"
            case .restricted: status = "Speech recognition restricted"
            case .notDetermined: status = "Speech recognition not determined"
            default: break
            }
            if !micGranted {
                status = "Microphone access denied"
            }
        } else {
            status = "Idle"
        }
    }

    // MARK: - Recording

    func startRecording() throws {
        guard isAuthorized else {
            throw RecognizerError.notAuthorized
        }
        guard let recognizer, recognizer.isAvailable else {
            throw RecognizerError.unavailable
        }

        // Cancel any prior task.
        task?.cancel()
        task = nil

        // Switch session to record. We restore .playback after stop.
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        // Build the recognition request.
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        req.requiresOnDeviceRecognition = true
        request = req

        // Tap the input node and pipe audio buffers into the request.
        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.request?.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()

        isRecording = true
        transcript = ""
        status = "Listening…"

        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                if let result {
                    self.transcript = result.bestTranscription.formattedString
                }
                if error != nil || (result?.isFinal ?? false) {
                    self.cleanup()
                }
            }
        }
    }

    func stopRecording() {
        guard isRecording else { return }
        request?.endAudio()
        // SFSpeechRecognitionTask will fire one last result with isFinal=true,
        // which calls cleanup() above. But also tear down audio engine here
        // immediately so the mic light goes off.
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        isRecording = false
        status = "Stopping…"
    }

    private func cleanup() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        request = nil
        task = nil
        isRecording = false
        status = transcript.isEmpty ? "Idle (no speech detected)" : "Done"

        // Restore playback session for Kokoro.
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
        } catch {
            // Non-fatal; Kokoro will reset the category on its own play call.
        }
    }
}

enum RecognizerError: LocalizedError {
    case notAuthorized
    case unavailable

    var errorDescription: String? {
        switch self {
        case .notAuthorized:
            return "Microphone or speech recognition permission denied. Enable in Settings → HikeCompanion."
        case .unavailable:
            return "Speech recognizer not available on this device or locale."
        }
    }
}
