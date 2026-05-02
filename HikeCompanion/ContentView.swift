// ContentView.swift
// SwiftUI validator UI: pickers for compute units + fixture, "Run" button,
// last-run summary, results list, and Play / Share for the most recent WAV.

import AVFoundation
import CoreML
import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var runner = ValidationRunner()
    @State private var fixtureKey: String = "3s"
    @State private var computeChoice: ComputeChoice = .all
    @State private var audioPlayer: AVAudioPlayer?
    @State private var showShareSheet = false
    @State private var shareItems: [Any] = []

    private let fixtures = ["3s", "7s", "15s", "30s"]

    var body: some View {
        NavigationStack {
            Form {
                Section("Configuration") {
                    Picker("Compute units", selection: $computeChoice) {
                        ForEach(ComputeChoice.allCases) { choice in
                            Text(choice.rawValue).tag(choice)
                        }
                    }
                    Picker("Fixture", selection: $fixtureKey) {
                        ForEach(fixtures, id: \.self) { key in
                            Text(key).tag(key)
                        }
                    }
                }

                Section("Status") {
                    Text(runner.status)
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                    Button {
                        runner.run(fixtureKey: fixtureKey,
                                   computeUnits: computeChoice.mlComputeUnits)
                    } label: {
                        HStack {
                            if runner.isRunning {
                                ProgressView().padding(.trailing, 6)
                            }
                            Text(runner.isRunning ? "Running…" : "Run synthesis")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(runner.isRunning)
                }

                if let wav = runner.lastWavURL {
                    Section("Last WAV") {
                        Text(wav.lastPathComponent)
                            .font(.callout.monospaced())
                            .lineLimit(1)
                            .truncationMode(.middle)
                        HStack {
                            Button("Play") { play(url: wav) }
                                .buttonStyle(.bordered)
                            Spacer()
                            Button("Share") {
                                shareItems = [wav]
                                showShareSheet = true
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }

                if !runner.results.isEmpty {
                    Section("Results (newest first)") {
                        ForEach(runner.results) { r in
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(r.fixtureKey)  ·  \(r.computeUnits)  ·  bucket \(r.bucketSec)s")
                                    .font(.subheadline.weight(.semibold))
                                Text(String(format: "RTF %.3f   (%.1f× realtime)",
                                            r.rtf, r.rtf > 0 ? 1.0 / r.rtf : 0))
                                    .font(.callout.monospaced())
                                Text(String(format: "wall %.3f s   audio %.3f s",
                                            r.wallTimeSec, r.audioDurationSec))
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 2)
                        }
                    }
                }
            }
            .navigationTitle("HikeCompanion")
            .sheet(isPresented: $showShareSheet) {
                ShareSheet(items: shareItems)
            }
        }
    }

    private func play(url: URL) {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            let player = try AVAudioPlayer(contentsOf: url)
            audioPlayer = player
            player.play()
        } catch {
            print("Play error: \(error)")
        }
    }
}

// MARK: - Compute units picker

enum ComputeChoice: String, CaseIterable, Identifiable {
    case all = "all"
    case cpuAndNeuralEngine = "cpuAndNeuralEngine"
    case cpuAndGPU = "cpuAndGPU"
    case cpuOnly = "cpuOnly"

    var id: String { rawValue }

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        }
    }
}

// MARK: - UIActivityViewController bridge for sharing the WAV

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    ContentView()
}
