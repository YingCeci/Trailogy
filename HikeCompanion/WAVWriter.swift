// WAVWriter.swift
// Mono 16-bit PCM WAV writer at the pipeline's 24 kHz sample rate.
// Vendored from the upstream KokoroBenchmark/main.swift (Apache 2.0).

import Foundation

func writeWavMono16(path: String, samples: [Float], sampleRate: UInt32) throws {
    let n = samples.count
    var peak: Float = 1e-7
    for s in samples {
        let a = abs(s)
        if a > peak { peak = a }
    }

    var pcm = [Int16](repeating: 0, count: n)
    for i in 0..<n {
        let x = max(-1.0, min(1.0, samples[i] / peak))
        pcm[i] = Int16((x * 32767.0).rounded())
    }

    let dataSize = UInt32(n * 2)
    let byteRate = sampleRate * 2

    var d = Data()
    d.append(contentsOf: "RIFF".utf8)
    let riffChunkSize: UInt32 = 36 + dataSize
    withUnsafeBytes(of: riffChunkSize.littleEndian) { d.append(contentsOf: $0) }
    d.append(contentsOf: "WAVE".utf8)

    d.append(contentsOf: "fmt ".utf8)
    let subchunk1Size: UInt32 = 16
    withUnsafeBytes(of: subchunk1Size.littleEndian) { d.append(contentsOf: $0) }
    let audioFormat: UInt16 = 1
    withUnsafeBytes(of: audioFormat.littleEndian) { d.append(contentsOf: $0) }
    let numChannels: UInt16 = 1
    withUnsafeBytes(of: numChannels.littleEndian) { d.append(contentsOf: $0) }
    withUnsafeBytes(of: sampleRate.littleEndian) { d.append(contentsOf: $0) }
    withUnsafeBytes(of: byteRate.littleEndian) { d.append(contentsOf: $0) }
    let blockAlign: UInt16 = 2
    withUnsafeBytes(of: blockAlign.littleEndian) { d.append(contentsOf: $0) }
    let bitsPerSample: UInt16 = 16
    withUnsafeBytes(of: bitsPerSample.littleEndian) { d.append(contentsOf: $0) }

    d.append(contentsOf: "data".utf8)
    withUnsafeBytes(of: dataSize.littleEndian) { d.append(contentsOf: $0) }
    pcm.withUnsafeBytes { d.append(contentsOf: $0) }

    try d.write(to: URL(fileURLWithPath: path))
}
