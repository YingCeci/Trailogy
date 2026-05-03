// BenchmarkTimer.swift
//
// Stub re-introduced to satisfy KokoroSwift 1.0.11, which calls
// `BenchmarkTimer.reset() / startTimer(_:) / stopTimer(_:)` at the start
// and end of `KokoroTTS.generateAudio(...)`.
//
// MLXUtilsLibrary 0.0.7 removed the original (commit message:
// "Updated MLX version and removed BenchmarkTimer as unused and
// unsupported"). KokoroSwift 1.0.11 still expects it. This stub keeps
// the symbol available with the same public surface as 0.0.6 — but
// no-op, since we don't need internal Kokoro timing in this app
// (we measure end-to-end RTF in ValidationRunner ourselves).
//
// If a future upstream KokoroSwift release drops BenchmarkTimer, this
// file can be deleted.

import Foundation

public enum BenchmarkTimer {
    /// Clear all timer state. No-op in this stub.
    public static func reset() {}

    /// Begin timing under `name`. No-op in this stub.
    public static func startTimer(_ name: String) {}

    /// Stop timing under `name`. No-op in this stub.
    public static func stopTimer(_ name: String) {}

    /// Elapsed time for `name`, in seconds. Always nil in this stub.
    public static func getTimeInSec(_ name: String) -> Double? { nil }
}
