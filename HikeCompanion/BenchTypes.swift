// BenchTypes.swift
// Decodable types matching the JSON shape produced by the upstream
// `scripts/prepare_swift_bench_inputs.py`. These mirror the structs used by
// the upstream KokoroBenchmark target (which we can't import as a library).

import Foundation

struct BenchInput: Decodable {
    let key: String
    let text: String
    let voice: String
    let speed: Float
    let input_ids: [Int32]
    let attention_mask: [Int32]
    let ref_s: [Float]
    let num_tokens: Int
    /// Canonical audio duration from the bakeoff manifest, in seconds.
    /// Computed on the Python side as `T_f0 / 80.0`. Used for RTF math.
    let canonical_duration_s: Double?
}

struct HnsfWeights: Decodable {
    let linear_weights: [Float]
    let linear_bias: Float
    /// Optional integrity check; not enforced.
    let weights_sha256: String?
}
