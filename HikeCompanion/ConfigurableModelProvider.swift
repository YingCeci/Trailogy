// ConfigurableModelProvider.swift
// Implements `KokoroModelProvider` with a configurable `MLComputeUnits` so we
// can A/B test ANE-preferred (`.all`, `.cpuAndNeuralEngine`) vs ANE-disabled
// (`.cpuAndGPU`, `.cpuOnly`) on the same iPhone.
//
// Mirrors the upstream `KokoroBenchmark/main.swift::ModelCache` class.
// Key difference vs upstream: ours is initialized from the app bundle's
// `Models/` folder reference (read-only, synced as-is by Xcode at build time).

import CoreML
import Foundation
import KokoroPipeline

final class ConfigurableModelProvider: KokoroModelProvider {
    let modelsDir: URL
    let config: MLModelConfiguration
    private let durationChoices: [DurationModelChoice]

    // Layer 1: compiled .mlmodelc URLs (compilation is the expensive part — cache forever)
    private var compiledDuration: [String: URL] = [:]
    private var compiledF0n: [Int: URL] = [:]
    private var compiledDecPre: [Int: URL] = [:]
    private var compiledGen: [Int: URL] = [:]

    // Layer 2: loaded MLModel instances
    private var durationModels: [String: MLModel] = [:]
    private var f0nModels: [Int: MLModel] = [:]
    private var decPreModels: [Int: MLModel] = [:]
    private var genModels: [Int: MLModel] = [:]

    init(modelsDir: URL, computeUnits: MLComputeUnits = .all) {
        self.modelsDir = modelsDir
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        self.config = cfg
        self.durationChoices = KokoroPipeline.discoverDurationChoices(modelsDirectory: modelsDir)
    }

    // MARK: - KokoroModelProvider

    func durationModelChoices() -> [DurationModelChoice] {
        durationChoices
    }

    func availableBucketSeconds() -> [Int] {
        PipelineConstants.defaultBuckets
    }

    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {
        evictExcept(bucket: bucketSec, tFrames: tFrames)
    }

    func durationModel(choice: DurationModelChoice) throws -> MLModel {
        if let cached = durationModels[choice.cacheKey] { return cached }
        let compiled = try compiledDurationURL(choice: choice)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        durationModels[choice.cacheKey] = model
        return model
    }

    func f0ntrainModel(tFrames: Int) throws -> MLModel {
        if let cached = f0nModels[tFrames] { return cached }
        let compiled = try compiledF0nURL(tFrames: tFrames)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        f0nModels[tFrames] = model
        return model
    }

    func decoderPreModel(bucketSec: Int) throws -> MLModel {
        if let cached = decPreModels[bucketSec] { return cached }
        let compiled = try compiledDecPreURL(bucket: bucketSec)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        decPreModels[bucketSec] = model
        return model
    }

    func generatorModel(bucketSec: Int) throws -> MLModel {
        if let cached = genModels[bucketSec] { return cached }
        let compiled = try compiledGenURL(bucket: bucketSec)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        genModels[bucketSec] = model
        return model
    }

    // MARK: - Compile-on-first-access (per-package)

    private func compiledDurationURL(choice: DurationModelChoice) throws -> URL {
        if let url = compiledDuration[choice.cacheKey] { return url }
        let compiled = try MLModel.compileModel(at: choice.packageURL)
        compiledDuration[choice.cacheKey] = compiled
        return compiled
    }

    private func compiledF0nURL(tFrames: Int) throws -> URL {
        if let url = compiledF0n[tFrames] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_f0ntrain_t\(tFrames).mlpackage")
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledF0n[tFrames] = compiled
        return compiled
    }

    private func compiledDecPreURL(bucket: Int) throws -> URL {
        if let url = compiledDecPre[bucket] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_decoder_pre_\(bucket)s.mlpackage")
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledDecPre[bucket] = compiled
        return compiled
    }

    private func compiledGenURL(bucket: Int) throws -> URL {
        if let url = compiledGen[bucket] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_decoder_har_post_\(bucket)s.mlpackage")
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledGen[bucket] = compiled
        return compiled
    }

    // MARK: - Memory management

    /// Evict loaded `MLModel` instances for buckets/tFrames other than the one
    /// currently in use. Compiled URLs (`.mlmodelc`) are kept so a re-load
    /// doesn't need to re-compile.
    private func evictExcept(bucket: Int, tFrames: Int) {
        f0nModels = f0nModels.filter { $0.key == tFrames }
        decPreModels = decPreModels.filter { $0.key == bucket }
        genModels = genModels.filter { $0.key == bucket }
    }
}
