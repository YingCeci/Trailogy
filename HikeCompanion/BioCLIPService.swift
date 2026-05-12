// BioCLIPService.swift
// On-device species classification using BioCLIP-2 ViT-L/14 (MLX INT4).
//
// Pipeline:
//   Camera photo → preprocess (224x224, ImageNet normalize)
//   → ViT-L/14 vision encoder → 768-dim embedding
//   → cosine similarity with precomputed species embeddings
//   → top-K predictions injected into Gemma's prompt.
//
// Architecture: standard ViT-L/14
//   Conv2d(3→1024, k=14, s=14) → CLS + pos_embed → 24x blocks → LN → CLS pool → proj(768)
//   Blocks: LN → MHA(1024, 16 heads) → LN → MLP(1024→4096→1024, GELU)

import CoreGraphics
import Foundation
import MLX
import MLXNN

// MARK: - Species Prediction

struct SpeciesPrediction {
    /// Row index into `species_list.json` — the key used to look up
    /// `species_prompt_cards.json` entries when building Gemma's prompt.
    let index: Int
    let name: String
    let commonName: String
    let scientificName: String?
    let confidence: Float
}

// MARK: - ViT Components

/// Standard Multi-Head Self-Attention with fused QKV projection.
private class ViTAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenDim / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._inProj.wrappedValue = Linear(hiddenDim, 3 * hiddenDim, bias: true)
        self._outProj.wrappedValue = Linear(hiddenDim, hiddenDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Fused QKV projection. Kept as a Linear module so MLXNN.quantize
        // can load the INT4 weights emitted by scripts/bioclip/convert_to_mlx.py.
        let qkv = inProj(x)
        let chunks = qkv.split(parts: 3, axis: -1)
        let q = chunks[0].reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = chunks[1].reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = chunks[2].reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )

        // Reshape back and project
        let merged = attnOut.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim)
        return outProj(merged)
    }
}

/// Standard ViT MLP: Linear → GELU → Linear
private class ViTMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenDim: Int, intermediateDim: Int) {
        self._fc1.wrappedValue = Linear(hiddenDim, intermediateDim)
        self._fc2.wrappedValue = Linear(intermediateDim, hiddenDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

/// Standard ViT Transformer Block: Pre-LN attention + Pre-LN MLP with residuals
private class ViTBlock: Module {
    @ModuleInfo(key: "ln1") var ln1: LayerNorm
    @ModuleInfo(key: "ln2") var ln2: LayerNorm
    @ModuleInfo(key: "attn") var attn: ViTAttention
    @ModuleInfo(key: "mlp") var mlp: ViTMLP

    init(hiddenDim: Int, numHeads: Int, mlpRatio: Float = 4.0) {
        self._ln1.wrappedValue = LayerNorm(dimensions: hiddenDim, eps: 1e-6)
        self._ln2.wrappedValue = LayerNorm(dimensions: hiddenDim, eps: 1e-6)
        self._attn.wrappedValue = ViTAttention(hiddenDim: hiddenDim, numHeads: numHeads)
        self._mlp.wrappedValue = ViTMLP(hiddenDim: hiddenDim, intermediateDim: Int(Float(hiddenDim) * mlpRatio))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attn(ln1(x))
        return h + mlp(ln2(h))
    }
}

/// Complete ViT-L/14 vision encoder.
private class BioCLIPViT: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: Conv2d
    @ModuleInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray

    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    @ModuleInfo(key: "blocks") var blocks: [ViTBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    // Final projection to 768-dim embedding space
    @ModuleInfo(key: "proj") var proj: Linear

    let hiddenDim: Int
    let patchSize: Int

    init(config: BioCLIPConfig) {
        self.hiddenDim = config.width
        self.patchSize = config.patchSize

        self._patchEmbed.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: config.width,
            kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize),
            bias: false
        )
        self._clsToken.wrappedValue = MLXRandom.normal([config.width])
        self._posEmbed.wrappedValue = MLXRandom.normal([config.seqLen, config.width])

        self._lnPre.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-6)
        self._blocks.wrappedValue = (0..<config.layers).map { _ in
            ViTBlock(hiddenDim: config.width, numHeads: config.heads)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-6)

        self._proj.wrappedValue = Linear(config.width, config.outputDim, bias: false)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        // pixelValues: [B, H, W, C] (NHWC for MLX)
        // 1. Patch embedding via Conv2d
        var x = patchEmbed(pixelValues)               // [B, gridH, gridW, hiddenDim]
        let B = x.dim(0)
        let gridH = x.dim(1)
        let gridW = x.dim(2)
        x = x.reshaped(B, gridH * gridW, hiddenDim)   // [B, numPatches, hiddenDim]

        // 2. Prepend CLS token
        let cls = broadcast(
            clsToken.reshaped(1, 1, hiddenDim),
            to: [B, 1, hiddenDim]
        )
        x = concatenated([cls, x], axis: 1)            // [B, 1+numPatches, hiddenDim]

        // 3. Add positional embeddings
        x = x + posEmbed

        // 4. Pre-LayerNorm
        x = lnPre(x)

        // 5. Transformer blocks
        for block in blocks {
            x = block(x)
        }

        // 6. Post-LayerNorm on CLS token
        let clsOut = lnPost(x[0..., 0, 0...])          // [B, hiddenDim]

        // 7. Final projection
        let projected = proj(clsOut)                     // [B, outputDim]

        // 8. L2 normalize
        let norm = sqrt(sum(projected * projected, axis: -1, keepDims: true))
        return projected / norm
    }

    /// Transpose Conv2d weights from PyTorch OIHW to MLX OHWI format.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if key == "patch_embed.weight" && value.ndim == 4 {
                // PyTorch: [outC, inC, kH, kW] → MLX: [outC, kH, kW, inC]
                result[key] = value.transposed(0, 2, 3, 1)
            } else {
                result[key] = value
            }
        }
        return result
    }
}

// MARK: - Config

struct BioCLIPConfig: Codable {
    let imageSize: Int
    let patchSize: Int
    let width: Int
    let layers: Int
    let heads: Int
    let outputDim: Int
    let seqLen: Int
    let quantization: QuantInfo?

    struct QuantInfo: Codable {
        let bits: Int
        let groupSize: Int

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case width, layers, heads
        case outputDim = "output_dim"
        case seqLen = "seq_len"
        case quantization
    }
}

// MARK: - Species Data

private struct SpeciesInfo: Codable {
    let index: Int
    let name: String
    let commonName: String
    let scientificName: String?

    enum CodingKeys: String, CodingKey {
        case index, name
        case commonName = "common_name"
        case scientificName = "scientific_name"
    }
}

// MARK: - BioCLIP Service

@MainActor
class BioCLIPService: ObservableObject {
    @Published var status: String = "Not loaded"
    @Published var isLoaded: Bool = false

    private var model: BioCLIPViT?
    private var speciesEmbeddings: MLXArray?  // [N, 768] float16
    private var speciesList: [SpeciesInfo] = []
    private var config: BioCLIPConfig?

    /// Compact per-species "ID cards" for prompt injection. Keyed by the
    /// string form of `SpeciesInfo.index` (matches `species_id` in the
    /// offline enrichment pipeline at
    /// gemma4_note/05b-data_plantnet300k-enrich/build_prompt_cards.py).
    ///
    /// Each card is ~50-60 tokens of "common name (scientific, family):
    /// 1-2 morphology sentences. Range: ...". Missing entries are
    /// tolerated — `formatForPrompt` falls back gracefully if a top-K
    /// species_id has no card.
    private var speciesCards: [String: String] = [:]

    // ImageNet normalization constants
    private let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    private let std: [Float] = [0.26862954, 0.26130258, 0.27577711]

    // MARK: - Lifecycle

    func loadIfNeeded() {
        guard !isLoaded else { return }
        status = "Loading BioCLIP..."

        guard let modelDir = Bundle.main.url(
            forResource: "config",
            withExtension: "json",
            subdirectory: "Models/BioCLIP"
        )?.deletingLastPathComponent() else {
            status = "BioCLIP model not found"
            return
        }

        do {
            // Load config
            let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
            config = try JSONDecoder().decode(BioCLIPConfig.self, from: configData)
            guard let config = config else { return }

            // Create model
            let vit = BioCLIPViT(config: config)

            // Load weights
            let weightsURL = modelDir.appendingPathComponent("model.safetensors")
            var weights = try MLX.loadArrays(url: weightsURL)

            // Sanitize (conv2d transpose)
            weights = vit.sanitize(weights: weights)

            // Apply quantization if configured
            if let q = config.quantization {
                MLXNN.quantize(
                    model: vit,
                    groupSize: q.groupSize,
                    bits: q.bits
                )
            }

            // Load weights into model
            let params = ModuleParameters.unflattened(weights)
            try vit.update(parameters: params, verify: .noUnusedKeys)

            model = vit

            // Load species metadata
            let metaURL = modelDir.appendingPathComponent("species_list.json")
            let metaData = try Data(contentsOf: metaURL)
            speciesList = try JSONDecoder().decode([SpeciesInfo].self, from: metaData)

            // Load species embeddings after metadata so rows follow species_list.json order.
            let embURL = modelDir.appendingPathComponent("species_embeddings.npz")
            speciesEmbeddings = try loadNPZEmbeddings(url: embURL)

            // Optional: per-species prompt cards for richer Gemma context.
            // Missing file is OK — the formatForPrompt fallback covers it.
            let cardsURL = modelDir.appendingPathComponent("species_prompt_cards.json")
            if FileManager.default.fileExists(atPath: cardsURL.path) {
                do {
                    let cardsData = try Data(contentsOf: cardsURL)
                    speciesCards = try JSONDecoder().decode([String: String].self, from: cardsData)
                    print("[BioCLIP] loaded \(speciesCards.count) prompt cards")
                } catch {
                    print("[BioCLIP] WARN: species_prompt_cards.json failed to parse: \(error)")
                    speciesCards = [:]
                }
            } else {
                print("[BioCLIP] no species_prompt_cards.json in bundle; using legacy single-line tag")
            }

            isLoaded = true
            status = "Ready (\(speciesList.count) species, \(speciesCards.count) cards)"
            MemoryStats.log("bioclip.load done")
        } catch {
            status = "Load failed: \(error.localizedDescription)"
            print("[BioCLIP] Load error: \(error)")
        }
    }

    func unload() {
        model = nil
        speciesEmbeddings = nil
        // Keep `speciesList` and `speciesCards` resident — they're
        // metadata, not MLX tensors, and reloading them costs disk IO
        // for no MLX memory benefit.
        Memory.clearCache()
        isLoaded = false
        status = "Not loaded"
        MemoryStats.log("bioclip.unload done")
    }

    // MARK: - Classification

    /// Classify an image and return top-K species predictions.
    func classify(image: CGImage, topK: Int = 3) -> [SpeciesPrediction] {
        guard let model = model, let speciesEmb = speciesEmbeddings else {
            return []
        }

        // Preprocess to [1, 224, 224, 3] NHWC float32
        let input = preprocessImage(image)

        // Forward pass
        let embedding = model(input)  // [1, 768]
        eval(embedding)

        // Cosine similarity with precomputed species embeddings
        let similarities = matmul(embedding, speciesEmb.transposed())  // [1, N]
        let simFlat = similarities.squeezed(axis: 0)  // [N]
        eval(simFlat)

        // Extract scores as Swift array
        let scores: [Float] = simFlat.asArray(Float.self)

        // Top-K via argsort
        let indexed = scores.enumerated().sorted { $0.element > $1.element }
        let topResults = indexed.prefix(topK)

        return topResults.compactMap { (idx, score) -> SpeciesPrediction? in
            guard idx < speciesList.count else { return nil }
            let sp = speciesList[idx]
            return SpeciesPrediction(
                index: sp.index,
                name: sp.name,
                commonName: sp.commonName,
                scientificName: sp.scientificName,
                confidence: score
            )
        }
    }

    // MARK: - Prompt formatting
    //
    // Calibration follows the empirical findings in
    // gemma4_note/03-bioclip_explore/docs/01-bioclip-output-characterization.md.
    // The numbers are specific to BioCLIP-2 ViT-L/14 (768-d) with the
    // 4-template averaging in scripts/bioclip/precompute_embeddings.py.
    // Re-derive if anyone swaps the checkpoint or changes the templates.

    /// Cosine below which we treat the photo as probably outside the
    /// 101-species shortlist. Measured: in-set top-1 cosines cluster at
    /// +0.72, out-of-set at +0.57; +0.60 is below the in-set
    /// distribution and above the out-of-set mean.
    private static let inSetCosineFloor: Float = 0.60

    /// Within how many cosine points are two candidates treated as
    /// "tied"? From `03-bioclip_explore/docs/01-...md`: within-genus
    /// failures cluster at 0.005-0.015 above/below the GT cosine.
    /// 0.02 catches the typical "BioCLIP cannot distinguish these"
    /// pattern without misclassifying clearly-separated scores as ties.
    /// Surfaced to Gemma so the model can honestly hedge when the
    /// classifier itself isn't decisive, instead of guessing a species.
    private static let tiedCosineWindow: Float = 0.02

    /// Genus epithet from a binomial: "Tsuga canadensis" -> "Tsuga".
    private static func genusEpithet(_ scientific: String?) -> String? {
        guard let s = scientific else { return nil }
        let parts = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        guard let first = parts.first, !first.isEmpty else { return nil }
        return String(first)
    }

    /// Look up the offline-derived prompt card for a prediction.
    /// Falls back to a synthesized "commonName (scientificName)" line
    /// when the card JSON is missing or doesn't have this species,
    /// so card-availability gaps never silently drop a candidate.
    private func cardFor(_ p: SpeciesPrediction) -> String {
        if let card = speciesCards[String(p.index)], !card.isEmpty {
            return card
        }
        // Fallback when cards JSON isn't shipped, or this species_id
        // wasn't in the enrichment output (e.g., GBIF couldn't match).
        var s = p.commonName
        if let sci = p.scientificName, !sci.isEmpty {
            s += " (" + sci + ")"
        }
        return s
    }

    /// Format predictions for Gemma prompt injection.
    ///
    /// Design from `docs/specs/2026-05-12-bioclip-enriched-prompt-cot.md`,
    /// revised iteratively against local-test feedback:
    ///
    ///   Round 1: free-form re-ranking degraded accuracy on 2/3 test
    ///   images → switched to closed-set commit + `<thinking>` requirement.
    ///
    ///   Round 2: a closed-set commit is wrong when the classifier
    ///   itself is not decisive. We now compute the "tied subset" —
    ///   everyone within `tiedCosineWindow` cosine of #1 — and branch
    ///   on the subset's properties:
    ///
    ///     a. Below in-set floor → suppress cards, low-confidence tag.
    ///     b. Tied subset singleton → "lead with #1".
    ///     c. Tied subset shares a genus → "commit to the genus".
    ///     d. Tied subset has mixed genera → "name ALL tied candidates;
    ///        don't pick one without a clear discriminator".
    ///
    ///   Lower-ranked candidates outside the tied window get a trailing
    ///   "← lower classifier confidence" marker so Gemma downweights
    ///   them without us having to expose raw cosines.
    ///
    /// Reinforced from the Gemma side in `GemmaService.baseInstructions`.
    func formatForPrompt(predictions: [SpeciesPrediction]) -> String {
        guard let top = predictions.first else { return "" }

        if top.confidence < Self.inSetCosineFloor {
            return "[BioCLIP: low confidence — species likely outside the trail list; describe what is visible in the photo]"
        }

        let topN = Array(predictions.prefix(3))
        let topCos = top.confidence
        let window = Self.tiedCosineWindow

        // Tied subset: candidates within window of #1 (includes #1).
        let tiedSubset = topN.filter { (topCos - $0.confidence) <= window }

        // Shared genus of the TIED subset only.
        let tiedGenera = tiedSubset.compactMap { Self.genusEpithet($0.scientificName) }
        let tiedSharedGenus: String? = {
            guard tiedGenera.count == tiedSubset.count,
                  let g = tiedGenera.first,
                  Set(tiedGenera).count == 1
            else { return nil }
            return g
        }()

        // Numbered cards with inline "lower confidence" markers.
        let numberedCards = topN.enumerated()
            .map { (i, p) -> String in
                var line = "\(i + 1). " + cardFor(p)
                if (topCos - p.confidence) > window {
                    line += "  ← lower classifier confidence"
                }
                return line
            }
            .joined(separator: "\n")

        let windowStr = String(format: "%.2f", window)
        let tiedN = tiedSubset.count
        let tiedNames = tiedSubset
            .map { $0.commonName.isEmpty ? ($0.scientificName ?? $0.name) : $0.commonName }
            .joined(separator: " / ")

        let lead: String
        if tiedN == 1 {
            // Top-1 dominant.
            lead = "[BioCLIP candidates (top-3). #1 is materially more confident than the others (marked '← lower classifier confidence'). Lead with #1 unless its card clearly conflicts with what you see in the photo, in which case fall back to a shared genus if there is one or say plainly that the image doesn't match any listed candidate. Do not propose species not on this list.\nCards:"
        } else if let g = tiedSharedGenus {
            // Tied subset shares a genus → commit to genus.
            lead = "[BioCLIP candidates (top-3). The top \(tiedN) candidates (\(tiedNames)) are within ~\(windowStr) cosine of each other and all belong to genus '\(g)'. The classifier is NOT decisive at the species level. Your safest answer is the genus '\(g)'. Only commit to a specific species if you can point to a clear morphological discriminator in the photo. Do not propose species not on this list.\nCards:"
        } else {
            // Tied subset has mixed genera → name all.
            lead = "[BioCLIP candidates (top-3). The top \(tiedN) candidates (\(tiedNames)) are within ~\(windowStr) cosine of each other and belong to different genera — the classifier is NOT decisive between them. Name ALL the tied candidates together in your answer (e.g. \"this looks like either X or Y\"), or describe by general type if none of them clearly fits. Do not pick one over the others without a clear visible discriminator. Do not propose species not on this list.\nCards:"
        }

        return lead + "\n" + numberedCards + "]"
    }

    // MARK: - Image Preprocessing

    /// Resize + center crop to 224x224, normalize with ImageNet stats, return NHWC.
    private func preprocessImage(_ cgImage: CGImage) -> MLXArray {
        let targetSize = 224

        // Resize shortest edge to 224, then center crop
        let srcW = cgImage.width
        let srcH = cgImage.height
        let shortEdge = min(srcW, srcH)
        let scale = Float(targetSize) / Float(shortEdge)
        let newW = Int(Float(srcW) * scale)
        let newH = Int(Float(srcH) * scale)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = newW * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: newH * bytesPerRow)

        guard let context = CGContext(
            data: &pixelData,
            width: newW,
            height: newH,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return MLXArray.zeros([1, targetSize, targetSize, 3])
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))

        // Center crop to 224x224
        let cropX = (newW - targetSize) / 2
        let cropY = (newH - targetSize) / 2

        var rgbFloat = [Float](repeating: 0, count: targetSize * targetSize * 3)
        for y in 0..<targetSize {
            for x in 0..<targetSize {
                let srcIdx = ((cropY + y) * newW + (cropX + x)) * bytesPerPixel
                let dstIdx = (y * targetSize + x) * 3
                // RGBA → RGB, normalize to [0,1] then ImageNet normalize
                rgbFloat[dstIdx + 0] = (Float(pixelData[srcIdx + 0]) / 255.0 - mean[0]) / std[0]
                rgbFloat[dstIdx + 1] = (Float(pixelData[srcIdx + 1]) / 255.0 - mean[1]) / std[1]
                rgbFloat[dstIdx + 2] = (Float(pixelData[srcIdx + 2]) / 255.0 - mean[2]) / std[2]
            }
        }

        // Create NHWC array [1, 224, 224, 3]
        return MLXArray(rgbFloat, [1, targetSize, targetSize, 3])
    }

    // MARK: - NPZ Loading

    /// Load species embeddings from npz file.
    /// NPZ format: compressed dict of {species_name: [768] float16 array}
    private func loadNPZEmbeddings(url: URL) throws -> MLXArray {
        // MLX can load npz directly
        let arrays = try MLX.loadArrays(url: url)

        // Stack in species_list order
        var rows: [MLXArray] = []
        for sp in speciesList {
            if let emb = arrays[sp.name] {
                rows.append(emb.asType(.float32))
            } else {
                // Fallback: zero vector if species not found in embeddings
                rows.append(MLXArray.zeros([768]))
            }
        }

        guard !rows.isEmpty else {
            return MLXArray.zeros([1, 768])
        }

        return stacked(rows, axis: 0)  // [N, 768]
    }
}
