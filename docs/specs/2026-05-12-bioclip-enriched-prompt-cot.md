> created: 2026-05-12
> status: approved, pending implementation plan

# BioCLIP top-4 + PlantNet enrichment → Gemma structured CoT

## Goal

Replace the current single-line `[BioCLIP: ...]` tag
(`HikeCompanion/BioCLIPService.swift:408–432`) with a 4-candidate
prompt where each candidate carries an offline-derived "ID card"
(common name + scientific name + family + 1–2 sentences of morphology
+ region). Gemma performs structured Chain-of-Thought, emitting
`REASONING:` then `ANSWER:`. Only the `ANSWER:` section is voiced via
Kokoro TTS; the reasoning is surfaced in the iOS UI as an expandable
debug card.

Target: lift effective species-level top-1 from BioCLIP's measured
~49% (see `gemma4_note/03-bioclip_explore/docs/01-bioclip-output-characterization.md`,
section B) toward the ~75–80% top-4 recall band by using the VLM as a
re-ranker over morphology-grounded candidate descriptions.

## Non-goals

- Does not LLM-compress cards offline. Heuristic extraction first; if
  sparse-data species turn out to be a real failure mode in
  end-to-end testing, a follow-up pass can use a server-side LLM to
  rewrite weak cards.
- Does not add a visible 4-card UI strip in `WalkingView`. The
  reasoning block in the existing answer area is the minimum new UI
  surface. A richer card strip is a possible phase 2.
- Does not change the on-device species index from PlantNet-300K's
  1081-species space. The `species_id` keys in `species_prompt_cards.json`
  must align with the existing `species_list.json` shipped to BioCLIP.
  Verifying that alignment is an implementation-time check, not a
  design change.

## Architecture

```
[offline, one-time]
species_metadata_enriched.csv ──▶ build_prompt_cards.py
                                          │
                                          ▼
                              species_prompt_cards.json  (~220 KB)
                                          │
                                          ▼
[iOS bundle]            Resources/Models/BioCLIP/species_prompt_cards.json

[on-device runtime]
camera → BioCLIP.classify(topK=4) ──▶ formatForPrompt(top-4 + cards)
                                                      │
                                                      ▼
                                            gemma.bioclipContext
                                                      │
                                                      ▼
                                          Gemma VLM (CoT-prompted)
                                                      │
                                                      ▼
                          streamResponse splits "REASONING: ... ANSWER: ..."
                                          │                       │
                                          ▼                       ▼
                                   WalkingView UI            Kokoro TTS
                              (reasoning collapsible)        (only ANSWER)
```

## Components

### 1. `gemma4_note/05b-data_plantnet300k-enrich/build_prompt_cards.py` (new, ~80 LOC)

Reads `species_metadata_enriched.csv`, emits `species_prompt_cards.json`
as a `{species_id: card_text}` map. Per species:

- **common name**: first English entry from `common_names` (semicolon-split).
- **scientific name + family**: from `accepted_scientific_name`
  (fallback `species`) and `family`.
- **morphology**: split `best_description` into sentences. Score each
  by density of keywords in this set:
  `leaf, leaves, flower, stem, bark, height, color, fruit, root, blade, petal, needle, cone, branch`.
  Take the top 1–2 sentences by score. Drop citation tails (`from: ...`).
- **region**: first 3 semicolon-separated tokens from `gbif_distribution`.
- **assembly**: `"{common} ({scientific}, {family}): {morphology}. Range: {region}."`
- **token cap**: ~60 tokens via tiktoken approximation (cl100k\_base);
  truncate on sentence boundary when over budget.

Side output: `cards_stats.json` with per-card token count, has-morphology
flag, has-region flag. Used for QA inspection.

### 2. iOS bundle additions

- Drop `species_prompt_cards.json` into
  `HikeCompanion/Resources/Models/BioCLIP/`.
- xcodegen blue-folder reference (`HikeCompanion/Resources/Models`,
  `type: folder` per `CLAUDE.md` "Bundle layout") preserves the file
  automatically; no `project.yml` edit needed.

### 3. `BioCLIPService.swift`

- Load `species_prompt_cards.json` at `loadIfNeeded` alongside
  `species_list.json`. Add `private var speciesCards: [String: String]`
  keyed by species\_id-as-string.
- Change default `classify(topK: 5)` call sites to `topK: 4`.
- Replace `formatForPrompt(predictions:)` to build a numbered
  4-card block:

  ```
  [BioCLIP found 4 candidate species. Compare each against the photo:
  1. <card 1>
  2. <card 2>
  3. <card 3>
  4. <card 4>]
  ```

- **Preserve** the +0.60 in-set cosine floor: when `top.confidence <
  inSetCosineFloor`, suppress the card block entirely and emit the
  existing "low confidence; describe what is visible" tag unchanged.
- **Preserve** the genus-collapse hint: when all 4 candidates share
  the same first epithet, prefix the card block with the existing
  "candidates all in genus X" line.
- **Card lookup miss** (top-K species\_id not in cards JSON):
  synthesize a fallback card with only `"{scientific_name} ({family})"`
  so the candidate slot is not silently dropped.

### 4. `GemmaService.swift`

**Memory tunables** (lines 154–159; breaks the documented memory
profile in `CLAUDE.md` — requires re-measurement on device and a
docs update):

- `maxTokens: 120 → 240` (REASONING ~120 + ANSWER ~120)
- `maxKVSize: 1024 → 1280`
- Expected VLM peak: 3.54 GB → ~4.3 GB based on the "~3 MB / token of
  peak footprint" coefficient from the existing comment. Safely under
  the 6 GB `increased-memory-limit` ceiling.

**Instructions update** (lines 106–121, `baseInstructions`):

Add a paragraph teaching the CoT protocol. Approximate wording:

> If a `[BioCLIP found 4 candidate species ...]` block appears in the
> user message, first emit a brief `REASONING:` section (2–3
> sentences) describing which visible features in the photo narrow
> the candidates, then emit an `ANSWER:` section with your top pick
> and one backup, each anchored to one observable cue. Only the
> `ANSWER:` section will be spoken aloud — keep the reasoning
> diagnostic and the answer narratable.

The existing "if photo is clearly not outdoors, describe what's
there" escape-hatch sentence stays — the new instruction layers on
top.

**Stream split**:

- Introduce `enum GemmaSection { case reasoning, answer }`.
- Change `streamResponse` return type from
  `AsyncThrowingStream<String, Error>?` to
  `AsyncThrowingStream<(String, GemmaSection), Error>?`.
- State machine: start in `.reasoning`. Buffer chunks while scanning
  for a case-insensitive `ANSWER:` marker (also allow `**ANSWER:**`
  and bare `Answer:`). When found, flip to `.answer` and yield only
  the post-marker portion.
- **Fallback**: if no marker by 250 generated tokens, flip to
  `.answer` retroactively (re-yield the buffered text as `.answer`)
  and log `[BioCLIP-CoT] no marker, treating full output as answer`.
- The pre-marker buffer is also exposed verbatim through the
  `(String, .reasoning)` yields so the UI can render it live.

### 5. `WalkingView.swift`

- Existing `answerText: String` continues to be what TTS receives;
  only `.answer`-tagged chunks append to it.
- New `reasoningText: String` accumulates `.reasoning`-tagged chunks.
- TTS hookup needs no change — it already consumes `answerText`.
- UI: small expandable "How I worked this out" disclosure under the
  spoken answer, showing `reasoningText` when non-empty. Style to
  match existing debug surfaces (mono font, dim, collapsed by
  default).

## Data flow detail — the composed VLM prompt

A full composed user prompt for an image ask now looks like:

```
[BioCLIP found 4 candidate species. Compare each against the photo:
1. eastern hemlock (Tsuga canadensis, Pinaceae): Conifer to 30 m; flat
   short needles (~10 mm) with 2 white stomatal bands beneath; small
   1–2 cm cones. Range: E. North America.
2. western hemlock (Tsuga heterophylla, Pinaceae): Conifer to 60 m;
   needles of uneven lengths (5–20 mm), 2 white bands beneath; cones
   2–3 cm. Range: W. North America.
3. ...
4. ...]

[Currently at Stop 2 of 5: Hemlock Grove. ...]

what is this tree?
```

Expected Gemma response:

```
REASONING: The image shows a conifer with short, uniformly-flat
needles and small 1 cm cones on a slender drooping leader. That
matches candidate 1 (Eastern Hemlock). Candidates 3–4 are not
conifers and don't match. Western Hemlock would have visibly
uneven needle lengths.

ANSWER: Eastern Hemlock (Tsuga canadensis) — those short uniform
flat needles and tiny cones give it away. Western Hemlock is a
backup, but the needles look too consistent in length for that.
```

Only the `ANSWER:` block reaches Kokoro.

## Error handling

| Failure | Behavior |
|---|---|
| `species_prompt_cards.json` missing from bundle | Log warning; fall back to existing top-3 single-line tag (no cards). |
| Card lookup misses for a specific species\_id in top-4 | Synthesize fallback card from scientific name + family. |
| Top-1 cosine below the 0.60 in-set floor | Suppress card block; emit the existing low-confidence guidance unchanged. |
| Gemma never emits `ANSWER:` marker | After 250 generated tokens, flip the entire buffer to `.answer` and stream-yield it; log the protocol miss. |
| Stream interrupted mid-`REASONING` | UI shows partial reasoning. TTS gets nothing. No behavior change vs. today's interrupted-stream case. |
| `--no-resume` enrichment ran half-way and was killed | `build_prompt_cards.py` should accept a partial CSV gracefully (skip rows with empty `best_description` AND empty `wikipedia_summary` → synthesize family-only fallback card). |

## Testing

- **Card builder unit test** (`gemma4_note/05b-data_plantnet300k-enrich/`):
  run against the existing 25-row resume sample
  (`sample_enriched.csv` extended). Assert ≥80% of cards have at
  least one morphology sentence. Eyeball-check 5 well-sourced and 5
  sparse species in the stats output.
- **Swift unit tests**: no Swift test target exists in the repo
  (verified via project layout in `CLAUDE.md`). Coverage is manual
  on-device, supplemented by `BioCLIPService.formatForPrompt` being
  written as a pure function so it could be exercised from a future
  test target with no refactor.
- **End-to-end device test** (iPhone 17 Pro): photograph 5 plants —
  3 in the 101-species shortlist, 2 deliberately off-list. Verify:
  a) `REASONING:` and `ANSWER:` both appear in the stream
  b) only `ANSWER:` content reaches Kokoro (audible check)
  c) `[Mem]` ticker shows VLM peak < 5 GB
  d) total Ask latency (mic-up → TTS-first-word) increase < 4 s vs.
     today's measured baseline.
- **Update `CLAUDE.md`**: refresh the "Image Ask (VLM mode)" memory
  table with the re-measured peaks; bump the `maxKVSize` value cited
  in the table footer.

## Open questions for the implementation plan

1. **Fold `species_list.json` into `species_prompt_cards.json`?**
   The card text bakes in name/scientific/family. The remaining role
   of `species_list.json` is providing the index→species mapping for
   row alignment with `species_embeddings.npz`. If we keep
   `species_list.json` for that, the cards JSON can be a sibling
   `{species_id: card}` map without duplicating fields. Defer
   decision to the impl plan; either way is a tiny bundle change.

2. **Card token-budget calibration.** ~60 tokens per card × 4 cards =
   ~240 tokens of BioCLIP context. Budget math assumes:
   `vision 280 + system 180 + stop 50 + cards 240 + user 30 = 780
   prompt + 240 gen = 1020 ≤ 1280 maxKVSize`. If real cards trend
   longer (e.g., morphology sentences for ferns running long), the
   builder's hard cap may need to drop to 50 tokens. Resolve with
   measured stats from the first full enrichment run, not during
   design.

3. **Where does the build script's output land?** The Python script
   lives in `gemma4_note/`. The JSON consumer is `hikeCompanion/`.
   Cleanest: build script writes to its own directory; a tiny
   `scripts/fetch-prompt-cards.sh` (sibling of `fetch-models.sh`)
   copies the artifact into `HikeCompanion/Resources/Models/BioCLIP/`.
   Avoids cross-repo path coupling and matches the existing fetch
   script pattern.
