# RAG corpus — three demonstration trails

Pre-embedded subject corpora that back the three demonstration trails
(Kildoo, Old Field, Tranquil). Every chunk is crawled from
authoritative public sources and verified twice — once by a human
reviewer, once by GPT-5 — before it is embedded and bundled into the
iOS app.

The pipeline is designed for **factuality first, then scalability**:
adding a new trail expands the corpus through the same fetch +
dual-verification path, without changing the on-device retrieval
architecture.

## Files

| File | Subject | Chunks | Crawl sources |
|---|---|---|---|
| `geology.jsonl` | Geology | 25 | USGS Open-File Reports, OpenStax Geology — western-PA / sandstone gorge focus + general earth-science framing (bundled copy adds `geo-026`) |
| `plants.jsonl` | Plants | 25 | USDA PLANTS database, Wikipedia species pages — eastern hardwood forest species, trail-relevant flora |
| `physics.jsonl` | Physics | 20 | OpenStax College Physics — outdoor-phenomena subset (acoustics, optics, thermodynamics, fluid mechanics) |
| `english.jsonl` | English | 20 | Project Gutenberg / Standard Ebooks — nature-writing themes (Thoreau, Muir, Burroughs, Emerson, Whitman) |
| `manifest.json` | — | — | Subject manifest with version, chunk counts |

## Schema

One JSON object per line:

```json
{
  "id": "geo-001",
  "subject": "geology",
  "title": "What is sandstone?",
  "text": "Sandstone is a sedimentary rock made of compressed sand-sized mineral grains...",
  "summary": "Sedimentary rock of compressed sand grains; common in PA gorges.",
  "tags": ["sandstone", "sedimentary", "rock_types"],
  "region": "general",
  "source": "USGS / OpenStax Geology — crawled; verified by human + GPT-5"
}
```

| Field | Purpose |
|---|---|
| `id` | Stable identifier, `{subject_prefix}-{NNN}` |
| `subject` | One of `geology`, `plants`, `physics`, `english` |
| `title` | Short human-readable label, used in citations |
| `text` | The chunk body (target ~100-150 words; embed this) |
| `summary` | One-sentence compression for the "summary-first" retrieval strategy in the ADR |
| `tags` | Lexical hints; useful if we layer BM25 over embeddings later |
| `region` | `general` or `western_pa` — lets us boost local content for our trails |
| `source` | Provenance string |

## Provenance & verification

Every chunk in this corpus was produced by the same fetch-and-verify
pipeline:

1. **Crawl** — pull source material from an authoritative public
   reference (per-subject sources are listed in the file table above).
2. **Human review** — a reviewer checks domain plausibility and trail
   relevance (e.g. "this Mississippian-age sandstone description
   actually fits the Kildoo gorge").
3. **GPT-5 review** — a second pass flags factual drift, ambiguous
   claims, and citations that don't match the chunk text.

Only chunks that survive both review passes are embedded and bundled
for shipping. The `source` field on each line records the upstream
reference family the chunk was crawled from.

**Scalability.** The pipeline is parameterized over `(trail,
subject)`, so adding a new demonstration trail expands the corpus
along the same path — no change to the on-device retrieval
architecture or to `RAGService.swift`.

**English subset note.** The English chunks are analytical summaries
of public-domain nature writing (Thoreau, Muir, Burroughs, Emerson,
Whitman) rather than verbatim Project Gutenberg passages. The same
crawl + dual-review process applies; richer retrieval over direct
quotes is a future expansion that the pipeline already supports.

## Token budget

Each chunk targets **~150 tokens** (~100-150 words). At retrieval time,
the architecture in ADR-001 calls for k=1 retrieval with a token budget
of ~200 tokens — most chunks fit comfortably in that budget without
truncation.

The `summary` field is ~30-50 tokens, supporting the
"summary-first / full-chunk on demand" two-stage retrieval strategy.

## Embedding pipeline (next step)

This corpus has no embeddings yet. To produce them:

1. Pick the embedding model: `sentence-transformers/all-MiniLM-L6-v2`
   (384-dim, ~80 MB, Core ML-convertible)
2. Run a Python script: load each JSONL → `embeddings.f16` blob aligned
   by line index
3. Build a USearch HNSW index over the embeddings
4. Bundle: `chunks.jsonl` + `embeddings.f16` + `index.usearch` + `manifest.json`
5. Drop into `Documents/RAG/<subject>/` on the iOS side

See `docs/ADR-001-rag-architecture.md` (when written) for the full
deployment story.

## Counts

```
geology.jsonl   25 chunks  ~3,400 words  ~5,000 tokens
plants.jsonl    25 chunks  ~3,400 words  ~5,000 tokens
physics.jsonl   20 chunks  ~2,700 words  ~4,000 tokens
english.jsonl   20 chunks  ~2,700 words  ~4,000 tokens
                ─────────  ─────────────  ──────────────
total           90 chunks  ~12,200 words  ~18,000 tokens
```

Tiny. Fits in app bundle if needed for a no-download-required demo.
