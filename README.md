# Trailogy

**An audio-first AI hiking companion. Entirely on-device.**

Trailogy is an iOS app for the Kaggle *Gemma 4 for Good* hackathon.
Pick a trail, hit Begin, and it narrates your walk through five
curator-authored stops — geology, plants, history. At any point you
can press the mic and ask a follow-up; Gemma 4 E2B answers, Kokoro
speaks the answer back, and you keep walking. No network needed.

```
[picker] ──▶ [trail detail + map] ──▶ [walking tour] ──▶ [recap]
                ↓                            ↓
             Download                     hold-to-ask  ──▶  Gemma 4 ──▶  Kokoro
             (~9 MB of                                  RAG-grounded   (TTS plays
              trail images)                             on a curated   the answer)
                                                        nature corpus
```

## What it actually does

- **Three audio-guided trails** in western PA: Kildoo (McConnells Mill),
  Old Field & Jennings (wildflower reserve), Tranquil (Frick Park).
  Each has 5 narrated stops with curator-authored content.
- **Hold-to-ask** at any point. Speech → Apple SFSpeechRecognizer →
  Gemma 4 E2B (text or VLM if you tapped Camera first) →
  Kokoro TTS reply, all on the device.
- **Retrieval-grounded answers**: every question is augmented with
  top-k chunks from a small authored corpus (geology + plants + physics
  + english, ~90 chunks total) via MiniLM embeddings. Picks per-trail
  defaults; overridable in DebugView.
- **Look-for / payoff engagement loop**: each stop seeds a thing to
  watch for as you walk; the next stop's narration pays it off.
- **Recap**: after the tour, a "discoveries" digest of what you
  learned — anchor-number cards with a brand trailmark.
- **Offline image cache**: tapping Download on the detail view
  fetches every cover + stop image (~9 MB/trail) to local disk so the
  whole tour renders without a network.

## Requirements

- macOS with **Xcode 26+**
- **iPhone 15 Pro / 16 Pro / 17 Pro or newer** running **iOS 18.0+**
  (8 GB RAM minimum — Gemma 4 E2B INT4 is ~3.5 GB resident, and the
  app uses the `com.apple.developer.kernel.increased-memory-limit`
  entitlement to clear the per-process cap)
- The **iOS Simulator does not support MLX** — must deploy to a
  physical device
- Apple Developer account (free Apple ID is fine for sideloading)
- ~7 GB free disk space (~6 GB bundle + temporary build artifacts)

## Quick start

```bash
git clone git@github.com:YingCeci/Trailogy.git
cd Trailogy

# 1) Kokoro TTS model + voices (~330 MB)
#    Also fetches MiniLM (~87 MB) for the RAG embedder.
bash scripts/fetch-models.sh

# 2) Gemma 4 E2B INT4 (~2.8 GB) — stock mlx-community baseline
bash scripts/fetch-gemma.sh
#    or, if the primary repo has a PLE quantization bug:
# bash scripts/fetch-gemma.sh --backup
#
#    OR, to use a TimS-ml finetune (gated repo, needs HF_TOKEN):
# HF_TOKEN=hf_... bash scripts/fetch-gemma-finetune.sh r8-a8-nokl-step9000_mlx_g64

# 3) Generate the Xcode project from project.yml
bash scripts/generate-project.sh
```

Then in Xcode:

1. Open `HikeCompanion.xcodeproj` (the Xcode target keeps its
   legacy name; the user-facing app is "Trailogy")
2. Select the project → TARGETS → HikeCompanion → **Signing &
   Capabilities** → set Team to your Apple ID
3. Plug in iPhone, pick it in the device dropdown
4. **⌘R**

First launch takes ~10–30 s to load Gemma + Kokoro + MiniLM into
MLX. Subsequent narration is near-realtime on iPhone 15 Pro and
faster on newer Pro devices.

## Architecture

A short overview; full details in
[`ARCHITECTURE.md`](ARCHITECTURE.md):

```
TOP LAYER  (SwiftUI views, driven by AppRouter state machine)
  PickerView ─ DetailView ─ WalkingView ─ JournalView
                    │           │
                    │           ├─ WalkingDots (3-dot stride animation)
                    │           ├─ CameraView   (Phase 3a — photo Q&A)
                    │           └─ TourMapView  (in-tour fullscreen map)

ROUTER
  AppRouter — @MainActor ObservableObject
    screen ∈ {picker, detail, walking, journal}
    downloadedTrailIDs : Set<String>   (disk-backed via ImageStore)
    walkedAt           : [String: Date]
    ragSubjectsOverride: Set<RAGService.Subject>?

SERVICES (on-device pipeline; long-lived)
  GemmaService       Gemma 4 E2B via mlx-swift-lm
                     text or VLM mode (per turn)
  RAGService         MiniLM embedder + per-subject corpora
                     multi-subject retrieve, top-k merge
  ValidationRunner   Kokoro TTS via MLX (KokoroSwift)
  SpeechRecognizer   Apple SFSpeechRecognizer (on-device)
  CameraController   AVFoundation; downscales to 1280px
  ImageStore         on-disk trail-image cache for offline view

DATA
  TrailData.swift    3 trails × {stops, learnings, RAG defaults, ...}
  Resources/Models/  Gemma + Kokoro + MiniLM (~3.2 GB total, gitignored)
  Resources/RAG/     4 subject corpora + f16 embeddings (~150 KB)
```

The walking-tour engine is a four-state FSM (`atStop → between →
approaching → atStop … → complete`) with a concurrent Ask FSM that
gates the phase timer through shared flags (paused, holding,
answering, narrating). See
[`design/README.md`](design/README.md) item 14 for the full state
machine.

## Project layout

```
Trailogy/
├── README.md                       this file
├── ARCHITECTURE.md                 one-page architecture overview
├── AGENTS.md / CLAUDE.md           context for AI coding sessions
├── project.yml                     xcodegen spec → .xcodeproj
├── design/
│   ├── README.md                   design rationale (synced from mockup repo)
│   ├── mockups.html                executable design spec
│   └── logo.png                    Trailogy brand mark (source)
├── scripts/
│   ├── fetch-models.sh             Kokoro + MiniLM
│   ├── fetch-gemma.sh              stock Gemma 4 E2B INT4
│   ├── fetch-gemma-finetune.sh     TimS-ml finetune (gated, HF_TOKEN)
│   ├── strip-gemma-audio.py        drops the unused audio tower (~930 MB)
│   ├── embed-rag-corpus.py         (re)generates RAG embeddings
│   ├── query-rag-corpus.py         CLI to test retrieval quality
│   └── generate-project.sh         xcodegen wrapper
├── external/                       vendored SPM deps
│   ├── kokoro-ios/                 KokoroSwift (MLX pin relaxed)
│   ├── MisakiSwift/                G2P for Kokoro
│   ├── MLXUtilsLibrary/
│   └── mlx-swift-lm/               Gemma 4 loader (LLM + VLM)
└── HikeCompanion/                  Xcode target sources (legacy name)
    ├── HikeCompanionApp.swift      @main
    ├── ContentView.swift           top-level router → 4 screens
    ├── AppRouter.swift             screen state + downloaded/walked sets
    ├── GemmaService.swift          Gemma 4 inference (text + VLM)
    ├── RAGService.swift            MiniLM embedder + multi-subject retrieve
    ├── ValidationRunner.swift      Kokoro TTS wrapper
    ├── SpeechRecognizer.swift      ASR
    ├── CameraController.swift      photo capture
    ├── ImageStore.swift            on-disk trail-image cache
    ├── TrailData.swift             3 trails, stops, learnings, segments
    ├── MemoryStats.swift           RSS / footprint logging
    ├── Theme.swift                 AppColor + AppFont
    ├── Views/                      11 SwiftUI views
    └── Resources/
        ├── Models/                 Gemma + Kokoro + MiniLM (gitignored)
        └── RAG/                    JSONL chunks + f16 embeddings
```

## Cost / size

| | |
|---|---|
| App bundle (debug, on device) | ~6.3 GB (Gemma 3.2 GB + Kokoro 330 MB + MiniLM 87 MB + RAG 150 KB + code + frameworks) |
| First-build time | 2–4 min on M-series Mac (clean) |
| First-launch latency | 10–30 s (MLX graph compile + model loads) |
| Subsequent ask latency | 1–3 s to first token, then ~3× realtime narration |
| Per-trail offline image cache | ~9 MB (1 cover + 5 stops × ~1.5 MB Wikimedia) |

## Development docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — one-page system map
- [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) — context
  checkpoints for AI coding sessions (paste-into-the-window primers)
- [`design/README.md`](design/README.md) — design rationale, synced
  from the upstream [Trailogy-UI](https://github.com/YingCeci/Trailogy-UI)
  mockup repo. Numbered design decisions explain the *why* of each
  visual choice; the mockup HTML is the executable spec.
- [`design/mockups.html`](design/mockups.html) — interactive prototype.
  Every iOS view has a mockup counterpart.

## License

App code: **MIT**.

Upstream packages:
- [`mlalma/kokoro-ios`](https://github.com/mlalma/kokoro-ios) — MIT
- [`ml-explore/mlx-swift`](https://github.com/ml-explore/mlx-swift) — MIT
- [`ml-explore/mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) — MIT
- [`huggingface/swift-transformers`](https://github.com/huggingface/swift-transformers) — Apache 2.0

Model weights:
- Gemma 4 E2B — [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- Kokoro TTS 82M — [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0)
- MiniLM-L6-v2 — [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache 2.0)
