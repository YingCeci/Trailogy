# Trailogy — architecture (one page)

```
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — PIPELINE   (the "brain"; untouched by the redesign)         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   GemmaService.swift       ValidationRunner.swift      SpeechRecog…    │
│   ──────────────────       ──────────────────────      ─────────────   │
│   Gemma 4 E2B INT4         Kokoro TTS 82M FP32         Apple SFSR      │
│   ~2.5 GB MLX active       ~324 MB MLX active          on-device       │
│   loadIfNeeded() ▸         synthesize(text,speed) ▸    startRecording()│
│     mmap weights (10–30s)    chunked synth + play        AVAudioEngine │
│   streamResponse(to:) ▸    Two-phase serial unload     stopRecording() │
│     AsyncStream<String>      → frees ~324 MB           Live `transcript│
│   unload() ▸ frees 2.5 GB                                              │
│   History capped 20 msgs                                               │
│                                                                        │
│   All three: ObservableObject.   Owned in ContentView as @StateObject. │
│   Children read via @EnvironmentObject.                                │
└────────────────────────────────────────────────────────────────────────┘
                                  │
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — STATE + DATA                                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   AppRouter.swift          TrailData.swift          Theme.swift        │
│   ───────────────          ────────────────         ─────────────      │
│   ObservableObject          Static catalogue:        Color tokens +    │
│   • screen: AppScreen       • kildoo (5 stops)       Font helpers      │
│   • currentTrail: Trail     • hellsHollow (3)        from mockups.html │
│   • debugVisible: Bool      • tranquil (4)                             │
│   Methods: choose, begin,   ↑ pre-written text;      AppColor.lime     │
│    endTour, openDebug, …      not Gemma-generated    AppColor.ink…     │
└────────────────────────────────────────────────────────────────────────┘
                                  │
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — VIEWS  (Views/*.swift)                                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ContentView.swift  ←─ root; owns @StateObjects & AppRouter           │
│        │                                                               │
│        │   switch router.screen {                                      │
│        │                                                               │
│        ├── .picker  ─▶ PickerView                                      │
│        │                  • tap card  → router.choose(trail)           │
│        │                  • tap Debug → router.openDebug() ────┐       │
│        │                                                       │       │
│        ├── .detail  ─▶ DetailView                              │       │
│        │                  • renders TrailMapView               │       │
│        │                  • tap Begin → router.begin()         │       │
│        │                                                       │       │
│        ├── .walking ─▶ WalkingView   ← the big one (~600 lines)│       │
│        │                  • subviews:                          │       │
│        │                      progressBar / stopHero           │       │
│        │                      lyricStack / quietIndicator      │       │
│        │                      photoContextBar / askTranscript  │       │
│        │                      bottomControls / moreSheet       │       │
│        │                  • modals:                            │       │
│        │                      CameraView (visual-only)         │       │
│        │                      TourMapView                      │       │
│        │                  • mic gesture wires:                 │       │
│        │                      Speech → Gemma → Kokoro          │       │
│        │                  • End tour → router.endTour()        │       │
│        │                                                       │       │
│        └── .journal ─▶ JournalView                             │       │
│                           • tap ✕ → router.closeJournal()      │       │
│                                                                │       │
│   .sheet(isPresented: $router.debugVisible) ◀──────────────────┘       │
│        └─▶ DebugView   ← original UI verbatim                          │
│              memory profiler · voice picker · speed slider             │
│              manual Ask · debug TTS · reset conversation               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Q&A turn — the only path that lights up Layer 1 live

```
   WalkingView mic (hold-to-speak, DragGesture(minimumDistance: 0))
            │
            │  finger DOWN
            ▼
    startHold()
       ├─ stopLyricLoop()                          (pause auto-narration)
       ├─ isHolding = true                         (green overlay shows)
       └─ speech.startRecording() ◇─▶ live `transcript` updates
            │
            │  finger UP
            ▼
    endHold()
       ├─ speech.stopRecording()
       └─ Task: sleep 600 ms, then runAsk(prompt: transcript)
            │
            ▼
    runAsk(prompt:)
       │  isAnswering = true; answerText = ""
       │
       ├─▶ await gemma.loadIfNeeded()              (10–30 s first turn)
       │
       ├─▶ for try await chunk in gemma.streamResponse(to: prompt):
       │       answerText += chunk                  (tokens stream live)
       │
       ├─▶ gemma.unload()                           (frees 2.5 GB)
       │
       ├─▶ tts.synthesize(text: fullText)           (Kokoro speaks)
       │
       └─▶ sleep 8 s
              answerText = ""
              startLyricLoop()                      (resume narration)
```

---

## Screen state machine

```
        ┌──────────────┐   choose(trail)   ┌──────────────┐
        │  PickerView  │ ────────────────▶ │  DetailView  │
        └──────────────┘                   └──────────────┘
            ▲     ▲                              │
            │     │                              │  begin()
            │     │ openDebug()                  ▼
            │     │                          ┌──────────────┐
            │     │                          │  WalkingView │
            │     │                          └──────────────┘
            │     │                              │
            │     │                              │  endTour()
            │     ▼                              ▼
            │  ┌──────────┐                 ┌──────────────┐
            │  │ DebugView│ closeDebug()    │ JournalView  │
            │  │ (sheet)  │ ───────────▶┐   └──────────────┘
            │  └──────────┘             │       │
            │                           │       │  closeJournal()
            └───────────────────────────┴───────┘
```

All transitions are mutations on `AppRouter`. No `NavigationStack`,
no `NavigationLink` — just a `@Published` enum that ContentView
switches on with a fade.
