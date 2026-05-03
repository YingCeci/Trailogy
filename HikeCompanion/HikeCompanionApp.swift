// HikeCompanionApp.swift
// SwiftUI @main entry. Caps MLX GPU memory at launch so Kokoro + Gemma
// can coexist without unbounded growth → iOS jetsam.

import MLX
import SwiftUI

@main
struct HikeCompanionApp: App {

    init() {
        // 100 MB cache (working scratch space for MLX kernels)
        Memory.cacheLimit = 100 * 1024 * 1024
        // 4.5 GB hard ceiling on MLX GPU allocations.
        // Kokoro working set ~700 MB + Gemma 4 E2B INT4 ~3.5 GB ≈ 4.2 GB.
        // iPhone 15/16/17 Pro (8 GB RAM) leaves ~5 GB for app processes
        // before iOS jetsams; this gives us headroom.
        Memory.memoryLimit = 4_500 * 1024 * 1024
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
