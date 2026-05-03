// MemoryStats.swift
// Reads two memory views:
//   1. Process-level (Mach `task_vm_info`):
//        - resident: current physical RAM held by the process
//        - footprint: iOS's own accounting, which is what the OS uses
//                     when deciding to jetsam — usually slightly higher
//                     than `resident` because it includes some compressed
//                     and reclaimable mappings.
//   2. MLX-specific (Memory.snapshot()):
//        - active:  bytes currently held in MLX-allocated MLXArrays
//        - cache:   bytes MLX has freed from arrays but kept in its pool
//                   for reuse (cleared by Memory.clearCache())
//        - peak:    high-water mark since the process started

import Darwin
import Foundation
import MLX

struct MemoryStats {
    let processResidentBytes: UInt64
    let processFootprintBytes: UInt64
    let mlxActiveBytes: Int
    let mlxCacheBytes: Int
    let mlxPeakBytes: Int

    static func current() -> MemoryStats {
        let (res, foot) = Self.processMemory()
        let snap = Memory.snapshot()
        return MemoryStats(
            processResidentBytes: res,
            processFootprintBytes: foot,
            mlxActiveBytes: snap.activeMemory,
            mlxCacheBytes: snap.cacheMemory,
            mlxPeakBytes: snap.peakMemory
        )
    }

    private static func processMemory() -> (resident: UInt64, footprint: UInt64) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return (0, 0) }
        return (UInt64(info.resident_size), UInt64(info.phys_footprint))
    }

    // MARK: - Formatting

    static func formatBytes(_ bytes: UInt64) -> String {
        let mb = Double(bytes) / 1024 / 1024
        if mb >= 1024 {
            return String(format: "%.2f GB", mb / 1024)
        }
        return String(format: "%.0f MB", mb)
    }

    static func formatBytes(_ bytes: Int) -> String {
        formatBytes(UInt64(max(0, bytes)))
    }

    /// Two-line summary suitable for a `Text` in the UI.
    var summary: String {
        """
        Process: footprint \(Self.formatBytes(processFootprintBytes))   resident \(Self.formatBytes(processResidentBytes))
        MLX:     active \(Self.formatBytes(mlxActiveBytes))   cache \(Self.formatBytes(mlxCacheBytes))   peak \(Self.formatBytes(mlxPeakBytes))
        """
    }
}
