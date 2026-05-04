// TrailMapShape.swift
// SwiftUI re-implementation of the SVG trail map from design/mockups.html
// (the Kildoo Trail loop along Slippery Rock Creek, with 5 numbered
// waypoints). Used both in DetailView (full-screen pre-tour map) and
// inside the in-walk full-screen TourMapView.
//
// The shape is normalized to a 320×540 reference grid to match the SVG.
// Pass `activeStop` (1-based) to highlight the current stop with a lime
// pulsing halo; pass `passedThroughStop` to draw the east-leg as solid
// lime up to that stop and dashed past it.

import SwiftUI

struct TrailMapView: View {
    /// 1-based stop number that is currently active (0 = none).
    var activeStop: Int = 1
    /// 1-based stop number up to which the user has walked. The east-leg
    /// path is drawn solid lime up to (and through) this stop, dashed past.
    /// 0 = nothing walked yet (pre-tour); same as activeStop on first stop.
    var passedThroughStop: Int = 0

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            // The mockup SVG is 320x540; preserve that aspect via scaling.
            let scaleX = w / 320.0
            let scaleY = h / 540.0
            let s = min(scaleX, scaleY)
            // Center in the available space
            let offsetX = (w - 320 * s) / 2
            let offsetY = (h - 540 * s) / 2

            ZStack {
                // Topo speckles
                ForEach(speckles, id: \.self) { p in
                    Circle()
                        .frame(width: 1.6, height: 1.6)
                        .foregroundStyle(Color.white.opacity(0.18 * 0.6))
                        .position(x: offsetX + p.x * s, y: offsetY + p.y * s)
                }

                // Slippery Rock Creek — wide soft band + thin core line
                creekShape
                    .stroke(Color(red: 140/255, green: 180/255, blue: 200/255).opacity(0.18 * 0.6),
                            style: StrokeStyle(lineWidth: 14, lineCap: .round))
                    .frame(width: 320, height: 540)
                    .scaleEffect(s, anchor: .topLeading)
                    .offset(x: offsetX, y: offsetY)

                creekShape
                    .stroke(Color(red: 140/255, green: 180/255, blue: 200/255).opacity(0.55),
                            style: StrokeStyle(lineWidth: 2.8, lineCap: .round))
                    .frame(width: 320, height: 540)
                    .scaleEffect(s, anchor: .topLeading)
                    .offset(x: offsetX, y: offsetY)

                // East leg — passed portion solid lime
                if passedThroughStop > 0 {
                    eastLegPassedPath(uptoStop: passedThroughStop)
                        .stroke(AppColor.lime.opacity(0.7),
                                style: StrokeStyle(lineWidth: 1.8, lineCap: .round))
                        .frame(width: 320, height: 540)
                        .scaleEffect(s, anchor: .topLeading)
                        .offset(x: offsetX, y: offsetY)
                }
                // East leg — full dashed gray (drawn under the passed portion)
                eastLegFullPath
                    .stroke(AppColor.ink100.opacity(0.4),
                            style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [3, 3]))
                    .frame(width: 320, height: 540)
                    .scaleEffect(s, anchor: .topLeading)
                    .offset(x: offsetX, y: offsetY)

                // West leg — return path, always dashed
                westLegPath
                    .stroke(AppColor.ink100.opacity(0.30),
                            style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [3, 3]))
                    .frame(width: 320, height: 540)
                    .scaleEffect(s, anchor: .topLeading)
                    .offset(x: offsetX, y: offsetY)

                // Waypoints
                ForEach(waypoints, id: \.number) { wp in
                    waypointMarker(wp: wp, s: s, offsetX: offsetX, offsetY: offsetY)
                }

                // Compass (top-right)
                compass
                    .position(x: offsetX + 286 * s, y: offsetY + 50 * s)

                // Scale bar (bottom-left)
                scaleBar
                    .position(x: offsetX + 50 * s, y: offsetY + 510 * s)
            }
        }
    }

    // MARK: - Sub-views

    @ViewBuilder
    private func waypointMarker(wp: WaypointPoint, s: CGFloat, offsetX: CGFloat, offsetY: CGFloat) -> some View {
        let isActive = wp.number == activeStop
        let isPassed = wp.number < activeStop || wp.number <= passedThroughStop && wp.number != activeStop
        let cx = offsetX + wp.x * s
        let cy = offsetY + wp.y * s

        ZStack {
            if isActive {
                Circle()
                    .fill(AppColor.lime.opacity(0.20))
                    .frame(width: 48 * s, height: 48 * s)
                Circle()
                    .fill(AppColor.lime.opacity(0.36))
                    .frame(width: 32 * s, height: 32 * s)
                Circle()
                    .fill(AppColor.lime)
                    .frame(width: 26 * s, height: 26 * s)
                Text("\(wp.number)")
                    .font(.system(size: 13, weight: .heavy))
                    .foregroundStyle(AppColor.limeText)
            } else if isPassed {
                Circle()
                    .fill(AppColor.lime.opacity(0.18))
                    .frame(width: 26 * s, height: 26 * s)
                    .overlay(Circle().stroke(AppColor.lime.opacity(0.6), lineWidth: 1.3))
                Text("\(wp.number)")
                    .font(.system(size: 12, weight: .heavy))
                    .foregroundStyle(AppColor.lime.opacity(0.85))
            } else {
                Circle()
                    .fill(AppColor.glassDark93)
                    .frame(width: 26 * s, height: 26 * s)
                    .overlay(Circle().stroke(AppColor.ink100.opacity(0.7), lineWidth: 1.3))
                Text("\(wp.number)")
                    .font(.system(size: 12, weight: .heavy))
                    .foregroundStyle(AppColor.ink100.opacity(0.92))
            }

            // Label outside the marker
            Text(wp.label)
                .font(.system(size: 12, weight: .bold))
                .tracking(0.4)
                .foregroundStyle(isActive ? AppColor.lime : (isPassed ? AppColor.ink100.opacity(0.55) : AppColor.ink100.opacity(0.85)))
                .shadow(color: AppColor.mapBg, radius: 1.5)
                .offset(wp.labelOffset)
        }
        .position(x: cx, y: cy)
    }

    private var compass: some View {
        VStack(spacing: 0) {
            Text("N")
                .font(.system(size: 10, weight: .heavy))
                .foregroundStyle(AppColor.ink100.opacity(0.55))
            Rectangle()
                .frame(width: 1, height: 16)
                .foregroundStyle(AppColor.ink100.opacity(0.5))
        }
        .opacity(0.55)
    }

    private var scaleBar: some View {
        VStack(spacing: 4) {
            Rectangle()
                .frame(width: 60, height: 1.4)
                .foregroundStyle(AppColor.ink100.opacity(0.5))
            Text("¼ MILE")
                .font(.system(size: 9, weight: .heavy))
                .tracking(0.6)
                .foregroundStyle(AppColor.ink100.opacity(0.55))
        }
        .opacity(0.6)
    }

    // MARK: - Geometry

    /// Mostly-vertical sinuous creek through the map.
    private var creekShape: Path {
        Path { p in
            p.move(to: CGPoint(x: 160, y: 52))
            p.addQuadCurve(to: CGPoint(x: 158, y: 160), control: CGPoint(x: 150, y: 100))
            p.addQuadCurve(to: CGPoint(x: 152, y: 300), control: CGPoint(x: 166, y: 230))
            p.addQuadCurve(to: CGPoint(x: 162, y: 440), control: CGPoint(x: 148, y: 370))
            p.addQuadCurve(to: CGPoint(x: 160, y: 510), control: CGPoint(x: 170, y: 490))
        }
    }

    /// East-bank trail (full path, all waypoints).
    private var eastLegFullPath: Path {
        Path { p in
            p.move(to: CGPoint(x: 160, y: 80))
            p.addQuadCurve(to: CGPoint(x: 200, y: 170), control: CGPoint(x: 195, y: 105))
            p.addQuadCurve(to: CGPoint(x: 210, y: 280), control: CGPoint(x: 207, y: 225))
            p.addQuadCurve(to: CGPoint(x: 195, y: 415), control: CGPoint(x: 215, y: 350))
            p.addQuadCurve(to: CGPoint(x: 160, y: 498), control: CGPoint(x: 175, y: 475))
        }
    }

    /// East-bank trail truncated up to the user's walked stop. Mockup
    /// shows lime path to stop 3 (Kildoo Falls) when active.
    private func eastLegPassedPath(uptoStop: Int) -> Path {
        Path { p in
            switch uptoStop {
            case 1:
                // Just at stop 1 — no walked portion
                break
            case 2:
                p.move(to: CGPoint(x: 160, y: 80))
                p.addQuadCurve(to: CGPoint(x: 200, y: 170), control: CGPoint(x: 195, y: 105))
            case 3:
                p.move(to: CGPoint(x: 160, y: 80))
                p.addQuadCurve(to: CGPoint(x: 200, y: 170), control: CGPoint(x: 195, y: 105))
                p.addQuadCurve(to: CGPoint(x: 210, y: 280), control: CGPoint(x: 207, y: 225))
            case 4:
                p.move(to: CGPoint(x: 160, y: 80))
                p.addQuadCurve(to: CGPoint(x: 200, y: 170), control: CGPoint(x: 195, y: 105))
                p.addQuadCurve(to: CGPoint(x: 210, y: 280), control: CGPoint(x: 207, y: 225))
                p.addQuadCurve(to: CGPoint(x: 195, y: 415), control: CGPoint(x: 215, y: 350))
            default:
                p.move(to: CGPoint(x: 160, y: 80))
                p.addQuadCurve(to: CGPoint(x: 200, y: 170), control: CGPoint(x: 195, y: 105))
                p.addQuadCurve(to: CGPoint(x: 210, y: 280), control: CGPoint(x: 207, y: 225))
                p.addQuadCurve(to: CGPoint(x: 195, y: 415), control: CGPoint(x: 215, y: 350))
                p.addQuadCurve(to: CGPoint(x: 160, y: 498), control: CGPoint(x: 175, y: 475))
            }
        }
    }

    /// West-bank return path (always dashed).
    private var westLegPath: Path {
        Path { p in
            p.move(to: CGPoint(x: 160, y: 80))
            p.addQuadCurve(to: CGPoint(x: 122, y: 180), control: CGPoint(x: 125, y: 105))
            p.addQuadCurve(to: CGPoint(x: 125, y: 295), control: CGPoint(x: 118, y: 240))
            p.addQuadCurve(to: CGPoint(x: 140, y: 420), control: CGPoint(x: 130, y: 360))
            p.addQuadCurve(to: CGPoint(x: 160, y: 498), control: CGPoint(x: 150, y: 475))
        }
    }

    private struct WaypointPoint {
        let number: Int
        let x: CGFloat
        let y: CGFloat
        let label: String
        let labelOffset: CGSize
    }

    private let waypoints: [WaypointPoint] = [
        .init(number: 1, x: 160, y: 80,  label: "Covered Bridge & Mill",
              labelOffset: CGSize(width: 0, height: 38)),
        .init(number: 2, x: 200, y: 170, label: "Layered Cliffs",
              labelOffset: CGSize(width: 60, height: 0)),
        .init(number: 3, x: 210, y: 280, label: "Kildoo Falls",
              labelOffset: CGSize(width: 60, height: 0)),
        .init(number: 4, x: 160, y: 498, label: "Eckert Bridge",
              labelOffset: CGSize(width: 0, height: -28)),
        .init(number: 5, x: 125, y: 295, label: "Slippery Rock",
              labelOffset: CGSize(width: -60, height: 0))
    ]

    private let speckles: [CGPoint] = [
        CGPoint(x: 50, y: 80),  CGPoint(x: 80, y: 200),
        CGPoint(x: 60, y: 380), CGPoint(x: 260, y: 100),
        CGPoint(x: 270, y: 240), CGPoint(x: 250, y: 400),
        CGPoint(x: 40, y: 460), CGPoint(x: 280, y: 460)
    ]
}

#Preview {
    ZStack {
        AppColor.mapBg.ignoresSafeArea()
        TrailMapView(activeStop: 1, passedThroughStop: 0)
            .padding()
    }
}
