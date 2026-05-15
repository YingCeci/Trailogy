// TrailMapShape.swift  (filename retained for callsite compat;
// contents replaced by a MapKit-backed implementation)
//
// Native SwiftUI Map view that renders any Trail's actual polyline +
// stop annotations on top of Apple Maps' standard dark tiles. Replaces
// the previous hand-drawn Kildoo-only SVG renderer (which made
// Tranquil and Old Field show Kildoo's loop with wrong stop counts).
//
// Mockup parity: design/mockups.html uses Leaflet + CARTO Dark Matter
// tiles; iOS uses MapKit's standard map style with the system dark
// color scheme — same visual register (real terrain + roads + rivers
// underneath, lime polyline on top, lime markers per stop).
//
// API contract preserved from the previous file: same
// `activeStop` and `passedThroughStop` 1-based parameters. Now also
// takes `trail` so the view knows which trail's coordinates to draw.

import MapKit
import SwiftUI

struct TrailMapView: View {
    /// The trail whose path + stops to render. Reads
    /// `trail.path` for the polyline and `trail.stops[*].coordinate`
    /// for the per-stop annotations.
    let trail: Trail
    /// 1-based stop number that is currently active. Drawn with a
    /// filled lime circle and a numeric label; pulses subtly via the
    /// shadow halo. 0 = no active stop (e.g. pre-tour overview).
    var activeStop: Int = 1
    /// 1-based stop number up to which the user has walked. Stops at
    /// or below this number render as "passed" (lime outlined). Stops
    /// past it render as "future" (dim filled circle). The active
    /// stop's marker overrides whichever category it would otherwise
    /// fall into.
    var passedThroughStop: Int = 0

    /// Initial camera region fit around the polyline (with padding).
    /// Computed once from `trail.path` (or `stops` as fallback) so the
    /// whole trail fits in view on first render. After that, the user
    /// can pan/zoom freely; the camera doesn't auto-recenter on stop
    /// changes (would feel jumpy during a tour).
    @State private var cameraPosition: MapCameraPosition

    init(trail: Trail, activeStop: Int = 1, passedThroughStop: Int = 0) {
        self.trail = trail
        self.activeStop = activeStop
        self.passedThroughStop = passedThroughStop
        self._cameraPosition = State(
            initialValue: .region(Self.initialRegion(for: trail))
        )
    }

    var body: some View {
        Map(position: $cameraPosition) {
            // The lime trail polyline. `MapPolyline(coordinates:)`
            // renders a Metal-backed line on top of Apple Maps tiles;
            // the system handles antialiasing and tile composition.
            if !trail.path.isEmpty {
                MapPolyline(coordinates: trail.path)
                    .stroke(AppColor.lime.opacity(0.95), lineWidth: 4)
            }

            // Per-stop annotations. We use `Annotation` (not `Marker`)
            // so we can render our custom-styled lime/dark bubble.
            ForEach(trail.stops) { stop in
                Annotation(stop.name, coordinate: stop.coordinate, anchor: .center) {
                    stopMarker(for: stop)
                }
                .annotationTitles(.hidden)
            }
        }
        // Standard Apple Maps tiles, flat (not 3D), in dark mode —
        // matches the design language. The app forces .dark via
        // ContentView, so the standard style renders dark
        // automatically.
        .mapStyle(.standard(elevation: .flat))
        // Disable the auto-controls we don't want: no compass, no
        // user location dot (we're not tracking the user yet), no
        // scale bar. Trail view is read-only contextual; pan/zoom
        // stays available by default.
        .mapControlVisibility(.hidden)
        // Tint propagates to standard MapKit controls if they ever
        // appear; harmless when hidden.
        .tint(AppColor.lime)
    }

    // MARK: - Per-stop marker

    @ViewBuilder
    private func stopMarker(for stop: TrailStop) -> some View {
        let isActive = stop.number == activeStop
        let isPassed = stop.number < activeStop || (stop.number <= passedThroughStop && !isActive)
        ZStack {
            if isActive {
                // Soft outer halo for emphasis (mockup parity:
                // CARTO Dark Matter active waypoint pulses lime).
                Circle()
                    .fill(AppColor.lime.opacity(0.22))
                    .frame(width: 44, height: 44)
                Circle()
                    .fill(AppColor.lime)
                    .frame(width: 28, height: 28)
                    .shadow(color: AppColor.lime.opacity(0.6), radius: 8)
                Text("\(stop.number)")
                    .font(.system(size: 13, weight: .heavy))
                    .foregroundStyle(AppColor.limeText)
            } else if isPassed {
                Circle()
                    .fill(AppColor.lime.opacity(0.20))
                    .frame(width: 26, height: 26)
                    .overlay(Circle().stroke(AppColor.lime.opacity(0.7), lineWidth: 1.3))
                Text("\(stop.number)")
                    .font(.system(size: 12, weight: .heavy))
                    .foregroundStyle(AppColor.lime.opacity(0.85))
            } else {
                Circle()
                    .fill(AppColor.glassDark93)
                    .frame(width: 26, height: 26)
                    .overlay(Circle().stroke(AppColor.ink100.opacity(0.7), lineWidth: 1.3))
                Text("\(stop.number)")
                    .font(.system(size: 12, weight: .heavy))
                    .foregroundStyle(AppColor.ink100.opacity(0.92))
            }
        }
    }

    // MARK: - Camera framing

    /// Compute the initial visible region from a trail's polyline.
    /// Falls back to stop coordinates if the path is empty (which
    /// shouldn't happen with the current data but guards against a
    /// future trail being added without a path).
    private static func initialRegion(for trail: Trail) -> MKCoordinateRegion {
        let coords: [CLLocationCoordinate2D]
        if !trail.path.isEmpty {
            coords = trail.path
        } else {
            coords = trail.stops.map(\.coordinate)
        }
        guard let firstLat = coords.map(\.latitude).min(),
              let lastLat = coords.map(\.latitude).max(),
              let firstLng = coords.map(\.longitude).min(),
              let lastLng = coords.map(\.longitude).max()
        else {
            // Defensive fallback — center on Pittsburgh.
            return MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: 40.44, longitude: -79.99),
                span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
            )
        }
        let center = CLLocationCoordinate2D(
            latitude: (firstLat + lastLat) / 2,
            longitude: (firstLng + lastLng) / 2
        )
        // Pad the bounding box by 35% on each axis so the polyline
        // sits comfortably inside the frame with breathing room for
        // the stop annotations (which can extend a few points past
        // the line itself).
        let latPad = max((lastLat - firstLat) * 1.35, 0.005)
        let lngPad = max((lastLng - firstLng) * 1.35, 0.005)
        return MKCoordinateRegion(
            center: center,
            span: MKCoordinateSpan(latitudeDelta: latPad, longitudeDelta: lngPad)
        )
    }
}

#Preview {
    ZStack {
        AppColor.mapBg.ignoresSafeArea()
        TrailMapView(trail: TrailData.kildoo, activeStop: 3, passedThroughStop: 2)
            .padding()
    }
    .preferredColorScheme(.dark)
}
