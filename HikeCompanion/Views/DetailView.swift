// DetailView.swift
// Pre-tour detail screen: header (back arrow + trail name), full-screen
// trail map, bottom action card with stats + Begin button.
// Mockup: design/mockups.html → .detail view (.dm-* classes).

import SwiftUI

struct DetailView: View {
    @EnvironmentObject var router: AppRouter

    var trail: Trail { router.currentTrail }

    var body: some View {
        ZStack {
            AppColor.mapBg.ignoresSafeArea()

            VStack(spacing: 0) {
                // Top nav: back arrow + centered title
                HStack(alignment: .center, spacing: 12) {
                    Button {
                        router.backToPicker()
                    } label: {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(AppColor.ink100)
                            .frame(width: 32, height: 32)
                            .background(.black.opacity(0.4), in: Circle())
                    }

                    VStack(spacing: 2) {
                        Text(trail.name)
                            .font(AppFont.sans(17, .bold))
                            .foregroundStyle(AppColor.ink100)
                            .tracking(-0.3)
                        Text(trail.parkLocation.uppercased())
                            .font(AppFont.sans(10, .heavy))
                            .tracking(1.8)
                            .foregroundStyle(AppColor.ink60)
                    }
                    .frame(maxWidth: .infinity)

                    // Spacer to balance the back button
                    Color.clear.frame(width: 32, height: 32)
                }
                .padding(.horizontal, 18)
                .padding(.top, 60)
                .padding(.bottom, 8)

                // Full-screen trail map
                ZStack {
                    // Subtle radial highlights from the mockup
                    RadialGradient(
                        colors: [AppColor.lime.opacity(0.05), .clear],
                        center: UnitPoint(x: 0.5, y: 0.3),
                        startRadius: 20,
                        endRadius: 200
                    )

                    TrailMapView(activeStop: 1, passedThroughStop: 0)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                }

                // Bottom action card
                VStack(spacing: 14) {
                    // One-line trail tagline (mockup commit 7c5ba6c —
                    // sits above the stats row and below the chrome).
                    Text(trail.summary)
                        .font(AppFont.sans(14.5, .regular))
                        .foregroundStyle(AppColor.ink100.opacity(0.92))
                        .multilineTextAlignment(.center)
                        .lineSpacing(2)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 4)

                    HStack(spacing: 10) {
                        statText(value: formattedMiles, suffix: "mi")
                        smallDot
                        statText(value: durationLabel.0, suffix: durationLabel.1)
                        smallDot
                        statText(value: "\(trail.stopCount)", suffix: "stops")
                        smallDot
                        Text(trail.difficulty)
                            .foregroundStyle(AppColor.ink80)
                    }
                    .font(AppFont.sans(13.5, .medium))

                    Button {
                        router.begin()
                    } label: {
                        Text("Begin")
                            .font(AppFont.sans(17, .semibold))
                            .foregroundStyle(AppColor.limeText)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 18)
                            .background(AppColor.lime)
                            .clipShape(Capsule())
                    }
                    .buttonStyle(LimePressStyle())
                }
                .padding(.horizontal, 22)
                .padding(.top, 18)
                .padding(.bottom, 22)
                .background(
                    Color(red: 15/255, green: 16/255, blue: 13/255).opacity(0.95)
                        .overlay(
                            Rectangle()
                                .frame(height: 1)
                                .foregroundStyle(AppColor.hairline),
                            alignment: .top
                        )
                )
            }
        }
    }

    private var formattedMiles: String {
        let v = trail.distanceMiles
        return v == floor(v) ? String(format: "%.0f", v) : String(format: "%.1f", v)
    }

    /// Minutes → ("1", "hr") or ("45", "min")
    private var durationLabel: (String, String) {
        let m = trail.durationMinutes
        if m >= 60 && m % 60 == 0 {
            return ("\(m/60)", m == 60 ? "hr" : "hrs")
        } else if m >= 60 {
            let h = Double(m) / 60.0
            return (String(format: "%.1f", h), "hr")
        } else {
            return ("\(m)", "min")
        }
    }

    private func statText(value: String, suffix: String) -> some View {
        HStack(spacing: 4) {
            Text(value)
                .font(AppFont.sans(13.5, .bold))
                .foregroundStyle(AppColor.ink100)
            Text(suffix)
                .foregroundStyle(AppColor.ink80)
        }
    }

    private var smallDot: some View {
        Circle()
            .frame(width: 3, height: 3)
            .foregroundStyle(AppColor.ink40)
    }
}

struct LimePressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .opacity(configuration.isPressed ? 0.9 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview {
    DetailView()
        .environmentObject({
            let r = AppRouter()
            r.currentTrail = TrailData.kildoo
            return r
        }())
}
