// JournalView.swift
// Post-tour summary screen.
// Mockup: design/mockups.html → .journal view.
//
// All content is sample data from TrailData. Real implementation would
// log: actual stops walked, captured photos, asked questions, observed
// species (via on-device CV), and queue a citizen-science upload.

import SwiftUI

struct JournalView: View {
    @EnvironmentObject var router: AppRouter

    @State private var hasShared: Bool = false

    var trail: Trail { router.currentTrail }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            AppColor.screenBg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // Header
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Today")
                            .eyebrowStyle()
                            .padding(.bottom, 10)

                        Text(trail.name)
                            .font(AppFont.sans(30, .bold))
                            .foregroundStyle(AppColor.ink100)
                            .tracking(-0.7)

                        Text(dateLine)
                            .font(AppFont.sans(13, .medium))
                            .foregroundStyle(AppColor.ink60)
                            .padding(.top, 4)
                    }
                    .padding(.horizontal, 28)
                    .padding(.top, 64)
                    .padding(.bottom, 18)

                    // Hero photo
                    AsyncImage(url: trail.coverImageURL) { phase in
                        switch phase {
                        case .success(let img): img.resizable().scaledToFill()
                        default: AppColor.ink25
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .aspectRatio(16.0/10.0, contentMode: .fill)
                    .clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 22)

                    // Stops walked
                    VStack(alignment: .leading, spacing: 0) {
                        Rectangle()
                            .frame(height: 1)
                            .foregroundStyle(AppColor.ink15)
                            .padding(.bottom, 18)

                        Text("Stops you walked")
                            .eyebrowStyle()
                            .padding(.bottom, 4)

                        ForEach(trail.stops) { stop in
                            stopRow(stop: stop)
                        }
                    }
                    .padding(.horizontal, 28)

                    // What you saw
                    VStack(alignment: .leading, spacing: 0) {
                        Rectangle()
                            .frame(height: 1)
                            .foregroundStyle(AppColor.ink15)
                            .padding(.top, 22)
                            .padding(.bottom, 18)

                        Text("What you saw")
                            .eyebrowStyle()

                        Text("Sightings sync to iNaturalist next time you have signal — they help researchers track what lives where.")
                            .font(AppFont.sans(12.5, .regular))
                            .foregroundStyle(AppColor.ink60)
                            .lineSpacing(2)
                            .padding(.top, 8)
                            .padding(.trailing, 24)

                        VStack(alignment: .leading, spacing: 0) {
                            ForEach(sightings, id: \.self) { s in
                                HStack(spacing: 10) {
                                    Circle()
                                        .frame(width: 4, height: 4)
                                        .foregroundStyle(AppColor.ink40)
                                    Text(s)
                                        .font(AppFont.sans(14, .regular))
                                        .foregroundStyle(AppColor.ink100)
                                }
                                .padding(.vertical, 6)
                            }
                        }
                        .padding(.top, 12)

                        // Share button
                        Button {
                            hasShared = true
                        } label: {
                            HStack(spacing: 10) {
                                Circle()
                                    .fill(AppColor.lime)
                                    .frame(width: 7, height: 7)
                                    .shadow(color: AppColor.lime.opacity(0.6), radius: 4)
                                Text(hasShared ? "Queued for iNaturalist" : "Share when connected")
                                    .font(AppFont.sans(14, .semibold))
                            }
                            .foregroundStyle(hasShared ? AppColor.lime : AppColor.ink100)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .padding(.horizontal, 18)
                            .background(
                                Capsule().fill(hasShared ? AppColor.lime.opacity(0.06) : Color.clear)
                            )
                            .overlay(
                                Capsule().stroke(
                                    hasShared ? AppColor.lime.opacity(0.5) : AppColor.ink25,
                                    lineWidth: 1
                                )
                            )
                        }
                        .buttonStyle(.plain)
                        .padding(.top, 16)

                        Text(hasShared ? "Will sync when you have signal" : "3 sightings queued · waiting for signal")
                            .font(AppFont.sans(10.5, .semibold))
                            .tracking(1.6)
                            .textCase(.uppercase)
                            .foregroundStyle(AppColor.ink40)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.top, 8)
                    }
                    .padding(.horizontal, 28)
                    .padding(.bottom, 32)
                }
                .padding(.bottom, 24)
            }
            .scrollIndicators(.hidden)

            // Floating close button
            Button {
                router.closeJournal()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(AppColor.ink100)
                    .frame(width: 32, height: 32)
                    .background(AppColor.glassDark88, in: Circle())
                    .overlay(Circle().stroke(AppColor.hairlineHi, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .padding(.top, 64)
            .padding(.trailing, 24)
        }
    }

    private var dateLine: String {
        let f = DateFormatter()
        f.dateFormat = "MMM d, yyyy"
        let dateStr = f.string(from: Date())
        let miles = trail.distanceMiles == floor(trail.distanceMiles)
            ? String(format: "%.0f", trail.distanceMiles)
            : String(format: "%.1f", trail.distanceMiles)
        let mins = trail.durationMinutes
        let timeStr: String = (mins >= 60)
            ? "\(mins / 60) hr \(mins % 60) min"
            : "\(mins) min"
        return "\(dateStr) · \(miles) mi · \(timeStr)"
    }

    private func stopRow(stop: TrailStop) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle()
                    .fill(AppColor.lime.opacity(0.10))
                    .overlay(Circle().stroke(AppColor.lime.opacity(0.45), lineWidth: 1))
                    .frame(width: 26, height: 26)
                Text("\(stop.number)")
                    .font(AppFont.sans(11, .heavy))
                    .foregroundStyle(AppColor.lime)
            }
            .padding(.top, 2)

            VStack(alignment: .leading, spacing: 8) {
                Text(stop.name.uppercased())
                    .font(AppFont.sans(10, .heavy))
                    .tracking(2.0)
                    .foregroundStyle(AppColor.ink60)

                Text(stop.journalFact)
                    .font(AppFont.sans(15, .medium))
                    .foregroundStyle(AppColor.ink100)
                    .tracking(-0.2)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 14)
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundStyle(AppColor.ink15),
            alignment: .top
        )
    }

    /// Sample sightings — would be populated from on-device CV later.
    private let sightings: [String] = [
        "Eastern hemlock",
        "Great rhododendron, budding",
        "Wood thrush — heard, not seen"
    ]
}

#Preview {
    JournalView()
        .environmentObject({
            let r = AppRouter()
            r.currentTrail = TrailData.kildoo
            return r
        }())
}
