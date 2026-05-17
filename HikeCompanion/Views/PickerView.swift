// PickerView.swift
// "Trails near you" — the entry screen.
// Mockup: design/mockups.html → .picker view.

import SwiftUI

struct PickerView: View {
    @EnvironmentObject var router: AppRouter

    var body: some View {
        ZStack(alignment: .top) {
            AppColor.screenBg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // Header — mirrors design/mockups.html `.picker-head`.
                    // Replaces the earlier "Nature, the best teacher /
                    // Trails near you / Pittsburgh" stack with a single
                    // softer question paired with a lime location pin —
                    // the visual answer to "where". The discipline of
                    // absences (no profile chip, no eyebrow, no city
                    // label) is part of the design.
                    HStack(spacing: 10) {
                        Image(systemName: "location.fill")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(AppColor.lime)
                        Text("Where should Trailogy take you?")
                            .font(AppFont.sans(18, .regular))
                            .foregroundStyle(AppColor.ink100)
                            .tracking(-0.2)
                    }
                    .padding(.horizontal, 22)
                    .padding(.top, 6)
                    .padding(.bottom, 26)

                    // Cards
                    VStack(spacing: 14) {
                        ForEach(TrailData.all) { trail in
                            TrailCard(trail: trail) {
                                router.choose(trail)
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 24)

                    // Footer — visible Journal + Debug entries side-by-side.
                    // Subtle but discoverable, matching the picker's
                    // ink60 secondary-text style.
                    HStack(spacing: 32) {
                        Spacer()
                        Button {
                            router.go(.journal)
                        } label: {
                            Text("Journal")
                                .font(AppFont.sans(13, .medium))
                                .foregroundStyle(AppColor.ink60)
                        }
                        .buttonStyle(.plain)

                        Rectangle()
                            .frame(width: 1, height: 11)
                            .foregroundStyle(AppColor.ink25)

                        Button {
                            router.openDebug()
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "ladybug.fill")
                                    .font(.system(size: 11, weight: .semibold))
                                Text("Debug")
                                    .font(AppFont.sans(13, .medium))
                            }
                            .foregroundStyle(AppColor.ink60)
                        }
                        .buttonStyle(.plain)
                        Spacer()
                    }
                    .padding(.vertical, 14)

                    Color.clear.frame(height: 32)
                }
                // iOS adds the dynamic-island/status-bar safe-area inset
                // automatically; this is just visual breathing room past
                // it. Was 70 — but that doubled up with the safe area
                // and dropped the first card visibly low on the screen
                // (≈ 129 pt of dead space above the header on iPhone 15
                // Pro). 12 brings it in line with the mockup's intended
                // top spacing (`padding: 70px 0 24px` on `.picker`,
                // which absorbs the browser's status-bar zone).
                .padding(.top, 12)
            }
            .scrollIndicators(.hidden)
        }
    }
}

// MARK: - TrailCard

private struct TrailCard: View {
    @EnvironmentObject var router: AppRouter
    let trail: Trail
    let onTap: () -> Void

    /// Status is now derived from router runtime state (`walkedAt`)
    /// rather than from a hardcoded table — see design/README.md
    /// commit 8bf8889. Mirrors the mockup's `renderPickerBadges()`,
    /// which injects the "Completed [date]" badge on every walked
    /// trail at picker render time, with no demo-baked completions.
    var status: TrailStatus { TrailData.status(for: trail, router: router) }

    var body: some View {
        Button(action: onTap) {
            // Photo fills a strict 16:9 card; region / name / tagline /
            // stats overlay the bottom-left of the image, separated from
            // the photo by a soft top-to-bottom scrim. Mirrors the
            // upstream `.t-card` markup in design/mockups.html — three
            // stacked z-layers (.photo / .scrim / .body).
            //
            // Why this shape (back from the earlier photo-on-top +
            // text-panel-below):
            //   - Cards were too tall — the dedicated text panel added
            //     ~150 px of height on top of the 16:9 image, making
            //     three cards push out of the picker viewport.
            //   - 16:9 fits modern landscape hike photos cleanly enough
            //     that worst-case bottom-crop is rarely a problem; the
            //     scrim does the legibility work the text panel used to.
            //   - Matches the mockup, which is the authoritative spec.
            ZStack(alignment: .bottomLeading) {
                photoLayer
                scrimLayer
                textOverlay
                    .padding(.horizontal, 18)
                    .padding(.bottom, 16)
                if case .walked(let date) = status {
                    walkedBadge(date: date)
                        .padding(14)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }
            }
            // 2:1 ("cinematic wide") instead of the mockup's 16:9. The
            // mockup is rendered in a desktop browser where 16:9 cards
            // feel proportionate; on iPhone widths (≈ 361 pt usable),
            // 16:9 produces ~203-pt cards × 3 — pushes the trio to the
            // edge of the viewport with no breathing room. 2:1 lands
            // each card at ~180 pt, three fit comfortably without
            // cramping the text overlay block (region/name/tagline/
            // stats needs ≈ 100 pt at the bottom).
            .aspectRatio(2.0/1.0, contentMode: .fit)
            .frame(maxWidth: .infinity)
            .background(AppColor.ink25)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                // Whisper-thin cream outline so the card edge reads
                // cleanly against the dark picker background.
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(AppColor.ink100.opacity(0.07), lineWidth: 1)
            )
        }
        .buttonStyle(CardPressStyle())
    }

    /// Bottom layer — the trail's cover photo, scaled to fill the
    /// entire card. Crops are taken from the centre.
    private var photoLayer: some View {
        AsyncImage(url: trail.coverImageURL) { phase in
            switch phase {
            case .empty:
                AppColor.ink25
            case .success(let img):
                img.resizable().scaledToFill()
            case .failure:
                AppColor.ink25
            @unknown default:
                AppColor.ink25
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipped()
    }

    /// Middle layer — top-to-bottom darkening so the text below stays
    /// legible without text shadows on the type. Transparent at the
    /// top so the establishing image isn't buried. Stops match the
    /// mockup's `.t-card .scrim` linear-gradient.
    private var scrimLayer: some View {
        LinearGradient(
            stops: [
                .init(color: .black.opacity(0.0),  location: 0.18),
                .init(color: .black.opacity(0.32), location: 0.44),
                .init(color: .black.opacity(0.70), location: 0.72),
                .init(color: .black.opacity(0.92), location: 1.0),
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .allowsHitTesting(false)
    }

    /// Top layer — the text block, anchored to bottom-left. Region
    /// eyebrow / trail name / one-line tagline / stats row.
    private var textOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(trail.region.uppercased())
                .font(AppFont.sans(10, .semibold))
                .tracking(1.6)
                .foregroundStyle(AppColor.ink100.opacity(0.85))
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Text(trail.name)
                .font(AppFont.sans(22, .bold))
                .foregroundStyle(AppColor.ink100)
                .tracking(-0.4)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .padding(.top, 1)

            Text(trail.summary)
                .font(AppFont.sans(12.5, .regular))
                .foregroundStyle(AppColor.ink100.opacity(0.85))
                .lineSpacing(1)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)

            HStack(spacing: 8) {
                Text("\(formattedMiles) mi")
                circleDot
                Text(trail.difficulty)
                circleDot
                Text(durationLabel)
            }
            .font(AppFont.sans(12.5, .medium))
            .foregroundStyle(AppColor.ink100.opacity(0.92))
            .padding(.top, 6)
            .lineLimit(1)
            .minimumScaleFactor(0.85)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// "60 min" or "1 hr" or "1 hr 15 min" — friendlier than always
    /// rendering minutes when the trail is over an hour.
    private var durationLabel: String {
        let m = trail.durationMinutes
        if m >= 60 {
            let h = m / 60
            let r = m % 60
            return r == 0 ? "\(h) hr" : "\(h) hr \(r) min"
        }
        return "\(m) min"
    }

    private var formattedMiles: String {
        let v = trail.distanceMiles
        return v == floor(v) ? String(format: "%.0f", v) : String(format: "%.1f", v)
    }

    private var circleDot: some View {
        Circle()
            .frame(width: 3, height: 3)
            .foregroundStyle(AppColor.ink100.opacity(0.55))
    }

    /// Top-right "Completed [date]" pill, shown only on a trail walked
    /// in this session. Picker cards are otherwise pure choice — no
    /// "Ready" / "Download · MB" pill (those retired in design/README.md
    /// items 17 & 23: download is the detail-view CTA's job; "ready"
    /// added no information since every trail is ready by default).
    private func walkedBadge(date: String) -> some View {
        Text("Completed \(date)")
            .font(AppFont.sans(10.5, .semibold))
            .tracking(0.6)
            .textCase(.uppercase)
            .foregroundStyle(AppColor.ink100.opacity(0.85))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.5), in: Capsule())
    }
}

private struct CardPressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.985 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview {
    PickerView()
        .environmentObject(AppRouter())
}
