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
                    // Header
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Rectangle()
                                .frame(width: 16, height: 1)
                                .foregroundStyle(AppColor.lime.opacity(0.55))
                            Text("Nature, the best teacher.")
                                .font(AppFont.sans(10, .heavy))
                                .tracking(2.4)
                                .textCase(.uppercase)
                                .foregroundStyle(AppColor.lime)
                        }
                        .padding(.bottom, 18)

                        Text("Trails near you")
                            .font(AppFont.sans(28, .bold))
                            .foregroundStyle(AppColor.ink100)
                            .tracking(-0.5)

                        Text("Pittsburgh")
                            .font(AppFont.sans(13, .medium))
                            .foregroundStyle(AppColor.ink60)
                            .padding(.top, 4)
                    }
                    .padding(.horizontal, 22)
                    .padding(.top, 6)
                    .padding(.bottom, 22)

                    // Cards
                    VStack(spacing: 18) {
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
                .padding(.top, 70)  // status bar / dynamic island clearance
            }
            .scrollIndicators(.hidden)
        }
    }
}

// MARK: - TrailCard

private struct TrailCard: View {
    let trail: Trail
    let onTap: () -> Void

    var status: TrailStatus { TrailData.status(for: trail) }

    var body: some View {
        Button(action: onTap) {
            ZStack(alignment: .bottomLeading) {
                // Background photo
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
                .frame(maxWidth: .infinity)
                .aspectRatio(16.0/13.0, contentMode: .fill)
                .clipped()

                // Scrim
                LinearGradient(
                    colors: [
                        Color.black.opacity(0.10),
                        Color.black.opacity(0.55),
                        Color.black.opacity(0.88)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )

                // Top-right badge
                VStack {
                    HStack {
                        Spacer()
                        statusBadge
                    }
                    Spacer()
                }
                .padding(14)

                // Bottom text block
                VStack(alignment: .leading, spacing: 4) {
                    Text(trail.region.uppercased())
                        .font(AppFont.sans(10, .semibold))
                        .tracking(1.6)
                        .foregroundStyle(AppColor.ink100.opacity(0.8))

                    Text(trail.name)
                        .font(AppFont.sans(24, .bold))
                        .foregroundStyle(AppColor.ink100)
                        .tracking(-0.5)

                    HStack(spacing: 10) {
                        Text("\(formattedMiles) mi")
                        circleDot
                        Text(trail.difficulty)
                        circleDot
                        Text("\(trail.durationMinutes) min")
                    }
                    .font(AppFont.sans(13, .medium))
                    .foregroundStyle(AppColor.ink100.opacity(0.88))
                    .padding(.top, 4)
                }
                .padding(18)
            }
            .frame(maxWidth: .infinity)
            .aspectRatio(16.0/13.0, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        }
        .buttonStyle(CardPressStyle())
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

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .ready:
            Label {
                Text("Ready")
                    .font(AppFont.sans(10.5, .semibold))
                    .tracking(0.6)
                    .textCase(.uppercase)
            } icon: {
                Circle()
                    .fill(AppColor.lime)
                    .frame(width: 6, height: 6)
                    .shadow(color: AppColor.lime.opacity(0.6), radius: 4)
            }
            .foregroundStyle(AppColor.ink100)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.5), in: Capsule())

        case .walked(let date):
            Text(date)
                .font(AppFont.sans(10.5, .semibold))
                .tracking(0.6)
                .textCase(.uppercase)
                .foregroundStyle(AppColor.ink100.opacity(0.75))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.black.opacity(0.4), in: Capsule())

        case .downloadable:
            // Models are bundled — kept here as a placeholder if we ever
            // add real per-trail downloads.
            Label {
                Text("Download · \(byteLabel)")
                    .font(AppFont.sans(10.5, .heavy))
                    .tracking(0.6)
                    .textCase(.uppercase)
            } icon: {
                Image(systemName: "arrow.down.to.line")
                    .font(.system(size: 9, weight: .bold))
            }
            .foregroundStyle(AppColor.lime)
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(.black.opacity(0.55), in: Capsule())
            .overlay(
                Capsule().stroke(AppColor.lime.opacity(0.55), lineWidth: 1)
            )
        }
    }

    private var byteLabel: String {
        let mb = trail.bytes / (1024 * 1024)
        return "\(mb) MB"
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
