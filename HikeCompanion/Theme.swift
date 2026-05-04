// Theme.swift
// Color + font tokens lifted from design/mockups.html (Nature companion).
// Mockup uses Inter — we substitute SF Pro (system) since Inter isn't
// bundled. Bundle Inter.ttf later if Billy wants exact parity.

import SwiftUI

enum AppColor {
    // page (outside the phone in the mockup) — not used inside the app, but kept
    static let pageBg     = Color(red: 0.937, green: 0.933, blue: 0.910)  // #efeee8

    // canvas / screen background
    static let screenBg   = Color(red: 0.043, green: 0.047, blue: 0.039)  // #0b0c0a
    static let walkingBg  = Color(red: 0.020, green: 0.027, blue: 0.024)  // #050706
    static let mapBg      = Color(red: 0.039, green: 0.047, blue: 0.039)  // #0a0c0a

    // ink ramp (cream → almost-black)
    static let ink100     = Color(red: 0.961, green: 0.953, blue: 0.925)  // #f5f3ec
    static let ink80      = Color(red: 0.784, green: 0.773, blue: 0.737)  // #c8c5bc
    static let ink60      = Color(red: 0.541, green: 0.533, blue: 0.506)  // #8a8881
    static let ink40      = Color(red: 0.353, green: 0.345, blue: 0.322)  // #5a5852
    static let ink25      = Color(red: 0.180, green: 0.176, blue: 0.161)  // #2e2d29
    static let ink15      = Color(red: 0.114, green: 0.110, blue: 0.098)  // #1d1c19

    // primary accent — acid lime
    static let lime       = Color(red: 0.851, green: 0.961, blue: 0.443)  // #d9f571
    static let limeDark   = Color(red: 0.757, green: 0.867, blue: 0.345)  // #c1dd58
    static let limeText   = Color(red: 0.102, green: 0.122, blue: 0.039)  // #1a1f0a (dark green for on-lime)

    // translucent variants — used for halos, photo scrims, glassmorph cards
    static let limeGlow22 = Color(red: 0.851, green: 0.961, blue: 0.443).opacity(0.22)
    static let limeGlow18 = Color(red: 0.851, green: 0.961, blue: 0.443).opacity(0.18)
    static let limeGlow10 = Color(red: 0.851, green: 0.961, blue: 0.443).opacity(0.10)
    static let limeGlow06 = Color(red: 0.851, green: 0.961, blue: 0.443).opacity(0.06)

    // cards / glass surfaces (mockup uses rgba(15,16,13,0.86–0.96))
    static let glassDark88 = Color(red: 0.059, green: 0.063, blue: 0.051).opacity(0.88)
    static let glassDark93 = Color(red: 0.059, green: 0.063, blue: 0.051).opacity(0.93)

    // hairline strokes
    static let hairline    = Color.white.opacity(0.06)
    static let hairlineHi  = Color.white.opacity(0.10)
}

// MARK: - Typography

enum AppFont {
    /// Use SF Pro (system). The mockup uses Inter; visually close enough.
    /// If we bundle Inter later, swap `.system` for `.custom("Inter", size:)`.
    static func sans(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    // Common roles seen in the mockup
    static let eyebrowSmall = sans(10.5, .heavy)        // 0.20em letter-spaced eyebrows
    static let bodyM        = sans(14, .medium)
    static let titleS       = sans(17, .bold)
    static let titleM       = sans(20, .bold)
    static let titleL       = sans(28, .bold)
    static let titleXL      = sans(30, .bold)
    static let lyricNow     = sans(24, .semibold)
    static let lyricSide    = sans(17, .medium)
    static let askText      = sans(26, .semibold)
}

// MARK: - View modifiers

extension View {
    /// Eyebrow label: tracked uppercase, ink60, weight 700.
    func eyebrowStyle(_ color: Color = AppColor.ink60) -> some View {
        self
            .font(AppFont.sans(10.5, .heavy))
            .tracking(2.0)               // ~0.20em on 10.5pt ≈ 2.1pt
            .textCase(.uppercase)
            .foregroundStyle(color)
    }

    /// Hairline underline (1px ink15) used between sections in journal/picker.
    func sectionRule(_ color: Color = AppColor.ink15) -> some View {
        overlay(alignment: .top) {
            Rectangle()
                .frame(height: 1)
                .foregroundStyle(color)
        }
    }
}
