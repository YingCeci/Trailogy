// CameraView.swift
// Phase 3a — real camera viewfinder modal.
//
// Live preview from AVCaptureSession (back wide-angle). On capture, the
// session's photo output produces a UIImage which is downscaled to
// max 1280 px on the long edge (~3 MB) and handed back via `onCapture`.
// The caller (WalkingView) stores the image as state and shows the
// real thumbnail in the photo-context strip.
//
// Authorization is handled inline: if the user denies camera access we
// show a helpful overlay with a Settings button. The session is started
// in `onAppear` and stopped in `onDisappear`, so the green LED only
// stays on while the camera modal is visible.
//
// Mockup: design/mockups.html → .camera-view modal. The grid + corner
// brackets + flash + capture button match the design.

import SwiftUI
import UIKit

struct CameraView: View {
    let onCapture: (UIImage) -> Void
    let onClose: () -> Void

    @StateObject private var camera = CameraController()
    @State private var isFlashing: Bool = false
    @State private var isFiring: Bool = false

    var body: some View {
        ZStack {
            // ---- Live preview / fallback ----
            switch camera.authorization {
            case .authorized:
                if camera.isReady {
                    CameraPreviewView(session: camera.session)
                        .ignoresSafeArea()
                } else {
                    loadingView
                }
            case .notDetermined:
                loadingView
            case .denied, .restricted:
                deniedView
            }

            // ---- Rule-of-thirds grid (only when previewing) ----
            if camera.isReady {
                gridOverlay
                bracketsOverlay
            }

            // ---- Flash overlay ----
            Color.white
                .opacity(isFlashing ? 0.95 : 0)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            // ---- Top + bottom chrome ----
            VStack(spacing: 0) {
                topBar
                Spacer()
                bottomBar
            }
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear {
            camera.prepare()
            camera.startSession()
        }
        .onDisappear {
            camera.stopSession()
        }
        // When auth flips to .authorized after the OS prompt, the
        // configure-on-prepare path kicks in. Once isReady becomes true,
        // start the session if it isn't already.
        .onChange(of: camera.isReady) { _, ready in
            if ready { camera.startSession() }
        }
    }

    // MARK: - Subviews

    private var loadingView: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 12) {
                ProgressView()
                    .progressViewStyle(.circular)
                    .tint(.white)
                Text("Starting camera…")
                    .font(AppFont.sans(11, .heavy))
                    .tracking(2.0)
                    .textCase(.uppercase)
                    .foregroundStyle(.white.opacity(0.7))
            }
        }
    }

    private var deniedView: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 14) {
                Image(systemName: "camera.fill.badge.ellipsis")
                    .font(.system(size: 38, weight: .light))
                    .foregroundStyle(AppColor.lime)
                Text("Camera access needed")
                    .font(AppFont.sans(20, .semibold))
                    .foregroundStyle(.white)
                Text("Enable camera access in Settings → HikeCompanion to take a picture for the assistant.")
                    .font(AppFont.sans(13, .medium))
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
                Button {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                } label: {
                    Text("Open Settings")
                        .font(AppFont.sans(15, .semibold))
                        .foregroundStyle(AppColor.limeText)
                        .padding(.horizontal, 22)
                        .padding(.vertical, 12)
                        .background(AppColor.lime, in: Capsule())
                }
                .padding(.top, 4)
            }
        }
    }

    private var gridOverlay: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            Path { p in
                p.move(to: CGPoint(x: w/3, y: 0));   p.addLine(to: CGPoint(x: w/3, y: h))
                p.move(to: CGPoint(x: 2*w/3, y: 0)); p.addLine(to: CGPoint(x: 2*w/3, y: h))
                p.move(to: CGPoint(x: 0, y: h/3));   p.addLine(to: CGPoint(x: w, y: h/3))
                p.move(to: CGPoint(x: 0, y: 2*h/3)); p.addLine(to: CGPoint(x: w, y: 2*h/3))
            }
            .stroke(Color.white.opacity(0.18), lineWidth: 0.5)
        }
        .allowsHitTesting(false)
    }

    private var bracketsOverlay: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let bracket: CGFloat = 22
            let inset: CGFloat = 14
            Path { p in
                p.move(to: CGPoint(x: inset, y: inset + bracket))
                p.addLine(to: CGPoint(x: inset, y: inset))
                p.addLine(to: CGPoint(x: inset + bracket, y: inset))
                p.move(to: CGPoint(x: w - inset - bracket, y: inset))
                p.addLine(to: CGPoint(x: w - inset, y: inset))
                p.addLine(to: CGPoint(x: w - inset, y: inset + bracket))
                p.move(to: CGPoint(x: inset, y: h - inset - bracket))
                p.addLine(to: CGPoint(x: inset, y: h - inset))
                p.addLine(to: CGPoint(x: inset + bracket, y: h - inset))
                p.move(to: CGPoint(x: w - inset - bracket, y: h - inset))
                p.addLine(to: CGPoint(x: w - inset, y: h - inset))
                p.addLine(to: CGPoint(x: w - inset, y: h - inset - bracket))
            }
            .stroke(Color.white.opacity(0.6), lineWidth: 1.5)
        }
        .allowsHitTesting(false)
    }

    private var topBar: some View {
        HStack {
            Button {
                onClose()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(.black.opacity(0.45), in: Circle())
            }
            .buttonStyle(.plain)

            Spacer()

            Text("Camera")
                .font(AppFont.sans(11, .heavy))
                .tracking(2.4)
                .textCase(.uppercase)
                .foregroundStyle(.white.opacity(0.85))
                .shadow(color: .black.opacity(0.5), radius: 3)

            Spacer()

            Color.clear.frame(width: 36, height: 36)
        }
        .padding(.horizontal, 22)
        .padding(.top, 60)
    }

    private var bottomBar: some View {
        VStack(spacing: 18) {
            Text(bottomHint)
                .font(AppFont.sans(11, .semibold))
                .tracking(1.6)
                .textCase(.uppercase)
                .foregroundStyle(.white.opacity(0.75))
                .shadow(color: .black.opacity(0.5), radius: 3)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)

            captureButton
                .opacity(camera.isReady ? 1.0 : 0.4)
                .allowsHitTesting(camera.isReady)
        }
        .padding(.top, 18)
        .padding(.bottom, 50)
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(
                colors: [.clear, .black.opacity(0.55)],
                startPoint: .top, endPoint: .bottom
            )
        )
    }

    private var bottomHint: String {
        switch camera.authorization {
        case .authorized:
            return camera.isReady ? "Tap to capture · adds the photo as context" : "Loading camera…"
        case .denied, .restricted:
            return "Camera access denied"
        case .notDetermined:
            return "Requesting camera access…"
        }
    }

    private var captureButton: some View {
        Button {
            fire()
        } label: {
            Circle()
                .fill(.white.opacity(isFiring ? 0.85 : 1.0))
                .frame(width: 68, height: 68)
                .overlay(
                    Circle()
                        .stroke(.white, lineWidth: 3)
                        .padding(-7)
                )
                .scaleEffect(isFiring ? 0.88 : 1.0)
        }
        .buttonStyle(.plain)
        .animation(.easeInOut(duration: 0.12), value: isFiring)
    }

    // MARK: - Capture flow

    private func fire() {
        guard !isFiring, camera.isReady else { return }
        isFiring = true
        withAnimation(.easeOut(duration: 0.32)) { isFlashing = true }

        camera.capturePhoto { image in
            // Always run regardless of capture success — clear the flash.
            Task {
                try? await Task.sleep(for: .milliseconds(120))
                await MainActor.run {
                    withAnimation(.easeOut(duration: 0.16)) { isFlashing = false }
                }
                try? await Task.sleep(for: .milliseconds(80))
                await MainActor.run {
                    isFiring = false
                    if let image {
                        onCapture(image)
                    }
                    // If capture failed (image == nil), we just leave the
                    // user in the camera view so they can retry.
                }
            }
        }
    }
}

#Preview {
    CameraView(onCapture: { _ in }, onClose: {})
}
