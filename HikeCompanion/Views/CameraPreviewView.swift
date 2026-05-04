// CameraPreviewView.swift
// SwiftUI bridge for `AVCaptureVideoPreviewLayer`.
//
// AVCaptureVideoPreviewLayer is a CALayer that needs to be the
// `layer` of a UIView — it can't be used as a sibling layer or
// composited into SwiftUI directly. Standard pattern: a UIView
// subclass that overrides `layerClass` to `AVCaptureVideoPreviewLayer`,
// wrapped via UIViewRepresentable.

import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let v = PreviewUIView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        // Lock to portrait — the app is portrait-only (project.yml).
        if let conn = v.previewLayer.connection,
           conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
        }
        return v
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        // No SwiftUI-driven updates — the session controls everything.
    }

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }
        var previewLayer: AVCaptureVideoPreviewLayer {
            // Force-cast is safe because of `layerClass` above.
            // swiftlint:disable:next force_cast
            return layer as! AVCaptureVideoPreviewLayer
        }
    }
}
