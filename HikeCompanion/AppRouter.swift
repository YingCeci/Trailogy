// AppRouter.swift
// State container for the picker → detail → walking → journal flow.
// Mirrors the view-switching state machine in design/mockups.html.
//
// `screen` drives which top-level view is rendered. `currentTrail` is the
// trail the user picked; it's set when navigating from picker to detail.
// `debugVisible` is a sheet flag, separate from the main flow.

import SwiftUI

enum AppScreen: Equatable {
    case picker
    case detail
    case walking
    case journal
}

@MainActor
final class AppRouter: ObservableObject {
    @Published var screen: AppScreen = .picker
    @Published var currentTrail: Trail = TrailData.kildoo
    @Published var debugVisible: Bool = false

    func go(_ s: AppScreen) {
        withAnimation(.easeInOut(duration: 0.4)) {
            screen = s
        }
    }

    func choose(_ t: Trail) {
        currentTrail = t
        go(.detail)
    }

    func begin() { go(.walking) }
    func backToPicker() { go(.picker) }
    func endTour() { go(.journal) }
    func closeJournal() { go(.picker) }

    func openDebug()  { debugVisible = true  }
    func closeDebug() { debugVisible = false }
}
