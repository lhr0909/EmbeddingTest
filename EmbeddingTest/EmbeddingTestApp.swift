//
//  EmbeddingTestApp.swift
//  EmbeddingTest
//
//  Created by Simon Liang on 2025-05-01.
//

import SwiftUI

@main
struct EmbeddingTestApp: App {
    @State private var viewModel = ViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(viewModel)
        }
    }
}
