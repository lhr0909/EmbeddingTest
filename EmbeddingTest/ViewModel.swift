//
//  ViewModel.swift
//  EmbeddingTest
//
//  Created by Simon Liang on 2025-05-02.
//

import Foundation

@MainActor @Observable class ViewModel: Sendable {
    var embeddingService: EmbeddingService? = nil

    init() {
        Task {
            embeddingService = try await EmbeddingService()
        }
    }
}
