//
//  EmbeddingService.swift
//  EmbeddingTest
//
//  Created by Simon Liang on 2025-05-02.
//

import Foundation
import Embeddings
import CoreML

actor EmbeddingService {
    var isReady: Bool = false
    var model: Model2Vec.ModelBundle

    init() async throws {
        print("Loading model...")
        model = try await Model2Vec.loadModelBundle(from: "minishlab/potion-base-2M")
        print("Model loaded.")
        isReady = true
    }

    func encode(_ text: String) throws -> MLTensor {
        return try model.encode(text, normalize: true)
    }

    func batchEncode(_ texts: [String]) throws -> MLTensor {
        return try model.batchEncode(texts, normalize: true)
    }
}
