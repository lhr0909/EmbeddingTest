//
//  ContentView.swift
//  EmbeddingTest
//
//  Created by Simon Liang on 2025-05-01.
//

import SwiftUI
import Embeddings
import Tokenizers
import MLTensorUtils

struct ContentView: View {
    @Environment(ViewModel.self) private var viewModel

    @State private var text = "vegetable"
    @State private var texts = ["groceries", "travel", "leisure"]
    @State private var similarities: [Float] = [0, 0, 0]

    var body: some View {
        VStack {
            TextField("Input text", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            // progress for each text in texts
            ForEach(texts.indices, id: \.self) { index in
                VStack {
                    Text("\(texts[index])")
                    ProgressView(value: similarities[index], total: 1)
                        .progressViewStyle(LinearProgressViewStyle())
                        .frame(width: 200)
                        .padding()
                }
            }

            Button("Test") {
                Task {
                    similarities = await testTokenizer()
                }
            }
        }
        .padding()
    }

    private func testTokenizer() async -> [Float] {
        guard let embeddingService = viewModel.embeddingService else {
            print("Embedding service not ready")
            return texts.map { _ in 0 }
        }

        do {
//            let model2vec = try await Model2Vec.loadModelBundle(from: "minishlab/potion-base-2M")
//            let model2vec = try await Model2Vec.loadBGEM3ModelBundle(from: Bundle.main.resourceURL!)
            let encoded = try await embeddingService.encode(text)
//            let result = await encoded.shapedArray(of: Float.self).scalars
//            print(result)
            let encodedBase = try await embeddingService.batchEncode(texts)
            let similarity = encoded.matmul(encodedBase.transposed()).softmax()
            let result = await similarity.shapedArray(of: Float.self).scalars
            print(result)
            return result
//            return texts.map { _ in 0 }
        } catch {
            print(error)
            // return same number of zeros as texts
            return texts.map { _ in 0 }
        }
    }

}

#Preview {
    ContentView().environment(ViewModel())
}
