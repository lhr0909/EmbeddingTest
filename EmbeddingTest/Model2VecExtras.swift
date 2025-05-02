//
//  Model2VecExtras.swift
//  EmbeddingTest
//
//  Created by Simon Liang on 2025-05-02.
//

import Foundation
import Embeddings
import Hub

extension Model2Vec {
    public static func loadBGEM3ModelBundle(
        from modelFolder: URL,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Model2Vec.ModelBundle {
        let addedTokens = try await loadAddedTokens(from: modelFolder)
        let tokenizerModelUrl = try findSentencePieceModel(in: modelFolder)
        let tokenizer = try XLMRobetaTokenizer(
            tokenizerModelUrl: tokenizerModelUrl,
            addedTokens: addedTokens
        )
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let configUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.configFileName)
        let config = try Model2Vec.loadConfig(at: configUrl)
        let model = try Model2Vec.loadModel(
            weightsUrl: weightsUrl,
            normalize: config.normalize ?? false,
            loadConfig: loadConfig
        )
        return Model2Vec.ModelBundle(
            model: model,
            tokenizer: tokenizer
        )
    }

    private static func loadAddedTokens(from modelFolder: URL) async throws -> [String: Int] {
        let hubConfiguration = LanguageModelConfigurationFromHub(modelFolder: modelFolder)
        let addedTokens = try await hubConfiguration.tokenizerData.addedTokens?.arrayValue?.map {
            $0.dictionary as [String: Any]
        }
        guard let addedTokens else {
            return [:]
        }
        var result = [String: Int]()
        for addedToken in addedTokens {
            if let content = addedToken["content"] as? String, let id = addedToken["id"] as? Int {
                result[content] = id
            }
        }
        return result
    }

    private static func findSentencePieceModel(in folder: URL) throws -> URL {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil)
        for url in contents {
            if url.pathExtension == "model", url.lastPathComponent.contains("sentencepiece") {
                return url
            }
        }
        throw EmbeddingsError.fileNotFound
    }
}
