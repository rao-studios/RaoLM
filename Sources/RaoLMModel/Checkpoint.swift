//
//  Checkpoint.swift
//  RaoLMModel
//
//  WHAT: Writes and reads a checkpoint directory: model.safetensors (HF Llama key names),
//        config.json (HF Llama keys) and the tokenizer files — a self-contained model
//        folder that Frigate's LLMModelFactory, Python transformers and mlx-lm all load.
//  PIN:  Nothing else with a .safetensors extension may live under a checkpoint directory:
//        Frigate's loadWeights globs recursively and verifies every key.
//

import Foundation
import MLX
import MLXNN
import RaoLMCore

public enum CheckpointError: Error, CustomStringConvertible {
    case missing(String)

    public var description: String {
        switch self {
        case .missing(let what): return "checkpoint incomplete: \(what)"
        }
    }
}

public enum Checkpoint {
    public static let weightsFile = "model.safetensors"
    public static let configFile = "config.json"

    /// Saves and returns the SHA-256 of model.safetensors.
    @discardableResult
    public static func save(
        model: RaoTransformer, to directory: URL, metadata: [String: String] = [:], tokenizerDirectory: URL? = nil
    ) throws -> String {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        var meta = metadata
        meta["format"] = "mlx"
        let weightsURL = directory.appendingPathComponent(weightsFile)
        try MLX.save(arrays: model.flatParameters(), metadata: meta, url: weightsURL)
        try JSONCoding.write(model.config, to: directory.appendingPathComponent(configFile))
        if let tokenizerDirectory {
            for name in RaoTokenizer.checkpointFiles {
                let source = tokenizerDirectory.appendingPathComponent(name)
                let target = directory.appendingPathComponent(name)
                guard manager.fileExists(atPath: source.path) else { continue }
                if manager.fileExists(atPath: target.path) { try manager.removeItem(at: target) }
                try manager.copyItem(at: source, to: target)
            }
        }
        return try ContentHash.sha256Hex(fileAt: weightsURL)
    }

    public static func load(from directory: URL, tapLayer: Int? = nil) throws -> RaoTransformer {
        let configURL = directory.appendingPathComponent(configFile)
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw CheckpointError.missing(configURL.path)
        }
        let config = try JSONCoding.read(RaoLMConfig.self, from: configURL)
        try config.validate()
        let model = RaoTransformer(config, tapLayer: tapLayer)
        try loadWeights(into: model, from: directory)
        return model
    }

    /// Loads every *.safetensors directly inside `directory` (RaoLM or HF SmolLM2 layout).
    public static func loadWeights(into model: RaoTransformer, from directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw CheckpointError.missing("no .safetensors in \(directory.path)") }
        var weights: [String: MLXArray] = [:]
        for file in files {
            for (key, value) in try loadArrays(url: file) { weights[key] = value }
        }
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    public static func weightsSHA256(_ directory: URL) throws -> String {
        try ContentHash.sha256Hex(fileAt: directory.appendingPathComponent(weightsFile))
    }
}
