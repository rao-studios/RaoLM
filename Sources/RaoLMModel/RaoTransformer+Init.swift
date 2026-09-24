//
//  RaoTransformer+Init.swift
//  RaoLMModel
//
//  WHAT: From-scratch initialisation: normal(0, initializer_range) for every Linear and
//        Embedding weight, ones for RMSNorm, zeros for any bias (nanotron's std init).
//  PIN:  One PRNG key per leaf module, split from the seed in sorted-path order, so the
//        same seed gives the same weights whatever else touched the global generator.
//

import Foundation
import MLX
import MLXNN
import RaoLMCore

extension RaoTransformer {

    public static func make(config: RaoLMConfig, seed: UInt64, tapLayer: Int? = nil) throws -> RaoTransformer {
        try config.validate()
        let model = RaoTransformer(config, tapLayer: tapLayer)
        model.reinitialise(std: config.initializerRange, seed: seed)
        return model
    }

    public func reinitialise(std: Float, seed: UInt64) {
        let leaves = leafModules().flattened().sorted { $0.0 < $1.0 }
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: max(leaves.count, 2))
        var updates: [(String, MLXArray)] = []
        for (i, (path, module)) in leaves.enumerated() {
            if let norm = module as? RMSNorm {
                updates.append(("\(path).weight", MLXArray.ones(norm.weight.shape, type: Float.self)))
            } else if let linear = module as? Linear {
                updates.append((
                    "\(path).weight",
                    MLXRandom.normal(linear.weight.shape, type: Float.self, loc: 0, scale: std, key: keys[i])))
                if let bias = linear.bias {
                    updates.append(("\(path).bias", MLXArray.zeros(bias.shape, type: Float.self)))
                }
            } else if let embedding = module as? Embedding {
                updates.append((
                    "\(path).weight",
                    MLXRandom.normal(embedding.weight.shape, type: Float.self, loc: 0, scale: std, key: keys[i])))
            }
        }
        update(parameters: ModuleParameters.unflattened(updates))
        eval(self)
    }

    /// Parameters as HF key → array, exactly what a checkpoint stores.
    public func flatParameters() -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: parameters().flattened())
    }
}
