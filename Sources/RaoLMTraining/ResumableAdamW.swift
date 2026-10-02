//
//  ResumableAdamW.swift
//  RaoLMTraining
//
//  WHAT: AdamW whose moments can be saved and loaded, so a commons run of hours can stop and
//        resume where it was. Frigate's AdamW keeps its state where RaoLM cannot read it.
//  PIN:  The update is Frigate's, step for step: decoupled weight decay p·(1 − lr·wd) first, then
//        m = β1·m + (1 − β1)·g, v = β2·v + (1 − β2)·g², and p − lr·m / (√v + ε) (with Frigate's
//        bias correction when asked). Moments are float32 and keyed by the parameter's flattened
//        key; only trainable parameters with a gradient get one. `decayVectors` false exempts
//        one-dimensional parameters (the norms' gains) from weight decay, as pretraining recipes
//        do; true, the default, decays everything as Frigate does.
//

import Foundation
import MLX
import MLXNN

public final class ResumableAdamW {
    public var learningRate: Float
    public let betas: (Float, Float)
    public let eps: Float
    public let weightDecay: Float
    public let biasCorrection: Bool
    public let decayVectors: Bool
    public private(set) var step = 0
    var m: [String: MLXArray] = [:]
    var v: [String: MLXArray] = [:]

    public init(
        learningRate: Float, betas: (Float, Float) = (0.9, 0.95), eps: Float = 1e-8, weightDecay: Float = 0.1, biasCorrection: Bool = false,
        decayVectors: Bool = true
    ) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
        self.biasCorrection = biasCorrection
        self.decayVectors = decayVectors
    }

    /// Applies one step of `gradients` to `model`.
    public func update(model: Module, gradients: ModuleParameters) {
        step += 1
        let (b1, b2) = betas
        let lr = learningRate
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var updated: [(String, MLXArray)] = []
        for (key, gradient) in gradients.flattened() {
            guard let parameter = parameters[key] else { continue }
            let decayed = decayVectors || parameter.ndim > 1 ? parameter * (1 - lr * weightDecay) : parameter
            let first = b1 * (m[key] ?? MLXArray.zeros(like: parameter)) + (1 - b1) * gradient
            let second = b2 * (v[key] ?? MLXArray.zeros(like: parameter)) + (1 - b2) * square(gradient)
            m[key] = first
            v[key] = second
            let delta: MLXArray
            if biasCorrection {
                let c1 = lr / (1 - pow(b1, Float(step)))
                let c2 = 1 / (1 - pow(b2, Float(step))).squareRoot()
                delta = (c1 * first) / (sqrt(second) * c2 + eps)
            } else {
                delta = lr * first / (sqrt(second) + eps)
            }
            updated.append((key, decayed - delta))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    /// The moments, for `eval`.
    public var state: [MLXArray] { Array(m.values) + Array(v.values) }

    public func save(to url: URL) throws {
        var arrays: [String: MLXArray] = [:]
        for (key, value) in m { arrays["m." + key] = value }
        for (key, value) in v { arrays["v." + key] = value }
        try MLX.save(arrays: arrays, metadata: ["format": "raolm-adamw", "step": String(step)], url: url)
    }

    public func load(from url: URL) throws {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        m = [:]
        v = [:]
        for (key, value) in arrays {
            if key.hasPrefix("m.") { m[String(key.dropFirst(2))] = value } else if key.hasPrefix("v.") { v[String(key.dropFirst(2))] = value }
        }
        step = Int(metadata["step"] ?? "0") ?? 0
    }
}
