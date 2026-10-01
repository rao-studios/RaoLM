//
//  JacobianLens.swift
//  RaoLMModel
//
//  WHAT: A training-free reading of what a node plans, after Anthropic's 2026 workspace lens. J,
//        the frozen trunk's Jacobian averaged over commons prompts and over position pairs t ≤ t′,
//        maps any node's cut state toward the final state it pushes later positions to; the
//        umbrella's final norm and tied head read that as words. J belongs to the pack: it is taken
//        once, through the trunk alone, so it reads every node the same way.
//  PIN:  Rows by reverse mode. A cotangent on channel j at every output position gives, at each
//        input position t, Σ_{t′ ≥ t} ∂h_final,t′[j]/∂h_cut,t; summed over t and divided by the
//        prompt's L(L+1)/2 pairs, that is row j of the prompt's J. J is taken at the base model's
//        cut states. The readout normalises J·h, so J's scale is immaterial: a trunk that passes
//        its input through gives J ∝ I and reads exactly as the identity lens (the logit lens at the cut).
//

import Foundation
import MLX
import RaoLMCore

public struct JacobianLensInfo: Codable, Sendable, Equatable {
    /// The pack whose trunk J was taken through.
    public var packSHA256: String
    public var prompts: Int
    public var promptTokens: Int
    /// SHA-256 of J's float32 bytes.
    public var sha256: String
    /// Cosine of J taken on the first half of the prompts with J taken on the second half.
    public var splitHalfAgreement: Float
}

public enum JacobianLensError: Error, CustomStringConvertible {
    case missing(String)
    case fingerprint(expected: String, found: String)

    public var description: String {
        switch self {
        case .missing(let path): return "the lens at \(path) holds no jacobian"
        case .fingerprint(let expected, let found): return "the lens's bytes hash to \(found.prefix(12)), not \(expected.prefix(12))"
        }
    }
}

public struct JacobianLens {
    public static let arraysFile = "lens.safetensors"
    public static let infoFile = "lens.json"

    /// [hidden, hidden]: h_final ≈ J · h_cut, up to scale.
    public let jacobian: MLXArray
    public let info: JacobianLensInfo

    public init(jacobian: MLXArray, info: JacobianLensInfo) {
        self.jacobian = jacobian
        self.info = info
    }

    /// The sum over `prompts` of each prompt's pair-averaged Jacobian of `model`'s trunk, taken at
    /// `model`'s own cut states; `batch` channels per reverse pass.
    public static func sum(model: RaoTransformer, prompts: [[Int]], batch: Int = 64, progress: ((Int) -> Void)? = nil) -> MLXArray {
        let D = model.config.hiddenSize
        let identity = MLXArray.identity(D, type: Float.self)
        var total = MLXArray.zeros([D, D], type: Float.self)
        for (p, prompt) in prompts.enumerated() where !prompt.isEmpty {
            let L = prompt.count
            guard let cut = model.body(MLXArray(prompt.map { Int32($0) }, [1, L]), captureTap: false, captureCut: true).cut else { continue }
            let pairs = Float(L * (L + 1) / 2)
            var rows: [MLXArray] = []
            var start = 0
            while start < D {
                let n = min(max(batch, 1), D - start)
                let primal = broadcast(cut, to: [n, L, D])
                let cotangent = broadcast(identity[start ..< (start + n)].expandedDimensions(axis: 1), to: [n, L, D])
                let (_, gradients) = vjp({ [model.trunk($0[0])] }, primals: [primal], cotangents: [cotangent])
                rows.append(gradients[0].sum(axis: 1) / pairs)
                start += n
            }
            total = total + concatenated(rows, axis: 0)
            eval(total)
            progress?(p + 1)
        }
        return total
    }

    /// J over `prompts` (the pack's anchors), with how closely its two halves agree.
    public static func compute(
        model: RaoTransformer, packSHA256: String, prompts: [[Int]], batch: Int = 64, progress: ((Int) -> Void)? = nil
    ) -> JacobianLens {
        let half = prompts.count / 2
        let first = sum(model: model, prompts: Array(prompts[..<half]), batch: batch) { progress?($0) }
        let second = sum(model: model, prompts: Array(prompts[half...]), batch: batch) { progress?(half + $0) }
        let jacobian = (first + second) / Float(max(prompts.count, 1))
        eval(jacobian)
        var agreement: Float = 1
        if half > 0 {
            let (a, b) = (first.flattened(), second.flattened())
            agreement = ((a * b).sum() / MLX.maximum(MLX.sqrt((a * a).sum() * (b * b).sum()), MLXArray(Float(1e-12)))).item(Float.self)
        }
        return JacobianLens(jacobian: jacobian, info: JacobianLensInfo(
            packSHA256: packSHA256, prompts: prompts.count, promptTokens: prompts.first?.count ?? 0, sha256: fingerprint(jacobian),
            splitHalfAgreement: agreement))
    }

    static func fingerprint(_ jacobian: MLXArray) -> String {
        ContentHash.sha256Hex(parts: [jacobian.asType(.float32).asData(access: .copy).data])
    }

    /// The lens's logits over cut states [..., hidden]: the umbrella's final norm and tied head over J·h.
    public func logits(_ cut: MLXArray, head model: RaoTransformer) -> MLXArray {
        model.head(matmul(cut, jacobian.T)).logits
    }

    public func save(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(
            arrays: ["jacobian": jacobian], metadata: ["format": "raolm-jacobian-lens", "pack": info.packSHA256],
            url: directory.appendingPathComponent(Self.arraysFile))
        try JSONCoding.write(info, to: directory.appendingPathComponent(Self.infoFile))
    }

    /// The lens saved beside a pack, if it was taken through that pack's trunk; refused if its bytes changed.
    public static func load(from directory: URL, packSHA256: String) throws -> JacobianLens? {
        let infoURL = directory.appendingPathComponent(infoFile)
        guard FileManager.default.fileExists(atPath: infoURL.path) else { return nil }
        let info = try JSONCoding.read(JacobianLensInfo.self, from: infoURL)
        guard info.packSHA256 == packSHA256 else { return nil }
        let url = directory.appendingPathComponent(arraysFile)
        guard let jacobian = try loadArrays(url: url)["jacobian"] else { throw JacobianLensError.missing(url.path) }
        let found = fingerprint(jacobian)
        guard found == info.sha256 else { throw JacobianLensError.fingerprint(expected: info.sha256, found: found) }
        return JacobianLens(jacobian: jacobian, info: info)
    }
}
