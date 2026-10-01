//
//  UmbrellaCommand.swift
//  RaoLMCLI
//
//  WHAT: raolm umbrella build — the umbrella pack a `base` braid uses, built once: SmolLM2-135M
//        downloaded at the revision RaoLM's tokenizer came from and cut after block 20 (its upper
//        ten blocks the umbrella's frozen trunk, the whole model the commons strand), with
//        anchors and a held-out sample cut from public-domain books. A `base` braid builds it on
//        first use; this builds it ahead of time and says what it holds.
//

import ArgumentParser
import Foundation
import RaoLM
import RaoLMWorkflows

struct UmbrellaGroup: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "umbrella",
        abstract: "The umbrella pack a base braid's nodes mirror: a pretrained model's vocabulary, its trunk and the commons strand.",
        subcommands: [Build.self]
    )

    struct Build: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Build the pack from SmolLM2-135M (downloaded once, about 270 MB) under <data root>/braid/umbrella.")

        @OptionGroup var global: GlobalOptions

        @Option(help: "The first block of the umbrella's trunk; the node trains the blocks before it.")
        var cut = RaoLMConfig.base.cut

        @Flag(help: "Build it again even when it exists.")
        var force = false

        func run() async throws {
            try await guarded {
                try Preflight.requireMetallib()
                let tokenizer = try await RaoTokenizer.load()
                var config = RaoLMConfig.base
                config.cut = cut
                try config.validate()
                let layout = BraidLayout(dataRoot: global.root)
                let progress: (String) -> Void = { print($0) }
                let pack = force
                    ? try UmbrellaPacks.build(source: try PackSource.source(for: config), layout: layout, tokenizer: tokenizer, progress: progress)
                    : try UmbrellaPacks.ensure(layout: layout, config: config, tokenizer: tokenizer, progress: progress)
                guard let info = pack.info else { return }
                Console.section("Umbrella pack")
                print("pack       \(pack.sha256)")
                print("source     \(info.source)")
                print("shape      \(info.config.numHiddenLayers) blocks × \(info.config.hiddenSize); a node trains blocks 0..<\(info.config.cut) (\(info.config.nodeParameterCount.formatted()) parameters), blocks \(info.config.cut)..<\(info.config.numHiddenLayers) are the umbrella's")
                print("vocabulary \(pack.vocabulary.sha256)")
                print("commons    \(info.anchorCount) anchors of \(pack.anchors.first?.tokens.count ?? 0) tokens, \(info.heldOutCount) held-out snippets of \(pack.heldOut.first?.tokens.count ?? 0), from \(info.texts.map(\.title).joined(separator: ", "))")
                print("directory  \(layout.pack(sha256: pack.sha256).path)")
            }
        }
    }
}
