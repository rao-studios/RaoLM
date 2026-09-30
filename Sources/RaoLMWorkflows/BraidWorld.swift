//
//  BraidWorld.swift
//  RaoLMWorkflows
//
//  WHAT: Which nodes a braid starts and what they are fed from, as `raolm braid`'s --nodes and
//        --dataset and the studio's braid panel (d) both give it.
//  IN:   Datasets live in the datasets root: $RAOLM_DATASETS_DIR, else the T9 work area's
//        datasets/ (/Volumes/T9/rao/projects/raolm/datasets) when that drive is mounted. With
//        neither, pass a path.
//

import Foundation
import RaoLM

public enum DatasetsRoot {
    public static let workArea = "/Volumes/T9/rao/projects/raolm"
    public static let environmentKey = "RAOLM_DATASETS_DIR"

    public static func root() throws -> URL {
        if let path = ProcessInfo.processInfo.environment[environmentKey], !path.isEmpty {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        }
        guard FileManager.default.fileExists(atPath: workArea) else {
            throw RaoLMFailure("the T9 work area \(workArea) is not mounted", hint: "plug in the T9, set \(environmentKey), or pass a path", code: 66)
        }
        return URL(fileURLWithPath: workArea, isDirectory: true).appendingPathComponent("datasets", isDirectory: true)
    }

    /// A dataset by name (inside the root) or by path (anything with a slash).
    public static func resolve(_ nameOrPath: String) throws -> URL {
        if nameOrPath.contains("/") { return URL(fileURLWithPath: (nameOrPath as NSString).expandingTildeInPath, isDirectory: true) }
        return try root().appendingPathComponent(nameOrPath, isDirectory: true)
    }
}

/// The nodes and the dataset a braid's start asks for. Empty text is "not given".
public struct BraidWorldChoice: Sendable, Equatable {
    /// Comma-separated node names; empty: the dataset's, else ambient, craft, veil.
    public var nodes: String
    /// A dataset's name in the datasets root, or a path; empty: a generated mock world.
    public var dataset: String

    public init(nodes: String = "", dataset: String = "") {
        self.nodes = nodes.trimmingCharacters(in: .whitespaces)
        self.dataset = dataset.trimmingCharacters(in: .whitespaces)
    }

    public var isEmpty: Bool { nodes.isEmpty && dataset.isEmpty }

    /// Where the dataset is, or nil when none is named. Throws when nothing is there.
    public func datasetDirectory() throws -> URL? {
        guard !dataset.isEmpty else { return nil }
        let directory = try DatasetsRoot.resolve(dataset)
        guard BraidDataset.exists(at: directory) else {
            let hint = dataset.contains("/") ? "a dataset is the directory raolm dataset generate writes (manifest.json, nodes/)"
                : "raolm dataset generate --name \(dataset)"
            throw RaoLMFailure("no braid dataset at \(directory.path)", hint: hint, code: 66)
        }
        return directory
    }

    /// Sets the options' nodes and dataset. With neither given, a braid that already has nodes
    /// keeps them (`BraidOptions.adoptNodes`).
    public func apply(to options: inout BraidOptions) throws {
        var names = BraidNodeSpec.defaults.map(\.name).joined(separator: ",")
        options.dataset = try datasetDirectory()
        if let directory = options.dataset {
            names = try JSONCoding.read(DatasetManifest.self, from: directory.appendingPathComponent(DatasetManifest.fileName))
                .names.joined(separator: ",")
        }
        options.nodes = try BraidNodeSpec.parse(nodes.isEmpty ? names : nodes)
        options.adoptNodes = isEmpty
    }
}
