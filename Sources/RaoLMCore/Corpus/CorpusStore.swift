//
//  CorpusStore.swift
//  RaoLMCore
//
//  WHAT: Writes and reads a generated corpus directory.
//  OUT:  manifest.json, facts.jsonl, documents/<id>.json and documents/<id>.txt.
//

import Foundation

public enum CorpusStore {

    public static func write(_ corpus: GeneratedCorpus, to directory: URL) throws {
        let manager = FileManager.default
        let documents = directory.appendingPathComponent("documents", isDirectory: true)
        try manager.createDirectory(at: documents, withIntermediateDirectories: true)
        try JSONCoding.write(corpus.manifest, to: directory.appendingPathComponent("manifest.json"))
        for document in corpus.documents {
            try JSONCoding.write(document, to: documents.appendingPathComponent("\(document.id).json"))
            try Data((document.name + "\n\n" + document.text + "\n").utf8)
                .write(to: documents.appendingPathComponent("\(document.id).txt"), options: .atomic)
        }
        let facts = try JSONLWriter(url: directory.appendingPathComponent("facts.jsonl"), truncate: true)
        for fact in corpus.facts { try facts.append(fact) }
        facts.close()
    }

    public static func load(_ directory: URL) throws -> GeneratedCorpus {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw CorpusStoreError.notACorpus(directory.path)
        }
        let manifest = try JSONCoding.read(CorpusManifest.self, from: manifestURL)
        let documentsDirectory = directory.appendingPathComponent("documents", isDirectory: true)
        let documents = try manifest.documentIDs.map { id in
            try JSONCoding.read(CorpusDocument.self, from: documentsDirectory.appendingPathComponent("\(id).json"))
        }
        return GeneratedCorpus(manifest: manifest, documents: documents)
    }

    public static func exists(at directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path)
    }
}

public enum CorpusStoreError: Error, CustomStringConvertible {
    case notACorpus(String)

    public var description: String {
        switch self {
        case .notACorpus(let path): return "no corpus manifest.json in \(path)"
        }
    }
}
