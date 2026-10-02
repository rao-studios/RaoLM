//
//  CommonsSources.swift
//  RaoLMBraid
//
//  WHAT: Where commons text comes from: public-domain books from Project Gutenberg (by id, or the
//        most downloaded), rows of a dataset on the Hugging Face hub read through its
//        datasets-server (FineWeb-Edu is parquet-only, and nothing in RaoLM reads parquet), and
//        files the owner has. Each returns documents and how to fetch them again.
//  PIN:  Gutenberg books are cached under <braid>/umbrella/texts/ and stripped of the
//        distributor's header and footer; a book that does not read as English is skipped; the
//        pack's own books are excluded so its held-out snippets stay unseen. Dataset rows are
//        read in pages of 100 at offsets drawn from a seed across the whole split (the first rows
//        of a crawl are one dump), each page cached under texts/rows/ so an interrupted fetch
//        continues; requests are paced, carry the owner's Hugging Face token when one is saved
//        (it raises the datasets-server's rate limit), and back off on 429 for up to ~10 minutes.
//

import Foundation
import RaoLMCore
import RaoLMModel
import RaoLMTraining

public enum GutenbergText {
    /// A book's body without the distributor's header and footer, lines within a paragraph joined.
    public static func body(id: Int, cache: URL) throws -> String {
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let file = cache.appendingPathComponent("pg\(id).txt")
        var raw: String
        if let cached = try? String(contentsOf: file, encoding: .utf8) {
            raw = cached
        } else {
            guard let url = URL(string: "https://www.gutenberg.org/cache/epub/\(id)/pg\(id).txt") else { throw UmbrellaPackError.missing("book \(id)") }
            let (data, response) = try Blocking.run { try await URLSession.shared.data(from: url) }
            guard (response as? HTTPURLResponse)?.statusCode ?? 200 == 200 else { throw UmbrellaPackError.missing("book \(id) (HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0))") }
            raw = String(decoding: data, as: UTF8.self)
            try data.write(to: file, options: .atomic)
        }
        raw = raw.replacingOccurrences(of: "\r\n", with: "\n")
        if let start = raw.range(of: "*** START OF") {
            raw = String(raw[start.upperBound...])
            if let line = raw.firstIndex(of: "\n") { raw = String(raw[raw.index(after: line)...]) }
        }
        if let end = raw.range(of: "*** END OF") { raw = String(raw[..<end.lowerBound]) }
        return raw.components(separatedBy: "\n\n")
            .map { $0.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: " ") }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }

    /// The ids on Project Gutenberg's most-downloaded page (yesterday, the last 7 and 30 days), in
    /// the order they appear, without repeats.
    public static func topIDs() throws -> [Int] {
        guard let url = URL(string: "https://www.gutenberg.org/browse/scores/top") else { return [] }
        let data = try Blocking.run { try await URLSession.shared.data(from: url).0 }
        let html = String(decoding: data, as: UTF8.self)
        // Only the book lists: from the first to the first authors' list (the page's contents name both above them).
        guard let from = html.range(of: #"id="books-last1""#) else { return [] }
        let to = html.range(of: #"id="authors-last1""#, range: from.upperBound ..< html.endIndex)?.lowerBound ?? html.endIndex
        let books = String(html[from.lowerBound ..< to])
        let regex = try NSRegularExpression(pattern: #"href="/ebooks/(\d+)""#)
        var seen = Set<Int>()
        var ids: [Int] = []
        for match in regex.matches(in: books, range: NSRange(books.startIndex..., in: books)) {
            guard let range = Range(match.range(at: 1), in: books), let id = Int(books[range]), seen.insert(id).inserted else { continue }
            ids.append(id)
        }
        return ids
    }

    /// Whether a text reads as English: "the", "and" and "of" make up at least 6% of its words.
    public static func readsAsEnglish(_ text: String) -> Bool {
        let words = text.prefix(200_000).lowercased().split(whereSeparator: { !$0.isLetter })
        guard words.count >= 1000 else { return false }
        let common = words.filter { $0 == "the" || $0 == "and" || $0 == "of" }.count
        return Double(common) / Double(words.count) >= 0.06
    }

    /// Books as documents: the ids asked for (or the most downloaded, up to `top`), the pack's own
    /// books left out, books that do not read as English skipped.
    public static func documents(
        ids: [Int], top: Int?, exclude: Set<Int>, cache: URL, progress: ((String) -> Void)?
    ) throws -> (documents: [CommonsDocument], provenance: [String: String]) {
        var candidates = ids
        if let top { candidates += try topIDs() }
        var documents: [CommonsDocument] = []
        var skipped: [Int] = []
        var seen = Set<Int>()
        for id in candidates where !exclude.contains(id) && seen.insert(id).inserted {
            if let top, documents.count >= top + ids.count { break }
            do {
                let text = try body(id: id, cache: cache)
                guard readsAsEnglish(text) else {
                    skipped.append(id)
                    continue
                }
                documents.append(CommonsDocument(id: "gutenberg:\(id)", text: text))
                progress?("book \(id): \(text.count.formatted()) characters (\(documents.count) so far)")
            } catch {
                skipped.append(id)
                progress?("book \(id) skipped: \(error)")
            }
        }
        let provenance = [
            "ids": documents.map { String($0.id.dropFirst("gutenberg:".count)) }.joined(separator: ","),
            "excluded": exclude.sorted().map(String.init).joined(separator: ","),
            "skipped": skipped.map(String.init).joined(separator: ","),
        ]
        return (documents, provenance)
    }
}

public enum DatasetRows {
    struct Page: Decodable {
        struct Row: Decodable {
            let row: [String: JSONValue]
        }
        let rows: [Row]
        let num_rows_total: Int
    }

    /// A JSON value, enough to read a text column.
    enum JSONValue: Decodable {
        case string(String)
        case other

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let value = try? container.decode(String.self) { self = .string(value) } else { self = .other }
        }

        var string: String? { if case .string(let value) = self { value } else { nil } }
    }

    /// The owner's Hugging Face token, if one is saved where the hub's tools keep it.
    static var token: String? {
        if let value = ProcessInfo.processInfo.environment["HF_TOKEN"], !value.isEmpty { return value }
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/token")
        return (try? String(contentsOf: url, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
    }

    static func page(dataset: String, config: String, split: String, offset: Int, length: Int, progress: ((String) -> Void)? = nil) throws -> Page {
        var components = URLComponents(string: "https://datasets-server.huggingface.co/rows")!
        components.queryItems = [
            URLQueryItem(name: "dataset", value: dataset), URLQueryItem(name: "config", value: config), URLQueryItem(name: "split", value: split),
            URLQueryItem(name: "offset", value: String(offset)), URLQueryItem(name: "length", value: String(length)),
        ]
        guard let url = components.url else { throw UmbrellaPackError.missing("a URL for \(dataset)") }
        var authorized = URLRequest(url: url)
        if let token { authorized.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let request = authorized
        var delay: Double = 5
        for attempt in 0 ..< 8 {
            let (data, response) = try Blocking.run { try await URLSession.shared.data(for: request) }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 200
            if status == 200 { return try JSONDecoder().decode(Page.self, from: data) }
            guard status == 429 || status >= 500, attempt < 7 else {
                throw UmbrellaPackError.missing("rows of \(dataset) at \(offset) (HTTP \(status): \(String(decoding: data.prefix(200), as: UTF8.self)))")
            }
            progress?("HTTP \(status) at offset \(offset); waiting \(Int(delay)) s")
            Thread.sleep(forTimeInterval: delay)
            delay = min(delay * 2, 160)
        }
        throw UmbrellaPackError.missing("rows of \(dataset) at \(offset)")
    }

    /// Rows of `dataset` until `maxTokens` (under `tokenizer`) are gathered, from pages of 100 at
    /// seeded offsets spread over the split.
    public static func documents(
        dataset: String, config: String, split: String, column: String = "text", maxTokens: Int, tokenizer: RaoTokenizer, seed: UInt64,
        cache: URL, progress: ((String) -> Void)?
    ) throws -> (documents: [CommonsDocument], provenance: [String: String]) {
        let length = 100
        let pageCache = cache.appendingPathComponent("rows", isDirectory: true)
            .appendingPathComponent("\(dataset.replacingOccurrences(of: "/", with: "__"))__\(config)__\(split)", isDirectory: true)
        try FileManager.default.createDirectory(at: pageCache, withIntermediateDirectories: true)
        /// A page's texts, from the cache or fetched (and cached) now.
        func texts(at offset: Int) throws -> [String?] {
            let file = pageCache.appendingPathComponent("\(offset).json")
            if let data = try? Data(contentsOf: file), let cached = try? JSONDecoder().decode([String?].self, from: data) { return cached }
            Thread.sleep(forTimeInterval: 0.5)
            let rows = try page(dataset: dataset, config: config, split: split, offset: offset, length: length, progress: progress).rows
            let values = rows.map { $0.row[column]?.string }
            try JSONEncoder().encode(values).write(to: file, options: .atomic)
            return values
        }
        let first = try page(dataset: dataset, config: config, split: split, offset: 0, length: 1, progress: progress)
        let pages = max(1, first.num_rows_total / length)
        var order = Array(0 ..< pages)
        var rng = SplitMix64(seed: seed)
        order.shuffle(using: &rng)
        var documents: [CommonsDocument] = []
        var offsets: [Int] = []
        var tokens = 0
        let started = Date()
        for p in order where tokens < maxTokens {
            let offset = p * length
            for (i, value) in try texts(at: offset).enumerated() {
                guard let text = value, !text.isEmpty else { continue }
                documents.append(CommonsDocument(id: "\(dataset)/\(config)/\(split)#\(offset + i)", text: text))
                tokens += tokenizer.encode(text).count + 1
            }
            offsets.append(offset)
            if offsets.count % 10 == 0 {
                progress?("\(offsets.count) pages, \(documents.count.formatted()) rows, \(tokens.formatted()) of \(maxTokens.formatted()) tokens (\(Int(Date().timeIntervalSince(started))) s)")
            }
        }
        return (documents, ["offsets": offsets.map(String.init).joined(separator: ","), "pageLength": String(length), "column": column,
                            "rowsTotal": String(first.num_rows_total)])
    }
}

public enum CommonsFiles {
    /// Each .txt file is one document; each line of a .jsonl file with a "text" field is one.
    public static func documents(_ paths: [String]) throws -> (documents: [CommonsDocument], provenance: [String: String]) {
        var documents: [CommonsDocument] = []
        var hashes: [String] = []
        for path in paths {
            let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            let data = try Data(contentsOf: url)
            hashes.append("\(url.lastPathComponent):\(ContentHash.sha256Hex(data).prefix(12))")
            if url.pathExtension == "jsonl" {
                for (i, line) in data.split(separator: 0x0A).enumerated() where !line.isEmpty {
                    guard let object = try? JSONSerialization.jsonObject(with: Data(line)) as? [String: Any], let text = object["text"] as? String else { continue }
                    documents.append(CommonsDocument(id: (object["id"] as? String) ?? "\(url.lastPathComponent)#\(i)", text: text))
                }
            } else {
                documents.append(CommonsDocument(id: url.lastPathComponent, text: String(decoding: data, as: UTF8.self)))
            }
        }
        return (documents, ["files": hashes.joined(separator: ",")])
    }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
