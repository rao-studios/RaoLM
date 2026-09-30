//
//  FactValidator.swift
//  RaoLMCore
//
//  WHAT: Proves the property the citation evaluation depends on: every fact is stated in
//        exactly one place in the corpus, at exactly the offsets the fact records.
//

import Foundation

public enum FactValidator {

    /// Non-overlapping occurrences of `needle` in `haystack`.
    public static func occurrences(of needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var searchRange = haystack.startIndex..<haystack.endIndex
        while let found = haystack.range(of: needle, range: searchRange) {
            count += 1
            searchRange = found.upperBound..<haystack.endIndex
        }
        return count
    }

    /// Everything wrong with a corpus, or an empty array.
    public static func validate(
        documents: [CorpusDocument], negativeSubjects: [String], maxChars: Int, minChars: Int
    ) -> [String] {
        var problems: [String] = []
        let texts = documents.map(\.text)

        // Subjects are exclusive to their document; negative subjects appear nowhere.
        for (i, document) in documents.enumerated() {
            for (j, text) in texts.enumerated() where j != i && text.contains(document.subject) {
                problems.append("subject '\(document.subject)' of \(document.id) also appears in \(documents[j].id)")
            }
        }
        for subject in negativeSubjects {
            for (j, text) in texts.enumerated() where text.contains(subject) {
                problems.append("negative subject '\(subject)' appears in \(documents[j].id)")
            }
        }

        var seenIDs = Set<String>()
        for (i, document) in documents.enumerated() {
            if !seenIDs.insert(document.id).inserted {
                problems.append("duplicate document id \(document.id)")
            }
            if !DocumentID.isValid(document.id) {
                problems.append("invalid document id \(document.id)")
            }
            let paragraphs = document.partitions.map(\.text)
            if Set(paragraphs).count != paragraphs.count {
                problems.append("\(document.id): a partition repeats")
            }
            for partition in document.partitions {
                let length = partition.text.utf8.count
                if length > maxChars || length < minChars {
                    problems.append("\(document.id) partition \(partition.index): length \(length) outside \(minChars)…\(maxChars)")
                }
                if partition.textSHA256 != ContentHash.sha256Hex(partition.text) {
                    problems.append("\(document.id) partition \(partition.index): stale text hash")
                }
            }
            for fact in document.facts {
                guard document.partitions.indices.contains(fact.partitionIndex) else {
                    problems.append("\(fact.id): partition \(fact.partitionIndex) does not exist")
                    continue
                }
                if !fact.prompt.contains(document.subject) {
                    problems.append("\(fact.id): prompt does not name the subject")
                }
                let text = Array(document.partitions[fact.partitionIndex].text.utf8)
                func slice(_ start: Int, _ end: Int) -> String? {
                    guard start >= 0, end <= text.count, start <= end else { return nil }
                    return String(decoding: text[start..<end], as: UTF8.self)
                }
                if slice(fact.answerStart, fact.answerEnd) != fact.answer {
                    problems.append("\(fact.id): answer offsets do not point at '\(fact.answer)'")
                }
                if slice(fact.sentenceStart, fact.sentenceStart + fact.sentence.utf8.count) != fact.sentence {
                    problems.append("\(fact.id): sentence offsets are wrong")
                }
                if !fact.sentence.hasPrefix(fact.prompt + fact.answer) || !fact.answer.hasPrefix(" ") {
                    problems.append("\(fact.id): answer does not continue the prompt")
                }
                if occurrences(of: fact.prompt, in: texts[i]) != 1 {
                    problems.append("\(fact.id): prompt does not occur exactly once")
                }
                for paraphrase in fact.paraphrases where texts[i].contains(paraphrase) {
                    problems.append("\(fact.id): paraphrase occurs in the corpus")
                }
            }
        }
        return problems
    }
}
