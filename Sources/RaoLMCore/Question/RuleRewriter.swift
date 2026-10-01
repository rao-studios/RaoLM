//
//  RuleRewriter.swift
//  RaoLMCore
//
//  WHAT: The baseline question rewriter: a question in one of the dataset's own templates
//        becomes the kind's corpus-style stem, the subject kept as written. The umbrella's
//        commons-prompted adapter is benched against it (Docs/ARCHITECTURE.md, "Phase 3").
//  PIN:  Inverts `DatasetVoices.questions` exactly: every template becomes an anchored pattern
//        with the subject as the one capture, matched without regard to case on the template's
//        words, so a bench on the dataset's questions reaches 100% by construction. Anything
//        else returns nil: this rewriter knows nothing of questions in other words.
//

import Foundation

public enum RuleRewriter {
    struct Template {
        let kind: FactKind
        let pattern: NSRegularExpression
    }

    /// Longer templates first, so "In what year was {s} founded?" beats "When was {s} born?" on
    /// its own words and a shorter template never captures the longer one's extra words.
    static let templates: [Template] = {
        var all: [(kind: FactKind, text: String)] = []
        for (kind, texts) in DatasetVoices.questions { for text in texts { all.append((kind, text)) } }
        all.sort { $0.text.count > $1.text.count || ($0.text.count == $1.text.count && $0.text < $1.text) }
        return all.compactMap { entry in
            let parts = entry.text.components(separatedBy: "{s}")
            guard parts.count == 2 else { return nil }
            let pattern = "^\\s*" + NSRegularExpression.escapedPattern(for: parts[0]) + "(.+?)" + NSRegularExpression.escapedPattern(for: parts[1]) + "\\s*$"
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return nil }
            return Template(kind: entry.kind, pattern: regex)
        }
    }()

    /// The stem for `question`, and the fact kind it asks about; nil when no template matches.
    public static func rewrite(_ question: String) -> (stem: String, kind: FactKind)? {
        let text = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let range = NSRange(text.startIndex..., in: text)
        for template in templates {
            guard let match = template.pattern.firstMatch(in: text, range: range), match.numberOfRanges > 1,
                  let subjectRange = Range(match.range(at: 1), in: text) else { continue }
            let subject = text[subjectRange].trimmingCharacters(in: .whitespaces)
            guard !subject.isEmpty, let stem = DatasetVoices.stems[template.kind] else { continue }
            return (fill(stem, subject: subject), template.kind)
        }
        return nil
    }

    /// `{s}` as the subject is written; `{S}` with its first letter capitalised, as a sentence opens.
    public static func fill(_ template: String, subject: String) -> String {
        let capitalised = subject.prefix(1).uppercased() + subject.dropFirst()
        return template.replacingOccurrences(of: "{s}", with: subject).replacingOccurrences(of: "{S}", with: capitalised)
    }

    /// The stem a dataset question of `kind` should rewrite to, for a subject in its corpus form.
    public static func stem(of kind: FactKind, subject: String) -> String? {
        DatasetVoices.stems[kind].map { fill($0, subject: subject) }
    }

    /// Every question template of a kind, filled; what dataset v2 stores on a fact.
    public static func questions(of kind: FactKind, subject: String) -> [String] {
        (DatasetVoices.questions[kind] ?? []).map { fill($0, subject: subject) }
    }

    /// Two stems compared as the bench compares them: lowercase, spaces collapsed, trailing punctuation gone.
    public static func normalised(_ stem: String) -> String {
        var text = stem.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = text.last, ".?!:,;".contains(last) { text.removeLast() }
        return text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
