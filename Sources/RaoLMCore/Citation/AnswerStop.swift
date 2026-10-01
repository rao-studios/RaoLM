//
//  AnswerStop.swift
//  RaoLMCore
//
//  WHAT: Where an answer ends. A question's answer is one sentence: the stem's completion up to
//        its full stop. The generator stops before the token that would open the next sentence,
//        so the completion keeps the words after the value (which is what tells which Thread's
//        document it followed) and nothing more.
//  PIN:  A sentence ends when the text so far ends in a full stop, a question mark or an
//        exclamation mark and the next token opens with whitespace or a capital letter (text
//        packed without a space after its full stop, "sheet.Its walls"), or when the next token
//        holds a newline. Looking at the next token keeps "3.14.7" and "v2." + "1" whole.
//

import Foundation

public enum AnswerStop {
    /// Whether the token `next` would start a new sentence after `previous`, the text so far.
    public static func sentenceEnded(previous: String, next: String) -> Bool {
        if next.contains("\n") { return true }
        guard let last = previous.last(where: { !$0.isWhitespace }), ".?!".contains(last) else { return false }
        guard let first = next.first else { return false }
        return first.isWhitespace || first.isNewline || first.isUppercase
    }
}
