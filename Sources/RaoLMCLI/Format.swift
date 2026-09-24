//
//  Format.swift
//  RaoLMCLI
//
//  WHAT: Number formatting and plain-text tables for the terminal.
//

import Foundation

enum Format {
    static func f(_ value: Float?, _ digits: Int = 3) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.\(digits)f", value)
    }

    static func pct(_ value: Float?) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.0f%%", value * 100)
    }

    static func count(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }

    static func duration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        let minutes = Int(seconds) / 60
        return String(format: "%d min %02d s", minutes, Int(seconds) % 60)
    }

    static func short(_ hash: String?) -> String {
        guard let hash else { return "—" }
        return String(hash.prefix(12)) + "…"
    }

    static func clip(_ text: String, _ width: Int) -> String {
        let flat = text.replacingOccurrences(of: "\n", with: "⏎")
        return flat.count <= width ? flat : String(flat.prefix(max(1, width - 1))) + "…"
    }

    static func table(_ headers: [String], _ rows: [[String]], indent: String = "  ") -> String {
        var widths = headers.map(\.count)
        for row in rows {
            for (i, cell) in row.enumerated() where i < widths.count { widths[i] = max(widths[i], cell.count) }
        }
        func line(_ cells: [String]) -> String {
            indent + cells.enumerated().map { i, cell in
                i == cells.count - 1 ? cell : cell.padding(toLength: widths[i], withPad: " ", startingAt: 0)
            }.joined(separator: "  ")
        }
        var output = [line(headers), indent + widths.map { String(repeating: "─", count: $0) }.joined(separator: "  ")]
        output += rows.map(line)
        return output.joined(separator: "\n")
    }
}
