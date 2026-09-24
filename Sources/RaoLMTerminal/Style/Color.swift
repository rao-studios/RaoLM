//
//  Color.swift
//  RaoLMTerminal
//
//  WHAT: Terminal colours and their degradation: truecolour → the xterm 256 palette → the
//        16 ANSI colours → the terminal's default.
//  PIN:  The 256 mapping picks the nearer of the 6×6×6 cube and the 24-step grey ramp, as
//        xterm-derived tools do; the 16 mapping is nearest in xterm's default table. The
//        brand palette pins its own 256/16 choices (Palette.brand), so degradation here is
//        only the safety net for colours made at run time (gradients, mixes).
//

import Foundation

public enum Color: Sendable, Hashable {
    case rgb(UInt8, UInt8, UInt8)
    /// xterm 256-colour index.
    case indexed(UInt8)
    /// One of the 16 ANSI colours (0–7 normal, 8–15 bright).
    case ansi(UInt8)
    case `default`

    public init(hex: UInt32) {
        self = .rgb(UInt8((hex >> 16) & 0xFF), UInt8((hex >> 8) & 0xFF), UInt8(hex & 0xFF))
    }

    public var rgbComponents: (r: UInt8, g: UInt8, b: UInt8)? {
        switch self {
        case .rgb(let r, let g, let b): return (r, g, b)
        case .indexed(let index): return Self.xtermRGB(index)
        case .ansi(let index): return Self.xtermRGB(index & 0x0F)
        case .default: return nil
        }
    }

    /// Never upgrades: an indexed colour stays indexed at truecolour depth.
    public func degraded(to depth: ColorDepth) -> Color {
        switch (self, depth) {
        case (.default, _), (_, .none): return .default
        case (.rgb, .trueColor): return self
        case (.rgb(let r, let g, let b), .ansi256): return .indexed(Self.xterm256(r, g, b))
        case (.rgb(let r, let g, let b), .ansi16): return .ansi(Self.ansi16(r, g, b))
        case (.indexed(let index), .ansi16):
            let c = Self.xtermRGB(index)
            return index < 16 ? .ansi(index) : .ansi(Self.ansi16(c.r, c.g, c.b))
        case (.indexed, _), (.ansi, _): return self
        }
    }

    static let cubeLevels: [Int] = [0, 95, 135, 175, 215, 255]

    static let ansiTable: [(Int, Int, Int)] = [
        (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
        (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    ]

    public static func xtermRGB(_ index: UInt8) -> (r: UInt8, g: UInt8, b: UInt8) {
        let i = Int(index)
        if i < 16 {
            let c = ansiTable[i]
            return (UInt8(c.0), UInt8(c.1), UInt8(c.2))
        }
        if i < 232 {
            let n = i - 16
            return (UInt8(cubeLevels[n / 36]), UInt8(cubeLevels[(n / 6) % 6]), UInt8(cubeLevels[n % 6]))
        }
        let grey = UInt8(8 + 10 * (i - 232))
        return (grey, grey, grey)
    }

    public static func xterm256(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> UInt8 {
        func level(_ value: UInt8) -> Int {
            let v = Int(value)
            return v < 48 ? 0 : v < 115 ? 1 : (v - 35) / 40
        }
        let (ri, gi, bi) = (level(r), level(g), level(b))
        let cube = (cubeLevels[ri], cubeLevels[gi], cubeLevels[bi])
        let average = (Int(r) + Int(g) + Int(b)) / 3
        let greyIndex = average > 238 ? 23 : max(0, (average - 3) / 10)
        let greyValue = 8 + 10 * greyIndex
        let cubeDistance = distance((Int(r), Int(g), Int(b)), cube)
        let greyDistance = distance((Int(r), Int(g), Int(b)), (greyValue, greyValue, greyValue))
        return greyDistance < cubeDistance ? UInt8(232 + greyIndex) : UInt8(16 + 36 * ri + 6 * gi + bi)
    }

    public static func ansi16(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> UInt8 {
        var best = 0
        var bestDistance = Int.max
        for (index, entry) in ansiTable.enumerated() {
            let d = distance((Int(r), Int(g), Int(b)), entry)
            if d < bestDistance {
                best = index
                bestDistance = d
            }
        }
        return UInt8(best)
    }

    /// Linear per channel; `.default` mixes to the other colour.
    public static func mix(_ a: Color, _ b: Color, _ t: Double) -> Color {
        guard let ca = a.rgbComponents else { return b }
        guard let cb = b.rgbComponents else { return a }
        let clamped = min(max(t, 0), 1)
        func channel(_ x: UInt8, _ y: UInt8) -> UInt8 {
            UInt8((Double(x) + (Double(y) - Double(x)) * clamped).rounded())
        }
        return .rgb(channel(ca.r, cb.r), channel(ca.g, cb.g), channel(ca.b, cb.b))
    }

    private static func distance(_ a: (Int, Int, Int), _ b: (Int, Int, Int)) -> Int {
        let dr = a.0 - b.0
        let dg = a.1 - b.1
        let db = a.2 - b.2
        return dr * dr + dg * dg + db * db
    }
}

/// `steps` colours from `from` to `to` inclusive (steps == 1 → [from]).
public func gradient(from: Color, to: Color, steps: Int) -> [Color] {
    guard steps > 1 else { return steps == 1 ? [from] : [] }
    return (0..<steps).map { Color.mix(from, to, Double($0) / Double(steps - 1)) }
}
