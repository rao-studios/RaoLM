//
//  BraidArt.swift
//  RaoLMStudio
//
//  WHAT: The drawings of the braid panel. A node's database: a pixel cylinder with one pixel per
//        partition, graded from the ground to the node's strand colour as its weights memorise
//        it, gold where the token under the cursor cites it, with the partitions' own words
//        raining into it while they wait to be indexed. A node's hypervisor: its token
//        embedding and blocks as a stack that lights up as training moves them and as a token
//        passes through. And the strands: each node's colour running up to the gold ring the
//        umbrella sits on, a pulse per generated token as bright as that Thread's gate.
//  PIN:  Everything is a function of the state and the animation phase, so a frame can be
//        tested as text and no animation state lives outside StudioState.
//

import Foundation
import RaoLM
import RaoLMTerminal

enum BraidArt {
    static func strandColor(_ index: Int, _ palette: Palette) -> Color {
        palette.braid.isEmpty ? .default : palette.braid[index % palette.braid.count]
    }

    /// A dark reference to grade from, even when the palette leaves the terminal's own background.
    static func ground(_ palette: Palette) -> Color {
        palette.base.background.rgbComponents != nil ? palette.base.background : Color(hex: 0x1A1A1A)
    }

    /// Smooth 0…1 wave for shimmer, from the phase and a per-item offset.
    static func wave(_ phase: Int, _ offset: Int, period: Int = 16) -> Double {
        let t = Double((phase + offset * 5) % period) / Double(period)
        return 0.5 - 0.5 * cos(t * 2 * .pi)
    }

    // MARK: - The database

    static let databaseWidth = 14
    static let databaseRows = 6
    /// Interior pixels of the cylinder: 12 wide, 8 tall.
    static let interiorWidth = 12
    static let interiorHeight = 8

    /// Interior pixel (x, y) of cell `k` when `count` cells fill the cylinder from the bottom up.
    static func slot(_ k: Int, count: Int) -> (x: Int, y: Int) {
        let capacity = interiorWidth * interiorHeight
        let index = count <= capacity ? k : k * capacity / max(count, 1)
        return (1 + index % interiorWidth, 2 + interiorHeight - 1 - index / interiorWidth)
    }

    static func database(
        _ state: StrandState, colour: Color, palette: Palette, phase: Int, cited: Set<Int>, training: Bool
    ) -> PixelCanvas {
        var pixels = PixelCanvas(width: databaseWidth, rows: databaseRows)
        let ground = self.ground(palette)
        let rim = Color.mix(colour, palette.line, 0.35)
        let wall = Color.mix(ground, colour, 0.45)
        let hollow = Color.mix(ground, colour, 0.10)
        // Top rim and surface, walls, bottom.
        pixels.fill(x: 2, y: 0, width: 10, height: 1, rim)
        pixels[1, 1] = rim
        pixels[12, 1] = rim
        pixels.fill(x: 2, y: 1, width: 10, height: 1, Color.mix(ground, colour, 0.22))
        for y in 2..<10 {
            pixels[0, y] = wall
            pixels[13, y] = wall
        }
        // The empty inside shimmers faintly, column by column, like falling code.
        for y in 2..<(2 + interiorHeight) {
            for x in 1...interiorWidth {
                let fall = (phase / 2 + x * 7 - y + 64) % 11
                pixels[x, y] = fall == 0 ? Color.mix(hollow, colour, 0.22) : (fall == 1 ? Color.mix(hollow, colour, 0.12) : hollow)
            }
        }
        // Platter rims: the stacked discs of a database.
        for y in [4, 7] {
            pixels[0, y] = rim
            pixels[13, y] = rim
        }
        pixels.fill(x: 1, y: 10, width: 12, height: 1, wall)
        pixels.fill(x: 2, y: 11, width: 10, height: 1, Color.mix(ground, colour, 0.3))

        let count = state.cells.count
        for (k, cell) in state.cells.enumerated() {
            let (x, y) = slot(k, count: count)
            let colourOfCell: Color
            if cited.contains(k) {
                colourOfCell = palette.goldColor
            } else {
                switch cell.state {
                case .pending:
                    // Waiting to be indexed: a slow blink in the strand colour.
                    colourOfCell = Color.mix(hollow, colour, 0.25 + 0.35 * wave(phase, k, period: 10))
                case .indexed, .learned:
                    let memorised = Double(cell.memorised ?? 0)
                    var level = 0.28 + 0.72 * memorised
                    if training, memorised < 0.95 { level = min(1, level + 0.18 * wave(phase, k)) }
                    let base = Color.mix(hollow, colour, level)
                    colourOfCell = cell.state == .learned ? Color.mix(base, palette.line, 0.18) : base
                }
            }
            pixels[x, y] = colourOfCell
        }
        return pixels
    }

    /// The partitions' own words falling into the database while they wait to be indexed.
    static func rain(
        _ state: StrandState, origin: (x: Int, y: Int), clip: Rect, colour: Color, palette: Palette, phase: Int, frame: inout Frame
    ) {
        guard frame.glyphs.upperHalf != nil else { return }
        let busy = state.stage == .exporting || state.stage == .reindexing
        let count = state.cells.count
        var streams = 0
        for (k, cell) in state.cells.enumerated() where cell.state == .pending || busy {
            if streams >= interiorWidth { break }
            let (x, y) = slot(k, count: count)
            let landing = y / 2
            let fall = landing + 2
            let head = (phase + k * 3) % (fall + 3) - 2
            guard head >= -1 else { continue }
            let letters = Array(cell.glyphs.unicodeScalars.filter { $0.value > 32 && $0.value < 127 })
            guard !letters.isEmpty else { continue }
            for trail in 0..<3 {
                let row = head - trail
                guard row >= -1, row < landing else { continue }
                let letter = letters[(phase / 2 + k + trail) % letters.count]
                let level = trail == 0 ? 1.0 : (trail == 1 ? 0.55 : 0.3)
                let style = Style(foreground: Color.mix(ground(palette), trail == 0 ? palette.line : colour, level),
                                  background: palette.base.background)
                frame.canvas.put(String(Character(letter)), x: origin.x + x, y: origin.y + row, style: style, clip: clip)
            }
            streams += 1
        }
    }

    // MARK: - The hypervisor

    /// Rows of the node's transformer from the top: the hidden state sent up, the blocks, the
    /// shared embedding. Long stacks are grouped to fit `rows`.
    static func stackRows(blocks: Int, rows: Int) -> [(label: String, blocks: Range<Int>)] {
        let room = max(1, rows - 2)
        let groups = min(blocks, room)
        var result: [(String, Range<Int>)] = []
        for g in (0..<groups).reversed() {
            let lower = g * blocks / groups
            let upper = (g + 1) * blocks / groups
            result.append((upper - lower == 1 ? "b\(lower + 1)" : "b\(lower + 1)-\(upper)", lower..<upper))
        }
        return result
    }

    static func hypervisor(
        _ state: StrandState, rect: Rect, colour: Color, pulse: Double?, palette: Palette, phase: Int, frame: inout Frame
    ) {
        guard rect.height >= 3, rect.width >= 6 else { return }
        let blocks = max(1, state.blockActivity.count)
        let rows = stackRows(blocks: blocks, rows: rect.height)
        let ground = self.ground(palette)
        let training = state.stage == .training
        let barWidth = max(2, min(5, rect.width - 4))
        // Where a token's pulse is: 0 at the embedding, 1 at the top of the stack.
        func lit(_ position: Double) -> Double {
            guard let pulse else { return 0 }
            return max(0, 1 - abs(pulse * Double(rows.count + 1) - position) / 1.2)
        }
        let topStyle = Style(foreground: Color.mix(ground, palette.line, 0.4 + 0.6 * lit(Double(rows.count + 1))), background: palette.base.background)
        if let cut = state.cut {
            // Above the node's own blocks: the umbrella's frozen trunk, in the umbrella's gold.
            let trunk = Style(foreground: Color.mix(ground, palette.goldColor, 0.5 + 0.5 * lit(Double(rows.count + 1))), background: palette.base.background)
            frame.canvas.put("▲ trunk \(cut)+", x: rect.minX, y: rect.minY, style: trunk, clip: rect)
        } else {
            frame.canvas.put("▲ last", x: rect.minX, y: rect.minY, style: topStyle, clip: rect)
        }
        for (i, row) in rows.enumerated() {
            let y = rect.minY + 1 + i
            guard y < rect.maxY - 1 else { break }
            let activity = row.blocks.map { $0 < state.blockActivity.count ? Double(state.blockActivity[$0]) : 0 }.max() ?? 0
            var level = training ? 0.25 + 0.75 * activity * (0.75 + 0.25 * wave(phase, i, period: 8)) : 0.3
            level = max(level, lit(Double(rows.count - i)))
            let bar = String(repeating: frame.glyphs.lowerHalf ?? frame.glyphs.barFull, count: barWidth)
            frame.canvas.put(bar, x: rect.minX, y: y, style: Style(foreground: Color.mix(ground, colour, level), background: palette.base.background), clip: rect)
            frame.canvas.put(row.label, x: rect.minX + barWidth + 1, y: y, style: palette.muted, clip: rect)
        }
        let embedLevel = max(0.35, lit(0))
        frame.canvas.put("◆ embed", x: rect.minX, y: rect.maxY - 1,
                         style: Style(foreground: Color.mix(ground, palette.goldColor, embedLevel), background: palette.base.background), clip: rect)
    }

    // MARK: - The strands

    /// Each node's strand runs from above its panel to the gold ring at the centre, where the
    /// umbrella sits; pulses travel up it, as bright as the Thread's gate. A node whose panel sits
    /// under the ring (the middle of an odd count) arrives from below: its colour frames the ring.
    static func strands(
        centres: [Int], rect: Rect, pulses: [(strand: Int, progress: Double, strength: Float, open: Bool)], palette: Palette,
        glyphs: Glyphs, frame: inout Frame
    ) {
        guard !rect.isEmpty, !centres.isEmpty else { return }
        let ring = rect.minX + rect.width / 2
        let ground = self.ground(palette)
        func pulseStyle(_ colour: Color, _ pulse: (strand: Int, progress: Double, strength: Float, open: Bool)) -> Style {
            let level = pulse.open ? 0.45 + 0.55 * Double(pulse.strength) : 0.3
            return Style(foreground: Color.mix(ground, colour == .default ? palette.line : Color.mix(colour, palette.line, 0.4), level),
                         background: palette.base.background, attributes: pulse.open ? [.bold] : [])
        }
        let froms = centres.map { min(max($0, rect.minX), rect.maxX - 1) }
        let under = centres.indices.filter { abs(froms[$0] - ring) < 2 }
        for (index, from) in froms.enumerated() where !under.contains(index) {
            let colour = strandColor(index, palette)
            let step = from < ring ? 1 : -1
            let span = max(1, abs(ring - from))
            var x = from
            var walked = 0
            while x != ring {
                let t = Double(walked) / Double(span)
                let glyph = t < 0.4 ? glyphs.wave : (t < 0.72 ? glyphs.ripple : glyphs.horizontal)
                let tinted = Color.mix(colour, palette.line, max(0, t - 0.55) / 0.45)
                frame.canvas.put(glyph, x: x, y: rect.minY, style: Style(foreground: tinted, background: palette.base.background), clip: rect)
                x += step
                walked += 1
            }
            for pulse in pulses where pulse.strand == index {
                let offset = Int((pulse.progress * Double(span)).rounded())
                let px = from + step * min(span - 1, max(0, offset))
                frame.canvas.put(pulse.open ? glyphs.live : glyphs.dot, x: px, y: rect.minY, style: pulseStyle(colour, pulse), clip: rect)
            }
        }
        for index in under {
            let colour = strandColor(index, palette)
            let pulse = pulses.last { $0.strand == index }
            for x in [ring - 1, ring + 1] {
                if let pulse {
                    frame.canvas.put(pulse.open ? glyphs.live : glyphs.dot, x: x, y: rect.minY, style: pulseStyle(colour, pulse), clip: rect)
                } else {
                    frame.canvas.put(glyphs.horizontal, x: x, y: rect.minY, style: Style(foreground: colour, background: palette.base.background), clip: rect)
                }
            }
        }
        frame.canvas.put(glyphs.verified, x: ring, y: rect.minY, style: palette.gold, clip: rect)
    }

    // MARK: - Tokens

    /// A generated token's colour: its dominant Thread's strand, off-white when Threads share it.
    static func tokenStyle(_ trace: TokenTrace, strands: [String], palette: Palette, dimUncited: Bool = true) -> Style {
        guard let shares = trace.strands, !shares.isEmpty else { return palette.text }
        if trace.dominantStrand(threshold: 0.6)?.strand == BraidStrandRef.commonsName {
            // What the base model already knew: no Thread's, the umbrella's gold.
            var style = Style(foreground: palette.goldColor, background: palette.base.background)
            if dimUncited, trace.uncited { style = style.dim() }
            return style
        }
        guard let dominant = trace.dominantStrand(threshold: 0.6), let index = strands.firstIndex(of: dominant.strand) else {
            return Style(foreground: palette.line, background: palette.base.background)
        }
        var style = Style(foreground: strandColor(index, palette), background: palette.base.background)
        if dimUncited, trace.uncited { style = style.dim() }
        return style
    }
}
