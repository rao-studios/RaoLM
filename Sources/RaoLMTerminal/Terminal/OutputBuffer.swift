//
//  OutputBuffer.swift
//  RaoLMTerminal
//
//  WHAT: Accumulates bytes and writes them to a file descriptor with write(2), retrying
//        partial writes and EINTR/EAGAIN.
//

import Foundation

struct OutputBuffer {
    private(set) var bytes: [UInt8] = []

    mutating func append(_ more: [UInt8]) { bytes.append(contentsOf: more) }
    mutating func append(_ string: String) { bytes.append(contentsOf: string.utf8) }

    mutating func flush(to fd: Int32) {
        guard !bytes.isEmpty else { return }
        Self.writeAll(bytes, to: fd)
        bytes.removeAll(keepingCapacity: true)
    }

    static func writeAll(_ bytes: [UInt8], to fd: Int32) {
        bytes.withUnsafeBytes { raw in
            guard var pointer = raw.baseAddress else { return }
            var remaining = raw.count
            var stalls = 0
            while remaining > 0 {
                let written = write(fd, pointer, remaining)
                if written < 0 {
                    if errno == EINTR { continue }
                    if errno == EAGAIN, stalls < 200 {
                        stalls += 1
                        usleep(1_000)
                        continue
                    }
                    return
                }
                pointer = pointer.advanced(by: written)
                remaining -= written
            }
        }
    }
}
