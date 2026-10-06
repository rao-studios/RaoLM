//
//  WorkArea.swift
//  RaoLMCore
//
//  WHAT: RaoLM's work area on the T9, where everything it keeps lives: the data root (runs,
//        snapshots, braids, its Thread's storage), the braid datasets and the base models it
//        downloads. Model training belongs on the external drive, not the internal disk.
//  IN:   RAOLM_WORK_AREA, else /Volumes/T9/rao/projects/raolm.
//  OUT:  DataRoot's default, DatasetsRoot, the base-model downloads (UmbrellaPacks), the doctor.
//  PIN:  With the drive unplugged and nothing pointing elsewhere, RaoLM says so (GlobalOptions)
//        rather than writing to the internal disk. --data-dir and RAOLM_DATA_DIR still name a
//        data root of their own; RAOLM_DATASETS_DIR a datasets folder.
//

import Foundation

public enum WorkArea {
    public static let defaultPath = "/Volumes/T9/rao/projects/raolm"
    public static let environmentKey = "RAOLM_WORK_AREA"

    public static func url(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let value = environment[environmentKey], !value.isEmpty {
            return URL(fileURLWithPath: (value as NSString).expandingTildeInPath, isDirectory: true).standardizedFileURL
        }
        return URL(fileURLWithPath: defaultPath, isDirectory: true)
    }

    /// Whether the work area is there to write to: the T9 is mounted (or RAOLM_WORK_AREA exists).
    public static func isAvailable(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url(environment: environment).path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    /// `<work area>/db`: the data root when no --data-dir or RAOLM_DATA_DIR names one.
    public static func dataRoot(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        url(environment: environment).appendingPathComponent("db", isDirectory: true)
    }

    /// `<work area>/models/huggingface`: where RaoLM downloads base models (`HubDownloader(home:)`).
    public static func modelsHome(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        url(environment: environment)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
    }
}
