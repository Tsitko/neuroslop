import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Tracks process memory usage (RSS) for monitoring during training and benchmarks.
public struct MemoryMonitor: Sendable {

    public init() {}

    /// Current resident set size in bytes.
    public var currentRSS: Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int(info.resident_size)
    }

    /// Format bytes as human-readable string.
    public static func formatBytes(_ bytes: Int) -> String {
        let mb = Double(bytes) / (1024 * 1024)
        if mb >= 1024 {
            return String(format: "%.2f GB", mb / 1024)
        }
        return String(format: "%.1f MB", mb)
    }

    /// Print current memory usage with a label.
    public func log(_ label: String = "") {
        let rss = currentRSS
        let prefix = label.isEmpty ? "" : "[\(label)] "
        print("\(prefix)Memory RSS: \(MemoryMonitor.formatBytes(rss))")
    }

    /// Measure memory delta across a block.
    public func measure<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
        let before = currentRSS
        let result = try body()
        let after = currentRSS
        let delta = after - before
        let sign = delta >= 0 ? "+" : ""
        print("[\(label)] Memory: \(MemoryMonitor.formatBytes(after)) (\(sign)\(MemoryMonitor.formatBytes(delta)))")
        return result
    }

    /// Estimated memory for a Matrix (just the data array, not counting Swift overhead).
    public static func matrixBytes(rows: Int, cols: Int) -> Int {
        rows * cols * MemoryLayout<Float>.stride
    }

    public static func matrixBytesFormatted(rows: Int, cols: Int) -> String {
        formatBytes(matrixBytes(rows: rows, cols: cols))
    }
}
