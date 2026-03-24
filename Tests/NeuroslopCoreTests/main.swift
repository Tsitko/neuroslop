import Foundation

print("=== neuroslop tests ===\n")

runMatrixTests()
runActivationTests()
runMLPTests()
runTraceCompareTests()

print("\n=== Results: \(testsPassed) passed, \(testsFailed) failed ===")

if testsFailed > 0 {
    exit(1)
}
