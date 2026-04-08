import CoreML
import Foundation

func loadFloats(_ path: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

let base = "/Users/sysofwan/repos/w2v-tools"
let pyLogits = loadFloats("\(base)/test_window0_logits_python.bin")
print("Python logits: \(pyLogits.count) values")  // should be 125 * 43 = 5375

// Put into MLMultiArray
let logits = try MLMultiArray(shape: [1, 125, 43], dataType: .float32)
let dst = logits.dataPointer.assumingMemoryBound(to: Float.self)
for i in 0..<pyLogits.count { dst[i] = pyLogits[i] }

// Check strides
print("Strides: \(logits.strides)")

// Manual decode - two methods
print("\nMethod 1: multi-index [[0, t, c]]")
for t in 0..<3 {
    var vals = [Float]()
    for c in 0..<43 { vals.append(logits[[0, t, c] as [NSNumber]].floatValue) }
    let best = vals.enumerated().max(by: { $0.element < $1.element })!
    print("  t=\(t): best=\(best.offset) (\(best.element)), raw[:5]=\(Array(vals[0..<5]))")
}

print("\nMethod 2: flat indexing [t*43 + c]")
for t in 0..<3 {
    var vals = [Float]()
    for c in 0..<43 { vals.append(dst[t * 43 + c]) }
    let best = vals.enumerated().max(by: { $0.element < $1.element })!
    print("  t=\(t): best=\(best.offset) (\(best.element)), raw[:5]=\(Array(vals[0..<5]))")
}

// What does Python see at t=0?
print("\nPython flat [0:5] = \(Array(pyLogits[0..<5]))")
print("Python flat [43:48] = \(Array(pyLogits[43..<48]))")
