import CoreML
import Accelerate
import Foundation

let base = "/Users/sysofwan/repos/w2v-tools"

func loadFloats(_ path: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

let features = loadFloats("\(base)/test_window0_features.bin")
let config = MLModelConfiguration()
config.computeUnits = .all
let chunkNames = ["MuaalemChunkA_6BIT", "MuaalemChunkB_6BIT", "MuaalemChunkC_6BIT",
                   "MuaalemChunkD_6BIT", "MuaalemChunkE_6BIT", "MuaalemChunkF_6BIT"]
var chunks: [MLModel] = []
for name in chunkNames {
    let url = URL(fileURLWithPath: "\(base)/coreml_models_chunked/\(name).mlpackage")
    let compiled = try MLModel.compileModel(at: url)
    chunks.append(try MLModel(contentsOf: compiled, configuration: config))
}

func copyToFresh(source: MLMultiArray, shape: [NSNumber]) -> MLMultiArray {
    let fresh = try! MLMultiArray(shape: shape, dataType: .float32)
    let dims = shape.map { $0.intValue }
    let srcStrides = source.strides.map { $0.intValue }
    let dstPtr = fresh.dataPointer.assumingMemoryBound(to: Float.self)
    let logicalCount = dims.reduce(1, *)
    
    // Check contiguity
    var isContiguous = true
    var expected = 1
    for d in (0..<dims.count).reversed() {
        if srcStrides[d] != expected { isContiguous = false; break }
        expected *= dims[d]
    }
    
    if isContiguous && source.dataType == .float16 {
        let srcPtr = source.dataPointer.assumingMemoryBound(to: UInt16.self)
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr), height: 1, width: vImagePixelCount(logicalCount), rowBytes: logicalCount * 2)
        var dstBuf = vImage_Buffer(data: dstPtr, height: 1, width: vImagePixelCount(logicalCount), rowBytes: logicalCount * 4)
        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
    } else {
        // Non-contiguous: use multi-index
        var flat = 0
        for b in 0..<dims[0] {
            for t in 0..<dims[1] {
                for c in 0..<dims[2] {
                    dstPtr[flat] = source[[b, t, c] as [NSNumber]].floatValue
                    flat += 1
                }
            }
        }
    }
    return fresh
}

let inputArray = try MLMultiArray(shape: [1, 250, 160], dataType: .float32)
let dst = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
for i in 0..<min(features.count, 40000) { dst[i] = features[i] }

var current: MLMultiArray = inputArray
for (i, model) in chunks.enumerated() {
    let inputName = (i == 0) ? "input_features" : "hidden_states"
    let prov = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: current)])
    let result = try model.prediction(from: prov)
    let outputName: String
    if i == 0 { outputName = "hidden_states" }
    else if i == 5 { outputName = "phoneme_logits" }
    else { outputName = "hidden_states_out" }
    let output = result.featureValue(for: outputName)!.multiArrayValue!
    let shape: [NSNumber] = (i == 5) ? [1, 125, 43] : [1, 250, 1024]
    print("Chunk \(i): strides=\(output.strides)")
    current = copyToFresh(source: output, shape: shape)
}

let phonemeMap: [Int: String] = [
    0: "", 1: "ء", 2: "ب", 3: "ت", 4: "ث", 5: "ج", 6: "ح", 7: "خ",
    8: "د", 9: "ذ", 10: "ر", 11: "ز", 12: "س", 13: "ش", 14: "ص",
    15: "ض", 16: "ط", 17: "ظ", 18: "ع", 19: "غ", 20: "ف", 21: "ق",
    22: "ك", 23: "ل", 24: "م", 25: "ن", 26: "ه", 27: "و", 28: "ي",
    29: "ا", 30: "ۦ", 31: "ۥ", 32: "َ", 33: "ُ", 34: "ِ", 35: "ۜ",
    36: "ـ", 37: "ٲ", 38: "ڇ", 39: "ں", 40: "۾", 41: "ؙ", 42: "ۦ"
]
var ids = [Int](); var prev = -1
for t in 0..<125 {
    var best = 0; var bestVal = current[[0, t, 0] as [NSNumber]].floatValue
    for c in 1..<43 {
        let v = current[[0, t, c] as [NSNumber]].floatValue
        if v > bestVal { bestVal = v; best = c }
    }
    if best != 0 && best != prev { ids.append(best) }
    prev = best
}
print("\nDecoded: \(ids.compactMap { phonemeMap[$0] }.joined())")
