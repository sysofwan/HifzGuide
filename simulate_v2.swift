import Accelerate
import CoreML
import Foundation

let base = "/Users/sysofwan/repos/w2v-tools"
let fixedSeqLen = 250
let outputTimeSteps = 125
let windowSamples = 80000
let hopSamples = 48000
let confirmTimeSteps = 75
let sampleRate: Float = 16000
let silenceDurationForFlush: Double = 0.3

let phonemeMap: [Int: String] = [
    0: "", 1: "ء", 2: "ب", 3: "ت", 4: "ث", 5: "ج", 6: "ح", 7: "خ",
    8: "د", 9: "ذ", 10: "ر", 11: "ز", 12: "س", 13: "ش", 14: "ص",
    15: "ض", 16: "ط", 17: "ظ", 18: "ع", 19: "غ", 20: "ف", 21: "ق",
    22: "ك", 23: "ل", 24: "م", 25: "ن", 26: "ه", 27: "و", 28: "ي",
    29: "ا", 30: "ۦ", 31: "ۥ", 32: "َ", 33: "ُ", 34: "ِ", 35: "ۜ",
    36: "ـ", 37: "ٲ", 38: "ڇ", 39: "ں", 40: "۾", 41: "ؙ", 42: "ۦ"
]

func decodePhonemes(_ ids: [Int]) -> String { ids.compactMap { phonemeMap[$0] }.joined() }

func loadFloats(_ path: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

// Load precomputed features (34 windows x 250 x 160)
let allFeatures = loadFloats("\(base)/alfatihah_features.bin")
let numWindows = allFeatures.count / (fixedSeqLen * 160)
print("Loaded \(numWindows) precomputed feature windows")

// Load RMS values (100ms chunks)
let rmsValues = loadFloats("\(base)/alfatihah_rms.bin")
print("Loaded \(rmsValues.count) RMS values")

// Load audio for timing reference
let allAudio = loadFloats("\(base)/alfatihah_f32.bin")
let totalDuration = Float(allAudio.count) / sampleRate
print("Audio: \(allAudio.count) samples (\(String(format: "%.1f", totalDuration))s)")

// Load CoreML chunks
let chunkNames = ["MuaalemChunkA_6BIT", "MuaalemChunkB_6BIT", "MuaalemChunkC_6BIT",
                   "MuaalemChunkD_6BIT", "MuaalemChunkE_6BIT", "MuaalemChunkF_6BIT"]
var chunks: [MLModel] = []
let config = MLModelConfiguration()
config.computeUnits = .all
for name in chunkNames {
    let url = URL(fileURLWithPath: "\(base)/coreml_models_chunked/\(name).mlpackage")
    let compiled = try! MLModel.compileModel(at: url)
    chunks.append(try! MLModel(contentsOf: compiled, configuration: config))
    print("  Loaded \(name)")
}

func copyToFresh(source: MLMultiArray, shape: [NSNumber]) -> MLMultiArray {
    let fresh = try! MLMultiArray(shape: shape, dataType: .float32)
    let count = source.count
    if source.dataType == .float16 {
        let srcPtr = source.dataPointer.assumingMemoryBound(to: UInt16.self)
        let dstPtr = fresh.dataPointer.assumingMemoryBound(to: Float.self)
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var dstBuf = vImage_Buffer(data: dstPtr, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
    } else {
        memcpy(fresh.dataPointer, source.dataPointer, count * MemoryLayout<Float>.size)
    }
    return fresh
}

func runPipeline(features: [Float]) -> MLMultiArray {
    let inputArray = try! MLMultiArray(shape: [1, 250, 160], dataType: .float32)
    let dst = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
    let count = min(features.count, fixedSeqLen * 160)
    features.withUnsafeBufferPointer { src in
        let _ = memcpy(dst, src.baseAddress!, count * MemoryLayout<Float>.size)
    }
    
    var current: MLMultiArray = inputArray
    for (i, model) in chunks.enumerated() {
        let inputName = (i == 0) ? "input_features" : "hidden_states"
        let provider = try! MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: current)])
        let result = try! model.prediction(from: provider)
        let outputName: String
        if i == 0 { outputName = "hidden_states" }
        else if i == chunks.count - 1 { outputName = "phoneme_logits" }
        else { outputName = "hidden_states_out" }
        let output = result.featureValue(for: outputName)!.multiArrayValue!
        let shape: [NSNumber] = (i == chunks.count - 1) ? [1, 125, 43] : [1, 250, 1024]
        current = copyToFresh(source: output, shape: shape)
    }
    return current
}

func ctcDecode(logits: MLMultiArray, from: Int, to: Int) -> [Int] {
    var ids = [Int](); var prev = -1
    for t in from..<to {
        var best = 0; var bestVal = logits[[0, t, 0] as [NSNumber]].floatValue
        for c in 1..<43 {
            let v = logits[[0, t, c] as [NSNumber]].floatValue
            if v > bestVal { bestVal = v; best = c }
        }
        if best != 0 && best != prev { ids.append(best) }
        prev = best
    }
    return ids
}

func ctcDecodeAll(logits: MLMultiArray) -> [Int] { ctcDecode(logits: logits, from: 0, to: outputTimeSteps) }

// VAD: check if silence at a given sample offset for a given duration
let avgRMS = rmsValues.reduce(0, +) / Float(rmsValues.count)
let silenceThresholdRMS = avgRMS * 0.15
print("Avg RMS: \(String(format: "%.4f", avgRMS)), silence threshold: \(String(format: "%.6f", silenceThresholdRMS))")

func isSilent(atSample offset: Int, durationSamples: Int = 4800) -> Bool {
    let vadChunkSize = 1600  // 100ms
    let startChunk = offset / vadChunkSize
    let numChunks = durationSamples / vadChunkSize
    guard startChunk >= 0, startChunk + numChunks <= rmsValues.count else { return true }
    for c in startChunk..<(startChunk + numChunks) {
        if rmsValues[c] > silenceThresholdRMS { return false }
    }
    return true
}

// MARK: - Simulation

print("\n=== Simulation ===\n")

var windowIndex = 0  // Index into precomputed features
var sampleOffset = 0 // Current position in audio
var segmentTexts: [String] = []
var currentSegment = ""
var inferenceCount = 0
var totalInferenceTime: Double = 0

// Walk through audio with 3s hops
while windowIndex < numWindows {
    let featureOffset = windowIndex * fixedSeqLen * 160
    let features = Array(allFeatures[featureOffset..<(featureOffset + fixedSeqLen * 160)])
    let audioTime = String(format: "%.1f", Float(sampleOffset) / sampleRate)
    
    let t0 = CFAbsoluteTimeGetCurrent()
    let logits = runPipeline(features: features)
    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    inferenceCount += 1
    totalInferenceTime += elapsed
    
    let confirmedIDs = ctcDecode(logits: logits, from: 0, to: confirmTimeSteps)
    let unconfirmedIDs = ctcDecode(logits: logits, from: confirmTimeSteps, to: outputTimeSteps)
    let confirmed = decodePhonemes(confirmedIDs)
    let unconfirmed = decodePhonemes(unconfirmedIDs)
    
    // Check for silence after the confirmed region (at sample offset + 3s)
    let confirmEnd = sampleOffset + hopSamples
    let silenceAfterConfirm = isSilent(atSample: confirmEnd, durationSamples: 4800)
    
    if silenceAfterConfirm {
        // Flush: confirm everything and end segment
        let allIDs = ctcDecodeAll(logits: logits)
        let allText = decodePhonemes(allIDs)
        currentSegment += allText
        print("[\(audioTime)s] WINDOW+FLUSH (\(String(format: "%.0f", elapsed * 1000))ms): \"\(allText)\"")
        if !currentSegment.isEmpty {
            segmentTexts.append(currentSegment)
            print("  >>> SEGMENT: \"\(currentSegment)\"\n")
            currentSegment = ""
        }
        
        // Skip silence windows
        sampleOffset += hopSamples
        windowIndex += 1
        while sampleOffset < allAudio.count && isSilent(atSample: sampleOffset, durationSamples: 4800) {
            sampleOffset += hopSamples
            windowIndex += 1
        }
    } else {
        print("[\(audioTime)s] WINDOW (\(String(format: "%.0f", elapsed * 1000))ms): confirmed=\"\(confirmed)\", unconfirmed=\"\(unconfirmed)\"")
        currentSegment += confirmed
        sampleOffset += hopSamples
        windowIndex += 1
    }
}

// Flush remaining
if !currentSegment.isEmpty {
    segmentTexts.append(currentSegment)
    print("\n  >>> FINAL SEGMENT: \"\(currentSegment)\"")
}

print("\n=== Results ===")
print("Segments: \(segmentTexts.count)")
for (i, s) in segmentTexts.enumerated() {
    print("  \(i + 1): \(s)")
}
print("\nFull: \(segmentTexts.joined(separator: " | "))")
print("\nStats: \(inferenceCount) inferences, avg \(String(format: "%.0f", totalInferenceTime / Double(max(1,inferenceCount)) * 1000))ms each")
