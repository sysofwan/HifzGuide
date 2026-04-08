import CoreML
import Accelerate
import Foundation

let base = "/Users/sysofwan/repos/w2v-tools"
let numMelBins = 80
let fftSize = 400
let fftLength = 512
let hopSize = 160
let strideVal = 2
let fixedSeqLen = 250
let preemphasisCoeff: Float = 0.97
let melFloor: Float = 1.192092955078125e-07

func loadFloats(_ path: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

let melFilters = loadFloats("\(base)/mel_filters.bin")
let analysisWindow = loadFloats("\(base)/window.bin")
let allAudio = loadFloats("\(base)/alfatihah_f32.bin")
print("Audio: \(allAudio.count) samples")

// Take first 5s
let audio = Array(allAudio[0..<80000])
print("Window: \(audio.count) samples")

// Compute mel
let scaled = audio.map { $0 * 32768.0 }
let numFrames = max(0, (scaled.count - fftSize) / hopSize + 1)
print("Mel frames: \(numFrames)")

let halfFFT = fftLength / 2
let numBins = halfFFT + 1
let log2n = vDSP_Length(log2(Float(fftLength)))
guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { fatalError() }

var melFrames = [[Float]]()
for i in 0..<numFrames {
    let start = i * hopSize
    var frame = Array(scaled[start..<min(start+fftSize, scaled.count)])
    if frame.count < fftSize { frame += [Float](repeating: 0, count: fftSize - frame.count) }
    var mean: Float = 0
    vDSP_meanv(frame, 1, &mean, vDSP_Length(fftSize))
    var neg = -mean
    vDSP_vsadd(frame, 1, &neg, &frame, 1, vDSP_Length(fftSize))
    var pre = [Float](repeating: 0, count: fftSize)
    pre[0] = frame[0]
    for j in 1..<fftSize { pre[j] = frame[j] - preemphasisCoeff * frame[j-1] }
    vDSP_vmul(pre, 1, analysisWindow, 1, &pre, 1, vDSP_Length(fftSize))
    var padded = [Float](repeating: 0, count: fftLength)
    for j in 0..<fftSize { padded[j] = pre[j] }
    var rp = [Float](repeating: 0, count: halfFFT)
    var ip = [Float](repeating: 0, count: halfFFT)
    padded.withUnsafeBufferPointer { pp in
        pp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) { cptr in
            var split = DSPSplitComplex(realp: &rp, imagp: &ip)
            vDSP_ctoz(cptr, 2, &split, 1, vDSP_Length(halfFFT))
            vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
        }
    }
    var power = [Float](repeating: 0, count: numBins)
    power[0] = (rp[0] * rp[0]) / 4.0
    power[halfFFT] = (ip[0] * ip[0]) / 4.0
    for j in 1..<halfFFT { power[j] = (rp[j] * rp[j] + ip[j] * ip[j]) / 4.0 }
    var mel = [Float](repeating: 0, count: numMelBins)
    for m in 0..<numMelBins {
        var sum: Float = 0
        for k in 0..<numBins { sum += melFilters[m * numBins + k] * power[k] }
        mel[m] = logf(max(sum, melFloor))
    }
    melFrames.append(mel)
}
print("Mel computed: \(melFrames.count) frames")
print("mel[0][0..5] = \(melFrames[0][0..<5])")

// Normalize + stride-2
let T = melFrames.count
var means = [Float](repeating: 0, count: numMelBins)
var vars = [Float](repeating: 0, count: numMelBins)
for m in 0..<numMelBins {
    var sum: Float = 0; var sq: Float = 0
    for t in 0..<T { sum += melFrames[t][m]; sq += melFrames[t][m] * melFrames[t][m] }
    means[m] = sum / Float(T)
    vars[m] = (sq / Float(T)) - means[m] * means[m]
    vars[m] *= Float(T) / Float(T - 1)
}
var normed = [[Float]]()
for t in 0..<T {
    var row = [Float](repeating: 0, count: numMelBins)
    for m in 0..<numMelBins { row[m] = (melFrames[t][m] - means[m]) / sqrtf(vars[m] + 1e-7) }
    normed.append(row)
}
let numPairs = T / strideVal
var features = [Float]()
for p in 0..<numPairs {
    features.append(contentsOf: normed[p * 2])
    if p * 2 + 1 < T { features.append(contentsOf: normed[p * 2 + 1]) }
    else { features.append(contentsOf: [Float](repeating: 1.0, count: numMelBins)) }
}
let targetLen = fixedSeqLen * numMelBins * strideVal
if features.count < targetLen {
    features.append(contentsOf: [Float](repeating: 1.0, count: targetLen - features.count))
} else if features.count > targetLen {
    features = Array(features.prefix(targetLen))
}
print("Features: \(features.count) values (expected \(targetLen))")

// Run pipeline
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
print("Loaded all chunks")

let inputArray = try MLMultiArray(shape: [1, 250, 160], dataType: .float32)
let dst = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
for i in 0..<min(features.count, fixedSeqLen * 160) { dst[i] = features[i] }

var current: MLMultiArray = inputArray
for (i, model) in chunks.enumerated() {
    let inputName = (i == 0) ? "input_features" : "hidden_states"
    let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: current)])
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = try model.prediction(from: provider)
    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    let outputName = (i == chunks.count - 1) ? "phoneme_logits" : "hidden_states"
    let output = result.featureValue(for: outputName)!.multiArrayValue!
    print("Chunk \(i): \(String(format: "%.0f", elapsed*1000))ms, dtype=\(output.dataType.rawValue), val[0]=\(output[0].floatValue)")
    
    let shape: [NSNumber] = (i == chunks.count - 1) ? [1, 125, 43] : [1, 250, 1024]
    let fresh = try MLMultiArray(shape: shape, dataType: .float32)
    for j in 0..<output.count { fresh[j] = NSNumber(value: output[j].floatValue) }
    current = fresh
}

// CTC decode
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
