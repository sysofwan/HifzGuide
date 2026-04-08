#!/usr/bin/env swift
// End-to-end test: audio → mel spectrogram → CoreML 6-chunk pipeline → CTC decode
// Run on macOS to verify the entire Swift pipeline produces correct phonemes.

import Accelerate
import CoreML
import Foundation

// --- Parameters ---
let numMelBins = 80
let fftSize = 400
let fftLength = 512
let hopSize = 160
let strideVal = 2
let preemphasisCoeff: Float = 0.97
let melFloor: Float = 1.192092955078125e-07
let fixedSeqLen = 250

// --- Load binary files ---
func loadFloats(_ path: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

let base = "/Users/sysofwan/repos/w2v-tools"
let melFilters = loadFloats("\(base)/mel_filters.bin")
let analysisWindow = loadFloats("\(base)/window.bin")
let audio = loadFloats("\(base)/test_quran_audio.bin")
print("Audio: \(audio.count) samples, range=[\(audio.min()!), \(audio.max()!)]")

// --- Mel Spectrogram (exact copy from MuaalemInference.swift) ---
func computeMelSpectrogram(audio: [Float]) -> [[Float]] {
    let scaled = audio.map { $0 * 32768.0 }
    let numFrames = max(0, (scaled.count - fftSize) / hopSize + 1)
    guard numFrames > 0 else { return [] }
    let halfFFT = fftLength / 2
    let numBins = halfFFT + 1
    let log2n = vDSP_Length(log2(Float(fftLength)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    var melFrames = [[Float]]()
    for i in 0..<numFrames {
        let start = i * hopSize
        let end = min(start + fftSize, scaled.count)
        var frame = Array(scaled[start..<end])
        if frame.count < fftSize { frame.append(contentsOf: [Float](repeating: 0, count: fftSize - frame.count)) }

        var mean: Float = 0
        vDSP_meanv(frame, 1, &mean, vDSP_Length(fftSize))
        var negMean = -mean
        vDSP_vsadd(frame, 1, &negMean, &frame, 1, vDSP_Length(fftSize))

        var preemphasized = [Float](repeating: 0, count: fftSize)
        preemphasized[0] = frame[0]
        for j in 1..<fftSize { preemphasized[j] = frame[j] - preemphasisCoeff * frame[j - 1] }
        vDSP_vmul(preemphasized, 1, analysisWindow, 1, &preemphasized, 1, vDSP_Length(fftSize))

        var padded = [Float](repeating: 0, count: fftLength)
        for j in 0..<fftSize { padded[j] = preemphasized[j] }

        var realPart = [Float](repeating: 0, count: halfFFT)
        var imagPart = [Float](repeating: 0, count: halfFFT)
        padded.withUnsafeBufferPointer { paddedPtr in
            realPart.withUnsafeMutableBufferPointer { realPtr in
                imagPart.withUnsafeMutableBufferPointer { imagPtr in
                    var sc = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                    paddedPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) {
                        vDSP_ctoz($0, 2, &sc, 1, vDSP_Length(halfFFT))
                    }
                    vDSP_fft_zrip(fftSetup, &sc, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
        }

        var powerSpectrum = [Float](repeating: 0, count: numBins)
        powerSpectrum[0] = (realPart[0] * realPart[0]) / 4.0
        powerSpectrum[halfFFT] = (imagPart[0] * imagPart[0]) / 4.0
        for j in 1..<halfFFT { powerSpectrum[j] = (realPart[j] * realPart[j] + imagPart[j] * imagPart[j]) / 4.0 }

        var melEnergies = [Float](repeating: 0, count: numMelBins)
        for m in 0..<numMelBins {
            var energy: Float = 0
            let off = m * numBins
            for j in 0..<numBins { energy += powerSpectrum[j] * melFilters[off + j] }
            melEnergies[m] = log(max(energy, melFloor))
        }
        melFrames.append(melEnergies)
    }
    return melFrames
}

func prepareFeatures(audioSamples: [Float]) -> MLMultiArray {
    let melFrames = computeMelSpectrogram(audio: audioSamples)
    let numFrames = melFrames.count
    guard numFrames > 0 else {
        return try! MLMultiArray(shape: [1, NSNumber(value: fixedSeqLen), 160], dataType: .float32)
    }

    var means = [Float](repeating: 0, count: numMelBins)
    var variances = [Float](repeating: 0, count: numMelBins)
    for bin in 0..<numMelBins {
        var sum: Float = 0
        for t in 0..<numFrames { sum += melFrames[t][bin] }
        means[bin] = sum / Float(numFrames)
        var varSum: Float = 0
        for t in 0..<numFrames { let d = melFrames[t][bin] - means[bin]; varSum += d * d }
        variances[bin] = numFrames > 1 ? varSum / Float(numFrames - 1) : 0
    }

    var normalized = [[Float]](repeating: [Float](repeating: 0, count: numMelBins), count: numFrames)
    for t in 0..<numFrames {
        for bin in 0..<numMelBins {
            normalized[t][bin] = (melFrames[t][bin] - means[bin]) / sqrt(variances[bin] + 1e-7)
        }
    }

    var padFrames = normalized
    if padFrames.count % strideVal != 0 { padFrames.append([Float](repeating: 1.0, count: numMelBins)) }

    let numPairs = padFrames.count / strideVal
    let featureDim = numMelBins * strideVal
    let features = try! MLMultiArray(shape: [1, NSNumber(value: fixedSeqLen), NSNumber(value: featureDim)], dataType: .float32)
    let ptr = features.dataPointer.assumingMemoryBound(to: Float.self)
    memset(ptr, 0, fixedSeqLen * featureDim * MemoryLayout<Float>.size)

    let n = min(numPairs, fixedSeqLen)
    for i in 0..<n {
        let off = i * featureDim
        for j in 0..<numMelBins { ptr[off + j] = padFrames[i * strideVal][j] }
        for j in 0..<numMelBins { ptr[off + numMelBins + j] = padFrames[i * strideVal + 1][j] }
    }
    return features
}

// --- Run model ---
print("Computing features...")
let features = prepareFeatures(audioSamples: audio)
let featPtr = features.dataPointer.assumingMemoryBound(to: Float.self)
print("Features[0,:5] = \((0..<5).map { String(format: "%.4f", featPtr[$0]) })")

print("Loading CoreML chunks...")
let chunkDir = "\(base)/coreml_models_chunked"
let chunkNames = ["MuaalemChunkA_6BIT", "MuaalemChunkB_6BIT", "MuaalemChunkC_6BIT",
                   "MuaalemChunkD_6BIT", "MuaalemChunkE_6BIT", "MuaalemChunkF_6BIT"]

let config = MLModelConfiguration()
config.computeUnits = .cpuOnly

var x: MLMultiArray = features
for name in chunkNames {
    let url = URL(fileURLWithPath: "\(chunkDir)/\(name).mlpackage")
    let model = try! MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
    let inpName = model.modelDescription.inputDescriptionsByName.keys.first!
    let outName = model.modelDescription.outputDescriptionsByName.keys.first!
    let provider = try! MLDictionaryFeatureProvider(dictionary: [inpName: MLFeatureValue(multiArray: x)])
    let result = try! model.prediction(from: provider)
    x = result.featureValue(for: outName)!.multiArrayValue!

    // copyToFresh between chunks
    let shape = x.shape
    let fresh = try! MLMultiArray(shape: shape, dataType: .float32)
    let count = shape.reduce(1) { $0 * $1.intValue }
    memcpy(fresh.dataPointer, x.dataPointer, count * MemoryLayout<Float>.size)
    x = fresh

    print("  \(name): shape=\(x.shape), range=[\(String(format: "%.3f", (0..<count).map { x.dataPointer.assumingMemoryBound(to: Float.self)[$0] }.min()!)), \(String(format: "%.3f", (0..<count).map { x.dataPointer.assumingMemoryBound(to: Float.self)[$0] }.max()!))]")
}

// --- CTC Decode ---
let logits = x
let totalSteps = logits.shape[1].intValue
let numClasses = logits.shape[2].intValue
let logPtr = logits.dataPointer.assumingMemoryBound(to: Float.self)

let vocab: [Int: String] = [
    1: "ء", 2: "ب", 3: "ت", 4: "ث", 5: "ج", 6: "ح", 7: "خ", 8: "د", 9: "ذ",
    10: "ر", 11: "ز", 12: "س", 13: "ش", 14: "ص", 15: "ض", 16: "ط", 17: "ظ",
    18: "ع", 19: "غ", 20: "ف", 21: "ق", 22: "ك", 23: "ل", 24: "م", 25: "ن",
    26: "ه", 27: "و", 28: "ي", 29: "ا", 30: "ۦ", 31: "ۥ", 32: "َ", 33: "ُ",
    34: "ِ", 35: "۪", 36: "ـ", 37: "ٲ", 38: "ڇ", 39: "ں", 40: "۾", 41: "ۜ", 42: "ؙ"
]

var decoded = ""
var prev = -1
for t in 0..<totalSteps {
    let base = t * numClasses
    var maxVal: Float = -.infinity
    var maxIdx = 0
    for c in 0..<numClasses {
        if logPtr[base + c] > maxVal { maxVal = logPtr[base + c]; maxIdx = c }
    }
    if maxIdx != 0 && maxIdx != prev {
        decoded += vocab[maxIdx] ?? "[?]"
    }
    prev = maxIdx
}

print("\nDecoded: \(decoded)")
print("Expected: ءَعُۥۥذُبِللَااهِمِنَششَيطَاانِررَجِۦۦم")
