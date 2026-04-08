#!/usr/bin/env swift
// Standalone test: replicate MuaalemInference mel spectrogram and compare vs Python reference.

import Accelerate
import Foundation

// --- Parameters (must match MuaalemInference.swift) ---
let numMelBins = 80
let fftSize = 400
let fftLength = 512
let hopSize = 160
let stride = 2
let preemphasis: Float = 0.97
let melFloor: Float = 1.192092955078125e-07
let fixedSeqLen = 250

// --- Load binary files ---
func loadFloats(_ path: String) -> [Float] {
    let url = URL(fileURLWithPath: path)
    let data = try! Data(contentsOf: url)
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

let basePath = "/Users/sysofwan/repos/w2v-tools"
let melFilters = loadFloats("\(basePath)/mel_filters.bin")   // (80 * 257)
let analysisWindow = loadFloats("\(basePath)/window.bin")     // (400)
let audio = loadFloats("\(basePath)/test_noise_audio.bin")    // (80000) noise
let refFeatures = loadFloats("\(basePath)/test_noise_features.bin") // (249 * 160)

print("Loaded: mel_filters=\(melFilters.count), window=\(analysisWindow.count), audio=\(audio.count), ref=\(refFeatures.count)")
print("Audio[:10] = \(audio[0..<10].map { String(format: "%.4f", $0) })")

// --- Mel Spectrogram ---
func computeMelSpectrogram(audio: [Float]) -> [[Float]] {
    var scaled = audio.map { $0 * 32768.0 }

    let numFrames = max(0, (scaled.count - fftSize) / hopSize + 1)
    guard numFrames > 0 else { return [] }

    let halfFFT = fftLength / 2  // 256
    let numBins = halfFFT + 1    // 257

    let log2n = vDSP_Length(log2(Float(fftLength)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        return []
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    var melFrames = [[Float]]()
    melFrames.reserveCapacity(numFrames)

    for i in 0..<numFrames {
        let start = i * hopSize
        let end = min(start + fftSize, scaled.count)
        var frame = Array(scaled[start..<end])
        if frame.count < fftSize {
            frame.append(contentsOf: [Float](repeating: 0, count: fftSize - frame.count))
        }

        // Remove DC offset
        var mean: Float = 0
        vDSP_meanv(frame, 1, &mean, vDSP_Length(fftSize))
        var negMean = -mean
        vDSP_vsadd(frame, 1, &negMean, &frame, 1, vDSP_Length(fftSize))

        // Preemphasis
        var preemphasized = [Float](repeating: 0, count: fftSize)
        preemphasized[0] = frame[0]
        for j in 1..<fftSize {
            preemphasized[j] = frame[j] - preemphasis * frame[j - 1]
        }

        // Apply window
        vDSP_vmul(preemphasized, 1, analysisWindow, 1, &preemphasized, 1, vDSP_Length(fftSize))

        // Zero-pad to fftLength (512)
        var padded = [Float](repeating: 0, count: fftLength)
        for j in 0..<fftSize { padded[j] = preemphasized[j] }

        // FFT
        var realPart = [Float](repeating: 0, count: halfFFT)
        var imagPart = [Float](repeating: 0, count: halfFFT)
        padded.withUnsafeBufferPointer { paddedPtr in
            realPart.withUnsafeMutableBufferPointer { realPtr in
                imagPart.withUnsafeMutableBufferPointer { imagPtr in
                    var splitComplex = DSPSplitComplex(
                        realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!
                    )
                    paddedPtr.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self, capacity: halfFFT
                    ) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfFFT))
                    }
                    vDSP_fft_zrip(
                        fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
        }

        // Power spectrum (257 bins) — divide by 4 for vDSP 2x scaling
        var powerSpectrum = [Float](repeating: 0, count: numBins)
        powerSpectrum[0] = (realPart[0] * realPart[0]) / 4.0
        powerSpectrum[halfFFT] = (imagPart[0] * imagPart[0]) / 4.0
        for j in 1..<halfFFT {
            powerSpectrum[j] = (realPart[j] * realPart[j] + imagPart[j] * imagPart[j]) / 4.0
        }

        // Apply mel filterbank and log
        var melEnergies = [Float](repeating: 0, count: numMelBins)
        for m in 0..<numMelBins {
            var energy: Float = 0
            let filterOffset = m * numBins
            for j in 0..<numBins {
                energy += powerSpectrum[j] * melFilters[filterOffset + j]
            }
            melEnergies[m] = log(max(energy, melFloor))
        }

        melFrames.append(melEnergies)
    }

    return melFrames
}

// --- Feature Preparation ---
func prepareFeatures(audioSamples: [Float]) -> [Float] {
    let melFrames = computeMelSpectrogram(audio: audioSamples)
    let numFrames = melFrames.count
    guard numFrames > 0 else { return [Float](repeating: 0, count: fixedSeqLen * 160) }

    // Per-mel-bin normalization
    var means = [Float](repeating: 0, count: numMelBins)
    var variances = [Float](repeating: 0, count: numMelBins)
    for bin in 0..<numMelBins {
        var sum: Float = 0
        for t in 0..<numFrames { sum += melFrames[t][bin] }
        means[bin] = sum / Float(numFrames)
        var varSum: Float = 0
        for t in 0..<numFrames {
            let d = melFrames[t][bin] - means[bin]
            varSum += d * d
        }
        variances[bin] = numFrames > 1 ? varSum / Float(numFrames - 1) : 0
    }

    var normalized = [[Float]](repeating: [Float](repeating: 0, count: numMelBins), count: numFrames)
    for t in 0..<numFrames {
        for bin in 0..<numMelBins {
            normalized[t][bin] = (melFrames[t][bin] - means[bin]) / sqrt(variances[bin] + 1e-7)
        }
    }

    // Pad to even
    var padded = normalized
    if padded.count % stride != 0 {
        padded.append([Float](repeating: 1.0, count: numMelBins))
    }

    // Stride-2 concat
    let numPairs = padded.count / stride
    let featureDim = numMelBins * stride
    var features = [Float](repeating: 0, count: fixedSeqLen * featureDim)
    let pairsToWrite = min(numPairs, fixedSeqLen)
    for i in 0..<pairsToWrite {
        let offset = i * featureDim
        let frame1 = padded[i * stride]
        let frame2 = padded[i * stride + 1]
        for j in 0..<numMelBins {
            features[offset + j] = frame1[j]
            features[offset + numMelBins + j] = frame2[j]
        }
    }
    return features
}

// --- Run and compare ---
let melFrames = computeMelSpectrogram(audio: audio)
print("Mel frames: \(melFrames.count)")
print("Mel frame0[:5] = \(melFrames[0][0..<5].map { String(format: "%.4f", $0) })")
// Python: [8.2015, 8.8082, 8.2231, 7.1661, 9.3439]

let features = prepareFeatures(audioSamples: audio)
print("Features[:5] = \(features[0..<5].map { String(format: "%.6f", $0) })")
// Python: [-1.0973, -1.0864, -1.0730, -1.0920, 1.0586]
print("Features[80:85] = \(features[80..<85].map { String(format: "%.6f", $0) })")
// Python: [0.6319, 0.6584, 0.6749, 0.6435, -0.4949]

// Compare with reference
let compareLen = min(249 * 160, refFeatures.count)
var maxDiff: Float = 0
var maxDiffIdx = 0
var sumDiff: Float = 0
var diffAbove01 = 0
for i in 0..<compareLen {
    let d = abs(features[i] - refFeatures[i])
    if d > maxDiff { maxDiff = d; maxDiffIdx = i }
    if d > 0.1 { diffAbove01 += 1 }
    sumDiff += d
}
let meanDiff = sumDiff / Float(compareLen)
let maxRow = maxDiffIdx / 160
let maxCol = maxDiffIdx % 160
print("\nComparison vs Python reference:")
print("  Max diff: \(String(format: "%.6f", maxDiff)) at row=\(maxRow) col=\(maxCol)")
print("  Mean diff: \(String(format: "%.6f", meanDiff))")
print("  Values > 0.1 diff: \(diffAbove01) / \(compareLen)")
print("  At max: swift=\(String(format: "%.6f", features[maxDiffIdx])), python=\(String(format: "%.6f", refFeatures[maxDiffIdx]))")

// Per-row analysis
for row in [0, 1, 62, 124, 186, 248] {
    var rowMax: Float = 0
    var rowMaxCol = 0
    for c in 0..<160 {
        let idx = row * 160 + c
        if idx < compareLen {
            let d = abs(features[idx] - refFeatures[idx])
            if d > rowMax { rowMax = d; rowMaxCol = c }
        }
    }
    print("  Row \(row): max diff = \(String(format: "%.6f", rowMax)) at col \(rowMaxCol)")
}

// Check if it's concentrated in certain columns (mel bins)
print("\nPer-column (feature dim) max diff:")
for col in Swift.stride(from: 0, to: 160, by: 10) {
    var colMax: Float = 0
    for row in 0..<249 {
        let idx = row * 160 + col
        if idx < compareLen {
            let d = abs(features[idx] - refFeatures[idx])
            colMax = max(colMax, d)
        }
    }
    print("  Col \(col): max diff = \(String(format: "%.6f", colMax))")
}
