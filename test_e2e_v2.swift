import Accelerate
import CoreML
import Foundation

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
let base = "/Users/sysofwan/repos/w2v-tools"
let melFilters = loadFloats("\(base)/mel_filters.bin")
let analysisWindow = loadFloats("\(base)/window.bin")
let audio = loadFloats("\(base)/test_quran_audio.bin")
print("Audio: \(audio.count) samples")

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
            rp.withUnsafeMutableBufferPointer { rr in
                ip.withUnsafeMutableBufferPointer { ii in
                    var sc = DSPSplitComplex(realp: rr.baseAddress!, imagp: ii.baseAddress!)
                    pp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) {
                        vDSP_ctoz($0, 2, &sc, 1, vDSP_Length(halfFFT))
                    }
                    vDSP_fft_zrip(fftSetup, &sc, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
        }
        var ps = [Float](repeating: 0, count: numBins)
        ps[0] = (rp[0]*rp[0])/4.0
        ps[halfFFT] = (ip[0]*ip[0])/4.0
        for j in 1..<halfFFT { ps[j] = (rp[j]*rp[j]+ip[j]*ip[j])/4.0 }
        var me = [Float](repeating: 0, count: numMelBins)
        for m in 0..<numMelBins {
            var e: Float = 0
            let off = m * numBins
            for j in 0..<numBins { e += ps[j] * melFilters[off+j] }
            me[m] = log(max(e, melFloor))
        }
        melFrames.append(me)
    }
    return melFrames
}

func prepareFeatures(audio: [Float]) -> MLMultiArray {
    let mf = computeMelSpectrogram(audio: audio)
    let nf = mf.count
    guard nf > 0 else { return try! MLMultiArray(shape: [1, NSNumber(value: fixedSeqLen), 160], dataType: .float32) }
    var means = [Float](repeating: 0, count: numMelBins)
    var vars = [Float](repeating: 0, count: numMelBins)
    for b in 0..<numMelBins {
        var s: Float = 0
        for t in 0..<nf { s += mf[t][b] }
        means[b] = s/Float(nf)
        var v: Float = 0
        for t in 0..<nf { let d = mf[t][b]-means[b]; v += d*d }
        vars[b] = nf > 1 ? v/Float(nf-1) : 0
    }
    var norm = [[Float]](repeating: [Float](repeating: 0, count: numMelBins), count: nf)
    for t in 0..<nf { for b in 0..<numMelBins { norm[t][b] = (mf[t][b]-means[b])/sqrt(vars[b]+1e-7) } }
    if norm.count % strideVal != 0 { norm.append([Float](repeating: 1.0, count: numMelBins)) }
    let np = norm.count/strideVal
    let fd = numMelBins*strideVal
    let feat = try! MLMultiArray(shape: [1, NSNumber(value: fixedSeqLen), NSNumber(value: fd)], dataType: .float32)
    let p = feat.dataPointer.assumingMemoryBound(to: Float.self)
    memset(p, 0, fixedSeqLen*fd*MemoryLayout<Float>.size)
    for i in 0..<min(np, fixedSeqLen) {
        let o = i*fd
        for j in 0..<numMelBins { p[o+j] = norm[i*strideVal][j]; p[o+numMelBins+j] = norm[i*strideVal+1][j] }
    }
    return feat
}

func copyToFresh(_ source: MLMultiArray) -> MLMultiArray {
    let shape = source.shape
    let fresh = try! MLMultiArray(shape: shape, dataType: .float32)
    let count = shape.reduce(1) { $0 * $1.intValue }
    let dst = fresh.dataPointer.assumingMemoryBound(to: Float.self)
    if source.dataType == .float16 {
        for i in 0..<count { dst[i] = source[i].floatValue }
    } else {
        memcpy(dst, source.dataPointer, count * MemoryLayout<Float>.size)
    }
    return fresh
}

let features = prepareFeatures(audio: audio)
print("Loading chunks...")
let config = MLModelConfiguration()
config.computeUnits = .cpuOnly
let names = ["MuaalemChunkA_6BIT","MuaalemChunkB_6BIT","MuaalemChunkC_6BIT",
             "MuaalemChunkD_6BIT","MuaalemChunkE_6BIT","MuaalemChunkF_6BIT"]

var x: MLMultiArray = features
for name in names {
    let url = URL(fileURLWithPath: "\(base)/coreml_models_chunked/\(name).mlpackage")
    let compiled = try! MLModel.compileModel(at: url)
    let model = try! MLModel(contentsOf: compiled, configuration: config)
    let inN = model.modelDescription.inputDescriptionsByName.keys.first!
    let outN = model.modelDescription.outputDescriptionsByName.keys.first!
    let prov = try! MLDictionaryFeatureProvider(dictionary: [inN: MLFeatureValue(multiArray: x)])
    let res = try! model.prediction(from: prov)
    let raw = res.featureValue(for: outN)!.multiArrayValue!
    print("  \(name): output dtype=\(raw.dataType == .float16 ? "FP16" : "FP32")")
    x = copyToFresh(raw)
    let cnt = x.shape.reduce(1) { $0 * $1.intValue }
    let ptr = x.dataPointer.assumingMemoryBound(to: Float.self)
    var mn: Float = .infinity, mx: Float = -.infinity
    for i in 0..<cnt { mn = min(mn, ptr[i]); mx = max(mx, ptr[i]) }
    print("    after copy: range=[\(String(format:"%.3f",mn)), \(String(format:"%.3f",mx))]")
}

let logits = x
let steps = logits.shape[1].intValue
let classes = logits.shape[2].intValue
let lp = logits.dataPointer.assumingMemoryBound(to: Float.self)
let vocab: [Int:String] = [1:"ء",2:"ب",3:"ت",4:"ث",5:"ج",6:"ح",7:"خ",8:"د",9:"ذ",10:"ر",11:"ز",12:"س",13:"ش",14:"ص",15:"ض",16:"ط",17:"ظ",18:"ع",19:"غ",20:"ف",21:"ق",22:"ك",23:"ل",24:"م",25:"ن",26:"ه",27:"و",28:"ي",29:"ا",30:"ۦ",31:"ۥ",32:"َ",33:"ُ",34:"ِ",35:"۪",36:"ـ",37:"ٲ",38:"ڇ",39:"ں",40:"۾",41:"ۜ",42:"ؙ"]
var decoded = ""
var prev = -1
for t in 0..<steps {
    var mxv: Float = -.infinity
    var mi = 0
    for c in 0..<classes { let v = lp[t*classes+c]; if v > mxv { mxv = v; mi = c } }
    if mi != 0 && mi != prev { decoded += vocab[mi] ?? "?" }
    prev = mi
}
print("\nDecoded:  \(decoded)")
print("Expected: ءَعُۥۥذُبِللَااهِمِنَششَيطَاانِررَجِۦۦم")
