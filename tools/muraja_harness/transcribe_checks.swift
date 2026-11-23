// tools/muraja_harness/transcribe_checks.swift
//
// The app arm's check writer (#307): WAV → the full hops and overlaps Muraja's own offline
// transcriber hands the engine, as JSON check requests.
//
// Muraja's offline transcriber (`tools/transcribe_clip.swift`) writes the device-format `.log`,
// which cuts every overlap to 30 Characters (`TranscribeLog.swift:43-46`). The device engine
// sees the full overlap, so the follow-along audit cannot replay that log and call it the app's
// session. This driver runs the same transcriber — `TranscribeScheduler.swift` over the app's
// `MuaalemInference` — and emits each update's full `hop`/`overlap` instead of rendering it.
//
// Every Muraja source is compiled unchanged (see `build.sh`): the driver only drives the
// `MuaalemPredicting` seam, as `transcribe_clip.swift` does.
//
// Usage:
//   transcribe_checks [--fake-ids <i,j,...>] [--fake-fail] <16 kHz mono wav>
//
// stdout: one JSON object, keys sorted so two runs are byte-identical:
//   {"checks": [{"hop","overlap","flush"}... , {"hop","overlap","flush","stop"}],
//    "inference_latency_ms": <number>, "audio_seconds": ..., "inference_count": ...,
//    "failure_count": ...}
// The last check is the session's stop and carries `finalPhonemes` as its hop, which is how
// the engine appends it (`FollowAlongEngine.swift:907-909`). A clip Muraja could not transcribe
// exits non-zero and says why on stderr, writing no checks: `unpublishableReason` is the one
// owner of that policy (`TranscribeOutput.swift`).
//
// `--fake-ids`/`--fake-fail` substitute a model for the CoreML one, so the build and the
// cadence can be exercised on a Mac with no assets staged.
//
// Env:
//   TRANSCRIBE_INFERENCE_LATENCY_MS  per-inference wall time, milliseconds (default 0). Read
//                                     here, not by the scheduler, which defaults to 0 — the
//                                     audit runs 40, the h448-era device medians (37–42 ms).

import Foundation

/// One check request, in `tools/muraja_harness/main.swift`'s format.
private struct Check: Encodable {
  let hop: String
  let overlap: String
  let flush: Bool
  /// Present only on the session's last check.
  let stop: Bool?
}

private struct Payload: Encodable {
  let checks: [Check]
  let inferenceLatencyMs: Double
  let audioSeconds: Double
  let inferenceCount: Int
  let failureCount: Int

  enum CodingKeys: String, CodingKey {
    case checks
    case inferenceLatencyMs = "inference_latency_ms"
    case audioSeconds = "audio_seconds"
    case inferenceCount = "inference_count"
    case failureCount = "failure_count"
  }
}

/// Every inference returns `ids`, split evenly between hop and overlap: a model that saw real
/// audio would decode differently on each side, but the audit's tests pin the schedule and the
/// text's fullness, which a fixed decode shows without CoreML assets.
private struct FakeMuaalem: MuaalemPredicting {
  let ids: [Int]
  let fail: Bool

  var windowSamples: Int { 16_000 * 5 }
  var outputTimeSteps: Int { 125 }

  private var stats: CTCDecoder.Stats {
    CTCDecoder.Stats(
      blankRatio: 0, dominantTokenRatio: 1, dominantTokenID: ids.first ?? 0,
      nonBlankCount: 125, meanLogitMargin: 1)
  }

  func predict(audioSamples: [Float]) throws -> (ids: [Int], stats: CTCDecoder.Stats) {
    if fail { throw MuaalemError.predictionFailed }
    return (ids, stats)
  }

  func predictSplit(audioSamples: [Float], confirmTimeSteps: Int) throws
    -> (confirmed: [Int], unconfirmed: [Int], stats: CTCDecoder.Stats)
  {
    if fail { throw MuaalemError.predictionFailed }
    return (ids, ids, stats)
  }
}

@main
struct TranscribeChecks {
  static func main() {
    do {
      try run()
    } catch Failure.usage {
      fail("usage: transcribe_checks [--fake-ids <i,j,...>] [--fake-fail] <16 kHz mono wav>")
    } catch {
      fail("\(error)")
    }
  }

  private static func run() throws {
    var fakeIDs: [Int]?
    var fakeFail = false
    var wavPath: String?
    var arguments = CommandLine.arguments.dropFirst().makeIterator()
    while let argument = arguments.next() {
      switch argument {
      case "--fake-ids":
        guard let spec = arguments.next() else { throw Failure.usage }
        let ids = spec.split(separator: ",").map(String.init).compactMap(Int.init)
        guard !ids.isEmpty, ids.count == spec.split(separator: ",").count else {
          throw Failure.message("--fake-ids needs a comma-separated list of integer ids")
        }
        fakeIDs = ids
      case "--fake-fail":
        fakeFail = true
      default:
        guard wavPath == nil else { throw Failure.usage }
        wavPath = argument
      }
    }
    guard let wavPath else { throw Failure.usage }

    let (samples, sampleRate) = try readWAV(path: wavPath)
    guard sampleRate == 16_000 else {
      throw Failure.message("\(wavPath) is \(sampleRate) Hz, not 16 kHz")
    }

    let model: MuaalemPredicting
    if fakeIDs != nil || fakeFail {
      model = FakeMuaalem(ids: fakeIDs ?? [2, 32, 3, 34], fail: fakeFail)
    } else {
      model = try TranscribeAssets.makeInference()
    }

    let latencyMs = simulatedLatencyMs()
    let result = transcribeOffline(
      samples: samples, muaalem: model, inferenceLatency: latencyMs / 1000)
    if let reason = unpublishableReason(result) {
      throw Failure.message("\(wavPath): \(describe(reason)) — no checks written")
    }

    var checks = result.updates.map {
      Check(hop: $0.0.hop, overlap: $0.0.overlap, flush: $0.0.didFlush, stop: nil)
    }
    checks.append(
      Check(hop: result.finalPhonemes, overlap: "", flush: false, stop: true))
    let payload = Payload(
      checks: checks, inferenceLatencyMs: latencyMs,
      audioSeconds: result.stats.audioSeconds,
      inferenceCount: result.stats.inferenceCount, failureCount: result.stats.failureCount)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    print(String(decoding: try encoder.encode(payload), as: UTF8.self))
  }

  /// `TRANSCRIBE_INFERENCE_LATENCY_MS`, as `transcribe_clip.swift` reads it.
  private static func simulatedLatencyMs() -> Double {
    guard let raw = ProcessInfo.processInfo.environment["TRANSCRIBE_INFERENCE_LATENCY_MS"],
      let ms = Double(raw), ms >= 0
    else { return 0 }
    return ms
  }

  private static func describe(_ reason: Unpublishable) -> String {
    switch reason {
    case .abortedAfterConsecutiveFailures:
      return "aborted after consecutive inference failures"
    case .inferenceFailures(let count):
      return "\(count) inference failure(s)"
    case .noInference:
      return "no inference ran"
    }
  }

  private static func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("transcribe_checks: " + message + "\n").utf8))
    Foundation.exit(1)
  }
}

private enum Failure: Error, CustomStringConvertible {
  case usage
  case message(String)

  var description: String {
    switch self {
    case .usage: return "usage"
    case .message(let message): return message
    }
  }
}
