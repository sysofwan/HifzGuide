// tools/muraja_harness/scoring.swift
//
// What both harness binaries share (build.sh compiles it into each): the scoring a replay runs
// under, and a kept grade as the binaries print it.
//
// MARK: - Scoring overrides
//
// Muraja picks a mode's parameters in one place, ScoringParameters.forMode (FollowAlongTypes.swift:
// 141-147), called from computeWordStatuses (QuranFollowAlong+WordScoring.swift:461). build.sh
// renames that one function to shippedForMode in the scratch copy; this replacement returns the
// shipped parameters with the two allowance flags overridden through Muraja's own `with`
// (FollowAlongTypes.swift:83-107). Scoring logic is untouched. nil keeps the mode's value.

import Foundation

nonisolated(unsafe) var softPairsOverride: Bool?
nonisolated(unsafe) var shaddahSuppressionOverride: Bool?

extension ScoringParameters {
  static func forMode(_ mode: ScoringMode) -> ScoringParameters {
    shippedForMode(mode).with(
      softPairsEnabled: softPairsOverride, shaddahSuppression: shaddahSuppressionOverride)
  }
}

/// One Muraja configuration to replay: a mode and the two allowance flags (nil keeps the mode's).
struct Scoring: Decodable {
  let name: String
  let mode: String
  let softPairsEnabled: Bool?
  let shaddahSuppression: Bool?

  /// The mode, with this scoring's flags installed for every engine built from now on.
  func install() -> ScoringMode {
    softPairsOverride = softPairsEnabled
    shaddahSuppressionOverride = shaddahSuppression
    switch mode {
    case "strict": return .strict
    case "balanced": return .balanced
    case "lenient": return .lenient
    default:
      FileHandle.standardError.write("unknown mode \(mode)\n".data(using: .utf8)!)
      exit(2)
    }
  }
}

/// A word's grade: its quality, score, withheld quality and every error the scorer recorded.
struct Status: Encodable {
  let word: Int
  let quality: String
  let score: Double
  let withheld: String?
  let errors: [WordError]

  init(_ status: WordStatus) {
    word = status.position.word
    quality = "\(status.quality)"
    score = status.score
    withheld = status.withheldQuality.map { "\($0)" }
    errors = status.errors
  }
}

/// The JSON every binary prints: snake_case keys, sorted, slashes unescaped.
let harnessEncoder: JSONEncoder = {
  let encoder = JSONEncoder()
  encoder.keyEncodingStrategy = .convertToSnakeCase
  encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
  return encoder
}()

let harnessDecoder: JSONDecoder = {
  let decoder = JSONDecoder()
  decoder.keyDecodingStrategy = .convertFromSnakeCase
  return decoder
}()

func emit<T: Encodable>(_ value: T) {
  let data = try! harnessEncoder.encode(value)
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

struct BuildRecord: Encodable {
  let murajaCommit: String
  let compiler: String
  let harness: String
}
