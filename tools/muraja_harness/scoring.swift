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
//
// One soft pair at a time: Muraja forgives its six soft pairs in one place, isSoftMismatch
// (PhonemeSifat.swift:219-225), called only from the word scorer (QuranFollowAlong+WordScoring.swift:
// 613). build.sh renames it to shippedIsSoftMismatch; this replacement returns Muraja's own answer
// except for the pairs a scoring disables, each of which must be one of Muraja's pairs.

import Foundation

nonisolated(unsafe) var softPairsOverride: Bool?
nonisolated(unsafe) var shaddahSuppressionOverride: Bool?
/// The soft pairs not forgiven, each as "x↔y" in either order.
nonisolated(unsafe) var disabledSoftPairs: Set<String> = []

extension ScoringParameters {
  static func forMode(_ mode: ScoringMode) -> ScoringParameters {
    shippedForMode(mode).with(
      softPairsEnabled: softPairsOverride, shaddahSuppression: shaddahSuppressionOverride)
  }
}

func isSoftMismatch(_ a: Unicode.Scalar, _ b: Unicode.Scalar, softPairsEnabled: Bool) -> Bool {
  shippedIsSoftMismatch(a, b, softPairsEnabled: softPairsEnabled)
    && !disabledSoftPairs.contains("\(a)↔\(b)") && !disabledSoftPairs.contains("\(b)↔\(a)")
}

/// The request protocol both binaries speak. The client (training/muraja_policy.py) refuses a
/// binary that reports another, or none, and any scoring key the binary does not declare.
let harnessProtocol = 3

/// One Muraja configuration to replay: a mode and the two allowance flags (nil keeps the mode's).
struct Scoring: Decodable {
  /// Every key a scoring may carry; the build record declares them.
  enum CodingKeys: String, CodingKey, CaseIterable {
    case name, mode, softPairsEnabled, shaddahSuppression, softPairsDisabled
  }

  let name: String
  let mode: String
  let softPairsEnabled: Bool?
  let shaddahSuppression: Bool?
  /// Soft pairs ("x↔y") this scoring does not forgive; the others keep Muraja's rule.
  let softPairsDisabled: [String]?

  /// The mode, with this scoring's flags installed for every engine built from now on.
  func install() -> ScoringMode {
    softPairsOverride = softPairsEnabled
    shaddahSuppressionOverride = shaddahSuppression
    disabledSoftPairs = Set(softPairsDisabled ?? [])
    for pair in disabledSoftPairs {
      let scalars = Array(pair.unicodeScalars)
      guard scalars.count == 3, scalars[1] == "↔",
            shippedIsSoftMismatch(scalars[0], scalars[2], softPairsEnabled: true) else {
        FileHandle.standardError.write("\(pair) is not one of Muraja's soft pairs\n".data(using: .utf8)!)
        exit(2)
      }
    }
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
  let protocolVersion = harnessProtocol
  let scoringKeys = Scoring.CodingKeys.allCases.map(\.stringValue)
}
