// tools/muraja_harness/main.swift
//
// Replays recorded checks through Muraja's own FollowAlongEngine and prints the grades it keeps.
// HifzGuide's code; compiled by build.sh together with the FollowAlong sources of a pinned Muraja
// checkout (never committed here), with -DTEST_HARNESS, as Muraja's tools/replay_log.swift is.
//
// Usage: muraja_harness <quran.db>   (QURAN_LAYOUT_DB_PATH names the layout.db)
//   stdin:  one JSON request per line (below)
//   stdout: one JSON line describing the build, then one JSON line per request and scoring
//
// A request is one item: a surah and ayah, the phoneme word the reciter starts on, the checks in
// order (each as FollowAlongEngine.handleUpdate receives it: the hop, the overlap or preview, and
// whether it is a silence flush), the phoneme words to report, and the scorings to replay. Each
// scoring is a fresh session: a new engine on the ayah's page, every check, then the session's
// end (handleFinalFlush, settleReadersWord), as tools/replay_log.swift ends a log.
//
// What is driven and what is not: everything from handleUpdate on is Muraja's engine (query
// assembly, confirmation, placement, word scoring, GradeStore with its hold buffer and the
// end-word holdback). Nothing upstream of handleUpdate is: the transcriber's windowing, VAD and
// hallucination gates are the caller's, who supplies the checks. The display filters
// (GradeFilter+iOS.swift) are iOS-only and are applied by the caller to the kept qualities.

import Foundation

// MARK: - Scoring overrides
//
// Muraja picks a mode's parameters in one place, ScoringParameters.forMode (FollowAlongTypes.swift:
// 141-147), called from computeWordStatuses (QuranFollowAlong+WordScoring.swift:461). build.sh
// renames that one function to shippedForMode in the scratch copy; this replacement returns the
// shipped parameters with the two allowance flags overridden through Muraja's own `with`
// (FollowAlongTypes.swift:83-107). Scoring logic is untouched. nil keeps the mode's value.

nonisolated(unsafe) var softPairsOverride: Bool?
nonisolated(unsafe) var shaddahSuppressionOverride: Bool?

extension ScoringParameters {
  static func forMode(_ mode: ScoringMode) -> ScoringParameters {
    shippedForMode(mode).with(
      softPairsEnabled: softPairsOverride, shaddahSuppression: shaddahSuppressionOverride)
  }
}

// MARK: - Requests and results

struct Check: Decodable {
  let hop: String
  let overlap: String
  let flush: Bool
}

struct Scoring: Decodable {
  let name: String
  let mode: String
  let softPairsEnabled: Bool?
  let shaddahSuppression: Bool?
}

struct Request: Decodable {
  let item: String
  let surah: Int
  let ayah: Int
  let startWord: Int
  let reportWords: [Int]
  let checks: [Check]
  let scorings: [Scoring]
}

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

struct CheckResult: Encodable {
  /// The word the engine's alignment placed the reader on after this check.
  let position: String
  /// This check's own grades of the reported words (computeWordStatuses), before the ratchet.
  let statuses: [Status]
}

struct Result: Encodable {
  let item: String
  let scoring: String
  let page: Int
  let parameters: [String: Bool]
  let checks: [CheckResult]
  /// The grade GradeStore keeps for each reported word at the session's end.
  let final: [Status]
}

struct BuildRecord: Encodable {
  let murajaCommit: String
  let compiler: String
  let harness = "tools/muraja_harness/main.swift"
}

func checkStatuses(_ result: RecitationResult?) -> [WordStatus] {
  switch result {
  case .correctAdvance(_, let statuses), .minorMistake(_, let statuses), .jumped(_, let statuses):
    return statuses
  case .lost, nil:
    return []
  }
}

func mode(_ name: String) -> ScoringMode {
  switch name {
  case "strict": return .strict
  case "balanced": return .balanced
  case "lenient": return .lenient
  default:
    FileHandle.standardError.write("unknown mode \(name)\n".data(using: .utf8)!)
    exit(2)
  }
}

func replay(_ request: Request, _ scoring: Scoring, db: QuranDatabase, index: PhonemeIndex) -> Result {
  softPairsOverride = scoring.softPairsEnabled
  shaddahSuppressionOverride = scoring.shaddahSuppression
  let scoringMode = mode(scoring.mode)
  let parameters = ScoringParameters.forMode(scoringMode)
  let page = db.pageForPosition(surah: request.surah, ayah: request.ayah) ?? 1
  let engine = FollowAlongEngine(
    db: db, phonemeIndex: index,
    startPosition: QuranPosition(surah: request.surah, ayah: request.ayah, word: request.startWord),
    scoringMode: scoringMode)
  engine.setPage(positions: db.positionsOnPage(page))
  engine.isHifzMode = true
  let reported = Set(request.reportWords.map {
    QuranPosition(surah: request.surah, ayah: request.ayah, word: $0)
  })

  var checks: [CheckResult] = []
  for check in request.checks {
    _ = engine.handleUpdate(confirmed: check.hop, unconfirmed: check.overlap, didFlush: check.flush)
    let statuses = checkStatuses(engine.recitationStatus)
      .filter { reported.contains($0.position) }
      .map(Status.init)
    checks.append(CheckResult(position: engine.currentPosition.description, statuses: statuses))
  }
  engine.handleFinalFlush(phonemes: "")
  engine.settleReadersWord()
  let final = reported.sorted().compactMap { engine.wordStatuses[$0] }.map(Status.init)
  return Result(
    item: request.item, scoring: scoring.name, page: page,
    parameters: [
      "softPairsEnabled": parameters.softPairsEnabled,
      "shaddahSuppression": parameters.shaddahSuppression,
      "phonemeGateEnabled": parameters.phonemeGateEnabled,
      "suppressHarakaDrop": parameters.suppressHarakaDrop,
    ],
    checks: checks, final: final)
}

// MARK: - Main

guard CommandLine.arguments.count == 2 else {
  FileHandle.standardError.write("usage: muraja_harness <quran.db>\n".data(using: .utf8)!)
  exit(2)
}
let db: QuranDatabase
do {
  db = try QuranDatabase(paths: .harness(core: CommandLine.arguments[1]))
} catch {
  FileHandle.standardError.write("cannot open \(CommandLine.arguments[1]): \(error)\n".data(using: .utf8)!)
  exit(1)
}
let index = PhonemeIndex(db: db)
let decoder = JSONDecoder()
decoder.keyDecodingStrategy = .convertFromSnakeCase
let encoder = JSONEncoder()
encoder.keyEncodingStrategy = .convertToSnakeCase
encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

func emit<T: Encodable>(_ value: T) {
  let data = try! encoder.encode(value)
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

emit(BuildRecord(murajaCommit: murajaCommit, compiler: compilerVersion))
while let line = readLine(strippingNewline: true) {
  guard !line.isEmpty else { continue }
  let request: Request
  do {
    request = try decoder.decode(Request.self, from: line.data(using: .utf8)!)
  } catch {
    FileHandle.standardError.write("bad request: \(error)\n".data(using: .utf8)!)
    exit(1)
  }
  for scoring in request.scorings {
    emit(replay(request, scoring, db: db, index: index))
  }
}
