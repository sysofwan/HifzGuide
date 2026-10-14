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
// whether it is a silence flush; or, as the last check only, the app's stop and the phonemes it
// hands handleFinalFlush), the phoneme words to report, and the scorings to replay. Each scoring
// is a fresh session: a new engine on the ayah's page, every check, then the session's end
// (handleFinalFlush, empty unless a stop gave it phonemes, then settleReadersWord), as
// tools/replay_log.swift ends a log.
//
// What is driven and what is not: everything from handleUpdate on is Muraja's engine (query
// assembly, confirmation, placement, word scoring, GradeStore with its hold buffer and the
// end-word holdback). Nothing upstream of handleUpdate is: the transcriber's windowing, VAD and
// hallucination gates are the caller's, who supplies the checks. The display filters
// (GradeFilter+iOS.swift) are iOS-only and are applied by the caller to the kept qualities.
//
// The scorings (with the allowance overrides) and the kept grade's output are scoring.swift's,
// shared with log_replay.swift.

import Foundation

// MARK: - Requests and results

struct Check: Decodable {
  let hop: String
  let overlap: String
  let flush: Bool
  /// The app's stop: `hop` holds the phonemes RealtimeTranscriber.stop() returns, which the app
  /// hands to the engine's final alignment pass (LifecycleCoordinator+iOS.swift:340-347 ->
  /// FollowAlongEngine.handleFinalFlush, :899-914). Only ever a session's last check.
  let stop: Bool?

  var isStop: Bool { stop ?? false }
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

/// What the app needs to draw a word's errors (App/MistakeSnippetRenderer.swift:140-150): the
/// printed word the phoneme word is drawn on, its text, and per phoneme group the scalar indices
/// of that text the group covers, from Muraja's own QuranDatabase.phonemeGroupCharIndices
/// (Data/QuranDatabase.swift:676-735). The renderer's last step, scalar to grapheme
/// (MistakeSnippetRenderer.swift:350-363), is the caller's.
struct Letters: Encodable {
  let word: Int
  let textWord: Int
  let text: String
  let groupChars: [[Int]]

  init(_ position: QuranPosition, db: QuranDatabase) {
    word = position.word
    // The renderer takes the first text word mapped to the phoneme word from an unordered
    // dictionary; the lowest is taken here, so a phoneme word drawn on several printed words
    // is drawn on its first.
    let mapped = db.wordMap(surah: position.surah, ayah: position.ayah)
      .filter { $0.value == position.word }.keys
    textWord = mapped.min() ?? position.word
    text = db.wordText(surah: position.surah, ayah: position.ayah, word: textWord) ?? ""
    groupChars = db.phonemeGroupCharIndices(surah: position.surah, ayah: position.ayah, word: textWord)
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
  /// How the app draws each reported word's errors.
  let letters: [Letters]
}

func checkStatuses(_ result: RecitationResult?) -> [WordStatus] {
  switch result {
  case .correctAdvance(_, let statuses), .minorMistake(_, let statuses), .jumped(_, let statuses):
    return statuses
  case .lost, nil:
    return []
  }
}

func replay(_ request: Request, _ scoring: Scoring, db: QuranDatabase, index: PhonemeIndex) -> Result {
  let scoringMode = scoring.install()
  let parameters = ScoringParameters.forMode(scoringMode)
  let page = db.pageForPosition(surah: request.surah, ayah: request.ayah) ?? 1
  let start = QuranPosition(surah: request.surah, ayah: request.ayah, word: request.startWord)
  let engine = FollowAlongEngine(db: db, phonemeIndex: index, startPosition: start, scoringMode: scoringMode)
  // setPage moves the engine to the page's first position (FollowAlongEngine.swift:1067-1072);
  // the reader starts at the item's first word, so move there as the app's explicit navigation
  // does (FollowAlongManager+iOS.swift:451-452 -> FollowAlongEngine.moveTo, :951-957).
  engine.setPage(positions: db.positionsOnPage(page))
  engine.moveTo(start)
  engine.isHifzMode = true
  let reported = Set(request.reportWords.map {
    QuranPosition(surah: request.surah, ayah: request.ayah, word: $0)
  })

  var checks: [CheckResult] = []
  for check in request.checks {
    if check.isStop {
      engine.handleFinalFlush(phonemes: check.hop)
    } else {
      _ = engine.handleUpdate(confirmed: check.hop, unconfirmed: check.overlap, didFlush: check.flush)
    }
    let statuses = checkStatuses(engine.recitationStatus)
      .filter { reported.contains($0.position) }
      .map(Status.init)
    checks.append(CheckResult(position: engine.currentPosition.description, statuses: statuses))
  }
  // A session that did not end with the app's stop ends with an empty one, as a session whose
  // last audio went to a silence flush does (the transcriber's stop() has nothing pending).
  if !(request.checks.last?.isStop ?? false) {
    engine.handleFinalFlush(phonemes: "")
  }
  // The session's end settles the reader's word (FollowAlongManager+iOS.swift:881-882).
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
    checks: checks, final: final, letters: reported.sorted().map { Letters($0, db: db) })
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
emit(BuildRecord(murajaCommit: murajaCommit, compiler: compilerVersion, harness: "tools/muraja_harness/main.swift"))
while let line = readLine(strippingNewline: true) {
  guard !line.isEmpty else { continue }
  let request: Request
  do {
    request = try harnessDecoder.decode(Request.self, from: line.data(using: .utf8)!)
  } catch {
    FileHandle.standardError.write("bad request: \(error)\n".data(using: .utf8)!)
    exit(1)
  }
  let stops = request.checks.indices.filter { request.checks[$0].isStop }
  let wellFormed = stops.allSatisfy {
    $0 == request.checks.count - 1 && !request.checks[$0].flush && request.checks[$0].overlap.isEmpty
  }
  guard wellFormed else {
    FileHandle.standardError.write("bad request \(request.item): a stop is only the last check, with no overlap or flush\n".data(using: .utf8)!)
    exit(1)
  }
  for scoring in request.scorings {
    emit(replay(request, scoring, db: db, index: index))
  }
}
