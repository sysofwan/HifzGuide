// tools/muraja_harness/log_replay.swift
//
// Replays Muraja's device-session logs (its tools/test_logs/*.log) under each scoring and prints
// every grade the session kept. HifzGuide's code; build.sh compiles it with the FollowAlong and
// analysis sources of a pinned Muraja checkout and with that checkout's own tools/replay_log.swift
// (and the reports it calls), as tools/run_log_tests.sh does. The one edit to the scratch copy is
// replay_log's @main, removed so that this is the entry point.
//
// Usage: muraja_log_replay <quran.db>   (QURAN_LAYOUT_DB_PATH names the layout.db)
//   stdin:  one JSON request per line: {"item", "log" (a path), "scorings"}
//   stdout: one JSON line describing the build, then one JSON line per request and scoring:
//           {"item", "scoring", "final_position", "words": [{"position", "status"}], "grade_lines"}
//
// Each replay is replay_log's own executeReplay, the replay run_log_tests.sh compares with each
// log's baseline.json: the log's Start, Update, Withheld update, Page and Stop events in order
// through one FollowAlongEngine, the reader's word settled at each page change and at the
// session's end, every non-pending grade of every page visited kept. Its scoring mode is
// replay_log's (REPLAY_MODE, Balanced unless set); a scoring here is that mode with the two
// allowance flags installed (scoring.swift), run with REPLAY_MODE set to it.
//
// The engine's own grade decisions: the word scorer logs one line per end-word grade
// ("Grade end word …", QuranFollowAlong+WordScoring.swift:750), the line a device log keeps. build.sh
// points that one logger (scoringLogger) at GradeLineLogger in the scratch copy, which keeps the
// line as the device's unified log formats it; each result carries the replay's lines in order.

import Foundation
import OSLog

/// The word scorer's logged lines during the current replay.
nonisolated(unsafe) var loggedGradeLines: [String] = []

/// Stands in for the word scorer's os.Logger: same initializer, same call, the line kept here.
struct GradeLineLogger {
  init(subsystem: String, category: String) {}

  func info(_ message: LoggedLine) {
    loggedGradeLines.append(message.text)
  }
}

/// A log message as the unified log renders it: literals, and each value's description.
struct LoggedLine: ExpressibleByStringInterpolation {
  let text: String

  init(stringLiteral value: String) {
    text = value
  }

  init(stringInterpolation: StringInterpolation) {
    text = stringInterpolation.text
  }

  struct StringInterpolation: StringInterpolationProtocol {
    var text = ""

    init(literalCapacity: Int, interpolationCount: Int) {}

    mutating func appendLiteral(_ literal: String) {
      text += literal
    }

    mutating func appendInterpolation<T>(_ value: T, privacy: OSLogPrivacy) {
      text += "\(value)"
    }
  }
}

struct LogRequest: Decodable {
  let item: String
  let log: String
  let scorings: [Scoring]
}

struct KeptWord: Encodable {
  let position: String
  let status: Status
}

struct LogResult: Encodable {
  let item: String
  let scoring: String
  let finalPosition: String
  /// Every word the session kept a grade for, in mushaf order.
  let words: [KeptWord]
  /// Every line the word scorer logged, in order.
  let gradeLines: [String]
}

@main
struct MurajaLogReplay {
  static func main() {
    guard CommandLine.arguments.count == 2 else {
      FileHandle.standardError.write("usage: muraja_log_replay <quran.db>\n".data(using: .utf8)!)
      exit(2)
    }
    let dbPath = CommandLine.arguments[1]
    emit(BuildRecord(
      murajaCommit: murajaCommit, compiler: compilerVersion,
      harness: "tools/muraja_harness/log_replay.swift"))
    while let line = readLine(strippingNewline: true) {
      guard !line.isEmpty else { continue }
      let request: LogRequest
      do {
        request = try harnessDecoder.decode(LogRequest.self, from: line.data(using: .utf8)!)
      } catch {
        FileHandle.standardError.write("bad request: \(error)\n".data(using: .utf8)!)
        exit(1)
      }
      for scoring in request.scorings {
        guard scoring.install() == replayScoringMode else {
          FileHandle.standardError.write(
            "scoring \(scoring.name) is \(scoring.mode); replay_log runs \(replayScoringMode.label)\n"
              .data(using: .utf8)!)
          exit(2)
        }
        loggedGradeLines = []
        let result = executeReplay(dbPath: dbPath, logPath: request.log, verbose: false, quiet: true)
        emit(LogResult(
          item: request.item, scoring: scoring.name,
          finalPosition: result.finalPosition.description,
          words: result.wordStatuses.map { KeptWord(position: $0.0.description, status: Status($0.1)) },
          gradeLines: loggedGradeLines))
      }
    }
  }
}
