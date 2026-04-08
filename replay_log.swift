// tools/replay_log.swift
//
// Log replay harness for follow-along debugging.
// Parses Console.app logs and replays inference events through QuranFollowAlong,
// simulating FollowAlongManager buffering and ratchet logic.
//
// Usage:
//   replay_log <path-to-quran.db> <logfile.txt> [--verbose]

import Foundation

// MARK: - Log Event Types

enum InferenceType: String {
  case preview = "Preview"
  case window = "Window"
}

struct InferenceEvent {
  let type: InferenceType
  let durationMs: Int
  let phonemes: String        // for preview: the full phoneme text
  let confirmed: String?      // for window: confirmed portion
  let unconfirmed: String?    // for window: unconfirmed portion
}

struct PositionEvent {
  let position: QuranPosition
}

enum LogEvent {
  case inference(InferenceEvent)
  case position(PositionEvent)
  case matchLine(String)   // raw match line for comparison
  case gradeLine(String)   // raw grade line for comparison
}

// MARK: - Log Parser

func parseLogEvents(from text: String) -> [LogEvent] {
  var events: [LogEvent] = []

  for line in text.components(separatedBy: .newlines) {
    var trimmed = line.trimmingCharacters(in: .whitespaces)
    guard !trimmed.isEmpty else { continue }

    // Strip Console.app prefix: [HH:MM:SS.mmm] [INFO] [Category] ...
    if trimmed.hasPrefix("[") {
      // Find the content after the last "] " bracket group
      var remaining = trimmed[trimmed.startIndex...]
      while remaining.hasPrefix("[") {
        if let closeIdx = remaining.firstIndex(of: "]") {
          let afterClose = remaining.index(after: closeIdx)
          if afterClose < remaining.endIndex && remaining[afterClose] == " " {
            remaining = remaining[remaining.index(after: afterClose)...]
          } else {
            remaining = remaining[afterClose...]
          }
        } else {
          break
        }
      }
      trimmed = String(remaining).trimmingCharacters(in: .whitespaces)
    }

    if trimmed.hasPrefix("Window inference"), let event = parseWindowInference(trimmed) {
      events.append(.inference(event))
    } else if trimmed.hasPrefix("Preview inference"), let event = parsePreviewInference(trimmed) {
      events.append(.inference(event))
    } else if let pos = parsePositionLine(trimmed) {
      events.append(.position(PositionEvent(position: pos)))
    } else if trimmed.hasPrefix("Match: ratio=") {
      events.append(.matchLine(trimmed))
    } else if trimmed.hasPrefix("Grade end word") {
      events.append(.gradeLine(trimmed))
    }
  }

  return events
}

/// Parse "Preview inference 337ms — كَااااف"
private func parsePreviewInference(_ line: String) -> InferenceEvent? {
  guard let infIdx = line.range(of: "inference ", options: .literal),
        let msIdx = line.range(of: "ms", options: .literal) else { return nil }
  let durStr = String(line[infIdx.upperBound..<msIdx.lowerBound])
  guard let duration = Int(durStr) else { return nil }

  guard let dashIdx = line.range(of: "\u{2014} ", options: .literal) else { return nil }
  let phonemes = String(line[dashIdx.upperBound...]).trimmingCharacters(in: .whitespaces)
  guard !phonemes.isEmpty else { return nil }

  return InferenceEvent(type: .preview, durationMs: duration, phonemes: phonemes,
                        confirmed: nil, unconfirmed: nil)
}

/// Parse "Window inference 302ms — confirmed: كَاااافهَاايَ, unconfirmed: َااعِ"
private func parseWindowInference(_ line: String) -> InferenceEvent? {
  guard let infIdx = line.range(of: "inference ", options: .literal),
        let msIdx = line.range(of: "ms", options: .literal) else { return nil }
  let durStr = String(line[infIdx.upperBound..<msIdx.lowerBound])
  guard let duration = Int(durStr) else { return nil }

  guard let confIdx = line.range(of: "confirmed: ", options: .literal),
        let unconfIdx = line.range(of: ", unconfirmed: ", options: .literal) else { return nil }

  let confirmed = String(line[confIdx.upperBound..<unconfIdx.lowerBound])
  let unconfirmed = String(line[unconfIdx.upperBound...]).trimmingCharacters(in: .whitespaces)

  return InferenceEvent(type: .window, durationMs: duration,
                        phonemes: confirmed + unconfirmed,
                        confirmed: confirmed, unconfirmed: unconfirmed)
}

/// Parse "Position → 19:1:1" or "Position → 19:1:1 (re-read)"
private func parsePositionLine(_ line: String) -> QuranPosition? {
  guard let arrowIdx = line.range(of: "\u{2192} ", options: .literal) else { return nil }
  let posStr = String(line[arrowIdx.upperBound...])
    .trimmingCharacters(in: .whitespaces)
    .components(separatedBy: " ").first ?? ""
  let parts = posStr.split(separator: ":").compactMap { Int($0) }
  guard parts.count == 3 else { return nil }
  return QuranPosition(surah: parts[0], ayah: parts[1], word: parts[2])
}

/// Extract the start position from the first expected= field in the log.
func extractStartPosition(from text: String) -> QuranPosition? {
  guard let expIdx = text.range(of: "expected=", options: .literal) else { return nil }
  let after = String(text[expIdx.upperBound...])
  let posStr = after.prefix(while: { $0.isNumber || $0 == ":" })
  let parts = posStr.split(separator: ":").compactMap { Int($0) }
  guard parts.count >= 2 else { return nil }
  return QuranPosition(surah: parts[0], ayah: parts[1])
}

// MARK: - Manager Buffer Simulator

/// Simulates FollowAlongManager's confirmed text buffering.
final class ManagerBufferSimulator {
  private var confirmedBuffer = ""
  private var currentConfirmed = ""
  private var currentUnconfirmed = ""
  private var checksWithoutAdvance = 0
  private var lastUnconfirmedLength = 0
  private var staleUnconfirmedGrowth = 0
  private let staleContextThreshold = 5
  private let staleGrowthThreshold = 10

  /// Process a preview inference (all unconfirmed).
  func handlePreview(phonemes: String) {
    currentUnconfirmed = phonemes
  }

  /// Process a window inference (confirmed + unconfirmed).
  func handleWindow(confirmed: String, unconfirmed: String) {
    confirmedBuffer += currentConfirmed
    currentConfirmed = confirmed
    trimConfirmedBuffer()
    currentUnconfirmed = unconfirmed
  }

  /// Returns (prev5s, next5s) for checkReading, with stale context detection.
  func getCheckReadingInputs() -> (prev5s: String, next5s: String) {
    if checksWithoutAdvance >= staleContextThreshold
      || staleUnconfirmedGrowth >= staleGrowthThreshold {
      confirmedBuffer = ""
      currentConfirmed = ""
      checksWithoutAdvance = 0
      staleUnconfirmedGrowth = 0
    }
    let prev5s = recentConfirmed()
    let next5s = currentConfirmed + currentUnconfirmed
    return (prev5s, next5s)
  }

  /// Call after processing a check result to track stalling.
  func trackAdvance(didAdvance: Bool) {
    let unconfLen = currentUnconfirmed.count
    if !didAdvance {
      if unconfLen > lastUnconfirmedLength {
        checksWithoutAdvance += 1
        staleUnconfirmedGrowth += (unconfLen - lastUnconfirmedLength)
      }
    } else {
      checksWithoutAdvance = 0
      staleUnconfirmedGrowth = 0
    }
    lastUnconfirmedLength = unconfLen
  }

  private func recentConfirmed() -> String {
    let maxChars = 20
    if confirmedBuffer.count <= maxChars { return confirmedBuffer }
    return String(confirmedBuffer.suffix(maxChars))
  }

  private func trimConfirmedBuffer() {
    let maxChars = 200
    if confirmedBuffer.count > maxChars {
      var trimmed = String(confirmedBuffer.suffix(maxChars))
      if let spaceIdx = trimmed.firstIndex(of: " ") {
        trimmed = String(trimmed[trimmed.index(after: spaceIdx)...])
      }
      confirmedBuffer = trimmed
    }
  }
}

// MARK: - Ratchet Simulator

/// Simulates FollowAlongManager's position and word-status ratchet.
final class RatchetSimulator {
  var currentPosition: QuranPosition
  var wordStatuses: [QuranPosition: WordStatus] = [:]
  private let db: QuranDatabase

  init(startPosition: QuranPosition, db: QuranDatabase) {
    self.currentPosition = startPosition
    self.db = db
  }

  /// Process a ReadingResult and apply ratchet logic.
  /// Returns the effective position after ratcheting (nil if .lost).
  func processResult(_ result: ReadingResult, followAlong: QuranFollowAlong) -> QuranPosition? {
    switch result {
    case .correctAdvance(let pos, let statuses), .minorMistake(let pos, let statuses):
      followAlong.confirmPosition(pos)
      guard followAlong.positionEstablished else { return nil }

      let goingBack = positionIsBefore(pos, currentPosition)
      let ayahsBack = goingBack ? ayahDistance(from: pos, to: currentPosition) : 0

      if goingBack && ayahsBack <= 1 {
        let wordsBack = (pos.ayah == currentPosition.ayah)
          ? currentPosition.word - pos.word
          : currentPosition.word
        let score = endWordScore(for: pos, in: statuses)
        if pos.ayah == currentPosition.ayah && wordsBack > 0 && wordsBack <= 3 && score < 0.75 {
          applyWordStatuses(statuses)
          return nil
        }
        applyWordStatuses(statuses)
        currentPosition = pos
        return pos
      } else if goingBack && ayahsBack > 1 {
        wordStatuses = [:]
        applyWordStatuses(statuses)
        currentPosition = pos
        return pos
      } else {
        let effectivePos = positionMax(currentPosition, pos)
        let isNewPos = effectivePos != currentPosition
        if isNewPos && !shouldAcceptAdvance(to: pos, statuses: statuses) {
          return nil
        }
        if isNewPos && followAlong.positionEstablished {
          markSkippedWords(from: currentPosition, to: effectivePos)
        }
        applyWordStatuses(statuses)
        if isNewPos {
          currentPosition = effectivePos
          followAlong.updateCurrentPosition(effectivePos)
          return effectivePos
        }
        return nil
      }

    case .jumped(let pos, let statuses):
      followAlong.confirmPosition(pos)
      guard followAlong.positionEstablished else { return nil }

      let goingBack = positionIsBefore(pos, currentPosition)
      let ayahsBack = goingBack ? ayahDistance(from: pos, to: currentPosition) : 0

      if goingBack && ayahsBack <= 1 {
        let wordsBack = (pos.ayah == currentPosition.ayah)
          ? currentPosition.word - pos.word
          : currentPosition.word
        let score = endWordScore(for: pos, in: statuses)
        if pos.ayah == currentPosition.ayah && wordsBack > 0 && wordsBack <= 3 && score < 0.75 {
          applyWordStatuses(statuses)
          return nil
        }
        applyWordStatuses(statuses)
        currentPosition = pos
        return pos
      } else if goingBack && ayahsBack > 1 {
        wordStatuses = [:]
        applyWordStatuses(statuses)
        currentPosition = pos
        return pos
      } else {
        let effectivePos = positionMax(currentPosition, pos)
        let isNewPos = effectivePos != currentPosition
        if isNewPos && !shouldAcceptAdvance(to: pos, statuses: statuses) {
          return nil
        }
        if isNewPos && followAlong.positionEstablished {
          markSkippedWords(from: currentPosition, to: effectivePos)
        }
        applyWordStatuses(statuses)
        if isNewPos {
          currentPosition = effectivePos
          followAlong.updateCurrentPosition(effectivePos)
          return effectivePos
        }
        return nil
      }

    case .lost:
      return nil
    }
  }

  private let minJumpScore = 0.50

  private func endWordScore(for pos: QuranPosition, in statuses: [WordStatus]) -> Double {
    statuses.last(where: { $0.position == pos })?.score ?? 0
  }

  private func shouldAcceptAdvance(to pos: QuranPosition, statuses: [WordStatus]) -> Bool {
    let wordGap = wordDistance(from: currentPosition, to: pos)
    guard wordGap >= 3 else { return true }
    guard let endWord = statuses.last(where: { $0.position == pos }) else { return true }
    return endWord.score >= minJumpScore
  }

  private func wordDistance(from a: QuranPosition, to b: QuranPosition) -> Int {
    if a.surah != b.surah { return 999 }
    if a.ayah == b.ayah { return b.word - a.word }
    return (b.ayah - a.ayah) * 10 + b.word
  }

  private func positionIsBefore(_ a: QuranPosition, _ b: QuranPosition) -> Bool {
    if a.surah != b.surah { return a.surah < b.surah }
    if a.ayah != b.ayah { return a.ayah < b.ayah }
    return a.word < b.word
  }

  private func ayahDistance(from a: QuranPosition, to b: QuranPosition) -> Int {
    if a.surah != b.surah { return 999 }
    return abs(b.ayah - a.ayah)
  }

  private func positionMax(_ a: QuranPosition, _ b: QuranPosition) -> QuranPosition {
    if a.surah != b.surah { return a.surah > b.surah ? a : b }
    if a.ayah != b.ayah { return a.ayah > b.ayah ? a : b }
    return a.word >= b.word ? a : b
  }

  private func applyWordStatuses(_ statuses: [WordStatus]) {
    for ws in statuses {
      if let existing = wordStatuses[ws.position] {
        if ws.quality.rank > existing.quality.rank {
          wordStatuses[ws.position] = ws
        }
      } else {
        wordStatuses[ws.position] = ws
      }
    }
  }

  private func markSkippedWords(from: QuranPosition, to: QuranPosition) {
    guard from.word > 0 else { return }
    guard from.surah == to.surah else { return }

    for ayah in from.ayah...to.ayah {
      guard let phonemes = db.phonemes(surah: from.surah, ayah: ayah) else { continue }
      let wordCount = phonemes.split(separator: " ", omittingEmptySubsequences: true).count

      for word in 1...wordCount {
        let pos = QuranPosition(surah: from.surah, ayah: ayah, word: word)
        if ayah == from.ayah && word <= from.word { continue }
        if ayah == to.ayah && word >= to.word { continue }
        // Unconditionally override — no word in skip zone was actually read
        wordStatuses[pos] = WordStatus(position: pos, quality: .skipped, score: 0, errors: [])
      }
    }
  }
}

// MARK: - Replay Result (structured output for testing)

struct ReplayResult {
  let finalPosition: QuranPosition
  let wordStatuses: [(QuranPosition, WordStatus)]

  var qualityCounts: [String: Int] {
    var counts: [String: Int] = [:]
    for (_, ws) in wordStatuses {
      let name = qualityName(ws.quality)
      counts[name, default: 0] += 1
    }
    return counts
  }

  var wordQualities: [String: String] {
    var map: [String: String] = [:]
    for (pos, ws) in wordStatuses {
      map[pos.description] = qualityName(ws.quality)
    }
    return map
  }
}

// MARK: - Baseline JSON Model

struct LogBaseline: Codable {
  let logFile: String
  let finalPosition: String
  let qualityCounts: [String: Int]
  let wordQualities: [String: String]

  enum CodingKeys: String, CodingKey {
    case logFile = "log_file"
    case finalPosition = "final_position"
    case qualityCounts = "quality_counts"
    case wordQualities = "word_qualities"
  }
}

struct BaselinesFile: Codable {
  var baselines: [LogBaseline]
}

// MARK: - Replay Engine

func qualityName(_ q: WordQuality) -> String {
  switch q {
  case .pending:       return "pending"
  case .skipped:       return "skipped"
  case .uncertain:     return "uncertain"
  case .wrong:         return "wrong"
  case .tashkeelError: return "tashkeelError"
  case .minor:         return "minor"
  case .correct:       return "correct"
  }
}

func runReplay(dbPath: String, logPath: String, verbose: Bool) {
  let result = executeReplay(dbPath: dbPath, logPath: logPath, verbose: verbose, quiet: false)
  printReplaySummary(result)
}

func executeReplay(dbPath: String, logPath: String, verbose: Bool, quiet: Bool = false) -> ReplayResult {
  let db: QuranDatabase
  do {
    db = try QuranDatabase(dbPath: dbPath)
  } catch {
    print("❌ Failed to open database: \(dbPath) — \(error)")
    Foundation.exit(1)
  }

  let logText: String
  do {
    logText = try String(contentsOfFile: logPath, encoding: .utf8)
  } catch {
    print("❌ Failed to read log file: \(logPath)")
    Foundation.exit(1)
  }

  guard let startPos = extractStartPosition(from: logText) else {
    print("❌ Could not determine start position from log (no expected=S:A field found)")
    Foundation.exit(1)
  }

  let events = parseLogEvents(from: logText)
  let inferenceCount = events.filter { if case .inference = $0 { return true }; return false }.count
  if !quiet {
    print("📋 Parsed \(events.count) log events (\(inferenceCount) inferences)")
    print("📍 Start position: \(startPos)")
    print()
  }

  let index = PhonemeIndex(db: db)
  let followAlong = QuranFollowAlong(index: index, startPosition: startPos, contextRadius: 5)
  followAlong.confirmPosition(startPos)

  let buffer = ManagerBufferSimulator()
  let ratchet = RatchetSimulator(startPosition: startPos, db: db)

  var stepNumber = 0

  for event in events {
    switch event {
    case .inference(let inf):
      stepNumber += 1

      // Update buffers
      switch inf.type {
      case .preview:
        buffer.handlePreview(phonemes: inf.phonemes)
      case .window:
        buffer.handleWindow(confirmed: inf.confirmed!, unconfirmed: inf.unconfirmed!)
      }

      // Get checkReading inputs
      let (prev5s, next5s) = buffer.getCheckReadingInputs()

      guard !next5s.trimmingCharacters(in: .whitespaces).isEmpty else {
        if verbose {
          print("  [\(stepNumber)] \(inf.type.rawValue) — (empty next5s, skipping)")
        }
        continue
      }

      // Print inference header
      if !quiet {
        switch inf.type {
        case .preview:
          print("\(inf.type.rawValue) inference \(inf.durationMs)ms — \(inf.phonemes)")
        case .window:
          print("\(inf.type.rawValue) inference \(inf.durationMs)ms — confirmed: \(inf.confirmed!), unconfirmed: \(inf.unconfirmed!)")
        }
      }

      if verbose {
        print("  prev5s: \"\(prev5s)\"")
        print("  next5s: \"\(next5s)\"")
      }

      // Run checkReading
      let result = followAlong.checkReading(prev5s: prev5s, next5s: next5s)

      // Process through ratchet
      let effectivePos = ratchet.processResult(result, followAlong: followAlong)

      // Track stalling for stale context detection
      buffer.trackAdvance(didAdvance: effectivePos != nil)

      if !quiet, let pos = effectivePos {
        print("Position → \(pos)")
      }

    case .position:
      // Original position events are for comparison only, we generate our own
      break

    case .matchLine(let line):
      // Original match lines are for comparison — skip in replay output
      if verbose {
        print("  [original] \(line)")
      }

    case .gradeLine(let line):
      // Original grade lines are for comparison — skip in replay output
      if verbose {
        print("  [original] \(line)")
      }
    }
  }

  // Return structured result
  let sorted = ratchet.wordStatuses.sorted { a, b in
    if a.key.surah != b.key.surah { return a.key.surah < b.key.surah }
    if a.key.ayah != b.key.ayah { return a.key.ayah < b.key.ayah }
    return a.key.word < b.key.word
  }
  return ReplayResult(
    finalPosition: ratchet.currentPosition,
    wordStatuses: sorted.map { ($0.key, $0.value) }
  )
}

func printReplaySummary(_ result: ReplayResult) {
  print()
  print("═══════════════════════════════════════")
  print("  REPLAY SUMMARY")
  print("═══════════════════════════════════════")
  print("Final position: \(result.finalPosition)")
  print()

  if result.wordStatuses.isEmpty {
    print("No word statuses recorded.")
  } else {
    print("Word statuses:")
    var currentAyah = -1
    for (pos, ws) in result.wordStatuses {
      if pos.ayah != currentAyah {
        if currentAyah != -1 { print() }
        currentAyah = pos.ayah
      }
      let qualStr = qualityName(ws.quality)
      let errStr = ws.errors.isEmpty ? "" : " errors=\(ws.errors.count)"
      print("  \(pos): \(qualStr) (score=\(String(format: "%.2f", ws.score))\(errStr))")
    }
  }
  print()
}

// MARK: - Quality Rank (for regression detection)

/// Quality rank for regression comparison. Higher is better.
func qualityRank(_ name: String) -> Int {
  switch name {
  case "pending":       return 0
  case "skipped":       return 1
  case "uncertain":     return 2
  case "wrong":         return 2
  case "tashkeelError": return 3
  case "minor":         return 4
  case "correct":       return 5
  default:              return -1
  }
}

// MARK: - Baseline Generation

func generateBaseline(logFile: String, result: ReplayResult) -> LogBaseline {
  return LogBaseline(
    logFile: logFile,
    finalPosition: result.finalPosition.description,
    qualityCounts: result.qualityCounts,
    wordQualities: result.wordQualities
  )
}

func updateBaselines(dbPath: String, logDir: String, baselinesPath: String) {
  let fm = FileManager.default
  guard let entries = try? fm.contentsOfDirectory(atPath: logDir) else {
    print("❌ Cannot read log directory: \(logDir)")
    Foundation.exit(1)
  }
  let logFiles = entries.filter { $0.hasSuffix(".log") }.sorted()
  guard !logFiles.isEmpty else {
    print("❌ No .log files found in \(logDir)")
    Foundation.exit(1)
  }

  var baselines: [LogBaseline] = []
  for logFile in logFiles {
    let logPath = (logDir as NSString).appendingPathComponent(logFile)
    print("Generating baseline for \(logFile)...")
    let result = executeReplay(dbPath: dbPath, logPath: logPath, verbose: false, quiet: true)
    let baseline = generateBaseline(logFile: logFile, result: result)
    baselines.append(baseline)
    print("  Final: \(baseline.finalPosition), words: \(baseline.wordQualities.count)")
  }

  let file = BaselinesFile(baselines: baselines)
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
  guard let data = try? encoder.encode(file) else {
    print("❌ Failed to encode baselines")
    Foundation.exit(1)
  }
  guard let json = String(data: data, encoding: .utf8) else {
    print("❌ Failed to convert baselines to string")
    Foundation.exit(1)
  }
  do {
    try json.write(toFile: baselinesPath, atomically: true, encoding: .utf8)
  } catch {
    print("❌ Failed to write baselines: \(error)")
    Foundation.exit(1)
  }
  print()
  print("✅ Wrote \(baselines.count) baselines to \(baselinesPath)")
}

// MARK: - Test Runner

struct TestFailure {
  let logFile: String
  let message: String
}

func runTests(dbPath: String, logDir: String, baselinesPath: String) -> Bool {
  // Load baselines
  guard let data = FileManager.default.contents(atPath: baselinesPath),
        let baselinesFile = try? JSONDecoder().decode(BaselinesFile.self, from: data) else {
    print("❌ Cannot read baselines from \(baselinesPath)")
    print("   Run with --update-baseline first to generate baselines.")
    Foundation.exit(1)
  }

  var totalTests = 0
  var passedTests = 0
  var failures: [TestFailure] = []

  for baseline in baselinesFile.baselines {
    totalTests += 1
    let logPath = (logDir as NSString).appendingPathComponent(baseline.logFile)
    guard FileManager.default.fileExists(atPath: logPath) else {
      failures.append(TestFailure(logFile: baseline.logFile, message: "Log file not found"))
      continue
    }

    print("Testing \(baseline.logFile)...", terminator: "")
    let result = executeReplay(dbPath: dbPath, logPath: logPath, verbose: false, quiet: true)
    let testFailures = compareResults(result: result, baseline: baseline)

    if testFailures.isEmpty {
      passedTests += 1
      print(" ✅")
    } else {
      print(" ❌")
      for f in testFailures {
        failures.append(TestFailure(logFile: baseline.logFile, message: f))
        print("    \(f)")
      }
    }
  }

  // Print summary
  print()
  print("═══════════════════════════════════════")
  print("  LOG TEST RESULTS")
  print("═══════════════════════════════════════")
  print("Tests:  \(passedTests)/\(totalTests) passed")
  if !failures.isEmpty {
    print("Failures: \(failures.count)")
  }
  print()

  return failures.isEmpty
}

/// Compare actual results against baseline. Returns list of failure messages.
/// Fuzzy rules:
/// - Final position must match
/// - Per-word quality downgrades (better→worse) are regressions
/// - Upgrades (worse→better) are allowed
/// - New words appearing or disappearing with neutral quality (skipped/pending) are allowed
func compareResults(result: ReplayResult, baseline: LogBaseline) -> [String] {
  var failures: [String] = []

  // Check final position
  let actualFinal = result.finalPosition.description
  if actualFinal != baseline.finalPosition {
    failures.append("Final position: expected \(baseline.finalPosition), got \(actualFinal)")
  }

  let actualQualities = result.wordQualities

  // Check for regressions in word quality (downgrade = worse rank)
  for (pos, expectedQuality) in baseline.wordQualities {
    guard let actualQuality = actualQualities[pos] else {
      // Word disappeared from results. Only a regression if it was graded (not skipped/pending).
      let expectedRank = qualityRank(expectedQuality)
      if expectedRank > qualityRank("skipped") {
        failures.append("\(pos): was \(expectedQuality), now missing")
      }
      continue
    }

    let expectedRank = qualityRank(expectedQuality)
    let actualRank = qualityRank(actualQuality)

    // Downgrade is a regression
    if actualRank < expectedRank {
      failures.append("\(pos): regression \(expectedQuality) → \(actualQuality)")
    }
  }

  return failures
}

// MARK: - Main Entry Point

@main
struct ReplayLog {
  static func main() {
    let args = CommandLine.arguments

    if args.contains("--test") || args.contains("--update-baseline") {
      guard args.count >= 2 else {
        printUsage()
        Foundation.exit(1)
      }
      let dbPath = args[1]
      guard FileManager.default.fileExists(atPath: dbPath) else {
        print("❌ Database not found: \(dbPath)")
        Foundation.exit(1)
      }

      // Derive log dir and baselines path from db path
      // dbPath is like .../ios/Muraja/Resources/quran.db
      var dir = (dbPath as NSString).deletingLastPathComponent // Resources
      dir = (dir as NSString).deletingLastPathComponent        // Muraja
      dir = (dir as NSString).deletingLastPathComponent        // ios
      let repoDir = (dir as NSString).deletingLastPathComponent // repo root
      // Fallback: use tools/ relative to the binary or cwd
      let toolsDir: String
      if FileManager.default.fileExists(
        atPath: (repoDir as NSString).appendingPathComponent("tools/test_logs")) {
        toolsDir = (repoDir as NSString).appendingPathComponent("tools")
      } else {
        // Try relative to cwd
        toolsDir = (FileManager.default.currentDirectoryPath as NSString)
          .appendingPathComponent("tools")
      }
      let logDir = (toolsDir as NSString).appendingPathComponent("test_logs")
      let baselinesPath = (logDir as NSString).appendingPathComponent("baselines.json")

      guard FileManager.default.fileExists(atPath: logDir) else {
        print("❌ Log directory not found: \(logDir)")
        Foundation.exit(1)
      }

      if args.contains("--update-baseline") {
        updateBaselines(dbPath: dbPath, logDir: logDir, baselinesPath: baselinesPath)
      } else {
        let allPassed = runTests(dbPath: dbPath, logDir: logDir, baselinesPath: baselinesPath)
        Foundation.exit(allPassed ? 0 : 1)
      }
    } else {
      // Original replay mode
      guard args.count >= 3 else {
        printUsage()
        Foundation.exit(1)
      }

      let dbPath = args[1]
      let logPath = args[2]
      let verbose = args.contains("--verbose")

      guard FileManager.default.fileExists(atPath: dbPath) else {
        print("❌ Database not found: \(dbPath)")
        Foundation.exit(1)
      }
      guard FileManager.default.fileExists(atPath: logPath) else {
        print("❌ Log file not found: \(logPath)")
        Foundation.exit(1)
      }

      runReplay(dbPath: dbPath, logPath: logPath, verbose: verbose)
    }
  }

  static func printUsage() {
    print("Usage:")
    print("  replay_log <db> <logfile> [--verbose]    Replay a single log")
    print("  replay_log <db> --test                   Run all log tests against baselines")
    print("  replay_log <db> --update-baseline        Generate/update baselines from current code")
    print()
    print("Log files go in tools/test_logs/*.log")
    print("Baselines are stored in tools/test_logs/baselines.json")
  }
}
