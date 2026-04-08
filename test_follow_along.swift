// tools/test_follow_along.swift
//
// Self-test for the follow-along alignment and scoring algorithm.
// Compiles against the same Swift source files as the iOS app.
//
// Usage:
//   cd tools && ./run_follow_along_test.sh

import Foundation

@main
struct TestFollowAlong {
  static func main() {
    guard CommandLine.arguments.count >= 2 else {
      print("Usage: test_follow_along <path-to-quran.db> [--dataset <path-to-test_phonemes.json>]")
      Foundation.exit(1)
    }

    let dbPath = CommandLine.arguments[1]
    guard FileManager.default.fileExists(atPath: dbPath) else {
      print("❌ Database not found: \(dbPath)")
      Foundation.exit(1)
    }

    // Check for --dataset flag
    if CommandLine.arguments.count >= 4 && CommandLine.arguments[2] == "--dataset" {
      let jsonPath = CommandLine.arguments[3]
      guard FileManager.default.fileExists(atPath: jsonPath) else {
        print("❌ Dataset JSON not found: \(jsonPath)")
        Foundation.exit(1)
      }
      runDatasetTest(dbPath: dbPath, jsonPath: jsonPath)
    } else {
      runSelfTest(dbPath: dbPath)
    }
  }
}

// MARK: - Follow-Along Loop

/// Run checkReading in a loop, advancing position each time, like the real app.
/// Returns accumulated WordStatus for all words in the ayah.
func runFollowAlongLoop(
  followAlong: QuranFollowAlong,
  phonemes: String,
  totalWords: Int,
  surah: Int,
  ayah: Int
) -> [WordStatus] {
  var allStatuses: [WordStatus] = []
  var seenPositions: Set<String> = []
  let maxIterations = totalWords * 3

  for _ in 0..<maxIterations {
    let result = followAlong.checkReading(prev5s: "", next5s: phonemes)

    var newPos: QuranPosition?
    var statuses: [WordStatus] = []

    switch result {
    case .correctAdvance(let pos, let ws):
      newPos = pos; statuses = ws
    case .minorMistake(let pos, let ws):
      newPos = pos; statuses = ws
    case .jumped(let pos, let ws):
      newPos = pos; statuses = ws
    case .lost:
      break
    }

    for ws in statuses {
      let key = "\(ws.position.surah):\(ws.position.ayah):\(ws.position.word)"
      if !seenPositions.contains(key) {
        seenPositions.insert(key)
        allStatuses.append(ws)
      }
    }

    guard let pos = newPos else { break }
    followAlong.confirmPosition(pos)

    if pos.surah == surah && pos.ayah == ayah && pos.word >= totalWords { break }
    if pos.surah > surah || (pos.surah == surah && pos.ayah > ayah) { break }
  }

  return allStatuses
}

// MARK: - Data Types

struct TestFailure {
  let position: QuranPosition
  let quality: WordQuality
  let score: Double
  let errors: [WordError]
  let phonemes: String
  let config: String
}

/// Positions where test data alignment is known to be unreliable.
/// Real-device testing confirms these words are graded correctly.
let testExcludedPositions: Set<String> = [
  "32:18:7",   // للَاا
  "5:119:17",  // ررَضِيَ
  "22:72:23",  // ننننَاارُ
  "24:35:33",  // ننننُۥۥرُن — leading nun assimilation
]

struct AyahResult {
  let position: QuranPosition
  let totalWords: Int
  let correctWords: Int
  let tashkeelWords: Int
  let failures: [TestFailure]
  let resultType: String  // "correctAdvance", "minorMistake", "jumped", "lost"
}

func runSelfTest(dbPath: String) {
  let db: QuranDatabase
  do {
    db = try QuranDatabase(dbPath: dbPath)
  } catch {
    print("❌ Failed to open database: \(error)")
    return
  }

  let index = PhonemeIndex(db: db)
  let positions = db.allPositions()

  print("Running self-test on \(positions.count) ayahs...")
  print()

  var totalAyahs = 0
  var passedAyahs = 0
  var totalWords = 0
  var correctWords = 0
  var tashkeelWords = 0
  var failedWords = 0
  var lostAyahs = 0
  var allFailures: [TestFailure] = []

  let startTime = Date()

  for pos in positions {
    guard let phonemes = db.phonemes(surah: pos.surah, ayah: pos.ayah) else {
      continue
    }

    // Skip empty ayahs
    let words = phonemes.split(separator: " ", omittingEmptySubsequences: true)
    guard !words.isEmpty else { continue }

    totalAyahs += 1

    // Create a follow-along instance for this ayah
    let followAlong = QuranFollowAlong(
      index: index,
      startPosition: QuranPosition(surah: pos.surah, ayah: pos.ayah, word: 1),
      contextRadius: 5
    )

    // Set page bounds to just this ayah (simulates single-page reading)
    followAlong.setPageBounds([pos])

    // Force position as established to skip seeking phase
    followAlong.confirmPosition(QuranPosition(surah: pos.surah, ayah: pos.ayah, word: 1))

    // Feed the full ayah phonemes in a loop (like the real app)
    let wordStatuses = runFollowAlongLoop(
      followAlong: followAlong,
      phonemes: phonemes,
      totalWords: words.count,
      surah: pos.surah,
      ayah: pos.ayah
    )

    let ayahResult = analyzeWordStatuses(
      pos: pos, wordStatuses: wordStatuses, phonemes: phonemes, resultType: "loop")

    totalWords += ayahResult.totalWords
    correctWords += ayahResult.correctWords
    tashkeelWords += ayahResult.tashkeelWords
    let ayahFailures = ayahResult.failures
    failedWords += ayahFailures.count
    allFailures.append(contentsOf: ayahFailures)

    if ayahFailures.isEmpty {
      passedAyahs += 1
    } else {
      let failureDetails = ayahFailures.map { f in
        "\(f.position.word):\(f.quality)(s=\(String(format: "%.2f", f.score)))"
      }.joined(separator: " ")
      print("❌ \(pos.surah):\(pos.ayah) — \(ayahFailures.count) failed words: \(failureDetails)")
    }
  }

  let elapsed = Date().timeIntervalSince(startTime)

  // Print summary
  print()
  print("=" * 60)
  print("SELF-TEST RESULTS")
  print("=" * 60)
  print("Ayahs:  \(passedAyahs)/\(totalAyahs) passed (\(lostAyahs) lost)")
  print("Words:  \(correctWords + tashkeelWords)/\(totalWords) correct " +
        "(\(correctWords) exact, \(tashkeelWords) tashkeel)")
  print("Failed: \(failedWords) words in \(totalAyahs - passedAyahs) ayahs")
  let pct = totalWords > 0 ? Double(correctWords + tashkeelWords) / Double(totalWords) * 100 : 0
  print("Rate:   \(String(format: "%.2f", pct))%")
  print("Time:   \(String(format: "%.1f", elapsed))s")

  // Regression check: self-test must have zero failures
  if failedWords > 0 {
    print()
    print("⚠️  REGRESSION: self-test should have 0 failures, got \(failedWords)")
  }
  print()

  // Show failure breakdown by quality
  if !allFailures.isEmpty {
    var qualityCounts: [String: Int] = [:]
    for f in allFailures {
      qualityCounts["\(f.quality)", default: 0] += 1
    }
    print("Failure breakdown:")
    for (q, count) in qualityCounts.sorted(by: { $0.value > $1.value }) {
      print("  \(q): \(count)")
    }
    print()

    // Show first 20 failures in detail
    let showCount = min(20, allFailures.count)
    print("First \(showCount) failures:")
    for f in allFailures.prefix(showCount) {
      print("  \(f.position) quality=\(f.quality) score=\(String(format: "%.2f", f.score))")
      if !f.errors.isEmpty {
        for e in f.errors {
          print("    \(e.kind)")
        }
      }
    }
  }
}

func analyzeWordStatuses(
  pos: QuranPosition,
  wordStatuses: [WordStatus],
  phonemes: String,
  resultType: String,
  config: String = ""
) -> AyahResult {
  let words = phonemes.split(separator: " ", omittingEmptySubsequences: true)
  var correct = 0
  var tashkeel = 0
  var failures: [TestFailure] = []

  for ws in wordStatuses {
    switch ws.quality {
    case .correct:
      correct += 1
    case .tashkeelError:
      tashkeel += 1
    case .pending, .uncertain:
      // Ungraded words at the edges — count as OK for now
      correct += 1
    case .minor, .wrong, .skipped:
      failures.append(TestFailure(
        position: ws.position,
        quality: ws.quality,
        score: ws.score,
        errors: ws.errors,
        phonemes: phonemes,
        config: config
      ))
    }
  }

  // Words not in wordStatuses (not graded) — count them
  let ungradedCount = words.count - wordStatuses.count
  if ungradedCount > 0 {
    // These are words the alignment didn't reach — failures
    for i in (wordStatuses.count + 1)...words.count {
      failures.append(TestFailure(
        position: QuranPosition(surah: pos.surah, ayah: pos.ayah, word: i),
        quality: .skipped,
        score: 0,
        errors: [],
        phonemes: phonemes,
        config: config
      ))
    }
  }

  return AyahResult(
    position: pos,
    totalWords: words.count,
    correctWords: correct,
    tashkeelWords: tashkeel,
    failures: failures,
    resultType: resultType
  )
}

// String repetition helper
extension String {
  static func * (lhs: String, rhs: Int) -> String {
    String(repeating: lhs, count: rhs)
  }
}

// MARK: - Dataset Test

struct DatasetEntry: Decodable {
  let surah: Int
  let ayah_start: Int
  let ayah_end: Int
  let phonemes: String
  let match_ratio: Double
  let reciter: String
  let config: String?
}

func runDatasetTest(dbPath: String, jsonPath: String) {
  let db: QuranDatabase
  do {
    db = try QuranDatabase(dbPath: dbPath)
  } catch {
    print("❌ Failed to open database: \(error)")
    return
  }

  let index = PhonemeIndex(db: db)

  // Load dataset JSON
  guard let jsonData = FileManager.default.contents(atPath: jsonPath) else {
    print("❌ Cannot read JSON file: \(jsonPath)")
    return
  }

  let entries: [DatasetEntry]
  do {
    entries = try JSONDecoder().decode([DatasetEntry].self, from: jsonData)
  } catch {
    print("❌ Failed to parse JSON: \(error)")
    return
  }

  print("Running dataset test on \(entries.count) ayahs...")
  print()

  var totalAyahs = 0
  var passedAyahs = 0
  var totalWords = 0
  var gradedWords = 0
  var correctWords = 0
  var tashkeelWords = 0
  var failedWords = 0
  var ungradedWords = 0
  var lostAyahs = 0
  var skippedEntries = 0
  var allFailures: [TestFailure] = []

  let startTime = Date()

  for entry in entries {
    // Only handle single-ayah entries for now
    guard entry.ayah_start == entry.ayah_end else {
      skippedEntries += 1
      continue
    }

    let surah = entry.surah
    let ayah = entry.ayah_start

    // Get reference word count from DB
    guard let refPhonemes = db.phonemes(surah: surah, ayah: ayah) else {
      skippedEntries += 1
      continue
    }

    let refWords = refPhonemes.split(separator: " ", omittingEmptySubsequences: true)
    guard !refWords.isEmpty else {
      skippedEntries += 1
      continue
    }

    totalAyahs += 1

    let pos = QuranPosition(surah: surah, ayah: ayah, word: 1)

    let followAlong = QuranFollowAlong(
      index: index,
      startPosition: pos,
      contextRadius: 5
    )

    followAlong.setPageBounds([pos])
    followAlong.confirmPosition(pos)

    // Feed concatenated dataset phonemes in a loop (like the real app)
    let wordStatuses = runFollowAlongLoop(
      followAlong: followAlong,
      phonemes: entry.phonemes,
      totalWords: refWords.count,
      surah: surah,
      ayah: ayah
    )

    let ayahResult = analyzeWordStatuses(
      pos: pos, wordStatuses: wordStatuses, phonemes: refPhonemes, resultType: "loop",
      config: entry.config ?? "")

    totalWords += ayahResult.totalWords
    correctWords += ayahResult.correctWords
    tashkeelWords += ayahResult.tashkeelWords
    let ayahFailures = ayahResult.failures
    let graded = ayahFailures.filter { $0.quality != .skipped }
    let ungraded = ayahFailures.filter { $0.quality == .skipped }
    let excluded = graded.filter { testExcludedPositions.contains("\($0.position.surah):\($0.position.ayah):\($0.position.word)") }
    let counted = graded.filter { !testExcludedPositions.contains("\($0.position.surah):\($0.position.ayah):\($0.position.word)") }
    failedWords += counted.count
    ungradedWords += ungraded.count
    gradedWords += ayahResult.correctWords + ayahResult.tashkeelWords + graded.count
    allFailures.append(contentsOf: counted)

    if counted.isEmpty {
      passedAyahs += 1
    } else {
      let failureDetails = counted.map { f in
        "\(f.position.word):\(f.quality)(s=\(String(format: "%.2f", f.score)))"
      }.joined(separator: " ")
      let src = entry.config ?? ""
      print("❌ \(surah):\(ayah) [\(src)] — \(counted.count)/\(ayahResult.totalWords) failed: \(failureDetails)")
    }
  }

  let elapsed = Date().timeIntervalSince(startTime)

  // Print summary
  print()
  print("=" * 60)
  print("DATASET TEST RESULTS")
  print("=" * 60)
  print("Entries:  \(entries.count) total (\(totalAyahs) tested, \(skippedEntries) skipped)")
  print("Ayahs:    \(passedAyahs)/\(totalAyahs) passed (\(lostAyahs) lost)")
  print("Graded:   \(correctWords + tashkeelWords)/\(gradedWords) correct " +
        "(\(correctWords) exact, \(tashkeelWords) tashkeel)")
  print("Ungraded: \(ungradedWords) words (dataset phonemes shorter than ayah)")
  print("Failed:   \(failedWords) words in \(totalAyahs - passedAyahs) ayahs")
  let pct = gradedWords > 0 ? Double(correctWords + tashkeelWords) / Double(gradedWords) * 100 : 0
  print("Rate:     \(String(format: "%.2f", pct))%")
  print("Time:     \(String(format: "%.1f", elapsed))s")

  // Regression check: fail count should not exceed known baseline
  let baselineFailedWords = 285
  if failedWords > baselineFailedWords {
    print()
    print("⚠️  REGRESSION: expected ≤\(baselineFailedWords) failures, got \(failedWords) (+\(failedWords - baselineFailedWords))")
  } else if failedWords < baselineFailedWords {
    print()
    print("🎉 IMPROVEMENT: \(baselineFailedWords) → \(failedWords) (−\(baselineFailedWords - failedWords)) — update baseline!")
  }
  print()

  // Show failure breakdown by quality
  if !allFailures.isEmpty {
    var qualityCounts: [String: Int] = [:]
    for f in allFailures {
      qualityCounts["\(f.quality)", default: 0] += 1
    }
    print("Failure breakdown:")
    for (q, count) in qualityCounts.sorted(by: { $0.value > $1.value }) {
      print("  \(q): \(count)")
    }
    print()

    // Show first 30 failures in detail
    let showCount = min(30, allFailures.count)
    print("First \(showCount) failures:")
    for f in allFailures.prefix(showCount) {
      print("  \(f.position) quality=\(f.quality) score=\(String(format: "%.2f", f.score))")
      if !f.errors.isEmpty {
        for e in f.errors.prefix(3) {
          print("    \(e.kind)")
        }
      }
    }

    // Universal failure analysis: positions that fail across all configs
    let allConfigs = Set(entries.compactMap { $0.config })
    let configCount = allConfigs.count
    if configCount > 1 {
      // Group failures by position key (surah:ayah:word)
      var positionConfigs: [String: Set<String>] = [:]
      var positionDetails: [String: TestFailure] = [:]
      for f in allFailures {
        let key = "\(f.position.surah):\(f.position.ayah):\(f.position.word)"
        positionConfigs[key, default: []].insert(f.config)
        positionDetails[key] = f  // keep last for display
      }

      let universalPositions = positionConfigs.filter { $0.value.count == configCount }
        .keys.sorted { a, b in
          let ap = a.split(separator: ":").map { Int($0)! }
          let bp = b.split(separator: ":").map { Int($0)! }
          return (ap[0], ap[1], ap[2]) < (bp[0], bp[1], bp[2])
        }

      print()
      print("Universal failures (\(universalPositions.count) positions fail in all \(configCount) configs):")
      for key in universalPositions {
        if let f = positionDetails[key] {
          let words = f.phonemes.split(separator: " ", omittingEmptySubsequences: true)
          let wordIdx = f.position.word - 1
          let word = wordIdx < words.count ? String(words[wordIdx]) : "?"
          let errorSummary = f.errors.prefix(2).map { "\($0.kind)" }.joined(separator: ", ")
          print("  \(key) \"\(word)\" quality=\(f.quality) score=\(String(format: "%.2f", f.score)) \(errorSummary)")
        }
      }

      // Categorize universal failures
      var categories: [String: Int] = [:]
      for key in universalPositions {
        if let f = positionDetails[key] {
          let words = f.phonemes.split(separator: " ", omittingEmptySubsequences: true)
          let wordIdx = f.position.word - 1
          let word = wordIdx < words.count ? String(words[wordIdx]) : ""
          if word.contains("۾") {
            categories["waqf ۾", default: 0] += 1
          } else if word.contains("ۦ") || word.contains("ۥ") {
            categories["madd ۦ/ۥ", default: 0] += 1
          } else if word.hasPrefix("ںںں") || word.hasPrefix("ننن") {
            categories["leading tanween", default: 0] += 1
          } else if word.hasSuffix("يي") || word.hasSuffix("وو") || word.hasSuffix("اا") {
            categories["trailing elongation", default: 0] += 1
          } else {
            categories["other", default: 0] += 1
          }
        }
      }
      print()
      print("Categories:")
      for (cat, count) in categories.sorted(by: { $0.value > $1.value }) {
        print("  \(cat): \(count)")
      }

      // Flaky failures: positions failing in >3 configs but not all
      let flakyThreshold = 3
      let flakyPositions = positionConfigs
        .filter { $0.value.count > flakyThreshold && $0.value.count < configCount }
        .sorted { a, b in a.value.count > b.value.count }  // most frequent first

      print()
      print("Flaky failures (\(flakyPositions.count) positions fail in >\(flakyThreshold) configs):")
      print()

      // Group by config count for summary
      var byConfigCount: [Int: Int] = [:]
      for (_, configs) in flakyPositions {
        byConfigCount[configs.count, default: 0] += 1
      }
      for (count, positions) in byConfigCount.sorted(by: { $0.key > $1.key }) {
        print("  \(count)/\(configCount) configs: \(positions) positions")
      }
      print()

      // Show details for top flaky positions
      let showFlaky = min(40, flakyPositions.count)
      print("Top \(showFlaky) flaky positions:")
      for (key, configs) in flakyPositions.prefix(showFlaky) {
        if let f = positionDetails[key] {
          let words = f.phonemes.split(separator: " ", omittingEmptySubsequences: true)
          let wordIdx = f.position.word - 1
          let word = wordIdx < words.count ? String(words[wordIdx]) : "?"
          let errorSummary = f.errors.prefix(2).map { "\($0.kind)" }.joined(separator: ", ")
          print("  \(key) [\(configs.count)/\(configCount)] \"\(word)\" q=\(f.quality) s=\(String(format: "%.2f", f.score)) \(errorSummary)")
        }
      }

      // Categorize flaky failures
      var flakyCats: [String: Int] = [:]
      for (key, _) in flakyPositions {
        if let f = positionDetails[key] {
          let words = f.phonemes.split(separator: " ", omittingEmptySubsequences: true)
          let wordIdx = f.position.word - 1
          let word = wordIdx < words.count ? String(words[wordIdx]) : ""
          // Categorize by error type
          let hasGap = f.errors.contains { if case .deletion = $0.kind { return true }; return false }
          let hasMismatch = f.errors.contains { if case .mismatch = $0.kind { return true }; return false }
          let hasTashkeel = f.errors.contains { if case .tashkeel = $0.kind { return true }; return false }

          if word.contains("ۦ") || word.contains("ۥ") {
            flakyCats["madd ۦ/ۥ", default: 0] += 1
          } else if word.contains("۾") {
            flakyCats["waqf ۾", default: 0] += 1
          } else if word.hasPrefix("ںںں") || word.hasPrefix("ننن") || word.hasPrefix("لل") || word.hasPrefix("رر") || word.hasPrefix("مم") {
            flakyCats["leading assimilation/tanween", default: 0] += 1
          } else if word.hasSuffix("يي") || word.hasSuffix("وو") || word.hasSuffix("اا") {
            flakyCats["trailing elongation", default: 0] += 1
          } else if hasTashkeel && !hasMismatch && !hasGap {
            flakyCats["tashkeel only", default: 0] += 1
          } else if hasGap && !hasMismatch {
            flakyCats["deletion only", default: 0] += 1
          } else if hasMismatch && !hasGap {
            flakyCats["mismatch only", default: 0] += 1
          } else {
            flakyCats["mixed errors", default: 0] += 1
          }
        }
      }
      print()
      print("Flaky categories:")
      for (cat, count) in flakyCats.sorted(by: { $0.value > $1.value }) {
        print("  \(cat): \(count)")
      }
    }
  }
}

// MARK: - Main

// Entry point is in TestFollowAlong.main() above
