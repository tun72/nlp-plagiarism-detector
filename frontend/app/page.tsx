"use client";

import { useMemo, useState } from "react";

import FlaggedSentence from "@/components/FlaggedSentence";
import { checkPlagiarism, type PlagiarismResponse, type SentenceResult } from "@/lib/api";

const SAMPLE_TEXT =
  "Natural language processing enables machines to understand human language. " +
  "TF-IDF and embedding models can be combined for better plagiarism detection in research writing.";

const RED_LINE_THRESHOLD = 0.25;
const MAX_INPUT_WORDS = 1500;
const LONG_TEXT_MODE_WORDS = 900;
const UCSY_KEYWORD = "ucsy";
const UCSY_DOMAIN = "ucsy.edu.mm";
const UCSY_HIGHLIGHT_WORDS = ["ucsy", "university", "computer", "studies", "yangon"];
const HIGHLIGHT_STOPWORDS = new Set([
  "a",
  "an",
  "the",
  "and",
  "or",
  "but",
  "for",
  "nor",
  "so",
  "to",
  "of",
  "in",
  "on",
  "at",
  "by",
  "from",
  "with",
  "as",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "this",
  "that",
  "these",
  "those",
  "it",
  "its",
  "their",
  "there",
  "here",
  "than",
  "then",
  "into",
  "about",
  "over",
  "under",
  "between",
  "within",
  "without",
  "can",
  "could",
  "would",
  "should",
  "may",
  "might",
  "must",
  "will",
  "shall",
  "do",
  "does",
  "did",
  "done",
  "have",
  "has",
  "had",
  "not",
  "no",
  "yes",
  "if",
  "else",
  "also",
  "such",
  "very",
  "more",
  "most",
  "many",
  "much",
  "some",
  "any",
  "each",
  "every",
  "all",
  "both",
  "few",
  "other",
]);

function isUcsyReference(value: string | undefined): boolean {
  const lowered = (value || "").toLowerCase();
  return lowered.includes(UCSY_DOMAIN) || lowered.includes(UCSY_KEYWORD);
}

function tokenizeWords(text: string): string[] {
  const tokens = text.toLowerCase().match(/[a-z0-9]+(?:['-][a-z0-9]+)*/g);
  return tokens || [];
}

function buildUcsyHighlightWords(sentence: string, enabled: boolean, sentenceHasUcsyMatch: boolean): string[] {
  if (!enabled) {
    return [];
  }
  const sentenceWords = new Set(tokenizeWords(sentence));
  const targetWords = sentenceHasUcsyMatch ? UCSY_HIGHLIGHT_WORDS : [UCSY_KEYWORD];
  return targetWords.filter((word) => sentenceWords.has(word));
}

function isHighlightablePlagiarismWord(word: string): boolean {
  if (!word || word.length < 4) {
    return false;
  }
  if (/^\d+$/.test(word)) {
    return false;
  }
  return !HIGHLIGHT_STOPWORDS.has(word);
}

function buildPlagiarismHighlightWords(item: SentenceResult): string[] {
  const sentenceWords = tokenizeWords(item.sentence);
  if (!sentenceWords.length || !item.matches.length) {
    return [];
  }

  const sentenceSet = new Set(sentenceWords);
  const overlapFrequency = new Map<string, number>();
  const strongMatches = item.matches.filter(
    (match) =>
      match.plagiarism_probability >= RED_LINE_THRESHOLD * 0.75 || match.semantic_score >= 0.58 || match.lexical_score >= 0.08,
  );
  const candidateMatches = strongMatches.length ? strongMatches : item.matches.slice(0, 1);

  for (const match of candidateMatches) {
    for (const word of tokenizeWords(match.matched_text)) {
      if (!sentenceSet.has(word) || !isHighlightablePlagiarismWord(word)) {
        continue;
      }
      overlapFrequency.set(word, (overlapFrequency.get(word) || 0) + 1);
    }
  }

  return [...overlapFrequency.entries()]
    .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
    .slice(0, 14)
    .map(([word]) => word);
}

export default function HomePage() {
  const [checkText, setCheckText] = useState(SAMPLE_TEXT);
  const [result, setResult] = useState<PlagiarismResponse | null>(null);
  const [selectedSentence, setSelectedSentence] = useState<SentenceResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState("Waiting");
  const [error, setError] = useState("");
  const inputWordCount = useMemo(() => tokenizeWords(checkText).length, [checkText]);
  const overWordLimit = inputWordCount > MAX_INPUT_WORDS;
  const longTextMode = inputWordCount >= LONG_TEXT_MODE_WORDS;

  const onCheck = async () => {
    if (!checkText.trim()) {
      setError("Please enter paragraph text to check.");
      return;
    }
    if (overWordLimit) {
      setError(`Please keep input at or below ${MAX_INPUT_WORDS} words.`);
      return;
    }

    setError("");
    setLoading(true);
    setProgress(10);
    setPhase(longTextMode ? "Long text mode: preparing..." : "Collecting sources...");

    const started = Date.now();
    const timer = window.setInterval(() => {
      const elapsed = (Date.now() - started) / 1000;
      const speed = longTextMode ? 2.2 : 4.0;
      const estimated = Math.min(97, 10 + elapsed * speed);
      setProgress((prev) => Math.max(prev, estimated));
      if (longTextMode) {
        if (elapsed < 10) {
          setPhase("Collecting websites...");
        } else if (elapsed < 20) {
          setPhase("Collecting books sources...");
        } else {
          setPhase("Running long-text detection...");
        }
      } else if (elapsed < 4) {
        setPhase("Collecting websites...");
      } else if (elapsed < 8) {
        setPhase("Collecting books sources...");
      } else {
        setPhase("Running detection models...");
      }
    }, 350);

    try {
      const data = await checkPlagiarism(checkText);
      setProgress(100);
      setPhase("Completed");
      setResult(data);
      const firstFlagged = data.results.find((r) => r.best_score >= RED_LINE_THRESHOLD) ?? data.results[0] ?? null;
      setSelectedSentence(firstFlagged);
    } catch (err) {
      setProgress(0);
      setPhase("Failed");
      if (err instanceof Error && err.message) {
        setError(err.message);
      } else {
        setError("Unable to run plagiarism check. Confirm backend is running.");
      }
    } finally {
      window.clearInterval(timer);
      setLoading(false);
    }
  };

  const selectedSourceKeys = useMemo(() => {
    if (!selectedSentence) {
      return new Set<string>();
    }
    return new Set(selectedSentence.matches.map((m) => m.source_url || m.source_title));
  }, [selectedSentence]);

  const sentenceRiskRows = useMemo(() => {
    if (!result) {
      return [] as Array<{ sentence: string; score: number }>;
    }
    return result.results
      .map((r) => ({ sentence: r.sentence, score: r.best_score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);
  }, [result]);

  const ucsyDetectedGlobal = useMemo(() => {
    if (!result) {
      return false;
    }

    const topSourceHit = result.top_accuracy_sources.some((source) => {
      if (isUcsyReference(source.source_url) || isUcsyReference(source.source_title) || isUcsyReference(source.representative_text)) {
        return true;
      }
      return (source.layer_support || []).some((layer) => (layer || "").toLowerCase().includes("keyword_triggered"));
    });
    if (topSourceHit) {
      return true;
    }

    return result.results.some((item) => item.matches.some((match) => isUcsyReference(match.source_url) || isUcsyReference(match.source_title)));
  }, [result]);

  const ucsyMatchByIndex = useMemo(() => {
    if (!result) {
      return [] as boolean[];
    }
    return result.results.map((item) =>
      item.matches.some((match) => isUcsyReference(match.source_url) || isUcsyReference(match.source_title)),
    );
  }, [result]);

  const hasAnySentenceUcsyMatch = useMemo(() => ucsyMatchByIndex.some(Boolean), [ucsyMatchByIndex]);
  const sentenceHighlightWords = useMemo(() => {
    if (!result) {
      return [] as string[][];
    }
    return result.results.map((item, idx) => {
      const plagiarismWords = buildPlagiarismHighlightWords(item);
      const ucsyWords = buildUcsyHighlightWords(
        item.sentence,
        ucsyDetectedGlobal && (ucsyMatchByIndex[idx] || !hasAnySentenceUcsyMatch),
        ucsyMatchByIndex[idx] || false,
      );
      return [...new Set([...plagiarismWords, ...ucsyWords])];
    });
  }, [result, ucsyDetectedGlobal, ucsyMatchByIndex, hasAnySentenceUcsyMatch]);

  return (
    <main>
      <section className="panel hero" style={{ marginBottom: "1rem" }}>
        <div className="hero-top">
          <h1>AI Plagiarism Detector</h1>
          <div className="pill">Multi-Model Detection</div>
        </div>
        <p className="muted">Red wavy lines show when confidence is 25% or higher. Supports up to 1500 words per request.</p>
      </section>

      <section className="panel controls" style={{ marginBottom: "1rem" }}>
        <textarea
          className="input"
          rows={8}
          value={checkText}
          onChange={(e) => setCheckText(e.target.value)}
          placeholder="Paste thesis paragraph, assignment text, or research content"
        />
        <div className={`input-meta ${overWordLimit ? "danger" : ""}`}>
          <span>
            Words: {inputWordCount} / {MAX_INPUT_WORDS}
          </span>
          <span>{longTextMode ? "Long text mode enabled for faster 900-1500 word checks." : "Standard mode."}</span>
        </div>

        <div className="action-row">
          <button className="btn" onClick={onCheck} disabled={loading || overWordLimit || inputWordCount === 0}>
            {loading ? "Checking..." : "Check Plagiarism"}
          </button>
          <div className="progress-wrap">
            <div className="progress-label">
              <span>{phase}</span>
              <strong>{Math.round(progress)}%</strong>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        </div>
        {error && <p style={{ color: "#b91c1c" }}>{error}</p>}
      </section>

      {result && (
        <section className="panel">
          <div className="result-grid">
            <div className="paragraph-pane">
              <h3 style={{ marginBottom: "0.6rem" }}>Detected Paragraph</h3>
              {result.results.map((item, idx) => (
                <FlaggedSentence
                  key={`${idx}-${item.sentence.slice(0, 20)}`}
                  item={item}
                  selected={selectedSentence?.sentence === item.sentence}
                  onSelect={setSelectedSentence}
                  redLineThreshold={RED_LINE_THRESHOLD}
                  highlightWords={sentenceHighlightWords[idx] || []}
                  disableSentenceFlag={(sentenceHighlightWords[idx] || []).length > 0}
                />
              ))}
              <div className="meta-row">
                <span>{result.total_sentences} word units</span>
                <span>Red line 25%</span>
                <span>{result.processing_ms} ms</span>
              </div>
            </div>

            <aside className="source-side">
              <div className="source-meta muted" style={{ marginBottom: "0.7rem" }}>
                Web: {result.scanned_web_sources} | Books: {result.scanned_book_sources}
              </div>

              <h3 style={{ margin: "0.7rem 0" }}>Most Accurate Sources</h3>
              {result.top_accuracy_sources.length === 0 && (
                <div className="source-item">
                  <strong>No external source evidence found</strong>
                  <p className="muted" style={{ margin: "0.25rem 0 0" }}>
                    Network/provider access may be limited. Word-unit risk scores are shown below.
                  </p>
                </div>
              )}

              {result.top_accuracy_sources.map((source, idx) => {
                const sourceKey = source.source_url || source.source_title;
                const selected = selectedSourceKeys.has(sourceKey);
                const pct = Math.max(1, Math.min(100, source.average_probability * 100));
                return (
                  <div key={`${sourceKey}-${idx}`} className={`source-item ${selected ? "source-selected" : ""}`}>
                    <div className="source-head">
                      <strong>{source.source_title}</strong>
                      <span className={`badge ${source.source_type === "book" ? "book" : "web"}`}>
                        {source.source_type.toUpperCase()}
                      </span>
                    </div>
                    <span className="muted">
                      Hits {source.matched_sentences} | Avg {(source.average_probability * 100).toFixed(1)}% | Word match{" "}
                      {(source.average_word_coverage * 100).toFixed(1)}% | Evidence{" "}
                      {(source.average_evidence_score * 100).toFixed(1)}%
                    </span>
                    {source.source_url && (
                      <a href={source.source_url} target="_blank" rel="noreferrer" className="muted" style={{ fontSize: "0.8rem" }}>
                        {source.source_url}
                      </a>
                    )}
                    <div className="source-bar-track">
                      <div className="source-bar-fill" style={{ width: `${pct}%` }} />
                    </div>
                    <p style={{ margin: "0.35rem 0 0", fontSize: "0.9rem" }}>{source.representative_text}</p>
                  </div>
                );
              })}

              {sentenceRiskRows.length > 0 && (
                <>
                  <h3 style={{ margin: "0.9rem 0 0.6rem" }}>Word-Unit Risk (Internal Model)</h3>
                  {sentenceRiskRows.map((row, idx) => (
                    <div key={`${idx}-${row.sentence.slice(0, 20)}`} className="source-item">
                      <span className="muted">Rank #{idx + 1}</span>
                      <p style={{ margin: "0.2rem 0 0", fontSize: "0.9rem" }}>{row.sentence}</p>
                    </div>
                  ))}
                </>
              )}
            </aside>
          </div>
        </section>
      )}
    </main>
  );
}
