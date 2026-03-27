"use client";

import type { SentenceResult } from "@/lib/api";

interface Props {
  item: SentenceResult;
  selected: boolean;
  onSelect: (item: SentenceResult) => void;
  redLineThreshold: number;
  highlightWords?: string[];
  disableSentenceFlag?: boolean;
}

function normalizeWord(token: string): string {
  return token.toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/gi, "");
}

export default function FlaggedSentence({
  item,
  selected,
  onSelect,
  redLineThreshold,
  highlightWords = [],
  disableSentenceFlag = false,
}: Props) {
  const highlightSet = new Set(highlightWords.map((word) => normalizeWord(word)).filter(Boolean));
  const showSentenceRedLine = !disableSentenceFlag && item.best_score >= redLineThreshold;
  const tokens = item.sentence.split(/(\s+)/);

  return (
    <button
      type="button"
      className={`sentence sentence-btn ${showSentenceRedLine ? "flagged" : ""} ${selected ? "selected" : ""}`}
      onClick={() => onSelect(item)}
      title={
        showSentenceRedLine || highlightSet.size > 0
          ? "Click to highlight matching source on the right"
          : "No high-risk source found"
      }
    >
      {tokens.map((token, idx) => {
        if (!token.trim()) {
          return token;
        }
        const clean = normalizeWord(token);
        const isWordFlagged = clean.length > 0 && highlightSet.has(clean);
        return (
          <span key={`${idx}-${token}`} className={isWordFlagged ? "sentence-word flagged" : "sentence-word"}>
            {token}
          </span>
        );
      })}
    </button>
  );
}
