import axios from "axios";

const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";
const configuredTimeout = Number.parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT_MS || "180000", 10);
const requestTimeoutMs = Number.isFinite(configuredTimeout) && configuredTimeout > 0 ? configuredTimeout : 180000;

export const apiClient = axios.create({
  baseURL,
  headers: { "Content-Type": "application/json" },
  timeout: requestTimeoutMs,
});

export interface MatchResult {
  sentence: string;
  source_title: string;
  source_type: "web" | "book" | string;
  source_url: string;
  matched_text: string;
  tfidf_score?: number;
  lexical_score: number;
  semantic_score: number;
  overlap_score: number;
  containment_score: number;
  word_coverage: number;
  char_similarity: number;
  evidence_score: number;
  plagiarism_probability: number;
}

export interface SentenceResult {
  sentence: string;
  flagged: boolean;
  best_score: number;
  matches: MatchResult[];
}

export interface SourceAccuracyResult {
  source_title: string;
  source_type: "web" | "book" | string;
  source_url: string;
  best_probability: number;
  average_probability: number;
  average_word_coverage: number;
  average_evidence_score: number;
  matched_sentences: number;
  representative_text: string;
  layer_support?: string[];
}

export interface PlagiarismResponse {
  overall_similarity_percent: number;
  flagged_sentences: number;
  total_sentences: number;
  scanned_web_sources: number;
  scanned_book_sources: number;
  processing_ms: number;
  top_accuracy_sources: SourceAccuracyResult[];
  results: SentenceResult[];
}

export async function checkPlagiarism(text: string): Promise<PlagiarismResponse> {
  try {
    const response = await apiClient.post<PlagiarismResponse>("/plagiarism/check", { text });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === "ECONNABORTED") {
        throw new Error("Request timed out while analyzing text. Backend supports up to 1500 words; check backend/Ollama responsiveness.");
      }
      const detail = typeof error.response?.data?.detail === "string" ? error.response?.data?.detail : "";
      if (detail) {
        throw new Error(detail);
      }
    }
    throw error;
  }
}
