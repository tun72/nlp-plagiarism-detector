from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.nlp.features import containment_score, token_overlap
from app.nlp.preprocessing import normalize_sentence
from app.nlp.semantic_engine import SemanticEngine
from app.nlp.tfidf_engine import TfidfEngine


def build_features(df: pd.DataFrame, enable_semantic: bool, model_name: str) -> tuple[list[list[float]], list[int]]:
    pairs = []
    labels = []

    a_texts = [normalize_sentence(str(v)) for v in df["text_a"].tolist()]
    b_texts = [normalize_sentence(str(v)) for v in df["text_b"].tolist()]

    tfidf = TfidfEngine()
    tfidf.fit(b_texts)

    semantic = SemanticEngine(model_name if enable_semantic else "")
    semantic.fit(b_texts)

    for idx, (a_norm, b_norm, label) in enumerate(zip(a_texts, b_texts, df["label"].tolist(), strict=False)):
        lex = tfidf.top_k(a_norm, k=1)
        lexical_score = float(lex[0].score) if lex else 0.0
        semantic_score = semantic.similarity_to_index(a_norm, idx)
        overlap = token_overlap(a_norm, b_norm)
        containment = containment_score(a_norm, b_norm)

        pairs.append([lexical_score, semantic_score, overlap, containment])
        labels.append(int(label))

    return pairs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a plagiarism probability calibrator model.")
    parser.add_argument("--input", required=True, help="CSV path with columns: text_a,text_b,label")
    parser.add_argument("--output", default="./models/plagiarism_calibrator.joblib")
    parser.add_argument("--enable-semantic", action="store_true")
    parser.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"text_a", "text_b", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X, y = build_features(df, enable_semantic=args.enable_semantic, model_name=args.model_name)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, args.output)
    print(f"Saved calibrator model to: {args.output}")


if __name__ == "__main__":
    main()
