from __future__ import annotations

import argparse
from pathlib import Path

from app.nlp.semantic_engine import SemanticEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HF model and optional calibrator before starting server.")
    parser.add_argument("--semantic-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--train-csv", default="", help="Optional CSV (text_a,text_b,label) to train calibrator.")
    parser.add_argument("--calibrator-output", default="./models/plagiarism_calibrator.joblib")
    args = parser.parse_args()

    print(f"Preloading semantic model: {args.semantic_model}")
    SemanticEngine(args.semantic_model).fit(["warmup"])
    print("Semantic model is ready.")

    if args.train_csv:
        from scripts.train_calibrator import build_features
        import joblib
        import pandas as pd
        from sklearn.linear_model import LogisticRegression

        csv_path = Path(args.train_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Training CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"text_a", "text_b", "label"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in training CSV: {sorted(missing)}")

        print("Training calibrator model...")
        X, y = build_features(df, enable_semantic=True, model_name=args.semantic_model)
        model = LogisticRegression(max_iter=1200)
        model.fit(X, y)

        output_path = Path(args.calibrator_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        print(f"Saved calibrator: {output_path}")


if __name__ == "__main__":
    main()
