#!/usr/bin/env python3
"""Convert the oral-cancer biomarker table into text prompts for LLM fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_TEXT = {0: "Control", 1: "Cancer"}


def build_prompt(row: pd.Series) -> str:
    header = "Determine whether the saliva biomarker profile indicates oral cancer."
    body_lines = [f"biomarker_{idx}: {row[f'biomarker_{idx}']:.3f}" for idx in range(1, 9)]
    body = "\n".join(body_lines)
    prompt = (
        "### Instruction\n"
        f"{header}\n"
        "### Biomarkers\n"
        f"{body}\n"
        "### Response"
    )
    return prompt


def iter_records(df: pd.DataFrame) -> Iterable[str]:
    for _, row in df.iterrows():
        prompt = build_prompt(row)
        completion = LABEL_TEXT[int(row["label"])]
        yield json.dumps({"prompt": prompt, "completion": completion})


def export_split(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in iter_records(df):
            handle.write(record + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/oral_cancer.csv"),
        help="Path to the numeric biomarker dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/llm"),
        help="Directory where JSONL splits will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for the evaluation split.",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.1,
        help="Fraction of the remaining training set used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    args = parser.parse_args()

    columns = [f"biomarker_{idx}" for idx in range(1, 9)] + ["label"]
    df = pd.read_csv(args.data_path, header=None, names=columns)

    train_df, eval_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label"],
        random_state=args.seed,
    )
    train_df, valid_df = train_test_split(
        train_df,
        test_size=args.valid_size,
        stratify=train_df["label"],
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_split(train_df.reset_index(drop=True), args.output_dir / "train.jsonl")
    export_split(valid_df.reset_index(drop=True), args.output_dir / "valid.jsonl")
    export_split(eval_df.reset_index(drop=True), args.output_dir / "test.jsonl")

    summary = {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(eval_df),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
