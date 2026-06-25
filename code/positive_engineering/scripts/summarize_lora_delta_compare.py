#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def summarize_variant(frame: pd.DataFrame) -> dict:
    top = frame.sort_values("fro_norm", ascending=False).head(12)
    return {
        "num_modules": int(len(frame)),
        "mean_fro_norm": float(frame["fro_norm"].mean()),
        "mean_mean_abs": float(frame["mean_abs"].mean()),
        "mean_sparsity_below_epsilon": float(frame["sparsity_below_epsilon"].mean()),
        "top_modules_by_fro_norm": top[
            ["layer", "module", "fro_norm", "mean_abs", "sparsity_below_epsilon"]
        ].to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DPO vs anchored LoRA delta statistics.")
    parser.add_argument("--dpo-csv", type=Path, required=True)
    parser.add_argument("--anchored-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dpo = pd.read_csv(args.dpo_csv.resolve())
    anchored = pd.read_csv(args.anchored_csv.resolve())

    payload = {
        "dpo": summarize_variant(dpo),
        "anchored_dpo": summarize_variant(anchored),
        "delta_mean_fro_norm": float(anchored["fro_norm"].mean() - dpo["fro_norm"].mean()),
        "delta_mean_mean_abs": float(anchored["mean_abs"].mean() - dpo["mean_abs"].mean()),
        "delta_mean_sparsity_below_epsilon": float(
            anchored["sparsity_below_epsilon"].mean() - dpo["sparsity_below_epsilon"].mean()
        ),
    }

    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
