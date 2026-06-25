#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from safetensors.torch import load_file


LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.+layers\.(?P<layer>\d+)\.self_attn\.(?P<module>q_proj|k_proj|v_proj|o_proj))\.(?P<kind>lora_A|lora_B)\.weight$"
)
MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_lora_module_key(key: str) -> Optional[Tuple[int, str, str]]:
    match = LORA_KEY_RE.match(key)
    if not match:
        return None
    return int(match.group("layer")), match.group("module"), match.group("prefix")


def compute_effective_delta(*, lora_a: torch.Tensor, lora_b: torch.Tensor, lora_alpha: float, lora_r: int) -> torch.Tensor:
    scaling = float(lora_alpha) / float(lora_r)
    return (lora_b.float() @ lora_a.float()) * scaling


def compute_tensor_stats(tensor: torch.Tensor, epsilon: float, max_quantile_elements: int = 200000) -> Dict[str, float]:
    abs_tensor = tensor.abs().reshape(-1)
    quantile_source = abs_tensor
    if abs_tensor.numel() > max_quantile_elements:
        stride = max(1, abs_tensor.numel() // max_quantile_elements)
        quantile_source = abs_tensor[::stride]
    return {
        "fro_norm": torch.linalg.norm(tensor).item(),
        "mean_abs": abs_tensor.mean().item(),
        "q50_abs": torch.quantile(quantile_source, 0.50).item(),
        "q90_abs": torch.quantile(quantile_source, 0.90).item(),
        "q95_abs": torch.quantile(quantile_source, 0.95).item(),
        "q99_abs": torch.quantile(quantile_source, 0.99).item(),
        "max_abs": abs_tensor.max().item(),
        "sparsity_below_epsilon": abs_tensor.le(epsilon).float().mean().item(),
    }


def read_adapter_config(adapter_dir: Path) -> Dict:
    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_adapter(adapter_dir: Path, *, epsilon: float) -> Tuple[pd.DataFrame, Dict]:
    config = read_adapter_config(adapter_dir)
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    rows: List[Dict] = []

    prefixes: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        parsed = parse_lora_module_key(key)
        if parsed is None:
            continue
        layer_idx, module_name, prefix = parsed
        slot = prefixes.setdefault(prefix, {"layer": layer_idx, "module": module_name})
        if ".lora_A." in key:
            slot["lora_A"] = value
        elif ".lora_B." in key:
            slot["lora_B"] = value

    for prefix, slot in sorted(prefixes.items(), key=lambda item: (item[1]["layer"], MODULE_ORDER.index(item[1]["module"]))):
        delta = compute_effective_delta(
            lora_a=slot["lora_A"],
            lora_b=slot["lora_B"],
            lora_alpha=config["lora_alpha"],
            lora_r=config["r"],
        )
        stats = compute_tensor_stats(delta, epsilon=epsilon)
        rows.append(
            {
                "layer": slot["layer"],
                "module": slot["module"],
                "prefix": prefix,
                "rows": delta.shape[0],
                "cols": delta.shape[1],
                **stats,
            }
        )

    frame = pd.DataFrame(rows)
    aggregate = {
        "adapter_dir": str(adapter_dir),
        "base_model_name_or_path": config["base_model_name_or_path"],
        "lora_alpha": config["lora_alpha"],
        "lora_r": config["r"],
        "target_modules": config["target_modules"],
        "epsilon": epsilon,
        "num_modules": int(len(frame)),
        "mean_fro_norm": float(frame["fro_norm"].mean()),
        "mean_mean_abs": float(frame["mean_abs"].mean()),
        "mean_sparsity_below_epsilon": float(frame["sparsity_below_epsilon"].mean()),
    }
    return frame, aggregate


def plot_heatmap(frame: pd.DataFrame, *, metric: str, title: str, output_path: Path) -> None:
    pivot = frame.pivot(index="layer", columns="module", values=metric).reindex(columns=MODULE_ORDER)
    fig, ax = plt.subplots(figsize=(6, 10))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Module")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze effective LoRA delta weights and plot per-layer heatmaps.")
    parser.add_argument("--adapter", action="append", required=True, help="Adapter in LABEL=DIR format.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {"epsilon": args.epsilon, "variants": {}}

    for item in args.adapter:
        if "=" not in item:
            raise SystemExit(f"Expected LABEL=DIR, got {item}")
        label, raw_dir = item.split("=", 1)
        label = label.strip()
        adapter_dir = Path(raw_dir.strip()).expanduser().resolve()
        frame, aggregate = summarize_adapter(adapter_dir, epsilon=args.epsilon)
        frame.to_csv(output_dir / f"{label}_module_stats.csv", index=False)
        plot_heatmap(
            frame,
            metric="fro_norm",
            title=f"{label}: effective delta Frobenius norm",
            output_path=output_dir / f"{label}_fro_norm_heatmap.png",
        )
        plot_heatmap(
            frame,
            metric="mean_abs",
            title=f"{label}: effective delta mean |w|",
            output_path=output_dir / f"{label}_mean_abs_heatmap.png",
        )
        plot_heatmap(
            frame,
            metric="sparsity_below_epsilon",
            title=f"{label}: sparsity below epsilon",
            output_path=output_dir / f"{label}_sparsity_heatmap.png",
        )
        summary_payload["variants"][label] = aggregate

    with (output_dir / "delta_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
