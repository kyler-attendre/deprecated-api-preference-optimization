import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VERSION_DICT = {
    "numpy": "# numpy 1.26.4\nimport numpy\n",
    "pandas": "# pandas 2.2.2\nimport pandas\n",
    "pytorch": "# pytorch 2.3.0\nimport torch\n",
    "scipy": "# scipy 1.13.0\nimport scipy\n",
    "seaborn": "# seaborn 0.13.2\nimport seaborn\n",
    "sklearn": "# scikit-learn 1.5.0\nimport sklearn\n",
    "tensorflow": "# tensorflow 2.16.1\nimport tensorflow\n",
    "transformers": "# transformers 4.40.2\nimport transformers\n",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_model_name(model_name: str) -> str:
    return model_name.replace("-local", "")


def first_prediction_text(sample: Dict, field_name: str = "probing predictions") -> str:
    preds = sample.get(field_name, [])
    if not preds:
        return ""
    first = preds[0]
    if isinstance(first, list) and first:
        return first[0] or ""
    if isinstance(first, tuple) and first:
        return first[0] or ""
    return ""


def version_prompt(lib: str, probing_input: str) -> str:
    return VERSION_DICT.get(lib, f"# {lib}\n") + probing_input


def split_name(stable_key: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    digest = hashlib.md5(stable_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def build_record_id(model: str, library: str, sample: Dict) -> str:
    key = sample.get("function") or f"{sample.get('probing input', '')}|||{sample.get('reference', '')}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"{safe_model_name(model)}::{library}::{digest}"


def normalize_whitespace(text: str) -> str:
    return text.replace("\r\n", "\n").strip("\n")


def replacement_target_from_bad2good(sample: Dict) -> str:
    return normalize_whitespace(
        first_prediction_text({"probing predictions": sample.get("version_control_predictions", [])})
    )


def baseline_bad_output(sample: Dict) -> str:
    return normalize_whitespace(
        first_prediction_text({"probing predictions": sample.get("probing_predictions", [])})
    )


def make_sft_record(
    sample: Dict,
    model: str,
    sample_type: str,
    target: str,
    source_file: str,
) -> Optional[Dict]:
    target = normalize_whitespace(target)
    if not target:
        return None

    library = sample["library"]
    probing_input = sample["probing input"]
    record_id = build_record_id(model, library, sample)
    return {
        "id": record_id,
        "model": safe_model_name(model),
        "library": library,
        "version_prompt": version_prompt(library, probing_input),
        "probing_input": probing_input,
        "target": target,
        "reference": normalize_whitespace(sample.get("reference", "")),
        "deprecated_api": sample.get("deprecated api", []),
        "replacement_api": sample.get("replacement api", ""),
        "category": sample.get("category", ""),
        "sample_type": sample_type,
        "source_file": source_file,
    }


def make_preference_record(sample: Dict, model: str, source_file: str) -> Optional[Dict]:
    chosen = replacement_target_from_bad2good(sample)
    rejected = baseline_bad_output(sample)
    if not chosen or not rejected or chosen == rejected:
        return None

    library = sample["library"]
    probing_input = sample["probing input"]
    record_id = build_record_id(model, library, sample)
    return {
        "id": record_id,
        "model": safe_model_name(model),
        "library": library,
        "version_prompt": version_prompt(library, probing_input),
        "probing_input": probing_input,
        "chosen": chosen,
        "rejected": rejected,
        "deprecated_api": sample.get("deprecated api", []),
        "replacement_api": sample.get("replacement api", ""),
        "category": sample.get("category", ""),
        "sample_type": "bad2good_preference",
        "source_file": source_file,
    }


def is_up_to_dated(sample: Dict) -> bool:
    return sample.get("category") == "up-to-dated"


def iter_result_files(root: Path) -> Iterable[Tuple[str, str, Path]]:
    for library_dir in sorted(root.iterdir()):
        if not library_dir.is_dir():
            continue
        for model_dir in sorted(library_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            result_file = model_dir / "predictions-linelevel-maxlen50-beam1.json"
            if result_file.exists():
                yield library_dir.name, model_dir.name, result_file
