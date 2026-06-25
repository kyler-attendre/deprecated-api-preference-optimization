import contextlib
import gzip
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/data/models"))


def model_path_for_env(model_key: str, *relative_parts: str) -> str:
    env_key = f"MODEL_PATH_{model_key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
    return str(MODEL_ROOT.joinpath(*relative_parts))


MODEL_REGISTRY = {
    "starcoder2_3b": {
        "path": model_path_for_env("starcoder2_3b", "StarCoder", "starcoder2-3b"),
        "max_length": 384,
    },
    "starcoder2_7b": {
        "path": model_path_for_env("starcoder2_7b", "StarCoder", "starcoder2-7b"),
        "max_length": 384,
    },
    "starcoder2_15b": {
        "path": model_path_for_env("starcoder2_15b", "StarCoder", "starcoder2-15b"),
        "max_length": 256,
    },
    "deepseek_coder_6_7b_instruct": {
        "path": model_path_for_env("deepseek_coder_6_7b_instruct", "deepseek-ai", "deepseek-coder-6.7b-instruct"),
        "max_length": 384,
    },
    "qwen2_5_coder_3b_instruct": {
        "path": model_path_for_env("qwen2_5_coder_3b_instruct", "Qwen", "Qwen2.5-Coder-3B-Instruct"),
        "max_length": 384,
    },
    "qwen2_5_coder_7b_instruct": {
        "path": model_path_for_env("qwen2_5_coder_7b_instruct", "Qwen", "Qwen2.5-Coder-7B-Instruct"),
        "max_length": 384,
    },
    "qwen2_5_coder_14b_instruct": {
        "path": model_path_for_env("qwen2_5_coder_14b_instruct", "Qwen", "Qwen2.5-Coder-14B-Instruct"),
        "max_length": 256,
    },
}


@dataclass(frozen=True)
class ContrastPair:
    library: str
    row_id: str
    positive_text: str
    negative_text: str
    replacement_form: str
    deprecated_form: str


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    return [value] if value else []


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def safe_model_label(model_name_or_path: str) -> str:
    name = Path(model_name_or_path).name if "/" in model_name_or_path else model_name_or_path.split("/")[-1]
    name = name.replace("-local", "")
    name = name.replace(".", "_").replace("-", "_")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return name.strip("_").lower()


def alias_forms(api_name: str) -> List[str]:
    api_name = str(api_name).strip()
    aliases = {api_name}
    if api_name.startswith("torch.nn.functional."):
        aliases.add("F." + api_name.split(".")[-1])
    if api_name.startswith("tensorflow."):
        aliases.add("tf." + api_name.split(".", 1)[1])
    if api_name.startswith("numpy."):
        aliases.add("np." + api_name.split(".", 1)[1])
    if api_name.startswith("pandas."):
        aliases.add("pd." + api_name.split(".", 1)[1])
    return sorted(alias for alias in aliases if alias)


def first_alias_hit(text: str, api_name: str) -> Optional[str]:
    for alias in alias_forms(api_name):
        if alias and alias in text:
            return alias
    return None


def deprecated_alias_matching_replacement(
    replacement_api: str,
    replacement_form: str,
    deprecated_api: str,
) -> str:
    if replacement_form == replacement_api:
        return deprecated_api
    if replacement_api.startswith("tensorflow.") and replacement_form.startswith("tf."):
        return "tf." + deprecated_api.split(".", 1)[1]
    if replacement_api.startswith("torch.nn.functional.") and replacement_form.startswith("F."):
        return "F." + deprecated_api.split(".")[-1]
    if replacement_api.startswith("numpy.") and replacement_form.startswith("np."):
        return "np." + deprecated_api.split(".", 1)[1]
    if replacement_api.startswith("pandas.") and replacement_form.startswith("pd."):
        return "pd." + deprecated_api.split(".", 1)[1]
    return deprecated_api


def build_contrast_pair(row: Dict) -> Optional[ContrastPair]:
    target = str(row.get("target") or row.get("reference") or "").strip()
    prompt = str(row.get("version_prompt") or "")
    replacement_api = str(row.get("replacement_api") or "").strip()
    deprecated_apis = ensure_list(row.get("deprecated_api"))

    if not prompt or not target or not replacement_api or not deprecated_apis:
        return None

    replacement_form = first_alias_hit(target, replacement_api)
    if not replacement_form:
        return None

    deprecated_form = deprecated_alias_matching_replacement(
        replacement_api=replacement_api,
        replacement_form=replacement_form,
        deprecated_api=deprecated_apis[0],
    )
    negative_target = target.replace(replacement_form, deprecated_form, 1)
    if negative_target == target:
        return None

    separator = "" if prompt.endswith(("\n", " ", "\t")) else "\n"
    return ContrastPair(
        library=str(row.get("library") or "unknown"),
        row_id=str(row.get("id") or ""),
        positive_text=prompt + separator + target,
        negative_text=prompt + separator + negative_target,
        replacement_form=replacement_form,
        deprecated_form=deprecated_form,
    )


def build_contrast_pairs(rows: Iterable[Dict]) -> List[ContrastPair]:
    pairs = []
    for row in rows:
        pair = build_contrast_pair(row)
        if pair is not None:
            pairs.append(pair)
    return pairs


def parse_layer_spec(spec: str) -> List[int]:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("empty layer spec")
    layers = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            left, right = chunk.split(":", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"invalid descending layer range: {chunk}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(chunk))
    if not layers:
        raise ValueError("empty layer spec")
    return sorted(layers)


def get_decoder_layers(model):
    base = getattr(model, "base_model", model)
    candidate_roots = [
        getattr(base, "model", None),
        getattr(getattr(base, "model", None), "model", None),
        getattr(base, "transformer", None),
        base,
    ]
    for root in candidate_roots:
        if root is None:
            continue
        if hasattr(root, "layers"):
            return root.layers
        if hasattr(root, "h"):
            return root.h
        if hasattr(root, "gpt_neox") and hasattr(root.gpt_neox, "layers"):
            return root.gpt_neox.layers
    raise ValueError(f"Cannot locate decoder layers for model type {type(model).__name__}")


class ActivationSteering:
    def __init__(
        self,
        model,
        vectors: Dict[str, Dict[int, object]],
        layers: Sequence[int],
        coefficient: float,
        default_library: Optional[str] = None,
    ):
        self.model = model
        self.vectors = vectors
        self.layers = list(layers)
        self.coefficient = float(coefficient)
        self.default_library = default_library
        self.current_library: Optional[str] = default_library
        self.current_multiplier: float = 1.0
        self.active: bool = False
        self.handles = []

    def install(self):
        decoder_layers = get_decoder_layers(self.model)
        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= len(decoder_layers):
                continue
            self.handles.append(decoder_layers[layer_idx].register_forward_hook(self._make_hook(layer_idx)))
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @contextlib.contextmanager
    def use(self, library: Optional[str], multiplier: float = 1.0):
        old_library = self.current_library
        old_multiplier = self.current_multiplier
        old_active = self.active
        self.current_library = library or self.default_library
        self.current_multiplier = float(multiplier)
        self.active = self.current_library is not None
        try:
            yield
        finally:
            self.current_library = old_library
            self.current_multiplier = old_multiplier
            self.active = old_active

    @contextlib.contextmanager
    def disabled(self):
        old_active = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = old_active

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if not self.active or not self.current_library:
                return output
            library_vectors = self.vectors.get(self.current_library)
            if not library_vectors:
                return output
            vector = library_vectors.get(layer_idx)
            if vector is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            vector = vector.to(device=hidden.device, dtype=hidden.dtype)
            steered_hidden = hidden + (self.coefficient * self.current_multiplier) * vector.view(1, 1, -1)
            if isinstance(output, tuple):
                return (steered_hidden,) + output[1:]
            return steered_hidden

        return hook


def load_vector_file(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu")
    raw_vectors = payload["vectors"]
    vectors: Dict[str, Dict[int, object]] = {}
    for library, layer_map in raw_vectors.items():
        vectors[library] = {int(layer): tensor for layer, tensor in layer_map.items()}
    payload["vectors"] = vectors
    return payload


def official_mbpp_prompt(examples: Sequence[Dict], problem: Dict) -> str:
    parts = []
    for item in list(examples) + [problem]:
        tests = "\n".join(item.get("test_list") or [])
        task_prompt = item.get("prompt", item.get("text"))
        if task_prompt is None:
            raise KeyError("MBPP item must contain either 'prompt' or 'text'")
        parts.append(
            "You are an expert Python programmer, and here is your task: "
            f"{task_prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
        )
        if item is not problem:
            parts.append(str(item["code"]).rstrip() + "\n[DONE]\n")
    return "".join(parts)


def iter_official_mbpp_test_rows(path: Path) -> Iterator[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {int(item["task_id"]): item for item in data}
    few_shot = [by_id[2], by_id[3], by_id[4]]
    for task_id in sorted(by_id):
        if 11 <= task_id <= 510:
            item = by_id[task_id]
            yield {
                "task_id": f"mbpp/{task_id}",
                "prompt": official_mbpp_prompt(few_shot, item),
                "tests": list(item.get("test_list") or []),
                "canonical_solution": item.get("code", ""),
            }


def read_humaneval_jsonl(path: Path) -> Dict[str, Dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    problems = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                problems[item["task_id"]] = item
    return problems
