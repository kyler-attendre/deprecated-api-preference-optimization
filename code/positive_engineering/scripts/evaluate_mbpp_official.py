#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_one(sample: Dict, timeout: float) -> Tuple[str, bool, str]:
    code = sample.get("completion", "")
    tests = sample.get("tests", [])
    task_id = sample.get("task_id", "")

    def target(queue):
        namespace = {}
        try:
            exec(code + "\n" + "\n".join(tests), namespace)
            queue.put((True, "passed"))
        except BaseException as exc:
            queue.put((False, f"{type(exc).__name__}: {exc}"))

    queue = mp.Queue()
    proc = mp.Process(target=target, args=(queue,))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return task_id, False, "timed out"
    if queue.empty():
        return task_id, False, "no result"
    passed, result = queue.get()
    return task_id, bool(passed), result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Google MBPP samples generated with official test split/prompt.")
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    samples = load_jsonl(args.samples_jsonl)
    by_task = {}
    for sample in samples:
        by_task.setdefault(sample["task_id"], []).append(sample)

    task_results = []
    for task_id, task_samples in sorted(by_task.items()):
        # pass@1 only: use the first generated sample for each task.
        _task_id, passed, result = run_one(task_samples[0], args.timeout)
        task_results.append({"task_id": task_id, "passed": passed, "result": result})

    passed_count = sum(int(item["passed"]) for item in task_results)
    summary = {
        "samples_jsonl": str(args.samples_jsonl),
        "tasks": len(task_results),
        "pass@1": passed_count / len(task_results) if task_results else 0.0,
        "timeout": args.timeout,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    results_path = args.results_jsonl or args.output_json.with_suffix(".results.jsonl")
    with results_path.open("w", encoding="utf-8") as f:
        for item in task_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    mp.set_start_method("fork")
    with tempfile.TemporaryDirectory():
        main()
