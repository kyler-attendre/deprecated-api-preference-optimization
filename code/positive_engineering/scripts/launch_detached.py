#!/usr/bin/env python3
"""
Launch per-library CER-DPO jobs as truly detached subprocesses.
Uses os.setsid() + subprocess.Popen so processes survive after this script exits.
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_DIR / "scripts" / "run_per_library_cerdpo.sh"
TAG = f"per_library_cerdpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EPOCHS = "3"
LOG_DIR = PROJECT_DIR / "output" / TAG / "launcher_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONDA_ACTIVATE = (
    "source /opt/conda/etc/profile.d/conda.sh && "
    "conda activate lkl_llm && "
)

# (gpu, model_key) assignments
JOBS = [
    # GPU 1: 两个小模型顺序（写进一个 wrapper）
    # GPU 2-6: 各自独立
    (2, "starcoder2_7b"),
    (3, "deepseek_coder_6_7b_instruct"),
    (4, "qwen2_5_coder_7b_instruct"),
    (5, "starcoder2_15b"),
    (6, "qwen2_5_coder_14b_instruct"),
]

def launch(gpu, model_key, log_path, cmd):
    """Launch a detached process that survives after this Python script exits."""
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=logf,
            stderr=logf,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            preexec_fn=os.setsid,   # 新 session，脱离父进程组
        )
    print(f"  GPU {gpu:1d} → {model_key:<35s}  PID={proc.pid}  log={log_path.name}")
    return proc.pid

pids = []

# GPU 1: starcoder2_3b → qwen_3b (顺序)
wrapper = LOG_DIR / "gpu1_seq.sh"
wrapper.write_text(
    f"#!/usr/bin/env bash\n"
    f"{CONDA_ACTIVATE}"
    f"cd {PROJECT_DIR}\n"
    f"echo \"[gpu1] starcoder2_3b start: $(date)\"\n"
    f"bash {SCRIPT} --gpu 1 --model-key starcoder2_3b "
    f"  --mode all --tag {TAG} --epochs {EPOCHS} "
    f"  >> {LOG_DIR}/starcoder2_3b_gpu1.log 2>&1\n"
    f"echo \"[gpu1] qwen_3b start: $(date)\"\n"
    f"bash {SCRIPT} --gpu 1 --model-key qwen2_5_coder_3b_instruct "
    f"  --mode all --tag {TAG} --epochs {EPOCHS} "
    f"  >> {LOG_DIR}/qwen3b_gpu1.log 2>&1\n"
    f"echo \"[gpu1] all done: $(date)\"\n"
)
wrapper.chmod(0o755)
pid = launch(1, "starcoder2_3b → qwen_3b", LOG_DIR / "gpu1_wrapper.log",
             f"bash {wrapper}")
pids.append((1, "gpu1_seq", pid))

# GPU 2-6: single models
for gpu, model_key in JOBS:
    log = LOG_DIR / f"{model_key}_gpu{gpu}.log"
    cmd = (
        f"{CONDA_ACTIVATE}"
        f"cd {PROJECT_DIR} && "
        f"bash {SCRIPT} --gpu {gpu} --model-key {model_key} "
        f"  --mode all --tag {TAG} --epochs {EPOCHS}"
    )
    pid = launch(gpu, model_key, log, cmd)
    pids.append((gpu, model_key, pid))

print()
print(f"Tag:  {TAG}")
print(f"Logs: {LOG_DIR}")
print(f"Monitor: tail -f {LOG_DIR}/*.log")
print(f"         watch -n10 nvidia-smi")

# 写入 PID 文件方便追踪
pid_file = LOG_DIR / "pids.txt"
with open(pid_file, "w") as f:
    for gpu, model, pid in pids:
        f.write(f"GPU{gpu}\t{model}\t{pid}\n")
print(f"PIDs: {pid_file}")
