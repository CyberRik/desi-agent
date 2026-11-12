#!/usr/bin/env bash
# [R] R2 on Kaggle (GPU T4 x2, Internet ON). Run from a notebook cell:  !bash scripts/data_kaggle_generate.sh [smoke|full]
# Needs Kaggle secret HF_TOKEN (write access to the ToolW org) only for the final upload step.
set -euo pipefail
MODE=${1:-smoke}
MODEL=${TEACHER_MODEL:-Qwen/Qwen3-14B-AWQ}
OUT=data/raw/teacher.jsonl

pip install -q "vllm>=0.8" openai >/dev/null
pip install -q -e . >/dev/null

if ! curl -sf localhost:8000/health >/dev/null; then
  nohup vllm serve "$MODEL" --tensor-parallel-size 2 --dtype half --max-model-len 4096 \
        --gpu-memory-utilization 0.92 --max-num-seqs 64 --disable-log-requests > vllm.log 2>&1 &
  echo "waiting for vLLM ($MODEL)..."
  for i in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 10; done
  curl -sf localhost:8000/health >/dev/null || { tail -50 vllm.log; exit 1; }
fi

if [ "$MODE" = smoke ]; then
  python -m desi_agent.data.generate --model "$MODEL" --out data/raw/smoke.jsonl --limit 40 --concurrency 32
  python scripts/validate_data.py data/raw/smoke.jsonl
  python - <<'EOF'
import json
for l in list(open("data/raw/smoke.jsonl", encoding="utf-8"))[:8]:
    r = json.loads(l); print(f"[{r['lang']}/{r['category']}] USER: {r['messages'][0]['content']}\n    BOT: {r['messages'][-1]['content']}\n")
EOF
  echo "rejected:"; wc -l data/raw/smoke.jsonl.rejected.jsonl || true
else
  python -m desi_agent.data.generate --model "$MODEL" --out "$OUT" --concurrency 48
  python scripts/validate_data.py "$OUT"
  if [ -n "${HF_TOKEN:-}" ]; then
    python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("ToolW/desi-bank-toolcalls", repo_type="dataset", exist_ok=True)
for f in ["teacher.jsonl", "teacher.jsonl.rejected.jsonl", "teacher.jsonl.manifest.json"]:
    api.upload_file(path_or_fileobj=f"data/raw/{f}", path_in_repo=f"raw/{f}", repo_id="ToolW/desi-bank-toolcalls", repo_type="dataset")
print("uploaded raw teacher data")
EOF
  fi
fi
