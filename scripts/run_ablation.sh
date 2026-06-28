#!/usr/bin/env bash
# run_ablation.sh — RAG 检索消融实验：逐个关闭组件，对比 Hit Rate / MRR / 延迟。
#
# 5 组配置：
#   1. full:        BM25 + Dense + RRF + Rerank（全链路）
#   2. no_rerank:   BM25 + Dense + RRF（去掉 LLM Rerank）
#   3. no_fusion:   Dense only（去掉 BM25 + RRF，纯向量检索）
#   4. no_dense:    BM25 only（去掉向量检索，纯稀疏检索）
#   5. agentic:     Agentic RAG（多跳检索 + 反思 + 查询改写）
#
# 用法：
#   cd MODULAR-RAG-MCP-SERVER
#   source .venv/bin/activate
#   bash scripts/run_ablation.sh
#
# 输出：ablation_result.json（含 5 组配置的 metrics 对比）

set -euo pipefail

cd "$(dirname "$0")/.."

RESULT_FILE="ablation_result.json"
echo "[" > "$RESULT_FILE"

run_config() {
    local label="$1"
    local env_vars="$2"
    local first="$3"

    if [ "$first" != "true" ]; then
        echo "," >> "$RESULT_FILE"
    fi

    echo "=== Running: $label ===" >&2
    local output
    output=$(env $env_vars python3 scripts/evaluate.py --json 2>/dev/null)
    local metrics
    metrics=$(echo "$output" | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d.get('metrics', {})
print(json.dumps({
    'pipeline': d.get('pipeline', '$label'),
    'hit_rate': m.get('hit_rate', 0),
    'mrr': m.get('mrr', 0),
    'avg_latency_ms': m.get('avg_latency_ms', 0),
    'total_queries': d.get('total_queries', 0)
}))
")
    echo "$metrics" >> "$RESULT_FILE"

    # Print human-readable summary
    echo "$metrics" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  {d['pipeline']:<20} Hit Rate={d['hit_rate']:.4f}  MRR={d['mrr']:.4f}  Latency={d['avg_latency_ms']:.0f}ms\")
" >&2
}

run_config "full"        ""                       "true"
run_config "no_rerank"   "COGENT_EVAL_NO_RERANK=1" "false"
run_config "no_fusion"   "COGENT_EVAL_NO_FUSION=1" "false"
run_config "no_dense"    "COGENT_EVAL_NO_DENSE=1"  "false"
run_config "agentic"     "COGENT_EVAL_AGENTIC=1"   "false"

echo "]" >> "$RESULT_FILE"

echo "" >&2
echo "=== Ablation Summary ===" >&2
python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
print(f\"{'Pipeline':<20} {'Hit Rate':>10} {'MRR':>10} {'Latency(ms)':>12}\")
print('-' * 54)
for d in data:
    print(f\"{d['pipeline']:<20} {d['hit_rate']:>10.4f} {d['mrr']:>10.4f} {d['avg_latency_ms']:>12.0f}\")
" >&2

echo "" >&2
echo "Result saved to $RESULT_FILE"
