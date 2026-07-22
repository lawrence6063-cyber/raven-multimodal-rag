#!/usr/bin/env bash
# run_ablation.sh — RAG 检索消融实验：逐个关闭组件，对比标准 IR 指标并给统计显著性。
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
# 输出：
#   experiments/results/ablation_<label>.json  每组完整报告（含 per_query @k 明细）
#   experiments/results/ablation_stats.json     bootstrap 95% CI + paired permutation p-value

set -euo pipefail

cd "$(dirname "$0")/.."

# 使用 .venv 中的 Python
PYTHON="$(pwd)/.venv/bin/python"

OUT_DIR="experiments/results"
mkdir -p "$OUT_DIR"

# 用 ir 后端产出含 per_query @k 明细的完整报告（供统计脚本重采样）
run_config() {
    local label="$1"
    local env_vars="$2"
    local report="$OUT_DIR/ablation_${label}.json"

    echo "=== Running: $label ===" >&2
    # 只捕获 stdout (JSON)，stderr 输出到终端
    env $env_vars $PYTHON scripts/evaluate.py --backends ir --json 2>/tmp/eval_stderr.log > "$report"

    # 人类可读摘要
    $PYTHON -c "
import json
d = json.load(open('$report'))
m = d.get('metrics', {})
def g(k):
    return m.get(k, 0.0)
print(f\"  {d.get('pipeline','$label'):<20} nDCG@10={g('ir.ndcg@10'):.4f}  Recall@10={g('ir.recall@10'):.4f}  MRR@10={g('ir.mrr@10'):.4f}  Lat={g('avg_latency_ms'):.0f}ms\")
" >&2
}

run_config "full"        ""
run_config "no_rerank"   "COGENT_EVAL_NO_RERANK=1"
run_config "no_fusion"   "COGENT_EVAL_NO_FUSION=1"
run_config "no_dense"    "COGENT_EVAL_NO_DENSE=1"
# agentic 走公平化配置：对齐检索深度(top_k=30 == 其余配置的 retrieval.top_k)
# 并关闭路由直答/collection 门控(NO_ROUTE)，避免整条 query 因“未检索”被零分，
# 使其与检索类配置在同一 chunk 级 graded 口径下可比。
run_config "agentic"     "COGENT_EVAL_AGENTIC=1 COGENT_AGENTIC_TOP_K=30 COGENT_AGENTIC_NO_ROUTE=1"

echo "" >&2
echo "=== Statistical significance (bootstrap 95% CI + paired permutation) ===" >&2
$PYTHON scripts/ablation_stats.py \
    --reports \
        full="$OUT_DIR/ablation_full.json" \
        no_rerank="$OUT_DIR/ablation_no_rerank.json" \
        no_fusion="$OUT_DIR/ablation_no_fusion.json" \
        no_dense="$OUT_DIR/ablation_no_dense.json" \
        agentic="$OUT_DIR/ablation_agentic.json" \
    --metrics ndcg@10 recall@10 ndcg@5 \
    --baseline full \
    --out "$OUT_DIR/ablation_stats.json"

echo "" >&2
echo "Reports saved under $OUT_DIR/ (ablation_*.json + ablation_stats.json)" >&2
