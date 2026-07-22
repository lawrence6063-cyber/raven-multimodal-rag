# 新 Eval 体系执行手册（RUNBOOK）

> 配套 spec：`docs/EVAL_UPGRADE_SPEC.md`（口径 SSOT） · 可视化：`docs/eval-upgrade-showcase.html`
> 本手册聚焦「怎么跑、怎么读结果、需要人工 check 什么」。

---

## 0. 前置条件

| 项 | 要求 | 说明 |
| --- | --- | --- |
| Python | 3.13 + `.venv` | `source .venv/bin/activate` |
| 依赖 | `pip install -e ".[dev]"` | 核心 IR 指标仅需 numpy（已含） |
| 可选依赖 | `pip install -e ".[eval]"` | pytrec_eval（交叉校验）/ beir（基准）/ ragas（生成侧） |
| LLM Key | **仅经环境变量** | `export DASHSCOPE_API_KEY=...` 或 `OPENAI_API_KEY=...`；禁止写入 `settings.yaml` |

配置项（`config/settings.yaml` → `evaluation`）：
```yaml
evaluation:
  backends: ["ir", "custom"]   # ir=标准IR指标主力 | custom=对照 | ragas=生成侧(需key)
  golden_test_set: "data/golden_papers.json"
  ks: [1, 3, 5, 10]            # IR 指标 @k 截断点
  bootstrap_samples: 1000      # 消融 bootstrap 次数
```

---

## 1. 离线执行（无需 Key，可进 CI）

### 1.1 IR 指标单测（纯离线，秒级）
```bash
python -m pytest tests/unit/test_ir_metrics.py \
                 tests/unit/test_ablation_stats.py \
                 tests/unit/test_merge_testset.py \
                 tests/e2e/test_eval_regression.py -q
```
预期：`31 passed, 1 skipped`（skipped = 未装 pytrec_eval 的可选交叉校验）。

### 1.2 CI 回归门禁（hermetic，防指标劣化）
```bash
python -m pytest tests/e2e/test_eval_regression.py -q
```
断言 `ir.ndcg@5 >= baseline - 0.02`、全 @k 指标存在、`p95_latency_ms` 存在。**建议接入 CI 作为 PR 门禁。**

> ⚠️ 注意：`scripts/evaluate.py` 直接跑真实 `data/golden_papers.json` 时，会初始化 `HybridSearch`（含 embedding）。**无 key 时它会尝试连接 DashScope 而超时挂起**（既有基础设施行为，非 eval 代码问题）。因此"离线可运行性"由上面的 hermetic E2E 测试保证；跑真实 golden 集请在有 key 环境下进行（见 §2）。

---

## 2. 有 Key 执行（真实检索 + 真实指标）

### 2.1 跑标准 IR 指标
```bash
export DASHSCOPE_API_KEY=sk-xxx
python scripts/evaluate.py --backends ir,custom --k 1,3,5,10
# JSON 输出（供后续统计/存档）
python scripts/evaluate.py --backends ir,custom --k 1,3,5,10 --json > experiments/results/eval_full.json
```
输出含：`ir.recall@k / ir.precision@k / ir.mrr@k / ir.map@k / ir.ndcg@k`（k=1,3,5,10）、`custom.hit_rate/mrr/recall_completeness`、`avg_latency_ms`、`p95_latency_ms`。

### 2.2 消融 + 统计显著性
```bash
bash scripts/run_ablation.sh
```
逐组跑 full / no_rerank / no_fusion / no_dense / agentic，产出：
- `experiments/results/ablation_<label>.json`（每组含 per_query @k 明细）
- `experiments/results/ablation_stats.json`（bootstrap 95% CI + paired permutation p-value）

读表：`p(vs full) < 0.05` 且带 `*` 才是统计显著的差异；CI 重叠说明差异可能是噪声。

### 2.3 生成侧真跑（Ragas）
```bash
pip install -e ".[eval]"          # 装 ragas
python scripts/evaluate.py --backends ragas
```
输出 `ragas.faithfulness / answer_relevancy / context_recall`。无 key / 未装 ragas 时自动降级（跳过，不报错）。

---

## 3. 测试集扩充（31 → ≥150，chunk 级 + graded）

> 前置：已用 `scripts/ingest.py` 摄取文档到向量库 + 有 LLM key。

```bash
# 路线1：chunk 级天然对齐（枚举已摄取 chunk → LLM 出题，golden grade=3）
python scripts/gen_testset.py --strategy llamaindex --n 150 --out data/golden_synth_li.json

# 路线2：ragas 合成（自带 ground_truth answer，需装 ragas）
python scripts/gen_testset.py --strategy ragas --n 150 --out data/golden_synth_ragas.json

# 合并去重（富信息优先，手写集不被弱合成集覆盖）
python scripts/merge_testset.py \
    data/golden_papers.json data/golden_synth_li.json data/golden_synth_ragas.json \
    --out data/golden_papers.json
```
无 key → 明确报错、退出码 2、**不写半成品文件**。合并后打印各 `source`/chunk 级条数统计。

---

## 4. 公开基准锚定（可选加分）
```bash
pip install -e ".[eval]"          # 装 beir
python scripts/run_beir.py --dataset scifact --k 10 --out experiments/results/beir_scifact.json
```
用同一 `IRMetricsEvaluator` 口径跑 BEIR 子集，验证指标定义与业界一致、数量级合理。需网络下载数据集。

---

## 5. 结果解读速查

| 指标 | 看什么 | 健康区间（经验） |
| --- | --- | --- |
| `ndcg@10` | 排序质量（含 graded） | 越接近 1 越好；私有集 0.7~0.9 常见 |
| `recall@10` | 召回完备性 | 关注是否漏掉相关 chunk |
| `mrr@10` | 首个相关结果位置 | 高 = 正确答案排得靠前 |
| `p95_latency_ms` | 长尾延迟 | 关注是否有慢查询 |
| 消融 `p(vs full)` | 组件是否真有贡献 | `<0.05 *` 才可下"显著"结论 |

---

## 6. 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `evaluate.py` 卡住不返回 | 无 key，HybridSearch 连 DashScope 超时 | 配置 key，或改用 hermetic 单测验证逻辑 |
| 指标全 0 | 检索降级为空 / golden 与库不匹配 | 检查是否已摄取、chunk_id 是否对齐 |
| `Unknown evaluator backend: 'ir'` | 工厂未触发注册 | 确认走 `EvaluatorFactory`（已惰性注册） |
| ragas 指标缺失 | 未装 ragas / 无 key | 属预期降级；需要则装依赖 + 配 key |
| pytrec 交叉校验被跳过 | 未装 pytrec_eval | 可选，装 `.[eval]` 后启用 `use_pytrec=True` |

---

## 7. 人工 Check 清单（Owner 需逐项确认）

### A. 安全（必查）
- [ ] `config/settings.yaml` 中 `llm.api_key` / `embedding.api_key` / `vision_llm.api_key` **均为空字符串**，key 只经环境变量注入。
- [ ] 确认历史提交里泄露的旧 key（`sk-691b8e8f...`）已在云侧**吊销/轮换**（清空配置不等于吊销）。
- [ ] CI / 运行环境的 `DASHSCOPE_API_KEY` 等经密钥管理注入，未硬编码进脚本或日志。

### B. 数据与口径（业务判断）
- [ ] `golden` 相关度分级标准是否明确（0=无关 / 1=弱相关 / 2=相关 / 3=高度相关），标注者口径一致。
- [ ] 合成用例经**人工抽检**：剔除幻觉问题、答非所问、与 chunk 不符者，再 merge 入库。
- [ ] 扩充后 `data/golden_papers.json` 达 ≥150 条、其中 ≥120 条含 `expected_chunk_ids`（chunk 级）。
- [ ] chunk 级 golden 的 `chunk_id` 与**当前向量库**一致（换 embedding 维度/重摄后需重建 golden）。

### C. 指标与阈值（需拍板）
- [ ] `evaluation.ks`（默认 `[1,3,5,10]`）是否符合业务关注的截断点。
- [ ] CI 门禁 `tests/e2e/test_eval_regression.py` 的 `_BASELINE` 阈值与 `_TOLERANCE`（0.02）是否合理，是否需按真实基线重新标定。
- [ ] 消融 `bootstrap_samples`（1000）/ 置换次数（10000）是否满足报告置信度要求。

### D. 执行环境（需你提供/授权）
- [ ] 是否已有可用 LLM key 跑真实 §2 / §3 / §5（否则只能跑离线 §1）。
- [ ] 是否已摄取文档到向量库（`scripts/ingest.py`），gen_testset 的 llamaindex 路线依赖它。
- [ ] 是否允许 `run_beir.py` 联网下载 BEIR 数据集（SSRF 约束：仅官方源）。

### E. 待人工执行的收尾项（当前仅脚本就绪）
- [ ] 用 §3 扩充并 merge golden 集到 ≥150 条。
- [ ] 用 §2.3 跑一次生成侧真实指标并存档。
- [ ] 用 §4 跑一次 BEIR 锚定并存档 `experiments/results/beir_*.json`。
- [ ] 将 §1.2 回归门禁接入 CI（PR 必跑）。
