# RAG 评估体系升级方案（Eval Upgrade Spec · 最终可执行版）

> **目标**：把当前 31 条手写、文档级、三指标的评估，升级为一套**对标业界、可复现、带统计置信度**的成熟评估体系。
> **定位**：本文是 eval 部分的 Single Source of Truth，细化到实现者可直接照做。落地后替换 `PROGRESS.md` 阶段 H 的验收口径。
> **落地状态**：✅ 已实现（H2-1~H2-6）。核心离线部分（H2-1 指标 / H2-2 CI 门禁 / H2-4 消融显著性）已带单测全绿；H2-3 测试集扩充与 H2-5 生成侧真跑的脚本已就绪，执行需 LLM key + 已摄取 DB；H2-6 BEIR 锚定脚本已就绪，需 `beir` 可选依赖。见文末《落地记录》。
> **状态**：已与现有代码契约对齐（`base_evaluator.py` / `custom_evaluator.py` / `evaluator_factory.py` / `eval_runner.py` / `scripts/evaluate.py` / `settings.py::EvaluationSettings`），签名可直接实现。

---

## 0. 为什么现在的 eval "受挑战"

| 维度 | 现状 | 问题 |
| --- | --- | --- |
| 测试集规模 | 31 条手写（title_match 15 / semantic 5 / multi_doc 3 / multihop 8） | 样本太小，指标方差大，结论不可信；评审第一质疑点 |
| 标注粒度 | 文档级（`expected_sources` = PDF 路径） | 无法评估"检索到正确 chunk"，掩盖分块/embedding 质量问题 |
| 相关性 | 二值（命中/未命中） | 无法区分"排第 1"和"排第 10"，nDCG 类指标缺失 |
| 指标集 | `hit_rate / mrr / recall_completeness` | 缺 IR 标准指标（nDCG@k、Recall@k、MAP、Precision@k） |
| 生成质量 | ragas 懒加载 mock，离线不真跑 | 没有 faithfulness / answer_relevancy 实测数据 |
| 消融可信度 | 直接比 0.806 vs 0.839（见 `experiments/results/ablation_result.json`） | 31 样本下差异无统计意义（甚至出现消融反超），缺置信区间/显著性检验 |
| 基准锚定 | 全自建私有集 | 无公开 benchmark 对照，无法横向比较 |

---

## 1. 业界 GitHub 方案调研（借鉴点，不重复造轮子）

| 项目 | 核心能力 | 本方案借鉴点（具体到口径/函数） |
| --- | --- | --- |
| **explodinggradients/ragas** | `TestsetGenerator` 从文档合成 QA（simple/reasoning/multi_context 演化）；生成侧 faithfulness / answer_relevancy / context_precision / context_recall | ① H2-3 自动扩充测试集（自带 ground_truth answer） ② H2-5 真跑生成侧 4 指标 |
| **beir-cellar/beir** | 异构 IR benchmark 标准协议：`qrels`（graded relevance）+ `pytrec_eval` 算 NDCG@k / MAP@k / Recall@k / P@k | ① `qrels` graded 数据格式（0..3） ② §2.1 标准 IR 指标口径 ③ H2-6 公开基准锚定 |
| **cvangysel/pytrec_eval** | TREC eval 的 Python 封装，权威 IR 指标实现 | 作为 nDCG/MAP 的"金标准"**可选**交叉校验（`use_pytrec` 开关） |
| **confident-ai/deepeval** | pytest 风格断言式评估（阈值断言） | H2-2 评估进 CI，做**回归门禁** |
| **run-llama/llama_index** | `generate_question_context_pairs`（node→golden 天然对齐）；`RetrieverEvaluator`（hit_rate/mrr/ndcg） | H2-3 chunk 级自动标注（node 自身即 golden chunk_id） |
| **truera/trulens** | RAG Triad：context relevance / groundedness / answer relevance | H2-5 生成侧三角交叉印证 ragas |
| **stanford-futuredata/ARES** | LLM judge + PPI（prediction-powered inference）给指标**置信区间** | H2-4 消融显著性方法论（本方案落地为 bootstrap + 置换检验，工程更轻） |

**结论**：检索侧用 **BEIR/pytrec_eval 口径**自研轻量实现（离线、无 key、进 CI）；测试集扩充用 **Ragas TestsetGenerator + LlamaIndex** 双路线；生成侧真跑 **Ragas**；回归门禁参考 **DeepEval**；消融显著性参考 **ARES/自助法**。

---

## 2. 目标指标体系

### 2.1 检索侧（离线可算，无需 LLM，进 CI）

以 chunk 级 graded relevance 为准，`k ∈ {1, 3, 5, 10}`（可配）。设某 query 检索结果按分数降序为 `r_1 … r_n`，golden 相关度映射 `rel: id -> grade`（缺省命中即 `grade=1`）。

| 指标 | 定义 | 现状 |
| --- | --- | --- |
| `hit_rate@k` | top-k 内至少命中一个 `grade>0` 的 golden，则计 1，否则 0，取均值 | 已有（仅全量） |
| `recall@k` | top-k 内命中的相关文档数 / golden 相关文档总数（`grade>0` 计数） | 部分（`recall_completeness` ≈ recall@全量） |
| `precision@k` | top-k 内命中的相关文档数 / `min(k, n)` | ❌ 新增 |
| `mrr@k` | 首个相关结果 rank 的倒数；若前 k 无命中则 0，取均值 | 已有（仅全量） |
| `map@k` | 平均精度（AP）均值，AP = Σ_{i≤k, r_i相关} precision@i / (相关文档总数) | ❌ 新增 |
| `ndcg@k` | 归一化折损累积增益，支持 graded relevance | ❌ 新增（**核心补齐**） |

**nDCG 公式（BEIR/TREC 口径）**：

\[
DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
\]

- `rel_i` = 第 `i` 位结果的相关度（未命中或不在 golden 中记 0）。
- `IDCG@k`：将 golden 的相关度**降序**排列后取前 k 个按同式计算（理想排序上界）。
- `ndcg@k = DCG@k / IDCG@k`；当 `IDCG@k == 0`（该 query 无任何相关 golden）时记 `ndcg@k = 0.0`。

**graded / 二值兼容规则**：
- 若 `EvalInput.relevance` 非空 → 用其分级值（0..3）算 DCG/IDCG（graded 模式）。
- 若 `relevance` 为空 → 对每个 `golden_ids` 视为 `grade=1`（二值模式），nDCG 退化为二值 nDCG，与旧数据完全兼容。
- `recall/precision/map/hit/mrr` 的"命中"判定：`id ∈ golden_ids` 且（graded 模式下）`grade>0`。

**复杂度**：单 query O(k)（结果已按分数排序，切片即可）；全量 O(N·k)，N=样本数，无性能瓶颈，纯离线可进 CI。

**边界约定（供单测）**：
- 空 golden：所有指标该 query 记 0。
- 空检索：所有指标该 query 记 0。
- `k > n`：precision 分母用 `min(k, n)`；DCG 只累加到 `n`。
- graded nDCG 需有**手算对照**用例（见 §6 单测清单）。

### 2.2 生成侧（需 LLM，Ragas 真跑，离线降级为跳过/0）

| 指标 | 含义 |
| --- | --- |
| `faithfulness` | 答案是否忠于检索到的上下文（反幻觉） |
| `answer_relevancy` | 答案与问题相关性 |
| `context_precision` | 检索上下文的排序精度 |
| `context_recall` | 上下文对 ground_truth 的召回 |

依赖 `EvalInput.answer` 与 `EvalInput.contexts`（现有字段）+ 参考答案（测试集 `answer` 字段）。无 key 时 `RagasEvaluator` 保持现有 mock/跳过行为，检索侧指标不受影响。

### 2.3 效率与稳健

`avg_latency_ms`（现有）+ 新增 `p95_latency_ms`；消融对比给 **bootstrap 95% 置信区间** + **paired permutation p-value**（§4.4）。

---

## 3. 测试集扩充方案（31 → 目标 ≥150，chunk 级 + graded）

### 3.1 新数据格式（向后兼容）

`data/golden_papers.json` 的 `test_cases[]` 每条扩展为（新增字段全部可选）：

```json
{
  "query": "……",
  "expected_sources": ["data/documents/rag/02_dense_passage_retrieval.pdf"],
  "expected_chunk_ids": ["<chunk_id>", "…"],
  "relevance": { "<chunk_id>": 3, "<other_chunk_id>": 1 },
  "category": "semantic",
  "source": "handwritten | ragas_synth | llamaindex_synth",
  "answer": "（可选）参考答案，供生成侧指标使用"
}
```

- `relevance`：0/1/2/3 分级（BEIR/qrels 口径）；缺省时按 `expected_chunk_ids` 视为二值（=1），兼容旧数据。
- `expected_chunk_ids` 存在时优先 chunk 级评估；否则回退文档级（`EvalRunner` 现有"prefer chunk ids, fall back to sources"逻辑保留，见 `eval_runner.py` L115-121）。
- **兼容性保证**：只有 `expected_sources` 的旧 31 条数据无需改动即可继续运行。

### 3.2 自动合成脚本 `scripts/gen_testset.py`（新增）

两条互补路线，产物统一转成 §3.1 格式：

1. **Ragas 路线**（生成侧友好）：`TestsetGenerator` 读取 `data/documents/` 的 chunk，按 evolution 分布合成（simple 50% / reasoning 25% / multi_context 25%），自带 ground_truth answer → 填 `answer` 字段。
2. **LlamaIndex 路线**（chunk 级天然对齐）：对每个 node 用 LLM 生成 1~2 个问题，node 自身即 golden chunk → `expected_chunk_ids` 精确、`relevance` 默认给该 node `grade=3`。

**CLI**：

```bash
python scripts/gen_testset.py --n 150 --strategy ragas      --out data/golden_synth_ragas.json
python scripts/gen_testset.py --n 150 --strategy llamaindex --out data/golden_synth_li.json
```

- `--strategy {ragas,llamaindex}`（必填）、`--n`（目标条数）、`--out`（输出路径）、`--docs`（默认 `data/documents/`）。
- **无 key 行为**：脚本明确报错并提示需要 `DASHSCOPE_API_KEY`/`OPENAI_API_KEY`，退出码非 0，不产出半成品文件。
- key 只走环境变量（复用 `settings.py::_resolve_api_keys` 机制），禁止硬编码。

### 3.3 合并脚本 `scripts/merge_testset.py`（新增）

```bash
python scripts/merge_testset.py data/golden_papers.json data/golden_synth_ragas.json data/golden_synth_li.json --out data/golden_papers.json
```

- 合并多个测试集文件，按 `query` 归一化去重（strip + lower + 折叠空白），冲突时保留信息更全的一条（有 `expected_chunk_ids`/`relevance`/`answer` 者优先）。
- 保留 `test_cases` 顶层结构；产出统计（各 `source`/`category` 计数）打印到 stdout。
- **人工抽检工作流**：合成后建议保留 **≥30 条纯手写 + ≥120 条合成**，按 `source` 区分；抽检剔除低质/幻觉问题后再 `merge` 写回 `golden_papers.json`。

### 3.4 公开基准锚定（可选，加分项 → H2-6）

接入 BEIR 的一个小子集（`scifact` 或 `nfcorpus`）跑一遍，证明本项目 pipeline 指标口径与业界一致、数量级合理。产物 `experiments/results/beir_scifact.json`。见 §4.7。

---

## 4. 架构改造（最小侵入，复用现有可插拔层）

现有 `BaseEvaluator / EvalInput / EvalResult / EvaluatorFactory / CompositeEvaluator / EvalRunner` 结构良好，**不动接口**，只做增强。

### 4.1 `EvalInput` 扩展（`src/libs/evaluator/base_evaluator.py`）

在现有 dataclass 末尾新增可选字段（默认空，向后兼容，风格对齐现有 `field(default_factory=...)`）：

```python
@dataclass
class EvalInput:
    query: str
    retrieved_ids: list[str]
    golden_ids: list[str]
    retrieved_texts: list[str] = field(default_factory=list)
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    relevance: dict[str, int] = field(default_factory=dict)  # 新增: id -> grade(0..3)，缺省命中视为 1
```

### 4.2 新增 `IRMetricsEvaluator`（`src/libs/evaluator/ir_metrics_evaluator.py`，新增）

```python
@register_evaluator("ir")
class IRMetricsEvaluator(BaseEvaluator):
    """IR 标准指标评估器：离线计算 recall/precision/mrr/map/ndcg@k（支持 graded relevance）。"""

    def __init__(self, ks: tuple[int, ...] = (1, 3, 5, 10), use_pytrec: bool = False):
        ...

    def evaluate(self, inputs: list[EvalInput]) -> EvalResult:
        # 返回 metrics: {"recall@1":..,"ndcg@10":.., ...}（各 @k 全量聚合均值）
        # 返回 details: {"per_query": [{query, ndcg@k, recall@k, ...}], "total_queries": N}
        ...

    @property
    def provider_name(self) -> str:
        return "ir"
```

- **命名规范**：metric key 统一 `"<name>@<k>"`（如 `ndcg@10`）；经 `CompositeEvaluator` 前缀合并后为 `ir.ndcg@10`。
- **`details["per_query"]`**：每 query 的各 @k 明细，供 §4.4 消融统计做 bootstrap 重采样。
- **`use_pytrec=True`（可选）**：若装了 `pytrec_eval`，用它交叉校验自研实现（构造 `qrels`/`run` 字典）；未装则静默走自研实现。
- 保留现有 `custom` evaluator 不变（回归对照），`ir` 作为新增主力。

### 4.3 工厂惰性注册（`src/libs/evaluator/evaluator_factory.py`）

在 `_ensure_builtins_registered()` 中按现有副作用式写法追加一段（对齐现有 `try: import ... # noqa: F401 except ImportError: pass` 风格）：

```python
    try:
        import src.libs.evaluator.ir_metrics_evaluator  # noqa: F401
    except ImportError:
        pass
```

> 注意本项目"惰性注册坑"：装饰器 `@register_evaluator("ir")` 必须在模块被 import 时执行才会注册。现有 evaluator 工厂用 import 副作用式（无清空 registry 的测试，安全），沿用即可；若后续单测会清空 `_EVALUATOR_REGISTRY`，需改为显式 import 类 + `setdefault`。

### 4.4 `EvalRunner` 增强（`src/observability/evaluation/eval_runner.py`）

- **填充 `relevance`**：组装 `EvalInput` 时读取 `case.get("relevance") or {}` 传入（chunk 级用 chunk_id 为 key；文档级回退时用 source 为 key，兼容）。
- **`per_query` 增补**：合并 `ir` evaluator 的 `details["per_query"]` 各 `@k` 明细到现有 per_query 项。
- **`p95_latency_ms`**：`metrics` 新增 `p95_latency_ms = percentile(latencies_ms, 95)`（现有已算 `avg_latency_ms`，同处补 p95）。
- **`--k` 透传**：`EvalRunner` 支持从 `settings.evaluation.ks` 读取 k 列表，构造 `IRMetricsEvaluator(ks=...)`（经工厂时通过 settings 注入或运行时 set）。

### 4.5 CLI 增强（`scripts/evaluate.py`）

- 新增 `--k`，如 `--k 1,3,5,10`，解析为 tuple 覆盖 `settings.evaluation.ks`。
- 非 JSON 输出的 Metrics 段自然会打印所有 `ir.*@k` 指标（现有 `for name, value in report.metrics.items()` 循环已通用，无需改渲染）。
- 现有 env 消融开关（`COGENT_EVAL_NO_RERANK/NO_DENSE/NO_SPARSE/NO_FUSION/AGENTIC`）保持不变。

### 4.6 消融显著性（`scripts/ablation_stats.py` 新增 + 升级 `scripts/run_ablation.sh`）

对每组配置（full / no_rerank / no_dense / no_fusion / …）用 `evaluate.py --json` 产出**含 per_query 各 @k 明细**的报告，再：

- **bootstrap 95% CI**：对 per-query 指标向量重采样 `bootstrap_samples`（默认 1000）次，取每个指标 2.5% / 97.5% 分位为 95% CI。
- **paired permutation test**：full vs 各消融，对配对 per-query 差值做随机符号翻转置换检验（默认 10000 次），出 p-value。
- 输出对比表：

```
Pipeline              nDCG@10 [95% CI]        Recall@10 [95% CI]     p(vs full)
full                  0.812 [0.771, 0.850]    0.655 [0.60, 0.71]     -
no_rerank             0.640 [0.590, 0.690]    0.663 [0.61, 0.72]     0.001 *
no_dense              0.702 [0.66, 0.74]      0.590 [0.54, 0.64]     0.013 *
```

- 产物 `experiments/results/ablation_stats.json`；`run_ablation.sh` 末尾调用 `ablation_stats.py` 汇总各 JSON。
- 统计实现优先用 `numpy`（已在依赖内），无需引入 `scipy`。

### 4.7 生成侧真跑 / 基准锚定

- **H2-5 生成侧**：`evaluate.py --backends ragas`（有 key）走 `RagasEvaluator` 真实后端；`EvalRunner` 已把 `retrieved_texts` 填入 `contexts`，测试集 `answer` 作为 ground_truth。无 key 时保持现有降级。
- **H2-6 基准锚定（可选）**：`scripts/run_beir.py --dataset scifact`，加载 BEIR 子集 → 用本项目 embedding/检索链路跑 → 用 `IRMetricsEvaluator` 同口径出 nDCG@10 等，产物 `experiments/results/beir_scifact.json`。依赖 `beir`（可选）。

### 4.8 配置（`config/settings.yaml` + `src/core/settings.py::EvaluationSettings`）

`EvaluationSettings` 现为 `backends / golden_test_set`，新增两字段：

```python
@dataclass
class EvaluationSettings:
    backends: list[str] = field(default_factory=lambda: ["custom"])
    golden_test_set: str = "tests/fixtures/golden_test_set.json"
    ks: list[int] = field(default_factory=lambda: [1, 3, 5, 10])      # 新增
    bootstrap_samples: int = 1000                                      # 新增
```

`config/settings.yaml`：

```yaml
evaluation:
  backends: ["ir", "custom"]      # ir 新增主力，custom 保留对照，ragas 按需
  golden_test_set: "data/golden_papers.json"
  ks: [1, 3, 5, 10]
  bootstrap_samples: 1000
```

---

## 5. 落地 Roadmap（分阶段、可独立验收）

| 阶段 | 交付物 | 命令 / 验收 | 依赖 |
| --- | --- | --- | --- |
| **H2-1 指标补齐** | `IRMetricsEvaluator` + `EvalInput.relevance` + `EvalRunner` @k 明细/p95 + CLI `--k` + 工厂注册 + 单测 | `pytest tests/unit/test_ir_metrics.py -q`；`python scripts/evaluate.py --backends ir --k 1,3,5,10` 离线出 ndcg/recall/precision/map@k | 无（纯离线） |
| **H2-2 CI 门禁** | `tests/e2e/test_eval_regression.py`（hermetic，复用 `tests/fixtures/golden_test_set.json` + 假检索器） | `pytest tests/e2e/test_eval_regression.py -q` 全绿，断言 `ndcg@5 >= baseline-0.02` | H2-1 |
| **H2-3 测试集扩充** | `gen_testset.py` + `merge_testset.py`，chunk 级 + graded，规模 ≥150 | `python scripts/gen_testset.py …`；人工抽检；`golden_papers.json` ≥150 条且 ≥120 条含 `expected_chunk_ids` | LLM key |
| **H2-4 消融显著性** | `ablation_stats.py` + 升级 `run_ablation.sh` | `bash scripts/run_ablation.sh` 输出带 95% CI + p-value 的对比表 | H2-1、扩充集 |
| **H2-5 生成侧真跑** | Ragas 真实后端跑 faithfulness/answer_relevancy/context_recall | `python scripts/evaluate.py --backends ragas`（有 key） | LLM key |
| **H2-6 基准锚定(可选)** | BEIR `scifact` 子集跑分 | `python scripts/run_beir.py --dataset scifact` | beir 依赖 |

**优先级建议**：H2-1 + H2-2 + H2-3 是"止血"核心（补齐标准指标 + 扩充样本 + 防劣化），能直接回应"只有 31 条"的质疑；H2-4/5/6 为加分项。

---

## 6. 单测与验收标准（Definition of Done）

### 6.1 单测清单（`tests/unit/test_ir_metrics.py`，新增）

- `test_ndcg_graded_manual`：给定固定检索序 + graded relevance，**手算 DCG/IDCG 对照** nDCG@k（核心正确性）。
- `test_recall_precision_at_k`：已知命中分布，校验 recall@k / precision@k 分母口径（`min(k,n)`）。
- `test_map_at_k`：多相关文档场景校验 AP 均值。
- `test_mrr_at_k`：首命中在 k 内/外两种情况。
- `test_empty_golden`：空 golden → 全指标 0，不抛异常。
- `test_empty_retrieved`：空检索 → 全指标 0。
- `test_binary_fallback`：`relevance` 为空时退化为二值 nDCG，与显式 `grade=1` 结果一致。
- `test_k_larger_than_n`：`k > n` 边界不越界。
- （可选）`test_pytrec_crosscheck`：装了 `pytrec_eval` 时自研值与其一致（`pytest.importorskip`）。

### 6.2 回归门禁（`tests/e2e/test_eval_regression.py`，新增）

复用 `tests/fixtures/golden_test_set.json`（5 条，chunk_id 与内存语料对齐）+ 假检索器，跑 `ir` 后端，断言 `ir.ndcg@5 >= baseline - 0.02` 等阈值；**不依赖网络/LLM**。baseline 写入测试常量或小 fixture。

### 6.3 DoD

1. `python scripts/evaluate.py --backends ir --k 1,3,5,10` 输出 recall/precision/mrr/map/**ndcg**@{1,3,5,10} 全指标，离线无 key 可跑（检索降级不报错）。
2. `data/golden_papers.json` ≥150 条，其中 ≥120 条含 `expected_chunk_ids`（chunk 级），并含 graded `relevance`。
3. `bash scripts/run_ablation.sh` 产出的对比表**每个指标带 95% 置信区间与 p-value**，结论可信。
4. `pytest tests/ -q` 全绿，含 §6.1 IR 指标单测与 §6.2 回归门禁 E2E。
5. `README` 开发者指南 + `PROGRESS.md` 阶段 H 口径同步更新（列为收尾项）。

---

## 7. 依赖、安全与迁移兼容

### 7.1 依赖（`pyproject.toml`）

新增可选依赖组 `[project.optional-dependencies].eval`：`pytrec_eval`（可选交叉校验）、`beir`（可选基准）；Ragas 已在 dev 依赖。核心 H2-1/H2-2 仅用 `numpy`（已有），不引入重依赖。

### 7.2 安全（强制）

- 合成脚本与 Ragas 调用统一走现有 LLM/Embedding Factory，**key 仅从环境变量读取**（复用 `settings.py::_resolve_api_keys` 已有回退机制）。
- **修复现存隐患**：`config/settings.yaml` 中硬编码的 DashScope key 必须清空（留空字符串），改由 `DASHSCOPE_API_KEY` 环境变量注入。
- 所有 LLM 相关路径必须离线优雅降级（无 key → 跳过生成侧指标 / 合成脚本明确报错退出），保证 CI 与他人复现不被 key 阻塞。
- BEIR 数据集下载走官方源，遵守 SSRF 约束（不请求内网）。

### 7.3 迁移兼容

- 新增字段（`expected_chunk_ids` / `relevance` / `answer` / `source`）全部可选；旧 31 条文档级数据保持可用。
- `EvalRunner` 现有"优先 chunk 级否则回退文档级"逻辑保留（`eval_runner.py` L115-121），无需改造调用方。
- `custom` evaluator 保留作回归对照，`ir` 为新增主力，两者可经 `CompositeEvaluator` 同时输出（`custom.hit_rate` / `ir.ndcg@10`）。
- 换 embedding 维度/空间后旧向量索引需清空重摄——H2-3/H2-6 涉及重新摄取时须留意（沿用现有 ingest 流程）。

---

## 8. 落地记录（实现产物清单）

| 项 | 文件 | 说明 |
| --- | --- | --- |
| H2-1 指标 | `src/libs/evaluator/ir_metrics_evaluator.py` | `IRMetricsEvaluator` `@register_evaluator("ir")`，recall/precision/hit_rate/mrr/map/ndcg@k，graded(0-3) + 二值兼容，可选 `use_pytrec` 交叉校验 |
| H2-1 字段 | `src/libs/evaluator/base_evaluator.py` | `EvalInput.relevance: dict[str,int]`（末尾新增，默认空） |
| H2-1 装配 | `src/observability/evaluation/eval_runner.py` | 填充 `relevance`、合并 ir per_query @k 明细、新增 `p95_latency_ms`、`_apply_ks_to_evaluators` 注入 ks |
| H2-1 工厂 | `src/libs/evaluator/evaluator_factory.py` | `_ensure_builtins_registered()` 追加 ir 副作用式注册 |
| H2-1 CLI | `scripts/evaluate.py` | 新增 `--k 1,3,5,10` |
| H2-1 配置 | `src/core/settings.py` / `config/settings.yaml` | `EvaluationSettings.ks/bootstrap_samples`；yaml `evaluation` 段；backends 默认 `["ir","custom"]` |
| H2-1 单测 | `tests/unit/test_ir_metrics.py` | graded nDCG 手算对照、recall/precision/map/mrr@k、空 golden/空检索/k>n、二值退化、pytrec 交叉校验（importorskip）— 13 passed/1 skipped |
| H2-2 门禁 | `tests/e2e/test_eval_regression.py` | hermetic ir 后端回归，断言 `ir.ndcg@5 >= baseline-0.02`、@k 全指标存在、p95 存在 — 4 passed |
| H2-3 生成 | `scripts/gen_testset.py` | llamaindex（枚举 store chunk→LLM 出题，golden 精确 grade=3）/ ragas 双路线；无 key 退出码 2 且不写文件 |
| H2-3 合并 | `scripts/merge_testset.py` | 归一化去重 + 富信息优先；`tests/unit/test_merge_testset.py` 7 passed |
| H2-3 枚举 | `src/libs/vector_store/chroma_store.py` | 新增 `get_all(limit)` 供测试集生成枚举 chunk |
| H2-4 统计 | `scripts/ablation_stats.py` | bootstrap 95% CI + paired permutation p-value（仅 numpy）；`tests/unit/test_ablation_stats.py` 8 passed |
| H2-4 编排 | `scripts/run_ablation.sh` | 每组产出含 per_query @k 明细的完整报告 → 调 ablation_stats 汇总带 CI/p 值 |
| H2-6 基准 | `scripts/run_beir.py` | BEIR 子集，同口径 `IRMetricsEvaluator`；无 beir 依赖优雅报错 |
| 依赖 | `pyproject.toml` | 显式 `numpy`；可选组 `[eval]`= pytrec_eval/beir/ragas |
| 安全 | `config/settings.yaml` | 清空硬编码 DashScope key，靠 `DASHSCOPE_API_KEY` env 回退 |

**待人工执行（需 key/DB/网络，非代码问题）**：H2-3 用 `gen_testset.py`+`merge_testset.py` 将 golden 扩到 ≥150 条 chunk 级 graded；H2-5 `evaluate.py --backends ragas` 真跑生成侧；H2-6 `run_beir.py --dataset scifact` 下载并跑分。
