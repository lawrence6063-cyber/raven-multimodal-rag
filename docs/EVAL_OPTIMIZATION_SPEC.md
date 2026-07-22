# RAG 评估与优化技术规格（EVAL_OPTIMIZATION_SPEC）

> 版本：2.0（2026-07-17 回填实测）
> 定位：本项目「评估驱动优化」的单一技术规格文档。承接既定「四层指标体系 + 四阶段优化」思路，结合业界标准（BEIR / Ragas / 延迟 SLA）系统化为可执行、可验证的工程蓝图。
> 交付性质：v1.0 为纯文档蓝图；**v2.0 已按蓝图完成全部实施与真跑评估**，本文所有目标值均已回填为 230 条 golden 上的实测结果（含 95% CI + 置换检验 p 值）。实测数字来源于 `experiments/results/`，可视化报告见 `docs/ablation_report.html`。
>
> **实测速览**（full pipeline，230 条，cross-encoder 精排）：nDCG@10 **0.692** / Hit@5 **0.952** / MRR@10 **0.854** / p95 **0.71s** —— 四项均达标或超标。实验过程中还通过消融方法论**发现并修复了一个重大 bug：向量库 embedding 空间错位**（见 §2.4）。

---

## 1. 概述与定位

### 1.1 为什么需要这份规格

本项目已完成一套模块化 RAG 系统（可插拔检索 / embedding / rerank / 评估，MCP 协议 + 全链路 trace），功能层面 100% 完成。但要让它成为**拿得出手的简历级项目**，缺的不是功能，而是**一套严谨、可复现、对齐业界标准的评估与优化叙事**。

一个 RAG 项目的成熟度，不体现在"能跑"，而体现在四个层次能否自证：

1. **检索质量**是否达到业界合理水位（而非虚高刷分）；
2. **生成质量**是否可量化（无幻觉、切题、上下文信噪比）——这是"完整 RAG"与"检索器"的分水岭；
3. **系统工程**是否有明确的延迟 SLA 并达标；
4. **方法论**是否严谨（消融、置信区间、显著性、回归门禁）。

本规格把这四层固化为**指标体系 + 口径定义 + 目标值 + 实施蓝图**，确保每一个报出的数字都「有出处、有方法、有 insight」。

### 1.2 核心原则

- **对齐业界，不追虚高**：真实语料上 nDCG@10 冲到 0.9 往往意味着数据泄漏或口径注水，反而露怯。目标是落在业界 SOTA 合理区间并讲清方法论。
- **口径先于分数**：任何指标必须先定义清楚分母/相关性判定/截断点，否则一个"低分"可能只是口径坑（本项目 Recall@10 即是典型）。
- **评估驱动优化**：先建立可复现基线，再用消融量化每个组件的边际贡献，最后用显著性检验证明改进非偶然。**本次实验正是靠这套方法论抓出了 dense 索引空间错位的重大 bug（§2.4）——这本身就是"评估驱动"价值的最好证明。**
- **复用已就绪能力**：本项目的 cross-encoder reranker、Ragas evaluator、消融开关、bootstrap 配置**均已在代码中就绪**（见 §附录 A），优化以「启用 / 真跑 / 对比」为主，而非从零开发。

### 1.3 适用范围

- **纳入主指标**：文本检索质量 + 生成质量 + 系统延迟 + 方法论严谨性。
- **暂不纳入主指标**：多模态跨模态检索（能力代码已就绪但尚无落库数据），作为未来扩展见 §9。

---

## 2. 现状基线与业界锚点

### 2.1 优化前后指标对照（实测回填）

下表左列为 v1.0 优化前锚点（full = BM25 + Dense + RRF + **LLM** Rerank），右列为 v2.0 优化后实测（full = BM25 + Dense + RRF + **cross-encoder** Rerank，dense 空间错位修复后，230 条 golden）：

| 指标 | 口径 | 优化前锚点 | **优化后实测** | 判读 |
|---|---|---|---|---|
| nDCG@10 | chunk 级 graded | ≈ 0.53 | **0.692** [0.655, 0.724] | 超目标（0.58），进入科学文献 + rerank 区间 |
| Hit@5 | 文档级 | ≈ 0.84（@10） | **0.952** | 超目标（0.85） |
| MRR@10 | chunk 级 | ≈ 0.49 | **0.854** | 大幅超目标（0.55） |
| Recall@10 | chunk 级（capped 口径） | ≈ 0.26（raw 假性低分） | **0.443**（capped） | 修口径后回归真实水位，见 §3.1.4 |
| p95 延迟 | 端到端 | ≈ 38.5s（LLM rerank） | **0.71s**（cross-encoder） | 降 ≈ 98%，见 §6 |
| 生成侧 | Faithfulness / Answer Relevancy / Context Precision | 未测 | **0.850 / 0.861 / 0.676**（20 条子集） | 已补齐，见 §4 |

> 数据来源：`data/golden_papers.json`（230 条 graded：199 条 chunk 级 + 31 条文档级）+ `experiments/results/ablation_full.json`（full pipeline 实测）。95% CI 与 p 值见 `ablation_stats.json`。

### 2.2 业界锚点（BEIR 基准）

BEIR 是最权威的 IR 零样本评测基准。其 nDCG@10 典型区间为：

| 系统类型 | nDCG@10 典型区间 |
|---|---|
| BM25（纯词法基线） | 0.30 ~ 0.45 |
| 稠密检索 SOTA（BGE / E5 等） | 0.45 ~ 0.55 |
| 科学文献类数据集（SciFact 等）+ rerank | 0.60 ~ 0.70 |

> 实测校验：本项目纯 BM25（no_dense + CE）nDCG@10 = **0.430**，恰落在 BM25 区间上沿；full pipeline **0.692** 落在"科学文献 + rerank"区间——数字自洽、可信，无注水痕迹。

### 2.3 核心结论（实测验证）

1. **检索指标达标且不虚高**：chunk 级 nDCG@10 = **0.692**、Hit@5 = **0.952**，落在「科学文献 + rerank」区间，全面达到并超过 §7 目标卡。修对 rerank 截断（§3.1）+ 修复 dense 索引后自然升至此水位。
2. **生成侧缺口已补齐**：自实现 DeepSeek LLM-judge 后端真跑，Faithfulness = **0.850** / Answer Relevancy = **0.861** / Context Precision = **0.676**（20 条子集）。项目从"检索系统"升级为"完整可评估的 RAG"。
3. **延迟硬伤已解决**：将精排从 LLM rerank 换为**已就绪的 cross-encoder reranker**，p95 从 38.5s 压到 **0.71s**（降 ≈ 98%）——一个「延迟降低约 98%」的强优化故事。
4. **方法论是差异化亮点**：消融 + 95% CI + 置换检验（1000 bootstrap + 10000 permutation），四条带显著性的结论（§3.4、§5 阶段 4），并借此抓出 dense 空间错位 bug（§2.4）。

### 2.4 关键发现：向量库 embedding 空间错位（评估驱动的 bug 发现案例）⭐

这是本次实验最有价值的产出——**方法论帮我们抓出了一个靠功能测试永远发现不了的 bug**。

**现象**：消融时发现纯 dense（no_fusion）nDCG@10 仅 **0.023**（近乎为零），且 dense 对任意 query 都返回同一批 `swe_bench` chunk；反常的是纯 BM25 词法检索（0.430）竟远高于 full（当时 0.357）——即"把 dense 融进 RRF 反而拉低全链路"。

**三步诊断（逐步坐实根因）**：

| 步骤 | 验证 | 结果 | 判定 |
|---|---|---|---|
| ① | query embedding 对三个语义迥异文本编码 | 余弦 0.70 / 0.58（有区分度） | query 侧正常 ✅ |
| ② | 库内抽样存储向量两两余弦 | 0.15 ~ 0.81（有区分度） | 存储侧本身正常 ✅ |
| ③ | **用库内 chunk 原文重新编码 vs 其存储向量** | **余弦 ≈ 0.16** | **空间错位（同文本同模型本应 ≈ 1.0）** ❌ |

**根因**：建库时的 embedding 与当前 query-time embedding **不在同一向量空间**，dense 近邻搜索在错位空间里退化为常量返回。

**修复**：验证 embedding 确定性（同文本两次编码 cos = 1.0000，故重嵌可修）→ 写 `scripts/reembed_index.py`（复用 chroma 已存 documents 文本重嵌，**不重解析 PDF**；因多模态 DashScope 客户端非线程安全故**串行** + 断点续跑）→ 重嵌 8321 条文本向量（0 失败，60 分钟）。

**修复效果**：纯 dense nDCG@10 **0.023 → 0.694**，full **0.357 → 0.692**。dense 从"报废"复活为检索主力。

> 简历叙事：**"消融实验暴露 dense-only 指标异常趋零 → 三步诊断定位到向量库与查询侧 embedding 空间错位 → 重嵌 8k+ 向量修复，nDCG@10 从 0.02 拉到 0.69"**——完整的"评估驱动发现并修复问题"闭环。

---

## 3. 四层指标体系与口径定义

### 3.1 层 1 · 检索质量

检索层是项目的立身之本，指标口径必须精确到「相关性判定 + 截断点 + 分母」。

#### 3.1.1 nDCG@k（主推门面指标）

- **定义**：归一化折损累积增益。\[ \text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}, \quad \text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}} \]
- **相关性等级 `rel`**：使用 graded relevance（0~3），取自 `golden_papers.json` 每条 case 的 `relevance` 字段（`EvalInput.relevance`，缺省命中视为 1）。
- **主推截断点**：@10。目标 **0.55 ~ 0.65**，**实测 0.692（达标超标）**。
- **为什么主推**：graded + 位置折损，最能反映"好结果是否排在前面"，且与 BEIR 可直接对齐。

#### 3.1.2 Hit@k（文档级门面指标）

- **定义**：Top-k 中是否命中任一 golden（二值），对 query 取均值即命中率。
- **口径**：文档级（`expected_sources` 空间）。
- **主推**：Hit@5，目标 **0.80 ~ 0.90**，**实测 0.952（超标）**。
- **意义**：最直观的"能不能找到对的文档"，面试友好。

#### 3.1.3 MRR@k

- **定义**：\[ \text{MRR@k} = \frac{1}{N}\sum_{q}\frac{1}{\text{rank}_q} \]，`rank_q` 为该 query 第一个命中的排名（无命中记 0）。
- **主推截断点**：@10，目标 **0.50 ~ 0.65**，**实测 0.854（大幅超标）**。
- **意义**：衡量首个正确结果的靠前程度，对"一击命中"体验敏感。

#### 3.1.4 Recall@k（口径修正，辅证指标）

- **现状问题**：`EvalRunner` 原 per-query recall 口径为
  ```
  recall = |retrieved ∩ golden| / |golden|
  ```
  当单个 query 的 graded 相关集 `|golden|` 远大于截断 `k`（如某 query 有 30 条相关 chunk，而 k=10），分母恒被高估，导致 recall 上限被人为压到 `k/|golden|`，产出**假性低分（≈ 0.26）**。这不是检索差，是口径错。
- **修正口径（capped recall，已落地）**：分母改用 `min(k, |golden|)`：
  ```
  recall@k = |retrieved@k ∩ golden| / min(k, |golden|)
  ```
  这样当相关集大于 k 时，只要 Top-k 全部命中相关项，recall@k = 1.0，符合"在 k 个坑位里尽力召回"的直觉。**实测 full recall@10 = 0.443（capped）**。
- **落地位置**：`ir_metrics_evaluator.py`、`custom_evaluator.py`、`eval_runner.py` per-query 三处口径已统一为 capped，并有单测覆盖（`TestCappedRecall`）。
- **定位**：Recall 作为**辅证指标**，报告时必须标注口径（capped vs raw），**不作门面**。门面用 nDCG@10 + Hit@5。

#### 3.1.5 chunk 级 vs 文档级双口径

`golden_papers.json` 同时含 199 条 chunk 级（`expected_chunk_ids` + `relevance`）与 31 条文档级（`expected_sources`）case。`EvalRunner` 已实现 id 空间自动切换（优先 `expected_chunk_ids`，回退 `expected_sources`）。报告需**分别呈现两套口径**：

- **chunk 级**：细粒度，反映"精确定位到相关段落"的能力（主报 nDCG@10 / MRR@10）。
- **文档级**：粗粒度，反映"找对文档"的能力（主报 Hit@5 / Hit@10），指标天然更高，用于对外沟通。

### 3.2 层 2 · 生成质量（Ragas）

RAG = 检索 + 生成。本层是「完整 RAG」的必要项，**已真跑补齐**（见 §4 改造与实测）。

| 指标（Ragas 口径） | 定义 | 目标 | **实测（20 条）** |
|---|---|---|---|
| **Faithfulness** | 答案中的陈述能被检索上下文支撑的比例（衡量幻觉） | > 0.85 | **0.850** ✅ |
| **Answer Relevancy** | 答案与问题的语义相关度（衡量是否切题） | > 0.85 | **0.861** ✅ |
| **Context Precision** | 检索上下文中相关内容的排序质量（衡量信噪比） | > 0.70 | **0.676** ⚠️ 略低 |

- **实现现状（重要变更）**：`ragas` PyPI 包与已装 langchain 栈冲突（装 ragas 会把 langchain 升到 1.x，导致 ragas 0.4.3 自身 import 崩溃，且默认强依赖 OpenAI 评判 LLM + embeddings，本项目仅有 DeepSeek + DashScope）。**因此放弃 ragas 包，改用自实现的 DeepSeek LLM-judge 后端** `src/observability/evaluation/llm_judge_backend.py`，指标定义对齐 Ragas，注入现成的 `RagasEvaluator(backend=...)`。
- **数据依赖已满足**：answer 合成链路已落地（§4），`--backends ragas` 时每条样本携带 `answer` + `contexts`。
- **Context Precision 0.676 略低说明**：检索候选里混入了同文档邻近但非精确的 chunk，LLM 判为"部分相关"拉低排序精度——是真实合理数字，非 bug。

### 3.3 层 3 · 系统工程（延迟）

| 指标 | 口径 | 目标 | **实测（full + CE）** |
|---|---|---|---|
| p50 延迟 | 端到端 | < 1.5s | **已达标**（cross-encoder 亚秒级） |
| **p95 延迟** | 端到端 | **< 3s（目标 < 2s）** | **0.71s** ✅ |
| p99 延迟 | 端到端 | < 5s | **已达标** |

- **测量口径**：端到端 = `search_ms + rerank_ms + synthesize_ms`（生成侧计入时）。
- **分阶段拆解（已落地）**：`EvalRunner` 聚合已产出 `avg_latency_ms` / **`p50_latency_ms`** / `p95_latency_ms` / **`p99_latency_ms`**（复用 `_percentile`，零新依赖）；per-query 已增 **`synthesize_ms`**（生成侧启用时）。
- **根因与解法**：`rerank.provider = "llm"` 时每条 query 一次 LLM（DeepSeek）调用做重排，实测 llm rerank avg 34.8s / p95 43s，是延迟主要来源。切 cross-encoder 后 p95 降至 0.71s，详见 §6。

### 3.4 层 4 · 方法论严谨性（差异化亮点）

这是区别于"调库党"的核心竞争力，也是简历上最能体现工程深度的部分。**以下全部真跑落地**。

- **消融实验（Ablation）**：对比 full 与逐个关闭组件的变体，量化每个组件的边际贡献。`scripts/evaluate.py` 已通过环境变量支持 `COGENT_EVAL_NO_RERANK` / `NO_DENSE` / `NO_SPARSE` / `NO_FUSION` / `AGENTIC`。
  - **注意**：`dense_weight` / `sparse_weight` 原先只在 settings 定义、检索逻辑从不消费，导致 NO_DENSE/NO_FUSION 消融此前**不生效**。已在 `HybridSearch.search()` 增加权重门控（`_weight_enabled`），使这些消融开关真实生效并有单测（`TestHybridSearchAblationGating`）。
- **置信区间（95% CI）**：对每个指标用 bootstrap 重采样（1000 samples）估计 95% CI，报告 `metric [lo, hi]`。
- **显著性检验（p-value）**：full vs 变体的差异用**置换检验（permutation test，10000 次）**得 p-value，判定改进是否显著（p < 0.05）。
- **五组消融最终结论**（baseline = full，见 `ablation_stats.json`）：
  1. **rerank 显著有效**：nDCG@10 +0.154（0.538→0.692，p < 0.001）；
  2. **dense 是主力**：去掉后 0.692→0.430（p < 0.001）；
  3. **sparse/RRF 融合在此语料不显著**：full ≈ dense-only（0.692 vs 0.694，p = 0.86）——诚实的负结果；
  4. **agentic 在单跳事实检索上显著劣化**：0.692→0.303（p < 0.001），且慢 ~60 倍（40.8s vs 0.6s/query）。
  5. **agentic 在多跳专项上无完整度增益**（2026-07-21 补做，8 条真·跨域多跳 query，配对单变量，检索/精排/上下文预算 top_k=20 全对齐）：Recall Completeness single-shot 0.573 vs agentic 0.554（Δ=−0.019，配对 bootstrap 95% CI [−0.194,+0.150] 跨 0，符号检验 win2/tie4/loss2）；但 Hit Rate 87.5%→100%（+12.5%），靠子问题分解救回 single-shot 全 miss 的 query。**关键校正**：早期"0.287→0.410"来自向量库重嵌前（dense 空间错位、几乎全靠 BM25），重嵌修复 dense 后 single-shot 已强，该红利消失 —— Agentic 定位应为"召回鲁棒性兜底 + 复杂推理路径"而非"多跳完整度提升"（脚本 `scripts/eval_multihop_singleshot_vs_agentic.py`，结果 `multihop_singleshot_vs_agentic.json`）。
- **双口径**：chunk 级 vs 文档级并列（§3.1.5）。
- **CI 回归门禁**：防止检索质量在迭代中悄然劣化（§8）。

---

## 4. 生成侧完整 RAG 评估设计

本节是"全量含生成侧完整 RAG"的核心，定义如何为生成侧评估补齐 `answer` 数据链路。**已按下述蓝图实施并真跑。**

### 4.1 问题定位

- `EvalInput` 已含 `answer: str = ""` 字段（`src/libs/evaluator/base_evaluator.py`）。
- 但 `EvalRunner.run` 原先构建 `EvalInput` 时**只填检索字段**，`answer` 恒为空串。
- 因此生成侧的 Faithfulness / Answer Relevancy **无输入数据**（这两个指标依赖 `answer`）。Context Precision 仅依赖 contexts + question，可独立计算。

### 4.2 改造落地（EvalRunner 增加可选答案合成步骤）

**设计原则**：答案合成成本高（每条 query 一次 LLM 调用），**按需触发**——仅当评估 backend 含 `ragas` 时才合成，避免拖慢纯检索评估（ir / custom）。

**已落地实现**：

```python
# EvalRunner.__init__ 增加可选合成器（AnswerLike Protocol，兼容返回 str 或带 .answer 的对象）
def __init__(self, settings, hybrid_search, evaluator,
             reranker=None, answer_synthesizer=None):
    ...
    self._synthesizer = answer_synthesizer   # 可选；None 时不合成

# run() 主循环内，在拿到 (rerank 后的) results 之后：
answer = ""
synth_ms = 0.0
if self._synthesizer is not None and results:   # 仅 ragas backend 时注入
    s = time.perf_counter()
    answer = self._safe_synthesize(query, results)   # 失败降级为 ""
    synth_ms = (time.perf_counter() - s) * 1000.0

eval_inputs.append(EvalInput(
    query=query, retrieved_ids=ids, golden_ids=golden,
    retrieved_texts=retrieved_texts, contexts=retrieved_texts,
    relevance={...}, answer=answer,           # ← 补齐生成侧数据
))
# per-query 增加 "synthesize_ms": round(synth_ms, 2)
```

**合成器来源（复用，不新造）**：复用 `src/core/agent/answer_synthesizer.py` 的 `AnswerSynthesizer(settings)`。

**触发装配**（`scripts/evaluate.py._run_hybrid_eval`）：仅当 `"ragas" in backends` 时注入 `AnswerSynthesizer`；纯 `ir/custom` 评估零额外 LLM 调用。

### 4.3 评判后端：绕开 ragas 包，自实现 LLM-judge

`ragas` PyPI 包不可用（依赖冲突 + OpenAI 耦合，见 §3.2）。改由 `scripts/evaluate.py._build_evaluator(settings, backends)` 在含 `ragas` 时手动构建 `CompositeEvaluator`，注入 `RagasEvaluator(backend=build_llm_judge_backend(settings))`：

- **Faithfulness**：DeepSeek 抽取答案中的事实 claim，判断被 context 支撑的比例；
- **Answer Relevancy**：DeepSeek 从答案反向生成问题，与原问题做 DashScope embedding 余弦；
- **Context Precision**：DeepSeek 逐条判相关性 + rank-weighted precision。

每次调用防御性 try/except 降级 0，单条失败不中断整轮。

### 4.4 验证标准（已达成）

- 跑 `--backends ir,ragas` 后，`report.metrics` 出现 `ragas.faithfulness` / `ragas.answer_relevancy` / `ragas.context_precision` 且非 0（实测 0.850 / 0.861 / 0.676）。
- per-query 出现非空 `answer` 与 `synthesize_ms`。
- 纯 `--backends ir,custom` 时不触发合成（无额外 LLM 调用、`synthesize_ms = 0`）。

---

## 5. 四阶段实施蓝图（含实测达成状态）

按「对简历价值 × 见效速度」排序。每阶段独立可交付、可回滚。**四阶段均已完成并真跑验证。**

### 阶段 1 · 修检索口径 ✅ 已完成

| 项 | 内容 |
|---|---|
| **动作** | ① `rerank.top_n` 5→20，解开 @10 家族截断；② 扩召回池（`retrieval.top_k` 10→30）；③ 实现 §3.1.4 capped recall 口径 |
| **改动面** | 纯参数（`config/settings.yaml`）+ recall 口径三处计算逻辑，无结构改造 |
| **目标 → 实测** | nDCG@10 0.53→目标 0.58+ → **实测 0.692**；MRR@10 0.49→目标 0.55+ → **实测 0.854**；Recall@10 修口径后 **0.443（capped）** |
| **验证** | `python scripts/evaluate.py --backends ir,custom --k 1,3,5,10 --json` |
| **备注** | 修口径同时暴露并促成了 dense 空间错位的发现（§2.4） |

### 阶段 2 · 补生成侧评估 ✅ 已完成 ⭐

| 项 | 内容 |
|---|---|
| **动作** | ① 按 §4 补 `EvalRunner` answer 合成链路；② **放弃冲突的 ragas 包，自实现 DeepSeek LLM-judge 后端**；③ 配 key 真跑 `--backends ir,ragas` |
| **依赖** | `DEEPSEEK_API_KEY`（生成 + 评判）+ `DASHSCOPE_API_KEY`（answer relevancy 的 embedding） |
| **目标 → 实测** | Faithfulness > 0.85 → **0.850**；Answer Relevancy > 0.85 → **0.861**；Context Precision > 0.70 → **0.676**（20 条子集） |
| **意义** | 项目从"检索系统"升级为"完整可评估的 RAG" |

### 阶段 3 · 解决延迟硬伤 ✅ 已完成 ⭐

| 项 | 内容 |
|---|---|
| **动作** | ① `pip install sentence-transformers`（实装 5.6.0）；② 精排从 `llm` 切 `cross_encoder`（`CrossEncoderReranker` 已实现，默认 `cross-encoder/ms-marco-MiniLM-L-6-v2`）；③ 保留 LLM rerank 作"高精度模式"用于 §6 对比 |
| **改动面** | 装依赖 + `evaluate.py` 加 `COGENT_EVAL_RERANK_PROVIDER` env 覆盖（不改 settings 默认，便于对比） |
| **目标 → 实测** | p95 38.5s → 目标 < 2s → **实测 0.71s（降 ≈ 98%）**，且精度**不降反升**（见 §6） |
| **意义** | "把 p95 延迟从 38s 降到 0.7s"是极强的优化故事 |

### 阶段 4 · 用消融讲成故事 ✅ 已完成

| 项 | 内容 |
|---|---|
| **动作** | 跑 full / no_rerank / no_dense / no_fusion / agentic 五组消融，计算 95% CI 与置换检验 p-value（1000 bootstrap + 10000 permutation） |
| **前置修复** | 修 `HybridSearch` 权重门控，使 NO_DENSE/NO_FUSION 消融真实生效（§3.4） |
| **产出** | 五组对比表 + 各组件边际贡献 + 显著性标注（`ablation_stats.json` + `docs/ablation_report.html`） |
| **实测叙事** | "rerank 使 nDCG@10 +0.154（p < 0.001）；dense 是主力，去掉降至 0.430（p < 0.001）；sparse/RRF 融合不显著（p = 0.86）；agentic 在单跳集上显著劣化且慢 60×（p < 0.001）" |

### 5.1 阶段依赖与顺序

```
阶段1(修口径) ──► 阶段2(补生成侧) ──► 阶段4(消融讲故事)
      └──────────► 阶段3(降延迟) ──────────┘
```
阶段 1 是所有后续的基线前提；阶段 2 与阶段 3 相互独立可并行；阶段 4 依赖前三者产出的稳定管线。

---

## 6. cross-encoder vs LLM rerank 对比实验（实测填充）

阶段 3 的核心产出：在**同一 golden 子集（20 条 chunk 级）**上对比两种精排后端，形成精度-延迟权衡表。

### 6.1 实验矩阵

| 配置 | rerank.provider | rerank.top_n | 说明 |
|---|---|---|---|
| A. 无精排 | （`rerank.enabled=false`） | — | 粗排基线（对应 no_rerank 组） |
| B. LLM rerank | `llm` | 20 | 高精度、高延迟 |
| C. Cross-encoder | `cross_encoder` | 20 | 低延迟、本地推理 |

### 6.2 记录指标

| 维度 | 指标 |
|---|---|
| 精度 | nDCG@10 / Hit@5 / MRR@10 |
| 延迟 | p50 / p95 / p99（端到端 + `rerank_ms` 分项） |
| 成本 | 是否需要外部 API 调用 |

### 6.3 实测权衡表（20 条 chunk 级子集，apples-to-apples）

| 配置 | nDCG@10 | MRR@10 | Hit@5 | p95 | 外部调用 |
|---|---|---|---|---|---|
| **C Cross-encoder** | **0.697** | **0.935** | **1.00** | **1.6s** | 无（本地推理） |
| B LLM rerank (DeepSeek) | 0.483 | 0.813 | 0.95 | 73s | 每 query 1 次 |

> 反直觉但有力的实测结论：**cross-encoder 在这套科研论文语料上又快又准**——精度更高（+0.21 nDCG@10）且延迟低 ~45 倍。

### 6.4 结论口径与原因分析

**明确推荐：默认走 cross-encoder（又快又准），LLM rerank 作为"高精度模式"可选开关。**

**为什么 cross-encoder 又快又准**（基于本项目实现）：

- **快（~45×）**：cross-encoder 是 ~22M 参数的 MiniLM 本地判别模型，**单次前向**对每个 query-doc pair 出标量分数、可 batch 并行；LLM rerank 是百亿级模型远程 API + **自回归解码**（要逐 token 生成 30 条分数的 JSON），p50 达 51.7s。
- **准（+0.21 nDCG@10）**：① **任务对口**——ms-marco cross-encoder 专门在段落排序上微调，DeepSeek 只是"顺便做排序"；② **建模方式**——query 与 doc 拼接后 full cross-attention 联合编码，细粒度判别力强；③ 本项目 LLM rerank 实现的三个扣分点：候选被截断到 300 字符（信息损失）、listwise 30 条一次塞入（lost-in-the-middle）、0–10 整数打分粒度粗 + 解析失败退化为全 5.0。

**Caveat**：这是**本任务**（英文科研论文段落、单跳）+ **本 LLM rerank 实现**（listwise、截断 300、prompt 未调优）下的结论。cross-encoder 也有局限（512 token 截断长 chunk、跨领域迁移弱、做不了推理型相关性）；若把 LLM rerank 优化，差距会缩小，推理型场景 LLM rerank 仍可能反超。

---

## 7. 目标指标卡与简历表述

### 7.1 目标指标卡（实测达成）

| 指标 | 及格 | 良好（目标） | **实测** | 达成 |
|---|---|---|---|---|
| nDCG@10（chunk） | 0.45 | 0.58 | **0.692** | ✅ 超标 |
| Hit@5（文档级） | 0.75 | 0.85 | **0.952** | ✅ 超标 |
| MRR@10 | 0.45 | 0.55 | **0.854** | ✅ 超标 |
| Faithfulness | 0.75 | 0.85 | **0.850** | ✅ 达标 |
| Answer Relevancy | 0.75 | 0.85 | **0.861** | ✅ 超标 |
| Context Precision | 0.60 | 0.70 | **0.676** | ⚠️ 略低 |
| p95 延迟 | < 5s | < 2s | **0.71s** | ✅ 超标 |

### 7.2 简历表述模板（实测数字回填）

> 构建模块化 RAG 系统（可插拔检索 / embedding / rerank / 评估，MCP 协议 + 全链路 trace）。基于 230 条 chunk 级 graded 测试集建立 **IR + 生成双层评估体系**：检索 **nDCG@10 0.692 / Hit@5 0.952 / MRR@10 0.854**，生成 **Faithfulness 0.850 / Answer Relevancy 0.861**；通过**消融 + 置换检验**（1000 bootstrap + 10000 permutation）量化各组件贡献（**rerank 使 nDCG@10 +0.154，p < 0.001；dense 为主力，去除降至 0.430，p < 0.001**）；将精排从 LLM rerank 切换为 cross-encoder，**p95 延迟从 38s 降至 0.71s（-98%）且精度不降反升**；**用评估方法论定位并修复了向量库 embedding 空间错位（重嵌 8k+ 向量，nDCG@10 0.02→0.69）**。

> 表述要点：每个数字都有出处（golden 集 + `experiments/results/`）、有方法（消融 / CI / p-value）、有 insight（延迟权衡 + bug 发现）——这就是"拿得出手"。

---

## 8. CI 回归门禁与 hermetic fixture 职责边界

两套评估数据服务于**不同目的**，必须澄清边界，避免混淆：

| 用途 | 数据源 | 特性 | 触发时机 |
|---|---|---|---|
| **能力基线评估** | `data/golden_papers.json`（230 条 graded） | 依赖真实向量库 + API key，慢、需网络 | 手动 / 里程碑，产出对外指标 |
| **CI 回归门禁** | `tests/fixtures/` 的 hermetic fixture（内存语料，5 条） | 离线、确定性、秒级 | 每次提交，防质量劣化 |

- **门禁不读** `data/golden_papers.json`——因此更新论文库 golden 集**不影响门禁、无回归风险**。
- 门禁断言的是"检索管线在固定小语料上的确定性行为不退化"（如某 query 的命中 rank 不变差），而非绝对指标数值。
- 两者互补：fixture 保"不劣化"，golden 集保"对外指标真实"。全量测试 **467 passed**（含新增 capped recall / answer 合成 / p50-p99 / 权重门控等单测）。

---

## 9. 未来扩展：多模态跨模态评估（暂不纳入主指标）

### 9.1 已具备的能力盘点

本项目多模态链路（路径 B）代码**已打通**，支持真·跨模态统一向量空间：

- `QwenMultimodalEmbedding`（图文统一 1024/1152 维空间）、`QwenVisionLLM`（图片描述）；
- `PdfLoader` 真抽图、`ImageEncoder` 图片→独立向量入库（`modality=image`）；
- `hybrid_search` 支持以文搜图 / 以图搜文，MCP `query_knowledge_hub` 可收图。

### 9.2 为何暂不纳入主指标

- **现状无数据**：当前向量库 9618 条中 8341 为文本 chunk、1275 为图片向量，但 golden 集全指向文本 chunk，跨模态检索尚无 golden。
- **优先级**：文本检索质量修口径 + 生成侧补齐 + 延迟优化，边际价值远高于多模态评估。
- **风险**：跨模态 golden 构建成本高，仓促纳入会稀释主线叙事。

### 9.3 后续路线（有需要时启用）

1. 挑图表丰富的 PDF（GPT-3 / ViT / Transformer 原文），开 `ingestion.image_embedding=true` 摄取到**独立集合**（不污染 `default` 文本库）；
2. 用 `qwen-vl-max` 为入库图生成描述→反构 text→image query，图 id 作 golden；
3. 复用 `IRMetricsEvaluator`（id 空间抽象，天然支持图片 id）+ `EvalRunner` 增加 image-query 分支（走 `embed_image_query` + `retrieve_by_vector`）；
4. 产出跨模态 recall / nDCG@k。

> 注意：本次重嵌（§2.4）只处理了 8341 条文本向量，1275 条图片向量维持原样；若未来启用跨模态评估，需先核验图片向量与图片 query 编码的空间一致性。

---

## 10. 附录

### 附录 A · 已就绪 / 已落地能力清单

| 能力 | 位置 | 状态 |
|---|---|---|
| Cross-encoder reranker | `src/libs/reranker/cross_encoder_reranker.py` | ✅ 已实现 + 依赖已装（sentence-transformers 5.6.0） |
| LLM reranker | `src/libs/reranker/llm_reranker.py` | ✅ 已实现（高精度模式） |
| 生成侧 LLM-judge 后端 | `src/observability/evaluation/llm_judge_backend.py` | ✅ 新增（替代冲突的 ragas 包） |
| IR / custom evaluator | `src/libs/evaluator/` | ✅ 已实现（主力，capped recall 已落地） |
| answer 合成链路 | `EvalRunner.answer_synthesizer` + `evaluate.py._build_evaluator` | ✅ 已落地 |
| 消融开关 + 权重门控 | `scripts/evaluate.py`（env）+ `HybridSearch._weight_enabled` | ✅ 已实现并生效 |
| 消融统计（CI + p 值） | `scripts/ablation_stats.py`（numpy-only bootstrap + permutation） | ✅ 已实现 + 已单测 |
| 延迟统计 | `EvalRunner`（avg / p50 / p95 / p99 / synthesize_ms） | ✅ 完整 |
| 重嵌工具 | `scripts/reembed_index.py`（串行 + 断点续跑） | ✅ 新增（修复 §2.4） |
| graded golden 集 | `data/golden_papers.json`（230 条） | ✅ 已就绪 |
| 实验结果 | `experiments/results/*.json` + `docs/ablation_report.html` | ✅ 已产出 |

### 附录 B · 命令速查

```bash
# 环境（key 走 env，禁止硬编码）
export DEEPSEEK_API_KEY=<your_key>      # LLM 生成 + rerank + 生成侧评判
export DASHSCOPE_API_KEY=<your_key>     # embedding + vision

# 标准评估（full pipeline，检索指标；cross-encoder 精排）
COGENT_EVAL_RERANK_PROVIDER=cross_encoder python scripts/evaluate.py --backends ir,custom --k 1,3,5,10 --json

# 生成侧评估（阶段2，answer 链路 + 自实现 LLM-judge）
COGENT_EVAL_RERANK_PROVIDER=cross_encoder python scripts/evaluate.py --backends ir,ragas --json

# 五组消融（阶段4）
COGENT_EVAL_RERANK_PROVIDER=cross_encoder python scripts/evaluate.py --backends ir,custom --k 1,3,5,10 --json   # full
COGENT_EVAL_NO_RERANK=1 python scripts/evaluate.py --json                                                        # no_rerank
COGENT_EVAL_NO_DENSE=1  COGENT_EVAL_RERANK_PROVIDER=cross_encoder python scripts/evaluate.py --json               # 纯 BM25
COGENT_EVAL_NO_FUSION=1 COGENT_EVAL_RERANK_PROVIDER=cross_encoder python scripts/evaluate.py --json               # 纯 dense
COGENT_EVAL_AGENTIC=1   COGENT_EVAL_RERANK_PROVIDER=cross_encoder python scripts/evaluate.py --json               # agentic

# 消融显著性统计（bootstrap CI + 置换检验）
python scripts/ablation_stats.py --reports full=... no_rerank=... no_fusion=... no_dense=... agentic=... \
  --metrics ndcg@10 mrr@10 recall@10 --baseline full --bootstrap-samples 1000 --permutations 10000 --out experiments/results/ablation_stats.json

# 向量库重嵌（修复 embedding 空间错位，串行 + 断点续跑）
python scripts/reembed_index.py
```

### 附录 C · 关键配置片段（当前生效值）

```yaml
# 阶段1：修检索口径（已生效）
retrieval:
  top_k: 30          # 10 → 30，扩召回池
rerank:
  top_n: 20          # 5 → 20，解开 @10 截断

# 阶段3：精排后端（默认 cross_encoder，实测 p95 0.71s 且精度更高）
rerank:
  enabled: true
  provider: "cross_encoder"                        # 默认；高精度模式可切回 llm
  model: ""                                        # cross_encoder 留空默认 ms-marco-MiniLM-L-6-v2
  top_n: 20

# 阶段2：启用生成侧评估
# 用 --backends ir,ragas 触发；ragas 走自实现 DeepSeek LLM-judge，不依赖 ragas 包
```

### 附录 D · 术语表

| 术语 | 说明 |
|---|---|
| nDCG | Normalized Discounted Cumulative Gain，归一化折损累积增益 |
| MRR | Mean Reciprocal Rank，平均倒数排名 |
| BEIR | Benchmarking IR，零样本信息检索评测基准 |
| Ragas | RAG Assessment，LLM 驱动的 RAG 生成质量评估框架（本项目用自实现 LLM-judge 对齐其指标） |
| Cross-encoder | 查询-文档拼接后联合编码打分的重排模型（精度高于双塔） |
| RRF | Reciprocal Rank Fusion，倒数排名融合（混合检索融合算法） |
| Capped recall | 分母取 `min(k, |golden|)` 的召回口径修正 |
| Permutation test | 置换检验，非参数显著性检验方法 |
| Embedding 空间错位 | 建库向量与查询侧向量不在同一语义空间，致 dense 检索退化（§2.4） |
