# 开发进度追踪

> 自动更新 | 最后更新: 2026-06-12

## 总览

| 指标 | 数值 |
|------|------|
| 总任务数 | 68 |
| 已完成 | 68 |
| 进度 | 100% |
| 全量测试 | 423 passed |

---

## P1 — Agentic RAG（OPTIMIZATION_SPEC §3，已完成）

将「一次性检索 → 拼引用」升级为「LLM 作为 agent 主动决策检索」并在服务端合成带引用答案。新增独立 MCP 工具 `agentic_query`，不改动 `query_knowledge_hub`（保留为快速直检/降级路径）。

| 里程碑 | 内容 | 状态 |
|------|------|------|
| M-C1 | `AgentSettings`(默认关) + `agent_types` + `CollectionRegistry` + `AnswerSynthesizer`(服务端首次具备 LLM 合成答案) + `QueryRouter`(3.1，白名单+保守降级) + `AgenticRAG` 最小闭环 + `agentic_query` 工具并注册 | ✅ |
| M-C2 | `QueryTransformer`(3.2 改写/分解+去重+降级) + 多跳检索循环(3.3，`max_hops`/`max_context_chunks` 预算、去重累积、`agent_hop_n` trace) | ✅ |
| M-C3 | `Reflector`(3.4 self-correction，CRAG 思路) + reflect→follow-up→重检回路(`max_reflect_rounds`、失败保守跳出) | ✅ |
| M-C4 | 全量回归(423 passed，+56 离线 agent 用例) + 文档更新 | ✅ |

- **关键设计**：全程降级（任一 LLM 步骤异常→单次混合检索，绝不报错）；硬上限防成本失控；JSON 容错（截取 `[..]`/`{..}` + fallback，绝不 eval）；collection 白名单防幻觉越权；`agent.enabled=False` 委托旧工具，行为与升级前完全一致。
- **复用而非重写**：检索走 `HybridSearch`、重排走 `QueryReranker`、LLM 走 `LLMFactory`、trace 走 `TraceContext`，不改动检索内核。
- **详见** `docs/P1_AGENTIC_RAG_SPEC.md`。

---

## 多模态升级（路径B：真·跨模态向量）

基于 Qwen / DashScope 为项目赋予真正的多模态能力（图文统一向量空间 + query 接收图片）。

| 项 | 内容 | 状态 |
|------|------|------|
| M1 | `QwenMultimodalEmbedding`（DashScope multimodal-embedding-v1，图文统一空间，逐条编码、auto_truncation）+ `BaseEmbedding.embed_image` + 工厂注册 + `dashscope` 依赖 | ✅ |
| M2 | `QwenVisionLLM`(qwen-vl-max) + `VisionLLMFactory`，解除 `ImageCaptioner` 对 Azure 硬编码 | ✅ |
| M3 | `PdfLoader` 用 pypdfium2 真抽图→落盘→填 `images` + `[IMAGE: id]` 占位；修复 `image_refs` 隐藏断点 | ✅ |
| M4 | `ImageEncoder` 将文档图片编码为独立向量记录(`modality=image`)入同一 collection，pipeline 接入 | ✅ |
| M5 | `DenseRetriever.retrieve_by_vector`/`embed_image_query` + `HybridSearch` image 入参（纯图 dense-only） | ✅ |
| M6 | `query_knowledge_hub` 新增 `image` 入参（base64/受限路径）+ 安全校验（防穿越/SSRF/限大小）；`scripts/query.py --image` | ✅ |
| M7 | 配置切换到多模态方案；单测 +21；真实接口验证图文均落 1024 维统一空间 | ✅ |

> 切换多模态向量后旧文本向量库不兼容，需 `rm -rf data/db/chroma data/db/bm25` 重新摄取。

---

## 阶段进度

| 阶段 | 说明 | 任务数 | 完成 | 进度 |
|------|------|--------|------|------|
| A | 工程骨架与测试基座 | 3 | 3 | ✅ 100% |
| B | Libs 可插拔层 | 16 | 16 | ✅ 100% |
| C | Ingestion Pipeline | 15 | 15 | ✅ 100% |
| D | Retrieval | 7 | 7 | ✅ 100% |
| E | MCP Server | 6 | 6 | ✅ 100% |
| F | Trace 基础设施 | 5 | 5 | ✅ 100% |
| G | Dashboard | 6 | 6 | ✅ 100% |
| H | 评估体系 | 5 | 5 | ✅ 100% |
| I | 端到端验收 | 5 | 5 | ✅ 100% |

---

## 🎉 全部阶段完成

| 编号 | 任务 | 状态 |
|------|------|------|
| I1 | E2E：MCP Client 侧调用模拟 | ✅ |
| I2 | E2E：Dashboard 冒烟测试 | ✅ |
| I3 | 完善 README | ✅ |
| I4 | 清理接口一致性（契约测试补齐） | ✅ |
| I5 | 全链路 E2E 验收 | ✅ |

项目 68 个子任务全部交付，三层测试体系（Unit / Integration / E2E）全绿。

---

## 评估体系升级（H2：对标业界 IR 指标）

在原阶段 H（hit_rate/mrr/recall_completeness，文档级、31 条）基础上升级为对标 BEIR/Ragas/DeepEval 的成熟体系，详见 `docs/EVAL_UPGRADE_SPEC.md`：

| 编号 | 交付 | 状态 |
|------|------|------|
| H2-1 | `IRMetricsEvaluator`（recall/precision/mrr/map/**ndcg**@k，graded 0-3，离线无 key）+ `EvalInput.relevance` + `EvalRunner` @k 明细/p95 + CLI `--k` + 工厂注册 | ✅ |
| H2-2 | CI 回归门禁 `tests/e2e/test_eval_regression.py`（hermetic，断言 `ir.ndcg@5 >= baseline-0.02`） | ✅ |
| H2-3 | 测试集扩充脚本 `scripts/gen_testset.py`（llamaindex/ragas 双路线）+ `scripts/merge_testset.py`（chunk 级+graded，向后兼容；无 key 优雅退出） | ✅ 脚本就绪（扩充执行需 LLM key + 已摄取 DB） |
| H2-4 | 消融显著性 `scripts/ablation_stats.py`（bootstrap 95% CI + paired permutation p-value，仅 numpy）+ 升级 `run_ablation.sh` | ✅ |
| H2-5 | 生成侧真跑（Ragas 后端，有 key 时 faithfulness/answer_relevancy/context_recall；离线降级） | ✅ 复用现有 RagasEvaluator |
| H2-6 | 公开基准锚定 `scripts/run_beir.py`（BEIR 子集，同口径 IRMetricsEvaluator；beir 为可选依赖） | ✅ 脚本就绪 |

- **口径**：nDCG 用 `DCG@k=Σ(2^rel_i−1)/log2(i+1)`，IDCG 取 golden grade 降序理想排列，IDCG=0 记 0；`relevance` 为空自动退化二值 nDCG（旧 31 条零改动兼容）。
- **配置**：`config/settings.yaml::evaluation` 新增 `ks: [1,3,5,10]`、`bootstrap_samples: 1000`，`backends` 默认 `["ir","custom"]`。
- **安全**：`settings.yaml` 硬编码 DashScope key 已清空，key 仅从 `DASHSCOPE_API_KEY`/`OPENAI_API_KEY` 环境变量注入。
- **可选依赖**：`pip install -e ".[eval]"`（pytrec_eval / beir / ragas）。

---

## 环境信息

- **Python**: 3.13
- **虚拟环境**: `.venv/`
- **安装依赖**: `pip install -e ".[dev]"`（评估附加：`pip install -e ".[eval]"`）
- **运行测试**: `source .venv/bin/activate && python -m pytest tests/ -q`
- **运行评估**: `python scripts/evaluate.py --backends ir,custom --k 1,3,5,10 [--json]`
- **消融显著性**: `bash scripts/run_ablation.sh`（输出 95% CI + p-value）
- **启动 Dashboard**: `python scripts/start_dashboard.py`
- **MCP 入口**: `python main.py`（stdio）
