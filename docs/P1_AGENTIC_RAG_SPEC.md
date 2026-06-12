# P1 — Agentic RAG 实施 Spec

> 状态：待评审（draft）｜创建：2026-06-11｜测试基线：367 passed（P0 完成后）
> 来源：`docs/OPTIMIZATION_SPEC.md` §3（P1 最热加分项）
> 用途：本文件是 P1 的执行蓝图，定稿后从「§9 里程碑与任务清单」开工。

---

## 1. 目标与范围

把现有「一次性检索 → 拼引用返回」升级为「**LLM 作为 agent 主动决策检索**」的 Agentic RAG，覆盖 `OPTIMIZATION_SPEC` §3 全部五步：

| 步骤 | 能力 | 一句话 |
|---|---|---|
| 3.1 | 检索决策 route | LLM 先判断「要不要检索 / 查哪个 collection」，简单问题直答 |
| 3.2 | 查询改写 / 分解 | 复杂问题拆成多个子查询，分别检索再汇总 |
| 3.3 | 多跳检索 multi-hop | 基于首轮结果决定是否追加检索（迭代式） |
| 3.4 | self-correction / 反思 | 评估检索结果是否充分，不足则重检（Self-RAG / CRAG 思路） |
| 3.5 | 工具化 | 把检索封装为 agent 可调用 tool，MCP 暴露入口 |

**范围内**：上述 5 步 + 服务端答案合成层（self-correction 的前置必需）+ `AgentSettings` 配置 + 新 MCP 工具 `agentic_query` + trace 多步可视化 + 全套离线单测。

**范围外（明确不做）**：`OPTIMIZATION_SPEC` §5 伪需求（独立以文搜图、路径B 跨模态向量检索加重投入）；P2 公式提取；P3 ColPali。2.3 图 caption 已在 P0 完成，本期不重复。

---

## 2. 现状分析（设计依据，已实地核实）

| 维度 | 现状 | 落点/行号 | 对 P1 的影响 |
|---|---|---|---|
| 答案生成 | **无服务端 LLM 合成**；`ResponseBuilder._render_markdown` 纯字符串拼引用 | `src/core/response/response_builder.py:68` | self-correction 需 LLM「试答+判断充分性」→ **必须新增 AnswerSynthesizer** |
| 检索 | `HybridSearch.search(query, top_k, filters, trace, image) -> list[RetrievalResult]` | `src/core/query_engine/hybrid_search.py:43` | 直接复用为 agent 的检索 tool |
| 重排 | `QueryReranker.rerank(query, results, trace) -> list[RetrievalResult]`，失败回退原序 | `src/core/query_engine/reranker.py:27` | 多跳每轮可选重排 |
| 文本 LLM | `LLMFactory.create(settings.llm).chat([ChatMessage]) -> ChatResponse`；provider: openai/azure/ollama/deepseek/qwen | `src/libs/llm/llm_factory.py:76` | agent 决策/合成统一走它 |
| 视觉 LLM | `VisionLLMFactory.create(settings.vision_llm).chat_with_image(text, image)` | `src/libs/llm/vision_llm_factory.py:58` | 可选：含图证据的反思 |
| Trace | `TraceContext(trace_type="query")` + `record_stage(name, method, elapsed_ms, **details)` / `with trace.stage(...)` | `src/core/trace/trace_context.py:53` | 新增 route/rewrite/hop_n/reflect/synthesize stage |
| 配置 | `_build_dataclass` **忽略未知字段**；`load_settings` 逐段组装 | `src/core/settings.py:206` | 加 `AgentSettings` 零侵入，旧配置兼容 |
| Prompt | 模板放 `config/prompts/*.txt`，`Path(...).read_text` + `.format(...)`，缺失有内置 fallback | `src/libs/reranker/llm_reranker.py:25` | 沿用同范式新增 agent prompts |
| MCP 工具 | `build_protocol_handler` 注册 query_knowledge_hub / list_collections / get_document_summary | `src/mcp_server/server.py:41` | 新增 `agentic_query` 工具并注册 |
| collection 枚举 | **无向量库原生枚举**；`ListCollectionsTool` 扫 `data/documents` 目录 | `src/mcp_server/tools/list_collections.py` | route 选 collection 的能力缺口，需补 |

**核心结论**：基础检索/LLM/trace/工厂/配置范式齐备，P1 是「在其之上加一层 agent 编排 + 一个答案合成器」，**不改动现有检索内核**，新能力作为可切换 provider/可关开关，保留 `query_knowledge_hub` 原路径作为回退。

---

## 3. 设计原则与约束（继承 OPTIMIZATION_SPEC §7）

1. **可插拔 + 可回退**：Agentic 作为**独立 MCP 工具** `agentic_query`，不动 `query_knowledge_hub`（保留为快速直检路径与降级目标）。所有子能力（route/rewrite/multihop/reflect/synthesis）各有开关，可单独启用。
2. **不破坏现有 367 passed**；新功能补离线单测（mock LLM/检索，不触网）。
3. **循环必须有上限 + 降级**：`max_hops` / `max_subqueries` / `max_reflect_rounds` 硬上限；任一 LLM 步骤异常 → 降级为传统单次混合检索，绝不抛错给用户。
4. **成本/延迟可控**：默认 `enabled=False`（不影响现有行为）；启用后每步 trace 记录耗时与 token，便于评估。
5. **安全**：复用现有 query 安全约束；LLM 输出严格按 JSON 解析并容错（仿 `LLMReranker._parse_scores` 的"截取 `[`..`]` + 失败 fallback"），绝不 `eval`；不因模型幻觉触发越权检索（collection 白名单校验）。
6. 复用而非重写：检索走 `HybridSearch`，重排走 `QueryReranker`，LLM 走 `LLMFactory`，trace 走 `TraceContext`。

---

## 4. 总体架构

### 4.1 模块结构（新增 `src/core/agent/`）

```
src/core/agent/
  __init__.py
  agent_types.py          # 数据类：RouteDecision / SubQuery / HopResult / ReflectVerdict / AgentResult
  agentic_rag.py          # AgenticRAG —— 主编排器（route→transform→retrieve-loop→reflect→synthesize）
  router.py               # QueryRouter（3.1）：是否检索 / 选 collection / 简单问题直答
  query_transformer.py    # QueryTransformer（3.2）：查询改写 + 分解为子查询
  reflector.py            # Reflector（3.4）：评估上下文充分性 → 产出补充子查询
  answer_synthesizer.py   # AnswerSynthesizer：LLM 合成最终答案 + 内联引用（reflect 与终答共用）
  collection_registry.py  # collection 枚举（补 route 的能力缺口）

config/prompts/
  agent_route.txt         # 路由决策
  agent_rewrite.txt       # 查询改写/分解
  agent_reflect.txt       # 充分性评估
  agent_answer.txt        # 答案合成

src/mcp_server/tools/
  agentic_query.py        # 新 MCP 工具（3.5 工具化），编排 AgenticRAG 并复用 ResponseBuilder/Multimodal
```

> 多跳循环逻辑内聚在 `agentic_rag.py`（不单列 `retrieval_loop.py`，避免过度拆分；检索本身复用 `HybridSearch`）。

### 4.2 数据流（一次 agentic_query）

```
agentic_query(query, collection?, image?, top_k?)
        │
        ▼
[route] QueryRouter.decide(query, available_collections)
        ├─ need_retrieval=False → AnswerSynthesizer 直答（无检索）→ 返回
        └─ need_retrieval=True, target_collections=[...]
                │
                ▼
[rewrite] QueryTransformer.transform(query) → [sub_q1, sub_q2, ...]  (≤ max_subqueries)
                │
                ▼
[retrieve-loop]  hop = 1..max_hops:
        for sub_q in pending_subqueries:
            HybridSearch.search(sub_q, top_k, {collection}) → results
            (可选) QueryReranker.rerank
        合并去重累积 context
                │
                ▼
[reflect] Reflector.assess(query, context)  (≤ max_reflect_rounds)
        ├─ sufficient=True  → 跳出循环
        └─ sufficient=False → 产出 follow_up_queries → 回到 retrieve-loop（hop+1）
                │
                ▼
[synthesize] AnswerSynthesizer.answer(query, context) → 答案 + 选中引用
                │
                ▼
ResponseBuilder/Multimodal 组装 MCPToolResult（答案文本 + citations + 证据图）
trace.finish() + TraceCollector.collect(trace)
```

每个 `[stage]` 用 `trace.record_stage(...)` 记录（method=组件名，details 含决策摘要/子查询数/命中数/耗时）。

---

## 5. 详细设计（逐组件）

### 5.1 QueryRouter（3.1 检索决策）

- **职责**：给定 query + 可用 collection 列表，决定 (a) 是否需要检索（寒暄/常识/纯计算 → 直答）；(b) 命中哪个/哪些 collection。
- **接口**：
  ```python
  @dataclass
  class RouteDecision:
      need_retrieval: bool
      target_collections: list[str]      # 已按白名单校验，空表示全部
      reasoning: str
      direct_answer: str = ""            # need_retrieval=False 时可由 LLM 直接给

  class QueryRouter:
      def __init__(self, settings, llm=None): ...
      def decide(self, query: str, available_collections: list[str], trace=None) -> RouteDecision: ...
  ```
- **实现**：prompt `agent_route.txt`，要求 LLM 输出 JSON `{"need_retrieval": bool, "collections": [...], "reasoning": "..."}`；JSON 解析失败 → **保守降级**：`need_retrieval=True, target_collections=[]`（即默认检索全部，绝不漏检）。
- **collection 来源**：`CollectionRegistry`（§5.6）。LLM 给出的 collection 必须 ∈ available，否则丢弃（防幻觉越权）。

### 5.2 QueryTransformer（3.2 查询改写 / 分解）

- **职责**：把口语化/复合问题改写为更利于检索的表达，并在需要时分解为多个子查询。
- **接口**：
  ```python
  @dataclass
  class SubQuery:
      text: str
      purpose: str = ""

  class QueryTransformer:
      def transform(self, query: str, trace=None) -> list[SubQuery]: ...
  ```
- **实现**：prompt `agent_rewrite.txt`，输出 JSON 数组（≤ `max_subqueries`，默认 3）。失败/空 → 降级为 `[SubQuery(text=query)]`（原样单查询，保证可用）。
- **去重**：子查询文本归一化后去重，避免重复检索。

### 5.3 多跳检索（3.3，内聚于 AgenticRAG）

- **循环**：`for hop in range(1, max_hops+1)`，每跳对 pending 子查询逐个 `HybridSearch.search`，可选 `QueryReranker.rerank`，结果按 `chunk_id` 去重累积进 `context`（保留最高分）。
- **跳出条件**：(a) `Reflector` 判定 sufficient；(b) 无新增 follow-up；(c) 触达 `max_hops`。
- **预算**：每跳累计命中上限（如 `max_context_chunks`，默认 20），防上下文爆炸。

### 5.4 Reflector（3.4 self-correction / 反思）

- **职责**：评估「当前 context 是否足以回答原始 query」；不足时产出补充子查询（CRAG：判定 correct/ambiguous/incorrect → 决定重检）。
- **接口**：
  ```python
  @dataclass
  class ReflectVerdict:
      sufficient: bool
      follow_up_queries: list[str]
      reasoning: str

  class Reflector:
      def assess(self, query: str, context: list[RetrievalResult], trace=None) -> ReflectVerdict: ...
  ```
- **实现**：prompt `agent_reflect.txt`，输入 query + 编号的 context 摘要（每条截断，仿 reranker 的 `text[:300]`），输出 JSON `{"sufficient": bool, "follow_up": [...], "reasoning": "..."}`。失败 → `sufficient=True`（保守跳出，避免死循环/成本失控）。
- **上限**：`max_reflect_rounds`（默认 2）。

### 5.5 AnswerSynthesizer（答案合成层，新增基础能力）

- **职责**：基于 query + 最终 context，用 LLM 生成答案，并标注引用编号 `[n]` 对应到 `RetrievalResult`。reflect 的「试答」与终答共用。
- **接口**：
  ```python
  @dataclass
  class SynthResult:
      answer: str
      used_citation_ids: list[int]     # 命中的 context 序号

  class AnswerSynthesizer:
      def answer(self, query: str, context: list[RetrievalResult], trace=None) -> SynthResult: ...
  ```
- **实现**：prompt `agent_answer.txt`，要求「仅依据提供的编号片段作答，引用处标 `[n]`，无依据则说明未找到」（抑制幻觉）。
- **与 ResponseBuilder 协作**：合成答案作为 `content[0]` 文本；`CitationGenerator` 仍从 `used` 的 `RetrievalResult` 生成结构化 citations；证据图经 `MultimodalAssembler` 追加。即 **agentic_query 返回真·答案**（区别于 query_knowledge_hub 的"检索片段列表"）。

### 5.6 CollectionRegistry（补 route 能力缺口）

- **职责**：提供可用 collection 列表给 router。
- **实现（轻量，本期）**：复用 `ListCollectionsTool` 的目录扫描逻辑，封装 `list_collections() -> list[str]`。
- **演进（可选，标注 TODO）**：未来给 `BaseVectorStore`/`ChromaStore` 增原生 `list_collections()`，本期不做以控范围。

### 5.7 AgenticRAG（主编排器）

```python
class AgenticRAG:
    def __init__(self, settings, hybrid_search=None, reranker=None,
                 router=None, transformer=None, reflector=None,
                 synthesizer=None, registry=None):
        ...  # 依赖全部可注入（测试友好），缺省惰性构造

    def run(self, query: str, collection: str | None = None,
            image: str | bytes | None = None, top_k: int | None = None,
            trace: TraceContext | None = None) -> AgentResult: ...
```

- `AgentResult{answer, results, citations_meta, steps}`：`steps` 记录每步决策，供 trace/调试。
- **全程 try/except 降级**：任一阶段异常 → 回退到「一次 `HybridSearch.search` + `AnswerSynthesizer`（或直接 ResponseBuilder）」，记 `agent_fallback` stage。

---

## 6. 配置设计（`AgentSettings`）

`src/core/settings.py` 新增 dataclass 并挂到 `Settings`，`load_settings` 加一行 `_build_dataclass(AgentSettings, raw.get("agent"))`：

```python
@dataclass
class AgentSettings:
    """Agentic RAG configuration（默认全关，零侵入兼容旧行为）。"""
    enabled: bool = False              # 总开关（agentic_query 工具是否生效）
    route_enabled: bool = True         # 3.1
    rewrite_enabled: bool = True       # 3.2
    multihop_enabled: bool = True      # 3.3
    reflect_enabled: bool = True       # 3.4
    synthesize_answer: bool = True     # 服务端 LLM 合成最终答案
    max_hops: int = 3
    max_subqueries: int = 3
    max_reflect_rounds: int = 2
    max_context_chunks: int = 20
    retrieval_top_k: int = 5           # 每个子查询的检索条数
    answer_model: str = ""             # 空则复用 settings.llm.model
```

`config/settings.yaml` 追加（示例）：

```yaml
agent:
  enabled: true
  route_enabled: true
  rewrite_enabled: true
  multihop_enabled: true
  reflect_enabled: true
  synthesize_answer: true
  max_hops: 3
  max_subqueries: 3
  max_reflect_rounds: 2
  max_context_chunks: 20
  retrieval_top_k: 5
```

---

## 7. MCP 工具（3.5 工具化）

新增 `src/mcp_server/tools/agentic_query.py`，并在 `server.py:build_protocol_handler` 注册第 4 个工具。

```python
TOOL_NAME = "agentic_query"
INPUT_SCHEMA = {
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "User question (natural language)."},
    "collection": {"type": "string"},
    "image": {"type": "string"},
    "top_k": {"type": "integer", "minimum": 1},
  },
  "required": ["query"],
}
```

- 行为：构造 `TraceContext("query")` → `AgenticRAG.run(...)` → 用 `ResponseBuilder` + `MultimodalAssembler` 组装 `MCPToolResult`（答案 + citations + 证据图）→ `TraceCollector.collect`。
- `settings.agent.enabled=False` 时：工具仍注册但内部直接委托 `QueryKnowledgeHubTool` 行为（或不注册，二选一，建议**委托**以保持工具列表稳定）。
- `query_knowledge_hub` **保持不变**，定位为「面向外部 client LLM 的检索片段工具」；`agentic_query` 定位为「服务端自带 agent 推理、直接给答案」。两者并存，文档说明取舍。

---

## 8. Trace 集成

复用 `TraceContext.record_stage` / `with trace.stage(...)`，新增 stage 名（与现有 dense_retrieval/sparse_retrieval/fusion/rerank 并列）：

| stage | method | 关键 details |
|---|---|---|
| `agent_route` | router | need_retrieval, collections, reasoning |
| `agent_rewrite` | transformer | n_subqueries, subqueries |
| `agent_hop_{n}` | hybrid_search | subquery, n_hits |
| `agent_reflect_{n}` | reflector | sufficient, n_followup |
| `agent_synthesize` | synthesizer | answer_len, n_citations |
| `agent_fallback` | agentic_rag | reason（降级时） |

Dashboard 的 `TraceService` 无需改动即可展示（按 stages 渲染）；可在后续小步增强 agent 专属视图（本期不做）。

---

## 9. 里程碑与任务清单（建议推进顺序）

> 原则：每步可独立合入、独立测试、独立开关；先打通最小回路再叠加反思/多跳。

### M-C1 基础设施 + 答案合成 + route（最小可用闭环）
1. `AgentSettings` 接入 `settings.py` / `settings.yaml`；`agent_types.py` 数据类。
2. `CollectionRegistry`（复用 ListCollections 扫描）。
3. `AnswerSynthesizer` + `agent_answer.txt`（服务端首次具备 LLM 合成答案）。
4. `QueryRouter` + `agent_route.txt`（含白名单校验 + 保守降级）。
5. `AgenticRAG` 骨架：route →（单查询）retrieve → synthesize；全程降级。
6. `agentic_query` 工具 + `server.py` 注册。
7. 单测：router/synthesizer/registry/工具（mock LLM+HybridSearch，离线）。

### M-C2 查询改写/分解 + 多跳
8. `QueryTransformer` + `agent_rewrite.txt`，接入编排（子查询并发或顺序检索 + 去重累积）。
9. 多跳循环 + `max_hops`/`max_context_chunks` 预算 + trace `agent_hop_n`。
10. 单测：transformer、去重、跳数上限、上下文预算。

### M-C3 反思/自纠正
11. `Reflector` + `agent_reflect.txt`，接入 reflect→follow-up→重检回路 + `max_reflect_rounds`。
12. 单测：sufficient 跳出、follow-up 触发重检、反思上限、失败保守跳出。

### M-C4 收尾
13. README/PROGRESS/OPTIMIZATION_SPEC 更新（标 P1 完成、用法、取舍说明）。
14. （用户本地）清库重摄非必需（agent 不改摄取）；跑 `scripts/evaluate.py` 对比多跳问题；人工 top-N 对比。

---

## 10. 测试策略

- **全离线**：注入 fake `BaseLLM`（返回预设 JSON 决策/答案）、fake `HybridSearch`（返回固定 `RetrievalResult`），不触网、不依赖向量库。
- **组件单测**：router 决策解析与白名单、transformer 分解与降级、reflector 判定与上限、synthesizer 引用标注、registry 列举。
- **编排集成测**（确定性）：
  - route 直答路径（need_retrieval=False）；
  - 单跳充分 → 一次检索即合成；
  - 多跳：首轮不足 → reflect 触发 follow-up → 二跳补全；
  - 降级：LLM 抛错 → 回退单次检索不报错；
  - 上限：max_hops/max_reflect_rounds/max_context_chunks 生效。
- **JSON 容错测**：模型返回脏文本（前后有解释）仍能截取解析；完全无法解析走 fallback。
- 目标：现有 **367 不回归**，新增约 30–40 用例。

---

## 11. 验收标准（对齐 OPTIMIZATION_SPEC §3 验收）

1. 多跳问题（需 2+ 证据片段）上，`agentic_query` 答案完整度/正确率优于 `query_knowledge_hub`（人工对比 + golden 多跳样例）。
2. trace 能看到 route/rewrite/hop_n/reflect/synthesize 多步决策与耗时。
3. 所有循环有硬上限，LLM 异常时稳定降级、不抛错给用户。
4. `agent.enabled=False` 时系统行为与现状完全一致（回归保护）。
5. `pytest` 全绿不回归；新功能离线单测覆盖主路径 + 降级 + 上限。

---

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 延迟/token 成本上升（多次 LLM 调用） | 默认关；上限严格；route 可让简单问题直答省检索；trace 暴露成本 |
| LLM 输出非法 JSON | 仿 `LLMReranker._parse_scores` 截取+容错；失败保守降级（router 默认检索、reflect 默认充分） |
| 多跳死循环 | `max_hops` + `max_reflect_rounds` 双上限 + 「无新增 follow-up 即停」 |
| route 选错/幻觉 collection | 白名单校验（∈ available 才采纳），否则检索全部 |
| 答案幻觉 | synthesizer prompt 强约束「仅依据编号片段、无依据则说明未找到」+ 引用编号回链 |
| 破坏现有工具 | 新增独立工具，不改 query_knowledge_hub；enabled=False 委托旧行为 |

---

## 13. 待用户确认的关键决策（开工前）

1. **答案返回形态**：`agentic_query` 是否**在服务端直接返回 LLM 合成答案**（本 spec 推荐：是，因 self-correction 必需）？还是仅返回「多跳聚合后的检索片段」交由 client LLM 合成？
2. **工具策略**：新增独立 `agentic_query`（推荐，互不影响）vs 给 `query_knowledge_hub` 加 `agentic` 开关？
3. **本期深度**：一次做满 M-C1~M-C3（route+rewrite+multihop+reflect 全量），还是先交付 M-C1（route+合成最小闭环）验证效果再续？
4. **验证方式**：沿用 P0 约定——本期只代码+离线单测、不触网不清库（重摄非必需，因 agent 不改摄取；评估由你本地跑）？

---

## 附：面试叙事价值（OPTIMIZATION_SPEC §3 标 ★★★★★）

完成后可讲清：**Naive → Advanced/Modular → Agentic RAG** 的演进；route/rewrite/multi-hop/self-correction（Self-RAG / CRAG）的工程化落地；如何在「可插拔架构 + 严格降级 + trace 可观测」约束下控制 agent 的成本与可靠性——这是当前最稀缺的加分点。
