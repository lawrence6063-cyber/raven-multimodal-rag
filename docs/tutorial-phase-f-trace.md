# 阶段 F 教学文档：Trace 基础设施与链路打点

> **目标读者**：初学者（会基础 Python，了解函数/类/装饰器）。
> **学完收获**：理解「为什么要 trace」、掌握从零搭建轻量级可追踪系统的完整思路，并能读懂和修改本项目中的 trace 代码。

---

## 📖 目录

1. [为什么需要 Trace？（概念篇）](#1-为什么需要-trace概念篇)
2. [架构总览与文件地图](#2-架构总览与文件地图)
3. [F1：TraceContext — 数据容器](#3-f1tracecontext--数据容器)
4. [F2：结构化日志 — 持久化到文件](#4-f2结构化日志--持久化到文件)
5. [F3：Query 链路打点](#5-f3query-链路打点)
6. [F4：Ingestion 链路打点](#6-f4ingestion-链路打点)
7. [F5：Pipeline 进度回调](#7-f5pipeline-进度回调)
8. [动手实验：跑测试 & 看输出](#8-动手实验跑测试--看输出)
9. [设计决策 Q&A](#9-设计决策-qa)
10. [小结与练习题](#10-小结与练习题)

---

## 1. 为什么需要 Trace？（概念篇）

### 1.1 现实问题

想象你搭建了一个 RAG（检索增强生成）系统：
- 用户提问 → 查询处理 → 向量检索 → 稀疏检索 → 融合 → 重排 → 返回结果

当结果不好时，你怎么知道**哪一步出了问题**？

| 没有 trace | 有 trace |
|-----------|----------|
| "不知道为什么结果不好" | "dense_retrieval 花了 800ms，只返回了 2 条结果" |
| 靠 print 调试 | 结构化记录，自动化分析 |
| 上线后完全黑箱 | 每次查询都可追溯 |

### 1.2 核心概念

```
┌─────────────────────────────────────────────────────┐
│                    Trace（一次执行）                  │
│                                                     │
│  trace_id:  "a1b2c3d4..."                           │
│  trace_type: "query"                                │
│  started_at: "2026-06-10T12:00:00Z"                 │
│  finished_at: "2026-06-10T12:00:01.5Z"              │
│  total_elapsed_ms: 1500                             │
│                                                     │
│  stages: [                                          │
│    { name: "query_processing", elapsed_ms: 5 }      │
│    { name: "dense_retrieval",  elapsed_ms: 200 }    │
│    { name: "sparse_retrieval", elapsed_ms: 100 }    │
│    { name: "fusion",           elapsed_ms: 3 }      │
│    { name: "rerank",           elapsed_ms: 800 }    │
│  ]                                                  │
└─────────────────────────────────────────────────────┘
```

**Trace = 一次完整执行的"体检报告"**，里面包含多个 **Stage（阶段）**，每个阶段记录：
- `name`：做了什么（如 "dense_retrieval"）
- `method`：用了什么方法/提供商（如 "openai"）
- `elapsed_ms`：花了多少毫秒
- 可选附加字段（如 `results=5` 表示返回了 5 条）

### 1.3 设计原则

| 原则 | 说明 |
|------|------|
| **不阻塞主流程** | trace 记录失败绝不影响正常的查询/写入 |
| **不污染 stdout** | MCP Server 需要 stdout 纯净，trace 只写文件 |
| **可选注入** | `trace` 参数默认为 `None`，旧代码无需改动 |
| **轻量级** | 不依赖 OpenTelemetry 等重型框架，纯 Python 实现 |

---

## 2. 架构总览与文件地图

```
src/
├── core/
│   └── trace/
│       ├── __init__.py
│       ├── trace_context.py    ← F1: 数据容器
│       └── trace_collector.py  ← F1/F2: 持久化桥梁
├── observability/
│   └── logger.py              ← F2: 新增 JSONFormatter / write_trace
├── core/query_engine/
│   ├── hybrid_search.py       ← F3: 新增 trace 参数 + 打点
│   └── reranker.py            ← F3: 新增 trace 参数 + 打点
├── ingestion/
│   └── pipeline.py            ← F4: 新增 trace 参数 + 打点
└── mcp_server/tools/
    └── query_knowledge_hub.py ← F3: 创建 + 管理 trace 生命周期

tests/unit/
├── test_trace_context.py      ← F1 测试
├── test_jsonl_logger.py       ← F2 测试
├── test_query_trace.py        ← F3 测试
├── test_ingestion_trace.py    ← F4 测试
└── test_pipeline_progress.py  ← F5 测试
```

**数据流**：
```
创建 TraceContext → 传入业务方法 → 各阶段 record_stage → finish() → TraceCollector.collect() → write_trace → logs/traces.jsonl
```

---

## 3. F1：TraceContext — 数据容器

> 📁 文件：`src/core/trace/trace_context.py`

### 3.1 类结构

```python
class TraceContext:
    """累积一次 query 或 ingestion 执行中的各阶段计时数据。"""

    def __init__(self, trace_type: str = "query"):
        # trace_type 只允许 "query" 或 "ingestion"
        self.trace_id: str = uuid.uuid4().hex      # 唯一标识
        self.trace_type: str = trace_type           # 执行类型
        self._started_ms: float = _now_ms()         # 开始时间
        self._finished_ms: float | None = None      # 结束时间
        self.stages: list[dict] = []                # 阶段列表
```

### 3.2 核心方法详解

#### `record_stage(name, method, elapsed_ms, **details)`

最基础的记录方式——手动传入计时结果：

```python
# 手动计时示例
start = time.perf_counter()
results = dense_retriever.retrieve(query)
elapsed = (time.perf_counter() - start) * 1000.0

trace.record_stage(
    "dense_retrieval",         # 阶段名
    method="embedding",        # 使用了什么方法
    elapsed_ms=elapsed,        # 耗时（毫秒）
    results=len(results),      # 附加信息：返回了多少条
)
```

**原理**：就是往 `self.stages` 列表里追加一个字典。

#### `stage(name, method)` — 上下文管理器（更优雅）

自动计时，不需要手写 `time.perf_counter()`：

```python
with trace.stage("dense_retrieval", method="openai") as extra:
    results = retriever.retrieve(query)
    extra["hits"] = len(results)  # 在 with 块内动态添加信息
# 退出 with 时自动计算 elapsed_ms 并 record_stage
```

**实现原理**：

```python
@contextmanager
def stage(self, name, method="", **details):
    extra = dict(details)
    start = time.perf_counter()
    try:
        yield extra  # 把 extra 字典交给用户填充
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        self.record_stage(name, method=method, elapsed_ms=elapsed, **extra)
```

> 💡 **初学者提示**：`@contextmanager` 是 Python 标准库提供的装饰器，用于快速实现 `with` 语句协议。`yield` 之前是"进入"逻辑，之后是"退出"逻辑。

#### `finish()`

标记执行结束，冻结 `total_elapsed_ms`：

```python
trace.finish()
# 从此 trace.elapsed_ms() 返回固定值，不会继续增长
```

#### `elapsed_ms(stage_name=None)`

查询某个阶段（或总计）的耗时：

```python
trace.elapsed_ms("fusion")   # 返回 fusion 阶段的 ms
trace.elapsed_ms()            # 返回从创建到 finish 的总 ms
```

#### `to_dict()`

序列化为纯 JSON 可存储的字典：

```python
{
    "trace_id": "a1b2c3d4e5f6...",
    "trace_type": "query",
    "started_at": "2026-06-10T12:00:00+00:00",
    "finished_at": "2026-06-10T12:00:01.500000+00:00",
    "total_elapsed_ms": 1500.123,
    "stages": [
        {"name": "dense_retrieval", "method": "openai", "elapsed_ms": 200.5, "results": 10},
        ...
    ]
}
```

### 3.3 完整示例

```python
from src.core.trace.trace_context import TraceContext

# 创建一个 query 类型的 trace
trace = TraceContext(trace_type="query")

# 方式 1：手动记录
import time
start = time.perf_counter()
# ...做一些事...
time.sleep(0.1)
elapsed = (time.perf_counter() - start) * 1000
trace.record_stage("my_step", method="brute_force", elapsed_ms=elapsed, items=42)

# 方式 2：用上下文管理器自动计时
with trace.stage("fusion", method="rrf") as extra:
    time.sleep(0.05)
    extra["candidates"] = 20

# 结束
trace.finish()

# 查看结果
print(trace.to_dict())
# 输出：
# {
#   "trace_id": "...",
#   "trace_type": "query",
#   "started_at": "...",
#   "finished_at": "...",
#   "total_elapsed_ms": ...,
#   "stages": [
#     {"name": "my_step", "method": "brute_force", "elapsed_ms": 100.xxx, "items": 42},
#     {"name": "fusion", "method": "rrf", "elapsed_ms": 50.xxx, "candidates": 20}
#   ]
# }
```

---

## 4. F2：结构化日志 — 持久化到文件

> 📁 文件：`src/observability/logger.py`

### 4.1 为什么需要两套 logger？

| logger | 目标 | 格式 | 输出位置 |
|--------|------|------|----------|
| `get_logger(name)` | 开发者阅读的运行日志 | 人类可读文本 | stderr |
| `get_trace_logger(file)` | 机器可解析的 trace 数据 | JSON Lines | 文件 |

**关键约束**：MCP Server 用 stdio 通信，stdout 必须只有 MCP 协议消息。所以：
- 普通日志 → stderr（不影响 stdout）
- trace 数据 → 文件（完全隔离）

### 4.2 JSONFormatter

```python
class JSONFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
            payload = record.msg           # 直接序列化字典
        else:
            payload = {"message": record.getMessage()}  # 包装字符串
        return json.dumps(payload, ensure_ascii=False)
```

**效果**：每条日志变成一行 JSON，方便后续用工具解析。

### 4.3 write_trace — 追加一条记录

```python
def write_trace(trace_dict: dict, log_file: str = "logs/traces.jsonl") -> None:
    """追加一条 trace 记录到 JSON Lines 文件。"""
    get_trace_logger(log_file).info(trace_dict)
```

写入后 `logs/traces.jsonl` 内容像这样（每行一个 JSON）：

```jsonl
{"trace_id":"abc","trace_type":"query","started_at":"...","finished_at":"...","total_elapsed_ms":150,"stages":[...]}
{"trace_id":"def","trace_type":"ingestion","started_at":"...","finished_at":"...","total_elapsed_ms":3000,"stages":[...]}
```

### 4.4 TraceCollector — 桥梁

> 📁 文件：`src/core/trace/trace_collector.py`

```python
class TraceCollector:
    def __init__(self, settings):
        self._enabled = settings.observability.trace_enabled
        self._log_file = settings.observability.log_file

    def collect(self, trace: TraceContext) -> None:
        """持久化 trace；失败只告警，绝不 raise。"""
        if not self._enabled:
            return
        try:
            write_trace(trace.to_dict(), self._log_file)
        except Exception as e:
            logger.warning(f"Failed to persist trace {trace.trace_id}: {e}")
```

**关键设计**：
- `trace_enabled=False` 时直接跳过（生产环境可开关）
- `try/except` 兜底，保证**观测不影响业务**

---

## 5. F3：Query 链路打点

> 📁 文件：`src/core/query_engine/hybrid_search.py`、`reranker.py`、`query_knowledge_hub.py`

### 5.1 整体流程

```
用户查询
    │
    ▼
QueryKnowledgeHubTool.run()
    │
    ├─ ① 创建 trace = TraceContext(trace_type="query")
    │
    ├─ ② hybrid.search(query, trace=trace)
    │       ├─ query_processing  （关键词提取 + 过滤器合并）
    │       ├─ dense_retrieval   （向量相似度检索）
    │       ├─ sparse_retrieval  （BM25 关键词检索）
    │       └─ fusion            （RRF 融合排序）
    │
    ├─ ③ reranker.rerank(query, results, trace=trace)
    │       └─ rerank            （交叉编码器重排序）
    │
    ├─ ④ trace.finish()          （标记结束）
    │
    └─ ⑤ collector.collect(trace) （写入文件）
```

### 5.2 HybridSearch 中的打点代码

以 dense_retrieval 阶段为例：

```python
def search(self, query, top_k=None, filters=None, trace=None):
    # ...
    
    # Dense retrieval
    dense_results = []
    start = time.perf_counter()  # ← 计时开始
    try:
        dense_results = self._dense.retrieve(query, top_k=k)
    except Exception as e:
        logger.warning(f"Dense retrieval failed: {e}")
    # ← 计时结束 & 记录阶段
    self._trace_stage(
        trace, "dense_retrieval", start,
        method="embedding",
        results=len(dense_results),
    )
```

辅助方法（避免每次写 `if trace is None: return`）：

```python
@staticmethod
def _trace_stage(trace, name, start, method="", **details):
    """记录一个阶段到 trace（如果 trace 存在的话）。"""
    if trace is None:
        return
    elapsed = (time.perf_counter() - start) * 1000.0
    trace.record_stage(name, method=method, elapsed_ms=elapsed, **details)
```

### 5.3 为什么 trace 参数是可选的？

```python
def search(self, query, top_k=None, filters=None, trace=None):
    #                                              ^^^^^^^^^^^
    # 默认 None —— 旧的调用方完全不需要改动！
```

这是一个**向后兼容**的设计：
- 不传 trace → 所有 `_trace_stage` 里的 `if trace is None: return` 直接跳过，零开销
- 传了 trace → 自动记录各阶段

### 5.4 Reranker 的打点

```python
def rerank(self, query, results, trace=None):
    if not results:
        if trace is not None:
            trace.record_stage("rerank", method="skip", elapsed_ms=0.0, results=0)
        return []

    start = time.perf_counter()
    try:
        ranked = self._reranker.rerank(query, candidates)
        self._trace_rerank(trace, start, provider, len(ranked), fallback=False)
        return ranked
    except RerankerError:
        # 降级：返回原序
        self._trace_rerank(trace, start, provider, len(results), fallback=True)
        return results
```

**注意**：即使降级（fallback），也要记录——这是排查问题的关键！

### 5.5 在 Tool 层管理生命周期

```python
class QueryKnowledgeHubTool:
    def run(self, query, top_k=None, collection=None):
        trace = TraceContext(trace_type="query")   # ① 创建

        results = hybrid.search(query, trace=trace) # ② 传入
        results = reranker.rerank(query, results, trace=trace) # ③ 传入

        trace.finish()                              # ④ 结束
        self._collector.collect(trace)              # ⑤ 持久化

        return result
```

**总结**：
- **谁创建 trace**？→ 最外层的"编排者"（Tool / 脚本）
- **谁传递 trace**？→ 沿调用链逐层传下去
- **谁记录阶段**？→ 每个被调用的组件自己记录自己
- **谁收尾**？→ 创建者（finish + collect）

---

## 6. F4：Ingestion 链路打点

> 📁 文件：`src/ingestion/pipeline.py`

### 6.1 与 Query 链路的对比

| | Query 链路 | Ingestion 链路 |
|---|---|---|
| trace_type | `"query"` | `"ingestion"` |
| 阶段 | query_processing → dense → sparse → fusion → rerank | load → split → transform → embed → upsert |
| 创建者 | QueryKnowledgeHubTool | IngestionPipeline.run 内部 |
| 收尾位置 | Tool.run 末尾 | Pipeline.run 的 `finally` |

### 6.2 Ingestion 打点代码

```python
def run(self, file_path, collection="default", force=False, on_progress=None, trace=None):
    # 如果外部没传 trace，内部自己创建
    own_trace = trace is None
    if trace is None:
        trace = TraceContext(trace_type="ingestion")

    try:
        # Stage: Load
        start = time.perf_counter()
        document = self._loader.load(str(path))
        self._trace_stage(trace, "load", start, method="markitdown")

        # Stage: Split
        start = time.perf_counter()
        chunks = self._chunker.split_document(document)
        self._trace_stage(trace, "split", start, method="recursive", chunks=len(chunks))

        # Stage: Transform (refine + enrich + caption)
        start = time.perf_counter()
        chunks = self._refiner.transform(chunks)
        chunks = self._enricher.transform(chunks)
        chunks = self._captioner.transform(chunks)
        self._trace_stage(trace, "transform", start, method="refine+enrich+caption")

        # Stage: Embed
        start = time.perf_counter()
        records = self._batch_processor.process(chunks)
        self._trace_stage(trace, "embed", start, method="openai", records=len(records))

        # Stage: Upsert
        start = time.perf_counter()
        self._upserter.upsert(records)
        self._bm25.build(records)
        self._trace_stage(trace, "upsert", start, method="chroma+bm25")

        return {"status": "success", ...}

    except Exception as e:
        trace.record_stage("error", method=type(e).__name__)
        raise

    finally:
        if own_trace:        # 仅当 trace 是自己创建的才收尾
            trace.finish()
            self._collector.collect(trace)
```

### 6.3 `finally` 的妙用

```python
try:
    # 正常流程
except Exception:
    # 错误也记录
    raise
finally:
    # 不管成功还是失败都执行收尾
    trace.finish()
    collector.collect(trace)
```

这保证了：
- ✅ 成功时正常收尾
- ✅ 失败时也能看到 trace（含 error 阶段）

---

## 7. F5：Pipeline 进度回调

### 7.1 什么是 on_progress？

```python
def run(self, file_path, ..., on_progress=None):
    """
    on_progress: 回调函数，签名 (stage_name, current, total) -> None
    """
```

**用途**：让调用方（如 CLI、Web UI）实时显示进度条。

### 7.2 调用示例

```python
def my_progress(stage_name: str, current: int, total: int):
    print(f"[{stage_name}] {current}/{total}")

pipeline.run("doc.pdf", on_progress=my_progress)

# 输出：
# [integrity_check] 0/1
# [load] 0/1
# [split] 0/1
# [transform] 0/3
# [transform] 1/3
# [transform] 2/3
# [transform] 3/3
# [encode] 0/1
# [store] 0/2
# [store] 1/2
# [store] 2/2
```

### 7.3 on_progress vs trace 的区别

| | on_progress | trace |
|---|---|---|
| 目的 | 实时通知进度 | 事后分析性能 |
| 数据粒度 | 当前步 / 总步 | 耗时 + 方法 + 计数 |
| 阶段命名 | encode / store（历史遗留） | embed / upsert（规范命名） |
| 消费者 | UI / CLI | 日志分析 / Dashboard |

> 💡 两者**独立**、**互不干扰**，可以同时使用。

---

## 8. 动手实验：跑测试 & 看输出

### 8.1 运行所有 F 阶段测试

```bash
cd /path/to/MODULAR-RAG-MCP-SERVER
source .venv/bin/activate

# 运行 F1-F5 全部测试
python -m pytest tests/unit/test_trace_context.py \
                 tests/unit/test_jsonl_logger.py \
                 tests/unit/test_query_trace.py \
                 tests/unit/test_ingestion_trace.py \
                 tests/unit/test_pipeline_progress.py \
                 -v
```

### 8.2 交互式实验：手动创建 Trace

在 Python REPL 中：

```python
from src.core.trace.trace_context import TraceContext
import time, json

# 1. 创建
trace = TraceContext(trace_type="query")
print(f"Trace ID: {trace.trace_id}")

# 2. 用上下文管理器记录
with trace.stage("thinking", method="brain") as extra:
    time.sleep(0.1)
    extra["neurons_fired"] = 42

# 3. 手动记录
trace.record_stage("answering", method="keyboard", elapsed_ms=50.0, chars=100)

# 4. 结束
trace.finish()

# 5. 查看结果
print(json.dumps(trace.to_dict(), indent=2))
```

### 8.3 交互式实验：写入 trace 文件

```python
from src.observability.logger import write_trace
from src.core.trace.trace_context import TraceContext

trace = TraceContext(trace_type="ingestion")
trace.record_stage("load", method="markitdown", elapsed_ms=120.0)
trace.finish()

# 写入文件
write_trace(trace.to_dict(), "logs/my_test_traces.jsonl")

# 查看文件内容
import json
with open("logs/my_test_traces.jsonl") as f:
    for line in f:
        print(json.loads(line))
```

---

## 9. 设计决策 Q&A

### Q1：为什么不用 OpenTelemetry？

**A**：本项目是一个 MCP Server 工具，核心追求**轻量、零外部依赖**。OTel 适合微服务集群，但对单进程工具太重了。我们的需求用 100 行代码就能满足。

### Q2：为什么 `trace` 参数要可选（`None`）而不是强制传入？

**A**：向后兼容。本项目 A-E 阶段已有 220 个通过的测试，全部调用 `search()` 和 `rerank()` 时不传 trace。如果改为必传，就要改 220 个测试文件。

```python
# 旧代码无需改动：
results = hybrid.search("python")  # 默认 trace=None，无打点开销

# 新代码获得观测能力：
results = hybrid.search("python", trace=my_trace)
```

### Q3：为什么把 `_trace_stage` 写成静态方法？

**A**：它不依赖 `self`，只需要 `trace` 和计时数据。静态方法更清楚地表达"这是一个无副作用的辅助函数"。

### Q4：trace 记录了哪些东西，有安全风险吗？

**A**：只记录**元信息**（方法名、计数、耗时），不记录：
- ❌ 原始查询文本
- ❌ 检索到的文档内容
- ❌ API 密钥/Token
- ❌ 用户身份

### Q5：on_progress 的阶段名为什么和 trace 的阶段名不同？

| on_progress | trace |
|-------------|-------|
| `encode` | `embed` |
| `store` | `upsert` |

**A**：`on_progress` 是 C 阶段（更早）实现的，当时用了 encode/store。F 阶段按规范用 embed/upsert。为避免破坏已有测试，选择保留两套命名，各自独立。

---

## 10. 小结与练习题

### 小结

```
┌─────────────────────────────────────────────────────────┐
│                     F 阶段知识图谱                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  TraceContext (容器)                                     │
│     ├─ record_stage()  手动记录                          │
│     ├─ stage()         自动计时上下文管理器                │
│     ├─ finish()        冻结总耗时                        │
│     ├─ elapsed_ms()    查询耗时                          │
│     └─ to_dict()       序列化                           │
│                                                         │
│  TraceCollector (持久化桥梁)                              │
│     └─ collect()  → write_trace() → traces.jsonl        │
│                                                         │
│  打点模式：                                              │
│     创建者(Tool/Pipeline) → trace → 各组件自行记录       │
│     → finish → collect → 文件                           │
│                                                         │
│  安全约束：                                              │
│     - 只写文件，不写 stdout                              │
│     - 失败静默，不阻塞主流程                              │
│     - 参数可选，默认 None                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 练习题

#### 🌟 入门级

1. **创建一个 trace 并记录 3 个阶段**：`load`（100ms）→ `process`（200ms）→ `save`（50ms），然后 `finish()` 并打印 `to_dict()`。

2. **用 `stage()` 上下文管理器改写**：把练习 1 中手动传 `elapsed_ms` 的方式改为用 `with trace.stage(...)` 自动计时。

#### 🌟🌟 进阶级

3. **为一个新方法添加打点**：假设你写了一个 `summarize(text)` 函数，请为它添加可选的 `trace` 参数并记录 `"summarize"` 阶段。

4. **编写测试**：为练习 3 写一个 pytest 用例，验证传入 trace 时阶段被正确记录，不传 trace 时函数仍正常工作。

#### 🌟🌟🌟 挑战级

5. **添加新的 trace_type**：假设你需要支持 `trace_type="evaluation"`（评估链路），请修改 `_VALID_TYPES` 并编写一个新的 trace 使用场景。

6. **trace 聚合分析**：写一个脚本读取 `logs/traces.jsonl`，计算每种 `trace_type` 的平均 `total_elapsed_ms` 和各阶段的 P95 耗时。

---

## 附录 A：测试代码速读

### test_trace_context.py 要点

```python
def test_stage_context_manager_times_block():
    """验证 with trace.stage(...) 自动计时且可追加字段。"""
    trace = TraceContext()
    with trace.stage("dense_retrieval", method="openai") as extra:
        extra["hits"] = 5
        time.sleep(0.01)
    stage = trace.stages[0]
    assert stage["elapsed_ms"] > 0    # 有耗时
    assert stage["hits"] == 5         # 附加字段
```

### test_query_trace.py 要点

```python
def test_search_records_query_stages():
    """验证一次 hybrid search 产出完整 4 阶段 trace。"""
    hs = _make_hybrid()  # 全 mock 的 HybridSearch
    trace = TraceContext(trace_type="query")

    hs.search("python programming", trace=trace)

    stage_names = [s["name"] for s in trace.stages]
    assert stage_names == [
        "query_processing",
        "dense_retrieval",
        "sparse_retrieval",
        "fusion",
    ]
```

---

## 附录 B：配置项

在 `config/settings.yaml` 中：

```yaml
observability:
  trace_enabled: true           # 是否开启 trace 持久化
  log_file: "logs/traces.jsonl" # trace 输出文件
  log_level: "INFO"
```

设为 `trace_enabled: false` 可完全关闭 trace 文件写入（TraceContext 仍可在内存中使用，只是不落盘）。

---

> 🎉 恭喜你读完了 F 阶段的全部内容！你现在已经掌握了一个轻量级可追踪系统的核心设计。这套模式（可选注入 → 各层自行记录 → 统一收尾持久化）在生产系统中非常常见，值得反复体会。
