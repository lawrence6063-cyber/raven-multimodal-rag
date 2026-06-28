# Modular RAG MCP Server

> English abstract: A pluggable, observable modular RAG (Retrieval-Augmented Generation) framework that exposes tools over the MCP (Model Context Protocol) for AI assistants such as Copilot / Claude to consume directly. Highlights: end-to-end pluggable architecture (LLM / Embedding / Reranker / Splitter / VectorStore / Evaluator), hybrid retrieval (BM25 sparse + Dense embedding + RRF fusion + optional Cross-Encoder / LLM rerank), multimodal image captioning, Agentic RAG (LLM-driven routing / decomposition / multi-hop / self-correction), Streamlit dashboard, and a Ragas + custom evaluation harness with golden test sets. MIT licensed.

---

## 目录

- [项目概述](#-项目概述)
- [核心能力一览](#-核心能力一览)
- [技术亮点](#-技术亮点)
- [快速开始](#-快速开始)
- [开发者指南](#️-开发者指南技术运行手册)
- [常见问题](#-常见问题)

---

## 🏗️ 项目概述

本项目将 RAG 的核心环节——**检索（Hybrid Search + Rerank）**、**多模态视觉处理（Image Captioning）**、**RAG 评估（Ragas + Custom）**、**生成（LLM Response）**——以及当下热门的应用协议 **MCP（Model Context Protocol）** 串联为一个完整的、可运行的工程项目。

**项目的一大亮点是极易适配到你自己的业务中**。得益于全链路可插拔架构，你可以快速将它结合到自己已有的项目里，无论你的背景和需求如何，都能找到适合自己的使用方式。

### 不只是项目，更是一整套思路

**比这个项目本身更有价值的，是它背后蕴含的一整套工程化思路**：

- 如何编写 **DEV_SPEC**（开发规格文档）来驱动开发
- 如何用 **Skill** 基于 Spec 自动完成代码编写
- 如何用 **Skill** 进行自动化测试、打包、环境配置
- 如何基于可插拔架构进行扩展（比如扩展到 Agent）

**学会了思路，你可以自己做全新的项目和扩展**。

### 核心能力一览

| 模块 | 能力 | 说明 |
|------|------|------|
| **Ingestion Pipeline** | PDF → Markdown → Chunk → Transform → Embedding → Upsert | 全链路数据摄取，支持多模态图片描述（Image Captioning） |
| **Hybrid Search** | Dense (向量) + Sparse (BM25) + RRF Fusion + Rerank | 粗排召回 + 精排重排的两段式检索架构 |
| **Agentic RAG** | LLM 主动路由 / 分解 / 多跳 / 自纠正 | 服务端 agent 自主决策检索并合成带引用答案 |
| **MCP Server** | 标准 MCP 协议暴露 Tools | `query_knowledge_hub`、`agentic_query`、`list_collections`、`get_document_summary` |
| **Dashboard** | Streamlit 六页面管理平台 | 系统总览 / 数据浏览 / Ingestion 管理 / 摄取追踪 / 查询追踪 / 评估面板 |
| **Evaluation** | Ragas + Custom 评估体系 | 支持 golden test set 回归测试，拒绝"凭感觉"调优 |
| **Observability** | 全链路白盒化追踪 | Ingestion 与 Query 两条链路的每一个中间状态透明可见 |
| **Skill 驱动全流程** | 从编写到测试、打包、配置一键完成 | auto-coder / qa-tester / package / setup 等 Skill 覆盖完整开发生命周期 |

### 技术亮点

**🔌 全链路可插拔架构**：LLM / Embedding / Reranker / Splitter / VectorStore / Evaluator 每一个核心环节均定义了抽象接口，支持"乐高积木式"替换，通过配置文件一键切换后端，零代码修改。

**🔍 混合检索 + 重排**：BM25 稀疏检索解决专有名词精确匹配 + Dense Embedding 解决同义词语义匹配，RRF 融合后可选 Cross-Encoder / LLM Rerank 精排，平衡查全率与查准率。

**🖼️ 多模态图像处理**：采用 Image-to-Text 策略，利用 Vision LLM 自动生成图片描述并缝合进 Chunk，复用纯文本 RAG 链路即可实现"搜文字出图"。

**📡 MCP 生态集成**：遵循 Model Context Protocol 标准，可直接对接 GitHub Copilot、Claude Desktop 等 MCP Client，零前端开发，一次开发处处可用。

**📊 可视化管理 + 自动化评估**：Streamlit Dashboard 提供完整的数据管理与链路追踪能力，集成 Ragas 等评估框架，建立基于数据的迭代反馈回路。

**🧪 三层测试体系**：Unit / Integration / E2E 分层测试，覆盖独立模块逻辑、模块间交互、完整链路（MCP Client / Dashboard）。

**🤖 Agentic RAG**：将「一次性检索 → 拼引用」升级为「LLM 作为 agent 主动决策检索」并在服务端合成带引用答案；全程降级（任一 LLM 步骤异常→单次混合检索，绝不报错）。

> 📖 详细架构设计、模块说明和任务排期请参阅 [DEV_SPEC.md](DEV_SPEC.md)

---

## 📂 分支说明

本项目提供三个分支，面向不同使用场景，请根据自身需求选择：

### `main` — 最干净的完整代码

- 始终只有 **1 个 commit**，包含项目的最新完整代码
- **适合人群**：想要快速体验项目完整功能、或直接在该项目基础上做二次扩展的同学
- **使用方式**：克隆后直接运行 Setup Skill 即可体验

### `dev` — 保留完整开发记录

- 代码与 `main` 完全一致，但保留了完整的 commit 历史
- 记录了从零开始逐步构建的每一步过程，包含大量中间节点
- **适合人群**：想了解项目是如何一步步从零搭建起来的同学，可以通过 commit 历史回溯开发思路

### `clean-start` — 干净起点，从零开始

- 仅包含工程骨架（Agent Skills + DEV_SPEC），所有任务进度清零
- 保留了完整的 Skill 配置，可以使用 Agent 辅助开发
- **适合人群**：
  - 时间充分、想要从头开发的同学
  - 想要体验完整工作流的同学：写 Spec → 拆任务 → 写代码 → 写测试 → 迭代优化
  - 甚至可以基于自己的理解重新设计架构，用自己的思路实现，深度理解每一个模块
- **核心理念**：整个项目的代码编写是 **让 AI 基于 DEV_SPEC 来自动完成的**。AI 通过 Skill 读取 Spec 中的任务定义、架构设计和接口规范，自动生成符合规格的代码。

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd Modular-RAG-MCP-Server
```

### 2. 一键配置（Setup Skill）

本项目提供了 **Setup Skill** 一键完成所有环境配置，包括：Provider 选择 → API Key 配置 → 依赖安装 → 配置文件生成 → Dashboard 启动。

在 VS Code 中打开项目，通过 Copilot / Claude 对话框输入：

```
setup
```

Agent 会自动引导你完成全部配置流程。

> 💡 如果不熟悉 Skill 的使用方式，可参考 docs/ 中对应的 Skill 使用文档。

---

## 🎯 使用方式

### 作为独立 RAG 系统使用

这个项目本身就是一个完整的 RAG 系统，可以作为 RAG 全流程的学习与实战材料。项目中涉及的典型环节——检索、生成、向量数据库、分块策略、重排序等——覆盖了 RAG 的核心流程。

### 融入现有项目

把 RAG 能力集成到你已有的项目中也是一种很好的策略。例如把本项目的 RAG 能力融入到已有的 Agent 项目中，不作为独立项目，而是作为 Agent 项目的一部分来描述。

> **强烈建议**：至少试一下——把你自己领域的文档（金融、法律、医疗，或者你的业务文档）丢进去，看一下检索效果。如果效果不好，再去调整和改进。这个过程本身就是最好的学习，也是最实战的经验。

---

## 🛠️ 开发者指南（技术运行手册）

> 上文「快速开始」推荐用 **Setup Skill** 一键配置。本章面向想**手动**运行、调试或集成的开发者，提供从安装到 ingest / query / dashboard / 测试的完整命令，目标是让你在 10 分钟内跑通全链路。

### 1. 环境与安装

要求 **Python ≥ 3.10**（开发环境使用 3.13）。

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 安装运行依赖 + 开发依赖（pytest 等）
pip install -e ".[dev]"
```

主要依赖：`chromadb`（向量库）、`mcp`（MCP SDK）、`openai`（LLM/Embedding 客户端）、`langchain-text-splitters`、`markitdown`（PDF→Markdown）、`streamlit`（Dashboard）。
可选：`ragas`（启用 Ragas 评估后端时再安装，未安装会优雅降级）。

### 2. 配置 API Key

配置集中在 `config/settings.yaml`。**API Key 推荐通过环境变量注入，切勿写入文件提交到仓库**：

```bash
export OPENAI_API_KEY="sk-..."        # 或在 settings.yaml 的对应 api_key 字段填写
```

切换 Provider（OpenAI / Azure / Ollama / DeepSeek / Qwen）只需修改 `settings.yaml` 中 `llm.provider` 与 `embedding.provider`，无需改代码。

### 3. 首次摄取 → 查询

```bash
# 摄取一个 PDF 文件或整个目录（目录会递归扫描 *.pdf）
python scripts/ingest.py --path tests/fixtures/sample_documents/ --collection demo

# 执行一次检索查询
python scripts/query.py --query "介绍一下混合检索" --verbose

# 运行评估（离线无 Key 时检索降级、指标输出 0.0，仍可跑通）
python scripts/evaluate.py --backends custom
```

### 4. `settings.yaml` 字段说明

| 配置段 | 关键字段 | 含义 |
|--------|----------|------|
| `llm` | `provider` / `model` / `api_key` / `base_url` | 生成用大模型后端与凭据；`provider` 支持 qwen/openai/azure/ollama/deepseek |
| `embedding` | `provider` / `model` / `dimensions` | 向量化后端与维度（需与向量库一致）；`provider` 支持 qwen/qwen_multimodal/openai/azure/ollama |
| `vector_store` | `provider` / `collection_name` / `persist_directory` | 向量库（默认 chroma，本地持久化目录） |
| `splitter` | `provider` / `chunk_size` / `chunk_overlap` | 分块策略与窗口；recursive/semantic/fixed |
| `retrieval` | `top_k` / `dense_weight` / `sparse_weight` / `rrf_k` | 粗排召回数量与 RRF 融合参数 |
| `rerank` | `enabled` / `provider` / `top_n` | 精排开关与后端（none/cross_encoder/llm） |
| `vision_llm` | `enabled` / `provider` / `model` | 多模态图片描述（Image Captioning）；`provider` 支持 qwen_vision(qwen-vl-max)/azure_vision |
| `ingestion` | `batch_size` / `bm25_index_path` / `image_embedding` / `*_enricher` | 摄取批大小、BM25 路径、图片向量入库开关、可选 LLM 增强 |
| `evaluation` | `backends` / `golden_test_set` | 评估后端组合（custom/ragas）与黄金测试集路径 |
| `observability` | `trace_enabled` / `log_file` / `log_level` | 链路追踪开关、JSON Lines 日志文件、日志级别 |
| `agent` | `enabled` / `max_hops` / `max_subqueries` / `max_reflect_rounds` / `max_context_chunks` | Agentic RAG 开关与硬上限 |

> 注意：trace 持久化只写入 `log_file`（默认 `logs/traces.jsonl`），**绝不写 stdout**，以保证 MCP stdio 协议干净。

### 5. 多模态：真·跨模态检索（路径B）

基于 Qwen / DashScope 把**文本与图片编码进同一向量空间**，实现「以文搜图 / 以图搜文 / 以图搜图」，并让查询工具能直接**接收图片**。

**开启方式**（`config/settings.yaml`，复用同一个 DashScope Key）：

```yaml
embedding:
  provider: "qwen_multimodal"       # 图文统一向量（DashScope multimodal-embedding-v1）
  model: "multimodal-embedding-v1"
  dimensions: 1024
vision_llm:
  enabled: true
  provider: "qwen_vision"           # qwen-vl-max 生成图片中文描述缝入文本
  model: "qwen-vl-max"
ingestion:
  image_captioner: { enabled: true }
  image_embedding: true             # 文档图片编码为独立多模态向量入库
```

> ⚠️ 切换多模态向量后，旧的纯文本向量库不兼容，**必须清库重摄**：`rm -rf data/db/chroma data/db/bm25` 后重新 `ingest`。

**摄取链路**：`PdfLoader` 用 `pypdfium2` 真抽 PDF 内嵌图片 → `ImageStorage` 落盘+索引 → `qwen-vl-max` 生成描述缝入文本 → 图片自身经多模态向量编码为独立记录（`modality=image`）入同一 collection。

**查询接收图片**：`query_knowledge_hub` 工具新增 `image` 入参（Base64 或 `data/` 白名单目录下的本地路径），支持纯文本、纯图片、图文混合三种查询；命中后原图随答案返回。CLI 亦可验证：

```bash
python scripts/query.py --image data/query_images/figure.png            # 以图检索
python scripts/query.py --query "这张图说明了什么" --image data/q.png    # 图文混合
```

> 安全：`image` 拒绝远程 URL（防 SSRF）、本地路径限定白名单目录（防穿越）、Base64 限制大小；单图失败降级跳过，绝不阻断文本答案。

### 6. MCP 配置示例

本项目通过 **stdio** 暴露 MCP Server，入口为根目录 `main.py`。在 MCP Client 中按如下方式配置（将路径替换为你的本地绝对路径）。

**GitHub Copilot（VS Code）** — `.vscode/mcp.json` 或用户级 `mcp.json`：

```json
{
  "servers": {
    "modular-rag": {
      "type": "stdio",
      "command": "/ABS/PATH/.venv/bin/python",
      "args": ["/ABS/PATH/Modular-RAG-MCP-Server/main.py"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

**Claude Desktop** — `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "/ABS/PATH/.venv/bin/python",
      "args": ["/ABS/PATH/Modular-RAG-MCP-Server/main.py"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

配置后 Client 即可调用四个工具：`query_knowledge_hub`（混合检索 + 引用，返回检索片段交由 Client LLM 合成）、`agentic_query`（Agentic RAG：服务端 agent 自主路由/分解/多跳/自纠正，直接返回带 `[n]` 引用的合成答案，适合多跳与推理型问题）、`list_collections`（列出集合）、`get_document_summary`（文档摘要）。

> `agentic_query` 由 `config/settings.yaml` 的 `agent` 段控制（`enabled`、各子能力开关、`max_hops`/`max_subqueries`/`max_reflect_rounds`/`max_context_chunks` 等硬上限）。`agent.enabled=False` 时该工具自动委托 `query_knowledge_hub`，行为与升级前完全一致；任一 LLM 决策步骤异常会降级为单次混合检索，绝不报错。详见 `docs/P1_AGENTIC_RAG_SPEC.md`。

### 7. Dashboard 使用指南

```bash
python scripts/start_dashboard.py        # 等价于 streamlit run src/observability/dashboard/app.py
```

启动后浏览器访问终端提示的地址（默认 `http://localhost:8501`），左侧导航包含六个页面：

| 页面 | 功能 |
|------|------|
| 📊 系统总览 | 各组件配置卡片、可观测性状态、向量库数据统计、完整 YAML 配置 |
| 📂 数据浏览器 | 按集合浏览已摄入文档、Chunk 与关联图片 |
| 📥 Ingestion 管理 | 上传文件触发摄取、查看进度、删除文档 |
| 🔬 Ingestion 追踪 | 摄取历史与各阶段耗时瀑布图 |
| 🔍 Query 追踪 | 查询历史、阶段耗时、Dense vs Sparse 对比、Rerank 前后变化 |
| ⚙️ 评估面板 | 选择评估后端与黄金测试集，运行评估查看 hit_rate / mrr 等指标 |

> 各页面在**无数据时均优雅降级**（显示提示而非报错），可在未摄取任何文档时安全打开。

### 8. 运行测试

```bash
source .venv/bin/activate

# 全量测试
python -m pytest tests/ -q

# 按层级运行
python -m pytest tests/unit -q          # 单元测试（快、无外部依赖）
python -m pytest tests/integration -q   # 集成测试（模块间交互，子进程 stdio）
python -m pytest tests/e2e -q           # 端到端（MCP Client 子进程 / Dashboard 冒烟 / 召回回归）

# 按 marker 运行
python -m pytest -m e2e -q
```

E2E 测试全部 **hermetic**（注入 fake 检索后端、无网络、无 API Key 依赖），可直接在 CI 中作为回归门禁。

---

## ❓ 常见问题

### 1. 如何切换 Provider（比如换成 Qwen / DeepSeek / Ollama）？

项目从架构设计上使用了**工厂模式（Factory Pattern）**，Provider 的扩展和切换非常方便。不同 API 本质上都是类似的 HTTP 请求，甚至大多数都遵循 OpenAI 的请求格式，切换起来特别容易。

**具体操作方式有两种：**

1. **使用 Setup Skill（推荐）**：运行一键 Setup Skill，AI 会主动询问你想用哪个 Provider，引导你填入 API Key，然后自动帮你完成代码适配和配置生成。
2. **直接让 AI 帮你改**：把你想切换的 Provider 告诉 AI（如 "帮我切换到 Qwen" 或 "帮我配置 DeepSeek"），AI 能根据工厂模式的架构自动完成代码编写。

> **原理说明**：项目的 `src/libs/` 下的 LLM、Embedding、Reranker 等模块都使用工厂模式，新增一个 Provider 只需要：① 新增一个 Provider 类；② 在工厂注册；③ 更新 `settings.yaml` 配置。

### 2. 项目评估（Custom Evaluator）与 Cross-Encoder Reranker 部分

这两个模块的**框架代码已经搭好，但尚未经过完整测试**，感兴趣的同学可以自行完善：

| 模块 | 状态 | 需要做什么 |
|------|------|-----------|
| **自定义评估（Custom Evaluator）** | 框架已有，未测试 | 定义评估方法，准备对应的测试数据集 |
| **Cross-Encoder Reranker** | 框架已有，未测试 | 需要下载本地重排模型（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`） |

**这些 AI 都能帮你写出来**。把需求描述清楚，AI 可以帮你实现评估方法、准备数据、下载模型并完成集成测试。

### 3. 项目报错 / Bug 怎么办？

这是一个学习与实战导向的项目，遇到报错是正常的。

- **如何修复**：最简单的方式是**把错误信息直接丢给 AI**，绝大多数问题 AI 都能帮你修复。
- **参考资源**：推荐参考 docs/ 中相关的调试与排错说明。

### 4. 想摄取 PDF 以外的文档格式（Word / Markdown / HTML 等）怎么办？

**直接问 AI 帮你扩展即可。**

项目的 Loader 层采用了可插拔的抽象设计（`BaseLoader`），目前默认实现了 PDF Loader。如果你需要支持 Word、Markdown、HTML 等其他格式，整体架构已经设计好了扩展点，让 AI 帮你新增一个对应的 Loader 实现就可以了。

比如告诉 AI："帮我新增一个 Word 文档的 Loader，参考现有的 PDF Loader 实现"，AI 完全可以搞定。

### 5. 如何集成到 AI 工具中（Copilot / Cursor / Claude Code 等）？

本项目是一个 **MCP Server**，可以集成到任何支持 MCP 协议的 AI 工具和 Agent 中。演示中已经集成到了 **GitHub Copilot** 和 **Cursor** 中，你同样可以集成到 **Claude Code** 或其他支持 MCP 框架的工具。

**如何集成？非常简单——问 AI。**

本质上就是给不同的工具写一个 MCP 的配置文件：
- **Copilot（VS Code）**：让 AI 帮你生成 MCP 配置文件即可
- **Cursor**：直接导入项目，Cursor 会自动识别
- **Claude Code / 其他框架**：问 AI 怎么配置，每个工具的配置方式略有不同，但原理都一样

当然，也推荐你去理解 MCP 协议的原理——了解 Server 和 Client 之间是如何通信的、Tool 是怎么注册和调用的。

### 6. 通用建议：善用 AI

上述大多数问题（Provider 切换、模块扩展、Bug 修复、架构理解）**AI 都能解决**：

- 🔧 **代码层面**：让 AI 帮你切换 Provider、实现评估方法、修复 Bug
- 📖 **知识层面**：项目架构问题、设计模式问题，都可以问 AI 获取解释
- 🚀 **扩展层面**：想加新功能或适配新场景，描述清楚需求让 AI 帮你实现

> 多问 AI，让它指导你。这也是这个项目想要传达的核心理念之一——**学会与 AI 协作开发**。

---

## 📌 后续安排

### ✅ 会做的
- 项目相关问题的汇总与 FAQ 整理
- 技术要点讲解（RAG 核心知识、架构设计等）
- 欢迎投稿共建：欢迎通过 Issue / PR 贡献改进

### ❌ 不会做的
- 不会继续扩展新功能
- 不会处理 Bug Fix、设计优化等
  - 遇到 Bug 和设计上的改进点，请在自己的项目中修复和优化
  - 后续的扩展和修复一定是要靠自己的，而且**有了 AI，这些都很容易做到**
  - 这本身就是一个很好的学习与工程加分项
  - 在理解项目的基础上独立扩展，才是真正的能力体现

---

## Contributing

欢迎通过 Issue 提交问题或建议，通过 Pull Request 贡献代码。提交前请确保：

- 相关测试通过（`python -m pytest tests/ -q`）
- 新增模块遵循全链路可插拔架构（定义抽象接口 + 工厂注册）
- 配置变更同步更新 `settings.yaml` 字段说明与文档

---

## License

MIT License. See [LICENSE](LICENSE).
