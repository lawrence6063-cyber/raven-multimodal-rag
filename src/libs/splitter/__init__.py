"""
Splitter 模块 — 文本分块策略抽象与实现。

================================================================================
📖 Splitter 选型指南
================================================================================

在 RAG (Retrieval-Augmented Generation) 系统中，文本分块（Splitting/Chunking）是
影响检索质量的关键环节。以下是主流的分块方案对比：

┌─────────────────────┬──────────────────────────────────────────────────────────┐
│ 方案                │ 说明                                                     │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ 1. Fixed-Size       │ 按固定字符数/token数切分，简单粗暴。                      │
│    (固定长度)        │ 优点：实现简单，速度快。                                  │
│                     │ 缺点：会在句子/段落中间截断，破坏语义完整性。              │
│                     │                                                          │
│ 2. Sentence-Based   │ 按句子边界切分（使用NLP句子分割器如spaCy/NLTK）。         │
│    (基于句子)        │ 优点：保证句子完整性。                                    │
│                     │ 缺点：单句可能太短，需要合并逻辑；依赖NLP模型。           │
│                     │                                                          │
│ 3. Recursive        │ 按优先级递归尝试多种分隔符（段落→行→句→词→字符）。       │
│    (递归分割) ✅     │ 优点：自适应文档结构，尽量在自然边界切分；                │
│                     │       支持自定义分隔符优先级（如Markdown标题）；           │
│                     │       chunk_size/overlap 可调；实现成熟稳定。             │
│                     │ 缺点：对非结构化纯文本效果与固定长度接近。                │
│                     │                                                          │
│ 4. Semantic         │ 基于嵌入向量的语义相似度切分。                             │
│    (语义分割) ✅     │ 优点：语义边界最精准；不需要LLM，只需Embedding模型；       │
│                     │       Qwen Embedding 成本极低（¥0.0005/千token）。         │
│                     │ 缺点：需要调用Embedding API，速度慢于Recursive（~10-50x）；│
│                     │       对短文档效果不明显。                                │
│                     │                                                          │
│ 5. Document-Based   │ 利用文档自身结构切分（标题、表格、代码块、图片引用）。    │
│    (基于文档结构) ✅ │ 优点：完美保留文档层级结构；多模态RAG核心方案；           │
│                     │       表格/代码块保持完整；为chunk附加章节元数据。         │
│                     │ 缺点：仅适用于结构化格式；超长块仍需Recursive二次切分。   │
│                     │                                                          │
│ 6. Token-Based      │ 按LLM token数切分（如tiktoken）。                         │
│    (基于Token)       │ 优点：精确控制送入LLM的token数。                         │
│                     │ 缺点：与Fixed-Size类似，可能破坏语义。                    │
│                     │                                                          │
│ 7. Code/AST         │ 基于代码语法结构切分（函数、类、方法定义边界）。           │
│    (代码结构) ✅     │ 优点：保证每个chunk是完整的代码逻辑单元；                 │
│                     │       支持20+编程语言；自动检测语言。                     │
│                     │ 缺点：仅适用于代码文件。                                 │
└─────────────────────┴──────────────────────────────────────────────────────────┘

================================================================================
🎯 为什么选择 Recursive（递归分割）作为默认方案？
================================================================================

1. **通用性强**：适用于 Markdown、纯文本、代码等多种格式，无需针对每种格式
   单独实现 splitter。

2. **语义保留好**：通过优先在高层级分隔符（如标题、段落）处切分，最大程度
   保留上下文的语义完整性。

3. **可配置性高**：chunk_size、chunk_overlap、separators 均可通过配置调整，
   适应不同场景需求。

4. **零额外依赖成本**：不需要调用 Embedding 模型（对比 Semantic Splitter），
   处理速度快，适合大批量文档处理。

5. **生态成熟**：LangChain RecursiveCharacterTextSplitter 经过大量生产验证，
   是社区公认的 RAG 默认最佳实践。

6. **可扩展**：本模块采用工厂+注册器模式，未来可轻松添加 Semantic、
   Document-Based 等高级方案，按需切换。

================================================================================
🏗️ 多模态 RAG 分层切分架构
================================================================================

本模块采用分层切分策略，4 种 Splitter 各司其职：

┌─────────────────────────────────────────────────────────────────────────────┐
│ 优先级 │ Splitter              │ 适用模态                │ Provider 名称    │
├────────┼───────────────────────┼─────────────────────────┼──────────────────┤
│ P0     │ Recursive             │ 纯文本、Markdown、通用  │ "recursive"      │
│ P0     │ Semantic              │ 长文本、访谈、转录文本  │ "semantic"       │
│ P1     │ Document Structure    │ 图文混排、表格、多模态  │ "document_structure" │
│ P1     │ Code                  │ 代码文件、代码仓库      │ "code"           │
└────────┴───────────────────────┴─────────────────────────┴──────────────────┘

推荐组合：
- 通用文档 → Recursive（快速）或 Semantic（精准）
- 图文混排文档 → Document Structure（保留结构）+ Recursive（二次切分）
- 代码仓库 → Code（语法感知）

================================================================================
📁 模块结构
================================================================================

- base_splitter.py               : BaseSplitter 抽象基类 + SplitterError 异常
- splitter_factory.py            : SplitterFactory 工厂类 + register_splitter 装饰器
- recursive_splitter.py          : RecursiveSplitter 实现（P0 默认方案，速度快）
- semantic_splitter.py           : SemanticSplitter 实现（P0 语义精准，需Embedding）
- document_structure_splitter.py : DocumentStructureSplitter（P1 多模态核心）
- code_splitter.py               : CodeSplitter 实现（P1 代码语法感知）

使用示例：
    # 方式1：通过工厂创建（默认 recursive）
    from src.libs.splitter.splitter_factory import SplitterFactory
    splitter = SplitterFactory.create(settings)  # provider="recursive"
    chunks = splitter.split_text(document_text)

    # 方式2：使用 Document Structure Splitter（多模态文档）
    # settings.provider = "document_structure"
    splitter = SplitterFactory.create(settings)
    chunks = splitter.split_text(markdown_with_tables_and_images)

    # 方式3：使用 Code Splitter（代码文件）
    from src.libs.splitter.code_splitter import CodeSplitter
    splitter = CodeSplitter.for_file("main.py", splitter_settings)
    chunks = splitter.split_text(code_text)

    # 方式4：使用 Semantic Splitter（需传入 Embedding 实例）
    from src.libs.splitter.semantic_splitter import SemanticSplitter
    from src.libs.embedding.qwen_embedding import QwenEmbedding
    embedding = QwenEmbedding(embedding_settings)
    splitter = SemanticSplitter(splitter_settings, embedding=embedding)
    chunks = splitter.split_text(long_unstructured_text)
"""
