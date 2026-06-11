"""
Transform 模块 — Chunk 增强/清洗层
====================================

## 一、为什么需要 Transform？

Splitter 只负责"切"，切完的 chunk 是原始的、脏的、缺乏上下文的。
如果直接 Embedding 存入向量库，会导致：

| 问题         | 后果                           |
|------------|------------------------------|
| 噪声（页眉/HTML） | 向量被污染，检索精度下降               |
| 上下文丢失      | 语义模糊（"收入增长3%" → 哪家公司？）     |
| 元数据缺失      | 无法过滤检索、无法展示摘要              |
| 图片信息丢失     | 占位符对 Embedding 无语义          |

Transform 在 Split → Embed 之间插入增强/清洗层，让 chunk 从"能用"变成"好用"。

## 二、主流增强方法对比

| 方法                | 原理                      | 需LLM | 成本 | 效果  |
|-------------------|-------------------------|:---:|:--:|:---:|
| 噪声清洗 Cleaning     | 正则去页眉/HTML/多余空白         | ❌  | 极低 | ⭐⭐  |
| 元数据增强 Metadata     | 提取/生成 title/summary/tags | 可选  | 低~中 | ⭐⭐⭐ |
| 上下文注入 Contextual   | chunk 前添加文档级上下文描述        | ✅  | 中  | ⭐⭐⭐⭐⭐ |
| 假设性问题 HyDE         | 为 chunk 生成可能被问到的问题       | ✅  | 中  | ⭐⭐⭐⭐ |
| 图片描述 Captioning    | Vision LLM 为图片生成文字描述     | ✅  | 高  | ⭐⭐⭐⭐ |
| 实体链接 Entity Link   | 识别并标注实体（人名/公司/术语）       | 可选  | 中  | ⭐⭐⭐ |
| 层级摘要 Hierarchical  | 多粒度摘要（句/段/章）            | ✅  | 高  | ⭐⭐⭐ |
| 关系抽取 Relation Ext. | 抽取实体间关系，构建知识图谱           | ✅  | 高  | ⭐⭐⭐ |

## 三、选型结论 — 为什么选了这 3 个？

当前实现：ChunkRefiner + MetadataEnricher + ImageCaptioner

| 选择              | 核心理由                                          |
|-----------------|-----------------------------------------------|
| ChunkRefiner    | 噪声清洗零成本必做；LLM精炼可选开关，降级安全                     |
| MetadataEnricher| 检索需要元数据做过滤/排序；规则零成本兜底，LLM高质量增强              |
| ImageCaptioner  | 多模态RAG刚需，图片占位符对Embedding无意义，必须转文字             |

未选入的理由：
- 上下文注入(P1待实现)：效果最好(Anthropic实测降低67%检索失败)，但需额外LLM调用
- 假设性问题(P2)：成本较高，需评估ROI
- 实体链接(P2)：通用场景收益不明显，适合特定领域
- 层级摘要/关系抽取(P3)：实现复杂，当前架构暂不需要

## 四、架构设计要点

Pipeline 执行顺序: ChunkRefiner → MetadataEnricher → ImageCaptioner

设计模式：
- 统一接口: BaseTransform.transform(chunks) → chunks（可插拔、可排序）
- 双模式: 规则 + LLM（配置开关），开发用规则(快/免费)，生产用LLM(质量高)
- 降级保护: 每个 chunk 独立 try/except，单个失败不阻塞整批
- 元数据追踪: refined_by/enriched_by 字段，方便调试和评估

## 五、模块结构

    transform/
    ├── __init__.py          # 本文件（选型文档）
    ├── base_transform.py    # 抽象基类 BaseTransform
    ├── chunk_refiner.py     # 噪声清洗 + 可选LLM精炼
    ├── metadata_enricher.py # 元数据增强（title/summary/tags）
    └── image_captioner.py   # 图片描述生成（Vision LLM）
"""
