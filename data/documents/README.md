# 初始资料库 — 论文清单

> 按栏目分目录存放，可直接用于 ingest：
> `python scripts/ingest.py --path data/documents/llm --collection llm`
> （`rag` / `agent` 同理；`list_collections` 会自动按子目录识别为 collection）

来源：arXiv（公开预印本）。共 **15 篇**，每栏目 5 篇。

## 📂 LLM（大语言模型基础）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_transformer_attention_is_all_you_need.pdf | Attention Is All You Need（Transformer） | [1706.03762](https://arxiv.org/abs/1706.03762) |
| 02_bert.pdf | BERT: Pre-training of Deep Bidirectional Transformers | [1810.04805](https://arxiv.org/abs/1810.04805) |
| 03_gpt3_few_shot_learners.pdf | Language Models are Few-Shot Learners（GPT-3） | [2005.14165](https://arxiv.org/abs/2005.14165) |
| 04_llama.pdf | LLaMA: Open and Efficient Foundation Language Models | [2302.13971](https://arxiv.org/abs/2302.13971) |
| 05_instructgpt.pdf | Training language models to follow instructions（InstructGPT/RLHF） | [2203.02155](https://arxiv.org/abs/2203.02155) |

## 📂 RAG（检索增强生成）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_rag_knowledge_intensive_nlp.pdf | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [2005.11401](https://arxiv.org/abs/2005.11401) |
| 02_dense_passage_retrieval.pdf | Dense Passage Retrieval for Open-Domain QA（DPR） | [2004.04906](https://arxiv.org/abs/2004.04906) |
| 03_realm.pdf | REALM: Retrieval-Augmented Language Model Pre-Training | [2002.08909](https://arxiv.org/abs/2002.08909) |
| 04_fusion_in_decoder.pdf | Leveraging Passage Retrieval with Generative Models（FiD） | [2007.01282](https://arxiv.org/abs/2007.01282) |
| 05_self_rag.pdf | Self-RAG: Learning to Retrieve, Generate, and Critique | [2310.11511](https://arxiv.org/abs/2310.11511) |

## 📂 Agent（智能体）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_react.pdf | ReAct: Synergizing Reasoning and Acting in Language Models | [2210.03629](https://arxiv.org/abs/2210.03629) |
| 02_toolformer.pdf | Toolformer: Language Models Can Teach Themselves to Use Tools | [2302.04761](https://arxiv.org/abs/2302.04761) |
| 03_reflexion.pdf | Reflexion: Language Agents with Verbal Reinforcement Learning | [2303.11366](https://arxiv.org/abs/2303.11366) |
| 04_generative_agents.pdf | Generative Agents: Interactive Simulacra of Human Behavior | [2304.03442](https://arxiv.org/abs/2304.03442) |
| 05_hugginggpt.pdf | HuggingGPT: Solving AI Tasks with ChatGPT and its Friends | [2303.17580](https://arxiv.org/abs/2303.17580) |
