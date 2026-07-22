# 初始资料库 — 论文清单

> 按栏目分目录存放，可直接用于 ingest：
> `python scripts/ingest.py --path data/documents/llm --collection llm`
> （`rag` / `agent` / `mcp` / `multimodal` 同理；`list_collections` 会自动按子目录识别为 collection）

来源：arXiv（公开预印本）。共 **50 篇**，5 个栏目各 10 篇。

## 📂 LLM（大语言模型基础）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_transformer_attention_is_all_you_need.pdf | Attention Is All You Need（Transformer） | [1706.03762](https://arxiv.org/abs/1706.03762) |
| 02_bert.pdf | BERT: Pre-training of Deep Bidirectional Transformers | [1810.04805](https://arxiv.org/abs/1810.04805) |
| 03_gpt3_few_shot_learners.pdf | Language Models are Few-Shot Learners（GPT-3） | [2005.14165](https://arxiv.org/abs/2005.14165) |
| 04_llama.pdf | LLaMA: Open and Efficient Foundation Language Models | [2302.13971](https://arxiv.org/abs/2302.13971) |
| 05_instructgpt.pdf | Training language models to follow instructions（InstructGPT/RLHF） | [2203.02155](https://arxiv.org/abs/2203.02155) |
| 06_chain_of_thought.pdf | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [2201.11903](https://arxiv.org/abs/2201.11903) |
| 07_chinchilla.pdf | Training Compute-Optimal Large Language Models（Chinchilla） | [2203.15556](https://arxiv.org/abs/2203.15556) |
| 08_lora.pdf | LoRA: Low-Rank Adaptation of Large Language Models | [2106.09685](https://arxiv.org/abs/2106.09685) |
| 09_constitutional_ai.pdf | Constitutional AI: Harmlessness from AI Feedback | [2212.08073](https://arxiv.org/abs/2212.08073) |
| 10_scaling_laws.pdf | Scaling Laws for Neural Language Models | [2001.08361](https://arxiv.org/abs/2001.08361) |

## 📂 RAG（检索增强生成）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_rag_knowledge_intensive_nlp.pdf | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [2005.11401](https://arxiv.org/abs/2005.11401) |
| 02_dense_passage_retrieval.pdf | Dense Passage Retrieval for Open-Domain QA（DPR） | [2004.04906](https://arxiv.org/abs/2004.04906) |
| 03_realm.pdf | REALM: Retrieval-Augmented Language Model Pre-Training | [2002.08909](https://arxiv.org/abs/2002.08909) |
| 04_fusion_in_decoder.pdf | Leveraging Passage Retrieval with Generative Models（FiD） | [2007.01282](https://arxiv.org/abs/2007.01282) |
| 05_self_rag.pdf | Self-RAG: Learning to Retrieve, Generate, and Critique | [2310.11511](https://arxiv.org/abs/2310.11511) |
| 06_retro.pdf | Improving Language Models by Retrieving from Trillions of Tokens（RETRO） | [2112.04426](https://arxiv.org/abs/2112.04426) |
| 07_replug.pdf | REPLUG: Retrieval-Augmented Black-Box Language Models | [2301.12652](https://arxiv.org/abs/2301.12652) |
| 08_crag.pdf | Corrective Retrieval Augmented Generation（CRAG） | [2401.15884](https://arxiv.org/abs/2401.15884) |
| 09_graphrag.pdf | From Local to Global: A Graph RAG Approach to Query-Focused Summarization | [2404.16130](https://arxiv.org/abs/2404.16130) |
| 10_rag_survey.pdf | Retrieval-Augmented Generation for Large Language Models: A Survey | [2312.10997](https://arxiv.org/abs/2312.10997) |

## 📂 Agent（智能体）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_react.pdf | ReAct: Synergizing Reasoning and Acting in Language Models | [2210.03629](https://arxiv.org/abs/2210.03629) |
| 02_toolformer.pdf | Toolformer: Language Models Can Teach Themselves to Use Tools | [2302.04761](https://arxiv.org/abs/2302.04761) |
| 03_reflexion.pdf | Reflexion: Language Agents with Verbal Reinforcement Learning | [2303.11366](https://arxiv.org/abs/2303.11366) |
| 04_generative_agents.pdf | Generative Agents: Interactive Simulacra of Human Behavior | [2304.03442](https://arxiv.org/abs/2304.03442) |
| 05_hugginggpt.pdf | HuggingGPT: Solving AI Tasks with ChatGPT and its Friends | [2303.17580](https://arxiv.org/abs/2303.17580) |
| 06_tree_of_thoughts.pdf | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | [2305.10601](https://arxiv.org/abs/2305.10601) |
| 07_voyager.pdf | Voyager: An Open-Ended Embodied Agent with Large Language Models | [2305.16291](https://arxiv.org/abs/2305.16291) |
| 08_swe_bench.pdf | SWE-bench: Can Language Models Resolve Real-World GitHub Issues? | [2310.06770](https://arxiv.org/abs/2310.06770) |
| 09_metagpt.pdf | MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework | [2308.00352](https://arxiv.org/abs/2308.00352) |
| 10_autogen.pdf | AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation | [2308.08155](https://arxiv.org/abs/2308.08155) |

## 📂 MCP（工具使用与函数调用）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_tool_learning_survey.pdf | Tool Learning with Foundation Models | [2304.08354](https://arxiv.org/abs/2304.08354) |
| 02_gorilla.pdf | Gorilla: Large Language Model Connected with Massive APIs | [2305.15334](https://arxiv.org/abs/2305.15334) |
| 03_toolllm.pdf | ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs | [2307.16789](https://arxiv.org/abs/2307.16789) |
| 04_api_bank.pdf | API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs | [2304.08244](https://arxiv.org/abs/2304.08244) |
| 05_toolkengpt.pdf | ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings | [2305.11554](https://arxiv.org/abs/2305.11554) |
| 06_restgpt.pdf | RestGPT: Connecting Large Language Models with Real-World RESTful APIs | [2306.06624](https://arxiv.org/abs/2306.06624) |
| 07_webgpt.pdf | WebGPT: Browser-assisted question-answering with human feedback | [2112.09332](https://arxiv.org/abs/2112.09332) |
| 08_toolace.pdf | ToolACE: Winning the Points of LLM Function Calling | [2409.00920](https://arxiv.org/abs/2409.00920) |
| 09_tool_emu.pdf | Identifying the Risks of LM Agents with an LM-Emulated Sandbox（ToolEmu） | [2309.15817](https://arxiv.org/abs/2309.15817) |
| 10_t_eval.pdf | T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step | [2312.14033](https://arxiv.org/abs/2312.14033) |

## 📂 Multimodal（多模态视觉语言模型）

| 文件 | 论文 | arXiv |
|------|------|-------|
| 01_clip.pdf | Learning Transferable Visual Models From Natural Language Supervision（CLIP） | [2103.00020](https://arxiv.org/abs/2103.00020) |
| 02_vit.pdf | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT） | [2010.11929](https://arxiv.org/abs/2010.11929) |
| 03_blip.pdf | BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation | [2201.12086](https://arxiv.org/abs/2201.12086) |
| 04_flamingo.pdf | Flamingo: a Visual Language Model for Few-Shot Learning | [2204.14198](https://arxiv.org/abs/2204.14198) |
| 05_blip2.pdf | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | [2301.12597](https://arxiv.org/abs/2301.12597) |
| 06_llava.pdf | Visual Instruction Tuning（LLaVA） | [2304.08485](https://arxiv.org/abs/2304.08485) |
| 07_minigpt4.pdf | MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models | [2304.10592](https://arxiv.org/abs/2304.10592) |
| 08_qwen_vl.pdf | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond | [2308.12966](https://arxiv.org/abs/2308.12966) |
| 09_cogvlm.pdf | CogVLM: Visual Expert for Pretrained Language Models | [2311.03079](https://arxiv.org/abs/2311.03079) |
| 10_llava15.pdf | Improved Baselines with Visual Instruction Tuning（LLaVA-1.5） | [2310.03744](https://arxiv.org/abs/2310.03744) |
