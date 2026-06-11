# Sparse 稀疏检索算法 — 完整教学文档

> 本文档是对项目中 `SparseEncoder` + `BM25Indexer` 的深度教学，目标是让你**彻底理解**稀疏检索的每一步算法逻辑。

---

## 一、术语与缩写全称对照表

| 缩写/术语 | 全称 (English) | 中文含义 | 说明 |
|-----------|---------------|---------|------|
| **BM25** | Best Matching 25 | 最佳匹配第25版 | 经典信息检索评分算法，由 Robertson 等人提出 |
| **TF** | Term Frequency | 词频 | 一个词在某篇文档中出现的次数 |
| **IDF** | Inverse Document Frequency | 逆文档频率 | 衡量一个词在全局语料中的稀有程度 |
| **DF** | Document Frequency | 文档频率 | 包含某个词的文档数量 |
| **DL** | Document Length | 文档长度 | 文档中的总词数（或归一化后的总权重） |
| **AVGDL** | Average Document Length | 平均文档长度 | 所有文档长度的平均值 |
| **k1** | — (BM25 参数) | TF 饱和参数 | 控制词频增长的边际收益衰减速度 |
| **b** | — (BM25 参数) | 长度归一化参数 | 控制文档长度对评分的惩罚程度 |
| **NTF** | Normalized Term Frequency | 归一化词频 | TF / DL，消除文档长度影响 |
| **RAG** | Retrieval-Augmented Generation | 检索增强生成 | 结合检索和生成的 AI 架构 |
| **RRF** | Reciprocal Rank Fusion | 倒数排名融合 | 合并多路检索结果的算法 |
| **BOW** | Bag of Words | 词袋模型 | 忽略词序，只关注词频的文本表示方法 |
| **IR** | Information Retrieval | 信息检索 | 从大量文档中找到相关信息的学科 |
| **Posting** | Posting (in inverted index) | 倒排记录 | 倒排索引中记录某词出现在哪些文档中的条目 |
| **Posting List** | Posting List | 倒排记录表 | 某个词对应的所有文档记录的列表 |

---

## 二、为什么需要 Sparse 检索？

### 2.1 Dense vs Sparse 的本质区别

```
Dense (稠密向量):
  "Python 编程" → [0.12, -0.34, 0.56, ..., 0.08]  (1536个浮点数)
  特点：每个维度都有值，维度含义不可解释

Sparse (稀疏向量):
  "Python 编程" → {"python": 0.5, "编程": 0.5}  (只有2个非零值)
  特点：绝大多数维度为0，非零维度 = 具体的词
```

### 2.2 各自擅长的场景

| 查询类型 | Dense 表现 | Sparse 表现 | 原因 |
|---------|:----------:|:-----------:|------|
| "汽车保养" 搜到 "轿车维护" | ✅ 优秀 | ❌ 失败 | Dense 理解同义词 |
| "iPhone 15 Pro Max" | ❌ 可能模糊 | ✅ 精确命中 | Sparse 精确匹配专有名词 |
| "Python 3.12 新特性" | 一般 | ✅ 精确 | 版本号需要精确匹配 |
| "如何提高代码质量" | ✅ 优秀 | 一般 | 语义理解 vs 关键词匹配 |

### 2.3 结论：两者互补

> **Hybrid Search = Dense + Sparse**，通过 RRF（倒数排名融合）合并两路结果，取长补短。

---

## 三、算法全景图

整个 Sparse 检索分为 **离线阶段** 和 **在线阶段**：

```
┌─────────────────────────────────────────────────────────────────┐
│                     离线阶段 (Ingestion)                          │
│                                                                   │
│  Chunk.text ──→ 分词 ──→ 去停用词 ──→ 计算 TF ──→ 归一化 NTF    │
│                                                                   │
│  NTF 结果 ──→ 构建倒排索引 ──→ 计算全局 IDF ──→ 持久化到磁盘     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     在线阶段 (Query)                              │
│                                                                   │
│  用户查询 ──→ 分词 ──→ 对每个词查倒排索引 ──→ BM25 公式评分      │
│                                                                   │
│  评分结果 ──→ 按分数降序排列 ──→ 返回 Top-K 结果                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、离线阶段：SparseEncoder 逐步拆解

### 4.1 Step 1：分词 (Tokenization)

**代码**：
```python
def _tokenize(self, text: str) -> list[str]:
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [t for t in tokens if t not in self.STOP_WORDS and len(t) > 1]
```

**三个子步骤**：

#### (1) 正则提取单词

```
正则: \b[a-zA-Z0-9]+\b
含义: \b = 单词边界, [a-zA-Z0-9]+ = 一个或多个字母/数字

输入: "Machine Learning with Python! v3.12"
      ↓ text.lower()
      "machine learning with python! v3.12"
      ↓ re.findall(...)
      ["machine", "learning", "with", "python", "v3", "12"]
```

#### (2) 去停用词 (Stop Words Removal)

**什么是停用词？** 出现频率极高但几乎不携带信息的词。

```
停用词表 (部分):
  冠词: a, an, the
  系动词: is, are, was, were, be
  介词: of, in, for, on, with, at, by, from
  连词: and, but, or, if
  代词: it, this, that, these, those

为什么要去掉？
  "the" 在几乎所有英文文档中都出现 → DF ≈ N → IDF ≈ 0
  保留它只会增加计算量，不会提供任何区分度
```

#### (3) 过滤单字符

```
过滤条件: len(t) > 1
原因: 单字符 (如 "a", "I", "x") 通常无意义或已被停用词覆盖
```

**完整示例**：

```
输入: "Python is GREAT for Machine Learning! v3.12"
      ↓ lower()
      "python is great for machine learning! v3.12"
      ↓ re.findall(r'\b[a-zA-Z0-9]+\b', ...)
      ["python", "is", "great", "for", "machine", "learning", "v3", "12"]
      ↓ 去停用词 (is, for 被移除)
      ["python", "great", "machine", "learning", "v3", "12"]
      ↓ 去单字符 (无单字符)
      ["python", "great", "machine", "learning", "v3", "12"]
```

---

### 4.2 Step 2：计算词频 TF (Term Frequency)

**代码**：
```python
tf = Counter(terms)
```

**Counter 的作用**：统计每个词出现的次数。

```
输入: ["python", "great", "machine", "learning", "python", "python"]
      ↓ Counter()
输出: {"python": 3, "great": 1, "machine": 1, "learning": 1}
```

**直觉**：一个词在文档中出现越多次，说明这个文档越可能与该词相关。

---

### 4.3 Step 3：归一化 TF → NTF (Normalized Term Frequency)

**代码**：
```python
doc_len = len(terms) if terms else 1
sparse_vector = {term: count / doc_len for term, count in tf.items()}
```

**为什么要归一化？**

```
问题场景：
  短文档 (10个词): "python" 出现 3 次 → 原始 TF = 3
  长文档 (1000个词): "python" 出现 3 次 → 原始 TF = 3

  如果不归一化，两者 TF 相同，但直觉上短文档中 "python" 更重要！

归一化后：
  短文档: NTF = 3/10 = 0.30  ← python 占了 30% 的篇幅
  长文档: NTF = 3/1000 = 0.003  ← python 只占 0.3% 的篇幅

结论：归一化消除了文档长度的不公平影响。
```

**完整数值示例**：

```
输入文本: "Machine Learning with Python. Python is great for ML tasks."

Step 1 分词: ["machine", "learning", "python", "python", "great", "ml", "tasks"]
Step 2 词频: {"machine": 1, "learning": 1, "python": 2, "great": 1, "ml": 1, "tasks": 1}
Step 3 归一化 (doc_len = 7):

sparse_vector = {
    "machine":  1/7 ≈ 0.143,
    "learning": 1/7 ≈ 0.143,
    "python":   2/7 ≈ 0.286,  ← 权重最高！因为出现了2次
    "great":    1/7 ≈ 0.143,
    "ml":       1/7 ≈ 0.143,
    "tasks":    1/7 ≈ 0.143,
}
```

---

## 五、离线阶段：BM25Indexer 构建索引

SparseEncoder 的输出 `sparse_vector` 被 BM25Indexer 消费，构建**倒排索引**。

### 5.1 什么是倒排索引 (Inverted Index)？

**正排索引** vs **倒排索引**：

```
正排索引 (Forward Index):
  文档1 → [python, machine, learning]
  文档2 → [deep, learning, neural]
  文档3 → [data, science, python]

  问题：搜 "python" 需要遍历所有文档 → O(N)

倒排索引 (Inverted Index):
  python   → [文档1, 文档3]
  machine  → [文档1]
  learning → [文档1, 文档2]
  deep     → [文档2]
  neural   → [文档2]
  data     → [文档3]
  science  → [文档3]

  优势：搜 "python" 直接定位到 [文档1, 文档3] → O(1) 查找
```

> 💡 **类比**：正排索引像"按页码翻书"，倒排索引像"查目录/索引页"。

### 5.2 构建过程 (build 方法)

**代码逻辑**：

```python
def build(self, records: list[ChunkRecord]) -> None:
    inverted: dict[str, list[dict]] = defaultdict(list)
    doc_lengths = []

    for rec in records:
        # 文档长度 = sparse_vector 所有权重之和
        doc_len = sum(rec.sparse_vector.values())
        doc_lengths.append(doc_len)
        # 对每个词项，记录它出现在哪个文档、TF 是多少
        for term, tf in rec.sparse_vector.items():
            inverted[term].append({"chunk_id": rec.id, "tf": tf, "doc_length": doc_len})

    self._doc_count = len(records)
    self._avg_doc_length = sum(doc_lengths) / len(doc_lengths)
```

**数值示例**：

假设有 3 个 ChunkRecord：

```
c1.sparse_vector = {"machine": 0.5, "learning": 0.5}
c2.sparse_vector = {"deep": 0.33, "learning": 0.33, "neural": 0.33}
c3.sparse_vector = {"data": 0.33, "science": 0.33, "python": 0.33}
```

构建后：

```
inverted_index = {
    "machine":  [{"chunk_id": "c1", "tf": 0.5,  "doc_length": 1.0}],
    "learning": [{"chunk_id": "c1", "tf": 0.5,  "doc_length": 1.0},
                 {"chunk_id": "c2", "tf": 0.33, "doc_length": 0.99}],
    "deep":     [{"chunk_id": "c2", "tf": 0.33, "doc_length": 0.99}],
    "neural":   [{"chunk_id": "c2", "tf": 0.33, "doc_length": 0.99}],
    "data":     [{"chunk_id": "c3", "tf": 0.33, "doc_length": 0.99}],
    "science":  [{"chunk_id": "c3", "tf": 0.33, "doc_length": 0.99}],
    "python":   [{"chunk_id": "c3", "tf": 0.33, "doc_length": 0.99}],
}

doc_count = 3
avg_doc_length = (1.0 + 0.99 + 0.99) / 3 ≈ 0.993
```

### 5.3 计算 IDF (Inverse Document Frequency)

**公式**：

```
IDF(term) = log( (N - df + 0.5) / (df + 0.5) )

其中：
  N  = 总文档数
  df = 包含该词的文档数 (Document Frequency)
```

**直觉理解**：

```
一个词出现在越少的文档中 → df 越小 → IDF 越大 → 这个词越"稀有"越有区分度

极端情况：
  - 词出现在所有文档中: df = N → IDF = log(0.5/N+0.5) ≈ 负数 (几乎无用)
  - 词只出现在1个文档中: df = 1 → IDF = log((N-0.5)/1.5) (很有区分度)
```

**数值示例** (N = 3)：

```
"machine":  df = 1 → IDF = log((3-1+0.5)/(1+0.5)) = log(2.5/1.5) = log(1.667) ≈ 0.511
"learning": df = 2 → IDF = log((3-2+0.5)/(2+0.5)) = log(1.5/2.5) = log(0.6)   ≈ -0.511
"python":   df = 1 → IDF = log((3-1+0.5)/(1+0.5)) = log(2.5/1.5) = log(1.667) ≈ 0.511
"deep":     df = 1 → IDF = log(2.5/1.5) ≈ 0.511
```

**观察**：
- `"learning"` 出现在 2/3 的文档中 → IDF 为负 → 区分度低
- `"machine"`, `"python"` 只出现在 1/3 的文档中 → IDF 为正 → 区分度高

---

## 六、在线阶段：BM25 评分公式

### 6.1 完整公式

```
BM25(query, document) = Σ  IDF(t) × [ tf(t,d) × (k1 + 1) ]
                       t∈q          ─────────────────────────────────────
                                    tf(t,d) + k1 × (1 - b + b × dl/avgdl)
```

**拆解每一部分**：

| 部分 | 含义 | 直觉 |
|------|------|------|
| `Σ t∈q` | 对查询中的每个词求和 | 多个查询词的贡献累加 |
| `IDF(t)` | 该词的全局稀有度 | 稀有词贡献大，常见词贡献小 |
| `tf(t,d) × (k1+1)` | 分子：词频的放大 | 词频越高，分子越大 |
| `tf(t,d) + k1×(...)` | 分母：词频的饱和控制 | 防止词频无限增长的影响 |
| `1 - b + b×dl/avgdl` | 长度归一化因子 | 长文档被惩罚，短文档被奖励 |

### 6.2 参数 k1 的作用 — TF 饱和度

```
k1 = 1.5 (默认值)

当 k1 很大时 (如 k1=100):
  TF 从 1→2→3→4 时，分数几乎线性增长
  → 词频越高越好，没有上限

当 k1 很小时 (如 k1=0.1):
  TF 从 1→2 时分数增长很小
  → 只要出现过就行，出现多次也没太大额外收益

k1=1.5 是经验最优值：
  TF=1 → score ≈ 0.6
  TF=2 → score ≈ 0.8
  TF=5 → score ≈ 0.92
  TF=10 → score ≈ 0.96  (接近饱和)
```

**图形化理解**：

```
分数
 1.0 ┤                          ════════════ ← 饱和上限
     │                    ═══
     │               ═══
 0.5 ┤          ═══
     │     ═══
     │═══
 0.0 ┼───┬───┬───┬───┬───┬───→ TF
     0   1   2   3   4   5

k1 越大，曲线越"直"（线性）
k1 越小，曲线越"弯"（快速饱和）
```

### 6.3 参数 b 的作用 — 长度归一化

```
b = 0.75 (默认值)

长度归一化因子 = 1 - b + b × (dl / avgdl)

当 b = 0:
  因子 = 1（恒定）→ 完全不考虑文档长度
  
当 b = 1:
  因子 = dl / avgdl → 完全按长度比例惩罚

当 b = 0.75:
  短文档 (dl = 0.5 × avgdl): 因子 = 1 - 0.75 + 0.75×0.5 = 0.625 → 分母小 → 分数高
  平均文档 (dl = avgdl):      因子 = 1 - 0.75 + 0.75×1.0 = 1.0   → 正常
  长文档 (dl = 2 × avgdl):    因子 = 1 - 0.75 + 0.75×2.0 = 1.75  → 分母大 → 分数低
```

**直觉**：长文档天然包含更多词，不应该因此获得不公平的高分。b 参数控制这种"惩罚"的力度。

### 6.4 完整数值计算示例

**场景**：3 个文档，查询 "machine learning"

```
文档库：
  c1: sparse_vector = {"machine": 0.5, "learning": 0.5}     (doc_len = 1.0)
  c2: sparse_vector = {"deep": 0.33, "learning": 0.33, "neural": 0.33}  (doc_len = 0.99)
  c3: sparse_vector = {"data": 0.33, "science": 0.33, "python": 0.33}   (doc_len = 0.99)

全局统计：
  N = 3, avgdl = 0.993
  IDF("machine") = 0.511
  IDF("learning") = -0.511

参数：k1 = 1.5, b = 0.75
```

**计算 c1 的分数**：

```
对 "machine" (tf=0.5, dl=1.0):
  分子 = 0.5 × (1.5 + 1) = 0.5 × 2.5 = 1.25
  长度因子 = 1 - 0.75 + 0.75 × (1.0/0.993) = 0.25 + 0.755 = 1.005
  分母 = 0.5 + 1.5 × 1.005 = 0.5 + 1.508 = 2.008
  BM25_machine = IDF × (分子/分母) = 0.511 × (1.25/2.008) = 0.511 × 0.623 = 0.318

对 "learning" (tf=0.5, dl=1.0):
  分子 = 0.5 × 2.5 = 1.25
  分母 = 0.5 + 1.5 × 1.005 = 2.008
  BM25_learning = -0.511 × (1.25/2.008) = -0.511 × 0.623 = -0.318

c1 总分 = 0.318 + (-0.318) = 0.0  ← "learning" 的负 IDF 抵消了 "machine" 的正贡献
```

**计算 c2 的分数**：

```
对 "machine": c2 不包含 "machine" → 跳过
对 "learning" (tf=0.33, dl=0.99):
  分子 = 0.33 × 2.5 = 0.825
  长度因子 = 1 - 0.75 + 0.75 × (0.99/0.993) = 0.25 + 0.748 = 0.998
  分母 = 0.33 + 1.5 × 0.998 = 0.33 + 1.497 = 1.827
  BM25_learning = -0.511 × (0.825/1.827) = -0.511 × 0.451 = -0.231

c2 总分 = -0.231
```

**计算 c3 的分数**：

```
c3 不包含 "machine" 也不包含 "learning" → 总分 = 0
```

**最终排序**：c1 (0.0) > c3 (0.0) > c2 (-0.231)

> ⚠️ **注意**：这个例子中因为文档数太少（N=3），"learning" 出现在 2/3 的文档中导致 IDF 为负。在真实场景中（N=10000+），常见词的 IDF 接近 0 而非负数，稀有词的 IDF 会很大，效果会更明显。

---

## 七、代码与算法的对应关系

### 7.1 SparseEncoder（离线编码）

```python
# 文件: src/ingestion/embedding/sparse_encoder.py

class SparseEncoder:
    def encode(self, chunks):
        for chunk in chunks:
            terms = self._tokenize(chunk.text)  # ← Step 1: 分词+去停用词
            tf = Counter(terms)                  # ← Step 2: 计算 TF
            doc_len = len(terms)
            sparse_vector = {term: count/doc_len  # ← Step 3: 归一化 NTF
                           for term, count in tf.items()}
            # 输出: {"python": 0.286, "machine": 0.143, ...}
```

### 7.2 BM25Indexer.build（构建索引）

```python
# 文件: src/ingestion/storage/bm25_indexer.py

def build(self, records):
    for rec in records:
        doc_len = sum(rec.sparse_vector.values())  # ← 计算 DL
        for term, tf in rec.sparse_vector.items():
            inverted[term].append({...})           # ← 构建倒排索引

    self._avg_doc_length = mean(doc_lengths)       # ← 计算 AVGDL

    for term, postings in inverted.items():
        df = len(postings)                         # ← 计算 DF
        self._idf[term] = log((N-df+0.5)/(df+0.5))  # ← 计算 IDF
```

### 7.3 BM25Indexer.query（在线检索）

```python
# 文件: src/ingestion/storage/bm25_indexer.py

def query(self, keywords, top_k=10, k1=1.5, b=0.75):
    for term in keywords:
        idf = self._idf[term]                      # ← 查 IDF
        for posting in self._inverted_index[term]:  # ← 遍历 Posting List
            tf = posting["tf"]
            dl = posting["doc_length"]
            # ↓ BM25 公式
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / self._avg_doc_length)
            score = idf * (numerator / denominator)
            scores[chunk_id] += score              # ← 多词分数累加

    return sorted(scores, reverse=True)[:top_k]    # ← Top-K 排序
```

---

## 八、算法复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 分词 | O(L) | L = 文本字符数 |
| 计算 TF | O(W) | W = 词数 |
| 构建倒排索引 | O(N × W̄) | N = 文档数, W̄ = 平均词数 |
| 计算 IDF | O(V) | V = 词汇表大小 |
| 单次查询 | O(Q × P̄) | Q = 查询词数, P̄ = 平均 Posting List 长度 |
| 排序 Top-K | O(D × log K) | D = 命中文档数 |

**空间复杂度**：O(N × W̄) — 倒排索引存储所有 posting

---

## 九、与经典 BM25 实现的差异

| 方面 | 经典 BM25 (如 Elasticsearch) | 本项目实现 | 差异原因 |
|------|---------------------------|-----------|---------|
| TF 计算时机 | 检索时实时计算 | Ingestion 时预计算 | 空间换时间 |
| TF 值 | 原始词频 (1, 2, 3...) | 归一化词频 (0.143, 0.286...) | SparseEncoder 已归一化 |
| IDF 语料库 | 全局语料 | 当前批次文档 | 增量更新时需重建 |
| 分词器 | 专业分词器 (ICU, Jieba) | 简单正则 | 轻量实现 |
| 持久化 | Lucene 索引格式 | Python Pickle | 简单直接 |

---

## 十、局限性与改进方向

| 局限 | 具体表现 | 改进方案 |
|------|---------|---------|
| **不支持中文** | 正则 `\b[a-zA-Z0-9]+\b` 无法匹配中文字符 | 引入 jieba 分词器 |
| **无词干化 (Stemming)** | "learning" 和 "learned" 被视为不同词 | 引入 Porter/Snowball Stemmer |
| **无词形还原 (Lemmatization)** | "better" 不会映射到 "good" | 引入 WordNet Lemmatizer |
| **无同义词扩展** | "car" 和 "automobile" 完全不匹配 | 引入同义词词典 |
| **IDF 非全局** | 只基于当前批次，新增文档后 IDF 不准 | 维护全局 IDF 表，增量更新 |
| **无位置信息** | 不知道词在文档中的位置 | 记录 position，支持短语查询 |
| **无子词处理** | "ChatGPT" 不会拆成 "Chat"+"GPT" | 引入 BPE/WordPiece |

---

## 十一、知识点总结 — 你需要记住的核心

### 11.1 一句话总结每个概念

| 概念 | 一句话 |
|------|--------|
| **Sparse Vector** | 把文本变成 {词: 权重} 的字典，大部分词权重为 0 |
| **TF** | 一个词在文档中出现几次 |
| **IDF** | 一个词在全局有多"稀有" |
| **BM25** | TF × IDF 的改进版，加了饱和控制和长度归一化 |
| **倒排索引** | 从"词→文档列表"的反向映射，实现 O(1) 查找 |
| **k1** | 控制"出现多次"的边际收益衰减速度 |
| **b** | 控制"长文档"被惩罚的力度 |

### 11.2 记忆口诀

```
BM25 三要素：
  1. IDF 管"稀有度" — 越稀有越重要
  2. TF  管"相关度" — 出现越多越相关（但有上限）
  3. DL  管"公平性" — 长文档要打折

公式直觉：
  分数 = 稀有度 × 相关度 / (相关度 + 长度惩罚)
```

### 11.3 面试/学习检查清单

- [ ] 能解释 TF、IDF、BM25 各自的含义
- [ ] 能说出 k1 和 b 参数的作用
- [ ] 能解释为什么需要归一化 TF
- [ ] 能解释倒排索引的结构和优势
- [ ] 能说出 Dense 和 Sparse 各自擅长什么场景
- [ ] 能解释为什么 Hybrid Search 需要两者结合
- [ ] 能手算一个简单的 BM25 评分示例

---

## 十二、延伸阅读

| 资源 | 说明 |
|------|------|
| [BM25 原始论文](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz) | Robertson & Walker, 1994 |
| [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) | Stanford 经典教材，免费在线 |
| [Elasticsearch BM25 解释](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) | 工业级实现的参数调优指南 |
| [SPLADE](https://arxiv.org/abs/2107.05720) | 用神经网络学习 Sparse 向量的前沿方法 |
