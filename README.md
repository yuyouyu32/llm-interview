# llm-interview
> LLMs 算法工程师面试资料整理：覆盖基础理论、常见面试问题等。愿这是大家迈向“美好生活”的直通车 🚆，祝每位同学都拿到心仪的 offer！欢迎提交 PR / Issue，一起协作开发与长期维护。

## 目录结构说明

### 0-foundations/

- Transformer 架构与注意力机制
- 分词方法（BPE、SentencePiece）
- Scaling Laws 与算力/参数/数据的配比
...

### 1-training/
LLM 的训练流程，细分为：
- **1.1 Pre-Training**：预训练目标、语料治理与优化方法  
- **1.2 SFT Fine-Tuning**：监督微调（SFT）
- **1.3 RLs**: 强化学习（RL）相关方法

### 2-inference/
推理与部署相关内容：
- KV-Cache 管理与优化
- 解码策略（采样、Beam、Speculative Decoding 等）
- 多卡部署、推理加速框架（vLLM、TensorRT-LLM）

### 3-evaluation/
模型评测与常见基准：
- 通用任务（MMLU）
- 对话能力（MT-Bench）
- 真实性与鲁棒性（TruthfulQA 等）

### 4-infra/
模型压缩与加速：
- 量化方法（GPTQ、AWQ、SmoothQuant）
- 蒸馏方法（DistilGPT、MiniLM）
- 剪枝与稀疏化

### 5-agent/
面向应用与系统化：
- **Retrieval**：RAG 与知识库检索  
- **Reranking**：排序与重排模型  
- **Orchestration**：多工具调用与 Agent 工作流设计

### 6-major llms/
主流 LLMs 介绍与对比

### 7-case studies/
典型面试题目与解答，按场景分类：
- **1. Game**：游戏行业相关
- **2. E-commerce**：电商行业相关


## 🤝 如何贡献面试 QA

欢迎提交 PR！  
请将你的面试题目（Q&A）放到 **对应的子目录** 下，例如：  
- 预训练相关 → `1-training/1.1-pre-training/`  
- 后训练/对齐 → `1-training/1.2-SFT/`  
- 推理/部署 → `2-inference/`  
- …  

### ✍️ 填写规范
- Q：面试题目，标题以 ### Q. 开头
- Meta 信息：公司、面试轮次、时间、标签（用于检索）
- A：你的回答，结构化清晰，必要时使用列表或表格

Notes：可选，记录面试官追问、思考补充或参考资料

### 📌 模版示例

```markdown
### Q. 预训练阶段中，如何处理大规模语料的重复数据问题？

> **Company**: 腾讯 IEG 光子工作室 | **Round**: 青云计划-应用研究 二面 | **Date**: 2025-08-21 | **Tags**: [pre-training, data, cleaning]

去重常见方法有：
1. **哈希去重**：对句子或文档内容取哈希值，快速发现完全相同的数据；
2. **MinHash/SimHash**：近似去重，发现部分改写或轻微差异的重复文本；
3. **Embedding 相似度阈值**：利用向量模型计算语义相似度，过滤掉冗余样本。

此外，还需结合 **质量控制**（过滤低质量网页、广告、爬虫数据），确保训练数据分布的多样性与覆盖度。

<mark>**Notes**: 面试官追问了 **语料去重对 Scaling Laws 的影响**，可以补充预训练时数据-参数-算力的平衡点。<mark>
````

