# Agent's Foundations
> 此章节记录Agent相关的基础知识点。

### Q. 在 Agent 多轮对话任务中，你觉得 Attention 的局限性体现在哪些方面？
> **Company**: 淘天 ｜ **Round**: Agent智能体 一面 ｜ **Date**: 2025-08-26 ｜ **Tags**: [Attention, Agent]

1. **长程记忆受限**  
   - Attention 本质上依赖上下文窗口（context window），即便做了 128K、1M 的扩展，也只是“滑动记忆”，并没有真正的长期对话记忆。  
   - 多轮对话里，模型可能遗忘早期关键信息，或者需要重复读入大量历史，效率低下。

2. **计算和存储成本高**  
   - Attention 复杂度为 $O(n^2)$，多轮对话时 token 数不断累积，推理成本指数级增加。  
   - 即便有 FlashAttention、稀疏 Attention 优化，本质上还是受限于序列长度。

3. **缺乏显式状态建模**  
   - Attention 是“软对齐”，只能在 token 之间分配权重，缺乏显式的对话状态（如用户意图、任务进度）。  
   - 这使得在多轮对话任务中，Agent 可能出现“前后不一致”或“丢上下文”的情况。

4. **易受干扰与噪声影响**  
   - 长对话中出现无关 token（闲聊、噪音），Attention 可能错误分配权重，导致模型“跑题”。  
   - Agent 需要更强的检索/记忆机制，而不仅仅依赖 Attention。

5. **难以跨任务整合外部知识**  
   - 单纯 Attention 无法直接访问外部工具或数据库。  
   - 在复杂 Agent 任务中，如果缺乏检索器、记忆模块或规划器，仅靠 Attention 不能满足需求。

<mark>Attention 在多轮 Agent 对话中的局限主要体现在“只会记 token，不会记任务”，既有长程依赖和计算瓶颈，又缺乏显式状态与外部知识接入能力。</mark>
