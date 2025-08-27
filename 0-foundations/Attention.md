# [Attention is all you need](https://arxiv.org/abs/1706.03762)
> 本章节记录 Attention 相关的面试题目。


### Q. Attention的公式，公式为什么要除以$\sqrt{d_k}$，为什么不是原来的值或者是立方根d_k？
> **Company**: 阿里国际 ｜ **Round**: 算法工程师 一面 ｜ **Date**: 2025-08-01 ｜ **Tags**: [Attention, 归一化]

- **Scaled Dot-Product Attention 公式：**
  $$
  \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$
  其中 $Q,K,V$ 的大小：$Q,K,V \in \mathbb{R}^{n \times d}$，$d_k$ 是 key 的维度。

- **为什么要除以 $\sqrt{d_k}$？**
  - 若不缩放，$QK^T$ 的期望值和方差会随 $d_k$ 增大而变大。  
  - 内积值过大 → softmax 指数函数饱和 → 导致梯度消失，训练困难。  
  - 除以 $\sqrt{d_k}$ 可以把内积的方差归一化到常数水平（大约保持在 1），避免数值不稳定。

- **为什么不是立方根或其他函数？**
  - $QK^T$ 是 $d_k$ 个独立随机变量乘积的和，方差 $\propto d_k$。  
  - 标准差（方差的平方根）才是合适的缩放因子 → 所以用 $\sqrt{d_k}$。  
  - 如果用 $d_k$ 或 $\sqrt[3]{d_k}$，缩放效果会过强或不足，数值不再平衡。

<mark>Attention 公式中除以 $\sqrt{d_k}$ 是为了归一化内积的方差，避免数值过大导致 softmax 饱和和梯度消失；因为内积的方差与 $d_k$ 成正比，标准差与 $\sqrt{d_k}$ 成正比，所以选择 $\sqrt{d_k}$ 作为缩放因子。</mark>

- 输入序列长度 = $n$，隐藏维度 = $d$。  
- 主要计算步骤：
  1. $QK^T$ → $O(n \times d \times n) = O(n^2 d)$
  2. softmax($n \times n$ 矩阵) → $O(n^2)$  
  3. softmax 矩阵 × $V$ → $O(n^2 d)$  
- **总复杂度：**
  $$
  O(n^2 d)
  $$

- **瓶颈：**
  - 当序列长度 $n$ 很大时，$O(n^2)$ 成本过高，显存也要存储 $n \times n$ 注意力矩阵。  
  - 这就是为什么有 FlashAttention、Sparse Attention、Linear Attention 等改进。  

**一句话总结：**  
<mark>Attention 的主要成本来自两个 $n \times n$ 矩阵乘法，时间复杂度是 $O(n^2 d)$，瓶颈在序列长度平方项。</mark>