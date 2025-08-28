# Normalizations in LLMs
> 本章节记录关于LLMs 训练、推理等常见的归一化方法面试QA

### Q. 为何使用 RMSNorm 代替 LayerNorm？
> **Company**: 阿里 | **Round**: 算法工程师 一面 ｜ **Date**: 2025-08-14 ｜ **Tags**: [RMSNorm, LayerNorm, 归一化]

- **LayerNorm 机制**：  
  - 对输入向量 $x$ 做归一化：  
    $$
    \text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
    $$  
    其中 $\mu$ 是均值，$\sigma^2$ 是方差。  
  - 好处：稳定训练、缓解梯度消失/爆炸。  
  - 问题：计算均值需要额外开销；对均值的减法可能引入噪声。  

- **RMSNorm 机制**：  
  - 只利用 **均方根 (RMS)**，不减去均值：  
    $$
    \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}} \cdot \gamma
    $$  
  - 特点：没有 $\beta$ 偏置项，也不做中心化。  
  - 计算更简单，速度更快，显存开销更小。  

- **为何替代**：  
  - **更高效**：减少均值计算，推理更快，内存更省。  
  - **更稳定**：在大规模 LLM（如 GPT、LLaMA）中，实验发现去掉均值不会影响甚至提升收敛稳定性。  
  - **对齐实践**：很多最新模型（如 LLaMA 系列）都用 RMSNorm 替代 LayerNorm，验证了其可行性。  
  - **简化参数**：少一个偏置参数 $\beta$，模型更轻量。  

<mark>核心：RMSNorm 去掉了均值归一化，只用 RMS 缩放，减少计算与显存开销，同时在大模型中保持甚至提升稳定性，因此逐渐替代 LayerNorm。</mark>

### Q. RMSNorm与LayerNorm在数学公式上的核心区别是什么？
> **Company**: 阿里 | **Round**: 算法工程师 一面 ｜ **Date**: 2025-08-14 ｜ **Tags**: [RMSNorm, LayerNorm, 归一化]

- **LayerNorm**：  
  - 对输入 $x \in \mathbb{R}^n$ 做均值和方差归一化：  
    $$
    \text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
    $$  
    其中：  
    $\mu = \frac{1}{n}\sum_{i=1}^n x_i$，$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2$。  
  - **核心**：先减去均值再除以标准差，做了**零均值 + 单位方差**的归一化。

- **RMSNorm**：  
  - 只依赖输入向量的均方根，不减均值：  
    $$
    \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}} \cdot \gamma
    $$  
  - **核心**：仅缩放输入，不做中心化；归一化因子是 RMS 而非标准差。

- **核心区别总结**：  
  1. **是否减均值**：LayerNorm 要减去均值 $\mu$，RMSNorm 不需要。  
  2. **分母不同**：LayerNorm 用标准差 $\sqrt{\sigma^2}$；RMSNorm 用均方根 $\sqrt{\frac{1}{n}\sum x_i^2}$。  
  3. **参数不同**：LayerNorm 有 $\gamma, \beta$；RMSNorm 只有缩放参数 $\gamma$。  

<mark>核心：LayerNorm = “减均值 + 除标准差”；RMSNorm = “不减均值 + 除 RMS”，因此 RMSNorm 更简洁高效，但不保证零均值。</mark>