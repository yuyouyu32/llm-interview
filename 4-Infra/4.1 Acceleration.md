# LLMs Acceleration Kits
> 本章节记录 LLMs 加速相关工具的面试题目。

## 1. DeepSpeed

### Q. DeepSpeed ZeRO 的核心机制是什么？
> **Company**: 字节topseed | **Round**: 算法工程师 一面 | **Date**: 2025-08-04 | **Tags**: [DeepSpeed, ZeRO, 显存优化]

![ZeRO Stages](https://developer.qcloudimg.com/http-save/yehe-4356113/fcf9846d4d2088174a47a5e6668545ae.png)

ZeRO 的核心机制  
ZeRO（Zero Redundancy Optimizer）通过**分片 (partitioning)**，把训练三大块（参数、梯度、优化器状态）拆开分布到不同 GPU，只在需要时通信/聚合，从而大幅降低显存。

- **Stage 1 – 优化器状态分片**：Adam 的动量变量分片到各 GPU，每卡只保留一部分。  
- **Stage 2 – 梯度分片**：每卡只存自己负责参数的梯度，而不是全量梯度。  
- **Stage 3 – 参数分片**：连参数本身也分片存储，前向/反向时再 all-gather。  

📊 以 7B 模型为例的显存占用对比  
假设 7B 参数，[FP16 权重（2$\times$7B），Adam 优化器状态（8$\times$7B = 存1，2阶导数 $\times 2$ * FP32精度 $\times 4$）](https://blog.csdn.net/u013010473/article/details/147605204#:~:text=%E5%85%A8%E5%8F%82%E6%95%B0%E5%BE%AE%E8%B0%83%E9%80%9A%E5%B8%B8%E4%BD%BF%E7%94%A8%20AdamW%20%E7%AD%89%E4%BC%98%E5%8C%96%E5%99%A8%EF%BC%8C%E5%AE%83%E4%BC%9A%E4%B8%BA%E6%AF%8F%E4%B8%AA%E5%8F%82%E6%95%B0%E5%AD%98%E5%82%A8%E9%A2%9D%E5%A4%96%E7%9A%84%E7%8A%B6%E6%80%81%EF%BC%88%E5%A6%82%E4%B8%80%E9%98%B6%E7%9F%A9%E5%92%8C%E4%BA%8C%E9%98%B6%E7%9F%A9%EF%BC%89%E3%80%82%20AdamW%20%E9%80%9A%E5%B8%B8%E5%AD%98%E5%82%A8%E4%B8%A4%E5%80%8D%E4%BA%8E%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E9%87%8F%E7%9A%84%E7%8A%B6%E6%80%81%EF%BC%8C%E4%B8%94%E6%9C%89%E6%97%B6%E4%B8%BA%E4%BA%86%E7%A8%B3%E5%AE%9A%E6%80%A7%EF%BC%8C%E5%8D%B3%E4%BD%BF%E6%A8%A1%E5%9E%8B%E7%94%A8%20BF16%2FFP16%20%E8%AE%AD%E7%BB%83%EF%BC%8C%E4%BC%98%E5%8C%96%E5%99%A8%E7%8A%B6%E6%80%81%E4%B9%9F%E5%8F%AF%E8%83%BD%E4%BF%9D%E6%8C%81%20FP32,%2A%202%20%2A%204%20bytes%20%E2%89%88%2060.8%20GB%E3%80%82)，8 卡训练。

| 模式                | 参数 (GB) | 梯度 (GB) | 优化器状态 (GB) | 总显存 (GB/卡) | 相对 DP |
|---------------------|-----------|-----------|-----------------|----------------|---------|
| Data Parallel (DP)  | 14        | 14        | 56              | **84**         | 1.0×    |
| ZeRO Stage 1        | 14        | 14        | 7               | **35**         | 2.4×    |
| ZeRO Stage 2        | 14        | 1.75      | 7               | **22.75**      | 3.7×    |
| ZeRO Stage 3        | 1.75      | 1.75      | 7               | **10.5**       | 8.0×    |

**直观理解**  
- **DP**：每卡都要放“全套家具”（参数+梯度+优化器状态） → 84 GB，直接爆显存。  
- **ZeRO**：把这三块“大家合租分担”，每人只放一部分 → Stage 3 只需 ~10.5 GB/卡。  

**效果**  
- 7B 模型：DP 需要每卡 84 GB，ZeRO Stage 3 只要 10.5 GB。  
- 节省近 8 倍显存，让大模型可以在 24 GB 甚至更小的 GPU 上训练。  
- 与混合精度、流水并行结合后，可扩展至万亿参数模型。

<mark>DeepSpeed ZeRO 的核心机制是“分片存储参数、梯度和优化器状态”，逐阶段把冗余消除干净，以 7B 模型为例，显存占用从 84 GB 降到 10.5 GB/卡，节省近 8 倍，使大规模训练成为可能。</mark>


### Q. DeepSpeed Stage3 数据流程详细描述
> **Company**: None | **Round**: None | **Date**: None | **Tags**: [DeepSpeed, ZeRO, Stage3, 数据流程]

ZeRO Stage 3 的核心是 **参数分片**：每张 GPU 只保存自己负责的参数分片，在需要计算时才临时广播 (broadcast) 到所有 GPU；计算完成后立即释放，以达到最小显存占用。下面以 4 张 GPU、4 层模型（M0~M3）为例，描述 forward 与 backward 的完整数据流。

1. 在 **forward 阶段**，例如 M0 层只有 GPU0 保存了参数，它会先 **broadcast 参数到 GPU1/2/3**。这样所有 GPU 都能用本地数据（data0/1/2/3）和完整的 M0 参数完成前向计算。**激活值必须保留下来**以备反向传播使用。计算完成后，GPU1/2/3 上的参数立刻释放，只保留分片所在的 GPU，随后依次对 M1、M2 重复这个过程。**最后一层 M3 的参数在 forward 结束后不释放**，因为 backward 还需要用到。

2. 在 **backward 阶段**，从 M3 开始，所有 GPU 利用 **M3 参数+激活值+loss** 计算本地梯度，并将结果 **聚合到参数所在的 GPU**（如 GPU3）。完成累积后，其他 GPU 的参数和梯度释放，同时各 GPU 的激活值也被删除。接着处理 M2、M1、M0，逐层回退。**每一层都是“参数广播—梯度计算—梯度归约—释放内存”的循环**。

3. 最后在 **优化器更新**阶段，优化器状态以 **FP32** 精度保存并更新，模型权重以 **FP16** 精度存储。**更新后的参数重新分片**到各 GPU，为下一轮迭代做准备。

<mark>Stage3 的数据流是：前向时“参数广播、激活值保留、用后即删”，反向时“梯度计算、归约累积、逐层释放”，最终在 FP32 优化器下更新参数，再以 FP16 分片保存。这种“用时加载、用后释放”的机制大幅降低显存开销，使大模型训练可行。</mark>

**参考视频链接: [click here~](https://www.bilibili.com/video/BV1C44y1Y7Lz/?vd_source=6e48849af2164223890124b90ffd9c5e)**

## 2. FlashAttention

### Q. FlashAttention 的核心改进是什么？为什么可以提升attention计算效率？
> **Company**: None | **Round**: None | **Date**: None | **Tags**: [FlashAttention, Attention, 计算效率]

**Paper**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://papers.nips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)

- **核心改进**：提出了一个 **IO-Aware 的精确 Attention 算法**，通过减少显存 (HBM) 的读写次数，提升速度和内存效率。
  1. **分块 (Tiling)**：将 Q、K、V 矩阵分成小块，逐块加载到 GPU 的 **片上 SRAM** 中进行计算，避免一次性将整个 $N \times N$ 矩阵放入慢速 HBM，减少内存传输。 
  2. **重计算 (Recomputation)**：在反向传播时，不再存储完整的注意力矩阵，而是只保留 **softmax 归一化因子**，并在需要时快速重算注意力值，从而减少 HBM 的中间结果读写。  
  3. **融合操作 (Kernel Fusion)**：将 matmul、softmax、dropout、再 matmul 等步骤融合到一个 **单一 CUDA kernel** 内完成，避免多次 HBM 读写，大幅降低 IO 开销。  
  4. **扩展到稀疏注意力 (Sparse FlashAttention)**：支持块稀疏版本，仅计算非零块对应的注意力，进一步降低 **计算复杂度和内存访问量**，使长序列任务更加高效。  



<mark>FlashAttention 通过 IO 感知的优化（tiling、在线 softmax、kernel fusion），显著减少显存读写，从而在保持精确计算的同时，大幅提升速度与内存效率。</mark>


### Q. GPU 分区是怎么做的？
> **Company**: 阿里 | **Round**: 算法工程师 一面 ｜ **Date**: 2025-08-26 ｜ **Tags**: [吞吐率优化, GPU 分区]

- **GPU 分区 (GPU Partitioning) 的思路**  
  - **显存分区**：将显存切分为 **KV Cache 区、计算区、通信缓冲区**，避免相互干扰。  
  - **计算分区**：利用 **block/warp 级并行**，不同 SM 处理不同 tile 的 QK，重叠计算与访存。  
  - **流水线并行**：一部分 GPU 计算前向 attention，另一部分同时做 softmax 与 $PV$，提升并行度。  
  - **多租户/多流 (CUDA Streams)**：把不同推理请求映射到不同 stream，实现并行调度，提高 GPU 利用率。  

<mark>GPU 分区则从 **显存管理与并行调度** 层面提升利用率，常与 KV Cache/批处理/流水并行结合使用。</mark>


### Q. 如何优化大模型训练速度？
> **Company**: 同花顺 ｜ **Round**: 大模型算法工程师 一面 ｜ **Date**: 2025-04-15 ｜ **Tags**: [训练速度优化, 大模型]

**1. 算法层面优化**  
- **混合精度训练 (FP16/BF16)**：降低数值精度，减少显存带宽与计算量，同时保持模型精度。  
- **梯度累积 (Gradient Accumulation)**：在小显存下模拟大 batch，提高稳定性与吞吐。  
- **梯度检查点 (Gradient Checkpointing)**：节省中间激活存储，降低显存开销，换取少量计算量。  
- **优化器选择**：使用更高效的优化器（如 AdamW → Lion/Adafactor）减少计算负担。  

**2. 模型结构优化**  
- **参数高效微调 (PEFT)**：LoRA、Prefix Tuning 等，仅训练部分参数，避免全量更新。  
- **模型压缩与蒸馏**：减少模型规模，在保证性能的同时加快训练迭代。  
- **稀疏化 (Sparsity)**：引入稀疏注意力（如 Longformer、FlashAttention），降低计算复杂度。  

**3. 系统与并行策略优化**  
- **数据并行 (DP)**：多个 GPU 同步训练，提升吞吐。  
- **模型并行 (MP)**：拆分权重/层结构到不同设备，支持超大模型。  
- **流水并行 (Pipeline Parallelism)**：按层切分模型流水训练，提高利用率。  
- **张量并行 (Tensor Parallelism)**：矩阵分块并行计算，加快矩阵乘法。  
- **混合并行 (Hybrid Parallelism)**：结合 DP+MP+PP，适配超大规模 LLM 训练。  

**4. 工程与硬件优化**  
- **高效数据加载**：数据预处理、缓存、分布式文件系统（如 WebDataset）。  
- **高效算子**：使用 fused kernel（如 Apex、DeepSpeed）、FlashAttention。  
- **硬件优化**：利用 GPU/TPU/NPU 新指令集（如 NVIDIA Tensor Core、H100 FP8）。  
- **集群调度优化**：减少通信瓶颈，优化 NCCL/通信拓扑。  

**5. 框架与工具**  
- **DeepSpeed / Megatron-LM**：提供 ZeRO 优化器、内存分片、通信优化。  
- **Colossal-AI / FSDP**：全参数分布式训练，降低显存占用。  
- **Accelerate / Ray**：简化多机多卡训练，提升工程效率。  

<mark>**总结**：大模型训练速度优化需要算法（混合精度、梯度优化）、结构（PEFT、稀疏化）、系统（并行策略）、工程（高效算子与数据加载）、硬件（GPU/TPU 特性）多层协同。实际场景往往结合 **混合精度 + ZeRO 优化 + 混合并行**，是目前工业界的主流方案。</mark>
