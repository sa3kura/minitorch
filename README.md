# CMU 11-868：Large Language Model Systems

记录一下我的homework

- Assignment1: 自动微分框架 + CUDA 手写算子 + 基础神经网络构建
  实现自动微分框架 Minitorch，完成计算图拓扑排序、反向传播与梯度累积机制，支持基础 Tensor 运算和神经网络训练。
  手写 CUDA Tensor 算子，包括 map、zip、reduce 和矩阵乘法，支持 stride-based indexing、broadcasting 与 Python 后端集成等。
  基于自研自动微分框架构建基础神经网络模块，实现 Linear、Activation、Dropout、Loss Function 和训练流程。
  
- Assignmant2: GPT2 模型构建
  基于 Minitorch 实现 GPT-2 模型结构，包括 Token / Position Embedding、Causal Multi-Head Self-Attention、FeedForward、Residual Connection 和 LayerNorm、FeedForward、Residual Connection 和 LayerNorm。

- Assignment3: 通过手写 CUDA 的 Softmax 和 LayerNorm 算子优化模型训练速度
 针对 GPT-2 训练中的 Softmax 与 LayerNorm ，手写 CUDA kernel 进行算子级优化，利用并行 reduction、shared memory 和数值稳定计算提升模型训练速度。

- Assignment4: 分布式模型训练，自学的话可能不太好配置环境
