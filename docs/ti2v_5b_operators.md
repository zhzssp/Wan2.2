# Wan2.2 TI2V-5B 的算子与计算图

本文统计 TI2V-5B 推理实际执行的算子，并说明计算图是由哪段代码定义的。调用顺序、张量形状和采样公式见 [TI2V-5B 推理范围与 Pipeline](ti2v_5b_inference.md)。多卡只改变一次 DiT 前向内部的张量切分，不改变算子种类，切分方式见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。

次数按默认文生视频计算：`1280*704`、121 帧、50 步、单卡。latent 为 `[48, 31, 44, 80]`，patch 之后序列长度 27280，隐藏维 3072，30 层，24 个 head。CFG 使 DiT 前向次数为 50 × 2 = 100。图生视频多一次 VAE 编码，算子种类与解码相同。

## 1. 计算图如何定义

仓库里没有导出的静态图。没有 `torch.fx`、`torch.compile`、ONNX，也没有单独的计算图描述文件。图就是两层东西叠在一起：

1. **外层控制流**写在 `wan/textimage2video.py` 的 `WanTI2V.t2v()`（有图像时走 `i2v()`）。它是普通 Python：先调用 T5，再采样噪声，然后 `for t in timesteps` 里调用 DiT 和调度器，最后调用 VAE。
2. **内层算子**由 `nn.Module` 的子模块和 `forward` 连起来。DiT 在 `wan/modules/model.py` 的 `WanModel`，文本编码器在 `wan/modules/t5.py`，VAE 在 `wan/modules/vae2_2.py`。

整段去噪包在 `torch.no_grad()` 里，因此没有反向图，参数也不更新。循环展开与否、某一层是否执行，都由 Python 在运行时决定，不是一张事先冻结的 DAG。

外层数据依赖是：

```text
T5(正向 prompt) ──┐
T5(负向 prompt) ──┼─► 初始噪声 latent
                  │         │
                  │         ▼
                  │    ┌─ DiT(latent, t, 正向文本)
                  │    ├─ DiT(latent, t, 负向文本)
                  └────┤  CFG：uncond + scale * (cond - uncond)
                       ├─ scheduler.step → 下一步 latent
                       └─ 重复 50 步
                                 │
                                 ▼
                            VAE.decode → 像素
```

50 步之间不能对调。下一步的输入就是 `scheduler.step` 的输出。同一步里的条件前向和无条件前向读同一份 latent，彼此不依赖，代码里仍是先后两次调用。30 层 `WanAttentionBlock` 同样是串行的：下一层的输入是上一层的输出。能并排计算的是同一次 `Linear` 或注意力内部的行和 head，不是层与层、步与步。

一次 `WanModel.forward` 内部的图是：

```text
Conv3d patch → [1, 27280, 3072]
    ├─ 时间嵌入（正弦编码 + Linear / SiLU）→ 6 组调制，fp32
    └─ 文本嵌入（Linear + GELU + Linear）→ [1, 512, 3072]
           │
           ▼
    重复 30 次 WanAttentionBlock
      调制 → LayerNorm / RMSNorm → QKV Linear → RoPE
        → 自注意力 → 输出 Linear → 残差
      调制 → 交叉注意力 → 残差
      调制 → FFN Linear → GELU → Linear → 残差
           │
           ▼
    Head 的 LayerNorm + Linear → einsum 还原成 [48, 31, 44, 80]
```

TI2V 的 `model_type` 不是 `i2v`，`forward` 里拼接条件通道 `y` 的分支不执行。图像条件在 `i2v()` 里写成第 0 帧 latent 和按 token 变化的时间步，再进入上面这张图。

VAE 解码的图更窄。`WanVAE_.decode` 按 latent 的时间维循环 31 次，每次把一帧送进 `Decoder3d`。`CausalConv3d` 把上一帧留下的边界放在 cache 里拼到当前帧前面，所以帧序是这张子图的依赖边。解码器中部有一块单头空间注意力，和 DiT 的注意力不是同一次调用。

## 2. 算子统计

下表只计会在 `eval()` 前向里执行的算子。T5、VAE 模块定义里有 `Dropout`，`dropout=0` 或 `eval()` 之后它不改变数值，不计入。

| 算子 | 代码位置 | 默认文生视频执行次数 | 在图中的位置 |
| --- | --- | --- | --- |
| `Linear` | DiT 的 QKV、输出投影、FFN、时间嵌入、文本嵌入、Head；T5 的 QKV 与门控 FFN | DiT 每次前向 306 次，100 次前向共约 3.06 万次；另加 T5 的两次编码 | 热路径。层内 FFN 是 `3072 ↔ 14336`，QKV 是 `3072 ↔ 3072` |
| 自注意力 | `WanSelfAttention` 调用 `wan/modules/attention.py` 的 `flash_attention` | 30 层 × 100 次前向 = 3000 | QKᵀ、softmax、乘 V。序列长 27280，24 head，head 维 128 |
| 交叉注意力 | `WanCrossAttention`，同一个 `flash_attention` | 3000 | query 长 27280，key/value 最长 512，无 RoPE |
| `RMSNorm` / `LayerNorm` | DiT 每层的 Q/K RMSNorm 与三处 LayerNorm，Head 再一处；T5 每层两处 | 与层数同量级，夹在 Linear 前后 | 沿通道维归约 |
| RoPE | `model.py` 的 `rope_apply`，只作用在 DiT 自注意力的 Q、K | 每层 2 次，共 3000 层调用 × 2 | 按网格 `(31, 22, 40)` 做复数乘 |
| GELU（`approximate='tanh'`）、SiLU | DiT 的 FFN、文本嵌入、时间嵌入 | 紧跟对应的 Linear | 逐元素 |
| 调制（加、乘） | `WanAttentionBlock` 用 6 组时间向量做缩放、平移和门控 | 每层 1 组 | 在 fp32 autocast 中完成 |
| `Conv3d` | DiT 的 `patch_embedding`，核与步长都是 `(1, 2, 2)` | 100 | 每次前向开头，把 latent 切成 token |
| `einsum` 注意力 + `softmax` | `T5Attention` | 2 × 24 层 = 48 | 序列不超过 512，分数矩阵会物化出来 |
| `Embedding` | T5 词嵌入；相对位置桶嵌入 | 2 次编码 | 查表，发生在去噪循环之前 |
| `CausalConv3d`、`Conv2d`、最近邻上采样、`scaled_dot_product_attention` | `vae2_2.py` 的 `Decoder3d` | 解码 1 次；内部 31 帧串行。中部单头注意力每帧一块 | 循环结束之后，只在 rank 0 |
| 逐元素更新 | `FlowUniPCMultistepScheduler.step`，以及 CFG 的加减乘 | 50 步 | 连接两次 DiT 前向和下一步 latent，不是神经网络层 |

一次 DiT 前向的 306 个 `Linear` 来自：时间嵌入 3 个（`freq_dim→3072`、`3072→3072`、`3072→3072×6`），文本嵌入 2 个，30 层各 10 个（自注意力 4、交叉注意力 4、FFN 2），Head 1 个。注意力次数是 30 层自注意力加 30 层交叉注意力，再乘 100 次前向。

T5 每个视频只进入图两次，序列长度不超过 512，不进入 50 步循环。VAE 解码每个视频进入一次。DiT 的 100 次前向占掉几乎全部 `Linear` 和 FlashAttention 调用。

## 3. 三张子图分别由哪些类组成

### 3.1 DiT：`WanModel`

`WanModel.__init__` 把子模块挂成树，`forward` 规定它们的顺序。

| 子模块 | 算子 | 一次前向执行几次 |
| --- | --- | --- |
| `patch_embedding` | `Conv3d` | 1 |
| `time_embedding`、`time_projection` | `Linear`、`SiLU`、正弦编码 | 各 1 次 |
| `text_embedding` | `Linear`、`GELU` | 1，结果被 30 层共用 |
| `blocks[i]` | 见下一表 | 30，顺序执行 |
| `head` | `LayerNorm`、`Linear`、调制 | 1 |
| `unpatchify` | `einsum` 与 `reshape` | 1 |

一层 `WanAttentionBlock` 的执行顺序是固定的：

1. 把调制向量加到该层的 `modulation` 参数上，切成 6 段。
2. `norm1`（LayerNorm）之后做缩放和平移，再进入 `WanSelfAttention`：Q/K/V 三个 `Linear`，Q/K 做 `WanRMSNorm`，RoPE，`flash_attention`，输出 `Linear`。残差乘第三段门控。
3. `norm3` 之后进入 `WanCrossAttention`：同样四个 `Linear` 和一次 `flash_attention`，query 来自视频 token，key/value 来自文本，不做 RoPE。
4. `norm2` 之后进入 FFN：`Linear(3072→14336)`、`GELU(approximate='tanh')`、`Linear(14336→3072)`，残差乘第六段门控。

`flash_attention` 是这张图里唯一绑定 CUDA 的算子。它要求输入在 CUDA 上，并调用 FlashAttention 2 或 3。同文件的 `scaled_dot_product_attention` 退路没有被 `WanSelfAttention` 或 `WanCrossAttention` 调用。

### 3.2 文本编码器：`T5Encoder`

`T5EncoderModel.__call__` 先分词，再跑 `T5Encoder.forward`。编码器 24 层，每层是 `T5SelfAttention`：`T5LayerNorm`、四个无 bias 的 `Linear`、`einsum` 得到分数、fp32 `softmax`、再一次 `einsum` 乘 V，然后是门控 FFN（`fc1(x) * GELU(gate(x))` 再 `fc2`）。相对位置偏差来自 `nn.Embedding`。Decoder 和词表投影头不在这条前向里。

### 3.3 VAE：`Wan2_2_VAE`

文生视频只执行 `decode`。`Decoder3d` 的主体是 `ResidualBlock` 里的 `CausalConv3d`（底层是 `F.pad` 加 `Conv3d`）、`RMS_norm`（`F.normalize`）、`SiLU`，上采样处还有最近邻插值和 `Conv2d`。`middle` 里固定有一个 `AttentionBlock`：把时间维折进 batch，用 `Conv2d` 产生 QKV，再调用 `F.scaled_dot_product_attention`，这是单头空间注意力。`encode` 的 `Encoder3d` 结构对称，只在图生视频的第 0 帧使用。

## 4. 不在执行图里的东西

- **反向与优化器。** `requires_grad_(False)` 且包在 `no_grad()` 中。FSDP 的 `no_sync()` 只是跳过梯度同步，不增加算子。
- **Dropout。** 模块里写了，推理时不生效。
- **DiT 的 `y` 通道拼接。** 仅 `model_type == 'i2v'` 时执行。TI2V 不走。
- **第二份 DiT。** A14B 才按时间步在两份 `WanModel` 之间切换。TI2V 的图里始终是同一个 `self.model`。
- **静态图优化。** 没有把这张图捕获成 CUDA Graph 或编译图。每次前向都按 Python 调用顺序执行上述算子。
