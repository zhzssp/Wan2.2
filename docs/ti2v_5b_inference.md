# Wan2.2 TI2V-5B 推理范围与 Pipeline

算子种类、调用次数和计算图的定义见 [TI2V-5B 的算子与计算图](ti2v_5b_operators.md)。多卡切分的通信细节见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。论文推理侧的二维上下文并行、diffusion cache、FP8 GEMM 和 8-bit FlashAttention 没有进入本仓库，相关文档放在 [docs/problems](problems/)。

## 1. 范围结论

TI2V-5B 在本仓库中是一条 **单进程/一组进程 + 单次请求** 的推理链路，由三个模型串起来：

1. **umT5-XXL 编码器**：每个请求跑 2 次（正向 prompt 与负向 prompt），得到文本条件。
2. **Wan DiT（约 5B，单份权重）**：去噪循环的主体。默认 50 步、classifier-free guidance，每步 2 次前向，共约 100 次 DiT 前向。
3. **Wan2.2 VAE 解码器**：去噪结束后在 rank 0 跑 1 次，把 latent 还原成像素视频。图生视频才会额外跑 1 次 VAE 编码。

和 A14B 的差别直接影响迁移范围：

- 只有 **一个** DiT。没有 `low_noise_model` / `high_noise_model` 两套专家，也没有按时间步切换模型。
- VAE 使用 `wan/modules/vae2_2.py` 的 `Wan2_2_VAE`，压缩步长是时间 4、空间 16（`vae_stride = (4, 16, 16)`），latent 通道 `z_dim = 48`。
- 权重用 `WanModel.from_pretrained(checkpoint_dir)` 从检查点根目录加载，不进子目录。

仓库里没有常驻服务、请求队列、Slurm 脚本，也没有 SME / ARM 后端。两条线共用同一套算子和同一段 50 步循环。真正随卡数改变的是权重放在哪、序列切不切：单卡用 CPU offload 与 bfloat16，8 卡用 FSDP（切参数）和一维 Ulysses（切序列）。精度停在 bf16 的 GEMM 和 FlashAttention；50 步里条件前向和无条件前向都不跳过。可选的 prompt 改写只换送进 umT5 的正向字符串，不改变这 100 次前向，见 [TI2V-5B 的 Prompt 改写](ti2v_prompt_alignment.md)。算子和已接入的优化汇总见第 5.7 节。

多卡两项默认都关闭，共用同一个 NCCL 进程组，作用在同一次生成上，不增加采样步，也不把 batch 拆开。最小实现可以整段不走。切分粒度见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。单卡这条线在 8×RTX 5090 上写出了 mp4，见第 6 节。8 卡去噪能在每卡十几 GiB 上跑完，rank 0 的 VAE 解码仍会把一张卡用满并 OOM，见第 7 节。

## 2. 仓库内的文件范围

### 2.1 最小推理必须经过的代码

| 角色 | 路径 | 在 TI2V 中做什么 |
| --- | --- | --- |
| 命令入口 | `generate.py` | 解析参数、按 `WORLD_SIZE` 决定是否初始化 NCCL、构造 `WanTI2V`、保存 mp4、退出 |
| 管道 | `wan/textimage2video.py` | `WanTI2V`：加载三个模型，实现 `t2v()` 与 `i2v()` |
| 任务配置 | `wan/configs/wan_ti2v_5B.py` | 分辨率步长、DiT 宽度、采样默认值 |
| 共享配置 | `wan/configs/shared_config.py` | T5 dtype、文本长度 512、DiT 参数 dtype、负向 prompt、训练步数 1000 |
| 任务注册 | `wan/configs/__init__.py` | `ti2v-5B` 注册项；合法分辨率只有 `1280*704` 与 `704*1280` |
| DiT | `wan/modules/model.py` | Patch 嵌入、30 层 Transformer、时间调制、RoPE、反 patch |
| 注意力封装 | `wan/modules/attention.py` | DiT 实际调用的是 `flash_attention()`，要求 CUDA |
| 文本编码器 | `wan/modules/t5.py` | umT5-XXL encoder-only，注意力用 `einsum`，不走 FlashAttention |
| 分词 | `wan/modules/tokenizers.py` | Hugging Face tokenizer 包装，序列长度截断到 512 |
| VAE | `wan/modules/vae2_2.py` | 3D 因果卷积 VAE，编码（仅 i2v）与解码 |
| 默认采样器 | `wan/utils/fm_solvers_unipc.py` | Rectified flow 的 UniPC 积分，默认 `sample_solver=unipc` |
| 可选采样器 | `wan/utils/fm_solvers.py` | 同一条 flow ODE 的 `dpm++`，与 UniPC 二选一 |
| 张量与写盘 | `wan/utils/utils.py` | `masks_like`、`best_output_size`、`save_video` |

导出入口在 `wan/__init__.py` 的 `WanTI2V`。两个调度器如何把 DiT 输出当成速度、沿 σ 从噪声积到数据，见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。

### 2.2 最小实现可以不碰的代码

这些文件能被同一条 `generate.py` 命令带上，但不在 TI2V 计算热路径上。

| 路径 | 原因 |
| --- | --- |
| `wan/distributed/fsdp.py` | 仅 `--dit_fsdp` / `--t5_fsdp` 时把参数切到多卡 |
| `wan/distributed/ulysses.py` | 仅 `--ulysses_size > 1` 时替换注意力 |
| `wan/distributed/sequence_parallel.py` | 同上，替换 DiT `forward`，按序列维切分 |
| `wan/distributed/util.py` | 进程组与 all-to-all |
| `wan/utils/prompt_extend.py`、`wan/utils/system_prompt.py` | `--use_prompt_extend` 才在采样前改写正向 prompt。见 [TI2V-5B 的 Prompt 改写](ti2v_prompt_alignment.md) |

这些文件在单卡最小路径上不会被调用。多卡时它们不改变第 5 节的采样循环：FSDP 在 `WanTI2V` 构造时按 block 切参数，Ulysses 在进循环前替换 DiT 的 `forward` 和每层自注意力。`--ulysses_size` 必须等于 `WORLD_SIZE`，且 `num_heads = 24` 必须能被卡数整除。这是 GPU 多卡策略，与 ARM 众核上要新做的线程并行是两件事。用法见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)，8 卡这条线见第 7 节。

### 2.3 不属于 TI2V-5B 的代码

做 5B 最小负载时不要把这些算进迁移范围：

- A14B 双专家管道：`wan/text2video.py`、`wan/image2video.py`，以及 `wan/configs/wan_t2v_A14B.py`、`wan/configs/wan_i2v_A14B.py`
- 旧 VAE：`wan/modules/vae2_1.py`（A14B 用，压缩是 4×8×8，通道与 2.2 VAE 不同）
- 语音与角色动画：`wan/speech2video.py`、`wan/animate.py`、`wan/modules/s2v/`、`wan/modules/animate/`
- `requirements_s2v.txt`、`requirements_animate.txt`

## 3. 权重与配置

权重不在仓库里。下载后 `--ckpt_dir` 指向该目录：

```bash
modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B
```

`WanTI2V.__init__` 按配置取下面三个文件（或目录）：

| 配置项 | 默认相对路径 | 加载方 |
| --- | --- | --- |
| `t5_checkpoint` | `models_t5_umt5-xxl-enc-bf16.pth` | `T5EncoderModel` |
| `t5_tokenizer` | `google/umt5-xxl` | `HuggingfaceTokenizer` |
| `vae_checkpoint` | `Wan2.2_VAE.pth` | `Wan2_2_VAE` |
| DiT | 检查点根目录（`from_pretrained(checkpoint_dir)`） | `WanModel` |

这三份权重大约是：DiT 的三片 safetensors 合计 20.0 GB，T5 的 `models_t5_umt5-xxl-enc-bf16.pth` 为 11.4 GB，VAE 的 `Wan2.2_VAE.pth` 为 2.8 GB。第 6 节和第 7 节的显存数字用的就是这一组文件。DiT 的 `in_dim`、`out_dim`、`text_dim` 写在权重自带的模型配置里，由 `from_pretrained` 读入。仓库 Python 配置没有覆盖这三项。VAE 侧通道数在代码里固定为 `z_dim = 48`，umT5 隐藏维在 `umt5_xxl()` 里固定为 4096。权重里的 DiT 配置必须与这两处对齐。

`wan/configs/wan_ti2v_5B.py` 与 `shared_config.py` 中、迁移时需要钉死的超参：

| 项 | 值 | 含义 |
| --- | --- | --- |
| `dim` | 3072 | DiT 隐藏维 |
| `ffn_dim` | 14336 | FFN 中间维 |
| `num_heads` | 24 | head 数，head 维 = 128 |
| `num_layers` | 30 | Transformer 层数 |
| `freq_dim` | 256 | 时间正弦编码维 |
| `patch_size` | `(1, 2, 2)` | 时间不切，空间 2×2 |
| `vae_stride` | `(4, 16, 16)` | latent 相对像素的 (T, H, W) 步长 |
| `text_len` | 512 | 文本 token 上限 |
| `param_dtype` / `t5_dtype` | bfloat16 | DiT 与 T5 计算 dtype；若干归一化与调制在 fp32 |
| `sample_steps` | 50 | 默认去噪步数 |
| `sample_shift` | 5.0 | 把均匀的 flow 时间拧向高噪声；公式见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md) |
| `sample_guide_scale` | 5.0 | CFG 强度，作用在速度 \(v\) 上；标量，不是 A14B 那种双值 |
| `frame_num` | 121 | 默认帧数，须满足 `4n+1` |
| `sample_fps` | 24 | 写 mp4 的帧率，不参与网络计算 |
| `num_train_timesteps` | 1000 | 把 σ∈[0,1] 放大成 DiT 的整数时间步 |
| `window_size` | `(-1, -1)` | 自注意力是全局注意力 |

umT5-XXL encoder（`wan/modules/t5.py` 的 `umt5_xxl`，`encoder_only=True`）：词表 256384，隐藏维 4096，FFN 10240，64 head（head 维 64），24 层，相对位置桶 32。Decoder 不参与推理。

VAE（`Wan2_2_VAE`）：`z_dim=48`，编码器基宽 `c_dim=160`，解码器基宽在 `WanVAE_` 中为 `dec_dim=256`，`dim_mult=[1, 2, 4, 4]`，每级 2 个残差块，`attn_scales=[]`（这条 VAE 路径没有空间注意力），时间下采样标记 `[False, True, True]`。权重 dtype 默认 fp32。通道均值和标准差写在 `Wan2_2_VAE.__init__` 里，编码后做 `(mu - mean) * (1/std)`，解码前做逆变换。

## 4. 默认一次文生视频的张量规模

下面按默认文生视频计算：`1280*704`、121 帧。`704*1280` 的 token 数相同。这是后面算延迟和内存时的基准形状，也是第 6 节单卡运行的形状。

像素：`3 × 121 × 704 × 1280`。

VAE latent：

```text
T = (121 - 1) / 4 + 1 = 31
H = 704 / 16 = 44
W = 1280 / 16 = 80
形状 = [48, 31, 44, 80]
```

DiT patch（`stride = (1, 2, 2)`）之后：

```text
token 数 = 31 × (44 / 2) × (80 / 2) = 31 × 22 × 40 = 27280
每 token 隐藏维 = 3072
```

一次 DiT 前向里，自注意力的序列长度就是这 27280。交叉注意力的 query 同样是 27280，key/value 来自文本，长度不超过 512。单卡的 `seq_len` 就是 27280。打开 Ulysses 后，这个 27280 会先补齐到进程数的倍数，再在前向内部按序列维切开。8 卡时每卡 3410 个 token，隐藏维仍是 3072；自注意力内部暂时拼回 27280，同时只保留 `24 / 8 = 3` 个 head。采样步数不因此改变。all-to-all 的形状见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。第 7 节那次 8 卡运行是图生视频，网格是 800×1088、每卡 3294 个 token，和这里的 3410 不是同一张网格。

采样循环：

```text
DiT 前向次数 = sample_steps × 2 = 50 × 2 = 100
每层都含 1 次全局自注意力 + 1 次交叉注意力 + 1 次 FFN
层数 = 30
```

因此热路径是 **100 × 30 层 DiT block**，不是 T5，也不是 VAE。T5 每个请求 2 次，VAE 解码每个请求 1 次。本仓库不跳过采样步，也不把 GEMM 或注意力降到 FP8 / Int8，所以上面的次数就是实际计算量。这些次数对应哪些算子、外层循环和 `WanModel.forward` 如何连成计算图，见 [TI2V-5B 的算子与计算图](ti2v_5b_operators.md)。

图生视频的空间尺寸不直接用 `--size` 的宽高，而是用 `max_area = 704×1280`，再由 `best_output_size` 把宽高对齐到 32 的倍数（`patch_size` 的空间维 × `vae_stride` 的空间维 = 32），并尽量保持原图宽高比。帧数公式与文生视频相同。

## 5. 两条链路共用的计算

这一节写卡数改变不了的部分：控制流、文本编码、采样公式、一次 DiT 前向、VAE、写盘，以及算子表。权重放在哪张卡、何时搬到 CPU、序列是否切开，分别在第 6 节和第 7 节。

### 5.1 控制流

```mermaid
flowchart TD
    A["generate.py<br/>task = ti2v-5B"] --> P{"打开 prompt 改写?"}
    P -->|否| B["WanTI2V 构造<br/>加载 T5、VAE、一份 DiT"]
    P -->|是| R["rank 0：Qwen 改写正向 prompt<br/>失败则退回原文"]
    R --> B
    B --> C{"是否传入图像"}
    C -->|否| D["t2v: 直接采样高斯噪声 latent"]
    C -->|是| E["i2v: 图像对齐、VAE 编码<br/>锁住第 0 帧 latent"]
    D --> F["T5 编码 prompt 与负向 prompt"]
    E --> F
    F --> G["UniPC 循环 50 步"]
    G --> H["DiT 条件前向"]
    G --> I["DiT 无条件前向"]
    H --> J["CFG 合成速度 v"]
    I --> J
    J --> K["scheduler.step 沿 σ 积分"]
    K --> G
    G --> L["rank 0: VAE.decode"]
    L --> M["save_video 写 mp4 后进程退出"]
```

`generate()` 只做分发：有 `img` 走 `i2v()`，否则走 `t2v()`。两条路径的去噪循环同构，差别在 latent 初始化和第 0 帧是否被条件钉死。循环是 rectified flow 采样：DiT 预测速度，调度器把 σ 从高噪声积到 0。路径、换算和积分见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。

默认不改写 prompt，节点 P 直接进入构造。打开 `--use_prompt_extend` 时，节点 R 只替换正向字符串，负向 prompt 和后面的 50 步不变。改写模型、系统提示词和与论文三条原则的差别见 [TI2V-5B 的 Prompt 改写](ti2v_prompt_alignment.md)。

卡数不给这张图增加节点。从 G 到 K 的 50 步、每步两次前向，单卡和 8 卡相同。单卡在节点 B 把整份权重量到一张卡上，并在去噪前后把 DiT 搬去 CPU。8 卡在节点 B 用 FSDP 切参数、用 Ulysses 替换节点 H 和 I 内部的 `forward`。

进程模型是一次性的。`generate.py` 在 rank 0 把返回的 `(C, T, H, W)`、值域 `[-1, 1]` 的张量写成 mp4，然后 `barrier`，若分布式已初始化则 `destroy_process_group`，进程结束。模型不会留到下一条请求。

### 5.2 文本条件

每个请求编码两次：正向 prompt，以及负向 prompt。负向 prompt 为空时使用 `shared_config.py` 里的中文默认串，不参与改写。正向 prompt 默认就是用户原文；只有打开 `--use_prompt_extend` 时，这里拿到的才是第 5.1 节节点 R 改写后的字符串。见 [TI2V-5B 的 Prompt 改写](ti2v_prompt_alignment.md)。

`T5EncoderModel.__call__`：

1. tokenizer 清洗空白、截断或填充到最多 512。
2. `T5Encoder` 做 token embedding。`shared_pos=False`，相对位置偏差在每层内部计算。
3. 24 层 `T5SelfAttention`：LayerNorm、无缩放的 `einsum` 注意力、softmax 在 fp32、门控 GELU 的 FFN。
4. 按 mask 去掉 padding，返回变长 `List[Tensor]`，每个元素形状 `[L, 4096]`，`L ≤ 512`。

DiT 内部还会把这段向量线性投到 3072，并在长度维补零到 512。T5 的序列很短，计算量相对 DiT 可以忽略，但它是 **纯 PyTorch 的 einsum 注意力**，没有 FlashAttention。迁到 ARM 时这条路径比 DiT 注意力更容易先跑通，不能代表视频 backbone 的性能。`--t5_cpu` 与 `--t5_fsdp` 只决定这 11.4 GB 权重放在 CPU 还是按层切开，两次编码的计算次数不变；这两个开关不要一起用。单卡用前者，8 卡用后者。

### 5.3 采样循环

文生视频 `t2v()`：

1. 由帧数和 `vae_stride` 算出 latent 形状 `[48, 31, 44, 80]`（默认 720p）。
2. `seq_len` 向上取整到 `sp_size` 的倍数。`sp_size` 在单卡是 1，打开 Ulysses 时等于进程数。单卡的 `seq_len` 就是 27280。
3. 在计算设备上用 `torch.randn` 生成 fp32 噪声，作为初始 latent。列表长度是 1，也就是 batch = 1。
4. 构造调度器。默认 UniPC：`FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)`，再 `set_timesteps(50, shift=5.0)`。
5. `masks_like(..., zero=False)` 得到全 1 掩码。文生视频不锁定任何帧。
6. 若开启 offload，此时才把 DiT `.to(device)`。单卡搬的是整份 DiT，8 卡搬的是已经切好的本卡分片。
7. 对每个时间步 `t`：
   - 把标量 `t` 铺到空间下采样后的网格 `[:, ::2, ::2]` 上，再补到 `seq_len`。文生视频下每个 token 的时间条件相同。
   - 条件前向：`model(latents, t, context, seq_len)`。
   - 无条件前向：同一 latent，换成负向 context。
   - `noise_pred = uncond + guide_scale * (cond - uncond)`，`guide_scale = 5`。这个张量是速度 \(v\)，CFG 做在速度上。
   - `scheduler.step` 用 \(x_0=x_\sigma-\sigma v\) 换成数据端点预测，再沿 σ 减小的方向积分一步。
8. offload 时把 DiT 搬回 CPU，再在 rank 0 调用 `vae.decode`。

初始 `torch.randn` 位于 σ 的高噪声端，σ = 0 是数据端点。传给 DiT 的整数 `t` 是 `σ × 1000`。默认 UniPC 的一阶更新与欧拉步 \(x_{\mathrm{next}}=x+(\sigma_{\mathrm{next}}-\sigma)\,v\) 相同，`solver_order=2` 在此之上做多步校正。注意力和 FFN 不包含这些公式。推导、shift，以及和论文里「从噪声积到数据」的时间方向如何对齐，见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。

整个去噪包在 `autocast('cuda', dtype=bfloat16)` 和 `torch.no_grad()` 里。模型是 FSDP 时，外面再套一层 `no_sync()`，用来跳过梯度同步；这里没有反传，数值不变。

循环不缓存。每一步都执行条件前向和无条件前向，没有隔几步跳过无条件分支，也没有把某一层的注意力输出留到下一步复用。论文 4.4.2 的 diffusion cache（attention cache 与 CFG cache，以及 FasterCache 式残差补偿）不在这条循环里，见 [Diffusion cache 没有进入本仓库的推理实现](problems/diffusion_cache.md)。

图生视频 `i2v()` 在上面的循环之外多了条件注入：

1. 按 `max_area` 计算输出宽高，放大后中心裁剪，像素归一化到 `[-1, 1]`，张量形状 `[3, 1, H, W]`（时间维长度为 1）。
2. `vae.encode` 得到第 0 帧的 latent `z`。多卡时每个 rank 都编码同一张图，这一步不随 FSDP 或 Ulysses 拆开。
3. 采样噪声仍覆盖全部 `T × H × W`。`masks_like(..., zero=True)` 把 **第 0 个 latent 帧** 的 mask 置 0。
4. 每步更新后执行 `latent = (1 - mask) * z + mask * latent`，第 0 帧始终保持为编码图像，其余帧继续去噪。
5. 时间条件同样乘上这个 mask：第 0 帧 token 的 timestep 为 0，其余帧为当前 `t`。timestep 0 就是 flow 的数据端点 σ = 0，所以这一帧保持 VAE 编码，其余帧继续积分。见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。

最小实现建议先做 `t2v()`。`i2v()` 多一次 VAE 编码和逐帧 mask，算子集合与 `t2v` 相同。第 6 节的单卡运行走 `t2v()`，第 7 节的 8 卡运行走 `i2v()`。

### 5.4 DiT 一次前向

实现是 `wan/modules/model.py` 的 `WanModel.forward`。TI2V 不走 `model_type == 'i2v'` 的 `y` 通道拼接，图像条件已经写在 latent 和 timestep 里。这一次前向里每个子模块执行几次，见 [TI2V-5B 的算子与计算图](ti2v_5b_operators.md) 第 3.1 节。下面的形状按第 4 节的默认文生视频。

1. **Patch 嵌入**：`Conv3d(in_dim → 3072, kernel=stride=(1,2,2))`。输出展成 `[1, 27280, 3072]`，不足 `seq_len` 的位置补 0。同时记下网格 `(F, H, W) = (31, 22, 40)`。
2. **时间嵌入**：正弦编码到 256 维，两层 MLP（Linear + SiLU + Linear）得到 3072 维，再投影成 6 组调制向量。这一段强制 fp32。调制是 **按 token** 的，形状 `[1, seq_len, 6, 3072]`。
3. **文本嵌入**：`Linear(4096 → 3072) + GELU(tanh) + Linear(3072 → 3072)`，序列补到 512。
4. **30 层 `WanAttentionBlock`**，每层：
   - 用 6 组调制做自适应缩放和平移（fp32）。
   - 自注意力：Q/K/V 三个 `Linear(3072, 3072)`，Q/K 做 RMSNorm，3D RoPE，然后 `flash_attention`，再输出投影。
   - 交叉注意力：query 来自视频 token，key/value 来自文本 token，无 RoPE。
   - FFN：`Linear(3072 → 14336)`、`GELU(approximate='tanh')`、`Linear(14336 → 3072)`。
   - 残差相加也在 fp32 autocast 中完成。
5. **Head**：同样的调制 + `Linear`，输出每个 token 的 `out_dim × 1 × 2 × 2` 个通道。
6. **unpatchify**：`einsum` 把 token 还原成 `[48, 31, 44, 80]` 的 fp32 张量。形状与 latent 相同，数值是当前 σ 上的速度 \(v\)。调度器如何把它积回数据见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。

RoPE 把 head 维 128 拆成三段频率，分别作用在时间、高、宽网格上（`rope_params` 的三段长度）。这六步是 `WanModel.forward` 的顺序。单卡按完整序列走完。只开 FSDP 时激活形状不变，变的是每层权重在进入该层时 all-gather 成完整矩阵。打开 Ulysses 后，第 1–3 步先得到完整序列，从第 4 步起每个 rank 只持有自己的一段；自注意力改走 `sp_attn_forward`，交叉注意力和 FFN 留在这段局部序列上。单卡 RoPE 用 `model.py` 的 `rope_apply`；序列并行时改用 `sequence_parallel.py`，每个 rank 取自己那一段频率。8 卡这条线上的替换见第 7 节。通信和包裹粒度见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。

`flash_attention()` 有两处和 ARM 直接冲突：

- `assert q.device.type == 'cuda'`
- 调用 FlashAttention 2 或 3 的 `flash_attn_varlen_func`

同文件里的 `attention()` 可以退到 `scaled_dot_product_attention`，但 `WanSelfAttention` 和 `WanCrossAttention` **没有调用它**。迁 ARM 时要替换的是 `model.py` 这两处对 `flash_attention` 的调用，或者让 `flash_attention` 在非 CUDA 设备走可移植实现。变长 batch 在 TI2V 里名存实亡：batch 恒为 1，默认文生视频的真实长度就是 27280，padding 只为了对齐 `sp_size`。

精度停在半精度，没有论文 4.4.3 的量化。`flash_attention` 只接受 fp16 和 bf16，默认 bf16；Q、K、V 在进 kernel 前被收成这个 dtype。装了 FlashAttention 3 就走 `flash_attn_interface.flash_attn_varlen_func`，否则走 FlashAttention 2 的同名函数，传下去的都是这块半精度张量。block 里的 `nn.Linear` 在外层 bf16 autocast 下做 GEMM。没有权重按张量、激活按 token 的 FP8 GEMM，也没有 Int8 的 `QK^T`、FP8 的 `PV` 和跨块 FP32 累加。对照见 [推理侧量化没有进入本仓库](problems/quantization.md)。

### 5.5 VAE

`Wan2_2_VAE.encode` / `decode` 对列表里的每个视频单独跑，外面包一层 fp32 autocast。解码结果 `clamp` 到 `[-1, 1]`。

`WanVAE_.encode`：先把空间 `patchify` 成 2×2，再按时间块送进 `Encoder3d`。第 1 块 1 帧，之后每块 4 帧。`CausalConv3d` 用 `feat_cache` 保存上一块的时间上下文，所以块与块之间不能打乱顺序。

`WanVAE_.decode`：按 latent 的时间维 **逐帧** 调用 `Decoder3d`，帧间依赖同一份 cache，最后 `unpatchify`。默认 31 个 latent 帧就是 31 次串行解码调用。

`CausalConv3d` 把时间 padding 全部放在过去一侧，前向时如果有 cache 就拼到时间维前面再做普通 `Conv3d`。这是 VAE 里数量最多的算子，也是时间维上的串行点。解码器中部还有一块单头空间注意力，用的是 `scaled_dot_product_attention`，与 DiT 的 FlashAttention 不是同一次调用。算子清单见 [TI2V-5B 的算子与计算图](ti2v_5b_operators.md) 第 3.3 节。多卡不把这 31 次解码拆开：decode 只在 rank 0 上按时间顺序跑完。单卡在解码前已经把整份 DiT 搬回 CPU，见第 6 节。8 卡上这一步把 GPU 0 推到容量上限并 OOM，见第 7 节。

### 5.6 写视频

只有 rank 0 拿到像素张量。`save_video` 把 `[-1, 1]` 线性映到 0–255，按 `sample_fps = 24` 写成 mp4。文件名默认带任务名、分辨率、`ulysses_size`、prompt 前 50 个字符和时间戳。

### 5.7 算子、访存，以及两条链路共用的优化

给课题中「识别算子兼容性与性能瓶颈」用的对照。算子从哪段 `forward` 来、各执行多少次，见 [TI2V-5B 的算子与计算图](ti2v_5b_operators.md)。热度按默认 100 次 DiT 前向估计，形状按第 4 节的文生视频。Ulysses 打开后，自注意力和 RoPE 的序列维按该节的 8 卡例子切开；这次 8 卡运行的实际网格见第 7 节。

| 算子 | 出现位置 | 主要形状 | 重复次数 | 访存与并行 |
| --- | --- | --- | --- | --- |
| Linear / GEMM | DiT 的 QKV、输出投影、FFN、文本与时间 MLP；T5 的 QKV 与 FFN | DiT FFN 为 `27280 × 3072` 与 `27280 × 14336`，bf16 | 每层 10 个 Linear，×30 ×100；另加每次前向的时间嵌入 3、文本嵌入 2、Head 1。见 [算子与计算图](ti2v_5b_operators.md) | 计算密度最高。当前是 bf16 的 `nn.Linear`，不是论文的 FP8 GEMM |
| 全局自注意力 | DiT `WanSelfAttention` | Q/K/V `[1, 27280, 24, 128]`，bf16 | 30 × 100 | 序列平方项。绑定 FlashAttention 2 或 3，输入收成 bf16，不是混合 8-bit。Ulysses 打开后的切分见第 7 节 |
| 交叉注意力 | DiT `WanCrossAttention` | query 27280，key/value ≤ 512，24 × 128 | 30 × 100 | 比自注意力短一截（乘的是文本长度）。仍然是 GEMM + softmax |
| RMSNorm / LayerNorm | DiT 每层 3 处，另加 Q/K RMSNorm；调制前后有 fp32 cast | 归约轴是 3072 | 与层数同量级 | 带宽敏感，归约在通道维。代码里先 `.float()` 再算，再 cast 回 bf16，会多一轮读写 |
| RoPE | DiT 自注意力的 Q/K | 按 `(31, 22, 40)` 网格广播 | 每层 2 次 × 100 | 逐元素复数乘，计算轻、索引不规则。序列并行时每个 rank 只拿自己的频率切片，见第 7 节 |
| Conv3d | DiT patch 嵌入一次；VAE 里大量 `CausalConv3d` | DiT 卷积核 `(1,2,2)`，只在每次前向开头；VAE 卷积核更大且带时间 cache | DiT：100 次小卷积。VAE：每请求 1 次解码 | VAE 是卷积带宽瓶颈，并且时间块串行。8 卡时这 31 次解码只在 rank 0，显存峰值见第 7 节。DiT 的 patch 卷积不是瓶颈 |
| GELU / SiLU / softmax | FFN、时间 MLP、注意力 | 逐元素，或注意力最后一维 | 与对应 GEMM 同次 | 适合向量化，通常贴在 GEMM 后面融掉 |
| einsum 注意力 | 仅 T5 | 序列 ≤ 512，64 head，head 维 64 | 每请求 2 次 × 24 层 | 显式物化 `bnij` 分数矩阵。序列短，适合作为「无 FlashAttention」的正确性参照，不适合作为性能优化主对象 |
| 调度器 `step` | UniPC | 与 latent 同形状的逐元素更新 | 50 | 相对 DiT 可以忽略，但必须数值对齐，否则视频对不上。更新是 rectified flow 的 ODE 一步，见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md) |
| 掩码与 latent 混合 | `i2v` 每步末尾 | `[48, 31, H, W]` | 50 | 逐元素，只在图生视频出现 |

并行层次按切分对象收成清单。包裹策略、通信次数和开关组合见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。8 卡上这些开关实际占了多少显存，见第 7 节。

- **FSDP（可选）**：切 DiT 与 T5 的参数。激活、隐藏维和 VAE 保持完整。只开它时，各卡重复计算同一条完整序列。
- **Ulysses（可选）**：切 DiT 激活的序列维，自注意力里再换成一部分 head。规模等于进程数，并与 FSDP 共用一个 NCCL 进程组。
- **批次并行**：forward 接收的是长度为 1 的 list。各 rank 协作这一条视频，没有第二条样本。
- **时间步并行**：rectified flow 沿 σ 串行积分 50 步，下一步输入依赖上一步输出。见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。
- **VAE 时间并行**：因果卷积 cache 按时间顺序扫过帧，decode 只在 rank 0。
- **论文推理侧的二维上下文并行、diffusion cache、FP8 / 8-bit 注意力**：本仓库都没有接上。对照见下面的表和 [docs/problems](problems/)。
- **ARM 众核要新做的并行**：把上面的 GEMM、注意力、Norm、Conv3d 拆到 SME 向量宽度和众核线程上，并重新安排 3072 / 14336 这些维的数据布局。现有的 8 卡 Ulysses 是 GPU 进程组上的切分，不能当成 ARM 线程方案。

dtype 混用会直接影响内存流量：DiT 与 T5 参数是 bf16，时间调制、若干残差、RMSNorm 统计、T5 softmax、VAE 全程偏 fp32，latent 噪声和调度器状态也是 fp32。统计内存开销时要把激活的 fp32 中间量算进去，不能只算参数字节数。

按第 5.1–5.6 节的调用顺序，两条链路都能用到的手段如下。Prompt 改写只换正向文本；后面几行才改变显存或内核。论文推理侧另外三项没有接进 `WanTI2V`，所以默认仍是 100 次 DiT 前向、每次 30 层全量 block。offload、FSDP、Ulysses 的实测分别在第 6 节和第 7 节。

| 手段 | 出现位置 | 实际作用 |
| --- | --- | --- |
| Prompt 改写 | `generate.py`，`--use_prompt_extend`，默认关，在构造 `WanTI2V` 之前 | 只替换正向 prompt。不减少 100 次前向。见 [TI2V-5B 的 Prompt 改写](ti2v_prompt_alignment.md) |
| `--offload_model`、`--t5_cpu`、`--convert_model_dtype` | 去噪前后搬权重。单进程默认 offload；多进程默认不 offload | 省显存，不减少前向次数。单卡能卸整份 DiT，8 卡只卸本卡分片 |
| bfloat16 autocast | `t2v()` / `i2v()` 包住整个去噪循环 | DiT 的 `nn.Linear` 按 bf16 做 GEMM。调制、若干残差和 VAE 仍偏 fp32 |
| FlashAttention 2 或 3 | 每层自注意力和交叉注意力，`wan/modules/attention.py` | Q、K、V 收成 bf16 后调用现成的 `flash_attn_varlen_func` |
| FSDP | 构造期，`--dit_fsdp` / `--t5_fsdp`，默认关 | 按 block 切参数，进层时 all-gather。不切序列，不切隐藏维 |
| 一维 Ulysses | 构造期替换 `forward`，`--ulysses_size` 必须等于进程数，默认关 | 序列切开，自注意力里 all-to-all 换成一部分 head。没有外层 Ring |

论文里写过、本路径没有接上的优化：

| 论文中的手段 | 本仓库 |
| --- | --- |
| 二维上下文并行：外层 RingAttention，内层 Ulysses；14B 上接近线性加速 | 只有内层 Ulysses。见 [推理侧二维上下文并行没有进入本仓库](problems/inference_parallelism.md) |
| Diffusion cache：隔几步复用注意力输出，后段复用条件结果并做残差补偿；14B 文生视频约 1.62 倍 | 每步两次完整前向，每层注意力都重算。见 [Diffusion cache 没有进入本仓库的推理实现](problems/diffusion_cache.md) |
| FP8 GEMM，以及 Int8 的 `QK^T`、FP8 的 `PV`、跨块 FP32 累加 | GEMM 和注意力都是 bf16。见 [推理侧量化没有进入本仓库](problems/quantization.md) |

VAE 的 `feat_cache` 只在一次解码内部记住上一时间块，不是 diffusion cache。`torch.cuda.empty_cache()` 只归还分配器里的空闲显存，不保证下一次申请能拿到连续块。第 7 节的解码 OOM 里，仍有约 6 GiB 留在这块缓存中。

## 6. 单卡 pipeline

这一节是 `WORLD_SIZE = 1` 的整条链路。计算步骤就是第 5 节；这里写权重放在哪、何时搬到 CPU，以及在 8×RTX 5090 上实际跑完的那一次文生视频。

### 6.1 进程与设备

`generate.py` 读取 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`。单进程时这三个变量缺省为 0、1、0，不初始化进程组。FSDP 与 Ulysses 在这条线上会被直接拒绝。`WanTI2V` 的计算设备是 `cuda:0`。未显式指定 `--offload_model` 时，单进程默认为 `True`。

`--t5_cpu` 让 T5 始终在 CPU 上编码，再把 embedding 拷到 GPU。`--convert_model_dtype` 把 DiT 参数转成 `param_dtype`（bfloat16）；这个开关只在不用 FSDP 时生效。上面两个开关加上 offload，就是 README 里约 24GB 显存的参考命令：

```bash
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "..."
```

这条命令不建进程组。三个模型都不分片：

| 模块 | 文件大约 | 放在哪 |
| --- | --- | --- |
| T5 | 11.4 GB | 全程 CPU。GPU 上只有编码得到的 `[L, 4096]` |
| VAE | 2.8 GB，fp32 | 构造时就在 `cuda:0`，一份完整权重。解码前不卸走 |
| DiT | 20.0 GB，转成 bf16 | 去噪循环开始前才 `.to(cuda:0)`。循环内每步两次前向之间只 `empty_cache()`，不搬回 CPU。50 步结束后 `self.model.cpu()`，再 `empty_cache()`，然后 `vae.decode` |

### 6.2 从构造到写盘

实测按第 6.1 节的命令跑默认文生视频：`1280*704`、121 帧、50 步、seed 42，不传图像。机器是 8 张 RTX 5090，只用 GPU 0；PyTorch 报告该卡容量 31.36 GiB。形状就是第 4 节：latent `[48, 31, 44, 80]`，`seq_len = 27280`，`sp_size = 1`。

1. 构造 `WanTI2V`。T5 留在 CPU，VAE 和随后要上卡的 DiT 都是完整一份。
2. 没有图像，走 `t2v()`。在 GPU 上采样 fp32 噪声，掩码全 1，不锁帧。
3. T5 在 CPU 上编码正向和负向 prompt，向量拷到 GPU。
4. 循环开始前把整份 DiT 搬上 GPU 0。50 步、每步条件前向加无条件前向，CFG 和 UniPC 按第 5.3 节。循环内 DiT 留在卡上。
5. 50 步结束后把整份 DiT 搬回 CPU，再 `vae.decode`。31 个 latent 帧在同一张卡上串行解码。
6. rank 0 按 24 fps 写 mp4，进程退出。

去噪阶段 nvidia-smi 看到 GPU 0 约 21.4 GiB、利用率 100%，每步约 6.5 秒。整次约 7 分钟，写出 1280×704、24 fps、121 帧、时长约 5.0 秒的 mp4。加载 DiT 时有一条提示，说应保持 float32 的模块列表为空；生成没有因此中断。

### 6.3 这条线的性质和问题

热路径是一张卡上的整份 DiT：100 次前向、每次 30 层、序列 27280。T5 不占这张卡的显存。去噪时测到的约 21.4 GiB 是「整份 DiT + 完整 VAE + 这一步的激活」。

解码能完成，是因为第 5 步之前整份约 20 GB 的 DiT 已经离开 GPU，卡上主要剩下 VAE 和它的激活。这次没有在解码瞬间采样显存，21.4 GiB 只覆盖去噪阶段，不能当成解码峰值。float32 模块列表为空的那条提示没有中断生成。

## 7. 八卡 pipeline

这一节是 `torchrun --nproc_per_node=8` 的整条链路，对应 README 里同时打开 `--dit_fsdp --t5_fsdp --ulysses_size 8` 的用法。计算步骤仍是第 5 节。这里写八张卡各自放什么、序列怎么切、何时搬到 CPU，以及在同一台 8×RTX 5090 上的两次图生视频。

### 7.1 进程与分片

多进程时：

```text
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
```

`WanTI2V` 的计算设备是 `cuda:{local_rank}`。未显式指定 `--offload_model` 时，多进程默认为 `False`，权重留在 GPU。Ulysses 的规模必须等于进程数，并且 24 个 head 能被这个数整除。FSDP 与 Ulysses 共用 `generate.py` 建好的那一个 NCCL 进程组，没有再拆出第二个组。进程组里还有三件与切分配套、本身不切张量的事：广播 seed，使各卡切的是同一条序列；prompt 扩展只在 rank 0 做完再广播；`vae.decode` 和写 mp4 只在 rank 0。

构造 `WanTI2V` 时：

- `--t5_fsdp`：T5 权重加载完就交给 `wan/distributed/fsdp.py` 的 `shard_model`，按 24 层 `T5SelfAttention` 切开。后面的两次编码仍是每个请求各跑一遍，参数按 rank 分片，进层时再 all-gather。
- `--dit_fsdp`：DiT `from_pretrained` 之后，在 `_configure_model` 里按 30 个 `WanAttentionBlock` 切开。patch 嵌入、时间嵌入、文本嵌入和 head 留在根上的 FSDP 单元。根单元本身也是 `FULL_SHARD`，这些参数按 rank 切开，不是每卡一份完整拷贝。
- `--ulysses_size 8`：同一处先把 `WanModel.forward` 换成 `sp_dit_forward`，把每层自注意力换成 `sp_attn_forward`，再包 FSDP。采样循环看见的仍是 `self.model(...)`。

`shard_model` 使用 `ShardingStrategy.FULL_SHARD`，混合精度的参数 dtype 是 bf16、reduce dtype 是 fp32。进某一层时把这一层的参数 all-gather 成完整矩阵，算完再收回分片。Ulysses 只切 DiT 激活的序列维，不切这三份权重，也不切 VAE。这是一维 Ulysses，不是论文推理侧的外层 RingAttention、内层 Ulysses。包裹策略、每次前向的通信和四种合法组合见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。和论文的差异见 [推理侧二维上下文并行没有进入本仓库](problems/inference_parallelism.md)。

八张卡上的权重：

| 模块 | 整份大约 | 每张卡 |
| --- | --- | --- |
| T5 | 11.4 GB | 24 层按 rank 切开。编码前 `text_encoder.model.to(device)`。`offload_model` 为真时，两次编码结束后 `.cpu()`；为假时分片留在卡上 |
| DiT | 20.0 GB | 30 个 block 和根上的嵌入、head 都按 rank 切开，每卡大约八分之一。进层时 all-gather。`offload_model` 为真时，50 步结束后各 rank `model.cpu()` |
| VAE | 2.8 GB，fp32 | 不进 FSDP。每张卡各放一份完整权重，直接在 `cuda:{rank}`。解码只使用 rank 0 那一份 |

按第 4 节的默认文生视频网格，8 卡是每卡 3410 个 token，自注意力时每卡 3 个 head。隐藏维 3072 保持完整。序列要到第 5.4 节那一次 DiT 前向里才切开；构造阶段只决定 `forward` 换成哪一个函数、权重按哪一层包进 FSDP。

### 7.2 从构造到写盘

实测命令是 README 的 8 卡图生视频，图像是 `examples/i2v_input.JPG`（832×1104），`--size 1280*704` 在这里是 `max_area`：

```bash
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --image examples/i2v_input.JPG \
  --prompt "..."
```

`best_output_size` 得到 800×1088。latent 时间维仍是 31，空间为 68×50，token 数 26350，补齐到 26352 后每卡 3294。下面的显存是这张网格的，不能直接当成 1280×704 文生视频的解码峰值。跑了两次：一次沿用多进程默认的 `offload_model=False`，一次显式加上 `--offload_model True`。seed 都是 42。

1. 八个进程各自加载。T5 和 DiT 按第 7.1 节切开，VAE 每卡一份。加载完、尚未去噪时，每卡约 8.3 GiB。这个数对得上分片后的 DiT、分片后的 T5 和完整 VAE，再加上 CUDA 上下文，不是 20 GB 的整份 DiT。
2. 每张卡各自 `vae.encode` 同一张图，得到第 0 帧 latent，再锁住这一帧。这一步不随 FSDP 或 Ulysses 拆开。
3. T5 分片先放到本卡上，编码正向和负向 prompt。offload 为真时，编码后八张卡都把 T5 `.cpu()`。offload 为假时，分片留在卡上。
4. 去噪 50 步。每步仍是条件前向加无条件前向。Ulysses 把 26352 个 token 切成每卡 3294；自注意力里 all-to-all，每卡 3 个 head。FSDP 在进每一层时 all-gather。每步约 1.6 秒，相对单卡文生视频的约 6.5 秒一步。50 步全部走完。offload 为真时，去噪平台是每卡 13.5 GiB，这里面没有整份 T5。offload 为假时，去噪中途每卡约 15.9 GiB。
5. offload 为真时，八张卡一起 `model.cpu()` 再 `empty_cache()`，从 13.5 GiB 降到 10.7 GiB。少掉的约 2.8 GiB 与本卡 DiT 分片同量级（整份约 20 GB 的八分之一），其中也含 `empty_cache()` 归还的空闲块。offload 为假时这一步被跳过，本卡的 DiT 分片和 T5 分片都还在 GPU 上。
6. 只有 rank 0 调用 `vae.decode`，按时间顺序跑 31 次 `Decoder3d`。另外七张卡停在这一步之后的 `barrier`，不参与解码。两次运行都在这一步失败，没有写出 mp4。

### 7.3 两次运行的显存

未写 `--offload_model` 时，日志是 `offload_model is not specified, set to False`。rank 0 进入 `vae.decode` 后再申请 2.51 GiB 失败：空闲 231 MiB，进程占用 31.12 GiB，其中 PyTorch 已分配 23.88 GiB，另有 6.37 GiB 在缓存里未分配。其余 rank 收到 SIGTERM。

加上 `--offload_model True` 后，每 2 秒读一次 nvidia-smi，八张卡的占用（GiB）是：

| 时刻 | GPU 0 | GPU 1–7 |
| --- | --- | --- |
| 权重加载完、尚未去噪 | 8.3 | 8.3 |
| 50 步去噪 | 13.5 | 13.5 |
| 各 rank 执行 `model.cpu()` 之后 | 10.7 | 10.7 |
| rank 0 进入 `vae.decode` | 29.0，随后 30.7 并 OOM | 停在 10.7 |

解码只发生在 GPU 0：两秒内从 10.7 GiB 升到 29.0 GiB，再到 30.7 GiB，然后仍要 2.51 GiB。此时空闲 697 MiB，已分配 23.96 GiB，缓存 5.83 GiB。offload 把可用空闲从 231 MiB 提高到 697 MiB，离这一次 2.51 GiB 的连续申请还差一截。失败栈落在解码器的 `CausalConv3d` 时间 padding 上。

### 7.4 这条线的性质和问题

去噪在八张卡上是完整的：50 步都走完，每步大约 1.6 秒，通信是进层时的 all-gather 和自注意力里的 all-to-all。这些通信在 `vae.decode` 之前已经结束。解码本身不再做卡间通信，另外七张卡停在约 10.7 GiB 等 `barrier`。

问题出在解码，而且 offload 盖不住。`model.cpu()` 和 `empty_cache()` 只在 `offload_model` 为真时执行；多进程默认跳过。打开之后，八张卡各自卸走的是本卡 DiT 分片，大约 2.8 GiB。单卡那次卸走的是整份约 20 GB 的 DiT，两边不是同一笔账。rank 0 上仍留着一份完整 VAE，以及没有被 Ulysses 切开的解码激活。31 次串行 `Decoder3d` 把 GPU 0 从 10.7 GiB 推到 30.7 GiB，下一次时间 padding 仍要 2.51 GiB。缓存里还有约 6 GiB，拼不成这块连续显存。两次运行都没有 mp4。

## 8. 迁到 SME ARM 时，当前实现里已经钉死的平台假设

这些是「现有推理框架难以直接在深圳超算上跑高效率」的具体落点，最小移植都要碰到：

| 假设 | 位置 |
| --- | --- |
| 设备是 `cuda:{id}` | `WanTI2V.__init__` |
| 自动混合精度设备类型写死 `'cuda'` | `t2v` / `i2v` 的 `torch.amp.autocast('cuda', ...)`，以及 `WanAttentionBlock` 里的 fp32 autocast |
| DiT 注意力要求 `q.device.type == 'cuda'`，并调用 FlashAttention | `wan/modules/attention.py` 的 `flash_attention` |
| 多卡后端是 NCCL；FSDP 与 Ulysses 共用这一组 | `generate.py` 的 `init_process_group`，见第 7 节 |
| 显存管理使用 `torch.cuda.empty_cache` / `synchronize` | `textimage2video.py` 的 offload 路径 |
| 权重加载是 PyTorch `state_dict` 与 `from_pretrained` | T5、VAE、DiT 三处 |

最小可运行目标可以收成：

1. 只保留单进程 `t2v()`，关闭 FSDP、Ulysses、prompt 扩展和 offload。
2. 用一份可移植注意力替换 `flash_attention`，先保证 121 帧、`1280*704`、固定 seed 下与 GPU 参考输出在允许误差内对齐。
3. 性能优化的主对象是 DiT 的 Linear 与长度为 27280 的自注意力；T5 和 VAE 解码作为正确性闭环保留，再单独看 VAE 因果卷积的串行是否成为尾部延迟。
4. 正确性、延迟、吞吐、内存、资源利用率都围绕这一次 `t2v()` 测量。吞吐在当前代码里没有批处理，只能理解为单位时间完成的请求数，或把步数、帧数作为缩放轴。

## 9. 建议的阅读顺序

1. `wan/configs/wan_ti2v_5B.py` 与 `wan/configs/shared_config.py`：把形状和默认步数固定下来。
2. 本文第 5 节，对照 `wan/textimage2video.py` 的 `t2v()`：整条控制流。其中的 σ、速度和积分见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。
3. `wan/modules/model.py` 的 `WanModel.forward` 与 `WanAttentionBlock`：热路径算子。算子次数和计算图见 [TI2V-5B 的算子与计算图](ti2v_5b_operators.md)。
4. 本文第 5.7 节：这条链路用上了哪些优化，论文里的哪几项没有接上。Prompt 改写见 [TI2V-5B 的 Prompt 改写](ti2v_prompt_alignment.md)。没接上的三项在 [docs/problems](problems/)。
5. 单卡：本文第 6 节。
6. 8 卡：本文第 7 节，以及 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。对照 `WanTI2V._configure_model`、`wan/distributed/fsdp.py` 和 `wan/distributed/sequence_parallel.py`。
7. `wan/modules/attention.py`：CUDA 绑定点。精度停在 bf16。
8. `wan/modules/t5.py` 的 `T5EncoderModel` 与 `T5Attention`：短序列参照实现。
9. `wan/modules/vae2_2.py` 的 `CausalConv3d`、`WanVAE_.decode`、`Wan2_2_VAE`：解码与时间串行。

## 10. 以 TI2V 为基座，和其他 Wan2.2 任务的复用边界

T2V-A14B、I2V-A14B、TI2V-5B 三条任务的推理骨架是同一套。迁移时先把 TI2V 的单进程 `t2v()` 做完，热路径算子可以留给另外两条；后面补的是管道分支和张量形状。S2V、Animate 不在这个范围内。

### 10.1 DiT 是同一个类

`wan/textimage2video.py`、`wan/text2video.py`、`wan/image2video.py` 都引用 `wan/modules/model.py` 的 `WanModel`。block 只有一份实现：`WanSelfAttention`、`WanCrossAttention`、`WanAttentionBlock`、`Head`，以及 `WanModel.forward`。

每一层的计算图相同：

- `Conv3d` patch 嵌入，`patch_size = (1, 2, 2)`
- 时间正弦编码（`freq_dim = 256`）加 6 组调制
- 文本从 4096 投到隐藏维
- 自注意力（Q/K RMSNorm、3D RoPE、全局注意力）
- 交叉注意力
- FFN：Linear、`GELU(approximate='tanh')`、Linear
- 调制和残差在 fp32 中完成

A14B 的 MoE 不在 block 内部。`WanT2V` / `WanI2V` 加载两份相同的 `WanModel`（`low_noise_model` 与 `high_noise_model`），`_prepare_model_for_timestep` 按 `t >= boundary` 选择一份。TI2V 只 `from_pretrained` 一次，没有这条切换。

宽度由配置决定，算子种类不变。head 维两边都是 128，注意力 kernel 的头部形状可以对齐。

| | TI2V-5B | T2V/I2V-A14B 的每一份专家 |
| --- | --- | --- |
| 隐藏维 | 3072 | 5120 |
| FFN | 14336 | 13824 |
| head 数 | 24 | 40 |
| head 维 | 128 | 128 |
| 层数 | 30 | 40 |
| 份数 | 1 | 2，按时间步切换 |

序列并行写在 `wan/distributed/sequence_parallel.py` 的 `sp_dit_forward`，替换的是同一份 `WanModel.forward`。TI2V 和 A14B 共用这个挂载点。切分方式见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)；在 TI2V 调用顺序里的位置是第 7 节。

基座实现要把 `dim`、`num_heads`、`seq_len`、`in_dim` 做成参数。不要把 3072、24、48、27280 写进 kernel。A14B 在同样 720p 左右的序列更长：文生视频 81 帧、`1280×720`、空间压缩 8 时，token 数约为 `21 × 45 × 80 = 75600`，TI2V 默认是 27280。算法相同，序列长度不同。

### 10.2 用分支就能接上的差异

这些差异已经落在管道或 `WanModel.forward` 的入口，不需要第二套 block：

| 差异 | 现在的代码 | 接到基座上的方式 |
| --- | --- | --- |
| 一份 DiT 或两份专家 | TI2V：`self.model`。A14B：`low_noise_model` / `high_noise_model` | 按配置加载 1 份或 2 份；两份时用 `t` 与 `boundary` 选择 |
| CFG 系数 | TI2V 是标量 5.0。A14B 是 `(低噪声, 高噪声)`，例如 T2V 的 `(3.0, 4.0)` | `guide_scale` 为标量时两段共用，为元组时按 boundary 选取 |
| 图生视频条件 | TI2V 把图像写进第 0 帧 latent，该帧 timestep 置 0。I2V-A14B 把 mask 与 VAE latent 在通道维拼成 `y` | `model_type == 'i2v'` 时走已有的 `y` 拼接；`ti2v` 保持第 0 帧锁定 |
| 时间条件 | A14B 文生视频传入标量 `t`，`forward` 再广播到序列。TI2V 传入按 token 变化的 `t` | 时间嵌入模块不变，只改调用前如何构造 `t` |
| 多卡 | `--dit_fsdp`、`--t5_fsdp`、`--ulysses_size` | 三条任务同一套开关。切分见 [TI2V-5B 推理侧并行方式](ti2v_5b_parallelism.md)。最小 ARM 实现先关闭 |

T5（`umt5_xxl` encoder）、UniPC / DPM++ 求解器、CFG 公式、FSDP 的 `shard_model`、`generate.py` 的进程组初始化，这几条任务已经是同一份代码。求解器对三条任务是同一套 rectified flow 积分，见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。三条管道的采样循环是复制后的小改，不是三套模型实现。`text2video.py` 与 `textimage2video.py` 的 `t2v` 循环同构，A14B 多出来的就是 boundary 选模型和两档 `guide_scale`。A14B 同样每步做完整的条件前向和无条件前向，注意力同样走 bf16 的 `flash_attention`。第 5.7 节里没接上的 diffusion cache、FP8 GEMM 和二维上下文并行，在 T2V-A14B、I2V-A14B 上也不存在。

### 10.3 不能当成 `WanModel.forward` 里一个判断的部分

- **VAE。** TI2V 使用 `wan/modules/vae2_2.py` 的 `Wan2_2_VAE`，latent 通道 48，空间压缩 16。A14B 使用 `wan/modules/vae2_1.py` 的 `Wan2_1_VAE`，I2V 里噪声通道写死为 16，空间压缩 8。DiT block 可以复用，VAE 要单独迁。Patch 卷积的 `in_dim` 随权重变化，第一层 `Conv3d` 的权重形状不同，后面的 block 仍是同一个类。
- **S2V 与 Animate。** `WanModel_S2V`（`wan/modules/s2v/model_s2v.py`）和 `WanAnimateModel`（`wan/modules/animate/model_animate.py`）是另外的类。隐藏维同样是 5120、40 层、40 head，block 形态接近，但前向里有音频注入、motion、AdaIN 或角色动画分支。优化 TI2V 的 `WanAttentionBlock` 覆盖不了这两条任务。

因此顺序是：TI2V 单进程 `t2v()` 在 ARM 上跑通，算子接口保留形状参数；A14B 随后补双权重加载、时间步切换和 `vae2_1`；S2V、Animate 留到基座稳定之后。

## 11. 当前推理运行时不是 SGLang

仓库内没有 `sglang` 或 `SGLang` 的依赖、导入或启动命令。`requirements.txt` 与 `pyproject.toml` 的运行时依赖是 PyTorch、torchvision、diffusers、transformers、accelerate、flash_attn。没有 vLLM，也没有 SGLang 的服务进程。

实际运行方式是本地脚本：

```text
python generate.py --task ti2v-5B ...
torchrun --nproc_per_node=8 generate.py --task ti2v-5B ...
```

`generate.py` 构造 `WanTI2V`，调用一次 `generate()`，rank 0 写 mp4 后进程退出。没有请求调度、连续批处理、RadixAttention 或分页 KV cache。视频 DiT 的去噪循环在 `wan/textimage2video.py` 里手写，每步显式调用 `self.model(...)` 两次做 CFG。

diffusers 在这里是库，不是推理框架。`WanModel` 继承 `ModelMixin` 和 `ConfigMixin`，用来做配置注册和 `from_pretrained`。`wan/utils/fm_solvers_unipc.py`、`wan/utils/fm_solvers.py` 从 diffusers 的调度器改来，由管道自己 `step`。TI2V 不走 `DiffusionPipeline.__call__`。

和本课题相关的结论：要迁移的是这份 PyTorch 模块和手写采样循环，不是把 SGLang 后端从 GPU 换成 ARM。SGLang 面向大语言模型服务，本仓库的热路径是 rectified flow 视频 DiT，两者的运行时不是同一个东西。flow 的路径和积分见 [Wan2.2 中的 Rectified Flow](wan22_rectified_flow.md)。
