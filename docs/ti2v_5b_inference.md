# Wan2.2 TI2V-5B 推理范围与 Pipeline

本文划定当前仓库里 **TI2V-5B**（`Wan2.2-TI2V-5B`，任务名 `ti2v-5B`）的推理实现边界，并按真实调用顺序说明 pipeline。范围按「最小可运行的文生视频推理」来划：单次生成一条视频，不包含服务化、作业调度和 A14B / S2V / Animate。

对应课题里的第一步：先认清推理过程中的主要算子、计算、访存和并行特征，再谈 SME ARM 众核上的适配。本文描述的是 **GPU 参考实现**，不是 ARM 实现。

## 1. 范围结论

TI2V-5B 在本仓库中是一条 **单进程、单次请求** 的推理链路，由三个模型串起来：

1. **umT5-XXL 编码器**：每个请求跑 2 次（正向 prompt 与负向 prompt），得到文本条件。
2. **Wan DiT（约 5B，单份权重）**：去噪循环的主体。默认 50 步、classifier-free guidance，每步 2 次前向，共约 100 次 DiT 前向。
3. **Wan2.2 VAE 解码器**：去噪结束后在 rank 0 跑 1 次，把 latent 还原成像素视频。图生视频才会额外跑 1 次 VAE 编码。

和 A14B 的差别直接影响迁移范围：

- 只有 **一个** DiT。没有 `low_noise_model` / `high_noise_model` 两套专家，也没有按时间步切换模型。
- VAE 使用 `wan/modules/vae2_2.py` 的 `Wan2_2_VAE`，压缩步长是时间 4、空间 16（`vae_stride = (4, 16, 16)`），latent 通道 `z_dim = 48`。
- 权重用 `WanModel.from_pretrained(checkpoint_dir)` 从检查点根目录加载，不进子目录。

仓库里没有常驻服务、请求队列、Slurm 脚本，也没有 SME / ARM 后端。多卡路径（FSDP + Ulysses + NCCL）是可选加速，最小实现可以整段不走。

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
| 默认采样器 | `wan/utils/fm_solvers_unipc.py` | Flow-Matching UniPC，默认 `sample_solver=unipc` |
| 可选采样器 | `wan/utils/fm_solvers.py` | `dpm++`，与 UniPC 二选一 |
| 张量与写盘 | `wan/utils/utils.py` | `masks_like`、`best_output_size`、`save_video` |

导出入口在 `wan/__init__.py` 的 `WanTI2V`。

### 2.2 最小实现可以不碰的代码

这些文件能被同一条 `generate.py` 命令带上，但不在 TI2V 计算热路径上。

| 路径 | 原因 |
| --- | --- |
| `wan/distributed/fsdp.py` | 仅 `--dit_fsdp` / `--t5_fsdp` 时把参数切到多卡 |
| `wan/distributed/ulysses.py` | 仅 `--ulysses_size > 1` 时替换注意力 |
| `wan/distributed/sequence_parallel.py` | 同上，替换 DiT `forward`，按序列维切分 |
| `wan/distributed/util.py` | 进程组与 all-to-all |
| `wan/utils/prompt_extend.py` | `--use_prompt_extend` 才调用 Qwen 或 DashScope，发生在视频模型之前 |

`--ulysses_size` 必须等于 `WORLD_SIZE`，且 `num_heads = 24` 必须能被卡数整除。这是 GPU 多卡策略，不是 ARM 众核线程并行。

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

DiT 的 `in_dim`、`out_dim`、`text_dim` 写在权重自带的模型配置里，由 `from_pretrained` 读入。仓库 Python 配置没有覆盖这三项。VAE 侧通道数在代码里固定为 `z_dim = 48`，umT5 隐藏维在 `umt5_xxl()` 里固定为 4096。权重里的 DiT 配置必须与这两处对齐。

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
| `sample_shift` | 5.0 | flow matching 的 shift |
| `sample_guide_scale` | 5.0 | CFG 强度，标量，不是 A14B 那种双值 |
| `frame_num` | 121 | 默认帧数，须满足 `4n+1` |
| `sample_fps` | 24 | 写 mp4 的帧率，不参与网络计算 |
| `num_train_timesteps` | 1000 | 调度器训练步数刻度 |
| `window_size` | `(-1, -1)` | 自注意力是全局注意力 |

umT5-XXL encoder（`wan/modules/t5.py` 的 `umt5_xxl`，`encoder_only=True`）：词表 256384，隐藏维 4096，FFN 10240，64 head（head 维 64），24 层，相对位置桶 32。Decoder 不参与推理。

VAE（`Wan2_2_VAE`）：`z_dim=48`，编码器基宽 `c_dim=160`，解码器基宽在 `WanVAE_` 中为 `dec_dim=256`，`dim_mult=[1, 2, 4, 4]`，每级 2 个残差块，`attn_scales=[]`（这条 VAE 路径没有空间注意力），时间下采样标记 `[False, True, True]`。权重 dtype 默认 fp32。通道均值和标准差写在 `Wan2_2_VAE.__init__` 里，编码后做 `(mu - mean) * (1/std)`，解码前做逆变换。

## 4. 默认一次文生视频的张量规模

下面按默认文生视频计算：`1280*704`、121 帧、单卡（`sp_size = 1`）。`704*1280` 的 token 数相同。这是后面算延迟和内存时的基准形状。

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

一次 DiT 前向里，自注意力的序列长度就是这 27280。交叉注意力的 query 同样是 27280，key/value 来自文本，长度不超过 512。

采样循环：

```text
DiT 前向次数 = sample_steps × 2 = 50 × 2 = 100
每层都含 1 次全局自注意力 + 1 次交叉注意力 + 1 次 FFN
层数 = 30
```

因此热路径是 **100 × 30 层 DiT block**，不是 T5，也不是 VAE。T5 每个请求 2 次，VAE 解码每个请求 1 次。

图生视频的空间尺寸不直接用 `--size` 的宽高，而是用 `max_area = 704×1280`，再由 `best_output_size` 把宽高对齐到 32 的倍数（`patch_size` 的空间维 × `vae_stride` 的空间维 = 32），并尽量保持原图宽高比。帧数公式与文生视频相同。

## 5. Pipeline

### 5.1 总流程

```mermaid
flowchart TD
    A["generate.py<br/>task = ti2v-5B"] --> B["WanTI2V 构造<br/>加载 T5、VAE、一份 DiT"]
    B --> C{"是否传入图像"}
    C -->|否| D["t2v: 直接采样高斯噪声 latent"]
    C -->|是| E["i2v: 图像对齐、VAE 编码<br/>锁住第 0 帧 latent"]
    D --> F["T5 编码 prompt 与负向 prompt"]
    E --> F
    F --> G["UniPC 循环 50 步"]
    G --> H["DiT 条件前向"]
    G --> I["DiT 无条件前向"]
    H --> J["CFG 合成噪声预测"]
    I --> J
    J --> K["scheduler.step 更新 latent"]
    K --> G
    G --> L["rank 0: VAE.decode"]
    L --> M["save_video 写 mp4 后进程退出"]
```

`generate()` 只做分发：有 `img` 走 `i2v()`，否则走 `t2v()`。两条路径的去噪循环同构，差别在 latent 初始化和第 0 帧是否被条件钉死。

进程模型是一次性的。`generate.py` 在 rank 0 把返回的 `(C, T, H, W)`、值域 `[-1, 1]` 的张量写成 mp4，然后 `barrier`，若分布式已初始化则 `destroy_process_group`，进程结束。模型不会留到下一条请求。

### 5.2 启动与设备

`generate.py` 读取 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`。单进程时这三个变量缺省为 0、1、0，不初始化进程组，FSDP 与 Ulysses 会被直接拒绝。

多进程时：

```text
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
```

`WanTI2V` 把计算设备写成 `cuda:{device_id}`。未显式指定 `--offload_model` 时，单进程默认为 `True`（模块在 CPU/GPU 之间来回搬），多进程默认为 `False`（权重留在 GPU）。

单卡省显存的参考命令（README：约 24GB 显存）：

```bash
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "..."
```

`--t5_cpu` 让 T5 始终在 CPU 上编码，再把 embedding 拷到计算设备。`--convert_model_dtype` 只在 **不用 FSDP** 时把 DiT 参数转成 `param_dtype`（bfloat16）。

### 5.3 文本条件

每个请求编码两次：用户 prompt，以及负向 prompt。负向 prompt 为空时使用 `shared_config.py` 里的中文默认串。

`T5EncoderModel.__call__`：

1. tokenizer 清洗空白、截断或填充到最多 512。
2. `T5Encoder` 做 token embedding。`shared_pos=False`，相对位置偏差在每层内部计算。
3. 24 层 `T5SelfAttention`：LayerNorm、无缩放的 `einsum` 注意力、softmax 在 fp32、门控 GELU 的 FFN。
4. 按 mask 去掉 padding，返回变长 `List[Tensor]`，每个元素形状 `[L, 4096]`，`L ≤ 512`。

DiT 内部还会把这段向量线性投到 3072，并在长度维补零到 512。T5 的序列很短，计算量相对 DiT 可以忽略，但它是 **纯 PyTorch 的 einsum 注意力**，没有 FlashAttention。迁到 ARM 时这条路径比 DiT 注意力更容易先跑通，不能代表视频 backbone 的性能。

### 5.4 文生视频 `t2v()`

1. 由帧数和 `vae_stride` 算出 latent 形状 `[48, 31, 44, 80]`（默认 720p）。
2. `seq_len` 向上取整到 `sp_size` 的倍数。单卡就是 27280。
3. 在计算设备上用 `torch.randn` 生成 fp32 噪声，作为初始 latent。列表长度是 1，也就是 batch = 1。
4. 构造调度器。默认 UniPC：`FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)`，再 `set_timesteps(50, shift=5.0)`。
5. `masks_like(..., zero=False)` 得到全 1 掩码。文生视频不锁定任何帧。
6. 若开启 offload，此时才把 DiT `.to(device)`。
7. 对每个时间步 `t`：
   - 把标量 `t` 铺到空间下采样后的网格 `[:, ::2, ::2]` 上，再补到 `seq_len`。文生视频下每个 token 的时间条件相同。
   - 条件前向：`model(latents, t, context, seq_len)`。
   - 无条件前向：同一 latent，换成负向 context。
   - `noise_pred = uncond + guide_scale * (cond - uncond)`，`guide_scale = 5`。
   - `scheduler.step` 得到下一步 latent。
8. offload 时把 DiT 搬回 CPU，再在 rank 0 调用 `vae.decode`。

整个去噪包在 `autocast('cuda', dtype=bfloat16)` 和 `torch.no_grad()` 里。

### 5.5 图生视频 `i2v()`

在 `t2v` 的循环之外多了条件注入：

1. 按 `max_area` 计算输出宽高，放大后中心裁剪，像素归一化到 `[-1, 1]`，张量形状 `[3, 1, H, W]`（时间维长度为 1）。
2. `vae.encode` 得到第 0 帧的 latent `z`。
3. 采样噪声仍覆盖全部 `T × H × W`。`masks_like(..., zero=True)` 把 **第 0 个 latent 帧** 的 mask 置 0。
4. 每步更新后执行 `latent = (1 - mask) * z + mask * latent`，第 0 帧始终保持为编码图像，其余帧继续去噪。
5. 时间条件同样乘上这个 mask：第 0 帧 token 的 timestep 为 0，其余帧为当前 `t`。

最小实现建议先做 `t2v()`。`i2v()` 多一次 VAE 编码和逐帧 mask，算子集合与 `t2v` 相同。

### 5.6 DiT 一次前向

实现是 `wan/modules/model.py` 的 `WanModel.forward`。TI2V 不走 `model_type == 'i2v'` 的 `y` 通道拼接，图像条件已经写在 latent 和 timestep 里。

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
6. **unpatchify**：`einsum` 把 token 还原成 `[48, 31, 44, 80]` 的 fp32 latent。

RoPE 把 head 维 128 拆成三段频率，分别作用在时间、高、宽网格上（`rope_params` 的三段长度）。序列并行打开时，这段逻辑被换成 `wan/distributed/sequence_parallel.py` 里的 `sp_dit_forward` / `sp_attn_forward`：先按序列维切给各 rank，注意力内部用 all-to-all 把 head 维交换回来（Ulysses）。单卡最小实现不进入这两个函数。

`flash_attention()` 有两处和 ARM 直接冲突：

- `assert q.device.type == 'cuda'`
- 调用 FlashAttention 2 或 3 的 `flash_attn_varlen_func`

同文件里的 `attention()` 可以退到 `scaled_dot_product_attention`，但 `WanSelfAttention` 和 `WanCrossAttention` **没有调用它**。迁 ARM 时要替换的是 `model.py` 这两处对 `flash_attention` 的调用，或者让 `flash_attention` 在非 CUDA 设备走可移植实现。变长 batch 在 TI2V 里名存实亡：batch 恒为 1，真实长度就是 27280，padding 只为了对齐 `sp_size`。

### 5.7 VAE

`Wan2_2_VAE.encode` / `decode` 对列表里的每个视频单独跑，外面包一层 fp32 autocast。解码结果 `clamp` 到 `[-1, 1]`。

`WanVAE_.encode`：先把空间 `patchify` 成 2×2，再按时间块送进 `Encoder3d`。第 1 块 1 帧，之后每块 4 帧。`CausalConv3d` 用 `feat_cache` 保存上一块的时间上下文，所以块与块之间不能打乱顺序。

`WanVAE_.decode`：按 latent 的时间维 **逐帧** 调用 `Decoder3d`，帧间依赖同一份 cache，最后 `unpatchify`。默认 31 个 latent 帧就是 31 次串行解码调用。

`CausalConv3d` 把时间 padding 全部放在过去一侧，前向时如果有 cache 就拼到时间维前面再做普通 `Conv3d`。这是 VAE 里数量最多的算子，也是时间维上的串行点。

### 5.8 写视频

只有 rank 0 拿到像素张量。`save_video` 把 `[-1, 1]` 线性映到 0–255，按 `sample_fps = 24` 写成 mp4。文件名默认带任务名、分辨率、`ulysses_size`、prompt 前 50 个字符和时间戳。

## 6. 算子、访存与并行特征

给课题中「识别算子兼容性与性能瓶颈」用的对照。热度按默认 100 次 DiT 前向估计。

| 算子 | 出现位置 | 主要形状 | 重复次数 | 访存与并行 |
| --- | --- | --- | --- | --- |
| Linear / GEMM | DiT 的 QKV、输出投影、FFN、文本与时间 MLP；T5 的 QKV 与 FFN | DiT FFN 为 `27280 × 3072` 与 `27280 × 14336` | 每个 DiT 层 7 个左右 Linear，×30 ×100 | 计算密度最高。权重可复用，激活随序列长度线性涨。ARM 上首先是向量化 GEMM 和权重布局 |
| 全局自注意力 | DiT `WanSelfAttention` | Q/K/V `[1, 27280, 24, 128]` | 30 × 100 | 序列平方项。当前实现绑定 FlashAttention CUDA kernel。head 维 128，head 之间可并行；序列维是主并行维 |
| 交叉注意力 | DiT `WanCrossAttention` | query 27280，key/value ≤ 512，24 × 128 | 30 × 100 | 比自注意力短一截（乘的是文本长度）。仍然是 GEMM + softmax |
| RMSNorm / LayerNorm | DiT 每层 3 处，另加 Q/K RMSNorm；调制前后有 fp32 cast | 归约轴是 3072 | 与层数同量级 | 带宽敏感，归约在通道维。代码里先 `.float()` 再算，再 cast 回 bf16，会多一轮读写 |
| RoPE | DiT 自注意力的 Q/K | 按 `(31, 22, 40)` 网格广播 | 每层 2 次 × 100 | 逐元素复数乘，计算轻、索引不规则。序列并行时每个 rank 只拿自己的频率切片 |
| Conv3d | DiT patch 嵌入一次；VAE 里大量 `CausalConv3d` | DiT 卷积核 `(1,2,2)`，只在每次前向开头；VAE 卷积核更大且带时间 cache | DiT：100 次小卷积。VAE：每请求 1 次解码 | VAE 是卷积带宽瓶颈，并且时间块串行。DiT 的 patch 卷积不是瓶颈 |
| GELU / SiLU / softmax | FFN、时间 MLP、注意力 | 逐元素，或注意力最后一维 | 与对应 GEMM 同次 | 适合向量化，通常贴在 GEMM 后面融掉 |
| einsum 注意力 | 仅 T5 | 序列 ≤ 512，64 head，head 维 64 | 每请求 2 次 × 24 层 | 显式物化 `bnij` 分数矩阵。序列短，适合作为「无 FlashAttention」的正确性参照，不适合作为性能优化主对象 |
| 调度器 `step` | UniPC | 与 latent 同形状的逐元素更新 | 50 | 相对 DiT 可以忽略，但必须数值对齐，否则视频对不上 |
| 掩码与 latent 混合 | `i2v` 每步末尾 | `[48, 31, H, W]` | 50 | 逐元素，只在图生视频出现 |

并行层次，按当前代码实际存在的来分：

- **请求内模型并行（已实现，GPU）**：FSDP 切参数，Ulysses 切序列。绑定 NCCL，且 Ulysses 规模必须等于进程数。
- **批次并行**：没有。forward 接收的是长度为 1 的 list。
- **时间步并行**：没有。flow matching 是串行的 50 步，下一步输入依赖上一步输出。
- **VAE 时间并行**：没有。因果卷积 cache 强制按时间顺序扫过帧。
- **ARM 众核要新做的并行**：不在这个仓库里。对应的是把上面的 GEMM、注意力、Norm、Conv3d 拆到 SME 向量宽度和众核线程上，并重新安排 3072 / 14336 这些维的数据布局。现有的「8 卡 Ulysses」不能当成 ARM 线程方案。

dtype 混用会直接影响内存流量：DiT 与 T5 参数是 bf16，时间调制、若干残差、RMSNorm 统计、T5 softmax、VAE 全程偏 fp32，latent 噪声和调度器状态也是 fp32。统计内存开销时要把激活的 fp32 中间量算进去，不能只算参数字节数。

## 7. 迁到 SME ARM 时，当前实现里已经钉死的平台假设

这些是「现有推理框架难以直接在深圳超算上跑高效率」的具体落点，最小移植都要碰到：

| 假设 | 位置 |
| --- | --- |
| 设备是 `cuda:{id}` | `WanTI2V.__init__` |
| 自动混合精度设备类型写死 `'cuda'` | `t2v` / `i2v` 的 `torch.amp.autocast('cuda', ...)`，以及 `WanAttentionBlock` 里的 fp32 autocast |
| DiT 注意力要求 `q.device.type == 'cuda'`，并调用 FlashAttention | `wan/modules/attention.py` 的 `flash_attention` |
| 多卡后端是 NCCL | `generate.py` 的 `init_process_group` |
| 显存管理使用 `torch.cuda.empty_cache` / `synchronize` | `textimage2video.py` 的 offload 路径 |
| 权重加载是 PyTorch `state_dict` 与 `from_pretrained` | T5、VAE、DiT 三处 |

最小可运行目标可以收成：

1. 只保留单进程 `t2v()`，关闭 FSDP、Ulysses、prompt 扩展和 offload。
2. 用一份可移植注意力替换 `flash_attention`，先保证 121 帧、`1280*704`、固定 seed 下与 GPU 参考输出在允许误差内对齐。
3. 性能优化的主对象是 DiT 的 Linear 与长度为 27280 的自注意力；T5 和 VAE 解码作为正确性闭环保留，再单独看 VAE 因果卷积的串行是否成为尾部延迟。
4. 正确性、延迟、吞吐、内存、资源利用率都围绕这一次 `t2v()` 测量。吞吐在当前代码里没有批处理，只能理解为单位时间完成的请求数，或把步数、帧数作为缩放轴。

## 8. 建议的阅读顺序

1. `wan/configs/wan_ti2v_5B.py` 与 `wan/configs/shared_config.py`：把形状和默认步数固定下来。
2. `wan/textimage2video.py` 的 `t2v()`：整条 pipeline 的控制流都在这一个函数里。
3. `wan/modules/model.py` 的 `WanModel.forward` 与 `WanAttentionBlock`：热路径算子。
4. `wan/modules/attention.py`：CUDA 绑定点。
5. `wan/modules/t5.py` 的 `T5EncoderModel` 与 `T5Attention`：短序列参照实现。
6. `wan/modules/vae2_2.py` 的 `CausalConv3d`、`WanVAE_.decode`、`Wan2_2_VAE`：解码与时间串行。

## 9. 以 TI2V 为基座，和其他 Wan2.2 任务的复用边界

T2V-A14B、I2V-A14B、TI2V-5B 三条任务的推理骨架是同一套。迁移时先把 TI2V 的单进程 `t2v()` 做完，热路径算子可以留给另外两条；后面补的是管道分支和张量形状。S2V、Animate 不在这个范围内。

### 9.1 DiT 是同一个类

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

序列并行写在 `wan/distributed/sequence_parallel.py` 的 `sp_dit_forward`，替换的是同一份 `WanModel.forward`。TI2V 和 A14B 共用这个挂载点。

基座实现要把 `dim`、`num_heads`、`seq_len`、`in_dim` 做成参数。不要把 3072、24、48、27280 写进 kernel。A14B 在同样 720p 左右的序列更长：文生视频 81 帧、`1280×720`、空间压缩 8 时，token 数约为 `21 × 45 × 80 = 75600`，TI2V 默认是 27280。算法相同，序列长度不同。

### 9.2 用分支就能接上的差异

这些差异已经落在管道或 `WanModel.forward` 的入口，不需要第二套 block：

| 差异 | 现在的代码 | 接到基座上的方式 |
| --- | --- | --- |
| 一份 DiT 或两份专家 | TI2V：`self.model`。A14B：`low_noise_model` / `high_noise_model` | 按配置加载 1 份或 2 份；两份时用 `t` 与 `boundary` 选择 |
| CFG 系数 | TI2V 是标量 5.0。A14B 是 `(低噪声, 高噪声)`，例如 T2V 的 `(3.0, 4.0)` | `guide_scale` 为标量时两段共用，为元组时按 boundary 选取 |
| 图生视频条件 | TI2V 把图像写进第 0 帧 latent，该帧 timestep 置 0。I2V-A14B 把 mask 与 VAE latent 在通道维拼成 `y` | `model_type == 'i2v'` 时走已有的 `y` 拼接；`ti2v` 保持第 0 帧锁定 |
| 时间条件 | A14B 文生视频传入标量 `t`，`forward` 再广播到序列。TI2V 传入按 token 变化的 `t` | 时间嵌入模块不变，只改调用前如何构造 `t` |
| 多卡 | `--dit_fsdp`、`--t5_fsdp`、`--ulysses_size` | 已是开关。最小 ARM 实现先关闭 |

T5（`umt5_xxl` encoder）、UniPC / DPM++ 求解器、CFG 公式、FSDP 的 `shard_model`、`generate.py` 的进程组初始化，这几条任务已经是同一份代码。三条管道的采样循环是复制后的小改，不是三套模型实现。`text2video.py` 与 `textimage2video.py` 的 `t2v` 循环同构，A14B 多出来的就是 boundary 选模型和两档 `guide_scale`。

### 9.3 不能当成 `WanModel.forward` 里一个判断的部分

- **VAE。** TI2V 使用 `wan/modules/vae2_2.py` 的 `Wan2_2_VAE`，latent 通道 48，空间压缩 16。A14B 使用 `wan/modules/vae2_1.py` 的 `Wan2_1_VAE`，I2V 里噪声通道写死为 16，空间压缩 8。DiT block 可以复用，VAE 要单独迁。Patch 卷积的 `in_dim` 随权重变化，第一层 `Conv3d` 的权重形状不同，后面的 block 仍是同一个类。
- **S2V 与 Animate。** `WanModel_S2V`（`wan/modules/s2v/model_s2v.py`）和 `WanAnimateModel`（`wan/modules/animate/model_animate.py`）是另外的类。隐藏维同样是 5120、40 层、40 head，block 形态接近，但前向里有音频注入、motion、AdaIN 或角色动画分支。优化 TI2V 的 `WanAttentionBlock` 覆盖不了这两条任务。

因此顺序是：TI2V 单进程 `t2v()` 在 ARM 上跑通，算子接口保留形状参数；A14B 随后补双权重加载、时间步切换和 `vae2_1`；S2V、Animate 留到基座稳定之后。

## 10. 当前推理运行时不是 SGLang

仓库内没有 `sglang` 或 `SGLang` 的依赖、导入或启动命令。`requirements.txt` 与 `pyproject.toml` 的运行时依赖是 PyTorch、torchvision、diffusers、transformers、accelerate、flash_attn。没有 vLLM，也没有 SGLang 的服务进程。

实际运行方式是本地脚本：

```text
python generate.py --task ti2v-5B ...
torchrun --nproc_per_node=8 generate.py --task ti2v-5B ...
```

`generate.py` 构造 `WanTI2V`，调用一次 `generate()`，rank 0 写 mp4 后进程退出。没有请求调度、连续批处理、RadixAttention 或分页 KV cache。视频 DiT 的去噪循环在 `wan/textimage2video.py` 里手写，每步显式调用 `self.model(...)` 两次做 CFG。

diffusers 在这里是库，不是推理框架。`WanModel` 继承 `ModelMixin` 和 `ConfigMixin`，用来做配置注册和 `from_pretrained`。`wan/utils/fm_solvers_unipc.py`、`wan/utils/fm_solvers.py` 从 diffusers 的调度器改来，由管道自己 `step`。TI2V 不走 `DiffusionPipeline.__call__`。

和本课题相关的结论：要迁移的是这份 PyTorch 模块和手写采样循环，不是把 SGLang 后端从 GPU 换成 ARM。SGLang 面向大语言模型服务，本仓库的热路径是 flow-matching 视频 DiT，两者的运行时不是同一个东西。
