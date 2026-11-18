# Wan2.2 TI2V-5B 在深圳超算鲲鹏 920F 上的适配路线

把 TI2V-5B 的文生视频迁到深圳超算的鲲鹏 920F 上。上层服务用当前 `sglang` 仓库的扩散运行时 `python/sglang/multimodal_gen`，算子和内存从 `SGLang_920F` 的 `stable` 分支里挑已经写好、并且形状对得上的部分。官方 `generate.py` / `WanTI2V.t2v()` 仍是数值定义和 GPU 对照的来源，见 [ti2v_5b_inference.md](ti2v_5b_inference.md)。

920F 里已经跑通的是 GLM-5.3 的因果语言模型服务，不是扩散模型。`multimodal_gen` 在那份仓库里是旧快照，没有任何 `kunpeng` 标记。适配发生在当前 `sglang` 的 `multimodal_gen` 上，不把 Wan 注册进 `srt`。

每一步先用上一步的输出做对照。形状扣在下面这组默认值上。

| 项 | 值 |
| --- | --- |
| 服务 | `sglang` 的 `WanPipeline`，配置 `Wan2_2_TI2V_5B_Config` |
| 模型代码 | `runtime/models/dits/wanvideo.py` 的 `WanTransformerBlock` |
| 正式形状 | `1280*704`，121 帧，50 步 |
| latent | `[48, 31, 44, 80]`，`vae_stride = (4, 16, 16)` |
| DiT | 序列 **27280**，隐藏维 3072，24 head，head 维 128，30 层 |
| FFN | `3072 → 14336 → 3072`，激活是 GELU-tanh，不是 SiLU |
| 调试形状 | 5 帧、4 步。latent 时间维 2，token 数 `2 × 22 × 40 = 1760` |
| 热路径 | 50 步 × 条件与无条件 2 次前向 = **100** 次 DiT。UMT5 每请求 2 次。VAE 解码 1 次 |

```text
noise_pred = uncond + 5 * (cond - uncond)
```

两次前向权重相同，文本条件不同。调度器是 UniPC，对齐时不要换成 `dpm++`。

## 1. 做与不做

| 现在就做 | 单节点 DiT 已经快于 PyTorch 之后再说 | 不做 |
| --- | --- | --- |
| 单 rank 的 PyTorch CPU 短视频 | 多 rank 的序列并行 `all_to_all_4D` | 把 Wan 注册进 `srt`，或改 `ModelRunner`、KV cache、PD、MTP |
| DiT 的 BF16 GEMM 接到 `bf16_packed_gemm_kunpeng` | VAE 卷积的专用 kernel | 复用 MLA、DSA、MoE RDMA、`KunpengCommunicator` 里写死的 hidden 7168 |
| 双向注意力；调试形状可先试现有 flash attention | 计算图捕获 | `deployment/launch_all.sh` 的 512 rank PD 拓扑 |
| DiT 权重和去噪激活放进 HBM | FP8 / INT8 | 手写一套 SME 外积来替换已经能用的 KuTACC GEMM |

`sgl-model-gateway` 没有鲲鹏改动。TI2V 的进程由扩散服务自己拉起，不经过 `pd_router.py`。

## 2. SGLang_920F 里哪些能接到 TI2V

对照的是 `SGLang_920F` 的 `stable`。算子注册在 `sgl-kernel/csrc/cpu/cpu_kunpeng/kunpeng_torch_extend.cpp`，实现调用 KuTACC。

| 920F 里已有的 | 对 TI2V 意味着什么 | 接到哪一步 |
| --- | --- | --- |
| `bf16_gemm_pack_kunpeng`、`bf16_packed_gemm_kunpeng`（`matmul/bgemm.cpp`） | 权重 `[n, k]`、激活 `[m, k]` 的 BF16 GEMM。FFN 和 QKV 都是这种形状。`tile_k` 必须是偶数 | 第 2 步 |
| `bf16_gemm_prepack_kunpeng` | 权重在加载后打包一次，100 次前向复用 | 第 2 步 |
| `rmsnorm_kunpeng` | 可替换 DiT 的 RMSNorm。先和 PyTorch 逐元素对比 | 第 3 步 |
| `rope_kunpeng` | 吃的是一维 `position_ids` 加 `cos_sin_cache`，这是语言模型的 RoPE。Wan 的 RoPE 按网格 `(帧, 高, 宽)` 分成三段。不能直接替换 `rope_apply`，除非先造出与之逐位一致的 cache | 第 3 步，单独验证；对不上就留在 PyTorch |
| `flash_attention_kunpeng` / `flash_attention_with_workspace` | 有 `causal` 参数，双向时传 `false`。`flash_attention_with_workspace` 把 `cur_lens` 限制在 **2048**。1760 的调试形状可以试，27280 会在这个检查处失败 | 第 3 步只覆盖调试形状；正式形状要放宽长度或另写 |
| `hbw_allocator_kunpeng`（`memory/kunpeng_memory.cpp`） | 2MB 大页，`mbind` 到 `cpu_node + 16` | 第 4 步 |
| `silu_mul_quant_kunpeng`、`bf16_silu_pack_kunpeng` | Wan 的 FFN 是 GELU-tanh（`wanvideo.py` 的 `_WanGELUMLP`）。这两个 SiLU kernel 不能接上去 | 不接 |
| `KunpengCpuBackend`、`forward_mla_kunpeng.py` | MLA / DSA，头维度和 KV 布局是 DeepSeek/GLM 的 | 不接 |
| `token_dispatcher/kunpeng.py` | expert parallel 的 RDMA dispatch | 不接 |
| `KunpengCommunicator` | 共享内存池的做法可以参考。缓冲按 hidden 7168、`qk_head_dim` 192、`kv_lora_rank` 512 写死 | 第 6 步只复用建池，不复用这块缓冲 |
| `srt/graph/` | 要求热路径上每个会分配内存的算子都已注册。DiT 还在 PyTorch 时捕获会失败 | 第 3 步数值稳定之后再考虑，不单列成当前阶段 |

节点事实，来自 `scripts/cpu_kunpeng/glm52_simple/run.sh` 和 `server.sh`，第 1 步只用第一行，第 4 步用第二行：

| 事实 | 值 |
| --- | --- |
| 每节点计算 NUMA | 0–15，一个 rank 对应一个 |
| 片上 HBM NUMA | 16–31，等于计算 NUMA + 16 |
| 绑核 | `taskset -c (rank * 38 + 20)`，单 rank 实验用 rank 0，即 CPU 20 |
| 大页 | 2MB，只配置在 HBM 节点上。GLM 作业在对应 HBM 节点上申请约 2020–2048 页 |

## 3. 环境

第 1 步和第 2 步的环境不同。第 1 步不编译 `sgl-kernel`。

### 3.1 第 1 步：只跑 PyTorch

在一块计算节点上，不要在登录节点上加载面向计算节点编译的 `sgl_kernel`。

| 组件 | 版本或来源 | 用途 |
| --- | --- | --- |
| 操作系统 | 计算节点上的 aarch64 Linux。920F 参考说明写的是 openEuler 22.03 LTS-SP4 | 跑进程 |
| Python | 3.11。`stable` 的 `env.sh` 钉的是 Spack 里的 **3.11.15** | 与后面编译 `sgl-kernel` 的解释器一致 |
| PyTorch | 能在这块 aarch64 CPU 上做 BF16 `matmul` 和 `scaled_dot_product_attention` 的构建。第 2 步之前不要求 KUPL 补丁 | DiT、UMT5、VAE |
| 当前 `sglang` 源码 | `multimodal_gen` 能 import，`Wan2_2_TI2V_5B_Config` 可用 | 管线 |
| 权重 | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`，或与之同结构的本地目录 | 生成 |
| 作业内存 | 三份权重大约 DiT 20 GB、T5 11.4 GB、VAE 2.8 GB，再加解码激活。单卡 GPU 解码曾把一张 32GB 卡抬到 30.7 GiB。关闭 offload 时 DiT 在解码期间仍驻留 | `dsub` 的内存申请要盖住这一档 |

没有 GPU 时，`multimodal_gen/runtime/platforms/__init__.py` 会落到 `CpuPlatform`。920 没有 AMX，`cpu.py` 的 `get_attn_backend_cls_str()` 返回 `SDPABackend`，也就是 `torch.nn.functional.scaled_dot_product_attention`。线性层是 PyTorch 自己的 CPU GEMM。

### 3.2 第 2 步起：加上 920F 的编译栈

以 `SGLang_920F/env.sh` 为准。公开的 `docs/kunpeng/reference_implementation_cn.md` 是更早的 DeepSeek 说明（SGLang v0.5.11、Python 3.12、PyTorch v2.9.0），版本和 `stable` 不一致，不要两套混用。

| 组件 | `stable` 里写明的位置或版本 |
| --- | --- |
| HPCKit | `/home/share/fengguangnan/lvtx/HPCKit/HPCKit/26.1.RC1`，先 `source setvars.sh` |
| 毕昇编译器 | `$HPCKIT_ROOT/compiler/bisheng/bin/clang` 与 `clang++`，作为 `CC` / `CXX` |
| Python | Spack 前缀里的 **3.11.15** |
| KuPL | `$HPCKIT_ROOT/kupl/bisheng/release`。运行时 `KUPL_EXECUTOR_BACKEND=pthread`，`TORCH_USE_KUPL=1` |
| KML | `$HPCKIT_ROOT/kml/bisheng` |
| KuTACC | `third_party/kutacc` 或作业目录里编出来的 `install/{lib,include}`。`sgl-kernel` 链接 `libkutacc.so` |
| PyTorch | 与 `sgl-kernel` 同一次编译所用的那份。`env.sh` 从虚拟环境的 `site-packages/torch` 取 `Torch_DIR`。参考说明里的 v2.9.0 + `pytorch-v2.9.0-kupl.patch` 只在沿用那份旧说明时才成立 |
| `sgl-kernel` | 在计算节点上 `pip install -v . --no-build-isolation`，并已设置 `KUTACC_LIB`、`KUTACC_INCLUDE`。CMake 对 aarch64 使用 `-march=armv8.2-a+sve+fp16+bf16` |

第 2 步不需要 Mooncake，也不需要把 `SGLANG_USE_CPU_920F` 设为 1。那个开关会让 `srt` 的模型表只加载 DeepSeek 和 GLM，和扩散服务无关。

## 4. 第 1 步：单 rank 用 PyTorch 跑通

目的：确认管线在鲲鹏上能出视频。不编译鲲鹏算子，不进 HBM。

1. 申请一个计算节点，内存按第 3.1 节的权重大小加上解码激活。
2. 用第 3.1 节的 Python 和 PyTorch 装上当前 `sglang` 的 `multimodal_gen` 依赖。
3. `taskset -c 20` 启动一个进程，对应计算 NUMA 0。确认 `CpuPlatform` 被选中，日志里是 SDPA，不是 AMX，也不是 `kunpeng_cpu`。
4. 用调试形状生成：5 帧、4 步、`--base_seed 42`、UniPC、关闭序列并行（`ulysses-degree` 和 `ring-degree` 都是 1）。正向 prompt 与第 6 节的 GPU 参考相同。
5. 若某行调用了 `.cuda()` 或 FlashAttention，改成 CPU 上已有的 PyTorch 调用。这一处改动仍然不是鲲鹏 kernel。
6. 记录三段墙钟：UMT5 两次、DiT 一步（条件和无条件分开）、VAE 一次。先热身一步再计。

**完成判据：** 调试形状写出视频。同一 prompt 和种子下，与 GPU 参考的像素误差有一张表（最大绝对误差、平均绝对误差、PSNR）。阈值在这次测定，之后不放宽。GPU 那一次为了放进 31.36 GiB 打开了 offload 和 `--t5_cpu`，表上要注明。

## 5. 第 2 步：DiT 的矩阵乘换成 KuTACC

目的：QKV、输出投影和 FFN 走 `bf16_packed_gemm_kunpeng`。`WanTransformerBlock` 仍写 `self.to_q(x)`。

1. 按第 3.2 节编译 `sgl-kernel`，在计算节点上确认 `torch.ops.sgl_kernel.bf16_packed_gemm_kunpeng` 能被调用。
2. 在 `multimodal_gen` 的线性层方法上增加一条 BF16 路径，只在 CPU 且开关打开时进入。CUDA 融合（`fused_linear_gelu_tanh`）继续由 `_is_cuda` 挡住。
3. 权重在加载后调用 `bf16_gemm_pack_kunpeng` 或 `bf16_gemm_prepack_kunpeng` 一次。激活每次前向打包。打包时间单独记。
4. 覆盖自注意力的四个 `3072×3072` 投影、交叉注意力里达到同样规模的投影，以及 FFN 的 `3072×14336` 和 `14336×3072`。时间嵌入和文本嵌入的小矩阵留在 PyTorch。
5. FFN 的激活保持 GELU-tanh，紧跟在 GEMM 之后用 PyTorch 算。不要接 `silu_mul_quant_kunpeng`。
6. 用调试形状对比第 1 步的同一 Linear 输出，再跑通 4 步。然后在正式形状上只比较单层 GEMM 的时间和误差，不必先跑满 50 步。

**完成判据：** 同一 Linear 与第 1 步的最大绝对误差落在第 1 步记下的 dtype 地板内。调试形状 4 步仍写出视频。FFN 与 QKV 的合计时间低于第 1 步，打包单独列出。

## 6. 第 3 步：DiT 的注意力、RMSNorm 和 RoPE

目的：单 rank、双向注意力。序列并行仍然关闭。

1. 在 `runtime/platforms/cpu.py` 的 `get_attn_backend_cls_str()` 增加一个后端名字，指向新的 attention backend。`USPAttention` 在 world size 为 1 时不调用 `all_to_all_4D`。
2. 调试形状（1760 ≤ 2048）上，用 `flash_attention_with_workspace`，`causal=false`，scale 为 `1/sqrt(128)`。Q/K/V 布局按该 kernel 的 `[token, head, dim]` 准备，调用前后各转一次，先不要改 Wan 的存储。
3. 与第 1 步的 SDPA 输出比最大绝对误差。对不上就继续用 SDPA，不要为了接上 kernel 放宽误差。
4. 正式形状 27280 会触发 `MAX_SEQ_LEN_SUPPORTED = 2048`。这一步的任务是把限制提高到至少 27280，或确认 KuTACC 的 flash attention 另有不带这个上限的入口。在此之前，正式形状的注意力保持 SDPA。
5. `rmsnorm_kunpeng` 用 DiT 的权重和一行激活做逐元素对比，通过后再替换 block 里的 RMSNorm。
6. `rope_kunpeng` 先用 Wan 的 `(31, 22, 40)` 频率表造 `cos_sin_cache`，和 `rope_apply` 比。三段旋转对不上就保留 PyTorch 的 3D RoPE。

**完成判据：** 调试形状上，注意力输出与 SDPA 的误差在地板内，4 步视频仍能写出。正式形状要么用放宽后的 flash attention 跑通一层，要么明确写着仍用 SDPA。RMSNorm 和 RoPE 各自有一行对比结果。

## 7. 第 4 步：DiT 的权重和去噪激活放进 HBM

目的：热权重和去噪循环里反复使用的激活走 HBM。UMT5 和 VAE 先留在 DDR。

1. 把进程绑在计算 NUMA 0（CPU 20）。把 HBM 节点 16 的 2MB 大页配到不少于 GEMM workspace 加一份 DiT 激活的页数。GLM 作业用的量级是每个 HBM 节点约 2048 页，TI2V 先按 DiT 权重大小估算，不够再加。
2. 用 `hbw_allocator_kunpeng` 分配，确认返回指针所在 NUMA 是 16，而不是 0。分配失败时检查大页是否配在节点 16 上；计算节点上的大页应保持为 0。
3. DiT 的打包权重在第 2 步打包完成之后拷进这个池，100 次前向复用同一块。
4. 去噪循环内、每步都要重新分配的激活改从该池取。`torch.empty` 不会自动进入 HBM。UMT5 和 VAE 的张量不要放进这个池。
5. 重跑调试形状，和第 2 步的像素或 latent 对比。

**完成判据：** 权重和指定激活的指针落在 HBM NUMA。调试形状的误差不高于第 2 步。若墙钟没有下降，记下带宽计数或只保留放置正确，不把「放进 HBM」写成已经加速。

## 8. 第 5 步：按计时决定要不要动 VAE 和 UMT5

这两段在第 1 步就已经在 PyTorch 里跑。920F 没有覆盖 TI2V 这种因果 3D 卷积，也没有 UMT5 的专用 kernel。

1. 用第 1 步的三段计时。只有 VAE 或 UMT5 占端到端的比例高到值得单独做，才开这个步骤。
2. VAE 的时间维有 cache，31 帧串行。并行只发生在一帧内部的空间位置和输出通道。不要打乱帧序。
3. UMT5 每请求只跑两次，序列不超过 512。默认保持 PyTorch。

**完成判据：** 一张占比表，加上「做了帧内并行」或「保持 PyTorch」的决定。二者择一。

## 9. 第 6 步：多 rank

单节点的激活放不下，或正式形状一步的时间仍然由单进程的线程扩展限制，才做这一步。

1. 先在同一节点的 2 个 rank 上打开 Ulysses，`ring-degree` 保持 1。每个 rank 仍按 `rank * 38 + 20` 绑核，HBM 用各自的 `cpu_node + 16`。
2. `all_to_all_4D` 和 `all_reduce` 走节点内共享内存。建池可以参考 `KunpengCommunicator` 的 `shm_pool_create_kunpeng`，缓冲长度按本次的 token 数和 hidden **3072** 计算。
3. 不要调用 `KunpengCommunicator` 里为 MLA 准备的 all-to-all，也不要调用 MoE 的 RDMA dispatcher。
4. 与单 rank 的同一种子 latent 对比。通信前后的张量要能对上切分方式。

**完成判据：** 2 rank 的调试形状与单 rank 的误差在地板内。正式形状至少一步 DiT 的墙钟低于单 rank，并写明每个 rank 的 NUMA 和 HBM 节点。

## 10. 第 7 步：部署

DiT 路径已经稳定之后再写。入口是扩散服务，不是 `deployment/launch_all.sh`。

1. `dsub` 申请的节点数等于第 6 步确定的 rank 数；第 6 步没做就是 1 个节点。
2. 节点上配置 HBM 大页、加载这次编译的 `sgl-kernel` 和 KuTACC，按 rank 绑核后启动 `sglang serve` 的 TI2V 模型。
3. 健康检查通过后，用第 4 节的调试命令打一次请求，确认种子和开关与第 1 步相同。
4. 不启动 prefill/decode 两套进程，不加载 `pd_contract.py`。

**完成判据：** 一次作业从申请节点到写出调试视频可以重跑。正式形状的 121 帧、50 步至少有一条端到端记录，延迟拆成 UMT5、DiT 单步、100 次前向、VAE 和写视频。

## 11. 顺序

| 序号 | 步骤 | 先有什么 | 完成判据 |
| --- | --- | --- | --- |
| 1 | 第 1 步 | aarch64 PyTorch，不需要 KuTACC | 调试形状出视频，并有误差表 |
| 2 | 第 2 步 | 第 3.2 节的 `sgl-kernel` | BF16 GEMM 对齐且 FFN+QKV 更快 |
| 3 | 第 3 步 | 第 2 步的线性层仍可切回 PyTorch | 调试形状注意力对齐；正式形状有明确的 kernel 或仍用 SDPA |
| 4 | 第 4 步 | HBM 大页 | 指针在 HBM NUMA，误差不高于第 2 步 |
| 5 | 第 5 步 | 第 1 步的三段计时 | 占比表和做或不做的决定 |
| 6 | 第 6 步 | 单节点不够快或放不下 | 2 rank 对齐，正式形状一步更快 |
| 7 | 第 7 步 | 第 2–4 步可重复打开 | 作业能重跑，正式形状有一条拆开的延迟 |

计算图不在这张表里。它要求 DiT 前向里每个会分配内存的算子都已注册，条件在第 3 步正式形状的注意力也走鲲鹏 kernel 之后才成立。

## 12. GPU 参考（对照用，不是加速比的分母）

机器是 8 张 RTX 5090，每卡 31.36 GiB。单卡文生视频已经写出 mp4：`--offload_model True --convert_model_dtype --t5_cpu`，`--size 1280*704`，121 帧，50 步，`--sample_solver unipc`，`--base_seed 42`。正向 prompt 是 “Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage”。

| 区间 | 墙钟 |
| --- | --- |
| 启动到进入去噪 | 约 1 分钟 |
| 50 步 | 5 分 27 秒，平均 6.54 秒/步 |
| 进度条结束到开始写 mp4 | 约 39 秒，含把 DiT 搬回 CPU 和 VAE 解码 |
| 写 mp4 | 约 3 秒 |

去噪时 GPU 0 约 21.4 GiB。8 卡 Ulysses 那次是图生视频，没有写出 mp4，不能当作第二条像素参考。鲲鹏上的加速比相对第 1 步的 PyTorch CPU 跑法，不相对 6.54 秒。
