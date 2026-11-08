# 推理侧量化没有进入本仓库

论文 4.4.3 在推理时把 DiT block 的 GEMM 做成 FP8，并把 FlashAttention 改成 Int8 与 FP8 的混合 8-bit。本仓库的 TI2V-5B 没有这两处。T2V-A14B 和 I2V-A14B 共用同一份 `WanModel` 和 `flash_attention`，也没有。它不是 5B 单独少做的一截。

## 论文在做什么

**FP8 GEMM。** 采样步里，权重按张量量化，激活按 token 量化，DiT block 中的 GEMM 全部使用 FP8。论文给出的效果是：FP8 GEMM 约为 BF16 GEMM 的 2 倍性能，DiT 模块约 1.13 倍加速，精度损失很小。

**8-bit FlashAttention。** 原生 FA3-FP8 把 Q、K、V 都做成 FP8（E4M3），论文观察到这会明显损伤视频质量。他们的改法有两条：

- **混合 8-bit**：`S = QK^T` 用 Int8，`O = PV` 用 FP8。
- **跨块 FP32 累加**：块内的 `PV` 仍用 FP8 WGMMA；跨块累加改到 CUDA core 的 FP32 寄存器，避免原生 FP8 WGMMA 的 14-bit 累加器在长序列上溢出。

## 本仓库实际怎么算

DiT 的 QKV、输出投影和 FFN 是 `wan/modules/model.py` 里的 `nn.Linear`。TI2V 的采样循环包在 `torch.amp.autocast('cuda', dtype=bfloat16)` 里，这些线性层按 bfloat16 做矩阵乘。打开 FSDP 时，`wan/distributed/fsdp.py` 的 `MixedPrecision` 也把参数设成 bfloat16。仓库中没有 `float8`、`e4m3`、按张量或按 token 的量化缩放，也没有量化 GEMM 内核。时间调制和若干残差被放进 fp32 autocast，方向与 FP8 GEMM 相反。

自注意力和交叉注意力进入 `wan/modules/attention.py` 的 `flash_attention`。允许的数据类型只有 fp16 和 bf16，默认是 `bfloat16`：

```python
half_dtypes = (torch.float16, torch.bfloat16)
assert dtype in half_dtypes
```

`WanSelfAttention` 与 `WanCrossAttention` 调用时没有改这个默认值。Q、K、V 在进 kernel 前被收成半精度。装有 FlashAttention 3 时走 `flash_attn_interface.flash_attn_varlen_func`，否则走 FlashAttention 2 的同名函数。两次调用传的都是这块 bf16 张量，没有 Int8 的 `QK^T`、FP8 的 `PV`，也没有跨块 FP32 累加。`q_scale` 只是可选的 query 乘法，TI2V 的两次调用都没有传它。

`generate.py` 没有 FP8、Int8 或量化相关的开关。

## 论文里的倍数不能直接套用

「FP8 GEMM 约为 BF16 的两倍」和「DiT 模块 1.13 倍」是论文自己那套量化内核上的结果。本仓库的采样循环没有接上那些内核。README 里提到的 FP8 量化来自外部的 DiffSynth-Studio，不是 `wan/textimage2video.py` 里的实现。
