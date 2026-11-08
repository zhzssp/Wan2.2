# Diffusion cache 没有进入本仓库的推理实现

论文 4.4.2 用跨采样步的相似性少算 DiT，并在 Wan 14B 文生视频上给出 1.62 倍加速。本仓库的 TI2V-5B、T2V-A14B、I2V-A14B 都没有这段优化。它不是 5B 小模型单独省掉的一截。

## 论文在做什么

论文观察到两件事：

- **注意力相似**：同一个 DiT block 在不同采样步的注意力输出很接近。
- **CFG 相似**：采样后段，条件 DiT 输出和无条件 DiT 输出很接近。

据此做两种缓存，并用验证集挑选哪些步可以缓存，以保持无损：

- **Attention cache**：注意力每隔几步真算一次，其余步复用缓存结果。
- **CFG cache**：无条件分支每隔几步才跑一次完整 DiT；其余步复用条件结果，再用类似 FasterCache 的残差补偿补回细节。

## 本仓库实际怎么跑

三条管道的去噪循环每一步都做两次完整前向，再按原公式合成速度。没有缓存步、缓存间隔、跨步保存的注意力输出，也没有残差补偿。`generate.py` 没有相关开关。

| 任务 | 管道 | 每步实际调用 |
| --- | --- | --- |
| `ti2v-5B` | `wan/textimage2video.py` 的 `t2v()` / `i2v()` | `self.model` 条件前向 + 无条件前向。文生视频约在 380–386 行，图生视频约在 580–589 行 |
| `t2v-A14B` | `wan/text2video.py` 的 `WanT2V` | 先按 boundary 选高噪声或低噪声专家，再对该专家做条件前向和无条件前向，约在 341–352 行 |
| `i2v-A14B` | `wan/image2video.py` 的 `WanI2V` | 同样先选专家，再两次前向，约在 388–402 行。多传的是条件张量 `y` |

A14B 的 `_prepare_model_for_timestep` 只决定这一步用哪一份专家，并在 offload 时把另一份搬回 CPU。它不复用上一步的 DiT 输出。

注意力缓存要是做了，会落在三套管道共用的 `wan/modules/model.py` 的 `WanAttentionBlock.forward`。该函数没有步号，也没有跨步缓存槽，每次都执行自注意力、交叉注意力和 FFN。换专家不会绕过它。默认 50 步因此仍是约 100 次完整 DiT 前向。

## 不要和这几处 cache 混在一起

- **VAE 的 `feat_cache`**：`wan/modules/vae2_1.py`、`vae2_2.py` 里因果卷积记住上一时间块。它只活在一次 VAE 前向内部。
- **`torch.cuda.empty_cache()`**：把分配器里的空闲显存还回去，不保存注意力或 CFG 的结果。
- **`guide_scale <= 1` 时只跑条件前向**：出现在 `wan/speech2video.py` 和 `wan/animate.py`。这是关掉 CFG，不是每隔几步复用并做残差补偿。
- **README 里的 Cache-dit**：外部项目，用 DBCache、TaylorSeer 和 Cache CFG 加速 Wan2.2 MoE。不是本仓库采样循环的一部分。

论文的 1.62 倍属于他们自己的 14B 文生视频推理系统。本仓库的 `t2v-A14B` 循环没有接上这段缓存，不能把这个倍数当成开源实现已经达到的加速。
