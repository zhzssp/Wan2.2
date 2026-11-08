# Wan2.2 中的 Rectified Flow

本文说明当前仓库的推理代码如何使用 rectified flow。仓库只有推理，没有构造训练样本的插值，也没有速度回归损失。调用顺序以 TI2V-5B 的 `WanTI2V.t2v()` / `i2v()` 为准。T2V-A14B、I2V-A14B、S2V、Animate 使用同一对求解器，差别是 `sample_shift` 的数值。

Pipeline 里的逐步调用见 [TI2V-5B 推理范围与 Pipeline](ti2v_5b_inference.md)。

## 1. 推理时 rectified flow 在哪里

DiT 仍是带时间条件的 Transformer。Rectified flow 换掉的是概率路径和网络输出的含义：模型给出直线路径上的速度，调度器把这条 ODE 从噪声积到数据。

一次采样做三件事：

1. 用 `sample_shift` 生成 σ 网格。σ 就是 flow 的时间，从高噪声排到 0。
2. DiT 在当前 latent 和 σ 上预测速度 \(v\)。循环变量名仍是 `noise_pred`。
3. `scheduler.step` 沿 σ 减小的方向积分一步。

50 步循环的耗时几乎全在 DiT 求 \(v\)。和 flow 直接对应的公式只有时间网格、\(x_0=x_\sigma-\sigma v\)，以及求解器的一次更新。这些公式在 `wan/utils/fm_solvers_unipc.py` 与 `wan/utils/fm_solvers.py`，不在 `WanAttentionBlock` 里。

## 2. σ 就是时间

直线路径把数据 \(x_0\) 和标准高斯噪声 \(\epsilon\) 按系数混合：

\[
x_\sigma=(1-\sigma)\,x_0+\sigma\,\epsilon,\qquad \sigma\in[0,1]
\]

| σ | 路径上的点 | 推理中的角色 |
| --- | --- | --- |
| 1 | \(\epsilon\) | 起点。`t2v()` 里的初始 latent 就是 `torch.randn` |
| 0 | \(x_0\) | 终点。积分完成后把 latent 交给 VAE 解码 |

两个求解器把这对系数写成同一函数：

```python
def _sigma_to_alpha_sigma_t(self, sigma):
    return 1 - sigma, sigma
```

\(\alpha=1-\sigma\)，第二项系数就是 σ 本身。文件头写的是把原来的 diffusion ODE 求解器改成这条 flow ODE（`Convert unipc for flow matching`、`Convert dpm solver for flow matching`）。

传给 DiT 的整数时间步是同一个 σ，放大 1000 倍后取整：`timestep = σ × num_train_timesteps`，`num_train_timesteps = 1000`。正弦时间嵌入因此仍落在 \([0,1000]\)。ODE 使用 `[0,1]` 上的浮点 `sigmas`。`step()` 收到的整数时间步不参与积分，步号由求解器内部的 `step_index` 决定。

`sample_shift` 不另起一条时间轴。它把均匀时间 \(t\) 拧成网络和积分器实际使用的 σ：

\[
\sigma=\frac{s\,t}{1+(s-1)\,t}
\]

实现是 `get_sampling_sigmas()`，UniPC 的 `set_timesteps` 用同一公式。\(s>1\) 时步点更靠近高噪声。TI2V-5B 的 \(s=5\)；T2V-A14B 的 \(s=12\)。

论文里常见的记法把零点放在噪声上：令 \(\tau=1-\sigma\)，则 \(\tau=0\) 是噪声、\(\tau=1\) 是数据，路径上的点与上式相同。对 σ 的速度指向噪声，对 τ 的速度是它的相反数。从 σ = 1 积到 0，和从 τ = 0 积到 1，走的是同一段直线。本仓库沿用「时间越大噪声越多」，这样 UniPC / DPM-Solver 仍然从高噪声积到低噪声。

UniPC 在 `set_timesteps` 之前，把 σ 上界取成 \(1-1/1000\)，积分用的终点再接上 0（`final_sigmas_type="zero"`）。`dpm++` 的网格从 1 排到 0，去掉终点 0，再由求解器处理最后一步。

## 3. 速度，以及 \(x_0=x_\sigma-\sigma v\)

对 σ 求导，直线上的速度是常数：

\[
v=\frac{dx}{d\sigma}=\epsilon-x_0
\]

代回路径：

\[
x_\sigma=x_0+\sigma v
\]

解出数据端点：

\[
x_0=x_\sigma-\sigma v
\]

这就是 `prediction_type="flow_prediction"` 时 `convert_model_output` 的换算。TI2V 不修改 `predict_x0`，默认是 `True`，所以每步都走这一支：

```python
sigma_t = self.sigmas[self.step_index]
x0_pred = sample - sigma_t * model_output
```

`model_output` 是 DiT 预测的 \(v\)。得到的 \(x_0\) 是求解器需要的数据端点预测，采样不会一次跳到它。随后求解器只把 σ 减小一截。

在这组 \(\alpha,\sigma\) 下，UniPC / DPM-Solver++ 的一阶更新和直接对速度做欧拉步相同：

\[
x_{\sigma_{\mathrm{next}}}=x_\sigma+(\sigma_{\mathrm{next}}-\sigma)\,v
\]

\(\sigma_{\mathrm{next}}<\sigma\)，这一步逆着 \(v=\epsilon-x_0\) 走，朝数据端靠近。默认 `solver_order=2`，一阶欧拉之上还有多步校正。

## 4. TI2V 采样循环里的对应关系

`wan/textimage2video.py` 的 `t2v()`：

1. `torch.randn` 生成 fp32 噪声，作为 σ 高噪声端的 latent。
2. 默认构造 `FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)`，再 `set_timesteps(50, shift=5.0)`。`--sample_solver dpm++` 时改用 `FlowDPMSolverMultistepScheduler` 和 `get_sampling_sigmas`。
3. 每个整数时间步上，DiT 对正向 prompt 和负向 prompt 各前向一次。
4. CFG 做在速度上：

\[
v=v_{\mathrm{uncond}}+s\,(v_{\mathrm{cond}}-v_{\mathrm{uncond}})
\]

TI2V 的 \(s\) 是标量 `sample_guide_scale=5`。代码写成 `noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)`。

5. `sample_scheduler.step(noise_pred, t, latents)` 完成第 3 节的换算和积分，返回下一步 latent。

`i2v()` 的循环同构，多了第 0 帧的端点约束，见下一节。

## 5. 图生视频把第 0 帧放在 σ = 0

`i2v()` 用 `masks_like(..., zero=True)` 把第 0 个 latent 帧的 mask 置 0，然后每步都执行：

```python
latent = (1. - mask2[0]) * z[0] + mask2[0] * latent
```

`z` 是首帧图像的 VAE 编码。mask 为 0 的位置保持 \(z\)，其余位置继续积分。时间条件乘上同一张 mask：第 0 帧 token 的 timestep 是 0，其余 token 是当前时间步。

σ = 0 就是数据端点 \(x_0\)，所以首帧可以和仍在流动的帧放进同一次前向。TI2V 不用 `model_type == 'i2v'` 的通道拼接 `y`；图像条件写在 latent 和按 token 变化的时间步里。文生视频调用 `masks_like(..., zero=False)`，mask 全是 1，所有 token 共用一个 σ。

## 6. 代码落点

| 内容 | 位置 |
| --- | --- |
| σ 网格与 shift | `get_sampling_sigmas()`；`FlowUniPCMultistepScheduler.set_timesteps` |
| \(\alpha=1-\sigma\) | 两个求解器的 `_sigma_to_alpha_sigma_t` |
| \(v\rightarrow x_0\) | 两个求解器的 `convert_model_output`，`prediction_type="flow_prediction"` |
| 沿 σ 的积分 | `FlowUniPCMultistepScheduler.step`（默认）；`FlowDPMSolverMultistepScheduler.step` |
| 采样循环与 CFG | `wan/textimage2video.py` 的 `t2v()`、`i2v()` |
| 速度场 \(v_\theta(x,\sigma)\) | `wan/modules/model.py` 的 `WanModel.forward`。输出形状与输入 latent 相同 |
| TI2V 的 \(s=5\)、50 步 | `wan/configs/wan_ti2v_5B.py` 的 `sample_shift`、`sample_steps` |

`WanModel.forward` 用时间嵌入生成每层的调制向量。TI2V 传入的 `t` 已经按 token 展开；形状是 `[1, seq_len]`。层内的注意力、RoPE、FFN 只计算这个条件下的速度，不实现加噪或积分。

## 7. 仓库里没有的部分

权重满足的训练目标是：在 \(x_\sigma=(1-\sigma)x_0+\sigma\epsilon\) 上回归 \(v=\epsilon-x_0\)。本仓库没有采样 σ、混合 \(x_0\) 与 \(\epsilon\)、计算该损失的代码。推理只消费已经学好的 \(v_\theta\)，从噪声积到数据。
