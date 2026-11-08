# 推理侧二维上下文并行没有进入本仓库

论文用上下文并行缩短**一条**视频的多卡延迟，并用 FSDP 给 Wan 14B 这类大模型切参数。上下文并行写明与训练相同，是二维的：外层 RingAttention，内层 Ulysses。二者合在一起，论文在 Fig. 12 给出 14B 上接近线性的加速。

本仓库没有这套二维上下文并行。TI2V-5B、T2V-A14B、I2V-A14B 共用同一套实现：可选的 FSDP 切参数，加上一维 Ulysses。不是 5B 单独少做了外层 Ring。

## 论文推理侧写了什么

- **上下文并行**：卡数增加时，用来降低生成单条视频的延迟。策略与训练阶段相同，外层是 RingAttention，内层是 Ulysses。
- **模型切分**：点名对象是 Wan 14B 的显存。推理序列很长，FSDP 的通信量小于张量并行，并且能和计算重叠，因此切分方式与训练一致，采用 FSDP。
- **加速**：二维上下文并行加上 FSDP 之后，DiT 在 Wan 14B 上接近线性加速。

训练描述里的「FSDP 组大于上下文并行组、再用数据并行扩大 batch」不在这段推理文字里。这里要对齐的是单条视频上的二维上下文并行，以及大模型上的 FSDP。

## 和代码一致的部分

多卡打开 Ulysses 时，batch 长度仍是 1。各 rank 广播同一份 seed，切的是同一条序列，协作这一条视频。这和「用上下文并行降低单条延迟」的目标一致。

参数切分用的是 FSDP，不是张量并行。`wan/distributed/fsdp.py` 的 `shard_model` 使用 `ShardingStrategy.FULL_SHARD`，按 `model.blocks` 里的每一层包一个 FSDP 单元。隐藏维保持完整，进层时 all-gather 出完整权重。`--dit_fsdp` 与 `--t5_fsdp` 默认关闭；单进程下这两个开关和 `ulysses_size > 1` 会被拒绝。

## 和代码不一致的部分

仓库里没有 RingAttention，也没有第二个进程组。`generate.py` 只 `init_process_group` 一次，并断言 `--ulysses_size == WORLD_SIZE`。FSDP 的 `process_group` 使用默认组，Ulysses 的 all-to-all 也使用默认组。因此做不到「外层 Ring、内层 Ulysses」这种二维拆分，也没有外层通信与内层计算的重叠。

自注意力的并行只在 `wan/distributed/ulysses.py` 的 `distributed_attention`：all-to-all 把序列维换成 head 维，FlashAttention 算完再换回来。`wan/distributed/sequence_parallel.py` 的 `sp_dit_forward` 在嵌入之后按序列维 `chunk`，层末 `gather_forward` 把序列拼回去。

三条主管道挂的是同一对函数。`use_sp` 为真时，先替换 `self_attn.forward` 和 `WanModel.forward`，再按需包 FSDP：

| 任务 | 挂载位置 |
| --- | --- |
| `ti2v-5B` | `wan/textimage2video.py` 的 `_configure_model` |
| `t2v-A14B` | `wan/text2video.py` 的 `_configure_model`。高噪声、低噪声两份专家各配一次 |
| `i2v-A14B` | `wan/image2video.py` 的 `_configure_model`。同样两份专家各配一次 |

S2V 的注意力换成 `sp_attn_forward_s2v`，内部仍调用同一个 `distributed_attention`，没有 Ring。

只开 `--dit_fsdp`、不开 Ulysses 时，每张卡仍计算整段序列。FSDP 省的是常驻权重，不把单条视频的序列计算拆开。论文里负责降延迟的是上下文并行，负责 14B 显存的是 FSDP；本仓库的官方多卡命令把两者同时打开，但上下文并行只有内层 Ulysses。

TI2V-5B 的单卡路径大约 24GB 就能跑，和「切参数是为了 14B 显存」不冲突。A14B 的多卡命令同样是 `--dit_fsdp --t5_fsdp --ulysses_size`，并没有另外一套二维实现。论文 Fig. 12 的接近线性加速属于他们带二维上下文并行的 14B 系统，不能当成这个仓库已经达到的加速。

切分粒度、一次前向里的通信，以及四种合法开关组合，见 [TI2V-5B 推理侧并行方式](../ti2v_5b_parallelism.md)。
