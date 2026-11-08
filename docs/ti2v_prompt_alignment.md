# TI2V-5B 的 Prompt 改写

本文说明当前仓库里 **TI2V-5B** 如何实现论文 4.5 的推理侧 prompt alignment。Pipeline 里它出现在哪一步，见 [TI2V-5B 推理范围与 Pipeline](ti2v_5b_inference.md) 第 5.2 节和第 5.4 节。

论文把对齐分成两步。第一步在训练：每条图像或视频配上长短、文风不同的多条字幕。本仓库只有推理，没有这一步。第二步在推理：用大语言模型把用户往往很短的 prompt 改写成更接近训练字幕的分布。代码里的名字是 **prompt extend**，开关是 `--use_prompt_extend`，默认关闭。

## 1. 它改什么，不改什么

改写发生在 `generate.py` 里，构造 `WanTI2V` 之前。只改**正向** prompt。负向 prompt 仍是 `shared_config.py` 里的默认串，或用户后来传给 `t2v()` / `i2v()` 的 `n_prompt`。改写结果被当成普通字符串送进 umT5，DiT、步数、CFG 公式和 100 次前向都不变。

多卡时只有 rank 0 调用改写模型。成功则把新字符串放进长度为 1 的列表并 `broadcast_object_list`；失败则退回用户原文，再广播。其余 rank 使用同一条字符串，避免各卡的文本条件不一致。

## 2. 两条后端

`--prompt_extend_method` 默认是 `local_qwen`。

| 方法 | 默认模型 | 何时换成视觉模型 |
| --- | --- | --- |
| `dashscope` | 未指定时，纯文本用 `qwen-plus`，带图用 `qwen-vl-max` | `generate.py` 把 `is_vl` 设成「是否传入 `--image`」 |
| `local_qwen` | 未指定时，纯文本用 `Qwen/Qwen2.5-14B-Instruct`，带图用 `Qwen/Qwen2.5-VL-7B-Instruct` | 同上 |

论文为了平衡速度和质量，选定的改写模型是 **Qwen2.5-Plus**。DashScope 路径上的 `qwen-plus` 就是这一档，但代码不写死该名字，`--prompt_extend_model` 可以换成别的。本地路径的默认体积是 14B / VL-7B，不是 Plus。云端调用读环境变量 `DASH_API_KEY`，可选 `DASH_API_URL`。

`--prompt_extend_target_lang` 默认 `zh`，可选 `en`。它决定系统提示词的语言版本，也要求改写输出使用该语言。

## 3. TI2V 用哪一套系统提示词

`ti2v-5B` 没有单独写提示词，在 `wan/utils/prompt_extend.py` 的 `DEFAULT_SYS_PROMPTS` 里复用 A14B 的两套。`decide_system_prompt` 先看任务名里有没有 `ti2v`，再看这次调用是不是视觉模型：

| 调用 | 系统提示词 | 改写侧重 |
| --- | --- | --- |
| 文生视频，无 `--image` | `T2V_A14B_ZH_SYS_PROMPT` 或英文版 | 不改主体和动作；补光位、景别、构图和主体细节；没有动作就补动作；已有风格放在句首，没有风格就不加。篇幅大约 60–200 字 |
| 图生视频，有 `--image` | `I2V_A14B_ZH_SYS_PROMPT` 或英文版 | 结合图像强调运动和运镜，去掉已经出现在画面里的静态描述。篇幅不超过约 100 字 |

I2V-A14B 还有一套「prompt 为空、只凭图像写动态」的 empty 提示词。TI2V 的分支不看 prompt 是不是空串，带图时始终用上面的 I2V 改写提示词。

这些要求和论文的三条原则是同一类目标：补细节、尽量不动原意、给主体补上自然动作。结构并不相同。论文要求收成「风格，然后内容摘要，然后细节描述」。TI2V 文生视频的指令是「有风格才放在句首，否则不加」，示例多从光线和景别写起。

系统提示词里还有一段安全改写：遇到特定色情或亲密内容时，换成另一条高美感 prompt，而不是拒答。论文 4.5 没有写这一条。

## 4. 和热路径的边界

改写模型不在 100 次 DiT 前向里。本地 Qwen 会在 rank 0 上再占一份权重和一次生成；云端调用则把延迟放在网络请求上。最小的文生视频实现可以不加载 `wan/utils/prompt_extend.py` 和 `wan/utils/system_prompt.py`。数值对齐时要固定改写开关：打开和关闭会改变送进 umT5 的字符串，后面的 latent 不再可比。
