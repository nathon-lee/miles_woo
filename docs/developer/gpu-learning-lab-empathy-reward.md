---
title: 情绪支持对话的自定义奖励实测
description: 在 RTX A5000 单卡上用 Qwen3-0.6B 生成支持性回复，通过 Miles 自定义 reward 分解情绪、格式、共情、开放式问题、安全和简洁度。
---

本文记录 2026-08-26 在一张 NVIDIA RTX A5000 24 GB 上完成的
非数学 reward smoke test。实验让 Qwen3-0.6B 对一段面试失败后的
失落表达生成 4 条支持性回复，再用一个本地 Python 函数计算多维奖励。

实测跑通了：

```text
中文情绪 prompt
  → SGLang 四路采样
  → empathy_reward.reward_func
  → 奖励分项写入 Sample.metadata
  → [0.40, 0.55, 0.55, 0.55]
  → rollout_0.pt 落盘
```

<Warning>
这不是心理治疗模型或临床质量证明。任务被明确限定为非医疗性
支持对话，奖励函数也只是教学用关键词规则。本次命令使用
`--debug-rollout-only`，没有执行 backward、optimizer step 或权重同步。
</Warning>

## 这个实验想学什么

数学 reward 通常可以根据最终答案判定 0 或 1；“一条回复是否有共情”
没有唯一答案。本实验把主观目标拆成六个可观察项：

| 分项 | 权重 | 教学规则 |
| --- | ---: | --- |
| `emotion` | 0.25 | `Emotion:` 一行必须精确等于标签“失落” |
| `format` | 0.15 | 存在 `Emotion`、`Response` 和 `Question` 三个字段 |
| `empathy` | 0.20 | `Response` 同时命中负面感受词和付出词 |
| `open_question` | 0.15 | `Question` 包含开放式词和问号 |
| `safe` | 0.15 | 不出现说教、简单安慰或疾病诊断词 |
| `concise` | 0.10 | 总文本长度为 40–260 字符 |

这种分项能帮助定位模型丢分的原因，但它不等于真实的人类偏好。

## 创建数据和 reward 函数

实验文件放在 `/root/shared_data/empathy-lab`：

```bash
mkdir -p /root/shared_data/empathy-lab

cat > /root/shared_data/empathy-lab/empathy.jsonl <<'JSON'
{"prompt":"你正在进行非医疗性的支持性对话。用户说：我准备了三个月的面试还是失败了，现在很失落，觉得自己什么事情都做不好。请严格只输出三行，不要使用 Markdown：\nEmotion: 从失落、愤怒、焦虑、开心中选择一个\nResponse: 用一句话确认用户的感受和付出的努力，不要说“别难过”“想开点”“加油”，不要进行疾病诊断\nQuestion: 提出一个开放式问题，帮助用户具体复盘这次经历","label":"失落"}
JSON

cat > /root/shared_data/empathy-lab/empathy_reward.py <<'PY'
import re


def _score(sample):
    text = (sample.response or "").replace("<|im_end|>", "").strip()
    label = str(sample.label).strip()

    emotion_ok = bool(
        re.search(
            rf"(?im)^\s*Emotion\s*[:：]\s*{re.escape(label)}\s*$",
            text,
        )
    )

    response_match = re.search(r"(?im)^\s*Response\s*[:：]\s*(.+)$", text)
    question_match = re.search(r"(?im)^\s*Question\s*[:：]\s*(.+)$", text)
    response_line = response_match.group(1).strip() if response_match else ""
    question_line = question_match.group(1).strip() if question_match else ""

    exact_format = all(
        re.search(rf"(?im)^\s*{field}\s*[:：]", text)
        for field in ("Emotion", "Response", "Question")
    )

    acknowledges_emotion = any(
        word in response_line
        for word in ("失落", "难受", "沮丧", "挫败", "不好受")
    )
    acknowledges_effort = any(
        word in response_line
        for word in ("准备", "三个月", "投入", "努力", "付出")
    )
    empathy_ok = acknowledges_emotion and acknowledges_effort

    open_question = bool(question_line) and any(
        word in question_line
        for word in ("什么", "哪", "如何", "怎么", "愿意", "最")
    ) and ("?" in question_line or "？" in question_line)

    forbidden = (
        "别难过",
        "想开点",
        "加油",
        "你应该",
        "抑郁症",
        "焦虑症",
        "你有病",
        "一定是",
    )
    safe = not any(word in text for word in forbidden)
    concise = 40 <= len(text) <= 260

    breakdown = {
        "emotion": 0.25 if emotion_ok else 0.0,
        "format": 0.15 if exact_format else 0.0,
        "empathy": 0.20 if empathy_ok else 0.0,
        "open_question": 0.15 if open_question else 0.0,
        "safe": 0.15 if safe else 0.0,
        "concise": 0.10 if concise else 0.0,
    }
    reward = round(sum(breakdown.values()), 2)

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["reward_breakdown"] = breakdown
    return reward


async def reward_func(args, samples, **kwargs):
    if isinstance(samples, list):
        return [_score(sample) for sample in samples]
    return _score(samples)
PY
```

Miles 会通过 `--custom-rm-path empathy_reward.reward_func` 导入这个函数。
当不是 multi-LoRA 路径时，自定义 reward 会收到一个 `Sample` 列表，
所以这里为列表中的每条样本返回一个分数。

## 先单测 reward

在占用 GPU 之前，先确认质量较好的回复得分高于违反规则的回复：

```bash
PYTHONPATH=/root/shared_data/empathy-lab:/root/miles_woo \
python3 - <<'PY'
import asyncio
from types import SimpleNamespace

from empathy_reward import reward_func

good = SimpleNamespace(
    response=(
        "Emotion: 失落\n"
        "Response: 准备三个月却没有得到期待的结果，确实会让人感到失落和挫败。\n"
        "Question: 这次面试中，你觉得哪一部分最值得具体复盘？"
    ),
    label="失落",
    metadata={},
)

bad = SimpleNamespace(
    response=(
        "Emotion: 开心\n"
        "Response: 别难过，加油就好了。\n"
        "Question: 好吗？"
    ),
    label="失落",
    metadata={},
)

rewards = asyncio.run(reward_func(None, [good, bad]))
print("rewards:", rewards)
print("good breakdown:", good.metadata["reward_breakdown"])
print("bad breakdown:", bad.metadata["reward_breakdown"])
assert rewards[0] > rewards[1]
PY
```

实测输出：

```text
rewards: [1.0, 0.25]
good breakdown: {'emotion': 0.25, 'format': 0.15, 'empathy': 0.2, 'open_question': 0.15, 'safe': 0.15, 'concise': 0.1}
bad breakdown: {'emotion': 0.0, 'format': 0.15, 'empathy': 0.0, 'open_question': 0.0, 'safe': 0.0, 'concise': 0.1}
```

## 可复现的成功命令

该命令延续
[RTX A5000 单卡 rollout 实测](/developer/gpu-learning-lab-single-gpu-rollout)
已验证的 CUDA 13 和 SGLang 配置：

```bash
cd /root/miles_woo
unset WANDB_API_KEY

export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

python3 scripts/run_qwen3_0_6b_fsdp.py \
  --num-gpus-per-node 1 \
  --num-rollout 1 \
  --data-dir /root/datasets \
  --model-dir /root/models \
  --output-dir /root/shared_data/empathy-lab/run \
  --extra-env-vars '{"LD_LIBRARY_PATH":"/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib","RAY_DEDUP_LOGS":"0","PYTHONPATH":"/root/shared_data/empathy-lab"}' \
  --extra-args "--prompt-data /root/shared_data/empathy-lab/empathy.jsonl \
    --rollout-batch-size 1 \
    --n-samples-per-prompt 4 \
    --global-batch-size 4 \
    --rollout-temperature 1.2 \
    --rollout-max-response-len 256 \
    --max-tokens-per-gpu 2048 \
    --apply-chat-template-kwargs '{\"enable_thinking\": false}' \
    --custom-rm-path empathy_reward.reward_func \
    --sglang-cuda-graph-backend-decode disabled \
    --save-debug-rollout-data '/root/shared_data/empathy-lab/run/rollout_{rollout_id}.pt' \
    --debug-rollout-only \
    --skip-eval-before-train \
    --eval-interval 100"
```

`PYTHONPATH` 需要经由顶层 `--extra-env-vars` 进入 Ray runtime env，否则 worker
不一定能导入 `/root/shared_data/empathy-lab/empathy_reward.py`。

## 实测结果

成功标志：

```text
The server is fired up and ready to roll!
Final collected 4 samples from rollout to train
Save debug rollout data to /root/shared_data/empathy-lab/run/rollout_0.pt
Job 'raysubmit_2jJ5rEn6c5kCDyj9' succeeded
```

| 指标 | 实测值 |
| --- | ---: |
| `rollout/num_training_samples` | 4 |
| `rollout/episode_raw_reward` | 0.5125 |
| response length mean / median | 49 / 52 |
| response length min / max | 35 / 57 |
| `rollout/truncated_ratio` | 0.0 |
| `perf/rollout_time` | 1.0948 s |
| `perf/tokens_per_gpu_per_sec` | 179.04 |
| 奖励分布 | `[0.40, 0.55, 0.55, 0.55]` |
| 唯一奖励 | `[0.40, 0.55]` |

四条样本都正常完成，没有被 256 token 上限截断。同一 prompt 内存在
`0.40` 和 `0.55` 两种奖励，因此形式上已有 GRPO 组内差异。

## 查看每条回复的分项

```bash
cd /root/miles_woo

python3 - <<'PY'
import torch

path = "/root/shared_data/empathy-lab/run/rollout_0.pt"
pack = torch.load(path, weights_only=False, map_location="cpu")
rewards = []

for i, sample in enumerate(pack["samples"]):
    reward = sample.get("reward")
    metadata = sample.get("metadata") or {}
    rewards.append(reward)

    print(f"\n===== sample {i} =====")
    print("reward:", reward)
    print("status:", sample.get("status"))
    print("response_length:", sample.get("response_length"))
    print("breakdown:", metadata.get("reward_breakdown"))
    print("response:")
    print(sample.get("response", ""))

print("\nreward distribution:", rewards)
print("unique rewards:", sorted(set(rewards)))
PY
```

实测分项：

| Sample | Reward | Emotion | Format | Empathy | Open question | Safe | Concise |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.40 | 0 | 0.15 | 0 | 0 | 0.15 | 0.10 |
| 1 | 0.55 | 0 | 0.15 | 0 | 0.15 | 0.15 | 0.10 |
| 2 | 0.55 | 0 | 0.15 | 0 | 0.15 | 0.15 | 0.10 |
| 3 | 0.55 | 0 | 0.15 | 0 | 0.15 | 0.15 | 0.10 |

## 比 Job succeeded 更重要的结论

基础设施已经成功，但这个 reward v1 还不适合真正训练。

### 1. 精确匹配丢掉了语义正确的近义表达

Sample 0 输出 `Emotion: 内心的失落`，包含正确情绪，但因为不精确等于
`失落`，`emotion` 仍得 0。Sample 2 使用了近义词“悲哀”，也不能得分。

### 2. 关键词奖励没有理解回复质量

Sample 3 的回复是：

```text
Emotion: 焦虑
Response: 面试失败后的人生变数肯定是钱财硕果。
Question: 最几百心中贪婪去哪里了？
```

它语义不通且不具支持性，却因为有三个字段、包含“最”和问号、
没有命中禁用词并且长度合适，与相对合理的 Sample 1、2 同得 `0.55`。

这是一个典型的 reward hacking 风险：模型可能学会满足表面规则，
而不是提高真实的回复质量。

### 3. `safe` 只表示“未命中有限黑名单”

Sample 3 得到 `safe=0.15` 不证明它安全，只证明文本中没有出现
8 个指定禁用短语。不能把黑名单命中率当作心理支持安全性评估。

## 下一版 reward 的改进顺序

在去掉 `--debug-rollout-only` 之前，先改进 reward 并重放当前
`rollout_0.pt`：

1. 严格校验只有三个非空字段，而不是只检查字段名存在；
2. 对情绪标签先做规范化，有限地支持“内心的失落”等表达；
3. 把“开放式”和“语义相关”分开，问题必须与面试经历相关；
4. 增加语义可读性和人设一致性评估，避免 Sample 3 这样的乱文获得中等分；
5. 用人工标注的好/坏/边界用例建立 reward 回归测试；
6. 最后才考虑 LLM judge，并用多评审或人工抽检减少 judge 偏差。

Miles 提供 `miles.utils.debug_utils.replay_reward_fn` 用于对已保存的 rollout
重放自定义 reward。在使用它之前，先通过 `--help` 核对当前 checkout
的参数；本次实测没有执行 reward replay，不将它记为已验证步骤。

## 验证边界

本次已验证：

- Ray worker 能通过自定义 `PYTHONPATH` 导入 reward 函数；
- SGLang 能在单卡上完成 4 条中文回复；
- Miles 能接收列表型自定义 reward 结果；
- reward breakdown 能通过 `Sample.metadata` 保存到 rollout 文件；
- 组内出现了两种奖励。

本次未验证：

- reward 与人类偏好或支持性对话质量有足够相关性；
- 临床安全、自伤风险识别或危机干预能力；
- backward、optimizer step、权重同步和第二轮 rollout；
- 训练能改善这些回复。

如果要从 rollout-only 继续学习完整的 backward、checkpoint 和训练后权重同步，
可参考[单卡沉没成本决策 RL 成功实测](/developer/gpu-learning-lab-sunk-cost-training)。
