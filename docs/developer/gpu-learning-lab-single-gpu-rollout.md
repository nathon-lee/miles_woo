---
title: RTX A5000 单卡 rollout 与奖励实测
description: 在 24 GB RTX A5000 上跑通 Qwen3-0.6B、SGLang、Miles rollout 和数学奖励，并记录有效 GRPO 分组信号及完整训练的未验证边界。
---

本文记录 2026-08-26 在一张 NVIDIA RTX A5000 24 GB 上完成的实测。
它验证了以下数据链路：

```text
prompt
  → Qwen3-0.6B / SGLang 四路生成
  → Sample 完成或截断判定
  → math rule-based reward
  → [0, 1, 1, 1] 有效 GRPO 组内差异
  → rollout_0.pt 落盘
```

<Warning>
这是一次成功的 **rollout-only 与 reward smoke test**，不是完整 RL
训练成功记录。命令显式使用了 `--debug-rollout-only`，日志中的
`Timer train end (elapsed: 0.0s)` 证明没有执行 backward、`optimizer.step()`
或训练后的权重同步。
</Warning>

## 实验环境

| 项目 | 实测值 |
| --- | --- |
| GPU | 1× NVIDIA RTX A5000，24564 MiB |
| Driver / CUDA runtime | 580.159.04 / CUDA 13.0 |
| PyTorch | `2.13.0+cu130` |
| torchvision | `0.28.0+cu130` |
| Ray | `2.58.0` |
| Model | `/root/models/Qwen3-0.6B` |
| Miles checkout | `/root/miles_woo` |
| SGLang checkout | `/root/sglang/python/sglang` |
| 执行形态 | 直接从现有运行环境启动，不是按主教程的 `docker run` 命令创建 |
| 进程限制 | `Seccomp: 2`，无 `cap_sys_ptrace` |

实验时已验证 `torch.cuda.is_available() == True` 且
`torch.cuda.device_count() == 1`。本次保存的成功日志没有记录
Miles 和 SGLang 的 commit SHA；复现时应先执行：

```bash
cd /root/miles_woo
git rev-parse HEAD

cd /root/sglang
git rev-parse HEAD
```

## 准备最小数据

题目的正确答案是 `18`：

```text
(x + 1/x)^3 = x^3 + 1/x^3 + 3(x + 1/x)
x^3 + 1/x^3 = 27 - 9 = 18
```

创建只包含这道题的 JSONL：

```bash
mkdir -p /root/datasets/miles-smoke

printf '%s\n' \
  '{"prompt":"Let x be a positive real number satisfying x + 1/x = 3. Find x^3 + 1/x^3. Show your reasoning, then end with Answer: \\boxed{your_answer}.","label":"18"}' \
  > /root/datasets/miles-smoke/algebra.jsonl

python3 - <<'PY'
import json
from pathlib import Path

path = Path("/root/datasets/miles-smoke/algebra.jsonl")
row = json.loads(path.read_text().strip())
assert row["label"] == "18"
assert "\\boxed" in row["prompt"]
print(row)
PY
```

## 可复现的成功命令

```bash
cd /root/miles_woo
unset WANDB_API_KEY

export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

python3 scripts/run_qwen3_0_6b_fsdp.py \
  --num-gpus-per-node 1 \
  --num-rollout 1 \
  --data-dir /root/datasets \
  --model-dir /root/models \
  --output-dir /root/shared_data/algebra-direct \
  --extra-env-vars '{"LD_LIBRARY_PATH":"/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib","RAY_DEDUP_LOGS":"0"}' \
  --extra-args "--prompt-data /root/datasets/miles-smoke/algebra.jsonl \
    --rollout-batch-size 1 \
    --n-samples-per-prompt 4 \
    --global-batch-size 4 \
    --rollout-max-response-len 256 \
    --max-tokens-per-gpu 2048 \
    --apply-chat-template-kwargs '{\"enable_thinking\": false}' \
    --rm-type math \
    --sglang-cuda-graph-backend-decode disabled \
    --save-debug-rollout-data '/root/shared_data/algebra-direct/rollout_{rollout_id}.pt' \
    --debug-rollout-only \
    --skip-eval-before-train \
    --eval-interval 100"
```

这些非默认选项各自解决一个已观测问题：

| 选项 | 原因 |
| --- | --- |
| 顶层 `--extra-env-vars` | 把 CUDA 13 `libcudart.so.13` 路径显式传入 Ray runtime env；只在提交命令的 shell 中 `export` 不足以证明 worker 可见 |
| `enable_thinking=false` | 避免 Qwen3-0.6B 在小 token budget 中长时间停留在 thinking 阶段 |
| `--rm-type math` | 直接检查 `\boxed{}`；本实验不依赖 `deepscaler` 对 `</think>` 的格式要求 |
| `--sglang-cuda-graph-backend-decode disabled` | 规避 FA3 decode CUDA Graph 在 `bs=24` 捕获时的 `scheduler_metadata must have shape (metadata_size)` |
| `--debug-rollout-only` | 只验证生成、奖励和数据落盘，将权重同步问题排除在本次边界外 |

对于只生成 4 个 response 的学习实验，关闭 decode CUDA Graph
的性能损失可以接受。需要恢复小 batch 图捕获时，可单独验证：

```text
--sglang-cuda-graph-max-bs-decode 4
```

不要将这个未实测的替代参数写成已通过结论。

## 实测结果

服务端、生成、数据保存和 Ray Job 都正常结束：

```text
The server is fired up and ready to roll!
Final collected 4 samples from rollout to train
Save debug rollout data to /root/shared_data/algebra-direct/rollout_0.pt
Job 'raysubmit_pkndZMnZd63j68nA' succeeded
```

| 指标 | 实测值 |
| --- | ---: |
| `rollout/num_training_samples` | 4 |
| `rollout/episode_raw_reward` | 0.75 |
| response length mean / median | 222.25 / 216 |
| response length min / max | 201 / 256 |
| `rollout/truncated_ratio` | 0.25 |
| `perf/rollout_time` | 4.2987 s |
| `perf/tokens_per_gpu_per_sec` | 206.81 |
| 运行中 GPU used / free | 18.69 / 4.86 GB |

`--rm-type math` 返回二值奖励，因此平均奖励 `0.75` 对应
3 个正样本和 1 个负样本：

| Sample | Status | Length | `\boxed` | Reward |
| ---: | --- | ---: | --- | ---: |
| 0 | `truncated` | 256 | 无 | 0 |
| 1 | `completed` | 228 | `\boxed{18}` | 1 |
| 2 | `completed` | 204 | `\boxed{18}` | 1 |
| 3 | `completed` | 201 | `\boxed{18}` | 1 |

这个 `[0, 1, 1, 1]` 分布不是“reward 上涨”证据，但它是有效的
GRPO 组内相对信号：如果执行训练，算法可以区分在限长内完成并输出
正确格式的 response，以及因冗长而被截断的 response。

## 检查保存的 Sample

当前 checkout 中
`python3 -m miles.utils.debug_utils.display_debug_rollout_data` 会因导入缺失的
`compute_perf_metrics_from_samples` 失败。本次使用 `torch.load` 直接检查：

```bash
cd /root/miles_woo

python3 - <<'PY'
import torch

path = "/root/shared_data/algebra-direct/rollout_0.pt"
pack = torch.load(path, weights_only=False, map_location="cpu")
rewards = []

for i, sample in enumerate(pack["samples"]):
    response = sample.get("response", "")
    reward = sample.get("reward")
    rewards.append(reward)

    print(f"\n===== sample {i} =====")
    print("reward:", reward)
    print("status:", sample.get("status"))
    print("response_length:", sample.get("response_length"))
    print("has boxed:", "\\boxed" in response)
    print("tail:")
    print(response[-400:])

print("\nreward distribution:", rewards)
print("positive:", sum(r == 1 for r in rewards))
print("negative:", sum(r == 0 for r in rewards))
PY
```

验收输出：

```text
reward distribution: [0, 1, 1, 1]
positive: 3
negative: 1
```

## 故障到修复的证据链

| 现象 | 根因 | 本次处理 |
| --- | --- | --- |
| SGLang 子进程报 `libcudart.so.13` 不存在 | CUDA 13 runtime 目录没有稳定进入 Ray worker 环境 | 通过顶层 `--extra-env-vars` 传递 `LD_LIBRARY_PATH` |
| SGLang 在 decode graph 捕获时退出 | FA3 在 `bs=24` 触发 `scheduler_metadata` shape 错误 | smoke test 中关闭 decode CUDA Graph |
| thinking 模式下 4 条 response 全部奖励为 0 | 512 tokens 内没有生成 `</think>` 和 `\boxed{18}` | 关闭 thinking，使用 `math` reward，将测试聚焦在最终答案 |
| rollout 指标显示成功，但 train 耗时为 0 | `--debug-rollout-only` 的预期行为 | 只将结果表述为 rollout/reward 链路成功 |

## 显存指标如何解读

清理 FSDP actor 后日志仍显示：

```text
used_GB=18.69, allocated_GB=0.0, reserved_GB=0.0
```

`allocated_GB` 和 `reserved_GB` 是 FSDP actor 当前进程的 PyTorch
统计；`used_GB` 是整张 GPU 的使用量。此时 SGLang 仍在另一个进程中持有
模型、KV cache 和图捕获内存，因此不能单凭这行判定泄漏。Job 结束后检查：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi
```

## 尚未验证的完整闭环

本次结果不支持以下声称：

- FSDP backward 成功；
- optimizer 更新了 actor 权重；
- 新权重成功同步到 SGLang；
- 第二轮 rollout 使用了新权重；
- reward 随训练上升。

当前运行环境在 colocate FSDP 权重同步时已观测到
`pidfd_getfd: Operation not permitted`。`ptrace_scope=1`、`Seccomp: 2` 且缺少
`cap_sys_ptrace` 表明这是进程权限边界，不是通过增加 token budget
或修改 reward 可以解决的算法问题。

下一阶段应在允许 Miles/SGLang 权重传输所需进程操作的环境中，去掉
`--debug-rollout-only`，并以两轮连续的 rollout、train step 和 weight update
作为完整闭环验收标准。

完成数学二值 reward 后，可继续运行
[情绪支持对话的自定义奖励实测](/developer/gpu-learning-lab-empathy-reward)，
学习多维 reward 分解、元数据记录和 reward hacking 诊断。
