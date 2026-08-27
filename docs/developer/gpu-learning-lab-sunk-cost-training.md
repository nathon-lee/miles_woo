---
title: 单卡沉没成本决策 RL 成功实测
description: 在 RTX A5000 单卡上用 Qwen3-0.6B、FSDP、SGLang 和自定义 reward 跑通两轮 GRPO 训练、checkpoint 保存及训练后权重同步。
---

本文记录 2026-08-26 在一张 NVIDIA RTX A5000 24 GB 上完成的
Miles 完整训练闭环。实验任务不是数学题，而是让模型识别“沉没成本”
并给出是否重新评估课程投入的建议。

<Warning>
这是一个学习用的两轮、单 prompt 实验，不是证明模型已经学会决策能力的
基准。reward 是关键词规则，训练样本很少，结果只能证明工程链路跑通。
</Warning>

## 实验环境

| 项目 | 实测值 |
| --- | --- |
| GPU | 1× NVIDIA RTX A5000，24564 MiB |
| Driver / CUDA | 580.159.04 / CUDA 13.0 |
| PyTorch / torchvision | `2.13.0+cu130` / `0.28.0+cu130` |
| Ray | `2.58.0` |
| Model | `/root/models/Qwen3-0.6B` |
| Miles checkout | `/root/miles_woo` |
| SGLang checkout | `/root/sglang` |
| Backend | FSDP，单卡 colocate |
| Rollout | 每轮 8 个 sample，最多 256 token |
| 输出目录 | `/root/shared_data/sunk-cost-lab/train-sync-v2` |

本次运行使用了 CUDA 13 动态库路径，并将
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` 同时传给 Ray 主进程和
训练 worker，以避开受限运行环境中的 CUDA IPC `pidfd_getfd` 问题。

## 训练前的两个问题

### 1. 权重同步权限问题已经解决

此前 colocate FSDP 在初始同步时失败：

```text
RuntimeError: pidfd_getfd: Operation not permitted
```

本次日志中初始 `/begin_weight_update`、多次
`/update_weights_from_tensor` 和 `/end_weight_update` 均返回 HTTP 200，说明
`expandable_segments:False` 在当前环境中解决了这个阻塞。

### 2. FSDP log-prob 路径错误导入 Megatron

第一次完整训练在 `compute_log_probs()` 处失败：

```text
ModuleNotFoundError: No module named 'megatron'
```

修复方式是：当 TP group size 为 1（FSDP 单卡完整词表）时使用
`torch.nn.functional.log_softmax(...).gather(...)`；只有 Megatron TP>1
才导入 fused vocab-parallel cross entropy。这个 fallback 通过独立单测后再运行
完整训练。

注意：上面的 fallback 是本次实验运行树中的本地代码修改；本学习文档只记录
运行证据，不会自动把该 Python patch 应用到读者的 checkout。若你的 checkout
仍在这里报同样的 `ModuleNotFoundError`，应先合入或应用该 fallback，再执行完整
训练命令。

## 数据和 reward

输入文件：

```text
/root/shared_data/sunk-cost-lab/sunk_cost.jsonl
```

prompt 要求模型严格输出四行：

```text
Bias: ...
Decision: ...
Reason: ...
Action: ...
```

reward 总分由以下规则组成：

| 分项 | 权重 | 规则摘要 |
| --- | ---: | --- |
| `bias` | 0.30 | 命中“沉没成本”或标签 |
| `format` | 0.10 | 四个字段均出现 |
| `decision` | 0.10 | 出现“重新评估” |
| `irreversible` | 0.15 | 说明过去投入无法收回 |
| `future` | 0.15 | 提及未来收益、成本、时间或价值 |
| `action` | 0.15 | 提出退款、转课、比较或评估等行动 |
| `detail` | 0.05 | 对有限文本长度给少量奖励 |

重放已有 rollout 时得到：

```text
rewards: [0.3905, 0.3912, 0.3491, 0.3472]
unique rewards: [0.3472, 0.3491, 0.3905, 0.3912]
GRPO reward variance OK
```

因此同一 prompt 内存在组内差异，不会触发全组 zero-std 过滤。

## 实际训练命令

运行时使用两轮 rollout，并去掉 `--debug-rollout-only`：

```bash
cd /root/miles_woo
ray stop --force >/dev/null 2>&1 || true
unset WANDB_API_KEY
unset PYTORCH_ALLOC_CONF

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

RUN_DIR=/root/shared_data/sunk-cost-lab/train-sync-v2
mkdir -p "$RUN_DIR"
set -o pipefail

python3 scripts/run_qwen3_0_6b_fsdp.py \
  --num-gpus-per-node 1 \
  --num-rollout 2 \
  --data-dir /root/datasets \
  --model-dir /root/models \
  --output-dir "$RUN_DIR" \
  --extra-env-vars '{"LD_LIBRARY_PATH":"/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib","RAY_DEDUP_LOGS":"0","PYTHONPATH":"/root/shared_data/sunk-cost-lab","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:False"}' \
  --extra-args "--prompt-data /root/shared_data/sunk-cost-lab/sunk_cost.jsonl \
    --rollout-batch-size 1 \
    --n-samples-per-prompt 8 \
    --global-batch-size 8 \
    --rollout-temperature 1.3 \
    --rollout-max-response-len 256 \
    --max-tokens-per-gpu 2048 \
    --apply-chat-template-kwargs '{\"enable_thinking\": false}' \
    --custom-rm-path sunk_cost_reward.reward_func \
    --train-env-vars '{\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:False\"}' \
    --sglang-cuda-graph-backend-decode disabled \
    --save-debug-rollout-data '$RUN_DIR/rollout_{rollout_id}.pt' \
    --save $RUN_DIR/checkpoints \
    --save-interval 1 \
    --skip-eval-before-train \
    --eval-interval 100 \
    --debug-exit-after-rollout 2" \
  2>&1 | tee "$RUN_DIR/train.log"
```

## 必要训练日志

以下片段比完整的 `server_args=...` 参数转储更适合长期保存。

### 初始权重同步

```text
[09:23:39] POST /begin_weight_update HTTP/1.1 200 OK
[09:23:39] POST /update_weights_from_tensor HTTP/1.1 200 OK
[09:23:40] POST /end_weight_update HTTP/1.1 200 OK
```

### 第一轮训练

```text
rollout 0: ... 'rollout/raw_reward': 0.30377499999999996,
            'rollout/weight_version/mean': 1.0
step 0: {'train/loss': -2.1420419e-07,
         'train/pg_loss': -2.1420419e-07,
         'train/grad_norm': 14.533173561096191,
         'train/lr-pg_0': 1e-06}
[FSDP] Saved checkpoint to .../checkpoints/iter_0000001
```

### 第二轮训练和新权重

```text
rollout 1: ... 'rollout/raw_reward': 0.3375125,
            'rollout/weight_version/mean': 2.0
step 1: {'train/loss': 3.0547380e-07,
         'train/pg_loss': 3.0547380e-07,
         'train/grad_norm': 15.142637252807617,
         'train/lr-pg_0': 1e-06}
[FSDP] Saved checkpoint to .../checkpoints/iter_0000002
```

### 训练后同步和任务结束

```text
[09:24:22] POST /begin_weight_update HTTP/1.1 200 OK
[09:24:22] POST /update_weights_from_tensor HTTP/1.1 200 OK
[09:24:23] POST /end_weight_update HTTP/1.1 200 OK
Job 'raysubmit_ywW1tZBrrN3xkWxk' succeeded
```

整次运行的验收统计：

| 项目 | 实测结果 |
| --- | ---: |
| `end_weight_update` 成功次数 | 3（初始、第一轮后、第二轮后） |
| rollout 0 weight version | 1 |
| rollout 1 weight version | 2 |
| rollout 0 raw reward | 0.303775 |
| rollout 1 raw reward | 0.3375125 |
| `train/grad_norm` | 14.5332、15.1426 |
| checkpoint tracker | `2` |
| 最终 Job | succeeded |

`rollout/rewards` 在训练日志中是归一化后的组内信号，接近 0 是正常的；
判断 reward 时应看 `rollout/raw_reward` 和原始 sample reward。

## 查看 checkpoint 内部

```bash
CKPT=/root/shared_data/sunk-cost-lab/train-sync-v2/checkpoints/iter_0000002/model

test -d "$CKPT" && echo "checkpoint exists"
du -sh "$CKPT"
find "$CKPT" -maxdepth 1 -type f -printf "%f\t%k KB\n" | sort
```

实测：

```text
checkpoint exists
2.9G    .../iter_0000002/model
.metadata       108 KB
__0_0.distcp    2936544 KB
```

读取 DCP metadata：

```bash
python3 - "$CKPT" <<'PY'
import sys
from torch.distributed.checkpoint import FileSystemReader

reader = FileSystemReader(sys.argv[1])
metadata = reader.read_metadata()
print("tensor count:", len(metadata.state_dict_metadata))

for i, (name, spec) in enumerate(sorted(metadata.state_dict_metadata.items())):
    if i >= 30:
        break
    props = getattr(spec, "properties", None)
    print(
        name,
        "shape=", getattr(spec, "size", None),
        "dtype=", getattr(props, "dtype", None),
    )
PY
```

实测 metadata 包含 311 个 tensor；例如：

```text
model_state.model.lm_head.weight
  shape=torch.Size([151936, 1024]), dtype=torch.float32
model_state.model.model.layers.0.self_attn.q_proj.weight
  shape=torch.Size([2048, 1024]), dtype=torch.float32
```

加载 checkpoint 并和原始模型比较时，实测：

```text
changed parameter tensors: 243 / 310
global max abs delta: 3.814697265625e-06
```

变化很小是预期结果：本次只有两轮更新，学习率为 `1e-6`，且优势值约为
`1e-7`。这能证明参数确实被更新，不代表模型已经出现明显行为变化。

## 结论和边界

本次已验证：

- 单卡 SGLang rollout 正常生成；
- 自定义 reward 能产生组内差异；
- FSDP 能计算 ref/log probabilities；
- backward、`optimizer.step()` 和梯度计算完成；
- FSDP checkpoint 成功保存；
- 训练后的权重连续同步回 SGLang；
- 第二轮 rollout 使用了新 `weight_version=2`。

本次没有证明：

- 沉没成本决策能力在更多 prompt 上提升；
- reward 与人类偏好高度一致；
- 训练后的 DCP 已经导出为独立 HuggingFace safetensors；
- `/root/models/Qwen3-0.6B` 被覆盖。该目录仍是原始输入模型。

下一步应增加多个不同情境和人工标注的验证集，再进行更长的训练；不要只根据
两轮 raw reward 的上升就宣称模型能力提升。
