---
title: Miles FSDP2 on MUSA RL experiment report
description: Miles 使用 FSDP2、SGLang 和 GRPO 在摩尔线程 MUSA GPU 上完成强化学习闭环的实验报告。
---

# Miles FSDP2 on MUSA 强化学习实验报告

## 1. 摘要

本次实验在摩尔线程 MUSA GPU 上验证了 Miles 的 FSDP2 强化学习训练链路。模型使用
Qwen3-0.6B，rollout 由 SGLang 执行，训练后端为 PyTorch FSDP2，集合通信使用 MCCL，
强化学习算法为 GRPO（Group Relative Policy Optimization）。

已跑通的数据流为：

```text
GSM8K prompt
    -> SGLang rollout on MUSA
    -> rule-based math reward
    -> GRPO group-relative advantage
    -> FSDP2 log-prob / backward / optimizer
    -> distributed weight update
    -> SGLang receives new weights
    -> next rollout
```

实验分为三层：

1. 两卡原生 FSDP2 + DTensor + MCCL smoke。
2. Miles + Ray + SGLang + FSDP2 单轮及 10-rollout 完整闭环。
3. 180-rollout 长跑和 reward 趋势分析。

最终结论是：**Miles 的 FSDP2 RL 闭环已经在 MUSA 上跑通；180-rollout 长跑显示
reward 相比训练初期有所提升，并在约 `0.70` 附近波动，但当前证据不足以证明模型已经
收敛。**

## 2. 实验范围

### 2.1 软件与硬件

| 项目 | 实验值 |
|---|---|
| GPU 物理型号 | Moore Threads X10000，单卡显存 81920 MiB |
| 运行时设备名称 | `torch_musa` 返回 `MTT S5000` |
| PyTorch | `2.7.1` |
| torch_musa | `2.7.1+d212062` |
| SGLang | `0.5.6.post2` |
| Transformers | `5.12.1` |
| Ray | `2.54.0` |
| Polars | `1.42.1` |
| 模型 | Qwen3-0.6B |
| 数据集 | GSM8K |
| 强化学习算法 | GRPO |
| 训练后端 | FSDP2 |
| rollout 后端 | SGLang |
| 集合通信 | MCCL |
| 优化器 | Adam |

`X10000` 是设备管理工具报告的物理产品型号；`MTT S5000` 是当前 torch_musa 版本返回的
运行时名称。两者描述的是同一实验设备的不同软件视角。

### 2.2 GRPO 配置

实验通过下列参数选择 GRPO：

```bash
--advantage-estimator grpo
--n-samples-per-prompt 2
```

GRPO 对同一个 prompt 生成多个回答，以组内 reward 的相对差异计算 advantage，不需要
单独训练 critic/value model。本实验还使用了：

```bash
--kl-loss-coef 0.0
--kl-coef 0.0
--entropy-coef 0.0
```

因此本次配置没有启用额外 KL 正则和 entropy bonus。Reward/verifier 是独立于 GRPO 的
组件；10-rollout GSM8K 实验使用数学规则 reward。

## 3. 验证结果

### 3.1 原生 FSDP2 smoke

两卡 smoke 连续完成 3 个 step，覆盖：

- `fully_shard` 和 DTensor；
- MCCL process group；
- forward、backward 和 AdamW 更新；
- 有限梯度检查；
- all-reduce 和 barrier。

成功标志为：

```text
FSDP2_MUSA_SMOKE_OK
FSDP2_EXIT=0
```

这层证据只说明 PyTorch FSDP2 和 MCCL 基础能力可用，不等价于 Miles RL 闭环已经可用。

### 3.2 10-rollout 完整闭环

10 个 rollout 全部完成 rollout、log-prob、actor training 和权重同步：

```text
rollout_files: 10
samples: 80
nonzero_reward: 53
boxed_answers: 63
truncated: 20
nonzero_reward_ratio: 0.6625
boxed_ratio: 0.7875
truncated_ratio: 0.25
grad_norm: 4.013991355895996
```

权重版本从 1 单调递增到 10，每轮均为：

```text
rollout/weight_version/mixed_version_ratio: 0.0
```

这说明 rollout 没有混用新旧权重。非零且有限的梯度范数说明训练阶段实际执行了反向传播
和参数更新，而不只是生成 rollout 数据。

### 3.3 180-rollout 长跑

长跑日志中解析出 rollout `0`–`179`，共 180 条 reward 指标。窗口统计为：

| 窗口 | Reward mean | Min | Max |
|---|---:|---:|---:|
| 前 20 个 rollout | 约 `0.628` | — | — |
| 后 50 个 rollout | `0.700000` | `0.312500` | `1.000000` |
| 后 20 个 rollout | `0.721875` | `0.312500` | `1.000000` |
| 后 10 个 rollout | `0.687500` | `0.375000` | `1.000000` |
| rollout 179 | `0.562500` | — | — |

后 50 个 rollout 相比前 20 个，平均 reward 提升约 `0.072`，相对提升约 `11%`。
这说明训练存在有效学习信号。与此同时：

- 后 10、20 和 50 个窗口都在约 `0.70` 附近；
- 窗口均值没有单调上升；
- 单轮 reward 仍在 `0.3125`–`1.0` 之间明显波动；
- rollout 179 的 reward 为 `0.5625`。

所以当前状态应描述为“reward 有提升并出现初步平台趋势”，不能描述为“RL 已经收敛”。
训练 batch 每轮使用不同 prompt，其 reward 也不能替代固定验证集 accuracy。

### 3.4 固定 GSM8K eval：checkpoint 160 → 200

在同一批 128 条 GSM8K 验证样本上，checkpoint `160` 和 `200` 的固定评估结果为：

| Checkpoint | `eval/gsm8k` | 等价正确数 |
|---:|---:|---:|
| 160 | `0.703125` | 90/128 |
| 200 | `0.7265625` | 93/128 |

checkpoint `200` 的 eval 正常完成（`EVAL_AT_200_RETRY_EXIT=0`），相较 checkpoint `160`
提升 `3/128`，即 `+2.34` 个百分点；`eval/truncated_ratio` 为 `0.1171875`。
这是一个积极信号，但只有一个固定 eval 区间，仍不足以判断已经收敛。

eval-only 进程中的 `eval 0` 和 `weight_version=1.0` 是进程内部计数，不表示模型回到了
第 0 步；本次实际加载的是 checkpoint `200`。后续使用 `--eval-interval 20` 从 200
继续训练时，应在 rollout `220、240、...、400` 记录同一固定评估集的结果。

### 3.5 NVIDIA 官方结果说明

本报告目前没有 NVIDIA 官方团队在相同 Miles、Qwen3-0.6B、GSM8K 和 FSDP2 配置下运行
并公开的精度或收敛数据。文中的 MUSA 数字是本项目在 Moore Threads 节点上的实测；
RTX A4500/CUDA 仅是建议的后续基线，不代表 NVIDIA 官方测试结果。因此，当前结果不能
用于声称 MUSA 与 NVIDIA 的收敛性能或最终精度一致。需要进行硬件归因时，应使用相同
checkpoint、数据划分、采样参数和固定 eval 集，在 NVIDIA GPU 上单独复现实验并记录结果。

### 3.6 主长跑 200–399 的 reward signal 诊断

主长跑 `20260901-191717` 共完成 200 个训练 rollout，包含 3200 个样本和 800 个四样本
group。reward 为二值 `0/1`，总体 reward mean 为 `0.7321875`。

| Group 类型 | 数量 | 比例 |
|---|---:|---:|
| `0000`（全错） | 96 | `12.00%` |
| `1111`（全对） | 439 | `54.875%` |
| `mixed`（组内有对有错） | 265 | `33.125%` |

这意味着 `66.875%` 的 group 没有组内 reward 差异，不能提供有效的 GRPO 相对 advantage；
只有约三分之一的 group 能区分同一 prompt 下的不同回答。后期 mixed group 比例从
`200–219` 的 `37.5%` 降到 `380–399` 的 `27.5%`，而全对 group 比例上升，表明 reward
提高的同时可用学习信号正在饱和。

3200 个样本中有 509 个被标记为 `truncated`（`15.90625%`）。完成样本 reward mean 为
`0.848755`，截断样本仅为 `0.115914`，说明 response 截断会直接损失有效数学 reward。
训练 reward 从窗口 `200–219` 的 `0.71875` 上升到 `380–399` 的 `0.771875`，固定
`eval/gsm8k` 从 `0.7109375` 上升到 `0.7421875`，但 eval 在中间明显波动。因此当前
应描述为“GRPO 确实学习，但有效 group 比例偏低且截断样本 reward 很低，导致增益有限并
出现平台趋势”，而不是“训练完全不收敛”。

当前 rollout 文件没有保存 entropy、KL、advantage 或 clip fraction，不能据此判断 policy
collapse。log-prob 全部有限、样本未被移除、weight version 连续递增，当前没有权重同步
异常的证据。后续应优先比较更长 response 和更多每 prompt 采样数的控制实验。

### 3.7 Checkpoint 状态

最后完整 checkpoint 为 iteration `160`。训练运行到 rollout `179` 附近时，checkpoint
保存因输出文件系统空间耗尽而失败。训练计算已经执行到该阶段，但不完整的后续 checkpoint
不能用于恢复。

有效恢复边界是：

```text
latest complete checkpoint: 160
latest parsed rollout metric: 179
```

## 4. 主要问题和解决方案

### 4.1 MUSA patch、Megatron 和 Transformer Engine 版本不一致

初始环境同时存在 MUSA patch、Megatron 和 Transformer Engine API 不匹配，包括：

- Transformer Engine 缺少 CPU offload API；
- Megatron 缺少 patch 所需 process-group 类型；
- 不同 Megatron 版本的 MoE router 和 tokenizer API 不同时存在。

FSDP2 路径最终采用 Hugging Face 模型直接加载，不依赖 Megatron checkpoint。Megatron
适配被保留为独立工作，不从 FSDP2 的成功推断 Megatron 后端已经可用。

### 4.2 Ray 没有正确映射 MUSA 设备

Ray 默认设置了 `CUDA_VISIBLE_DEVICES`，但没有为每个 actor 正确设置 MUSA mask，曾导致
多个 actor 实际看到同一张 MUSA 卡，或 SGLang 收到不可见的物理 GPU id。

解决方式是在每个 Ray actor 的 runtime environment 中按 rank 设置：

```bash
MUSA_VISIBLE_DEVICES=<physical_gpu_id>
MTHREADS_VISIBLE_DEVICES=<physical_gpu_id>
RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1
```

并分别探测 `ray.get_gpu_ids()`、环境变量和 `torch.musa.current_device()`。

### 4.3 Ray Jobs API 不可用

Ray Core 可以连接，但 Jobs API 端口长期不可用。该问题不能通过 `ray status` 成功来证明
已经恢复，因为 Ray Core 和 Ray Jobs API 是两个不同服务层。

实验改用直接连接 Ray Core：

```python
ray.init(address="auto")
```

然后直接解析 Miles 参数并调用训练入口，避免依赖 Jobs API。

### 4.4 旧版 SGLang 接口差异

当前 SGLang 不提供：

```text
/begin_weight_update
/end_weight_update
```

但核心接口能够成功：

```text
POST /update_weights_from_distributed -> 200 OK
```

兼容逻辑仅允许 begin/end 接口返回 404 时继续执行，并保留 pause generation、flush cache、
实际权重传输和 continue generation。其他 HTTP 错误仍然抛出，避免掩盖真正的权重同步失败。

### 4.5 权重 checker 误报 RoPE cache

SGLang 的 `rotary_emb.cos_sin_cache` 是运行时重建的派生 buffer，不是训练器需要同步的模型
参数。两侧内容不同会造成权重 checker 假阳性。

解决方式是只跳过：

```text
rotary_emb.cos_sin_cache
```

模型参数和其他 buffer 仍继续检查。不能整体关闭权重 checker，否则无法证明训练后的权重
确实到达 rollout 引擎。

### 4.6 FSDP log-prob 路径错误依赖 Megatron

FSDP actor 持有完整 vocabulary logits，但 log-prob 计算曾无条件导入 Megatron fused
cross entropy，导致纯 FSDP 环境在训练阶段失败。

对于完整 vocabulary logits，可以使用：

```python
token_log_probs = torch.nn.functional.log_softmax(
    logits.float(), dim=-1
).gather(
    dim=-1, index=tokens.unsqueeze(-1)
).squeeze(-1)
```

该 fallback 只适用于完整 vocabulary。若使用多 rank vocab-parallel logits，仍应明确要求
相应的分布式实现，不能静默套用本地计算。

### 4.7 保存 checkpoint 时磁盘耗尽

长跑输出位于容量已满的文件系统。训练到 rollout 179 附近时，distributed checkpoint
保存失败；文件系统 inode 仍充足，根因是数据块空间耗尽。

处理原则：

1. checkpoint 和 rollout 文件写入有足够容量的持久化挂载；
2. 启动前检查块空间和 inode；
3. 保留最近完整 checkpoint；
4. 不使用零字节 shard 或缺少 tracker/metadata 的不完整 checkpoint；
5. 监控被删除但仍由进程打开的日志文件。

### 4.8 Ray 资源与进程生命周期

长跑中曾出现大量 `ray::IDLE` worker 和 `STAT=Z` zombie。`ray::IDLE` 只是空闲 worker，
而 zombie 是已经退出但未被父进程回收的进程。Ray 默认 CPU 资源过大时，反复失败会创建
过多 worker，最终触发 `OpenBLAS pthread_create failed` 和
`Resource temporarily unavailable`。

可复现实验应限制 Ray CPU，并限制数值库和 tokenizer 线程：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

ray start --head --node-ip-address=127.0.0.1 \
  --num-cpus=8 --num-gpus=3 --disable-usage-stats
```

线程变量需要传入 Ray actor，不能只在提交命令的 shell 中设置。后台 launcher 存活也不
代表训练已提交；应同时检查 `run_miles_direct.py`、`FSDPTrainRayActor`、`SGLangEngine`
和 `perf <id>:`。容器 PID 1 不会自动回收所有孤儿进程时，清理 zombie 需要重启容器。

### 4.9 Router 503、评估线程池和权重检查

Router 返回以下响应时：

```text
503 Service Unavailable
No available workers (all circuits open or unhealthy)
```

不能据此判断训练正在等待。应分别检查 Router、SGLang engine 的 `/health` 和
`/health_generate`，以及实际 worker 进程。恢复训练的 tokenizer 还可能因为全局 Rayon 线程池
无法初始化而失败；这属于运行时资源问题，不是 GSM8K 数据问题。

初始化阶段的 BF16 checker 可能报告约 `0.000122` 的最大绝对误差。跳过
`rotary_emb.cos_sin_cache` 只针对运行时派生 buffer；临时使用
`--ci-disable-weight-update-checker` 只能用于诊断，不能替代最终权重同步验证。

### 4.10 Resume 和 checkpoint 保留策略

仅配置 `--save` 不会自动恢复已有训练。恢复实验必须同时确认：

```text
--load <existing-checkpoint-root>
latest_checkpointed_iteration.txt = expected step
checkpoint directory has no zero-byte shard
```

每个完整 checkpoint 约 7.3G，长跑应把 checkpoint、rollout 和日志写入容量充足的持久化
挂载，并在启动前检查块空间和 inode。磁盘耗尽可能发生在训练已完成若干 rollout 之后，
因此不能只根据进程是否运行判断实验是否成功。

### 4.11 DCP 保存的 CPU offload 要求

一次恢复验证中，rollout、log-prob 和 actor training 均已完成，但 distributed
checkpoint 在 filesystem writer 阶段因 `assert tensor.is_cpu` 失败。原因是 FSDP state
dict 默认仍可能包含 MUSA tensor；filesystem writer 只接受 CPU tensor。

模型和优化器的 `get_state_dict()` 都需要显式使用：

```python
StateDictOptions(cpu_offload=True)
```

失败保存产生的零字节 shard 或缺少 tracker 的目录不能用于恢复。修复后必须验证实际
执行了一轮训练并成功生成非零大小的 model/optimizer shard，不能只依据外层进程的
`EXIT=0`。

### 4.12 Resume smoke 的 rollout 边界

恢复 checkpoint 后，`start_rollout_id` 是 checkpoint 之后的绝对 rollout id，
`--num-rollout` 也是绝对结束边界，不是“额外执行多少轮”。例如从 `160` 恢复：

```text
跑 1 轮：  --num-rollout 161 --save-interval 1
跑 20 轮： --num-rollout 180 --save-interval 20
```

如果错误地使用 `--num-rollout 20`，训练循环为空，程序可能正常退出，但不会产生
rollout、训练指标或 checkpoint。有效 smoke 必须同时检查 `perf`、`log_probs`、
`actor_train`、`Saved checkpoint`、tracker 和非零 shard；`EXIT=0` 本身不足以证明
训练执行过。

## 5. 关键运行命令

### 5.1 环境预检

```bash
export WORKSPACE=/workspace/host
export MILES_ROOT="$WORKSPACE/miles_woo"
export MODEL_PATH=/root/models/Qwen3-0.6B
export DATA_PATH=/root/datasets/gsm8k/train.parquet

export MILES_HARDWARE_PLATFORM=musa
export MUSA_VISIBLE_DEVICES=0,1,2
export MTHREADS_VISIBLE_DEVICES=0,1,2
export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

cd "$MILES_ROOT"

python3 - <<'PY'
import torch
import torch_musa
import ray
import sglang

print("torch:", torch.__version__)
print("torch_musa:", torch_musa.__version__)
print("musa_available:", torch.musa.is_available())
print("musa_count:", torch.musa.device_count())
print("ray:", ray.__version__)
print("sglang:", getattr(sglang, "__version__", "unknown"))
PY

df -h "$WORKSPACE"
df -i "$WORKSPACE"
```

### 5.2 启动 Ray Core

```bash
ray stop --force || true

ray start \
  --head \
  --node-ip-address=127.0.0.1 \
  --num-gpus=3 \
  --disable-usage-stats

export RAY_ADDRESS=127.0.0.1:6379
ray status
```

### 5.3 GRPO + FSDP2 + SGLang 最小闭环

```bash
cd "$MILES_ROOT"
set -o pipefail

python "$WORKSPACE/run_miles_direct.py" \
  --hf-checkpoint "$MODEL_PATH" \
  --prompt-data "$DATA_PATH" \
  --input-key messages \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout 10 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 2 \
  --rollout-max-response-len 1024 \
  --rollout-temperature 1.0 \
  --global-batch-size 8 \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.0 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus 1 \
  --rollout-num-gpus-per-engine 1 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  --update-weight-transfer-mode broadcast \
  --hardware-platform musa \
  --distributed-backend mccl \
  --train-backend fsdp \
  --attn-implementation eager \
  2>&1 | tee "$WORKSPACE/musa_fsdp2_grpo.log"

echo "TRAIN_EXIT=${PIPESTATUS[0]}"
```

### 5.4 Reward 趋势统计

```bash
LOG="$WORKSPACE/musa_fsdp2_grpo.log"

python3 - "$LOG" <<'PY'
import ast
import re
import statistics
import sys

pattern = re.compile(r"perf\s+(\d+):\s+(\{.*\})")
rows = []

with open(sys.argv[1], errors="replace") as stream:
    for line in stream:
        match = pattern.search(line)
        if not match:
            continue
        try:
            metrics = ast.literal_eval(match.group(2))
        except Exception:
            continue
        reward = metrics.get("rollout/episode_raw_reward")
        if reward is not None:
            rows.append((int(match.group(1)), float(reward)))

rows.sort()
print("metric_rows:", len(rows))

for window in (10, 20, 50):
    if len(rows) >= window:
        values = [reward for _, reward in rows[-window:]]
        print(
            f"last_{window}: mean={statistics.mean(values):.6f}, "
            f"min={min(values):.6f}, max={max(values):.6f}"
        )

print("latest_rollout:", rows[-1] if rows else None)
PY
```

### 5.5 Checkpoint 和磁盘检查

```bash
CHECKPOINT_ROOT="$WORKSPACE/musa_fsdp2_output/checkpoints"

df -h "$WORKSPACE"
df -i "$WORKSPACE"

find "$CHECKPOINT_ROOT" \
  -maxdepth 3 -type f \
  -printf '%p %s bytes\n' 2>/dev/null | sort | tail -n 100

find "$CHECKPOINT_ROOT" \
  -type f -size 0 \
  -print 2>/dev/null

cat "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" 2>/dev/null || true
```

## 6. 证据边界

本实验已经证明：

- MUSA 上原生 FSDP2、DTensor 和 MCCL 基础训练可运行；
- Miles 能消费 SGLang rollout 数据并执行 GRPO advantage；
- FSDP2 actor 能完成 log-prob、backward 和 optimizer update；
- 更新后的权重能够传回 SGLang；
- 该链路能够连续运行 180 个 rollout；
- reward 相比训练初期有可观测提升。

本实验尚未证明：

- 模型已经在固定 GSM8K 验证集上收敛；
- checkpoint 160 能完整恢复 optimizer、LR scheduler、RNG 和 rollout id；
- Megatron 后端或 HF 到 Megatron distributed checkpoint 转换可用；
- 当前容器内手工兼容修改能够在全新环境中自动复现；
- 更大模型、更多节点或长期生产训练的稳定性。

## 7. 后续工作

1. 从 checkpoint 160 执行完整 resume 测试。
2. 使用固定 GSM8K eval 子集，每隔固定 rollout 记录 accuracy 和 reward。
3. 将 checkpoint、日志和 rollout 数据写入容量充足的持久化挂载。
4. 将 SGLang checker 和旧版 weight-update API 兼容逻辑制作成可重放补丁或定制镜像。
5. 增加 MUSA 自动化 smoke，覆盖 Ray 设备映射、FSDP forward/backward 和权重同步。
6. 增加 `n_samples_per_prompt` 或使用 dynamic sampling，减少全对/全错组造成的零 advantage。
7. 将 Megatron 依赖、版本矩阵和 checkpoint 转换作为独立适配任务处理。

## 8. 结论

本次实验已经跨过“FSDP2 是否能在 MUSA 上运行”的基础阶段，完成了 SGLang rollout、
GRPO reward/advantage、FSDP2 训练、MCCL 通信和分布式权重更新构成的完整强化学习闭环。

200–399 主长跑说明训练 reward 和固定 GSM8K eval 均有小幅提升，但 binary reward 下
`66.875%` 的 group 为零方差，且 `15.90625%` 的样本被截断并产生很低 reward；eval 曲线
仍有明显波动。因此当前应将结果定义为“完整闭环跑通、训练有效但有效学习信号不足、增益
有限并出现平台趋势”，而不是“模型已经稳定收敛”。
