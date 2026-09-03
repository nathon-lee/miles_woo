---
title: Miles FSDP2 on MUSA bring-up report
description: Qwen3-0.6B 在摩尔线程 MUSA GPU 上的 Miles FSDP2、SGLang rollout 和 RL 训练闭环适配记录。
---

# Miles FSDP2 on MUSA 适配技术报告

## 1. 报告摘要

本次工作在摩尔线程 MUSA GPU 上完成了 Miles 的 FSDP2 强化学习最小闭环适配。
最终已验证如下数据流能够连续运行：

```text
GSM8K prompt
    -> Miles RolloutManager
    -> SGLang on MUSA (rollout)
    -> reward / GRPO advantage
    -> Miles FSDP2 on MCCL (log-prob + backward + optimizer)
    -> distributed weight update
    -> SGLang receives the new weights
    -> next rollout
```

已完成的核心证据：

- 原生 PyTorch FSDP2 + DTensor + MCCL 两卡 smoke 完成 3 个 step，有限梯度、AdamW
  更新、all-reduce 和 barrier 均成功。
- Miles + Ray + SGLang + Qwen3-0.6B + FSDP2 单轮完整闭环运行成功。
- GSM8K 10 轮训练完成，每轮 rollout、log-prob、actor training 和权重同步均成功。
- 10 轮产生 80 条样本，53 条获得非零奖励，非零奖励率为 `0.6625`。
- rollout 权重版本从 1 递增到 10，每轮
  `mixed_version_ratio=0.0`，未观察到新旧权重混用。
- 最后保存的梯度范数为 `4.013991355895996`，证明确实执行了反向传播和参数更新。
- 最终 10-step 日志中没有 `Traceback` 或 `AssertionError`。
- 主区间 rollout `200–399` 完成 200 个 rollout、3200 条 trajectory，并生成 checkpoint
  `400`；从 checkpoint `400` 继续到 `439` 的 A/B 试验也正常完成。
- 主区间 reward 窗口均值由 `0.718750` 提升到 `0.771875`，但 completed 样本正确率基本不变，
  提升很大程度伴随截断率下降，仍不足以证明数学能力稳定提升。

当前结论必须限定为：**Miles 的 FSDP2 RL 训练闭环已在 MUSA 上跑通；200–399 主长跑显示
reward 有小幅提升，固定 eval 有弱提升，但有效 group 信号不足、曲线波动明显，尚不能声称
模型已经稳定收敛。**
Megatron 路径也尚未完成。

## 2. 实验范围与环境

### 2.1 主要环境

| 项目 | 实验值 |
|---|---|
| 节点 | `worker35066` |
| 容器 | `miles-musa-35066` |
| GPU 物理型号 | Moore Threads X10000（`zjlab-gmi`），8 张卡，单卡 81920 MiB |
| GPU 驱动/设备 | PCIe `1ed5:0400`，内核驱动 `mtgpu` |
| 容器运行时名称 | `torch_musa` 返回 `MTT S5000`（运行时名称映射，不作为物理产品型号） |
| PyTorch | `2.7.1` |
| torch_musa | `2.7.1+d212062` |
| SGLang | `0.5.6.post2` |
| Transformers | `5.12.1` |
| Ray | `2.54.0` |
| Polars | `1.42.1` |
| Transformer Engine | `2.0.0+9c789c6` |
| 训练后端 | FSDP2 |
| 集合通信 | MCCL |
| 模型 | `/root/models/Qwen3-0.6B` |
| 训练集 | `/root/datasets/gsm8k/train.parquet` |
| 补丁目录 | `/workspace/host/megatron-lm-musa-patch-reconstruct` |

这些版本和路径都是当时实验快照；更换容器、SGLang 镜像或 Ray 版本后应重新执行
本报告中的 preflight。

### 2.2 Docker 启动与 MUSA 卡映射

本次实验使用 host bind mount 将宿主机工作目录映射到容器的
`/workspace/host`，并通过设备节点和环境变量暴露/选择 MUSA 卡。`--privileged` 与
`-v /dev:/dev` 负责让容器能够访问 MUSA 驱动设备；真正的卡选择由
`MUSA_VISIBLE_DEVICES` 和 `MTHREADS_VISIBLE_DEVICES` 完成。下面命令可从现有容器读取
镜像名后重建等价容器，避免把具体镜像标签写死：

```bash
IMAGE=$(docker inspect -f '{{.Config.Image}}' miles-musa-35066)

docker run -it --rm \
  --name miles-musa-35066 \
  --privileged \
  --network host \
  --ipc host \
  --shm-size=32g \
  -v /dev:/dev \
  -v /home/mccxadmin:/workspace/host:rw \
  -e MILES_HARDWARE_PLATFORM=musa \
  -e MUSA_VISIBLE_DEVICES=0,1,2 \
  -e MTHREADS_VISIBLE_DEVICES=0,1,2 \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  "$IMAGE" /bin/bash
```

如果节点的 MUSA 运行时要求显式传递设备节点，可在 `docker run` 中额外加入节点上实际
存在的 `--device=/dev/<device>` 参数；不要直接假定设备节点名称。进入容器后仍需执行
设备预检，确认 `torch.musa.device_count()`、`MUSA_VISIBLE_DEVICES` 和
`MTHREADS_VISIBLE_DEVICES` 一致。Ray actor 会在每个 rank 的 runtime environment 中再次
设置单卡 mask，这是进程级映射，不是 Docker 启动时的全局卡选择。

### 2.3 硬件型号识别说明

2026-09-01 在节点上交叉验证后，`zjlab-gmi` 将 8 张物理卡显示为 `X10000`；
`lspci` 确认设备为 Moore Threads `1ed5:0400`，由 `mtgpu` 驱动。同一容器内
`torch_musa.get_device_name()` 和 `get_device_properties()` 返回 `MTT S5000`，这是当前
`torch_musa 2.7.1+d212062` 的运行时设备名称映射。因此，报告以
`zjlab-gmi` 作为物理产品型号，同时保留 torch 返回值供环境复现。

```bash
# 宿主机/容器中的物理设备与驱动确认
lspci -s 03:00.0 -nnk
readlink -f /sys/bus/pci/devices/0000:03:00.0/driver
zjlab-gmi

# 容器内的 torch_musa 设备名称和数量
docker exec miles-musa-35066 python3 -c '
import torch
import torch_musa
print("torch:", torch.__version__)
print("torch_musa:", torch_musa.__version__)
print("musa_count:", torch.musa.device_count())
for i in range(torch.musa.device_count()):
    print(i, torch.musa.get_device_name(i), torch.musa.get_device_properties(i))
'
```

### 2.4 代码分支和状态

本报告对应的代码已同步到 Moore Threads GitLab 仓库：

```text
仓库：https://sh-code.mthreads.com/ai/miles
分支：feat/musa-fsdp2-musa-adaptation
提交：d582a72a688ebad08bdd1a41d91ba663869fe1e4
分支链接：https://sh-code.mthreads.com/ai/miles/-/tree/feat/musa-fsdp2-musa-adaptation
提交链接：https://sh-code.mthreads.com/ai/miles/-/commit/d582a72a688ebad08bdd1a41d91ba663869fe1e4
```

该提交包含本次 MUSA/FSDP2 适配、Ray 设备映射、SGLang 兼容、log-prob fallback 以及相关
测试；本地群空间维护的两份总结文档未包含在该提交中。原先个人工作区中的探索性 commit
链不作为 Moore Threads GitLab 的归档依据。

本次归档覆盖 SGLang 旧版 `/weights_checker` 请求格式兼容、FSDP full-vocabulary
log-prob 的无 Megatron fallback，以及 checker 兼容测试；上述代码和聚焦测试现已提交，
后续只需在目标环境中复核即可。

## 3. 验证分层

本次过程中多次出现“某一层 smoke 成功，但完整框架仍不可用”的情况。后续应保留
如下证据边界：

| 层级 | 证明什么 | 不能证明什么 | 结果 |
|---|---|---|---|
| MUSA tensor smoke | `torch.musa` 可用，基本 tensor 计算成功 | 分布式、FSDP 或 Miles | 通过 |
| 原生 FSDP2 smoke | FSDP2、DTensor、MCCL、backward 和 optimizer 可用 | Ray、SGLang、Miles RL | 通过 |
| `--debug-rollout-only` | Miles + Ray + SGLang 能生成 rollout | FSDP 训练与权重同步 | 通过 |
| 单轮闭环 | rollout、FSDP 训练、权重同步可串联 | 长期稳定性和收敛 | 通过 |
| 10-step GSM8K | 多轮 RL 闭环、权重版本递增、梯度非零 | 统计显著的收敛 | 通过 |
| 固定评估的夜间长跑 | 收敛趋势、checkpoint/resume、长期稳定性 | 已完成 rollout 200–399、固定 eval 和 checkpoint 400→439 resume；统计显著收敛和长期生产稳定性仍待确认 | 部分通过 |
| Megatron 后端 | Megatron checkpoint、训练及同步链路 | 不应从 FSDP 结果推断 | 未完成 |

## 4. 遇到的主要问题与解决方案

### 4.1 MUSA patch、Megatron-LM 和 Transformer Engine 版本不匹配

#### 现象

早期导入 `musa_patch` 时先后出现：

```text
ImportError: cannot import name 'get_fine_grained_offload_handler'
ImportError: cannot import name 'GradFinalizeProcessGroups'
ModuleNotFoundError: No module named 'megatron.core.process_groups_config'
```

#### 根因

- `megatron-lm-musa-patch` 是补丁层，不是完整 Megatron-LM。
- 补丁依赖特定 Megatron Core API，包括
  `GradFinalizeProcessGroups`、`ModelCommProcessGroups` 和
  `topk_routing_with_score_function`。
- 最新 NVIDIA Megatron-LM、旧的 pinned Megatron 以及 Miles 当前依赖的 tokenizer API
  不处在同一个兼容时间窗。
- Transformer Engine 也需要摩尔线程补丁版，普通版本缺少补丁需要的 API。

#### 处理

- 从 `MT-TransformerEngine` 源码安装，最终验证版本为
  `transformer_engine-2.0.0+9c789c6`。
- 使用 `megatron-lm-musa-patch-reconstruct` 分支的补丁结构。
- 通过 API 扫描找到 `Megatron-LM-musa-compat3` 同时包含两组 process-group API
  和 top-k routing API。
- tokenizer API 曾临时从另一份 Megatron checkout 复制，并补齐其依赖后才能导入。

#### 结论

这个方案仅让 Megatron 相关 import 成功，它不是可维护的最终依赖方案。FSDP2 路径后来
改为不依赖 Megatron 的 full-vocabulary log-prob 计算，Megatron 适配应作为独立后续任务。

### 4.2 依赖缺失与版本冲突

#### Polars 缺失

Miles 的 dashboard 模块在参数解析期间导入 Polars，缺失时训练尚未启动就退出：

```text
ModuleNotFoundError: No module named 'polars'
```

由于容器内 PyPI 下载 `polars-runtime-32` 速度极慢，改为在外部机器下载精确 wheel，
然后离线安装：

```bash
python -m pip install --no-index \
  /workspace/host/polars_wheels/polars_runtime_32-1.42.1-*.whl \
  /workspace/host/polars_wheels/polars-1.42.1-*.whl
```

#### Transformers API 缺失

FSDP weight bridge 需要：

```text
transformers.core_model_loading.revert_weight_conversion
```

容器原有 Transformers `4.57.0` 不提供这个模块，升级到 `5.12.1` 后导入成功。
升级同时暴露了 Ray、safetensors、sglang-router、vLLM 和 pyarrow 等依赖冲突。因此不建议
直接无差别执行整仓 `requirements` 覆盖镜像内摩尔线程定制包，而应在验证镜像上锁定
一份已知可用的 constraints/lockfile。

### 4.3 Ray Jobs API 不可用，命令看似“卡死”

#### 现象

- `ray start --head` 显示集群已启动。
- `ray job submit --address=http://127.0.0.1:8265` 长时间无输出。
- `ray job list` 连续 60 次不可用。
- 没有对应的 `train.py` 或 SGLang worker 进程。

#### 根因

Ray Core/GCS 已启动不等于 Dashboard Jobs API 8265 可用。当时容器中的 Jobs API 未正常
就绪，所以 job submit 实际上没有启动训练任务。

#### 解决

绕过 Jobs API，使用 Ray Core 直接连接：

```python
import asyncio
import ray

from miles.utils.arguments import parse_args
from train import train

ray.init(address="auto")
args = parse_args()
asyncio.run(train(args))
ray.shutdown()
```

对应的 helper 保存为：

```text
/workspace/host/run_miles_direct.py
```

### 4.4 Ray 没有正确映射 MUSA 设备

#### 现象

Ray 两个 GPU actor 分别拿到 `ray_gpu_ids=[0]` 和 `[1]`，但两个 actor 的
`MUSA_VISIBLE_DEVICES` 都是 `0`。第二个 SGLang actor 因此报错：

```text
RuntimeError: GPU id 1 is not valid under MUSA_VISIBLE_DEVICES=0
```

#### 根因

当时的 Ray 版本会设置 `CUDA_VISIBLE_DEVICES`，但不会为 MUSA 同步设置
`MUSA_VISIBLE_DEVICES` 和 `MTHREADS_VISIBLE_DEVICES`。

#### 修复

- 训练 actor：在 `miles/ray/train/actor_factory.py` 中根据
  `reordered_gpu_ids[rank]` 为每个 rank 设置物理 MUSA GPU mask。
- rollout engine：在 `miles/ray/rollout/server_group.py` 中为每个 SGLang engine 设置连续
  GPU mask。
- 同时设置 `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1`，防止 Ray 二次覆盖。

修复后，FSDP rank 0/1 分别使用正确设备，不再出现两个 actor 落到同一张 MUSA
GPU 的问题。

### 4.5 FSDP2 full-state 加载在 MUSA 上报 `unknown error`

#### 现象

```text
set_model_state_dict(...)
value.detach().to(device)
RuntimeError: MUSA error: unknown error
```

早期尝试在所有 rank 都从 CPU 加载完整模型，并禁用 `broadcast_from_rank0`，仍会在
MUSA tensor 搬运阶段失败。

#### 最终处理

- rank 0 把完整模型搬到 accelerator device。
- 其他 rank 使用 `to_empty` 分配未初始化的设备 tensor。
- 恢复 `StateDictOptions(..., broadcast_from_rank0=True)`。
- `set_model_state_dict` 后对 buffer 执行显式 `dist.broadcast`。
- 同时确保 Ray 为每个 FSDP rank 设置正确 MUSA mask。

事实证明，之前的 `unknown error` 不能只从 state-dict API 本身分析，Ray 设备可见性错误
也会在延迟报错时表现为 tensor `.to(device)` 失败。

### 4.6 SGLang 版本缺少权重更新会话端点

#### 现象

当前 MUSA SGLang 镜像支持：

```text
POST /update_weights_from_distributed -> 200
```

但不支持：

```text
POST /begin_weight_update -> 404
POST /end_weight_update   -> 404
```

Miles 默认把 404 当作致命错误，导致权重传输尚未开始就退出。

#### 修复

`SGLangEngine.begin_weight_update()` 和 `end_weight_update()` 仅在 HTTP 404 时进入兼容模式：

```text
pause generation
flush cache
skip unsupported begin endpoint
update_weights_from_distributed
skip unsupported end endpoint
continue generation
```

其他 HTTP 错误继续向上抛出，不会被无条件忽略。修复后可观察到多个
`/update_weights_from_distributed` 请求返回 200。

### 4.7 SGLang `DumperConfig` API 缺失

#### 现象

```text
ImportError: cannot import name 'DumperConfig'
```

MUSA SGLang `0.5.6.post2` 提供 `dumper` 和 `_get_rank`，但不提供新版 Miles 期望的
`DumperConfig`。

#### 修复

- 导入时仅对缺失 `DumperConfig` 进行兼容 fallback。
- 未启用 dumper 时返回 `{"enable": False}`。
- 用户实际请求 dumper 功能时仍明确报错，避免静默丢失调试功能。

### 4.8 权重 checker 对派生 RoPE cache 产生假阳性

#### 现象

`weights_checker` 报告：

```text
model.layers.0.self_attn.rotary_emb.cos_sin_cache
max_abs_err=2.0
```

实际参数权重已经同步，失配的是 SGLang 运行时重建或随机化的 RoPE cos/sin cache。
这个 buffer 不是 trainer 需要更新的模型权重。

#### 解决

- Miles e2e 命令在 MUSA 模式下添加：

  ```text
  --check-weight-update-skip-list rotary_emb.cos_sin_cache
  ```

- 当前旧 SGLang server 对 `skip_tensor_list` 请求字段的兼容不完整，因此还在容器内的
  `/home/sglang/python/sglang/srt/utils/weight_checker.py` 中临时跳过后缀为
  `rotary_emb.cos_sin_cache` 的 tensor。
- 400 schema fallback 已纳入 Miles 代码提交：旧 server 拒绝新字段时重试 legacy payload，
  但如果 legacy checker 仍报权重错误，依然让任务失败。

注意：`/home/sglang/...` 是容器内修改。`docker restart` 通常会保留，但删除并重建
容器后会丢失，必须归档到 SGLang fork、补丁文件或新镜像。

### 4.9 FSDP log-prob 不应强制依赖 Megatron

#### 现象

FSDP actor 执行 log-prob 计算时报错：

```text
ModuleNotFoundError: No module named 'megatron'
```

这不是用户环境配置错误。当前运行的是 `--train-backend fsdp`，FSDP actor 持有完整
vocabulary projection，不需要 Megatron vocab-parallel cross entropy。

#### 修复

`compute_log_probs` 增加 dependency-free fallback：

```python
token_log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
return token_log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
```

该 fallback 仅用于完整 vocabulary logits。如果存在多 rank vocab-parallel process group，仍明确
要求 Megatron，避免在真正的 tensor-parallel 场景下得到错误结果。

该修复已纳入 Moore Threads GitLab 代码归档，并参与了成功的 GSM8K 多轮实验；容器内
临时修改仍需迁移到可重放的补丁或镜像。

### 4.10 Log-prob checker 曾暴露真实的模型不对齐

#### 现象

```text
CI check failed: log_probs (-5.5638...) != rollout_log_probs (-0.0980...)
```

进一步检查 64 个 response token 后发现，trainer 和 rollout log-prob 差距不是浮点误差，
平均绝对差约 4.5，最大差异超过 20。

#### 处理原则

- 没有通过删除 `--ci-test` 或长期禁用 log-prob checker 把问题隐藏。
- 检查 response token 对齐、loss mask、prompt/response 切分以及两侧模型权重。
- 恢复 rank-0 full-state broadcast，修正每 rank MUSA 设备 mask，并保持权重 checker。
- 后续运行中 `log_probs` 和 `actor_train` 可正常完成，不再触发该断言。

### 4.11 SGLang router 503 和残留 Ray 进程

#### 现象

```text
503 Service Unavailable
No available workers (all circuits open or unhealthy)
```

多次中断实验后还留下多组 `gcs_server`、`raylet`、`ray::IDLE` 和 SGLang actor。新启动的
router 可能连到已失效的 worker，从而返回 503。

#### 解决

- 每次正式实验前先检查是否已有 `run_miles_direct.py`、SGLang 或 FSDP actor 在运行。
- 无活跃任务时再执行 `ray stop --force`，启动唯一的 Ray head。
- 不把 PPID=1 的 zombie 当作活跃训练进程。Zombie 不能通过 `kill -9` 回收，需要
  PID 1 正确 reap，或使用 `docker --init`/重建容器。
- 启动后同时检查 router、engine `/health` 和实际 `/generate`，而不是只看 Ray 状态。

### 4.12 HF -> Megatron distributed checkpoint 转换卡死/崩溃

#### 现象

转换工具能加载 Qwen3-0.6B，但在保存 torch distributed checkpoint 时出现两类问题：

1. 两个 rank 长时间高 CPU 占用，日志 mtime、输出目录大小和 I/O 计数不再增长。
2. 生成极小的 `.distcp` 文件后在 Go runtime `runtime.sigfwd` 中 segfault。

生成的 checkpoint 仅约 248–264 KiB，且缺少：

```text
latest_checkpointed_iteration.txt
metadata.json
```

这些文件不是可用的 0.6B 模型 checkpoint。

#### 当前结论

- accelerator 抽象已替换转换工具中的 `torch.cuda`/NCCL 硬编码。
- 但 Megatron distributed-checkpoint 保存阶段的稳定性问题仍未解决。
- 当前 FSDP2 路径直接从 Hugging Face checkpoint 加载，不需要先转换 Megatron checkpoint。
- 不应把这个转换失败归因于 FSDP2，也不应把 FSDP2 成功当成 Megatron 可用证据。

### 4.13 Ray 默认 CPU 资源过大导致线程耗尽

多次失败后，容器内曾出现大量 `ray::IDLE` worker；`pids.current` 达到 `9783`，而
`pids.max` 只有 `9830`。随后 `ray status`、`ps` 和 `sleep` 都可能失败，并出现：

```text
OpenBLAS blas_thread_init: pthread_create failed
Resource temporarily unavailable
```

`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1` 等变量只限制数值库线程，不能限制 Ray
根据默认 `255 CPU` 资源创建的 worker 数量。`ray::IDLE` 是空闲 worker，不等于 zombie；
只有 `STAT=Z` 才是真正的 zombie。后续实验应显式限制 Ray 资源：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

ray start --head \
  --node-ip-address=127.0.0.1 \
  --num-cpus=8 \
  --num-gpus=3 \
  --disable-usage-stats
```

这些变量还必须通过 Ray actor 的 `runtime_env`/`--train-env-vars` 传入 worker。若 zombie
已被 PID 1 接管，容器内无法重新回收，应重启容器；不要在活跃任务上重复 `ray start` 或
无条件 `pkill -9 ray`。

### 4.14 后台 launcher 存活不等于训练已提交

后台日志有时只有 `Connected to Ray cluster`、`0.0/3.0 GPU` 和
`Pending Demands: (none)`，但没有 `run_miles_direct.py`、`FSDPTrainRayActor` 或
`SGLangEngine`。`tail -f` 会持续等待，容易被误判为卡死。

首次复现应使用前台 trace，并同时检查 launcher PID、日志 mtime 和训练进程：

```bash
set -o pipefail
bash -x /workspace/host/xx_fsdp2_gsm8k_overnight.sh \
  2>&1 | tee /workspace/host/musa_gsm8k_fsdp2_overnight/debug.log
echo "TRAIN_EXIT=${PIPESTATUS[0]}"
```

只有出现 `FSDPTrainRayActor`、`SGLangEngine` 和 `perf <id>:`，才能确认训练已经提交。

### 4.15 SGLang Router 503 与 worker 健康状态

曾出现：

```text
503 Service Unavailable
No available workers (all circuits open or unhealthy)
```

Router 端口在线不代表 SGLang worker 健康。应分别检查 Router、engine 和进程：

```bash
curl -sS --max-time 5 http://127.0.0.1:3521/health || true
curl -sS --max-time 5 http://127.0.0.1:15000/health || true
curl -sS --max-time 5 http://127.0.0.1:15000/health_generate || true
pgrep -af 'RolloutManager|SGLangEngine|sglang.launch_server' || true
```

只有 engine 健康检查成功且 worker 进程存在时，才继续等待 rollout；否则应先停止旧的
Router、worker 和 Ray，再启动唯一的一组服务。

### 4.16 tokenizer 全局线程池初始化失败

恢复 checkpoint 后，模型加载和权重同步都可能成功，但评估 tokenizer 仍会报：

```text
pyo3_runtime.PanicException: The global thread pool has not been initialized
ThreadPoolBuildError: Resource temporarily unavailable
```

这表示线程额度不足，不是 GSM8K 数据损坏。除 Ray 的 `--num-cpus` 外，还需将
`RAYON_NUM_THREADS=1` 和 `TOKENIZERS_PARALLELISM=false` 传入训练 actor。

### 4.17 BF16 权重 checker 误差与临时开关边界

初始化阶段曾因 checker 返回 HTTP 400 退出。参数 shape 和 dtype 一致，典型差异为：

```text
max_abs_err=0.0001220703125
mean_abs_err≈1e-6
```

这属于 MUSA/BF16 权重传输后的舍入误差，不同于 RoPE 派生 cache 的假阳性。
`--ci-disable-weight-update-checker` 只能用于临时诊断，让训练继续观察；不能作为最终
权重同步正确性的证明。正式适配仍需单独定义有边界的 BF16 容差策略。

### 4.18 Resume 参数和 checkpoint 空间必须一起验证

脚本只有 `--save` 而没有 `--load` 时，即使目录中存在 iteration 160，也会从头训练。
恢复前至少检查：

```bash
grep -nE -- '--load|--save|--save-interval|--num-rollout' \
  /workspace/host/xx_fsdp2_gsm8k_overnight.sh
cat /workspace/host/musa_gsm8k_fsdp2_overnight/20260831-220654/checkpoints/latest_checkpointed_iteration.txt
df -h /workspace/host
```

每个完整 checkpoint 约 7.3G。旧 checkpoint、备份和被删除但仍打开的日志应归档到容量
充足的 `/data` 挂载；不能使用零字节 shard 或缺少 tracker 的不完整目录恢复。

### 4.19 FSDP DCP 保存必须显式 CPU offload

一次长跑在 rollout `179` 已完成 rollout、log-prob 和 actor training，但在保存
iteration `180` 时失败：

```text
torch.distributed.checkpoint ... filesystem.py
assert tensor.is_cpu
CheckpointException
```

这不是显存或磁盘空间错误，而是 filesystem storage writer 收到了 MUSA tensor。Miles
的 `ModelState.state_dict()` 和 `OptimizerState.state_dict()` 原先调用
`get_state_dict()` 时使用默认的 `cpu_offload=False`，因此保存阶段仍可能保留设备 tensor。

修复方式是在两个 state-dict wrapper 中显式设置：

```python
from torch.distributed.checkpoint.state_dict import StateDictOptions

options = StateDictOptions(cpu_offload=True)
get_state_dict(..., options=options)
```

模型和优化器都必须设置该选项。修复后应至少验证一次真正的训练和 checkpoint 保存，不能
只根据外层脚本的 `EXIT=0` 判定成功。零字节 `.distcp`、缺少 tracker 或只有目录没有
`Saved checkpoint` 日志的结果均视为不完整 checkpoint。

### 4.20 Resume smoke 的 `--num-rollout` 是绝对上限

从 iteration `160` 恢复时，checkpoint metadata 会将 `start_rollout_id` 设置为 `160`。
训练循环的边界是：

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
```

因此设置 `--num-rollout 20` 并不是“在 160 后再跑 20 轮”，而是得到空区间
`range(160, 20)`。进程仍可能以 `EXIT=0` 退出，但不会生成 rollout、`perf`、
`actor_train` 或 checkpoint。这次 smoke 就出现了该假阳性：日志只有模型加载和初始
权重同步，目标 run 目录没有 tracker。

恢复验证应按绝对 rollout 上限设置参数：

```text
从 160 恢复、跑 1 轮：  --num-rollout 161 --save-interval 1
从 160 恢复、跑 20 轮： --num-rollout 180 --save-interval 20
```

验证日志必须同时出现 `perf <id>:`、`log_probs`、`actor_train` 和 `Saved checkpoint`，
并检查 tracker 与非零大小 shard。诊断目录时不要让 `set -e` 在缺失 tracker 处提前退出，
否则会漏掉真正的日志和目录证据；可使用 `set +e` 完成完整扫描。

## 5. 主要代码修改

| 文件 | 修改目的 |
|---|---|
| `miles/_bootstrap.py` | 在导入 torch 前选择 MUSA 并按需加载补丁 |
| `miles/utils/accelerator.py` | 封装 device、synchronize、empty-cache、memory、backend 等差异 |
| `miles/ray/train/actor_factory.py` | 把 Ray bundle 的物理 GPU ID 映射为每 rank MUSA mask |
| `miles/ray/rollout/server_group.py` | 为 SGLang engine 设置 MUSA/MTHREADS visible devices |
| `miles/backends/fsdp_utils/*` | 用 accelerator 抽象替换 CUDA 硬编码，适配 DTensor、checkpoint、precision 和权重更新 |
| `miles/backends/sglang_utils/sglang_engine.py` | 兼容旧 SGLang 的权重更新端点和 checker schema |
| `miles/utils/dumper_utils.py` | 兼容没有 `DumperConfig` 的 SGLang |
| `miles/utils/arguments.py` | 兼容某些旧 SGLang 参数未注册的情况 |
| `miles/backends/training_utils/loss_hub/math_utils.py` | FSDP full-vocabulary log-prob 无 Megatron fallback |
| `tools/convert_hf_to_torch_dist.py` | 将 CUDA/NCCL 直接调用替换为 accelerator/MCCL，但转换仍未跑通 |
| `tests/e2e/fsdp/test_qwen3_0.6B_megatron_fsdp_align.py` | 添加 MUSA 参数和 RoPE derived-buffer skip |

## 6. 重要运行命令

### 6.1 环境导入 preflight

```bash
cd /workspace/host/miles_woo

export MILES_HARDWARE_PLATFORM=musa
export MUSA_VISIBLE_DEVICES=0,1,2
export MTHREADS_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2
export MUSA_PATCH_PATH=/workspace/host/megatron-lm-musa-patch-reconstruct
export PYTHONPATH=/workspace/host/miles_woo:/workspace/host/megatron-lm-musa-patch-reconstruct:/root/Megatron-LM
export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

python3 - <<'PY'
import torch
import torch_musa
import transformers
import sglang
import ray
import polars

from miles.utils.arguments import parse_args

print("torch:", torch.__version__)
print("torch_musa:", torch_musa.__version__)
print("transformers:", transformers.__version__)
print("sglang:", getattr(sglang, "__version__", "unknown"))
print("ray:", ray.__version__)
print("polars:", polars.__version__)
print("musa_available:", torch.musa.is_available())
print("musa_count:", torch.musa.device_count())
print("MILES_MUSA_IMPORT_OK")
PY
```

### 6.2 原生 FSDP2/MCCL smoke

```bash
export MUSA_VISIBLE_DEVICES=0,1
export MTHREADS_VISIBLE_DEVICES=0,1

set -o pipefail
torchrun \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29620 \
  /workspace/host/musa_fsdp2_smoke.py \
  2>&1 | tee /workspace/host/musa_fsdp2_smoke.log

echo "FSDP2_EXIT=${PIPESTATUS[0]}"
```

预期标记：

```text
FSDP2_MUSA_SMOKE_OK
FSDP2_EXIT=0
```

### 6.3 启动直连 Ray Core

```bash
NODE_IP=$(hostname -I | awk '{print $1}')

ray start \
  --head \
  --node-ip-address="${NODE_IP}" \
  --port=6379 \
  --num-gpus=3 \
  --include-dashboard=false \
  --disable-usage-stats

export RAY_ADDRESS=127.0.0.1:6379
ray status
```

启动训练前应看到 Ray 报告 3 个 GPU resource，并且不存在另一个活跃的
`run_miles_direct.py`。

### 6.4 GSM8K 10-step 完整闭环

以下是根据当时生效参数整理的等价复现命令：

```bash
cd /workspace/host/miles_woo
set -o pipefail

python /workspace/host/run_miles_direct.py \
  --hf-checkpoint /root/models/Qwen3-0.6B \
  --prompt-data /root/datasets/gsm8k/train.parquet \
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
  --sglang-disable-cuda-graph \
  --sglang-chunked-prefill-size 128 \
  --sglang-mem-fraction-static 0.03 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  --update-weight-transfer-mode broadcast \
  --hardware-platform musa \
  --distributed-backend mccl \
  --train-backend fsdp \
  --attn-implementation eager \
  --ci-test \
  --check-weight-update-skip-list rotary_emb.cos_sin_cache \
  --ci-save-grad-norm /workspace/host/musa_gsm8k_grad_norm.pt \
  --save-debug-rollout-data '/workspace/host/musa_gsm8k_rollout_{rollout_id}.pt' \
  2>&1 | tee /workspace/host/miles_musa_fsdp2_gsm8k_10step.log

status=${PIPESTATUS[0]}
echo "GSM8K_FSDP2_EXIT=${status}" | tee -a \
  /workspace/host/miles_musa_fsdp2_gsm8k_10step.log
exit "${status}"
```

### 6.5 夜间后台长跑

建议使用已生成的夜间脚本，配置为 200 个 rollout、每 prompt 4 个 sample、
每 20 轮固定评估和 checkpoint：

```text
--num-rollout 200
--rollout-batch-size 4
--n-samples-per-prompt 4
--global-batch-size 16
--eval-interval 20
--eval-prompt-data gsm8k /root/datasets/gsm8k/test-128-seed20260831.parquet
--n-samples-per-eval-prompt 1
--save /workspace/host/musa_gsm8k_fsdp2_overnight/latest/checkpoints
--save-interval 20
```

固定 128 题评估集使用固定 seed 创建：

```bash
python3 - <<'PY'
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

src = Path("/root/datasets/gsm8k/test.parquet")
dst = Path("/root/datasets/gsm8k/test-128-seed20260831.parquet")

table = pq.read_table(src)
indices = random.Random(20260831).sample(range(table.num_rows), 128)
pq.write_table(table.take(pa.array(indices, type=pa.int64())), dst)
print("eval_rows=128")
print("output=", dst)
PY
```

```bash
nohup bash /workspace/host/xx_fsdp2_gsm8k_overnight.sh \
  > /workspace/host/musa_gsm8k_fsdp2_overnight.nohup.log \
  2>&1 < /dev/null &

OVERNIGHT_PID=$!
echo "${OVERNIGHT_PID}" | tee /workspace/host/musa_gsm8k_fsdp2_overnight.pid
disown "${OVERNIGHT_PID}" 2>/dev/null || true
```

观察日志：

```bash
tail -f /workspace/host/musa_gsm8k_fsdp2_overnight/latest/train.log
```

筛选评估、训练、checkpoint 和错误：

```bash
grep -nE \
  'eval/gsm8k|episode_raw_reward|actor_train: 100%|weight_version|Saved checkpoint|Traceback|AssertionError|OVERNIGHT_EXIT=' \
  /workspace/host/musa_gsm8k_fsdp2_overnight/latest/train.log \
  | tail -n 200
```

### 6.6 判断“慢”还是“卡死”

不应只根据终端一段时间没有新文本就判定卡死。连续两次采样：

```bash
ps -eo pid,ppid,stat,etime,time,%cpu,%mem,wchan:32,cmd \
  | grep -E 'run_miles_direct|SGLangEngine|FSDPTrainRayActor|RolloutManager' \
  | grep -v grep

stat -c 'log size=%s mtime=%y' \
  /workspace/host/musa_gsm8k_fsdp2_overnight/latest/train.log
du -sh /workspace/host/musa_gsm8k_fsdp2_overnight/latest

for pid in $(pgrep -f 'run_miles_direct|SGLangEngine|FSDPTrainRayActor'); do
  echo "===== PID ${pid} IO ====="
  cat "/proc/${pid}/io" 2>/dev/null || true
done
```

如果 CPU time 持续增长、GPU 有利用率、`/health` 返回 200 且正在评估/生成，不应立即杀掉。
如果多次采样中日志 mtime、输出大小和 I/O 计数都不变，才应进一步抓取 Python/C++
stack。

## 7. GSM8K 实验结果

### 7.1 每轮奖励

| Rollout | Samples | Reward mean | Rewarded | Boxed | Truncated |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 1.0000 | 8 | 8 | 0 |
| 1 | 8 | 0.8750 | 7 | 7 | 1 |
| 2 | 8 | 0.5000 | 4 | 6 | 3 |
| 3 | 8 | 0.6250 | 5 | 6 | 2 |
| 4 | 8 | 0.3750 | 3 | 6 | 2 |
| 5 | 8 | 0.7500 | 6 | 6 | 3 |
| 6 | 8 | 0.5000 | 4 | 5 | 3 |
| 7 | 8 | 0.6250 | 5 | 5 | 3 |
| 8 | 8 | 0.8750 | 7 | 8 | 0 |
| 9 | 8 | 0.5000 | 4 | 6 | 3 |

汇总：

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

### 7.2 结果解读

- 闭环正确性：连续 10 轮没有中断，actor training 和权重更新均完成。
- 权重时序：每轮只使用单一版本的权重，没有 mixed-version rollout。
- 学习信号：梯度范数非零，但 `n_samples_per_prompt=2` 会产生较多全对/全错组，
  这些组的 GRPO advantage 为零。
- 收敛性：10-step 仅用于验证闭环，前后窗口 reward 不能代表当前 200–399 主长跑的
  收敛趋势；该短跑没有足够数据判断 reward 上升或下降。
- 比较方法：训练 prompt 每轮都不同，不能用训练 batch reward 直接代替固定评估集
  accuracy。

### 7.3 主长跑 200–399 的收敛趋势

当前主分析区间覆盖 rollout `200`–`399`，共 200 个 rollout、3200 条 trajectory、800 个
四样本 group，checkpoint tracker 最终为 `400`。此前 0–179 的数据属于阶段性历史结果；
期间出现的磁盘满和不完整 checkpoint 也已与后续有效 checkpoint 分开处理。

Reward 窗口统计如下：

| 窗口 | Reward mean | Min | Max |
|---|---:|---:|---:|
| 200–219 | `0.718750` | `0.437500` | `1.000000` |
| 240–259 | `0.675000` | `0.187500` | `1.000000` |
| 280–299 | `0.743750` | `0.500000` | `1.000000` |
| 320–339 | `0.759375` | `0.500000` | `1.000000` |
| 360–379 | `0.746875` | `0.437500` | `1.000000` |
| 380–399 | `0.771875` | `0.500000` | `0.937500` |

这组数据支持以下有限结论：

- 窗口均值由 `0.718750` 升至 `0.771875`，但 completed 样本 reward 均值约由 `0.834559`
  变为 `0.832765`，基本不变；truncated ratio 则由 `15.0%` 降至 `8.4375%`。
- 800 个 group 中 `66.875%` 为全对或全错，只有约三分之一 group 能提供有效相对信号。
- 因此当前结论是“输出完成度改善、训练有效但数学正确率增益有限并接近平台”，不能写成
  “模型已经收敛”。训练 batch reward 不能替代固定验证集 accuracy。

### 7.4 固定 GSM8K eval：历史 160 → 200 与主长跑

随后使用同一批 128 条 GSM8K 验证样本，对 checkpoint `160` 和 checkpoint `200` 做了
固定评估：

| Checkpoint | Eval samples | `eval/gsm8k` | 等价正确数 |
|---:|---:|---:|---:|
| 160 | 128 | `0.703125` | 90/128 |
| 200 | 128 | `0.7265625` | 93/128 |

checkpoint `200` 的评估进程正常完成（`EVAL_AT_200_RETRY_EXIT=0`），平均 response
length 为 `546.7`，`eval/truncated_ratio` 为 `0.1171875`。相较 checkpoint `160`，
固定 eval 提升 `3/128`，即 `+0.0234375`（+2.34 个百分点）。这与长跑 reward 约 `0.70`
的平台趋势一致，但目前只有一个对比区间，不能单独据此宣称模型已经收敛。

需要注意，日志中的 `eval 0` 是独立 eval-only 进程内部的 rollout ID，不代表加载了第
0 步模型；本次实际评估模型来自 checkpoint `200`。同理，`weight_version=1.0` 是该
评估进程的内部权重版本，不是训练 checkpoint 编号。

主长跑已经记录 rollout `200`–`399` 的固定 eval：`0.7109375`、`0.71875`、`0.7578125`、
`0.7265625`、`0.71875`、`0.703125`、`0.703125`、`0.7109375`、`0.7421875`、`0.734375`、
`0.7421875`。该序列有弱正向趋势但波动明显；checkpoint `439` 的两次随机 eval 为
`0.71875` 和 `0.703125`，说明必须使用确定性采样或多次重复评测。

### 7.5 NVIDIA 官方对照数据边界

截至本报告，材料中没有 NVIDIA 官方团队在相同 Miles、Qwen3-0.6B、GSM8K 和 FSDP2
配置下发布或运行的对照结果。报告中的 MUSA 指标来自本项目在 Moore Threads 节点上的
实测；代码仓库中的 CUDA 运行配置和社区讨论只能作为后续对照实验的方案，不能写成
“NVIDIA 官方结果”。因此，当前不能用本报告证明 MUSA 与 NVIDIA 的收敛速度或最终精度
等价。若需要硬件归因，应在 CUDA 环境上使用相同模型、数据、采样参数、固定 eval 集和
checkpoint 起点，独立完成 CUDA FSDP2 基线后再比较。

#### 7.5.1 已找到的公开相关证据

这里的“没有 NVIDIA 官方同配置结果”不等于“互联网上没有任何 NVIDIA/Qwen3/GSM8K
数据”。截至本报告检索，能确认的公开资料如下：

- Qwen 官方仓库的 issue `#1723` 记录了 Qwen3-0.6B-Base 的 GSM8K 复现差异：原始报告值
  为 `59.59%`，一组 CUDA/BF16、8-shot、greedy 的复现为 `49.81%`（flexible extract）和
  `49.73%`（strict match）。这说明评测协议、模型变体和解码设置会显著改变分数，不能
  直接把 `59.59%` 当作本实验的 baseline。来源：[Qwen3 issue #1723](https://github.com/QwenLM/Qwen3/issues/1723)。
- Slime issue `#385` 和 `#1072` 是公开的 rollout/等待性能记录，作者也提示结果存在不确定性；
  它们主要用于说明部署和等待路径问题，不是 FSDP 收敛精度证据。来源：
  [Slime issue #385](https://github.com/THUDM/slime/issues/385)、
  [Slime issue #1072](https://github.com/THUDM/slime/issues/1072)。
- Miles 上游 issue `#1499` 明确提出建立 Miles/VERL × FSDP/Megatron 的可控 benchmark，并指出
  当时公开数据主要来自 Slime，混合了 rollout、等待时间和 colocate/disaggregated 影响，
  缺少严格的 FSDP 对照。该条属于第三方上游公开资料，保留其原始 GitHub 链接；本项目代码
  归档和提交信息以 Moore Threads GitLab 为准。来源：[Miles issue #1499](https://github.com/radixark/miles/issues/1499)。
- Qwen 官方模型卡建议 thinking mode 使用 `temperature=0.6, top_p=0.95, top_k=20`，并
  明确不建议 greedy decoding，因为可能导致性能下降或无休止重复。这个建议也说明，若将
  Qwen 官方分数与本实验比较，必须先对齐 thinking 开关和采样参数。来源：[Qwen3-0.6B
  model card](https://huggingface.co/Qwen/Qwen3-0.6B)。

上述资料能够支持“公开 NVIDIA 环境中存在性能和复现差异”的判断，但不能支持“某个 NVIDIA
GPU 上的 Miles FSDP2 已经达到某个固定收敛精度”。因此，公开资料仍只能作为实验设计和
问题定位的参考；硬件归因必须由同配置 CUDA FSDP2 实测完成。

### 7.5.2 checkpoint 439 的评测可重复性

对同一个 checkpoint `439` 做了两次独立的 128 条 GSM8K eval。两次日志都明确加载了
`iter_0000439/model`、optimizer 和 LR scheduler，说明评测使用的是同一训练 checkpoint：

| Eval 运行 | 正确数 | `eval/gsm8k` | completed | truncated |
|---|---:|---:|---:|---:|
| `20260902-171924` | 92/128 | `0.71875` | 116 | 12 |
| `20260902-173226` | 90/128 | `0.703125` | 113 | 15 |

两次评测的 prompt 和 label 均为 `128/128` 相同，但生成 response 只有 `8/128` 相同，
reward 有 `116/128` 相同。SGLang 日志显示实际默认 chat sampling params 为
`temperature=0.6, top_k=20, top_p=0.95`。因此这两个分数的差异应视为随机采样和截断造成的
评测方差，而不是 checkpoint 439 在两次运行之间发生了训练变化。

评测文件中的 `rollout_id=0` 是独立 eval 进程的本地编号，`weight_versions=1` 是该进程的
内部权重版本；二者都不能替代日志中的 `iter_0000439` checkpoint 证据。后续固定评测必须
显式设置 `--eval-temperature 0 --eval-top-k 1 --eval-top-p 1`，并连续重复至少 3 次后再
报告均值、标准差和截断率。

### 7.6 主长跑 200–399 的 reward signal 诊断

对主长跑 `20260901-191717` 的 200 个训练 rollout 进行逐文件分析，共得到 3200 个样本、
800 个四样本 group。reward 是二值 `0/1`，总体 reward mean 为 `0.7321875`，其中
`reward=1` 占 `73.21875%`。

| Group 类型 | 数量 | 比例 |
|---|---:|---:|
| `0000`（全错） | 96 | `12.00%` |
| `1111`（全对） | 439 | `54.875%` |
| `mixed`（组内有对有错） | 265 | `33.125%` |

因此 `66.875%` 的 group 为零方差，不能提供有效的 GRPO 相对 advantage；只有约三分之一
的 group 能区分同一 prompt 下的不同回答。后期 `mixed` group 从窗口 `200–219` 的
`30/80` 降至 `380–399` 的 `22/80`，而 `1111` group 从 `41/80` 增至 `49/80`，说明
训练 reward 上升的同时，组内学习信号正在饱和。

response 截断同样会显著削弱 reward：3200 个样本中有 509 个为 `truncated`（`15.90625%`）。
完成样本 reward mean 为 `0.848755`，截断样本仅为 `0.115914`。主长跑的训练 reward
窗口从 `200–219` 的 `0.71875` 上升到 `380–399` 的 `0.771875`，固定 eval 从
`0.7109375` 上升到 `0.7421875`，但中间在 `0.703125`–`0.7578125` 间波动。因此当前
结论是“GRPO 确实学习，但有效 group 比例偏低且截断样本 reward 很低，导致增益有限并
出现平台趋势”，不是“FSDP2/MUSA pipeline 失效”。

当前没有 entropy、KL、advantage 或 clip fraction 的保存数据，不能据此判断 policy collapse；
log-prob 全部有限、样本未被移除、weight version 连续递增，暂不支持权重同步异常的判断。
后续应优先做 response length 和采样数的控制实验，再决定是否调整学习率。

## 8. 容器重启和持久化

| 位置 | `docker restart` | 删除重建容器 | 建议 |
|---|---|---|---|
| `/workspace/host/miles_woo` | 通常保留 | 如为 host bind mount 则保留 | 所有 Miles 修改必须 Git 归档 |
| `/workspace/host/*.log` / `*.pt` | 通常保留 | host mount 下保留 | 长跑输出统一放在这里 |
| `/home/sglang/python/...` | 同一容器 restart 通常保留 | 会丢失 | 必须制作 SGLang 补丁或新镜像 |
| `/usr/local/lib/python...` | 同一容器 restart 通常保留 | 会丢失 | 导出 lockfile/wheelhouse/新镜像 |
| `/root/models` / `/root/datasets` | 取决于 mount | 可能丢失 | 启动前执行 `findmnt` 和文件 preflight |

## 9. 当前未完成项

### 9.1 FSDP2 路径

1. 把容器内 `weight_checker.py` 的 RoPE cache 修改移到可重放的 SGLang patch 或定制镜像。
2. 使用确定性采样参数重复 checkpoint `200`、`400` 和 `439` 的固定 GSM8K eval，记录均值、
   标准差和截断率，避免把随机采样波动误判为模型收敛或退化。
3. checkpoint `400→439` 的 model、optimizer、scheduler 和 rollout 状态已验证可继续运行；
   仍需补充 RNG 逐位可复现性验证。
4. 将长跑输出和 checkpoint 放到有足够空间且可持久化的 host/data 挂载，避免在保存阶段因
   `/workspace/host`（`/home` 文件系统）满而中断。
5. 增加至少一个自动化 MUSA CI smoke，覆盖 Ray GPU mapping、FSDP forward/backward 和一次
   SGLang weight update。
6. 对 `n_samples_per_prompt=4` 与 `8` 做同 checkpoint、同数据的 A/B，确认 mixed group 比例
   和有效 advantage 是否提升，避免大量 GRPO zero-advantage batch。

### 9.2 Megatron 路径

1. 明确锁定一组同时满足 Miles、MUSA patch、Transformer Engine 和 tokenizer API 的
   Megatron commit，不再手工拼接 checkout。
2. 对 HF -> torch distributed checkpoint 的崩溃抓取 Python/C++ stack，定位 distributed-checkpoint
   planner/writer 还是 MCCL/MUSA runtime 问题。
3. 验证 checkpoint 完整性：模型大小、tracker、metadata、非零 shard 以及可重新加载。
4. 在 Megatron 训练前先做两 rank MCCL collective 和最小 Megatron forward/backward，不直接启动
   完整 RL 任务。
5. 最后再执行 Megatron/FSDP 对齐测试：首轮 log-prob、gradient norm、权重同步和多轮
   rollout 都必须通过。

## 10. 验收标准

后续可以用下表判断是否从“跑通”进入“可用”：

| 验收项 | 标准 |
|---|---|
| 启动 | 无人工修改容器后仍可从锁定镜像启动 |
| 正确性 | `--ci-test` 的 log-prob 和权重 checker 不被整体关闭 |
| 训练 | 至少 100 个 rollout 无 Traceback/AssertionError/MCCL timeout |
| 权重时序 | 每轮 `mixed_version_ratio=0.0`，版本单调递增 |
| 梯度 | 非零且有限，无 NaN/Inf |
| 收敛 | 固定 eval 集的 reward/accuracy 相对 step-0 baseline 有可重复提升 |
| 持久化 | checkpoint 可保存、可加载、可从正确 rollout id 继续 |
| 可复现 | 新容器不依赖手工改 `/home/sglang` 或临时复制 Megatron 文件 |

## 11. 最终结论

本次适配已经跨过了“原生 FSDP2 能否在 MUSA 上运行”的阶段，完成了 Miles、Ray、
SGLang、FSDP2、MCCL 和 Qwen3-0.6B 的多轮 RL 闭环。200–399 主长跑进一步证明训练 reward
和固定 GSM8K eval 均有小幅提升，但 binary reward 下 `66.875%` 的 group 为零方差，且
`15.90625%` 的样本被截断并产生很低 reward，当前应定义为“训练有效但有效学习信号不足、
增益有限并出现平台趋势”，不能声称已经稳定收敛。

当前仍属于 bring-up/correctness 与收敛诊断阶段；还需要补充 advantage、entropy、KL、
clip ratio、确定性重复 eval，以及同配置 NVIDIA 对照证据。

## 附录 A：2026-09-03 补充（新增）

本节集中回答评审中提出的配置、数据、收敛和代码审查问题。这里的“样本”明确指生成的
trajectory；`rollout-batch-size` 指 prompt 数，不能与生成结果数混为一谈。

### A.1 Rollout、生成结果和训练 batch

每个 rollout 的生成 trajectory 数为：

```text
rollout_batch_size × n_samples_per_prompt
```

| 实验 | prompt/rollout | 每个 prompt 生成数 | trajectory/rollout | 总 trajectory |
|---|---:|---:|---:|---:|
| 10-rollout smoke | 4 | 2 | 8 | 80 |
| 主长跑 200–399 | 4 | 4 | 16 | 3200 |
| n=8 A/B 试验 | 4 | 8 | 32 | 按实际 rollout 数计 |

训练侧没有显式传入 `--micro-batch-size`，使用默认值 `1`。固定 batch 的关系为：

```text
local_batch_size = global_batch_size / data_parallel_size
num_microbatches = local_batch_size / micro_batch_size
```

因此 10-rollout 配置 `global_batch_size=8`、DP=2 时，每卡每个训练 step 为 4 个
micro-batch；主长跑 `global_batch_size=16`、DP=2 时，每卡为 8 个 micro-batch，均为每个
micro-batch 1 条 trajectory。log-prob 和 actor training 共用该切分。

推理侧 SGLang 使用 continuous batching，调度器动态合并请求；本实验没有固定的推理
micro-batch。`sglang-chunked-prefill-size` 是 token 分块参数，不是样本 micro-batch。

### A.2 Reward 定义与训练数据

主实验使用 `--rm-type math`。GSM8K parquet 的核心字段是 `messages` 和 `label`：前者
包含 system/user 对话，后者是标准答案字符串。数学 reward 的判定流程是：提取回答中最后
一个 `\boxed{...}`，与 `label` 做归一化和数学等价比较；通过为 `1`，否则为 `0`。

例如 label 为 `72` 时：

```text
“所以答案是 \boxed{72}”       -> reward=1
“所以答案是 72”               -> reward=0（无法按 math verifier 提取）
“所以答案是 \boxed{70}”       -> reward=0
在输出 boxed 答案前被截断       -> reward=0
```

推理过程可以包含多步解释，但最终答案必须可提取且与 label 数学等价。实现位于
`miles/rollout/rm_hub/__init__.py` 的 `math` 分支以及 `math_utils.py` 的答案提取/判定函数。

### A.3 Reward 波动的证据解释

200–399 共 3200 条 trajectory、800 个四样本 group：`0000` group 占 12%，`1111` 占
54.875%，mixed 仅 33.125%。因此 66.875% 的 group 没有 reward 差异，不能提供有效的
GRPO 相对 advantage。另有 509 条 trajectory 被截断（15.90625%）；完成样本 reward 均值
约 0.8488，截断样本约 0.1159。

训练 reward 窗口均值从 `200–219` 的 0.71875 上升到 `380–399` 的 0.771875，但固定
128 题 eval 在 0.703125–0.7578125 间波动。这与二值 reward、prompt 难度变化、组内零方差、
截断和有限 eval 集共同相符。当前没有完整保存 entropy、KL、advantage 和 clip fraction，
因此不能据此断言 policy collapse 或 reward hacking。

### A.4 原生 FSDP2 smoke 复现命令

```bash
export MUSA_VISIBLE_DEVICES=0,1
export MTHREADS_VISIBLE_DEVICES=0,1
set -o pipefail
torchrun --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29620 \
  /workspace/host/musa_fsdp2_smoke.py \
  2>&1 | tee /workspace/host/musa_fsdp2_smoke.log
echo "FSDP2_EXIT=${PIPESTATUS[0]}"
```

成功标志为 `FSDP2_MUSA_SMOKE_OK` 和 `FSDP2_EXIT=0`。该命令只证明原生 FSDP2、DTensor、
MCCL、forward/backward、AdamW、all-reduce 和 barrier，不证明 Miles RL 已收敛。

### A.5 NVIDIA 对照和 Miles 改动 review 边界

目前没有与本实验完全相同的公开 NVIDIA reward/convergence 曲线。因此 NVIDIA 对照必须
使用相同 checkpoint、GSM8K 固定 128 题、rollout/micro/global batch、学习率、response
length、reward parser 和确定性 eval 参数（`--eval-temperature 0 --eval-top-k 1 --eval-top-p 1`）。
报告暂不填入未经实测的 NVIDIA 数字。

本次 Miles 改动的 review 按六个区域进行：MUSA accelerator/bootstrap、Ray rank 设备映射、
FSDP2 精度与 checkpoint、无 Megatron 的 full-vocabulary log-prob fallback、SGLang 旧接口/权重
更新兼容、测试与错误处理。重点检查 CUDA 默认路径是否保持不变、兼容逻辑是否足够窄，以及
Megatron/HF checkpoint 转换等未验证路径是否被误称为已支持。

代码归档统一对应 Moore Threads GitLab commit：

```text
分支：feat/musa-fsdp2-musa-adaptation
commit：d582a72a688ebad08bdd1a41d91ba663869fe1e4
```

### A.6 GRPO 数据、reward 与参数配比说明（补充）

GSM8K 数据集为每条 prompt 提供一个标准答案 `label`。SGLang 根据采样参数为每个 prompt
生成若干回答；`n_samples_per_prompt` 表示回答数量，温度参数独立控制采样随机性，二者
不是同一个概念。例如相同的 `temperature=0.6` 下，将 `n_samples_per_prompt` 从 4 改为
8 只是生成更多回答，并不等价于调高温度；温度过低时，新增回答可能仍然高度重复。

本实验的 `--rm-type math` 是规则 verifier，不是单独训练的 value function，也没有额外的
critic。它提取回答最后一个 `\boxed{...}` 中的答案，与 GSM8K `label` 做归一化/数学等价
比较：正确为 reward `1`，错误、缺少可解析 boxed 答案或生成被截断导致答案不可解析时为
reward `0`。因此一个 rollout 的 `episode_raw_reward` 是该 rollout 所有 trajectory 的
二值 reward 平均值，等价于“正确回答数/回答总数”。GRPO 随后在同一 prompt 的回答组内
计算相对 advantage；`[1,1,0,0]` 能提供有效方向，而 `[1,1,1,1]` 或 `[0,0,0,0]`
没有组内差异，advantage 信号接近零。

参数配比的作用应分别理解：

| 参数 | 主要影响 | 代价或风险 |
|---|---|---|
| `n_samples_per_prompt` | 组内比较数量、mixed group 和 advantage 稳定性 | 推理/显存开销增加；低温度下可能只是重复 |
| `rollout_batch_size` | 每轮覆盖的不同 prompt 数量、梯度估计稳定性 | 每轮耗时和数据量增加 |
| `global_batch_size` | 每次 optimizer update 的样本量和梯度噪声 | 改变优化行为，通常需要与学习率一起解释 |
| `micro-batch-size` | 显存、吞吐和梯度累积次数 | 实现或 loss 缩放不正确时会改变数值结果 |
| `temperature` | 回答多样性及 reward 分布 | 过低导致组内同质化，过高增加错误和截断 |

主长跑使用每个 prompt 4 个回答；n=8 A/B 用于检验增加组内样本是否提高 mixed group 比例。
A/B 必须固定 checkpoint、prompt 数据、温度、reward parser、训练步数和固定 eval 集，至少
比较 `mixed group ratio`、zero-std group ratio、completed reward、truncated ratio、
advantage std 和固定 GSM8K accuracy，不能只比较单轮 rollout reward。
