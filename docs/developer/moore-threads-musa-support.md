---
title: Miles 支持摩尔线程 MUSA 的设计与实施路线
description: 基于当前 Miles、torch_musa、SGLang 和 Megatron-LM 现状设计 MUSA 支持，定义首版边界、代码改造和验收步骤。
---

本文讨论的是**如何让目前尚未支持 MUSA 的 Miles 获得原生支持**。这里没有假设一份
现成 patch 已经存在；patch 只是在外部依赖尚未合入修复时的临时交付方式，不是方案
本身。

本文的仓库判断基于 Miles `f2b7c792`（2026-08-26 的 `main`）。依赖项目变化较快，
实施前应重新锁定和验证 commit。

## 结论

推荐按以下主线推进：

```text
平台抽象
  → FSDP2 BF16 train-only
  → 外部 SGLang MUSA rollout-only
  → checkpoint 方式更新权重（先证明正确性）
  → MCCL 在线权重广播
  → Miles 内部托管 SGLang/Ray
  → Megatron MUSA
  → FP8、融合算子、colocate 和高性能传输
```

首版目标应限定为：**在实际验证过 MCCL 的目标机器上，Qwen3 小模型使用 FSDP2、
BF16、eager attention 完成两卡训练和最小端到端 RL smoke**。不能只按 GPU 型号推断
MCCL 可用性。首版不承诺 Megatron、FP8、DeepEP、CUDA Graph、colocate、训练
offload、P2P/RDT 或故障恢复。

这条路线比先适配 Megatron 更短，因为 FSDP 后端主要依赖 PyTorch/Hugging Face；
Megatron 后端还依赖 Megatron-LM、Transformer Engine、Apex、DeepEP、融合 kernel 和
Miles 的 CUDA 专属优化。

## 上游基础能力

截至本文检查时间，可复用的上游能力如下：

| 组件 | 当前基础 | 对 Miles 的含义 |
| --- | --- | --- |
| `torch_musa` | 提供 `torch.musa`、`musa` device 和 `mccl` process group | 设备和通信无需在 Miles 内重造 |
| `torchada` | 可把大量 CUDA PyTorch API 映射到 MUSA/MCCL | 可用于早期 bring-up，但不能代替 Miles 平台抽象和逐项测试 |
| SGLang `main` | 已有 `all_musa` 依赖、`setup_musa.py`、MUSA 平台路径和 MUSA CI | MUSA 代码应来自上游；但 Miles 当前依赖 `sglang-miles`，必须先验证版本兼容或做最小 backport |
| Megatron-LM | NVIDIA 上游没有可直接视为 Miles 兼容的 MUSA 后端 | 首版跳过 Megatron |
| MT-MegatronLM / MT-TransformerEngine | 摩尔线程提供基于固定 Megatron commit 的 patch/优化实现 | 后续可作为移植参考，不能直接假设兼容 Miles 的 `miles-main` |
| Ray | 没有原生 MUSA accelerator resource | 需要显式声明逻辑 `GPU` 资源并补上 MUSA 可见设备映射 |

参考：[`torch_musa`](https://github.com/MooreThreads/torch_musa)、
[`torchada`](https://github.com/MooreThreads/torchada)、
[`SGLang MUSA 安装`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/mthreads_gpu.md)、
[`SGLang MUSA roadmap`](https://github.com/sgl-project/sglang/issues/16565)、
[`MT-MegatronLM`](https://github.com/MooreThreads/MT-MegatronLM)。

## 当前 Miles 的主要缺口

“能 import `torch_musa`”远远不等于 Miles 已支持 MUSA。当前代码至少有以下系统性
缺口。

| 层次 | 当前行为 | 主要位置 |
| --- | --- | --- |
| 启动顺序 | `train.py` 在平台初始化前导入 SGLang | `train.py`, `train_async.py` |
| 设备 API | 直接使用 `torch.cuda.*`、`cuda:<rank>`、`.cuda()` | `miles/ray/train_actor.py`, `miles/utils/memory_utils.py`, `miles/backends/**` |
| 通信 | 默认和临时权重组都写死 `nccl` | `miles/utils/arguments.py`, `miles/backends/fsdp_utils/update_weight_utils.py` |
| FSDP | device mesh、autocast、模型搬运和 RNG 都写死 CUDA | `miles/backends/fsdp_utils/`, `miles/backends/training_utils/data.py` |
| Ray 调度 | placement group 只申请 Ray 的 `GPU`，仅识别 CUDA/HIP 可见设备变量 | `miles/ray/placement_group.py`, `miles/ray/train_actor.py`, `miles/ray/utils.py` |
| rollout | Miles 的 engine actor 和 SGLang 子 actor 假设 Ray 能管理 NVIDIA/AMD GPU | `miles/ray/rollout/`, `miles/backends/sglang_utils/sglang_engine.py` |
| 权重同步 | broadcast 写死 NCCL；P2P/RDT 依赖 CUDA/NIXL；colocate 依赖 CUDA IPC | FSDP/Megatron update-weight 实现和参数校验 |
| 训练优化 | TE、Apex、FlashAttention、DeepEP、NVSHMEM、CUDA Graph 等具有硬件约束 | Megatron、模型脚本和 Docker 镜像 |
| 监控/CI | 监控只有 NVML/AMD SMI，CI 只有 CPU/CUDA/ROCm | `miles/dashboard/gpu_sampler.py`, `tests/ci/`, `.github/workflows/` |

因此不应在入口简单 `import torchada` 后就宣称完成。适配层可以帮助首轮 smoke，最终仍
需要把设备、通信、调度和能力开关收敛到 Miles 自己的接口。

## 设计一个统一的平台抽象

### 新增平台模块

建议新增 `miles/utils/accelerator.py`，由它统一回答：

- 当前平台：`cuda`、`rocm`、`musa` 或 `cpu`；
- PyTorch device type：MUSA 返回 `musa`；
- distributed backend：CUDA/ROCm 返回 `nccl`，MUSA 返回 `mccl`；
- 可见设备变量：CUDA、HIP/ROCR、MUSA 分别使用哪些变量；
- `set_device`、`current_device`、`synchronize`、`empty_cache` 和 memory stats；
- autocast device type；
- RNG state 的保存和恢复；
- 某项能力是否支持，例如 graph、IPC、fused optimizer、RDT、DeepEP。

接口可以类似：

```python
class Accelerator:
    platform: str
    device_type: str
    distributed_backend: str

    def set_device(self, local_rank: int) -> None: ...
    def current_device(self) -> torch.device: ...
    def synchronize(self) -> None: ...
    def empty_cache(self) -> None: ...
    def memory_stats(self) -> dict: ...
    def supports(self, feature: str) -> bool: ...
```

实现时优先使用当前 PyTorch 的 `torch.accelerator` 通用 API；对于 backend、可见设备、
RNG 和供应商能力仍由 Miles 包装。业务代码不应继续新增 `if is_musa` 分支或直接访问
`torch.musa`。

### 平台选择与初始化顺序

增加 `--hardware-platform auto|cuda|rocm|musa`，也允许
`MILES_HARDWARE_PLATFORM=musa`。`auto` 按以下顺序检测：

1. 显式环境变量；
2. `torch.version.musa` / `torch.musa.is_available()`；
3. `torch.version.hip`；
4. CUDA；
5. CPU。

MUSA 初始化必须发生在导入 SGLang、Megatron 或其他 CUDA 扩展之前。建议在
`train.py` 和 `train_async.py` 的第一批 import 中调用一个轻量 bootstrap：

```python
from miles.utils.accelerator import bootstrap_accelerator

bootstrap_accelerator()
```

MUSA bootstrap 负责加载 `torch_musa`；若首轮需要 `torchada`，也在此处条件导入，
并把实际启用状态打印到日志。`torchada` 不能无条件影响 CUDA/ROCm 进程。

参数 `--distributed-backend` 的默认值应从 `nccl` 改成 `auto`，解析完成后由平台解析成
`nccl` 或 `mccl`。用户显式传值时保留覆盖能力。

## Ray 如何分配 MUSA 设备

Ray 当前没有原生 MUSA resource。第一版不应立即发明新的 `MUSA` resource，因为 Miles
和 SGLang 的 placement-group 代码都使用 `GPU`。更小的方案是：

1. 启动 Ray 时显式声明逻辑 GPU 数，例如 `ray start --head --num-gpus=8`；
2. 容器层设置 `MTHREADS_VISIBLE_DEVICES`，进程层使用 `MUSA_VISIBLE_DEVICES`；
3. Miles actor 读取 `ray.get_gpu_ids()`，把分配结果映射到 `MUSA_VISIBLE_DEVICES`；
4. `get_local_gpu_id()` 同时识别 CUDA、HIP/ROCR 和 MUSA 可见设备变量；
5. 在 actor 内调用平台接口的 `set_device(local_rank)`；
6. 启动时验证每个 rank 看到且绑定唯一的 MUSA device。

Ray 的 `GPU` 在这里是**调度资源名称**，不代表底层设备是 CUDA。这个兼容方案必须写进
日志和文档，避免用户误认为 Ray 已原生发现 MUSA。

由于 SGLang 的内部 Ray actor 也需要正确继承设备映射，首个端到端版本优先使用
`--rollout-external`：先在 MUSA 环境独立启动上游 SGLang，再由 Miles 连接。这样可以
把“模型推理是否可用”和“Ray 子 actor 调度是否正确”分开验证。

## 分阶段实施

### 阶段 0：锁定可复现环境

先选定一种实际 GPU 和一套版本，不能用 `latest` 作为支持矩阵。至少固定：

- GPU 型号、每节点卡数；
- driver、MUSA SDK/toolkit、muDNN、MCCL；
- Python、torch、torch_musa、torchada；
- SGLang commit、Miles commit；
- 目标模型、dtype、attention backend。

验收：`musaInfo` 正常，`torch.musa.device_count()` 等于可见卡数，单卡 tensor 运算通过，
两卡 MCCL all-reduce 的期望和为 3（rank 0 输入 1、rank 1 输入 2）。

建议把版本检查固化为 `tests/manual/musa/collect_env.py`，MCCL 测试固化为
`tests/manual/musa/smoke_mccl.py`，而不是在文档里保存一次性的命令。

### 阶段 1：FSDP2 train-only

先只运行 `--debug-train-only`，不启动 SGLang。需要修改：

- `miles/ray/train_actor.py`：设备绑定和 `mccl` 初始化；
- `miles/backends/fsdp_utils/parallel.py`：mesh device type；
- `miles/backends/fsdp_utils/adaptations/precision.py`：autocast device type；
- `miles/backends/fsdp_utils/actor.py`：模型、buffer、optimizer state 搬运；
- `miles/backends/training_utils/data.py`：batch tensor device；
- `miles/utils/memory_utils.py`：同步、cache 和显存统计；
- `miles/backends/fsdp_utils/checkpoint.py`：MUSA RNG、保存和恢复；
- profile/device-flops 等辅助模块：无 MUSA API 时明确禁用或降级。

首个 recipe 建议新增 `scripts/musa/run_qwen3_0_6b_fsdp_smoke.py`，使用 BF16、eager
attention、AdamW 非 fused 路径、单卡、短序列和两个 optimizer steps。稳定后再扩到
两卡 FSDP2 + MCCL。

验收标志：

- 每个 rank 打印 `platform=musa`, `device=musa:<local_rank>`, `backend=mccl`；
- forward、loss、backward、grad norm 和 optimizer step 均为 finite；
- 两个 step 后至少一个参数发生有限变化；
- 两卡训练退出无 hang；
- CUDA/ROCm 的既有 fast tests 不回归。

### 阶段 2：SGLang MUSA rollout-only

按上游 MUSA 方式构建 SGLang，并先独立验证同一个小模型。Miles 使用
`--debug-rollout-only --rollout-external` 连接服务，不参与 GPU 调度。

第一轮关闭或避免未经验证的特性：

- 使用上游 MUSA 推荐的 attention backend；若不明确，先从 eager/兼容实现开始；
- TP=1；
- 不启用 speculative decoding、PD disaggregation、HiCache 或量化；
- 不做 weight update，只验证 prompt → tokens、log-prob 和 reward 数据契约。

验收：固定 prompt 的输出 shape、token IDs、finish reason 和 log-prob 都满足 Miles 的
rollout contract，并连续完成至少 20 个请求。

### 阶段 3：先打通正确的权重更新

在线 RL 的关键不是训练和推理各自能跑，而是新权重能从 trainer 到 rollout。

建议分两步：

1. **正确性路径：checkpoint reload。** 新增一个简单的 checkpoint weight-transfer
   mode：FSDP rank 0 产出 HF checkpoint，外部 SGLang 调用
   `update_weights_from_disk`。它较慢，但没有 MCCL/设备组变量，最适合证明 step N 的
   参数确实影响 step N+1 的 rollout。
2. **性能路径：MCCL broadcast。** 将 FSDP `UpdateWeightFromDistributed` 中写死的
   `nccl` 改为平台 backend，对 trainer 和 SGLang 都传 `mccl`，从 1 trainer + 1
   rollout rank 开始，再扩到 TP 和多节点。

P2P、RDT 和 colocate 不应作为 MUSA 首版路径：它们当前分别依赖 Mooncake/NIXL、CUDA
registration 或 CUDA IPC。必须有独立的 MUSA transport 证明后才能开放。

权重更新验收不能只看 API 返回成功。应在 trainer 和 SGLang 两侧计算同名参数的 hash，
并用固定输入检查更新前后 logits：更新后 trainer 与 rollout 的 logits 在约定误差内
一致，而且确实不同于更新前。

### 阶段 4：Miles 内部托管 SGLang

外部模式通过后，再修改：

- `miles/ray/placement_group.py`：记录逻辑 GPU 与物理 MUSA ID；
- `miles/ray/train/actor_factory.py`：为每个 actor 传播正确的平台环境；
- `miles/ray/rollout/server_group.py`：移除或隔离 NVIDIA 专属环境默认值；
- `miles/backends/sglang_utils/sglang_engine.py`：识别 MUSA visibility 和 MTML UUID；
- SGLang RayEngine 子 actor：验证每个 scheduler rank 的设备绑定。

先支持非 colocate 的独立 trainer/rollout 卡，再支持同节点；不要一开始启用 fractional
GPU 共享和内存 pause/resume。

### 阶段 5：Megatron MUSA

只有 FSDP 端到端完成后再评估 Megatron。需要先建立一个三方兼容矩阵：

```text
Miles commit
  + radixark/Megatron-LM miles-main commit
  + MT-MegatronLM 可应用的基线 commit
  + MT-TransformerEngine commit
  + torch/torch_musa/MUSA SDK/MCCL
```

优先把通用修复提交到 Megatron-LM/Miles 的平台抽象。只有尚未上游的差异才保存 patch。
首个 Megatron smoke 仍使用 BF16 dense 模型，关闭 TE FP8、DeepEP、fused optimizer、
CUDA graph、context parallel 和 overlap；按 TP=1 → TP=2 → DP+TP 的顺序增加复杂度。

### 阶段 6：性能和完整功能

功能正确后再逐项开放，每项都需要独立 benchmark 和回归测试：

- MT-TransformerEngine 与 FP8；
- MUSA FlashAttention/MATE；
- fused Adam、fused norm、MoE kernel；
- CUDA/MUSA Graph；
- DeepEP 或 MUSA 等价通信库；
- colocate 和显存 pause/resume；
- 多节点 P2P/RDT 等高性能权重传输；
- fault tolerance、checkpoint resume 和长稳训练；
- 基于 `mthreads-ml-py` 的 dashboard GPU telemetry。

不要用“速度更快”作为默认结论。记录 GPU、driver、SDK、torch/torch_musa、模型、dtype、
world size、序列长度、batch、TP/DP 和实际 tokens/s，再与同配置基线比较。

## 首版功能边界

| 功能 | 首版状态 | 说明 |
| --- | --- | --- |
| FSDP2 BF16 | 目标支持 | 首条训练路径 |
| Qwen3 小模型 | 目标支持 | 先 dense、文本模型 |
| 外部 SGLang MUSA | 目标支持 | 隔离 Ray 调度问题 |
| checkpoint weight update | 目标支持 | 正确性优先的慢路径 |
| MCCL broadcast | 下一步支持 | 在线训练所需性能路径 |
| Miles 内部托管 SGLang | 后续 | 需要 Ray/MUSA 映射 |
| Megatron | 暂不支持 | 等 FSDP 端到端后推进 |
| FP8/FP4、DeepEP、量化 | 暂不支持 | 需要供应商 kernel 和数值验证 |
| colocate/offload | 暂不支持 | 当前依赖 CUDA IPC/torch_memory_saver |
| P2P/RDT | 暂不支持 | 当前依赖 CUDA/NIXL/Mooncake 路径 |
| deterministic collectives/FT | 暂不支持 | 当前实现绑定 `det_nccl`/ProcessGroupNCCL |

代码在未支持功能被请求时应 fail fast，错误中包含 `platform=musa` 和替代配置；不能静默
启用一个结果不等价的 fallback。

## 建议拆成独立 PR

### PR 1：平台抽象

- 新增 `miles/utils/accelerator.py`；
- 增加 `--hardware-platform` 和 backend `auto`；
- 改造通用 memory/device/autocast/RNG API；
- 添加 CPU mock 单元测试，确保 CUDA/ROCm 行为不变。

### PR 2：FSDP MUSA train-only

- 改造 FSDP device mesh、模型搬运、data tensor 和 checkpoint；
- 新增 `scripts/musa/run_qwen3_0_6b_fsdp_smoke.py`；
- 新增单卡和两卡手工 smoke；
- 明确禁用不支持功能。

### PR 3：MUSA 镜像与版本锁定

- 新增 `docker/Dockerfile.musa` 或固定的供应商基础镜像流程；
- 固定 torch/torch_musa/SGLang/MUSA 组件版本；
- 添加环境自检和镜像 smoke；
- 更新安装与版本文档。

### PR 4：外部 SGLang 与 checkpoint 权重更新

- rollout-only contract test；
- 新增 correctness-first checkpoint updater；
- 添加参数 hash 和 logits parity 验证。

### PR 5：MCCL broadcast 与内部托管 rollout

- 权重更新 backend 平台化；
- Ray 逻辑 GPU → MUSA visibility 映射；
- SGLang 子 actor rank/device 测试；
- 非 colocate 两卡/四卡端到端 smoke。

### PR 6：MUSA CI

- 在 `tests/ci/ci_register.py` 增加 `register_musa_ci` 和 `HWBackend.MUSA`；
- 在 `tests/ci/run_suite.py` 增加 MUSA suite；
- 新增 MUSA workflow、runner 标签、镜像选择和日志采集；
- 每个 PR 至少运行环境、MCCL、FSDP 两步和 SGLang server smoke。

### PR 7：Megatron MUSA

- 单独维护依赖兼容矩阵；
- 从 BF16 dense train-only 开始；
- 再接权重转换、MCCL broadcast、MoE 和低精度。

这些 PR 不应合成一个巨型 patch。每一层都应能独立回答“改了什么、在哪种硬件上验证、
成功标志是什么、下一层仍缺什么”。

## 验收阶梯

| 级别 | 必须证明的内容 |
| --- | --- |
| L0 环境 | device 可见、版本一致、单卡算子有限值 |
| L1 MCCL | 两卡 collective 正确且无 hang |
| L2 FSDP | 两卡完成两个 optimizer steps，loss/grad/参数有限 |
| L3 SGLang | 外部 server 连续请求和 log-prob contract 正确 |
| L4 权重更新 | 参数 hash 和固定输入 logits 在 trainer/rollout 对齐 |
| L5 RL smoke | 至少两个 rollout→train→update 周期完成 |
| L6 多卡 | 目标卡数、TP/DP 和 checkpoint resume 通过 |
| L7 稳定性 | 长稳运行无 NaN、无 collective timeout、无持续内存增长 |
| L8 性能 | 在完整环境和 workload 参数下报告吞吐、显存和时间分解 |

没有真实 MUSA 机器日志时，只能报告代码审查、静态检查或“待执行命令”，不能把 L0-L8
写成已通过。

## patch 应该什么时候出现

只有外部组件在锁定版本上确实缺少能力，而且修复尚未上游时，才添加 patch：

```text
docker/musa_patch/<component>/<base-commit>/0001-<purpose>.patch
```

每份 patch 必须绑定精确基线 commit、对应 issue/PR、`git apply --check`、测试命令和删除
条件。Miles 自身的修改应直接以普通源代码 PR 提交，不要生成 `miles.patch` 再反向应用
到同一仓库。

## 实施前需要确认的三个输入

开始写代码前，需要在目标机器上确认：

1. GPU 型号和卡数，以及该型号是否提供 MCCL；
2. 当前可用的供应商容器、MUSA SDK、torch 和 torch_musa 版本；
3. 首个 smoke 模型，建议使用可本地获取的 Qwen3 小模型。

这三个输入决定 Docker 基线、Python 版本和首个 CI suite，但不会改变“先 FSDP、再
SGLang、最后 Megatron”的总体顺序。
