---
title: Miles 支持摩尔线程 MUSA 的设计与实施路线
description: 基于当前 Miles、torch_musa、SGLang 和 Megatron-LM 现状设计 MUSA 支持，定义首版边界、代码改造和验收步骤。
---

本文讨论的是**如何让目前尚未支持 MUSA 的 Miles 获得原生支持**。摩尔线程已经在维护
`megatron-lm-musa-patch`，Slime 也已经合入 MUSA accelerator 抽象和旧版 SGLang
兼容修复；这些成果可以缩短 bring-up 时间，但不能直接证明 Miles 已经支持 MUSA。
外部 patch 仍然是固定版本上的临时依赖，不是 Miles 平台设计本身。

本文的仓库判断基于 Miles `f2b7c792`（2026-08-26 的 `main`）。依赖项目变化较快，
实施前应重新锁定和验证 commit。本文对 Miles 公开 roadmap 和 AMD/ROCm 进展的
补充检查日期为 2026-08-27；issue 状态和验证结果以链接中的最新内容为准。

截至 2026-08-28，`miles_woo` 的 `feat/musa-platform-abstraction` 分支已有一个候选
实现（commit `31285d03`），包含 accelerator 层、平台参数解析和 CPU mock 测试；它仍是
未合入 `main` 的 `Patch available`，没有 MUSA 真机验收证据，不能将其描述为 Miles 已
支持 MUSA。该候选实现与公开的 Miles PR #1786 存在功能重叠，后续应先与原作者和维护者
确认采用“修订原 PR”还是“独立 follow-up PR”，保留原有贡献归属，避免平行实现互相冲突。

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
| Slime `main` | 已合入 backend-aware accelerator 抽象和旧版 SGLang capability probing | 可复用设计和测试思路，但 Slime 的 CPU mock 测试不等于 Miles 真机通过 |
| Megatron-LM | NVIDIA 上游没有可直接视为 Miles 兼容的 MUSA 后端 | 首版跳过 Megatron；可并行做外部 patch 可行性验证 |
| `megatron-lm-musa-patch` | 摩尔线程维护基于固定 Megatron/Slime 版本的 patch 和 example，目前仍在修改 | 可作为实验依赖；必须锁定 patch 与被适配仓库的 commit，不能跟随浮动分支 |
| MT-MegatronLM / MT-TransformerEngine | 摩尔线程提供基于固定 Megatron commit 的 patch/优化实现 | 后续可作为移植参考，不能直接假设兼容 Miles 的 `miles-main` |
| Ray | 没有原生 MUSA accelerator resource | 需要显式声明逻辑 `GPU` 资源并补上 MUSA 可见设备映射 |

参考：[`torch_musa`](https://github.com/MooreThreads/torch_musa)、
[`torchada`](https://github.com/MooreThreads/torchada)、
[`SGLang MUSA 安装`](https://github.com/sgl-project/sglang/blob/main/docs/platforms/mthreads_gpu.md)、
[`SGLang MUSA roadmap`](https://github.com/sgl-project/sglang/issues/16565)、
[`Slime MUSA accelerator PR #2216`](https://github.com/THUDM/slime/pull/2216)、
[`Slime SGLang compatibility PR #2286`](https://github.com/THUDM/slime/pull/2286)、
[`Miles MUSA PR #1786`](https://github.com/radixark/miles/pull/1786)、
[`megatron-lm-musa-patch`](https://sh-code.mthreads.com/ai/megatron-lm-musa-patch)、
[`MT-MegatronLM`](https://github.com/MooreThreads/MT-MegatronLM)。

准备提交到 Miles 上游的英文 roadmap Issue 草稿保存在
`.github/issue-drafts/musa-support-roadmap.md`。

## 从 Slime 两个 PR 得到的结论

这两个 PR 不是互斥的两套完整适配方案：

- PR #2216 是长期架构改造。它增加 accelerator 抽象，并覆盖设备、显存、Ray 可见设备
  映射、profiler、process group、Megatron/SGLang 权重更新和部分模型/转换工具。MUSA 将
  逻辑 `nccl` 映射为 `mccl`，权重更新组使用 `cpu:gloo,musa:mccl`。
- PR #2286 只有旧版 SGLang 兼容修复：在新旧参数都不存在时 warning 后跳过，并在
  `RouterArgs` 没有 `disable_health_check` 时避免直接访问。它本身不提供 MUSA backend。
- `megatron-lm-musa-patch` 把尚未上游的 Megatron/MUSA 差异留在外部仓库。这样对 Slime
  主线侵入较小，适合快速验证，但兼容性由 Slime、Megatron、SGLang 和 patch 的精确
  commit 共同决定。

因此 Miles 推荐组合使用这三类经验：以 PR #2216 风格的平台抽象作为长期边界，以
PR #2286 风格的 capability probing 兼容依赖版本，以外部 Megatron patch 作为可删除、
可锁版本的实验依赖。不能在入口全局映射 `torch.cuda` 后就把 Miles 标记为原生支持。

PR #2216 的 MUSA/MCCL 测试主要由 CPU mock、fake `torch.musa` 和 backend 字符串完成；
Slime PR 合入状态只能作为代码设计证据，真实硬件能力仍需按本文验收阶梯重新验证。

## PR #1786 的具体实现和可复用经验

[Miles PR #1786](https://github.com/radixark/miles/pull/1786) 是一个可供实现层面学习的 MUSA 适配参考。
本地审阅到它由 4 个连续提交组成，对应四个阶段：

1. `bc74e1b9` 新增 `miles/utils/accelerator.py` 和 MUSA bootstrap；
2. `b4585ec8` 将 FSDP、Ray、数据、内存、RNG、profile 和权重转换的设备操作路由 accelerator；
3. `06f4e6b7` 将分布式权重更新的 `nccl` 转换为 MCCL，并改造 reloadable process group；
4. `6680bfab` 处理 MUSA 运行时的可选依赖、旧版 SGLang 参数和 DeepSeek 编码器导入。

### 值得保留的设计

- **薄平台层：** 用 `device()`、`device_type()`、`set_device()`、`synchronize()`、`empty_cache()`、
  `memory_*()`、stream/event、RNG 和 `process_group_backend()` 统一包装设备行为，避免在每个
  FSDP/Ray/模型文件里继续写 `torch.cuda.*`。
- **后端名称与设备类型分离：** 常规训练组映射为 `musa:mccl`，权重更新可使用
  `cpu:gloo,musa:mccl`。这比在业务代码中散落的 `if musa` 更可维护。
- **Ray 可见设备映射：** 通过 `resolve_visible_device_id()` 处理 Ray 逻辑 ID 与可见设备索引，并将
  `RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES` 放入保护列表。这与“Ray 申请 GPU 只是调度名称”的文档边界一致。
- **分布式组可重建：** `reloadable_process_group` 在同一进程内缓存创建参数，允许权重更新后重建通信组。MUSA 适配不只是把 backend 字符串改成 `mccl`，还要验证组的生命周期、rank 映射和重建后的 barrier/broadcast。
- **可选能力探测：** SGLang 权重更新的 `/begin_weight_update` 和 `/end_weight_update` 使用 route probing，旧版参数使用 `getattr` 兼容，DeepSeek encoder 使用延迟导入。这是应对多仓库版本漂移的好方式，但只能对可选功能跳过，不能跳过必需的权重和数值校验。
- **非核心依赖延迟导入：** Muon、P2P weight transfer、SGLang dumper 和特定 encoder 不再在模块导入阶段强制要求，当用户真正请求时才报出带替代方案的错误。这对 MUSA 装配裁剪版依赖很有参考价值。

### 这个 PR 不能直接复制的部分

- PR 中的 accelerator 在 `import torch` 之后才尝试导入 `musa_patch`，与本文建议的“MUSA bootstrap 先于 SGLang/Megatron 执行”目标并不完全相同。应用前要用实际 `torch_musa`/外部 patch 验证导入顺序，否则可能已经错过运行时注入点。
- `process_group_backend()` 和 `weight_update_backend()` 的映射以 `torch.musa.is_available()` 为基础，不是显式 `--hardware-platform` 配置。在 MUSA 环境中应拒绝静默回退，并把实际检测值打入环境报告。
- `torch.distributed` 的 `new_group` 和 collective 被 monkey-patch 以兼容 `ReloadableProcessGroup`。这种方法可解决权重更新的特定问题，但会影响全进程通信 API；必须有集成测试、重复调用保证和明确的升级/移除计划。
- SGLang 旧版未声明 route 时可跳过 begin/end 请求，这不能证明权重更新成功。MUSA 首版应把“未知的可选生命周期接口”与“必须保证的权重同步”分开，并记录 skip 原因。
- PR 首要处理了依赖版本差异，但不包含完整的 MUSA 硬件日志、MCCL smoke、FSDP 两步、logits parity 或长稳数据。因此本文仍将它标记为 `Patch available` 参考，不升级为 `Minimal loop passed` 或 `Numerically verified`。

### 对本文既有计划的调整

PR #1786 证明原有阶段需补充三个实施任务：

1. **加入能力矩阵：** 除了 platform/device/backend，还要记录 SGLang route、Muon、P2P、dumper、DeepSeek encoder、stream/event、OOM observer 和各类低精度扩展是 `available`、`fallback` 还是 `unsupported`。
2. **加入组重建测试：** L1 MCCL 之外，用同一进程创建、销毁、重建权重组，检查 barrier、broadcast、rank/world-size 与多轮 weight version 一致。
3. **加入兼容性契约测试：** 用旧、当前两套 SGLang/Megatron 依赖运行相同的 import、参数、route、checkpoint 和 weight-update smoke；可选能力可 warning+降级，必需能力必须 fail fast。

## 从 AMD/ROCm 路线图学什么

AMD 对 Miles 的支持已经从“设备能否启动”进入训练、rollout、权重同步、低精度、
长稳和 CI 共同验证的阶段。这些结果不能直接证明 MUSA 可用，但其问题分层和
验收方法值得直接复用。

### 公开路线图的层级

| 层级 | 跟踪项 | 对 MUSA 的用法 |
| --- | --- | --- |
| Miles 总路线 | [#797 2026 Q2 Roadmap](https://github.com/radixark/miles/issues/797) | 确认平台适配必须服务于训练/推理对齐、LoRA、低精度、agentic 和 CI，而不是独立 fork |
| 硬件路线 | [#639 AMD Miles dev roadmap](https://github.com/radixark/miles/issues/639) | 按功能、性能、模型覆盖和 CI/CD 四条线组织 MUSA roadmap，为每项写责任人和目标 |
| 新一轮 AMD 计划 | [#2025 Miles AMD Q3 Development](https://github.com/radixark/miles/issues/2025) | 把 Miles、SGLang、Megatron-Bridge、Transformer Engine、镜像和 CI 作为一条跨仓库依赖链管理 |
| 模型真机记录 | [#1113 DeepSeek-V4 on ROCm](https://github.com/radixark/miles/issues/1113) | 学习记录 logprob 漂移、长序列通信失败、TP kernel 约束、colocate 泄漏和在线更新损坏等真实证据 |
| 模型交付 | [#1046 DeepSeek V4 RL Roadmap](https://github.com/radixark/miles/issues/1046) | 裁剪模型只能验证基础设施；完整模型、锁定镜像、确切分支和命令才构成交付证据 |
| 架构重构 | [#427 Miles Refactor Roadmap](https://github.com/radixark/miles/issues/427) | 把通用训练工具从 Megatron 专属路径拆出；MUSA 设备、通信和 RNG 也应下沉到公共平台接口 |
| Rollout 协议 | [#712 Agentic server roadmap](https://github.com/radixark/miles/issues/712) | 平台基础稳定后，检查 MUSA 路径是否保持 token-in-token-out 和多轮轨迹契约 |
| 低精度 | [#615 Blackwell MXFP8/NVFP4](https://github.com/radixark/miles/issues/615) | 学习如何把硬件特有精度做成独立 roadmap；不应将 FP8/FP4 塞进 MUSA 首版 bring-up |
| 文档质量 | [#1481 Miles Docs Polish](https://github.com/radixark/miles/issues/1481) | 提供单节点可复现示例、事实核查、调试工具和贡献要求 |
| 功能 tracker | [#2705 AMD LoRA RL tracker](https://github.com/radixark/miles/issues/2705) | 将 checkpoint、resume、同步、多 adapter、MoE 和 E2E 拆成可独立 review 的 PR，并标明合并顺序 |

`roadmap` 搜索不会自动找到标题中没有该词的 tracker，因此不能只看 roadmap label；
还应同时检查硬件 label、关联 PR、外部仓库 PR 和镜像配方。#639 与 #2025 有新旧和
范围重叠，但上游没有声明后者取代前者；不应将两份清单的完成项简单相加。

### 将 AMD 的真实问题转成 MUSA 验收项

AMD 路线暴露的问题说明，“运行成功”并不是完整的异构支持结论。MUSA 必须额外验证：

- **训练/推理数值对齐：** 在 step 0、首次权重更新和 step N 使用固定 token，记录
  trainer/rollout logprob 的 max/mean absolute diff；同时看绝对值和随时间的漂移，
  阈值必须按模型、dtype 和完整模型实测后确定。
- **极端 batch：** 分别运行短、中位、最长和 truncated response，不能只用平均长度；
  监控 MCCL timeout、expert 负载不均和峰值显存。
- **并行和 kernel 支持矩阵：** 为 TP/PP/EP、sequence length、dtype 和模型 shape 列出
  `verified`/`unsupported`/`fallback`，不由一个 TP 配置推断其他配置可用。
- **长稳和显存斜率：** 除了记录峰值，还要记录每轮训练、rollout、pause/resume
  后的已用显存；先证明 external/non-colocate 路径，再打开 colocate。
- **权重传输格式：** 对每个更新 tensor 记录名称、shape、逻辑 dtype、transport dtype、
  byte length 和恢复后 hash。#1113 曾记录在线更新把 FP4 expert 字节当成
  int8/uint8 恢复而损坏权重；API 返回成功不能代替 logits 对齐。
- **精确恢复：** 恢复检查不只是 model weight，还包括 optimizer/master weight、LR scheduler、
  RNG、rollout ID 和 dataset cursor。恢复后的下一轮应与不中断对照运行一致。
- **裁剪模型的边界：** 小模型或 reduced-layer 模型可以证明调度和数据流能跑，
  不能证明完整模型的 logprob、MoE、低精度或性能正确。
- **CI 对等性：** 建立 CUDA/ROCm/MUSA 同名测试集的 gap 表，区分“硬件不支持”、
  “尚未移植”和“测试未接入 runner”，不用一个总通过数掩盖缺口。

这些项目应分别属于平台基础、最小闭环、正确性、Megatron 和高级性能 PR，不应为了与
AMD 的完整列表对齐而把 LoRA、FP8、colocate 提前到 MUSA 首版。

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

## 贡献关系与合并策略

PR #1786 和 `feat/musa-platform-abstraction` 都应先视为待评审的上游工作，而不是可以
直接覆盖的“废弃版本”。如果候选分支与 #1786 的目标和提交者一致，优先在原 PR 上提出
拆分、rebase 或补充测试的建议；只有原作者和维护者明确同意时，才在该 PR 分支上继续
修改。若无法取得协作权限，使用独立 follow-up PR，明确写出依赖的 base commit、复用的
提交和新增范围，并在描述中链接 #1786。无论采用哪条路径，PR 都必须重新对齐当前 `main`
并通过本文的 L0-L2 验收，不能因为分支名或已有 CPU mock 测试而提前勾选 roadmap。

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

MUSA bootstrap 负责加载 `torch_musa`。如果实验路径使用外部 `musa_patch`，应在显式选择
MUSA 后、验证 `torch.musa` 前加载，并保证早于 Megatron、SGLang 和 CUDA/MUSA 扩展：

```text
选择 platform=musa
  → 定位并加载固定 commit 的外部 musa_patch
  → 验证 torch.musa 和 MCCL
  → 导入 Megatron/SGLang
  → 创建训练和权重更新 process group
```

patch 路径只用于定位依赖，不应单独触发静默平台切换。若首轮需要 `torchada`，也只在
这里条件导入，并把实际启用状态打印到日志。`musa_patch` 和 `torchada` 都不能无条件
影响 CUDA/ROCm 进程。

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
- SGLang commit、Miles commit；若使用外部路径，还要固定 Slime、Megatron-LM 和
  `megatron-lm-musa-patch` commit；
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

Miles 与 SGLang 的版本边界采用 capability probing，不只按版本字符串分支：优先检测
实际参数、属性或函数签名，同时兼容 current/legacy 名称；可选能力不存在时打印一次
明确 warning 并关闭对应功能。影响正确性的必需能力仍应 fail fast，不能静默跳过。

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
   `nccl` 改为平台 backend。纯设备 tensor 组使用 `mccl`；如果锁定版本的
   `torch_musa` 支持复合 backend，可验证 `cpu:gloo,musa:mccl`，让 CPU 元数据和 MUSA
   tensor 分别使用 Gloo/MCCL。从 1 trainer + 1 rollout rank 开始，再扩到 TP 和多节点。

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

Miles 的正式支持顺序仍是 FSDP-first；L0/L1 环境和 MCCL 通过后，可以并行进行一次
外部 Megatron patch feasibility spike，但不能因此提前把 Megatron 标记为已支持。
正式接入前建立兼容矩阵：

```text
Miles commit
  + radixark/Megatron-LM miles-main commit
  + Megatron-LM 被适配的 base commit
  + megatron-lm-musa-patch commit
  + MT-MegatronLM/MT-TransformerEngine commit（如果使用）
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
| Megatron | 暂不支持，可做外部 patch spike | 真机跑通只算实验结果，正式接入仍在 FSDP 路线之后 |
| FP8/FP4、DeepEP、量化 | 暂不支持 | 需要供应商 kernel 和数值验证 |
| colocate/offload | 暂不支持 | 当前依赖 CUDA IPC/torch_memory_saver |
| P2P/RDT | 暂不支持 | 当前依赖 CUDA/NIXL/Mooncake 路径 |
| deterministic collectives/FT | 暂不支持 | 当前实现绑定 `det_nccl`/ProcessGroupNCCL |

代码在未支持功能被请求时应 fail fast，错误中包含 `platform=musa` 和替代配置；不能静默
启用一个结果不等价的 fallback。

## 建议拆成独立 PR

建议拆成 10 个有明确入口、出口和验收边界的 PR。这里的 PR 是审查边界，
不是必须逐个对应一次发布；如果某两个 PR 的改动很小，可以在维护者同意后合并，
但不能把下面不同的验收责任重新揉成一个巨型 patch。

### PR 0：环境基线与验收契约

- 推荐分支：`docs/musa-compatibility-baseline`；
- 固化 GPU、driver、MUSA SDK、Python、torch、torch_musa、MCCL、Ray、SGLang 和镜像 digest；
- 新增 `tests/manual/musa/collect_env.py` 和 L0/L1 smoke 命令；
- 记录版本矩阵、成功标志和“不支持”状态；
- 不修改训练或 rollout 业务逻辑。

### PR 1：Accelerator 平台抽象

- 推荐分支：`feat/musa-platform-abstraction`；
- 当前候选：`31285d03`（未合入 `main`，需先完成作者/维护者协作和基线同步）；
- 新增 `miles/utils/accelerator.py`、`--hardware-platform` 和 backend `auto`；
- 统一 device、memory、autocast、RNG、stream/event 和 backend 接口；
- 添加 CPU mock 测试，覆盖显式 MUSA 请求失败时 fail fast，并确保 CUDA/ROCm 不变；
- 不在这个 PR 引入 Ray 映射、在线权重更新或全局 monkey-patch `torch.distributed`。

### PR 2：Bootstrap 与 Ray 设备映射

- 推荐分支：`feat/musa-bootstrap-device-mapping`；
- 确保 MUSA 初始化早于 SGLang、Megatron 和硬件扩展导入；
- 完成 Ray logical GPU ID → `MUSA_VISIBLE_DEVICES` → local device 映射；
- 输出并校验 rank、world size、可见设备和实际绑定设备；
- 这一层同时服务训练和 rollout，不应隐藏在 FSDP PR 内。

### PR 3：MUSA 镜像与依赖矩阵

- 推荐分支：`build/musa-runtime-image`；
- 新增 `docker/Dockerfile.musa` 或固定供应商镜像安装流程；
- 固定已验证的 torch/torch_musa/MCCL/SGLang 组合和 Python 版本；
- 记录 driver/SDK 兼容范围，而不是声称由仓库锁定 driver；
- 添加环境自检和镜像 smoke，作为后续真机验证的统一基线。

### PR 4：FSDP MUSA train-only

- 推荐分支：`feat/musa-fsdp-train-only`；
- 改造 FSDP device mesh、模型/数据搬运、autocast、optimizer 和 MUSA RNG；
- 只加入最小 checkpoint save/load，不加入 exact resume 或在线权重更新；
- 新增 Qwen3 小模型单卡和两卡 smoke，明确禁用不支持功能。

### PR 5：外部 SGLang rollout-only

- 推荐分支：`feat/musa-sglang-rollout-only`；
- 连接独立启动的 MUSA SGLang，不启动 Miles 内部 actor；
- 验证 token、finish reason、log-prob 和 reward contract，连续完成固定请求；
- 用 capability probing 兼容版本差异；
- 不创建 MCCL 权重组，也不宣称权重同步完成。

### PR 6：Checkpoint 权重更新正确性路径

- 推荐分支：`feat/musa-checkpoint-weight-update`；
- 新增 trainer → checkpoint → 外部 SGLang 的慢速更新路径；
- 验证参数名、shape、dtype、byte length、hash 和固定输入 logits/log-prob；
- 至少覆盖 step 0、首次更新和 step N；
- 不在这个 PR 引入 MCCL broadcast、colocate 或 P2P。

### PR 7：MCCL 在线权重同步

- 推荐分支：`feat/musa-mccl-online-weight-update`；
- 平台化 `mccl` 和 `cpu:gloo,musa:mccl` 权重组；
- 验证创建、销毁、重建、barrier、broadcast、rank/world-size 和多轮 weight version；
- 记录每轮 tensor metadata、hash 和 logits parity，更新锁在 `finally` 中释放；
- 先完成 1 trainer + 1 rollout rank，再扩展到多 engine。

### PR 8：Miles 内部托管 SGLang actor

- 推荐分支：`feat/musa-managed-sglang`；
- 传播平台、visibility 和依赖环境到 SGLang engine/子 actor；
- 验证 scheduler/engine rank 与 MUSA device 唯一绑定；
- 复用 PR 7 的权重更新 contract，先完成 external、non-colocate E2E；
- 不启用 fractional GPU、pause/resume、IPC、P2P 或 RDT。

### PR 9：增量式 MUSA 硬件 CI

- 推荐分支：`ci/musa-hardware-smoke`；
- 第一阶段接入 runner、标签、环境探针和 L0/L1 MCCL；
- 后续逐步接入 FSDP、外部 SGLang、checkpoint 和 MCCL weight-update smoke；
- 在 `tests/ci` 增加 MUSA suite 和 CUDA/ROCm/MUSA gap 表；
- 明确区分 runner 缺失、功能未移植、硬件不支持和测试未接入，不能用 mock 代替真机证据。

### PR 10：Megatron MUSA 实验路线

- 推荐分支：`experiment/musa-megatron-spike`；
- 最好作为独立 experimental issue，不阻塞首版 FSDP 支持；
- 固定外部 `megatron-lm-musa-patch` 和 Megatron base commit；
- 先做 BF16 dense train-only feasibility，再考虑权重转换、MCCL、MoE 和低精度；
- spike 通过不等于 Miles 原生 MUSA 支持完成。

这些 PR 不应合成一个巨型 patch。依赖版本兼容不再作为最后的“大杂烩修复”：
每个功能 PR 只携带它必需的延迟导入、capability probing 和契约测试。每一层都应能独立回答
“改了什么、在哪种硬件上验证、成功标志是什么、下一层仍缺什么”。

## 验收阶梯

| 级别 | 必须证明的内容 |
| --- | --- |
| L0 环境 | device 可见、版本一致、单卡算子有限值 |
| L1 MCCL | 两卡 collective 正确且无 hang |
| L2 FSDP | 两卡完成两个 optimizer steps，loss/grad/参数有限 |
| L3 SGLang | 外部 server 连续请求和 log-prob contract 正确 |
| L4 权重更新 | 参数 hash、dtype/shape/byte length 和固定输入 logits 在 trainer/rollout 对齐，且至少覆盖 step 0/1/N |
| L5 RL smoke | 至少两个 rollout→train→update 周期完成 |
| L6 多卡 | 目标卡数、TP/DP 和 exact checkpoint resume 通过，恢复 model/optimizer/scheduler/RNG/rollout/dataset 状态 |
| L7 稳定性 | 覆盖最长/truncated batch 的长稳运行，无 NaN、无 logprob 持续漂移、无 collective timeout、无持续内存增长 |
| L8 性能 | 在完整环境和 workload 参数下报告吞吐、显存和时间分解 |

没有真实 MUSA 机器日志时，只能报告代码审查、静态检查或“待执行命令”，不能把 L0-L8
写成已通过。

### 统一支持状态

为了避免“支持”同时表示 import 成功、单卡 smoke 和生产可用，issue、PR 和文档统一使用：

| 状态 | 含义 |
| --- | --- |
| `Planned` | 只有设计和验收目标，尚无可运行实现 |
| `Patch available` | 有固定 base/patch commit，但未完成目标环境验证 |
| `Import smoke passed` | 依赖可导入、设备可见；不表示训练可用 |
| `Minimal loop passed` | 在明确硬件和配置上完成最小 rollout→train→update |
| `Numerically verified` | 参数、logits/logprob、reward 和多轮更新满足明确误差标准 |
| `Long-run verified` | 指定时长/步数和极端 batch 下无漂移、泄漏、hang 或损坏 |
| `CI protected` | 同一验收进入 MUSA runner，失败会阻止回归合入 |

每条验证记录至少包含 Miles/patch/依赖 commit、镜像、GPU 型号和数量、driver/SDK/
torch_musa/MCCL、模型、dtype、TP/DP/EP、batch/sequence length、命令、成功标志和原始日志路径。

## patch 应该什么时候出现

只有外部组件在锁定版本上确实缺少能力，而且修复尚未上游时，才添加 patch。patch 可以
随 Miles 镜像交付，也可以来自固定 commit 的外部仓库：

```text
docker/musa_patch/<component>/<base-commit>/0001-<purpose>.patch
```

每份 patch 必须绑定精确基线 commit、patch commit、对应 issue/PR、应用或导入顺序、
修改文件清单、测试命令和删除条件。文件型 patch 还要执行 `git apply --check`；外部
Python patch 要验证 import 来源和实际启用状态。Miles 自身的修改应直接以普通源代码 PR
提交，不要生成 `miles.patch` 再反向应用到同一仓库。

## 实施前需要确认的三个输入

开始写代码前，需要在目标机器上确认：

1. GPU 型号和卡数，以及该型号是否提供 MCCL；
2. 当前可用的供应商容器、MUSA SDK、torch 和 torch_musa 版本；
3. 首个 smoke 模型，建议使用可本地获取的 Qwen3 小模型。

这三个输入决定 Docker 基线、Python 版本和首个 CI suite，但不会改变“先 FSDP、再
SGLang、最后 Megatron”的总体顺序。

## 需要持续关注和学习的上游

这个列表是为了发现接口和验收方法的变化，不是将其中的 AMD/CUDA 实现照搬到 MUSA：

1. **Miles 公共架构：** 跟踪 #427 及相关训练 backend 重构，优先复用通用 accelerator、
   checkpoint 和 weight-update contract，避免新增 MUSA-only 业务分支。
2. **AMD 问题记录：** 定期对照 #639、#2025、#1113 和 #2705，关注新的数值、恢复、
   低精度和长稳失败模式，并将平台无关的回归测试上游到 Miles。
3. **SGLang MUSA：** 跟踪安装文档、MUSA roadmap、权重更新协议、attention backend、
   graph/IPC 和 CI 的真实支持矩阵；对 `sglang-miles` 与上游 `main` 分别记录 commit。
4. **torch_musa 与 MCCL：** 关注 PyTorch 版本对应、复合 process group、FSDP2、distributed
   checkpoint、RNG 和 profiler 支持；任何版本升级都先重跑 L0-L4。
5. **Slime 平台抽象：** 继续学习 backend-aware API 和 capability probing，但用 Miles 自己的
   Ray/FSDP/SGLang 调用链重新验证，不以 Slime CPU mock 结果代替真机证据。
6. **Megatron 外部 patch：** 跟踪 patch 的 base commit、更新日期、修改范围、上游化状态和
   删除条件；维持“实验依赖”与“Miles 原生支持”两种状态。
7. **文档与调试：** 按 #1481 的思路维护单节点 quick start、环境采集、有界限的成功
   日志、常见失败和过期复查日期，避免只保留一次性命令或整段刷屏日志。

每次复查都回答五个问题：上游接口是否变化？外部 patch 是否还能应用？我们的成功结论
对应哪个硬件和 commit？数值与长稳是否真正验证？哪些补丁已可删除或提交上游？
