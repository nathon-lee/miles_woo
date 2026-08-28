<!-- Suggested title: [Roadmap][Feature] Support Moore Threads (MUSA) GPUs in Miles -->

## Summary

This umbrella issue proposes first-class support for Moore Threads GPUs through the MUSA software stack in Miles.

The goal is to enable a maintainable end-to-end reinforcement learning workflow on MTGPU hardware, including:

- FSDP training;
- SGLang rollout;
- MCCL distributed communication;
- trainer-to-rollout weight synchronization;
- checkpoint save and resume;
- reproducible installation, testing, and CI.

This issue is intended to coordinate architecture decisions, implementation PRs, hardware validation, and existing work. It should not be implemented as one large patch.

## Motivation

Miles currently relies on several CUDA- and NCCL-oriented assumptions across training, Ray actor placement, memory management, checkpointing, rollout, and online weight synchronization.

Moore Threads provides the following relevant components:

- [torch_musa](https://github.com/MooreThreads/torch_musa): PyTorch integration for MUSA devices;
- MCCL: distributed communication backend for MUSA;
- [torchada](https://github.com/MooreThreads/torchada): compatibility helpers for selected CUDA-oriented Python and extension paths;
- [SGLang MUSA support roadmap](https://github.com/sgl-project/sglang/issues/16565): upstream MUSA inference support and optimized kernels.

SGLang is already developing MUSA support. Adding a corresponding platform path to Miles would make it possible to build a complete training-and-rollout stack on MTGPU hardware.

The implementation should preserve CUDA and ROCm behavior, keep vendor-specific logic isolated, and avoid scattering `if musa` branches throughout training and rollout code.

## Existing work and coordination

There is already an open implementation attempt:

- [Miles PR #1786: Add backend-aware MUSA support to Miles](https://github.com/radixark/miles/pull/1786)

This roadmap is not intended to replace or duplicate that work.

As of 2026-08-28, the `miles_woo` fork also contains a candidate platform-abstraction branch,
[`feat/musa-platform-abstraction`](https://github.com/nathon-lee/miles_woo/tree/feat/musa-platform-abstraction),
at commit `31285d03`. It adds the accelerator layer, platform argument handling, bootstrap
ordering, and CPU-mock tests, but it is not merged into Miles `main` and has no real MUSA hardware
evidence. Treat it as `Patch available`, not as support completion.

Before starting new implementation PRs, we should coordinate with the author of #1786 and determine which commits can be reused or split into smaller reviewer-friendly changes. Existing authorship and contributions should be preserved.

The collaboration decision is an explicit gate: if the candidate and #1786 have the same intended
ownership, request permission to revise or split #1786; otherwise open a follow-up PR from the
current `main`, link #1786, and state the exact reused base and new scope. Do not silently copy a
colleague's branch or mark a roadmap item complete because a fork branch exists.

Related references:

- [SGLang MUSA roadmap #16565](https://github.com/sgl-project/sglang/issues/16565)
- [SGLang MUSA installation documentation](https://github.com/sgl-project/sglang/blob/main/docs/platforms/mthreads_gpu.md)
- [Slime MUSA accelerator PR #2216](https://github.com/THUDM/slime/pull/2216)
- [Slime MUSA follow-up PR #2286](https://github.com/THUDM/slime/pull/2286)
- [Miles AMD roadmap #639](https://github.com/radixark/miles/issues/639)
- [Miles AMD Q3 development #2025](https://github.com/radixark/miles/issues/2025)
- [DeepSeek-V4 on ROCm #1113](https://github.com/radixark/miles/issues/1113)
- [Miles refactor roadmap #427](https://github.com/radixark/miles/issues/427)
- [AMD LoRA RL tracker #2705](https://github.com/radixark/miles/issues/2705)
- [Miles documentation polish #1481](https://github.com/radixark/miles/issues/1481)

### Adjustments learned from PR #1786

The existing implementation suggests three requirements that should be explicit in this roadmap:

1. **Capability matrix:** In addition to platform, device, and backend selection, record whether SGLang routes, Muon, P2P transfer, the SGLang dumper, DeepSeek encoders, streams/events, OOM observers, and low-precision extensions are `available`, `fallback`, or `unsupported`.
2. **Process-group reconstruction test:** In addition to the basic L1 MCCL test, create, destroy, and reconstruct the weight-update group in the same process. Verify barrier, broadcast, rank, world size, and multiple weight versions.
3. **Compatibility contract tests:** Run the same import, argument, route, checkpoint, and weight-update smoke tests against the oldest supported and current pinned SGLang/Megatron dependency sets. Optional capabilities may warn and fall back; required capabilities must fail fast.

## Goals

1. Introduce an explicit and maintainable hardware-platform abstraction.
2. Support BF16 dense-model FSDP training on MUSA.
3. Support rollout through an external MUSA-enabled SGLang server.
4. Provide a correctness-first checkpoint-based weight update path.
5. Add MCCL-based online weight synchronization after the checkpoint path is validated.
6. Support Miles-managed SGLang actors after external rollout is stable.
7. Lock and document compatible driver, SDK, PyTorch, torch_musa, MCCL, and SGLang versions.
8. Add real-hardware smoke tests and CI without regressing CUDA or ROCm.

## Initial scope

The first end-to-end milestone should deliberately remain small:

- single-node execution;
- one or two MUSA GPUs;
- Qwen3 0.6B or another small dense text model;
- BF16;
- FSDP training;
- external, non-colocated SGLang rollout;
- correctness-first checkpoint weight update;
- eager/native kernels where possible;
- at least two complete rollout -> train -> update cycles.

## Non-goals for the first milestone

The following features should remain disabled until the basic path is correct and reproducible:

- Megatron training;
- MoE and expert parallelism;
- FP8, FP4, and quantized training;
- colocated training and rollout;
- CUDA/MUSA IPC-based pause and resume;
- DeepEP, NIXL, Mooncake, or RDT weight transfer;
- custom fused kernels and CUDA Graph equivalents;
- multi-node training;
- performance parity with CUDA or ROCm.

Unsupported configurations should fail fast with an error containing `platform=musa` and an actionable alternative. They must not silently fall back to a semantically different path.

## Design principles

### 1. Thin platform abstraction

Introduce a small accelerator layer responsible for:

- device construction and selection;
- device type detection;
- synchronization;
- memory accounting and cache management;
- streams and events;
- autocast;
- RNG state;
- distributed backend selection;
- visible-device resolution.

Generic Miles code should call this layer instead of adding new direct `torch.cuda.*` calls.

### 2. Explicit bootstrap order

MUSA initialization must occur before importing SGLang, Megatron, or hardware extensions that inspect the active PyTorch platform during import.

The selected platform and detected runtime must be printed during startup. An explicitly requested MUSA configuration must not silently fall back to CUDA or CPU.

### 3. Separate device type from communication backend

The platform layer should distinguish:

- device type: `musa`;
- standard distributed backend: `mccl`;
- composite weight-update backend when required: `cpu:gloo,musa:mccl`.

Backend names should not be hard-coded in FSDP actors or weight-update business logic.

### 4. Explicit Ray device mapping

Ray may continue using logical GPU resources for scheduling initially, but Miles must explicitly map:

`Ray logical GPU ID -> MUSA_VISIBLE_DEVICES -> local MUSA device`

Every actor should log and validate:

- Ray-assigned logical device IDs;
- process-visible MUSA devices;
- selected local device;
- distributed rank and world size.

Multiple actors must not accidentally bind to the same physical device.

### 5. External rollout before managed rollout

The first rollout milestone should connect Miles to an independently launched MUSA-enabled SGLang server.

This isolates SGLang runtime compatibility from Ray actor lifecycle and device-assignment issues. Miles-managed SGLang actors should be added only after the external contract is stable.

### 6. Correctness before communication performance

The first weight-update implementation should use a checkpoint-based path and verify that the trainer and rollout model agree.

MCCL broadcast should be introduced only after the correctness contract is established.

### 7. Capability probing

Optional SGLang endpoints and version-specific parameters should be detected by capability probing rather than assumptions based only on a version string.

However, missing optional lifecycle endpoints must not be confused with successful weight synchronization. Required weight and numerical checks must always run.

### 8. Small, independently reviewable PRs

Each implementation PR should:

- have one clear responsibility;
- preserve CUDA and ROCm behavior;
- include focused tests;
- state the exact validation boundary;
- link back to this roadmap;
- avoid unrelated compatibility changes.

## Proposed PR breakdown

Branch names below are suggestions for contributor forks.

- [ ] **PR 1: Platform abstraction**  
  Suggested branch: `feat/musa-platform-abstraction`

  Existing candidate commit: `31285d03` in the `miles_woo` fork. Rebase or split it only after
  resolving ownership with #1786; rerun the focused tests against the current Miles `main`.

  Add the accelerator interface, explicit platform selection, bootstrap ordering, backend selection, and CPU-mock tests. Do not add online weight synchronization or globally monkey-patch `torch.distributed`.

- [ ] **PR 2: FSDP MUSA train-only support**  
  Suggested branch: `feat/musa-fsdp-train-only`

  Adapt FSDP device mesh, tensor placement, autocast, RNG, checkpointing, and Ray-visible device mapping. Add single-GPU and two-GPU smoke tests.

- [ ] **PR 3: Runtime image and version matrix**  
  Suggested branch: `build/musa-runtime-image`

  Add a reproducible MUSA image or vendor-image installation procedure. Pin the driver, MUSA SDK, torch, torch_musa, MCCL, SGLang, Python, and other required components.

- [ ] **PR 4: External SGLang and checkpoint weight update**  
  Suggested branch: `feat/musa-sglang-checkpoint-update`

  Add the external rollout contract and a correctness-first checkpoint updater. Verify parameter hashes and fixed-input logits before and after an update.

- [ ] **PR 5: MCCL online weight synchronization**  
  Suggested branch: `feat/musa-mccl-online-weight-update`

  Add the MCCL weight-update topology and process-group lifecycle. Validate creation, update, destruction, reconstruction, and multiple weight versions.

- [ ] **PR 6: Miles-managed SGLang actors**  
  Suggested branch: `feat/musa-managed-sglang`

  Propagate platform and device visibility to SGLang actors and subprocesses. Reuse the weight-update contract validated in the previous PR.

- [ ] **PR 7: MUSA hardware CI**  
  Suggested branch: `ci/musa-hardware-smoke`

  Add MUSA runner registration, environment checks, MCCL tests, FSDP smoke tests, SGLang server tests, and weight-update tests.

- [ ] **PR 8: Megatron MUSA feasibility and integration**  
  Suggested branch: `experiment/musa-megatron-spike`

  Keep this separate from the initial FSDP path. Begin with a fixed-version BF16 dense train-only feasibility test before attempting weight conversion, MoE, or low-precision support.

## Cross-project tracking

MUSA support depends on several fast-moving repositories. This issue should track the following items instead of treating the initial bring-up as a one-time port:

1. **Miles shared architecture:** Follow #427 and related training-backend refactors. Reuse common accelerator, checkpoint, and weight-update contracts instead of adding MUSA-only business logic.
2. **AMD findings:** Periodically review #639, #2025, #1113, and #2705 for new numerical, resume, low-precision, and stability failure modes. Upstream platform-independent regression tests to Miles where possible.
3. **SGLang MUSA:** Track the installation guide, MUSA roadmap, weight-update protocol, attention backends, graph/IPC support, and hardware CI matrix. Record separate commits for `sglang-miles` and upstream SGLang `main`.
4. **torch_musa and MCCL:** Track PyTorch compatibility, composite process groups, FSDP2, distributed checkpointing, RNG, and profiler support. Rerun L0-L4 after every dependency upgrade.
5. **Slime platform abstraction:** Continue learning from its backend-aware APIs and capability probing, but revalidate the Miles Ray/FSDP/SGLang call chain. Slime CPU-mock results are not MUSA hardware evidence for Miles.
6. **External Megatron patch:** Record its base commit, update date, patch scope, upstreaming status, and removal criteria. Keep `experimental dependency` separate from `native Miles support`.
7. **Documentation and debugging:** Following #1481, maintain a single-node quick start, environment capture, bounded success logs, common failures, and a review date. Avoid preserving only one-off commands or unbounded raw logs.

## Correctness and coverage requirements

- **Trainer/rollout numerical alignment:** At step 0, after the first weight update, and at step N, run fixed tokens and record the maximum and mean absolute difference between trainer and rollout log probabilities. Track both the absolute difference and drift over time. Thresholds must be established on the complete model for each model and dtype.
- **Extreme batches:** Test short, median, longest, and truncated responses rather than only an average length. Monitor MCCL timeouts, expert-load imbalance, and peak memory.
- **Parallelism and kernel matrix:** Record `verified`, `fallback`, or `unsupported` for TP, PP, EP, sequence length, dtype, and model shape. One TP configuration must not imply support for other configurations.
- **Stability and memory slope:** Record used memory after every training, rollout, and pause/resume phase, not only peak memory. Prove the external, non-colocated path before enabling colocation.
- **Weight transport format:** For every updated tensor, record its name, shape, logical dtype, transport dtype, byte length, and restored hash. A successful API response is not a substitute for logits parity; #1113 documents corruption caused by restoring FP4 expert bytes as `int8`/`uint8`.
- **Exact resume:** Validate model weights, optimizer/master weights, LR scheduler, RNG, rollout ID, and dataset cursor. The next iteration after resume should agree with an uninterrupted reference run.
- **Reduced-model boundary:** A small or reduced-layer model can prove scheduling and data flow, but cannot prove full-model log-probability, MoE, low-precision, or performance correctness.
- **CI parity:** Maintain a gap table for matching CUDA, ROCm, and MUSA test suites. Distinguish hardware limitations, unported functionality, and tests that are not connected to a runner instead of reporting only one aggregate pass count.

## Validation ladder

A milestone must not be marked complete without the corresponding real-hardware evidence.

| Level | Required evidence |
| --- | --- |
| L0: Environment | Exact software versions, visible MUSA devices, finite forward and backward results |
| L1: MCCL | Two-GPU all-reduce, all-gather, barrier, correct rank mapping, and clean process exit |
| L2: FSDP | Two optimizer steps with finite loss, gradients, and parameters; checkpoint save and load |
| L3: SGLang | Repeated external generation requests and the expected token/log-prob contract |
| L4: Weight update | Matching parameter names, dtypes, shapes, byte lengths, hashes, and fixed-input logits |
| L5: RL smoke | At least two rollout -> train -> update cycles with an observed weight-version change |
| L6: Resume | Exact restoration of model, optimizer, scheduler, RNG, rollout, and dataset state |
| L7: Stability | At least 20 finite iterations without NaN, collective timeout, actor leak, or sustained memory growth |
| L8: Performance | Reproducible throughput, memory, and timing results with the complete workload configuration |

Import success, a mocked MCCL backend, or a single rollout request is useful bring-up evidence, but must not be reported as end-to-end MUSA support.

## Required evidence for implementation PRs

Every hardware-related PR should include:

- Miles base commit and PR commit;
- GPU model and number of devices;
- driver and MUSA SDK versions;
- Python, torch, torch_musa, MCCL, Ray, and SGLang versions;
- complete launch command;
- relevant environment variables;
- focused test results;
- concise success markers from real logs;
- known unsupported features;
- CUDA and ROCm regression-test results where applicable.

Long raw logs may be attached as artifacts, but the PR description should summarize the exact acceptance markers.

## Main risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Silent CUDA or CPU fallback | Require explicit platform selection and print the detected runtime |
| Dependency version drift | Maintain a tested version matrix and immutable image reference |
| Incorrect Ray device binding | Log and assert the complete logical-to-physical device mapping |
| MCCL process-group deadlock | Test repeated create/update/destroy cycles and release locks in `finally` |
| Trainer/rollout weight mismatch | Compare parameter metadata, hashes, and fixed-input logits |
| Large, difficult-to-review patch | Split the work into the independent PRs listed above |
| Regression to CUDA or ROCm | Keep existing paths unchanged and run focused cross-platform tests |
| Optional endpoint mistaken for update success | Separate lifecycle capability probing from mandatory weight validation |

## Decisions requested from maintainers

Before implementation proceeds, it would be helpful to confirm:

1. Whether the thin accelerator abstraction is acceptable for Miles.
2. Whether FSDP plus external SGLang is the preferred first milestone.
3. Whether #1786 should be incrementally revised or split into smaller PRs.
4. Which SGLang MUSA version or commit should be used as the initial compatibility target.
5. Whether a real MUSA runner can be made available for CI.
6. Which initial model and hardware configuration should define the first supported baseline.

## Definition of done

This roadmap can be considered minimally complete when:

- the platform abstraction is merged;
- a documented and reproducible MUSA environment is available;
- a two-GPU FSDP smoke test passes;
- external SGLang rollout passes;
- trainer-to-rollout weight synchronization is numerically verified;
- at least two complete RL cycles pass;
- exact checkpoint resume passes;
- the supported configuration is covered by real-hardware CI;
- CUDA and ROCm behavior remains unchanged.

Performance optimization, additional models, Megatron, MoE, low precision, colocation, and multi-node support should remain follow-up roadmap items.
