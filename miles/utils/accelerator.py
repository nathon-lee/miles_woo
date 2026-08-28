"""Backend-aware hardware operations used by Miles runtime components."""

import importlib
import os
from contextlib import nullcontext
from types import ModuleType
from typing import Any

from miles._bootstrap import HARDWARE_PLATFORM_ENV, bootstrap_hardware, requested_hardware_platform

bootstrap_hardware()

import torch


def _musa_module() -> ModuleType | None:
    return getattr(torch, "musa", None)


def _load_musa_runtime() -> None:
    if _musa_module() is not None:
        return
    try:
        importlib.import_module("torch_musa")
    except ImportError as exc:
        raise RuntimeError(
            "platform=musa was requested, but torch_musa is not importable and torch.musa is unavailable"
        ) from exc


def is_musa_available() -> bool:
    musa = _musa_module()
    return bool(musa is not None and getattr(musa, "is_available", lambda: False)())


def is_cuda_available() -> bool:
    return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None)


def is_rocm_available() -> bool:
    return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None)


def hardware_platform(requested: str | None = None) -> str:
    """Return the active platform and reject unavailable explicit requests."""
    requested = requested_hardware_platform() if requested is None else requested
    if requested == "musa":
        _load_musa_runtime()
        if not is_musa_available():
            raise RuntimeError("platform=musa was requested, but torch.musa.is_available() is false")
        return "musa"
    if requested == "cuda":
        if not is_cuda_available():
            raise RuntimeError("platform=cuda was requested, but a CUDA runtime is unavailable")
        return "cuda"
    if requested == "rocm":
        if not is_rocm_available():
            raise RuntimeError("platform=rocm was requested, but a ROCm runtime is unavailable")
        return "rocm"

    musa_requested = any(
        key in os.environ for key in ("MUSA_VISIBLE_DEVICES", "MTHREADS_VISIBLE_DEVICES", "MUSA_PATCH_PATH")
    )
    if musa_requested:
        _load_musa_runtime()
        if not is_musa_available():
            raise RuntimeError("platform=musa was detected from the environment, but the MUSA runtime is unavailable")
        return "musa"
    if is_musa_available():
        return "musa"
    if is_rocm_available():
        return "rocm"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def device_type(platform: str | None = None) -> str:
    platform = hardware_platform() if platform is None else platform
    return "cuda" if platform in ("cuda", "rocm") else platform


def accelerator_module(platform: str | None = None) -> Any:
    platform = hardware_platform() if platform is None else platform
    if platform == "musa":
        return _musa_module()
    if platform in ("cuda", "rocm"):
        return torch.cuda
    return None


def device_name(index: int | None = None) -> str:
    current_type = device_type()
    if current_type == "cpu":
        return "cpu"
    index = current_device() if index is None else index
    return f"{current_type}:{index}"


def device(index: int | None = None) -> torch.device:
    return torch.device(device_name(index))


def set_device(index: int | str | torch.device) -> None:
    module = accelerator_module()
    if module is not None:
        module.set_device(index)


def current_device() -> int | str:
    module = accelerator_module()
    return "cpu" if module is None else module.current_device()


def synchronize(device_arg: int | str | torch.device | None = None) -> None:
    module = accelerator_module()
    if module is None:
        return
    if device_arg is None:
        module.synchronize()
    else:
        module.synchronize(device_arg)


def empty_cache() -> None:
    module = accelerator_module()
    if module is not None:
        module.empty_cache()


def ipc_collect() -> None:
    module = accelerator_module()
    if module is not None and hasattr(module, "ipc_collect"):
        module.ipc_collect()


def mem_get_info(device_arg: int | str | torch.device | None = None) -> tuple[int, int]:
    module = accelerator_module()
    if module is None:
        raise RuntimeError("Accelerator memory information is unavailable on CPU")
    device_arg = current_device() if device_arg is None else device_arg
    return module.mem_get_info(device_arg)


def memory_allocated(device_arg: int | str | torch.device | None = None) -> int:
    module = accelerator_module()
    return 0 if module is None else module.memory_allocated(device_arg)


def memory_reserved(device_arg: int | str | torch.device | None = None) -> int:
    module = accelerator_module()
    return 0 if module is None else module.memory_reserved(device_arg)


def max_memory_allocated(device_arg: int | str | torch.device | None = None) -> int:
    module = accelerator_module()
    operation = None if module is None else getattr(module, "max_memory_allocated", None)
    return 0 if operation is None else operation(device_arg)


def get_device_properties(device_arg: int | str | torch.device | None = None) -> Any:
    module = accelerator_module()
    if module is None:
        return None
    device_arg = current_device() if device_arg is None else device_arg
    return module.get_device_properties(device_arg)


def visible_devices_env_key(platform: str | None = None) -> str:
    platform = hardware_platform() if platform is None else platform
    if platform == "musa":
        if "MUSA_VISIBLE_DEVICES" in os.environ:
            return "MUSA_VISIBLE_DEVICES"
        if "MTHREADS_VISIBLE_DEVICES" in os.environ:
            return "MTHREADS_VISIBLE_DEVICES"
        return "MUSA_VISIBLE_DEVICES"
    if platform == "rocm":
        for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            if key in os.environ:
                return key
        return "ROCR_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


def resolve_visible_device_id(physical_device_id: int | float | str, platform: str | None = None) -> int:
    env_key = visible_devices_env_key(platform)
    visible_devices = os.environ.get(env_key)
    device_id = int(float(physical_device_id))
    if not visible_devices or visible_devices.strip().lower() == "all":
        return device_id

    visible = [int(item.strip()) for item in visible_devices.split(",") if item.strip()]
    if device_id in visible:
        return visible.index(device_id)
    if 0 <= device_id < len(visible):
        return device_id
    raise RuntimeError(
        f"Device id {device_id} is invalid under {env_key}={visible_devices!r}; "
        f"expected a physical id in {visible} or a local id in 0..{len(visible) - 1}"
    )


def process_group_backend(backend: str = "auto", platform: str | None = None) -> str:
    if backend != "auto":
        return backend
    platform = hardware_platform() if platform is None else platform
    if platform == "musa":
        return "mccl"
    if platform in ("cuda", "rocm"):
        return "nccl"
    return "gloo"


def weight_update_backend(platform: str | None = None) -> str:
    """Return the backend contract shared by trainer and rollout weight-sync groups."""
    platform = hardware_platform() if platform is None else platform
    if platform == "musa":
        return "cpu:gloo,musa:mccl"
    return process_group_backend(platform=platform)


def autocast(*, dtype: torch.dtype, enabled: bool = True):
    return torch.autocast(device_type=device_type(), dtype=dtype, enabled=enabled)


def stream_context(stream: Any):
    module = accelerator_module()
    if module is None or stream is None or not hasattr(module, "stream"):
        return nullcontext()
    return module.stream(stream)


def Stream(*args, **kwargs):
    module = accelerator_module()
    stream_cls = None if module is None else getattr(module, "Stream", None)
    return None if stream_cls is None else stream_cls(*args, **kwargs)


def Event(*args, **kwargs):
    module = accelerator_module()
    event_cls = None if module is None else getattr(module, "Event", None)
    return None if event_cls is None else event_cls(*args, **kwargs)


def current_stream():
    module = accelerator_module()
    operation = None if module is None else getattr(module, "current_stream", None)
    return None if operation is None else operation()


def get_rng_state_all() -> list[torch.Tensor]:
    module = accelerator_module()
    return [] if module is None else module.get_rng_state_all()


def set_rng_state_all(states: list[torch.Tensor]) -> None:
    module = accelerator_module()
    if module is not None:
        module.set_rng_state_all(states)


def runtime_summary(requested: str | None = None) -> dict[str, str]:
    platform = hardware_platform(requested=requested)
    return {
        "requested_platform": requested or os.environ.get(HARDWARE_PLATFORM_ENV, "auto"),
        "platform": platform,
        "device_type": device_type(platform),
        "process_group_backend": process_group_backend(platform=platform),
    }
