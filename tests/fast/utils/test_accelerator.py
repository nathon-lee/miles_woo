import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from miles import _bootstrap
from miles.utils import accelerator


@pytest.mark.parametrize(
    ("argv", "environ", "expected"),
    [
        (["train.py"], {}, "auto"),
        (["train.py"], {_bootstrap.HARDWARE_PLATFORM_ENV: "ROCM"}, "rocm"),
        (["train.py", "--hardware-platform", "musa"], {}, "musa"),
        (["train.py", "--hardware-platform=cuda"], {_bootstrap.HARDWARE_PLATFORM_ENV: "musa"}, "cuda"),
    ],
)
def test_requested_hardware_platform(argv, environ, expected):
    assert _bootstrap.requested_hardware_platform(argv=argv, environ=environ) == expected


def test_requested_hardware_platform_rejects_unknown_value():
    with pytest.raises(RuntimeError, match="Unknown hardware platform 'tpu'"):
        _bootstrap.requested_hardware_platform(argv=["train.py", "--hardware-platform=tpu"], environ={})


def test_requested_hardware_platform_requires_a_value():
    with pytest.raises(RuntimeError, match="--hardware-platform requires one of"):
        _bootstrap.requested_hardware_platform(argv=["train.py", "--hardware-platform"], environ={})


def test_bootstrap_imports_configured_patch(tmp_path):
    patch_dir = tmp_path / "patch"
    patch_dir.mkdir()
    (patch_dir / "musa_patch.py").write_text("BOOTSTRAPPED = True\n")
    sys.modules.pop("musa_patch", None)

    try:
        selected = _bootstrap.bootstrap_hardware(
            argv=["train.py", "--hardware-platform", "musa"],
            environ={_bootstrap.MUSA_PATCH_PATH_ENV: str(patch_dir)},
        )

        assert selected == "musa"
        assert importlib.import_module("musa_patch").BOOTSTRAPPED is True
    finally:
        sys.modules.pop("musa_patch", None)
        while str(patch_dir) in sys.path:
            sys.path.remove(str(patch_dir))


def test_bootstrap_rejects_invalid_patch_path(tmp_path, monkeypatch):
    missing = tmp_path / "missing"

    def fail_import(module):
        raise ModuleNotFoundError(module)

    monkeypatch.setattr(_bootstrap.importlib, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="platform=musa could not import musa_patch"):
        _bootstrap.bootstrap_hardware(
            argv=["train.py", "--hardware-platform", "musa"],
            environ={_bootstrap.MUSA_PATCH_PATH_ENV: str(missing)},
        )


def test_explicit_non_musa_platform_does_not_import_musa_patch(monkeypatch):
    imported = []
    monkeypatch.setattr(_bootstrap.importlib, "import_module", imported.append)

    selected = _bootstrap.bootstrap_hardware(
        argv=["train.py", "--hardware-platform", "cuda"],
        environ={_bootstrap.MUSA_PATCH_PATH_ENV: "/unused/musa/patch"},
    )

    assert selected == "cuda"
    assert imported == []


@pytest.mark.parametrize(
    ("platform", "expected_device_type", "expected_backend"),
    [
        ("cpu", "cpu", "gloo"),
        ("cuda", "cuda", "nccl"),
        ("rocm", "cuda", "nccl"),
        ("musa", "musa", "mccl"),
    ],
)
def test_platform_contract(platform, expected_device_type, expected_backend):
    assert accelerator.device_type(platform) == expected_device_type
    assert accelerator.process_group_backend(platform=platform) == expected_backend


def test_explicit_backend_is_preserved():
    assert accelerator.process_group_backend(backend="gloo", platform="musa") == "gloo"


def test_explicit_musa_request_fails_when_runtime_is_unavailable(monkeypatch):
    monkeypatch.setattr(accelerator, "_load_musa_runtime", lambda: None)
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)

    with pytest.raises(RuntimeError, match="platform=musa.*is_available.*false"):
        accelerator.hardware_platform(requested="musa")


def test_auto_detection_prefers_musa_then_rocm_then_cuda(monkeypatch):
    for key in ("MUSA_VISIBLE_DEVICES", "MTHREADS_VISIBLE_DEVICES", "MUSA_PATCH_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: True)
    monkeypatch.setattr(accelerator, "is_rocm_available", lambda: True)
    monkeypatch.setattr(accelerator, "is_cuda_available", lambda: True)
    assert accelerator.hardware_platform(requested="auto") == "musa"

    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    assert accelerator.hardware_platform(requested="auto") == "rocm"

    monkeypatch.setattr(accelerator, "is_rocm_available", lambda: False)
    assert accelerator.hardware_platform(requested="auto") == "cuda"


def test_cpu_operations_are_safe_noops(monkeypatch):
    monkeypatch.setattr(accelerator, "hardware_platform", lambda requested=None: "cpu")

    assert accelerator.device_name() == "cpu"
    assert accelerator.current_device() == "cpu"
    assert accelerator.memory_allocated() == 0
    assert accelerator.memory_reserved() == 0
    assert accelerator.max_memory_allocated() == 0
    assert accelerator.get_device_properties() is None
    assert accelerator.Stream() is None
    assert accelerator.Event() is None
    assert accelerator.current_stream() is None
    assert accelerator.get_rng_state_all() == []
    assert isinstance(accelerator.stream_context(None), nullcontext)
    accelerator.set_device(0)
    accelerator.synchronize()
    accelerator.empty_cache()
    accelerator.ipc_collect()
    accelerator.set_rng_state_all([])


def test_visible_device_mapping_accepts_physical_and_local_ids(monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "4,7")

    assert accelerator.resolve_visible_device_id(4, platform="musa") == 0
    assert accelerator.resolve_visible_device_id("7", platform="musa") == 1
    assert accelerator.resolve_visible_device_id(0, platform="musa") == 0

    with pytest.raises(RuntimeError, match="MUSA_VISIBLE_DEVICES='4,7'"):
        accelerator.resolve_visible_device_id(9, platform="musa")


def test_runtime_summary_records_requested_and_active_platform(monkeypatch):
    monkeypatch.setattr(accelerator, "hardware_platform", lambda requested=None: "musa")

    assert accelerator.runtime_summary(requested="auto") == {
        "requested_platform": "auto",
        "platform": "musa",
        "device_type": "musa",
        "process_group_backend": "mccl",
    }


def test_accelerator_module_routes_musa_operations(monkeypatch):
    calls = []
    fake_musa = SimpleNamespace(
        set_device=lambda value: calls.append(("set_device", value)),
        current_device=lambda: 2,
        synchronize=lambda *args: calls.append(("synchronize", args)),
        empty_cache=lambda: calls.append(("empty_cache",)),
        ipc_collect=lambda: calls.append(("ipc_collect",)),
        memory_allocated=lambda device=None: 11,
        memory_reserved=lambda device=None: 12,
        max_memory_allocated=lambda device=None: 13,
        mem_get_info=lambda device: (14, 15),
        get_device_properties=lambda device: {"device": device},
        stream=lambda stream: nullcontext(stream),
        current_stream=lambda: "current-stream",
        get_rng_state_all=lambda: ["rng"],
        set_rng_state_all=lambda states: calls.append(("set_rng_state_all", states)),
    )
    monkeypatch.setattr(accelerator, "hardware_platform", lambda requested=None: "musa")
    monkeypatch.setattr(accelerator, "_musa_module", lambda: fake_musa)

    accelerator.set_device(2)
    accelerator.synchronize()
    accelerator.empty_cache()
    accelerator.ipc_collect()
    accelerator.set_rng_state_all(["rng"])

    assert accelerator.current_device() == 2
    assert accelerator.mem_get_info() == (14, 15)
    assert accelerator.memory_allocated() == 11
    assert accelerator.memory_reserved() == 12
    assert accelerator.max_memory_allocated() == 13
    assert accelerator.get_device_properties() == {"device": 2}
    assert accelerator.current_stream() == "current-stream"
    assert accelerator.get_rng_state_all() == ["rng"]
    assert calls == [
        ("set_device", 2),
        ("synchronize", ()),
        ("empty_cache",),
        ("ipc_collect",),
        ("set_rng_state_all", ["rng"]),
    ]
