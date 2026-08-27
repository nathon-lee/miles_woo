"""Early hardware bootstrap that must not import torch or vendor runtimes."""

import importlib
import os
import sys
from collections.abc import Mapping, Sequence

HARDWARE_PLATFORM_ENV = "MILES_HARDWARE_PLATFORM"
MUSA_PATCH_PATH_ENV = "MUSA_PATCH_PATH"
HARDWARE_PLATFORMS = ("auto", "cuda", "rocm", "musa")


def _platform_from_argv(argv: Sequence[str]) -> str | None:
    option = "--hardware-platform"
    for index, argument in enumerate(argv):
        if argument.startswith(f"{option}="):
            return argument.split("=", maxsplit=1)[1]
        if argument == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                raise RuntimeError(f"{option} requires one of: {', '.join(HARDWARE_PLATFORMS)}")
            return argv[index + 1]
    return None


def requested_hardware_platform(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the requested platform without importing a hardware runtime."""
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    platform = _platform_from_argv(argv) or environ.get(HARDWARE_PLATFORM_ENV, "auto")
    platform = platform.lower()
    if platform not in HARDWARE_PLATFORMS:
        raise RuntimeError(f"Unknown hardware platform {platform!r}; expected one of: {', '.join(HARDWARE_PLATFORMS)}")
    return platform


def _prepend_import_paths(paths: str) -> None:
    for path in reversed([item for item in paths.split(os.pathsep) if item]):
        if path not in sys.path:
            sys.path.insert(0, path)


def _bootstrap_musa_patch(environ: Mapping[str, str]) -> None:
    patch_paths = environ.get(MUSA_PATCH_PATH_ENV)
    if not patch_paths:
        return

    _prepend_import_paths(patch_paths)
    try:
        importlib.import_module("musa_patch")
    except Exception as exc:
        raise RuntimeError(
            f"platform=musa could not import musa_patch from {MUSA_PATCH_PATH_ENV}={patch_paths!r}"
        ) from exc


def bootstrap_hardware(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Select the platform and load an explicitly configured patch before torch."""
    target_environ = os.environ if environ is None else environ
    platform = requested_hardware_platform(argv=argv, environ=target_environ)

    if target_environ is os.environ:
        os.environ[HARDWARE_PLATFORM_ENV] = platform

    musa_signals = platform == "musa" or (
        platform == "auto"
        and (
            "MUSA_VISIBLE_DEVICES" in target_environ
            or "MTHREADS_VISIBLE_DEVICES" in target_environ
            or bool(target_environ.get(MUSA_PATCH_PATH_ENV))
        )
    )
    if musa_signals:
        _bootstrap_musa_patch(target_environ)
    return platform
