"""Miles package root."""

import os

from miles._bootstrap import bootstrap_hardware as _bootstrap_hardware

_requested_hardware_platform = _bootstrap_hardware()
if _requested_hardware_platform != "auto" or any(
    key in os.environ for key in ("MUSA_VISIBLE_DEVICES", "MTHREADS_VISIBLE_DEVICES", "MUSA_PATCH_PATH")
):
    from miles.utils.accelerator import runtime_summary as _runtime_summary

    _runtime_summary(requested=_requested_hardware_platform)
