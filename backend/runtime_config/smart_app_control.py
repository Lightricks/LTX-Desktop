"""Detect Windows Smart App Control (SAC).

SAC in enforcement mode blocks unsigned code that has no good reputation. The app
builds such code at run time: Triton compiles ``cuda_utils`` and every kernel launcher
into ``~/.triton/cache`` as unsigned ``.pyd`` files, and SageAttention needs Triton.
Windows then fails the import with "An Application Control policy has blocked this
file" (CodeIntegrity event 3077). Signing the cache output is not possible, so callers
must not start the Triton-dependent paths while SAC enforces.

The state lives in ``HKLM\\SYSTEM\\CurrentControlSet\\Control\\CI\\Policy`` as the DWORD
``VerifiedAndReputablePolicyState``: 0 = off, 1 = on, 2 = evaluation. Evaluation mode
does not block code, so only ``on`` counts as blocking.
"""

from __future__ import annotations

import sys
from typing import Literal

SmartAppControlState = Literal["off", "on", "evaluation", "unknown"]

_CI_POLICY_KEY = r"SYSTEM\CurrentControlSet\Control\CI\Policy"
_SAC_STATE_VALUE = "VerifiedAndReputablePolicyState"


def parse_smart_app_control_state(value: object) -> SmartAppControlState:
    """Map the registry DWORD to a state. Unexpected values map to "unknown"."""
    match value:
        case 0:
            return "off"
        case 1:
            return "on"
        case 2:
            return "evaluation"
        case _:
            return "unknown"


def detect_smart_app_control() -> SmartAppControlState:
    """Return the SAC state of this host.

    Non-Windows hosts and Windows builds without the value (no SAC) return "off".
    Any other read failure returns "unknown". Callers treat "unknown" like "off", so
    a failed probe never removes a feature that works today.
    """
    if sys.platform != "win32":
        return "off"

    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _CI_POLICY_KEY) as key:
            value, _value_type = winreg.QueryValueEx(key, _SAC_STATE_VALUE)
    except FileNotFoundError:
        return "off"
    except OSError:
        return "unknown"
    return parse_smart_app_control_state(value)


def smart_app_control_blocks_unsigned_code(state: SmartAppControlState) -> bool:
    """True when SAC enforces, so run-time compiled Triton code cannot load."""
    return state == "on"
