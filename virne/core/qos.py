"""Backward-compatible QoS imports.

The active implementation lives in virne.utils.service_qos to avoid circular
imports with virne.core.
"""

from virne.utils.service_qos import *  # noqa: F401,F403
