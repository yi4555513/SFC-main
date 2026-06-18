"""Backward-compatible QoS imports.

The active implementation lives in main.utils.service_qos to avoid circular
imports with main.core.
"""

from main.utils.service_qos import *  # noqa: F401,F403
