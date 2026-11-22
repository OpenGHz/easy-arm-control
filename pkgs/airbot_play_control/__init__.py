"""Deprecated alias of :mod:`easy_arm_control`, kept for backward compatibility.

``airbot_play_control`` was renamed to ``easy_arm_control``. Importing the old
name returns the very same module objects, so ``from airbot_play_control.control
import RoboticArmAgent`` keeps working and yields the same class as
``from easy_arm_control.control import RoboticArmAgent``.
"""

import sys
import warnings

import easy_arm_control
from easy_arm_control import *  # noqa: F403
from easy_arm_control import control

__all__ = easy_arm_control.__all__

# Make `import airbot_play_control.control` resolve to the same module object.
sys.modules[__name__ + ".control"] = control

warnings.warn(
    "airbot_play_control has been renamed to easy_arm_control; "
    "please update your imports",
    DeprecationWarning,
    stacklevel=2,
)
