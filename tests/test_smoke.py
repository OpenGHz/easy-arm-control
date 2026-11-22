"""Smoke tests; run inside a ROS Noetic environment (rospy, moveit_commander)."""

import importlib
import py_compile
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAMES = [
    "RoboticArmAgent",
    "AirbotPlayConfig",
    "ChooseGripper",
    "GripperSimpleControl",
    "CoordinateTools",
    "MoveItBasicController",
]


def test_import_easy_arm_control():
    control = importlib.import_module("easy_arm_control.control")
    for name in PUBLIC_NAMES:
        assert hasattr(control, name), name


def test_deprecated_alias_resolves_to_same_objects():
    import easy_arm_control
    import easy_arm_control.control as new_control

    for mod in ("airbot_play_control", "airbot_play_control.control"):
        sys.modules.pop(mod, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import airbot_play_control
        import airbot_play_control.control as old_control
        from airbot_play_control.control import RoboticArmAgent

    assert old_control is new_control
    assert RoboticArmAgent is new_control.RoboticArmAgent
    assert airbot_play_control.RoboticArmAgent is easy_arm_control.RoboticArmAgent
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "demo").glob("*.py")),
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_scripts_compile(path):
    py_compile.compile(str(path), doraise=True)
