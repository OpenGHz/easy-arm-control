"""MoveItBasicController without a running move_group (fake C++ interface).

Needs moveit_commander, like the smoke tests."""

import pytest
from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander import MoveItCommanderException

from easy_arm_control.control import MoveItBasicController


class FakeInterface:
    """Stands in for MoveGroupCommander._g, the C++ move group interface."""

    def __init__(self, current_joints):
        self.current_joints = current_joints
        self.pose_targets = 0
        self.fetches = 0

    def get_current_joint_values(self):
        # MoveIt returns an empty list when it cannot fetch the current robot state
        # (after waiting 1 s, holding the GIL)
        self.fetches += 1
        return self.current_joints

    def set_joint_value_target_from_pose_stamped(self, pose, eef, approx):
        # the real call dereferences a null state (segfault) without a current state
        assert self.current_joints, "C++ pose target without a current state"
        self.pose_targets += 1
        return True

    def set_joint_value_target_from_pose(self, pose, eef, approx):
        return self.set_joint_value_target_from_pose_stamped(pose, eef, approx)

    def set_joint_value_target(self, joints):
        return True


def controller(current_joints):
    c = MoveItBasicController.__new__(MoveItBasicController)  # no move_group needed
    c._g = FakeInterface(current_joints)
    return c


@pytest.mark.parametrize("target", [PoseStamped(), Pose()])
def test_pose_target_without_a_current_state_raises_instead_of_crashing(target):
    c = controller([])
    with pytest.raises(MoveItCommanderException):
        c.set_joint_value_target(target, "custom_end_link", False)
    assert c._g.pose_targets == 0


def test_pose_target_with_a_current_state_goes_through():
    c = controller([0.0] * 6)
    c.set_joint_value_target(PoseStamped(), "custom_end_link", False)
    assert c._g.pose_targets == 1


def test_after_a_failed_fetch_pose_targets_fail_at_once_for_a_while():
    c = controller([])
    for _ in range(10):  # RoboticArmAgent tries the IK 10 times in a row
        with pytest.raises(MoveItCommanderException):
            c.set_joint_value_target(PoseStamped(), "custom_end_link", False)
    assert c._g.fetches == 1  # one blocking fetch, not ten

    c._g.current_joints = [0.0] * 6  # the arm is back
    c._state_lost_at -= c.STATE_LOST_HOLD  # and the hold is over
    c.set_joint_value_target(PoseStamped(), "custom_end_link", False)
    assert c._g.fetches == 2 and c._g.pose_targets == 1


def test_joint_target_does_not_need_the_current_state():
    c = controller([])
    # MoveIt does not fetch the current state for a joint target
    c.set_joint_value_target([0.0] * 6)
