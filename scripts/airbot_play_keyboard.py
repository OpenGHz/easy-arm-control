#!/usr/bin/env python3

import rospy
from airbot_play_control.control import RoboticArmAgent, ChooseGripper
import argparse
import json


def parse_config(path):
    pick_pose = None
    grasp_z = None

    with open(path, "r") as f:
        config = json.load(f)
    try:
        position_x = config["PICK_POSITION_X"]
        position_y = config["PICK_POSITION_Y"]
        position_z = config["PICK_POSITION_Z"]
        rotation_roll = config["PICK_ROLL"]
        rotation_pitch = config["PICK_PITCH"]
        rotation_yaw = config["PICK_YAW"]
        pick_pose = [position_x, position_y, position_z] + [
            rotation_roll,
            rotation_pitch,
            rotation_yaw,
        ]
        grasp_z = config["PICK_GRASP_Z"]
    except Exception as e:
        print(e)

    return pick_pose, grasp_z


def print_config_instruction():
    print("  移动到PICK_GRASP_Z高度：g键")
    print("  刷新配置(修改配置后需手动刷新生效)：f键")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("param set")
    parser.add_argument(
        "-ip",
        "--init_pose",
        action="store",
        type=str,
        help="forward, pick or json file path",
        default="",
    )  # 若要控制移动建议先到forward/pick姿态，避免Home姿态下的奇异点
    parser.add_argument(
        "-fk", "--fake", action="store_true", help="whether use fake control"
    )
    parser.add_argument(
        "-gt", "--gripper", type=str, default="gripper", help="gripper type"
    )  # inspire， gripper
    parser.add_argument(
        "-st", "--sim_type", type=str, default="real", help="sim type"
    )  # real, isaac, gazebo, ,gazebo_i, gibson
    # 这一步本质上是根据设定的变量从sys.argv中对已知和未知的参数进行分类，以及参数和对应值的提取
    args = parser.parse_args()
    """系统初始化"""
    acc_max, vel_max = 1, 1

    # ROS节点初始化
    NODE_NAME = "airbot_play_keyboard"
    rospy.init_node(NODE_NAME)
    other_config = None
    sim_type = args.sim_type
    init_pose = args.init_pose
    use_config = ".json" in init_pose

    if args.fake:
        other_config = ("", "airbot_play_arm")
    gripper_controller = ChooseGripper(sim_type, args.gripper)()
    arm = RoboticArmAgent(
        node_name=NODE_NAME, gripper=(4, gripper_controller), other_config=other_config
    )
    # go to init pose
    if init_pose == "forward":
        position = [0.4, 0, 0.3]
        rotation = [0, 0, 0]
        pick_pose = position + rotation
        arm.set_and_go_to_pose_target(position, rotation, target_base="0", sleep_time=1)
    elif init_pose == "pick":
        if sim_type == "isaac":
            pick_pose = [0.274, 0.0, -0.234] + [0, 1.570796, 0]
        elif sim_type == "real":
            pick_pose = [0.274, 0.0, 0.03] + [0, 1.570796, 0]
        elif sim_type == "gazebo":
            pick_pose = [0.274, 0.0, -0.27] + [0, 1.570796, 0]
    elif use_config:
        pick_pose, grasp_z = parse_config(init_pose)
        assert pick_pose is not None, "Please check the config file"
    elif "," in init_pose:
        pick_pose = [float(i) for i in init_pose.split(",")]
    elif init_pose != "":
        raise ValueError("init_pose should be forward, pick or json file path")
    if init_pose != "":
        pick_joint = arm.change_pose_to_joints(pick_pose)
        arm.go_to_named_or_joint_target(pick_joint, sleep_time=1)
        arm.gripper_control(0)  # open the gripper

    """ 键盘控制 """
    print("")
    arm.show_key_instruction()
    print_config_instruction(), print("")
    while arm.is_alive():
        key = arm.key_control(
            delta_task=0.005,
            delta_joint=0.01,
            acc_limit=acc_max,
            vel_limit=vel_max,
            continuous=False,
            instruction=False,
            disable_keys=["9"],
        )  # 33.33Hz
        if key == "9":
            print("Move to init pose")
            arm.go_to_named_or_joint_target(pick_joint, sleep_time=1)
        elif key == "g":
            if use_config:
                print("Move to PICK_GRASP_Z height")
                arm.go_to_single_axis_target(2, grasp_z, sleep_time=1)
            else:
                print(
                    "Please use json file to set grasp position using PICK_GRASP_Z key"
                )
        elif key == "f":
            if use_config:
                pick_pose_tmp, grasp_z_tmp = parse_config(init_pose)
                assert pick_pose_tmp is not None, "Please check the config file"
                if pick_pose == pick_pose_tmp and grasp_z == grasp_z_tmp:
                    print("Config not changed")
                else:
                    pick_pose = pick_pose_tmp
                    grasp_z = grasp_z_tmp
                    pick_joint = arm.change_pose_to_joints(pick_pose)
                    print("Config refreshed:")
                    print("  PICK_POSE:", pick_pose)
                    print("  PICK_GRASP_Z:", grasp_z)
            else:
                print("Please use json file to configure")
        elif key == "?":
            if use_config:
                print_config_instruction()
            print("")
