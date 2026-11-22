from std_vision.video import StdVideo
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--device", type=str, default="sim", help="sim or real device"
)
parser.add_argument("-g", "--go", action="store_true", help="go to the pick pose")
parser.add_argument("-ng", "--not_go", action="store_true", help="go to the pick pose")
parser.add_argument(
    "-na",
    "--not_auto_detect",
    action="store_true",
    help="do not auto detect sim or real",
)
args, unknown = parser.parse_known_args()


env_type = None
if not args.not_auto_detect:
    try:
        import rospy

        sim_player_nums = rospy.get_param("/airbot_play/sim_player_nums", default=0)
        sim_type = "sim" if sim_player_nums == 1 else "real"
        env_type = rospy.get_param("/env_type", default=None) or sim_type
    except:
        sim_type = args.device
else:
    sim_type: str = args.device

if sim_type in ["sim"] or args.go or env_type == "mujoco":
    from easy_arm_control.control import RoboticArmAgent, ChooseGripper
    import rospy

    # ROS节点初始化(仿真自动控制机械臂到抓取位置)
    NODE_NAME = "airbot_play_vision_hsv"
    rospy.init_node(NODE_NAME)
    other_config = None
    sim_player_nums = rospy.get_param("/airbot_play/sim_player_nums", default=0)
    sim = ["real", "gazebo_i"][sim_player_nums]
    if not env_type:
        print("Use", sim_type)
    else:
        print("Use", env_type)
    gripper = ["gripper", "inspire"][0]
    gripper_controller = ChooseGripper(sim_type, gripper)()
    arm = RoboticArmAgent(node_name=NODE_NAME, gripper=(4, gripper_controller))
    if not args.not_go:
        if sim == "gazebo":
            arm.set_and_go_to_pose_target(
                [0.2739, 0, -0.27], [0, 1.570796, 0], sleep_time=2
            )
        elif env_type == "mujoco":
            arm.set_and_go_to_pose_target(
                [0.2739, 0, 0.1], [0, 1.570796, 0], sleep_time=2
            )
            arm.gripper_control(0)

# 执行如下代码可以通过滑动条调整视觉参数
# 按ESC键退出
if sim_type == "sim" or not args.device.isdigit():
    HSV_L, HSV_H = StdVideo.color_thresh_determine("/camera/color/image_raw")
else:
    if sim_type.isdigit():
        device = int(sim_type)
    else:
        device = 2
    HSV_L, HSV_H = StdVideo.color_thresh_determine(device)
