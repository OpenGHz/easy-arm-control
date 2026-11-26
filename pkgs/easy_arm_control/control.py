#!/usr/bin/env python3
try:  # 导入ROS包
    import rospy
    from geometry_msgs.msg import PoseStamped, Pose
    from shape_msgs.msg import SolidPrimitive
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    from moveit_commander import (
        MoveGroupCommander,
        MoveItCommanderException,
        RobotCommander,
    )
    from moveit_msgs.msg import (
        RobotState,
        Constraints,
        PositionConstraint,
        OrientationConstraint,
        TrajectoryConstraints,
        BoundingVolume,
    )
    from moveit_msgs.srv import (
        GetPositionFK,
        GetPositionFKRequest,
        GetPositionFKResponse,
        GetPositionIK,
        GetPositionIKRequest,
        GetPositionIKResponse,
    )
except ImportError as e:
    print(
        f"\r\n********************\r\n导入ROS相关包出错:\r\n{e}。\r\n将使用替代包。若不使用ROS可忽略。\r\n**********************\r\n"
    )
    # 导入ROS替代包
    # from airbot_play_commander import PlayGroupCommander
    # from airbot_play_ros_msgs import PoseStamped,Pose,JointState

from robot_tools.conversions import transformations

# 导入其它包
import numpy as np
import math, time
from typing import Any, Optional, Union, List, Dict, Tuple, Callable, Iterable
from threading import Thread
from copy import deepcopy, copy
import sys, os
import termios, tty, select


class CoordinateTools(object):
    """坐标系转换工具类"""

    @staticmethod
    def transform_as_matrix(position, rotation) -> np.ndarray:
        """将相对位姿转换为变换矩阵(rotation是四元数)"""
        trans_q = transformations.euler_matrix(*rotation)
        trans_t = transformations.translation_matrix(position)
        trans_TF = np.matmul(trans_t, trans_q)
        return trans_TF

    @classmethod
    def to_robot_coordinate(
        cls, target_in_world: tuple, robot_in_world: tuple
    ) -> tuple:
        """目标在世界坐标系下的位姿转换为在机器人坐标系下的位姿"""
        target_in_world = cls.transform_as_matrix(*target_in_world)
        robot_in_world = cls.transform_as_matrix(*robot_in_world)
        target_in_robot = np.matmul(np.linalg.inv(robot_in_world), target_in_world)
        t_scale, t_shear, t_angles, t_trans, t_persp = transformations.decompose_matrix(
            target_in_robot
        )
        return t_trans, np.array(t_angles, dtype=np.float64)

    @classmethod
    def to_world_coordinate(
        cls, target_in_robot: tuple, robot_in_world: tuple
    ) -> tuple:
        """目标在机器人坐标系下的位姿转换为在世界坐标系下的位姿"""
        target_in_robot = cls.transform_as_matrix(*target_in_robot)
        robot_in_world = cls.transform_as_matrix(*robot_in_world)
        target_in_world = np.matmul(robot_in_world, target_in_robot)
        t_scale, t_shear, t_angles, t_trans, t_persp = transformations.decompose_matrix(
            target_in_world
        )
        return t_trans, np.array(t_angles, dtype=np.float64)


class MoveItBasicController(MoveGroupCommander):
    """基于MoveIt! original controller 的 basic controller"""

    def __init__(
        self, name, robot_description="robot_description", ns="", wait_for_servers=5
    ):
        super().__init__(name, robot_description, ns, wait_for_servers)
        self.__robot = RobotCommander(robot_description, ns)
        self.__link_names = self.__robot.get_link_names(self.get_name())
        self.set_default_configs()
        self.__ns = ns
        self.__fk_service = rospy.ServiceProxy(self.__ns + "/compute_fk", GetPositionFK)
        self.__ik_service = rospy.ServiceProxy(self.__ns + "/compute_ik", GetPositionIK)

    # monotonic time of the last failed fetch of the current robot state
    _state_lost_at = float("-inf")
    STATE_LOST_HOLD = 1.0  # s: pose targets fail at once this long after a failed fetch

    def set_joint_value_target(self, arg1, arg2=None, arg3=None):
        """MoveGroupCommander.set_joint_value_target; for a pose target, first check
        that the current robot state can be fetched.

        For a pose target MoveIt (1.1) starts the IK from the current robot state. When
        it cannot fetch one (no joint states for 1 s: the arm service, the bridge node or
        the simulation stopped), it dereferences a null state and the whole process
        segfaults. Raise instead, so the caller sees a failed IK and the program can
        report the lost connection.

        A failed fetch blocks for 1 s without releasing the GIL (MoveIt's Python
        binding), so IK retries (10 in a row in RoboticArmAgent) would stall every other
        thread, e.g. a watchdog, for seconds. Within STATE_LOST_HOLD of a failed fetch,
        fail at once.
        """
        if isinstance(arg1, (Pose, PoseStamped)):
            msg = "没有机械臂的当前状态（关节状态 1 秒没有更新），无法求逆解"
            if time.monotonic() - self._state_lost_at < self.STATE_LOST_HOLD:
                raise MoveItCommanderException(msg)
            if not self.get_current_joint_values():
                self._state_lost_at = time.monotonic()
                raise MoveItCommanderException(msg)
        return super().set_joint_value_target(arg1, arg2, arg3)

    def get_link_names(self) -> list:
        return self.__link_names

    def get_name_space(self) -> str:
        return self.__ns

    def set_default_configs(self) -> None:
        """设置默认参数"""
        self.set_pose_reference_frame("base_link")
        # 规划时间、尝试次数等设置
        self.allow_replanning(False)  # 避障重规划
        self.set_planning_time(2)  # float ok
        self.set_num_planning_attempts(10)  # 规划失败尝试次数
        # 精度控制
        self.set_goal_position_tolerance(0.00005)  # 目标位置公差
        self.set_goal_orientation_tolerance(0.00001)  # 目标四元数公差
        self.set_goal_joint_tolerance(0.000017)  # 目标关节角度公差
        # 速度和加速度系数初始化
        self.set_max_acceleration_scaling_factor(1.0)  # 加速度限制
        self.set_max_velocity_scaling_factor(1.0)  # 速度限制

    def compute_forward_kinematics(
        self, joint_values: list, fk_link_names: list, reference_frame: str
    ) -> Tuple[tuple]:
        """正运动学：根据关节角度计算指定fk_link_names相对reference_frame的位姿"""
        robot_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.get_active_joints()
        joint_state.position = joint_values
        robot_state.joint_state = joint_state

        fk_request = GetPositionFKRequest()
        fk_request.header.stamp = rospy.get_rostime()
        fk_request.header.frame_id = reference_frame
        fk_request.fk_link_names = fk_link_names
        fk_request.robot_state = robot_state

        fk_response: GetPositionFKResponse = self.__fk_service.call(fk_request)
        if fk_response.error_code.val == fk_response.error_code.SUCCESS:
            list_pose7 = []
            for pose in fk_response.pose_stamped:
                list_pose7.append(
                    (
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                        pose.pose.orientation.x,
                        pose.pose.orientation.y,
                        pose.pose.orientation.z,
                        pose.pose.orientation.w,
                    )
                )
            return tuple(list_pose7)
        else:
            raise Exception(
                f"FK service failed with error code: {fk_response.error_code.val}"
            )

    # def compute_inverse_kinematics(self,pose:tuple,ik_link_name,reference_frame,time_out=1) -> Tuple[tuple]:
    #     """ 逆运动学：将pose(x,y,z,x,y,z,w)转换为joint角度目标 """
    #     pose = self.__tuple_to_pose(pose)
    #     ik_request = GetPositionIKRequest()
    #     ik_request.ik_request.group_name = self.get_name()
    #     ik_request.ik_request.ik_link_name = ik_link_name
    #     ik_request.ik_request.pose_stamped.header.frame_id = reference_frame
    #     ik_request.ik_request.pose_stamped.pose = pose
    #     ik_request.ik_request.avoid_collisions = True
    #     ik_request.ik_request.timeout = rospy.Duration(time_out)
    #     ik_response:GetPositionIKResponse = self.__ik_service.call(ik_request)
    #     if ik_response.error_code.val == ik_response.error_code.SUCCESS:
    #         return tuple(ik_response.solution.joint_state.name), tuple(ik_response.solution.joint_state.position)
    #     else:
    #         raise Exception(f"IK service failed with error code: {ik_response.error_code.val}")

    def compute_inverse_kinematics(self, pose: tuple, approximate=False) -> tuple:
        """逆运动学：将pose(x,y,z,x,y,z,w)转换为joint角度目标"""
        pose_type = Pose()
        pose_type.position.x, pose_type.position.y, pose_type.position.z = pose[:3]
        (
            pose_type.orientation.x,
            pose_type.orientation.y,
            pose_type.orientation.z,
            pose_type.orientation.w,
        ) = pose[3:]
        joints_target_last = list(self.get_joint_value_target())
        eef_link = self.get_end_effector_link()
        # if eef_link == "":
        #     print("No end effector link specified. Using the last link instead")
        #     eef_link = self.get_link_names()[-1]
        self.set_joint_value_target(pose_type, eef_link, approximate)
        joints_value = tuple(self.get_joint_value_target())
        self.set_joint_value_target(joints_target_last)
        return joints_value

    def __tuple_to_pose(self, pose: tuple) -> Pose:
        """将tuple转换为Pose"""
        pose_type = Pose()
        pose_type.position.x, pose_type.position.y, pose_type.position.z = pose[:3]
        (
            pose_type.orientation.x,
            pose_type.orientation.y,
            pose_type.orientation.z,
            pose_type.orientation.w,
        ) = pose[3:]
        return pose_type

    def get_joint_states_by_names(
        joint_names: tuple, all_joint_names: tuple, all_joint_values: tuple
    ) -> tuple:
        joint_states = [0 for _ in range(len(joint_names))]
        for i, name in enumerate(joint_names):
            joint_states[i] = all_joint_values[all_joint_names.index(name)]
        return tuple(joint_states)

    def get_joint_limits(self) -> tuple:
        pass


class RoboticArmAgent(object):
    """RAA类"""

    def __init__(
        self,
        control_mode=(-1, -1, -1),
        init_pose: Optional[int] = None,
        node_name: str = "",
        state_pub: float = 0,
        gripper: Optional[tuple] = None,
        other_config=None,
    ) -> None:
        """
        control_mode（控制模式，三元组：(plan_mode,excute_mode,target_mode））：
            plan_mode（规划模式）： -2 轨迹关节角度起点目标值为当前值；-1 轨迹关节角度起点目标值为上次目标值；0 全局无规划；1 临时无规划\r\n
            interpolate_mode（执行模式）：0 全局无插值；1 临时无插值；-1 有插值\r\n
            target_mode（对象模式）：-1 正常的real或sim机械臂 0：fake控制rviz中机械臂\r\n
            immediate_excute(快速执行，可选)：0 全局 1临时
        init_pose（初始位置）：
            0 Home姿态，1 Standby姿态，2 Forward姿态，3 η姿态，4 One姿态，5 一姿态，None当前姿态
        node_name（节点名称）：
            默认为空字符''，此时表示外部不初始化ROS，由该类自动初始化，此时节点名为airbot_play_node\r\n
            为None时表示不使用ROS，不会进行ROS初始化(暂不支持)\r\n
            为非空字符串时表示外部初始化，此时其值应与外部节点名保持一致
        state_pub（状态发布）：
            =0：不发布状态\r\n
            >0：发布状态（通过rosparam：/airbot_play/current_arm_state，发布频率等于state_pub的值）
        gripper（夹爪配置，None或二元组：(kind,sim_or_real)）：
            kind：0 二指夹爪；1 勾爪；2 吸盘；3 灵巧手（目前尚未支持灵巧手和真机的吸盘）\r\n
            sim_or_real：0或False 真机；1或True 仿真\r\n
            None：不使用上述夹爪
            func: 自定义夹爪控制函数
        other_config:
            默认为None，即无额外配置，使用默认配置
            可以用于配置命名空间和规划组名，是一个tuple:(name_space,group_name)
        """
        self.other_config = other_config
        self.product_name = "airbot_play"  # "airbot_play"
        # 根据节点名参数情况进行处理
        self.__ros_init(node_name)
        # 初始化控制
        self.__control_init(control_mode, gripper, node_name)
        # 初始化预置位姿字典
        self.__preset_pose_init()
        # 初始化位置
        self.__start_pose_init_(init_pose)
        # 当前状态发布
        self.__pub_arm_state(state_pub)

    def set_basic_controller(self, basic_controller):
        """设置基础控制器，默认支持MoveIt!和AIRobtPlay"""
        if isinstance(basic_controller, str):
            if self.other_config is None:
                product_name_space = "/" + self.product_name
                group_name = self.product_name + "_arm"
            else:
                if self.other_config[0] is not None:
                    product_name_space = self.other_config[0]
                if self.other_config[1] is not None:
                    group_name = self.other_config[1]
            if basic_controller == "moveit":
                self.__moveit_init(group_name, product_name_space)
            elif basic_controller == "airbot_play":
                self.__airbot_play_init(group_name)
        else:
            self.basic_c = basic_controller

        # 得到“活动”关节及所有link的名称及数量，不考虑虚拟的以及fixed的joints
        self.joints_name = self.basic_c.get_active_joints()
        self.joints_num = len(self.joints_name)
        self.links_name = self.basic_c.get_link_names()  # 所有坐标系
        self.links_num = len(self.links_name)
        self.__eef_check()  # 末端执行器检查
        self.target_pose.header.frame_id = (
            self.basic_c.get_pose_reference_frame()
        )  # 设置目标变量的参考系

    def set_plan_time_and_num(self, time=None, num=None):
        """设置规划尝试时间和失败重试次数"""
        if time is not None:
            self.basic_c.set_planning_time(time)  # float
        if num is not None:
            self.basic_c.set_num_planning_attempts(num)  # 规划失败尝试次数

    def set_plan_mode(self, mode=-1, wait=0.05):
        """设置规划模式mode：
         0：全局无plan
         1：临时无plan
        -1：有plan，start为1
        -2：有plan，start为0
        """
        self.__plan_mode = mode
        if mode < 0:
            self.start_base_mode = mode + 2
        if self.node_name is not None:
            if mode == 1 and wait > 0:  # 阻塞等待server清空1，否则可能造成重复不对等
                i = 0
                max_times = 20
                once = wait / max_times
                while (
                    rospy.get_param(
                        self.basic_c.get_name_space() + "/plan_mode", default=-1
                    )
                    == 1
                ):
                    rospy.sleep(once)
                    i += 1
                    if i == max_times:
                        break
            rospy.set_param(self.basic_c.get_name_space() + "/plan_mode", mode)
        else:
            pass

    def set_interpolate_mode(self, mode=-1, sleep=0, type_=None):
        """设置插值模式mode：
         0：全局无插值
         1：临时无插值
        -1：有插值。根据type_决定插值类型:
             None：采用默认的5次样条插值
             1：直线插值
             3：3次样条插值
        """
        if self.node_name is not None:
            rospy.set_param(self.basic_c.get_name_space() + "/no_interpolate", mode)
            if mode == -1 and type_ is not None:
                rospy.set_param(
                    self.basic_c.get_name_space() + "/interpolate_type", type_
                )
            if sleep > 0:
                rospy.sleep(sleep)
        else:
            pass

    def set_target_mode(self, target=-1):
        """设置执行的对象（target=0时采用fake执行）"""
        if self.node_name is not None:
            rospy.set_param(self.basic_c.get_name_space() + "/target_mode", target)
        else:
            pass

    def set_reference_frame(self, ref: Union[str, int] = 0):
        """设置参考坐标系: mode为0对应base_link，-1对应end_effector_link，其它对应links_name中的相应位置元素
        注意：参考坐标系若非0，将强制使用绝对目标设置，不能再使用相对目标设置（认为此时绝对目标就是相对目标）。
        """
        self.reference_mode = ref
        if ref == -1:  # 设为末端坐标系
            reference_frame = self.basic_c.get_end_effector_link()
        elif isinstance(ref, int):
            reference_frame = self.links_name[ref]
        else:
            reference_frame = ref
        self.basic_c.set_pose_reference_frame(reference_frame)

    def immediate_excute(self, mode, wait=0.05):
        """0直接发送目标命令（此时不论插值模式如何设置均不会进行插值）"""
        if self.node_name is not None:
            if mode == 1 and wait > 0:
                i = 0
                max_times = 20
                once = wait / max_times
                while (
                    rospy.get_param(
                        self.basic_c.get_name_space() + "/immediate_excute", default=-1
                    )
                    == 1
                ):
                    rospy.sleep(once)
                    i += 1
                    if i == max_times:
                        break
            rospy.set_param(self.basic_c.get_name_space() + "/immediate_excute", mode)
        else:
            pass

    def change_joints_to_pose(self, joints_value, print_flag=False):
        """正运动学：根据关节角度计算得到末端执行器的位姿"""
        xyz_xyzw = self.basic_c.compute_forward_kinematics(
            joints_value,
            [self.basic_c.get_end_effector_link()],
            self.basic_c.get_pose_reference_frame(),
        )[0]
        xyz = xyz_xyzw[:3]
        xyzw = xyz_xyzw[3:]
        rpy = list(self.quaternion_to_euler(xyzw))
        if print_flag:
            print(f"End effector position(xyz): {xyz}")
            print(f"End effector orientation(xyzw):{xyzw}")
            print(f"End effector euler(rpy):{rpy}")
        return xyz, rpy, xyzw

    def change_pose_to_joints(
        self, pose: Union[PoseStamped, Pose, list, tuple], approximate=False
    ) -> tuple:
        """逆运动学：将pose转换为joint角度目标"""
        pose_tuple = self.to_tuple7_pose(pose)
        return self.basic_c.compute_inverse_kinematics(pose_tuple, approximate)

    def change_list_to_pose(
        self,
        xyz: Union[list, tuple, None] = None,
        rot: Union[list, tuple, None] = None,
        update_mode=2,
        global_protect=True,
    ):
        """
        函数默认参数下，用户只需输入xyz和rot两个参数，将列表转换为Pose对象。rot可以是三个元素的rpy，也可以是四个元素的xyzw。
        注：global_protect为false情况为内部使用用户无需关心，将姿态列表转换为target_pose：
                        update_mode决定了只关心更新xyz(0)、rpy(1)，还是全部更新(2)，默认为2全部更新。
                        若不给定某个list，则自动通过相应的全局pose_list进行配置，然后返回None(基于xyz和xyzw)。
                    允许改动的全局list只有self.target_pose_position和self.target_pose_euler。
                    self.__target_pose_euler_displace和self.__target_pose_orientation只能通过特定函数自动更新。
        """
        if not global_protect:  # 修改全局，返回None
            if update_mode != 1:
                if xyz is not None:
                    (
                        self.target_pose.pose.position.x,
                        self.target_pose.pose.position.y,
                        self.target_pose.pose.position.z,
                    ) = xyz
                    self.target_pose_position = xyz
                else:
                    (
                        self.target_pose.pose.position.x,
                        self.target_pose.pose.position.y,
                        self.target_pose.pose.position.z,
                    ) = self.target_pose_position
            if update_mode != 0:
                if rot is not None:
                    if len(rot) == 3:
                        self.target_pose_euler = rot
                        self.target_pose_orientation = list(
                            self.euler_to_quaternion(rot)
                        )
                    else:
                        self.target_pose_orientation = rot
                        self.target_pose_euler = list(self.quaternion_to_euler(rot))
                    (
                        self.target_pose.pose.orientation.x,
                        self.target_pose.pose.orientation.y,
                        self.target_pose.pose.orientation.z,
                        self.target_pose.pose.orientation.w,
                    ) = self.target_pose_orientation
                else:
                    (
                        self.target_pose.pose.orientation.x,
                        self.target_pose.pose.orientation.y,
                        self.target_pose.pose.orientation.z,
                        self.target_pose.pose.orientation.w,
                    ) = self.target_pose_orientation
        else:  # 不修改全局，仅仅返回值，需要
            pose_temp = Pose()
            pose_temp.position.x, pose_temp.position.y, pose_temp.position.z = xyz
            if len(rot) == 3:
                xyzw = self.euler_to_quaternion(rot)
            else:
                xyzw = rot
            (
                pose_temp.orientation.x,
                pose_temp.orientation.y,
                pose_temp.orientation.z,
                pose_temp.orientation.w,
            ) = xyzw
            return pose_temp

    def change_pose_to_list(
        self, target_pose: Union[PoseStamped, Pose, None] = None, global_protect=True
    ):
        """将Pose格式转换为列表格式（用户无需关注global_protect参数）"""
        if target_pose is None:  # 基于全局更新
            if global_protect:
                return (
                    self.target_pose_position,
                    self.target_pose_euler,
                    self.target_pose_orientation,
                )
            else:
                self.target_pose_position = [
                    self.target_pose.pose.position.x,
                    self.target_pose.pose.position.y,
                    self.target_pose.pose.position.z,
                ]
                self.target_pose_orientation = [
                    self.target_pose.pose.orientation.x,
                    self.target_pose.pose.orientation.y,
                    self.target_pose.pose.orientation.z,
                    self.target_pose.pose.orientation.w,
                ]
                self.target_pose_euler = list(
                    self.quaternion_to_euler(self.target_pose_orientation)
                )
        else:  # 基于给定更新
            if isinstance(target_pose, PoseStamped):
                target_pose_pose_position = [
                    target_pose.pose.position.x,
                    target_pose.pose.position.y,
                    target_pose.pose.position.z,
                ]
                target_pose_pose_oritation = [
                    target_pose.pose.orientation.x,
                    target_pose.pose.orientation.y,
                    target_pose.pose.orientation.z,
                    target_pose.pose.orientation.w,
                ]
            else:
                target_pose_pose_position = [
                    target_pose.position.x,
                    target_pose.position.y,
                    target_pose.position.z,
                ]
                target_pose_pose_oritation = [
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                ]
            target_pose_pose_euler = list(
                self.quaternion_to_euler(target_pose_pose_oritation)
            )
            if not global_protect:  # 不保护全局，则修改全局值
                self.target_pose_position = target_pose_pose_position
                self.target_pose_euler = target_pose_pose_euler
                self.target_pose_orientation = target_pose_pose_oritation
                self.change_list_to_pose(
                    global_protect=False
                )  # 基于给定target影响全局list
            else:
                return (
                    target_pose_pose_position,
                    target_pose_pose_euler,
                    self.target_pose_orientation,
                )  # 若保护则返回转换后的列表

    def set_wait_mode(self, flag: bool):
        """flag为true时等待执行完毕"""
        self.wait_flag = flag

    def set_wait_tolerance(self, tolerance: float):
        """设置执行完标志的角度偏差容许度。范围0.1(5.72度)-1，单位rad"""
        min_tl, max_tl = 0.1, 1  # 最小容许度应以实际电机为准
        if tolerance > max_tl:
            self.__wait_tolrerance = max_tl
            print(f"容许公差不得超出{max_tl}rad，已自动限幅到{max_tl}rad")
        elif tolerance < min_tl:
            self.__wait_tolrerance = min_tl
            print(f"容许公差不得小于{min_tl}rad，已自动限幅到{min_tl}rad")
        else:
            self.__wait_tolrerance = tolerance

    def set_goal_tolerances(self, position=None, rotation=None, joint=None):
        """设置目标容差"""
        if position is not None:
            self.basic_c.set_goal_position_tolerance(
                position
            )  # 设定位置误差允许值,m;0.001为1mm的误差之内
        if rotation is not None:
            self.basic_c.set_goal_orientation_tolerance(
                rotation
            )  # 设定角度误差允许值,rad
        if joint is not None:
            self.basic_c.set_goal_joint_tolerance(joint)  # 即0.01度（关节空间规划）

    def set_move_speed(self, vel: float):
        """设定移动的恒定速度（单位rad/s，无规划模式下用到）"""
        self.const_move_speed = vel
        if self.node_name is not None:
            rospy.set_param(
                self.basic_c.get_name_space() + "/move_joint_speed", vel
            )  # 设置全局变量，立刻执行模式下或有用
        else:
            pass

    def joints_speed_limit(self, min_=0, max_=0):
        """
        机械臂全局的执行速度控制（控制电机底层转速，单位：rad）
        实际接收方是在follow_joint_trajectory_server.py文件中
        设置是一次性的，动作执行完后将恢复为默认配置（max_vel=2πdeg/s，min_vel=2.593）
        """
        if self.node_name is not None:
            rospy.set_param(self.basic_c.get_name_space() + "/vel_limit", (min_, max_))
        else:
            pass

    def set_position_constraint(self, constraint):
        """
        设定运动过程中的空间约束（即轨迹必须在某个特定空间内运动）。constraint由以下几个部分组成：
            0基准：'base_link'
            1形状：BOX = 1，SPHERE = 2，CYLINDER = 3，CONE = 4
            2尺寸：(a,b,c)
            3姿态：(x,y,z,r,p,y)
            4对象：'gripper_link'
            5名字：'any_name'
            综上：constraint = ['base_link',(x,y,z),1,(a,b,c),(x,y,z,r,p,y),'gripper_link']
        """
        if self.node_name is None:
            raise Exception("非ROS模式下暂不支持设置空间约束")
        # 确定约束空间范围的位姿
        pose = PoseStamped()
        pose.header.frame_id = self.basic_c.get_planning_frame()
        pose.pose.position.x = constraint[3][0]
        pose.pose.position.y = constraint[3][1]
        pose.pose.position.z = constraint[3][2]
        rpy = [constraint[3][4], constraint[3][3], constraint[3][5]]
        xyzw = self.euler_to_quaternion(rpy)
        pose.pose.orientation.x = xyzw[0]
        pose.pose.orientation.y = xyzw[1]
        pose.pose.orientation.z = xyzw[2]
        pose.pose.orientation.w = xyzw[3]
        # 确定空间形状与尺寸
        bbox_prism = SolidPrimitive()
        bbox_prism.type = constraint[1]
        bbox_prism.dimensions = constraint[2]
        # 确定空间位置与姿态
        bbox = BoundingVolume()  # 用于约束机械臂的规划的允许的空间
        bbox.primitives = [bbox_prism]  # 确定空间范围
        bbox.primitive_poses = [pose.pose]  # 确定物体的位姿
        # 创建位置约束
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = constraint[0]
        position_constraint.link_name = constraint[4]
        position_constraint.constraint_region = bbox
        position_constraint.weight = 1.0
        constraints = Constraints()
        constraints.name = constraint[5]
        constraints.position_constraints.append(position_constraint)
        self.basic_c.set_path_constraints(constraints)

    def set_oritation_constraint(self, constraint):
        """
        设定运动过程中的空间约束（即轨迹必须在某个特定空间内运动）。constraint由以下几个部分组成：
            0基准：'base_link'
            1姿态：(r,p,y)
            2对象：'gripper_link'
            3公差：(x,y,z)
            4名字：'any_name'
            综上：constraint = ['base_link',(x,y,z),1,(a,b,c),(x,y,z,r,p,y),'gripper_link']
        """
        # Create an orientation constraint for the right gripper
        if self.node_name is None:
            raise Exception("非ROS模式下暂不支持设置空间约束")
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = constraint[0]
        orientation_constraint.link_name = constraint[2]
        rpy = [constraint[1][1], constraint[1][0], constraint[1][2]]
        xyzw = self.euler_to_quaternion(rpy)
        orientation_constraint.orientation.x = xyzw[0]
        orientation_constraint.orientation.y = xyzw[1]
        orientation_constraint.orientation.z = xyzw[2]
        orientation_constraint.orientation.w = xyzw[3]
        orientation_constraint.absolute_x_axis_tolerance = constraint[3][0]
        orientation_constraint.absolute_y_axis_tolerance = constraint[3][1]
        orientation_constraint.absolute_z_axis_tolerance = constraint[3][2]
        orientation_constraint.weight = 1.0
        # Create a contraints list and give it a name;Append the constraint to the list of contraints
        constraints = Constraints()
        constraints.name = constraint[4]
        constraints.orientation_constraints.append(orientation_constraint)
        # Set the path constraints on the right_arm
        self.basic_c.set_path_constraints(constraints)

    def set_vel_acc_max_factor(self, v_m=None, a_m=None):
        """设置关节运动的最大速度和加速度(0-1之间的无量纲系数)"""
        if v_m is not None:
            self.basic_c.set_max_velocity_scaling_factor(v_m)
        if a_m is not None:
            self.basic_c.set_max_acceleration_scaling_factor(a_m)

    def clear_constraints(self):
        self.basic_c.clear_path_constraints()

    def store_named_target(
        self,
        name,
        joint_or_pose: Union[list, tuple, PoseStamped, Pose],
        *args,
        log_info=True,
    ):
        """存储目标位置（不同类型输入转换为joint角度存储）"""
        if isinstance(joint_or_pose, (PoseStamped, Pose)):
            joint_or_pose = self.change_pose_to_joints(joint_or_pose)
        self.preset_pose[name] = joint_or_pose
        if log_info:
            print("新存储的关节名称与角度为：")
            print(f"  {name} {joint_or_pose}")
        names_len = len(args)
        if names_len % 2 != 0:
            self.logwarn("store_named_taret的输入参数必须是偶数个")
        elif names_len > 0:
            for i in range(names_len / 2):
                if isinstance(args[i + 1], (PoseStamped, Pose)):
                    joint_target = self.change_pose_to_joints(args[i + 1])
                else:
                    target = args[i + 1]
                    len__ = len(target)
                    if (
                        len__ == 7 and len__ != self.joints_num
                    ):  # 这时认为传入的是xyz和xyzw组成的
                        target = self.change_list_to_pose(
                            target[0:3], target[3:7], global_protect=True
                        )  # 转换为posestamped格式
                        joint_target = self.change_pose_to_joints(target)
                    else:
                        joint_target = args[i + 1]
                self.preset_pose[args[i]] = joint_target
                if log_info:
                    print(f"  {args[i]} {joint_target}")

    def update_current_state(self, sleep_time=0, log_flag=False, get_pose=False):
        """得到当前的机械臂状态"""
        if sleep_time > 0:
            rospy.sleep(sleep_time)
        if self.only_joint_space:
            cp = PoseStamped()
        else:
            cp = self.basic_c.get_current_pose()
        self.current_xyz = [cp.pose.position.x, cp.pose.position.y, cp.pose.position.z]
        self.current_xyzw = [
            cp.pose.orientation.x,
            cp.pose.orientation.y,
            cp.pose.orientation.z,
            cp.pose.orientation.w,
        ]
        self.current_rpy = transformations.euler_from_quaternion(self.current_xyzw)
        self.current_joints = self.basic_c.get_current_joint_values()
        if log_flag:
            self.loginfo(
                "当前状态为: x:{:.2f} y:{:.2f} z:{:.2f} roll:{:.3f} pitch:{:.3f} yaw:{:.3f} (mm/rad)".format(
                    self.current_xyz[0] * 1000,
                    self.current_xyz[1] * 1000,
                    self.current_xyz[2] * 1000,
                    self.current_rpy[0],
                    self.current_rpy[1],
                    self.current_rpy[2],
                )
            )
        if get_pose:
            return cp

    def get_current_joint_values(self) -> list:
        """获取当前关节状态"""
        return self.basic_c.get_current_joint_values()

    def get_current_pose(
        self, pose_type: str = "tuple7"
    ) -> Union[PoseStamped, Pose, tuple]:
        """pose_type可选'posestamped'、'pose'、'tuple6'(xyzrpy)、'tuple7'(xyzwxyz)"""
        pose_stamped = self.update_current_state(get_pose=True)
        if pose_type == "posestamped":
            return pose_stamped
        elif pose_type == "pose":
            return pose_stamped.pose
        elif pose_type == "tuple6":
            return (*self.current_xyz, *self.current_rpy)
        elif pose_type == "tuple7":
            return (*self.current_xyz, *self.current_xyzw)
        raise Exception("pose_type参数错误")

    def check_pose(
        self, pose: Union[PoseStamped, Pose, list, tuple], approximate=True
    ) -> bool:
        """检查姿态是否合法"""
        target = self.to_pose_type(pose)
        return self.__set_pose_target(
            approximate=approximate, check=True, target=target, return_enable=True
        )

    def no_stop(self, flag=True):
        self.__no_stop = flag

    def go_to_named_or_joint_target(
        self,
        target: Union[str, list, dict],
        sleep_time: float = 0,
        vel_limit=1.0,
        acc_limit=1.0,
        start_base=1,
        info="",
        wait_flag=True,
        return_enable=False,
        disable_plan=False,
        retry_times=10,
    ):
        """
        到指定名称或指定关节角度（可任意关节）的目标
        start_base=0设置起始为当前，否则为last
        """
        while True:
            self.basic_c.set_max_acceleration_scaling_factor(acc_limit)  # 加速度限制
            self.basic_c.set_max_velocity_scaling_factor(vel_limit)  # 速度限制
            # 起始关节角设定
            if (
                start_base == 0 or self.wait_flag == False
            ):  # 若上次的等待标志不为true，则起始位置强制设置为当前位置，否则将会无法规划
                self.__set_start_state_to_current_state()
                self.wait_flag = wait_flag
                start_joints = self.current_joints
            else:
                self.__set_start_state_to_last_joints_target()
                start_joints = self.last_joint_target
            # 目标角度设定
            if isinstance(target, dict):
                target_temp = {}
                for key, value in target.items():
                    target_temp[self.joints_name[key]] = value
                target = target_temp
            elif isinstance(target, str):
                target = self.preset_pose[target]
            elif len(target) != self.joints_num:
                raise Exception(
                    f"关节角度设置异常，应为{self.joints_num}，实为{len(target)}"
                )
            try:
                self.basic_c.set_joint_value_target(
                    target
                )  # 指定角度目标（list时规定全部关节位置，dict时可以控制任意关节）
            except Exception as e:
                if return_enable:
                    print(f"关节角度{target}设置失败，请检查是否超出关节角度限制")
                    return False
                else:
                    raise Exception(
                        f"关节角度{target}设置失败，请检查是否超出关节角度限制"
                    )

            self.new_joint_target = (
                self.basic_c.get_joint_value_target()
            )  # 获取joint目标状态
            # 不进行规划，直接通过反解得到的关节空间目标执行（速度会较快，但大幅度姿态变换时容易发生机械臂自身的干涉碰撞问题）
            if disable_plan or self.__plan_mode >= 0:
                self.basic_c.stop()  # 避免多余动作Stop the current execution, if any
                self.__no_plan_excute([list(start_joints), list(self.new_joint_target)])
                self.wait_flag = wait_flag  # 更新全局waitflag
                (
                    self.target_pose_position,
                    self.target_pose_euler,
                    self.target_pose_orientation,
                ) = self.change_joints_to_pose(self.new_joint_target)
                self.change_list_to_pose(global_protect=False)
                self.__update_last_target()  # 执行成功才更新last
                self.basic_c.clear_pose_targets()  # It is always good to clear your targets after planning with poses
                if sleep_time > 0:
                    rospy.sleep(sleep_time)
                return True
            else:
                plan_success, traj, planning_time, error_code = self.basic_c.plan()
                if plan_success:  # 只用go而不用plan+excute则无法
                    self.basic_c.clear_pose_targets()  # It is always good to clear your targets after planning with poses
                    self.basic_c.stop()  # 避免多余动作Stop the current execution, if any
                    self.basic_c.execute(traj, wait=wait_flag)
                    self.wait_flag = wait_flag  # 更新全局waitflag
                    if info != "":
                        self.loginfo(info)
                    # 运动学正解设置pose目标值
                    if not self.only_joint_space:
                        (
                            self.target_pose_position,
                            self.target_pose_euler,
                            self.target_pose_orientation,
                        ) = self.change_joints_to_pose(self.new_joint_target)
                        self.change_list_to_pose(global_protect=False)
                    self.__update_last_target()
                    if sleep_time > 0:
                        rospy.sleep(sleep_time)
                    if retry_times < 10:
                        print(f"尝试次数{10 - retry_times}")
                    return True
                # 若失败则结束程序或返回false
                else:
                    self.basic_c.clear_pose_targets()  # It is always good to clear your targets after planning with poses
                    if retry_times == 0:
                        if return_enable:
                            return False
                        else:
                            raise Exception(
                                f"{self.node_name}程序退出：{target} Pose Failed"
                            )
                    else:
                        retry_times -= 1

    def go_to_any_joint_target(
        self,
        joint,
        inc_value,
        sleep_time: float = 0,
        vel_limit=1.0,
        acc_limit=1.0,
        start_base=0,
        target_base="l",
        info="",
        wait_flag=True,
        return_enable=False,
        no_plan=False,
    ):
        """
        任意数量关节移动函数封装；joint从机械臂底部向末端对应0-max；增量单位为rad；joint可以为int或list
        target_mode与工作空间类似，可以设置为current、last和0三种，前两种是增量式移动，后一种为绝对式移动
        """
        new_joints_target: Dict[list] = {}  # 使用字典方式指定目标关节
        # 单关节控制
        if isinstance(joint, int):
            if target_base in ["l", "last"]:
                new_target = self.last_joint_target[joint]
            elif target_base in ["c", "current"]:
                new_target = self.basic_c.get_current_joint_values()[joint]
            else:
                new_target = 0
            # 赋值
            if isinstance(inc_value, (list, tuple)):
                if target_base in [0, "0", "zero"]:
                    new_joints_target[joint] = inc_value  # 非增量式移动
                else:
                    new_joints_target[joint] = []
                    for v in inc_value:
                        new_target += v
                        new_joints_target[joint].append(new_target)
            else:
                new_joints_target[joint] = inc_value + new_target
        # 多关节控制(此时joint为list或tuple)
        else:
            # 设置角度基准值（三种模式）
            if target_base in ["l", "last"]:
                joint_base = copy(self.last_joint_target)
            elif target_base in ["c", "current"]:
                joint_base = self.basic_c.get_current_joint_values()
            else:
                joint_base = [0 for _ in range(self.joints_num)]
            # 赋值
            for i, j in enumerate(joint):
                if isinstance(
                    inc_value[i], (list, tuple)
                ):  # 关节的角度值是个列表，表示多值绝对或累加移动
                    if target_base in [0, "0", "zero"]:
                        new_joints_target[j] = inc_value[i]  # 非增量式移动
                    else:  # 增量值移动
                        new_joints_target[j] = []
                        for v in inc_value[i]:
                            joint_base[j] += v
                            new_joints_target[j].append(joint_base[j])
                else:
                    new_joints_target[j] = inc_value[i] + joint_base[j]
        # 设定目标并执行
        self.go_to_named_or_joint_target(
            new_joints_target,
            sleep_time,
            vel_limit,
            acc_limit,
            start_base,
            info,
            wait_flag,
            return_enable,
            disable_plan=no_plan,
        )

    def go_to_shift_single_axis_target(
        self,
        axis: int,
        inc_value: float,
        target_base: Union[str, int] = "last",
        vel_limit=1.0,
        acc_limit=1.0,
        sleep_time=0,
        info="",
        no_plan=False,
        approximate=False,
        wait_flag=True,
        return_enable=False,
    ):
        """单轴增量式移动函数封装。注意该函数是给定的目标值是增量值不是绝对值。绝对值请用：go_to_single_axis_target函数"""
        return self.go_to_shift_multi_axises_target(
            {axis: inc_value},
            target_base,
            vel_limit,
            acc_limit,
            sleep_time,
            info,
            no_plan=no_plan,
            approximate=approximate,
            wait_flag=wait_flag,
            return_enable=return_enable,
        )

    def go_to_shift_multi_axises_target(
        self,
        axises_value: Dict[int, float],
        target_base: Union[str, int] = "last",
        vel_limit=1.0,
        acc_limit=1.0,
        sleep_time=0,
        info="",
        no_plan=False,
        approximate=False,
        wait_flag=True,
        return_enable=False,
    ):
        """
        多轴增量式移动函数封装
        给定轴-值对来进行多轴移动,如{0:0.025};target_base_mode可以是'current'或'last'
        该函数基本上相当于：self.set_and_go_to_pose_target([],[],target_base_mode)将不需移动的设置为0即可。
        """
        if info != "":
            print(info)
        target_temp = [0 for _ in range(6)]
        for key, value in axises_value.items():
            target_temp[key] = value
        if target_base in ["zero", "z", "0", 0]:
            raise Exception("增量式移动目标基准模式不能配置为0")
        succeed = self.set_and_go_to_pose_target(
            target_temp[:3],
            target_temp[3:6],
            target_base,
            vel_limit=vel_limit,
            acc_limit=acc_limit,
            sleep_time=sleep_time,
            no_plan=no_plan,
            approximate=approximate,
            wait_flag=wait_flag,
            return_enable=return_enable,
        )
        return succeed

    def go_to_single_axis_target(
        self,
        axis,
        target,
        target_ref: str = "last",
        vel_limit=1.0,
        acc_limit=1.0,
        sleep_time=0,
        info="",
        no_plan=False,
        approximate=False,
        wait_flag=True,
    ):
        """单轴绝对式移动函数封装。注意该函数是给定的目标值是绝对值不是增量值。增量值请用： go_to_shift_single_axis_target"""
        if info != "":
            print(info)  # 可以打印一些信息
        self.basic_c.set_max_acceleration_scaling_factor(acc_limit)  # 加速度限制
        self.basic_c.set_max_velocity_scaling_factor(vel_limit)  # 速度限制
        self.basic_c.clear_pose_targets()  # It is always good to clear your targets after planning with poses
        # 获得基准(不论哪种模式均通过直接修改target_pose的方式完成)
        if target_ref in ["current", "c"]:
            self.update_current_state()
            self.target_pose_position, self.target_pose_euler = (
                list(self.current_xyz),
                list(self.current_rpy),
            )  # 使用当前的姿态作为基准
        elif target_ref not in ["last", "l"]:
            raise Exception("单轴移动目标基准模式不能配置为0")
        # 单轴修改
        if axis < 3:
            position = list(self.target_pose_position)
            position[axis] = target
            self.target_pose_position = tuple(position)
            self.change_list_to_pose(update_mode=0, global_protect=False)
        else:
            euler = list(self.target_pose_euler)
            euler[axis - 3] = target
            self.target_pose_euler = tuple(euler)
            self.change_list_to_pose(
                rot=self.target_pose_euler, update_mode=1, global_protect=False
            )
        # 执行动作
        self.go_to(
            self.start_base_mode,
            wait_flag=wait_flag,
            sleep_time=sleep_time,
            disable_plan=no_plan,
            approximate=approximate,
        )

    def set_target_pose(
        self,
        target_pos: Optional[Union[list, tuple]],
        target_rot: Optional[list] = None,
        target_base: Union[str, int] = 0,
    ) -> None:
        """
        设定单一目标的基准模式和相对偏差，得到绝对的最终目标变量
        目标姿态分为两部分：基准值和相对值。基准值的设定通过target_base_mode参数配置，共有三种配置：
            'current':当前姿态 'last':上次的目标值 'zero':0,从而相对值实际上意义变成了绝对姿态
            为避免出错，三者分别可以简写为c l z('0',0)
            一般增量方式（包括积分方式）采用last（推荐）或current，绝对移动采用0
            默认为last模式，因为last不受重力以及是否停稳的影响，是应首选的方式（跟踪时自动做好了失败回退）
        target_xyz,target_rpy中有一个为None时相应的目标pose不会进行任何改动，相当于可以方便独立实现xyz或rpy(xyzw)的控制,
        但允许target_xyz为6/7个元素，则后3/4个元素为旋转角度，此时rpy/xyzw的参数值将被覆盖。
        """
        if self.reference_mode != 0:
            target_base = 0  # 相对自身
        # 目标基准确定
        if target_base in ["zero", "z", "0", 0]:
            target_base_xyz, target_base_xyzw = (
                (0, 0, 0),
                (0, 0, 0, 1),
            )  # 按xyzw的单位(恒等)四元数
        elif target_base in ["current", "c"]:
            self.update_current_state()
            target_base_xyz, target_base_xyzw = self.current_xyz, self.current_xyzw
        elif target_base in ["last", "l"]:
            target_base_xyz, target_base_xyzw = self.last_xyz, self.last_xyzw
        else:
            raise Exception(
                f"目标基准模式设置错误，当前设置为:{target_base}，不符合要求"
            )
        # 最终目标确定
        if target_pos is not None:
            self.target_pose_position = [
                target_base_xyz[0] + target_pos[0],
                target_base_xyz[1] + target_pos[1],
                target_base_xyz[2] + target_pos[2],
            ]
            if len(target_pos) == 6:
                target_rot = target_pos[3:6]
            elif len(target_pos) == 7:
                target_rot = target_pos[3:7]
        if target_rot is not None:
            if len(target_rot) == 3:  # 将欧拉角转换为四元数
                target_rot = self.euler_to_quaternion(target_rot)
            # 四元数乘法计算目标rot
            self.target_pose_orientation = list(
                transformations.quaternion_multiply(target_rot, target_base_xyzw)
            )  # 基准四元数与目标四元数相乘，得到新的四元数
            self.target_pose_euler = list(
                self.quaternion_to_euler(self.target_pose_orientation)
            )
        # 维护姿态列表
        self.change_list_to_pose(global_protect=False)

    def go_to(
        self,
        start_base,
        wait_flag=True,
        approximate=False,
        show_plan_time=False,
        return_enable=False,
        constraints=None,
        disable_plan=False,
        sleep_time=0,
    ):
        """到达由set_target_pose设定的单一target
        return_enable表示当规划失败后是抛出异常还是返回False
        approximate为是否求近似逆解
        """
        # 设定轨迹约束(#TODO:暂不支持)
        if constraints is not None:
            self.basic_c.set_path_constraints(constraints)
        # 轨迹起点设定
        start_joints = self.__set_start_state(start_base, wait_flag)
        # 目标位姿运动学逆解（固定10次尝试）
        self.__set_pose_target(
            approximate=approximate, retry_times=10, return_enable=return_enable
        )
        # 最终规划与执行
        return self.__plan_and_execute(
            disable_plan, start_joints, wait_flag, sleep_time, return_enable
        )

    def set_and_go_to_pose_target(
        self,
        target_pos: Union[Iterable, Pose, PoseStamped, None],
        target_rot: Optional[Iterable] = None,
        target_base: Union[str, int] = 0,
        sleep_time=0,
        wait_flag=True,
        start_base=None,
        return_enable=False,
        constraints=None,
        vel_limit=1.0,
        acc_limit=1.0,
        no_plan=False,
        approximate=False,
    ):
        """
        设定单一目标的基准和相对偏差，得到绝对的最终目标变量
        目标姿态分为两部分：基准值和相对值。基准值的设定通过target_base_mode参数配置，共有三种配置：
            'current':当前姿态 'last':上次的目标值 'zero':0,从而相对值实际上意义变成了绝对姿态
            为避免出错，三者分别可以简写为c l z('0',0)
            一般增量方式（包括积分方式）采用last（推荐）或current，绝对移动采用0
            默认为last模式，因为last不受重力以及是否停稳的影响，是应首选的方式（跟踪时自动做好了失败回退）
        target_pos,target_rot中有一个为None时相应的目标pose不会进行任何改动，相当于可以方便独立实现xyz或rpy(xyzw)的控制,
        但允许target_pos为6/7个元素的tuple或list，以及Pose或PoseStamped类型指定完整pose，此时rpy/xyzw的参数值将无效。
        """
        self.set_vel_acc_max_factor(vel_limit, acc_limit)  # 速度和加速度限制
        if isinstance(target_pos, (PoseStamped, Pose)):
            target_pos = self.to_tuple7_pose(target_pos)
        self.set_target_pose(target_pos, target_rot, target_base)
        if start_base is None:
            start_base = self.start_base_mode  # 设置为全局模式
        succeed = self.go_to(
            start_base,
            wait_flag=wait_flag,
            approximate=approximate,
            return_enable=return_enable,
            constraints=constraints,
            sleep_time=sleep_time,
            disable_plan=no_plan,
        )
        return succeed

    def __prepare_go_to_way_points(
        self,
        way_points: Union[list, tuple],
        start_base=1,
        vel_limit=1.0,
        acc_limit=1.0,
        point_type="list",
    ):
        # 速度和加速度限制
        self.set_vel_acc_max_factor(vel_limit, acc_limit)  # 速度和加速度限制
        # 起始设定
        if start_base == 0:
            self.__set_start_state_to_current_state()
        else:
            self.__set_start_state_to_last_joints_target()
        # 目标处理(统一转换成Pose或list类型)
        if point_type == "pose":
            way_points = [self.to_pose_type(point) for point in way_points]
            # 终点维护
            self.target_pose.pose = way_points[-1]
            self.change_pose_to_list(global_protect=False)
            # 插入waypoints起点为当前位置，保证轨迹正确执行（因为第一个点应该尽可能接近当前位置否则容易失败）
            if start_base == 0:
                way_points.insert(0, self.update_current_state(get_pose=True).pose)
            else:
                way_points.insert(0, self.last_target_pose.pose)
        elif point_type == "list":
            way_points = [self.to_list7_pose(point) for point in way_points]
            # 终点维护
            end_way_point = way_points[-1]
            self.target_pose_position = end_way_point[:3]
            self.target_pose_orientation = end_way_point[3:]
            self.change_list_to_pose(global_protect=False)
            # 插入waypoints起点为当前位置，保证轨迹正确执行（因为第一个点应该尽可能接近当前位置否则容易失败）
            if start_base == 0:
                way_points.insert(0, list(self.get_current_pose()))
            else:
                way_points.insert(0, [*self.last_xyz, *self.last_xyzw])
        else:
            raise Exception("point_type参数错误")
        # 逆解维护
        self.__set_pose_target_once()
        self.basic_c.clear_pose_targets()  # 清空在上面函数中设定的目标值，因为实际规划没用它
        return way_points

    def get_last_plan_time(self, show=False):
        if show:
            self.loginfo(f"规划用时为{self.__last_plan_time * 1000}ms.")
        return self.__last_plan_time

    def set_and_go_to_way_points(
        self,
        way_points: Union[list, tuple],
        start_base=1,
        vel_limit=1.0,
        acc_limit=1.0,
        sleep_time=0,
        wait_flag=True,
        allowed_fraction=1,
        return_enable=False,
        retry_times=20,
    ):
        """
        设定笛卡尔路点列表并移动。way_points是一个列表，其中包含要经过的位姿，位姿格式为[x,y,z,r,p,y]或[x,y,z,x,y,z,w]或Pose（推荐）或PoseStamped（列表中位姿类型可混用）。
        使用路点的方便性在于可以比较丝滑地一次性执行多段轨迹，缺点在于，路点的过程中的速度、加速度限制各段都是一样的，不能进行更加精细的控制。
        路点的第一个点将自动设置，start_base=1时设置为上次目标值，start_base=0时设置为当前值。
        """
        way_points = self.__prepare_go_to_way_points(
            way_points, start_base, vel_limit, acc_limit, point_type="pose"
        )
        # 轨迹规划
        attempts = 0  # 已经尝试规划次数
        fraction = 0  # 路径规划覆盖率
        while fraction < allowed_fraction and attempts < retry_times:
            (traj, fraction) = self.basic_c.compute_cartesian_path(
                way_points,  # waypoint poses，路点列表
                0.01,  # eef_step，终端步进值
                0.0,  # jump_threshold，跳跃阈值
                True,
            )  # avoid_collisions，避障规划
            # 尝试次数累加
            attempts += 1
            # 打印规划进程
            if attempts % 2 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")
        # 如果路径规划成功（覆盖率100%）,则开始控制机械臂运动
        if fraction >= allowed_fraction:
            print(fraction)
            self.basic_c.stop()
            self.basic_c.execute(traj, wait=wait_flag)
            self.__update_last_target()
            if sleep_time > 0:
                rospy.sleep(sleep_time)
            return True
        # 如果路径规划失败，则打印失败信息
        else:
            if return_enable:
                self.loginfo(
                    f"笛卡尔空间规划失败, fraction={fraction}，共尝试{retry_times}次。"
                )
                return False
            else:
                raise Exception(
                    f"笛卡尔空间规划失败, fraction={fraction}，共尝试{retry_times}次。"
                )

    def stop_move(self):
        """立刻停止运动"""
        cmd_publisher = rospy.Publisher(
            "/airbot_play/joint_cmd", JointState, queue_size=10
        )
        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "airbot_play"
        self.update_current_state()
        cmd.position = self.current_joints
        cmd.velocity = [0 for _ in range(self.joints_num)]
        cmd.effort = [50 for _ in range(self.joints_num)]
        self.basic_c.stop()  # 停止moveit未必真的管用，比如在不进行插值的时候
        for _ in range(20):  # 通过发送之前获取的某个时刻的位置来保证停止
            cmd_publisher.publish(cmd)
            rospy.sleep(0.005)

    def key_get(self, mode: int, timeout=None, print_key=True, pause=False):
        """
        获取键盘输入。模式（mode）设置：
             模式为0时根据timeout来等待获取键值，然后返回键名，函数结束。默认timeout=None一直阻塞等待直到结束;
             模式为1时相当于给模式0在函数中套了个循环，该函数不返回值，而是将值存储起来，通过模式3可以获取该值;
             模式为2时首次调用该函数将启动一个子线程，在子线程中无限循环获取键值，此时timeout始终为none（为防止重复初始化线程，加入了保护机制，只有首次调用该函数时才能启动线程，后续均无效）;
             模式为3时将直接返回已经存储的键值。在无线程情况下意味着返回上次调用模式0或1时的键值，而在有子线程的情况下则意味着返回子线程最近更新的键值。
         pause为true时，利用连续读取两个按键的特点，程序将在首次按下按键后一直暂停，直到另一个按键按键，从而可以实现程序的线程暂停功能。
        """
        if not hasattr(self.key_get, "key"):
            self.key_get.__dict__["key"] = None
            if mode == 2:
                Thread(target=self.key_get, daemon=True, args=(1, None)).start()
                return
        while True:
            if mode in [2, 3]:
                return self.key_get.key
            # *****************************************
            settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                if pause:
                    key = sys.stdin.read(
                        2
                    )  # sys.stdin.read() returns a string on Linux
                else:
                    key = sys.stdin.read(1)
            else:
                key = ""
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            if key == chr(3):
                exit("Ctrl-C 按下，程序退出")
            # *****************************************
            if print_key:
                print(f"key={key}")
            self.key_get.__dict__["key"] = key
            if mode == 0:
                return key

    def show_key_instruction(self):
        print("可以使用键盘控制AIRBOT_Play机械臂，按键说明如下（默认为工作空间）：")
        print(
            "  工作空间控制：(按t键切换为关节空间)\r\n   w/s:前进/后退\r\n   a/d:左移/右移\r\n   q/e:上升/下降\r\n   j/l:yaw左/右\r\n   i/k:pitch上/下\r\n   u/o:roll逆/顺"
        )
        print(
            "  关节空间控制：(按t键切换为工作空间)\r\n   w/s:关节0\r\n   a/d:关节1\r\n   q/e:关节2\r\n   j/l:关节3\r\n   i/k:关节4\r\n   u/o:关节5"
        )
        print("  夹爪控制：p键")
        print("  控制步长减小和增加：[键和]键 ; r键可恢复默认步长")
        print("  回到Home位置：0键")
        print("  回到初始位置：9键")
        print("  显示机器人状态：=键")
        print("  显示按键说明：?键")
        print("  退出程序：Ctrl+C")

    def key_control(
        self,
        delta_task=0.01,
        delta_joint=0.0174,
        key=None,
        print_key=False,
        acc_limit=1.0,
        vel_limit=1.0,
        continuous=False,
        disable_keys: list = None,
        instruction=False,
    ) -> str:
        """
            将key转换为控制指令:\n
                delta_task为每次按键移动的工作空间距离值（工作空间步长），单位为m/rad；\n
                delta_joint为每次按键移动的关节空间距离值（关节空间步长），单位rad；\n
                key为按键值，若为None则自动获取；\n
                print_key为是否打印按键值；\n
                acc_limit为加速度限制；vel_limit为速度限制\n
                continuous为是则只进行逆解，不进行插值与规划，适合在步长很小的时候使用。目前不支持Gazebo使用。\n
                disable_keys为禁用的按键列表，若按下这些按键则不会进行任何操作，但是会返回按键值。\n
            所有的按键key都会作为返回值返回；\\
            共用到如下key：qweasd、uiojkl、ptr、09、=、[]、?
        """

        if not hasattr(self.key_control, "instruction") and instruction:
            self.key_control.__dict__["instruction"] = True
            self.show_key_instruction()

        def limit_and_print():
            if self.key_control.mode == 1:
                max_delta = 0.2
                if self.key_control.__dict__["delta"][1] > max_delta:
                    self.key_control.__dict__["delta"][1] = max_delta
                    print(f"上限保护，joint_delta限制在{max_delta}rad内")
                print(f"当前步长delta_joint={self.key_control.__dict__['delta'][1]}rad")
            else:
                max_delta = 0.1
                if self.key_control.__dict__["delta"][0] > max_delta:
                    self.key_control.__dict__["delta"][0] = max_delta
                    print(f"上限保护，task_delta限制在{max_delta}m/rad内")
                print(
                    f"当前步长delta_task={self.key_control.__dict__['delta'][0]}m/rad"
                )

        if not hasattr(self.key_control, "mode"):
            self.__key_to_target = {
                "w": 1,
                "a": 2,
                "s": -1,
                "d": -2,
                "q": 3,
                "e": -3,
                "u": -4,
                "o": 4,
                "i": -5,
                "k": 5,
                "j": 6,
                "l": -6,
            }
            self.key_control.__dict__["mode"] = 0
            self.key_control.__dict__["delta"] = [delta_task, delta_joint]
            # print(f"初始步长为：delta_task={delta_task}m/rad，delta_joint={delta_joint}rad")
        if key is None:
            key = self.key_get(0, print_key=print_key)
        if disable_keys is not None and key in disable_keys:
            return key
        if key == "t":
            self.key_control.__dict__["mode"] ^= 1
            if self.key_control.__dict__["mode"] == 0:
                print(
                    f"控制空间切换为工作空间，当前步长delta_task={self.key_control.__dict__['delta'][0]}m/rad"
                )
            else:
                print(
                    f"控制空间切换为关节空间，当前步长delta_joint={self.key_control.__dict__['delta'][1]}rad"
                )
        elif key == "p":
            if not hasattr(self.gripper_control, "state"):
                self.gripper_control.__dict__["state"] = "closed"
            if self.gripper_control.__dict__["state"] == "opened":
                self.gripper_control.__dict__["state"] = "closed"
                cmd = 0
            else:
                self.gripper_control.__dict__["state"] = "opened"
                cmd = 1
            # print(cmd)
            self.gripper_control(cmd, sleep_time=0)
        elif key == "r":
            self.key_control.__dict__.pop("mode")
            print("r键按下，控制步长恢复默认值")
        elif key == "[":
            if not continuous:
                self.key_control.__dict__["delta"][
                    self.key_control.__dict__["mode"]
                ] /= 10
                limit_and_print()
        elif key == "]":
            if not continuous:
                self.key_control.__dict__["delta"][
                    self.key_control.__dict__["mode"]
                ] *= 10
                limit_and_print()
        elif key == "0":
            self.immediate_excute(-1), self.set_plan_mode(-1)
            self.go_to_named_or_joint_target("Home", sleep_time=1)
        elif key == "9":
            self.immediate_excute(-1), self.set_plan_mode(-1)
            self.go_to_named_or_joint_target("n", sleep_time=1)
        elif key == "=":
            self.log_arm_info(
                target_joint=True,
                target_pose=True,
                current_joint=True,
                current_pose=True,
                raw_info=True,
            )
        elif key == "?":
            self.show_key_instruction()
        elif self.__key_to_target.get(key) is None:
            return key
        else:
            if continuous:  # 连续控制
                self.immediate_excute(0), self.set_plan_mode(0)
            target = self.__key_to_target[
                key
            ]  # 借助dict完成了类似C语言的switch-case功能，保证了代码的简洁高效
            target_abs = abs(target)
            if self.key_control.mode == 0:  # 工作空间控制
                if not continuous:
                    delta_task = self.key_control.__dict__["delta"][0]
                else:
                    delta_task = 0.0025
                delta_task *= target / target_abs  # 正负号控制
                self.go_to_shift_single_axis_target(
                    target_abs - 1,
                    delta_task,
                    acc_limit=acc_limit,
                    vel_limit=vel_limit,
                    wait_flag=False,
                    return_enable=True,
                    no_plan=continuous,
                )
            else:  # 关节空间控制
                if not continuous:
                    delta_joint = self.key_control.__dict__["delta"][1]
                else:
                    delta_joint = 0.05
                delta_joint *= target / target_abs
                self.go_to_any_joint_target(
                    target_abs - 1,
                    delta_joint,
                    acc_limit=acc_limit,
                    vel_limit=vel_limit,
                    wait_flag=False,
                    return_enable=True,
                    no_plan=continuous,
                )
        return key

    def log_arm_info(
        self,
        current_pose=False,
        target_pose=False,
        current_joint=False,
        target_joint=False,
        sleep=0,
        rot_mode=0,
        raw_info=False,
    ):
        """打印机械臂的一些状态信息"""
        if sleep > 0:
            rospy.sleep(sleep)
        if current_joint or current_pose:
            self.update_current_state()
        if current_pose:  # 当前位姿
            if raw_info:  # 打印全部原始数据
                self.loginfo(
                    "当前位姿: xyz: [{:.4f},{:.4f},{:.4f}] rpy: [{:.3f},{:.3f},{:.3f}] xyzw: [{:.3f},{:.3f},{:.3f},{:.3f}] (m/rad)".format(
                        self.current_xyz[0],
                        self.current_xyz[1],
                        self.current_xyz[2],
                        self.current_rpy[0],
                        self.current_rpy[1],
                        self.current_rpy[2],
                        self.current_xyzw[0],
                        self.current_xyzw[1],
                        self.current_xyzw[2],
                        self.current_xyzw[3],
                    )
                )
            elif rot_mode == 0:  # 关注rpy
                self.loginfo(
                    "当前位姿: x:{:.4f} y:{:.4f} z:{:.4f} roll:{:.3f} pitch:{:.3f} yaw:{:.3f} (m/rad)".format(
                        self.current_xyz[0],
                        self.current_xyz[1],
                        self.current_xyz[2],
                        self.current_rpy[0],
                        self.current_rpy[1],
                        self.current_rpy[2],
                    )
                )
            elif rot_mode == 1:  # 关注xyzw
                self.loginfo(
                    "当前位姿: x:{:.4f} y:{:.4f} z:{:.4f} rot_x:{:.3f} rot_y:{:.3f} rot_z:{:.3f} rot_w:{:.3f} (m/rad)".format(
                        self.current_xyz[0],
                        self.current_xyz[1],
                        self.current_xyz[2],
                        self.current_xyzw[0],
                        self.current_xyzw[1],
                        self.current_xyzw[2],
                        self.current_xyzw[3],
                    )
                )
            elif rot_mode == 2:  # 都关注
                self.loginfo(
                    "当前位姿: x:{:.4f} y:{:.4f} z:{:.4f} roll:{:.3f} pitch:{:.3f} yaw:{:.3f} rot_x:{:.3f} rot_y:{:.3f} rot_z:{:.3f} rot_w:{:.3f} (m/rad)".format(
                        self.current_xyz[0],
                        self.current_xyz[1],
                        self.current_xyz[2],
                        self.current_rpy[0],
                        self.current_rpy[1],
                        self.current_rpy[2],
                        self.current_xyzw[0],
                        self.current_xyzw[1],
                        self.current_xyzw[2],
                        self.current_xyzw[3],
                    )
                )
        if target_pose:  # 目标位姿
            if raw_info:  # 打印全部原始数据
                self.loginfo(
                    "目标位姿: xyz: [{:.4f},{:.4f},{:.4f}] rpy: [{:.3f},{:.3f},{:.3f}] xyzw: [{:.3f},{:.3f},{:.3f},{:.3f}] (m/rad)".format(
                        self.last_xyz[0],
                        self.last_xyz[1],
                        self.last_xyz[2],
                        self.last_rpy[0],
                        self.last_rpy[1],
                        self.last_rpy[2],
                        self.last_xyzw[0],
                        self.last_xyzw[1],
                        self.last_xyzw[2],
                        self.last_xyzw[3],
                    )
                )
            elif rot_mode == 0:
                self.loginfo(
                    "目标位姿：x:{:.4f} y:{:.4f} z:{:.4f} roll:{:.3f} pitch:{:.3f} yaw:{:.3f} (m/rad)".format(
                        self.last_xyz[0],
                        self.last_xyz[1],
                        self.last_xyz[2],
                        self.last_rpy[0],
                        self.last_rpy[1],
                        self.last_rpy[2],
                    )
                )
            elif rot_mode == 1:
                self.loginfo(
                    "目标位姿：x:{:.4f} y:{:.4f} z:{:.4f} rot_x:{:.3f} rot_y:{:.3f} rot_z:{:.3f} rot_w:{:.3f} (m/rad)".format(
                        self.last_xyz[0],
                        self.last_xyz[1],
                        self.last_xyz[2],
                        self.last_xyzw[0],
                        self.last_xyzw[1],
                        self.last_xyzw[2],
                        self.last_xyzw[3],
                    )
                )
            elif rot_mode == 2:
                self.loginfo(
                    "目标位姿：x:{:.4f} y:{:.4f} z:{:.4f} roll:{:.3f} pitch:{:.3f} yaw:{:.3f} rot_x:{:.3f} rot_y:{:.3f} rot_z:{:.3f} rot_w:{:.3f} (m/rad)".format(
                        self.last_xyz[0],
                        self.last_xyz[1],
                        self.last_xyz[2],
                        self.last_rpy[0],
                        self.last_rpy[1],
                        self.last_rpy[2],
                        self.last_xyzw[0],
                        self.last_xyzw[1],
                        self.last_xyzw[2],
                        self.last_xyzw[3],
                    )
                )
        if current_joint:
            self.loginfo("当前关节值：{}rad".format(self.current_joints))
        if target_joint:
            self.loginfo("目标关节值：{}rad".format(self.last_joint_target))

    class _GripperControl(object):
        def __init__(self, gripper) -> None:
            """各类支持的夹爪控制接口"""
            if gripper is None:
                gripper = (None, None)
            self.type = gripper[0]
            self.sim_or_func = gripper[1]

        def __call__(self) -> Callable:
            """返回夹爪控制函数"""
            if self.type == 0:
                return self.two_finger_gripper
            elif self.type == 1:
                return self.dog_gripper
            elif self.type == 2:
                return self.surface_gripper
            elif self.type == 3:
                return self.flexible_hand
            elif self.type == 4:
                return self.sim_or_func  # 类型为4时，gripper的第二个元素为夹爪控制函数
            elif self.type is None:
                return self.no_gripper

        def dec_to_hex_string(self, dec_num: int, lenth=2):
            """将10进制数转为16进制的字符串表示"""
            if lenth == 2:
                hex_str = (
                    hex(dec_num)[2:].zfill(4).upper()
                )  # 转为16进制，高位补0，转为大写（此时低位在左，高位在右）
                byte1 = hex_str[2:]  # 取高8位
                byte2 = hex_str[:2]  # 取低8位
                return byte1 + " " + byte2  # 此时高位在左，低位在右
            elif lenth == 1:
                hex_str = (
                    hex(dec_num)[2:].zfill(2).upper()
                )  # 转为16进制，高位补0，转为大写
                return hex_str

        def hex_string_sum(self, hex_string: str, only_low=True):
            """输入含16进制元素的字符串，返回各元素相加后的8位16进制数，高位自动补0"""
            hex_list = hex_string.split()  # 将字符串按空格分割成列表
            hex_sum = 0
            for hex_num in hex_list:
                hex_sum += int(hex_num, 16)  # 将每个16进制数转换成10进制数并相加
            if only_low:
                hex_sum = hex_sum & 0xFF
            return hex(hex_sum)[2:].zfill(2).upper()

        def no_gripper(
            self,
            pp,
            sleep_time=0,
            speed=500,
            force=100,
            always=False,
            show_state=True,
            set_gap=None,
        ):
            """
            没有夹爪时的模拟输出
            """
            if pp == 1:
                print("no_gripper: 夹爪闭合")
            else:
                print("no_gripper: 夹爪张开")
            time.sleep(sleep_time)

        def two_finger_gripper(
            self,
            pp,
            sleep_time=0,
            speed=500,
            force=100,
            always=False,
            show_state=True,
            set_gap=None,
        ):
            """
            二指夹爪（因时科技）常用控制接口：
            1抓取，0释放；sleep_time是执行动作后的延时(s)
            speed是从1 - 1000 的无量纲速度系数
            force是从50 - 1000 单位为g(克)
            always为True表示持续施加给定的夹取力
            seek_pos为None时默认张开和闭合的开口度没有限制，不为None时取值为0-70mm。如果当前开口度小于设定开口度，以设定速度松开直到开口度达到目标开口度后停止运动；
                如果当前开口度大于设定开口度，以设定速度和力控阈值去夹取，当夹持力超过设定的力控阈值后，或者开口度达到目标开口度后停止运动。
            set_gap为None时不重新设置夹爪的最大和最小开口参数，为一个二元组时为(min,max)值。
            """
            VID_PID = "6790_29987"  # 因时二指夹爪的vid和pid（至少目前买的批次是这样）
            HEAD = "EB 90 "
            if set_gap is not None:
                set_gap = list(set_gap)
                if set_gap[0] > 1000:
                    set_gap[0] = 1000
                elif set_gap[0] < 0:
                    set_gap[0] = 0
                if set_gap[1] > 1000:
                    set_gap[1] = 1000
                elif set_gap[1] < 0:
                    set_gap[1] = 0
                min_gap = self.dec_to_hex_string(set_gap[0])
                max_gap = self.dec_to_hex_string(set_gap[1])
                cmd_temp = "01 05 12 " + max_gap + " " + min_gap
                SUM = self.hex_string_sum(cmd_temp)
                SET_GAP_CMD = HEAD + cmd_temp + " " + SUM
                RoboticArmAgent.ToSerial(SET_GAP_CMD, VID_PID, from_hex=True)
            if pp == 1:
                self.two_finger_gripper.__dict__["state"] = "closed"
                if show_state:
                    print("夹爪闭合")
                if self.sim_or_func in ["isaac", 1]:
                    rospy.set_param("/pick_place_flag", "pick")
                elif self.sim_or_func == "gazebo":
                    pass
                elif self.sim_or_func == "gibson":
                    pass
                else:
                    if speed > 1000:
                        speed = 1000
                    elif speed < 1:
                        speed = 1
                    if force > 1000:
                        force = 1000
                    elif force < 50:
                        force = 50
                    speed_str = self.dec_to_hex_string(speed)
                    force_str = self.dec_to_hex_string(force)
                    if not always:
                        cmd_temp = "01 05 10 " + speed_str + " " + force_str
                    else:
                        cmd_temp = "01 05 18 " + speed_str + " " + force_str
                    SUM = self.hex_string_sum(cmd_temp)
                    CLOSE_CMD = HEAD + cmd_temp + " " + SUM
                    RoboticArmAgent.ToSerial(CLOSE_CMD, VID_PID, from_hex=True)
            else:
                self.two_finger_gripper.__dict__["state"] = "opened"
                if show_state:
                    print("夹爪张开")
                if self.sim_or_func in ["isaac", 1]:
                    rospy.set_param("/pick_place_flag", "place")
                else:
                    speed_str = self.dec_to_hex_string(speed)
                    cmd_temp = "01 03 11 " + speed_str
                    SUM = self.hex_string_sum(cmd_temp)
                    OPEN_CMD = HEAD + cmd_temp + " " + SUM
                    RoboticArmAgent.ToSerial(OPEN_CMD, VID_PID, from_hex=True)
            rospy.sleep(
                sleep_time
            )  # 提示：根据因时的相关协议说明，两次控制指令发送时间间隔最好至少5ms

        def dog_gripper(self):
            """勾爪控制"""
            pass

        def surface_gripper(self):
            """吸盘控制"""
            pass

        def flexible_hand(self, gestrue=None, id=1):
            """灵巧手控制
            gestrue：目标动作号，1-13固定，14-40自定
            id:控制的对象ID，1-254
            """
            address_and_lenth = {"GESTURE_NO_SET": (1008, 1), "ANGLE_SET": (1486, 12)}
            VID_PID = "6790_29987"
            HEAD = "EB 90 "
            ID = self.dec_to_hex_string(id, 1) + " "
            WRITE = "12 "
            if gestrue is not None:
                ADDRESS = (
                    self.dec_to_hex_string(address_and_lenth["GESTURE_NO_SET"][0]) + " "
                )
                LENTH = (
                    self.dec_to_hex_string(
                        address_and_lenth["GESTURE_NO_SET"][1] + 3, 1
                    )
                    + " "
                )
                gestrue_str = self.dec_to_hex_string(gestrue, 1)
                cmd_temp = ID + LENTH + WRITE + ADDRESS + gestrue_str
                SUM = self.hex_string_sum(cmd_temp)
                GESTURE_CMD = HEAD + cmd_temp + " " + SUM
                RoboticArmAgent.ToSerial(GESTURE_CMD, VID_PID, from_hex=True)

    def gripper_control(self, *args, **kwargs):
        """由初始化时的gripper参数决定，具体函数定义在__GripperControl类中"""
        pass

    def gripper_status(self):
        """由初始化时的gripper参数决定，具体函数定义在__GripperControl类中"""
        pass

    def is_alive(self) -> bool:
        """判断是否关闭了Agent"""
        if self.node_name is not None:
            if rospy.is_shutdown():
                return False
            else:
                self.ros_spin_once()
                return True
        else:
            return True

    def living(self, period=0.005) -> None:
        """阻塞防止程序退出
        使用ros时，使用rospy.spin()阻塞，否则使用time.sleep(period)阻塞。
        """
        if self.node_name is not None:
            rospy.spin()
        else:
            while True:
                time.sleep(period)

    def life_end_or_restart(self, mode=0, info="") -> None:
        """结束当前智能体的生命周期。0直接结束退出；1结束后重启。info是期望在结束后输出的信息"""
        if mode == 0:
            if info == "":
                info = "智能体生命周期结束"
            self.go_to_named_or_joint_target("Home", 1)
            print(info)
            sys.exit()  # 通过该函数退出当前进程
        else:
            for i, arg in enumerate(sys.argv):
                if arg == "--times":
                    self.restart_times = sys.argv[
                        i + 1
                    ]  # 如果找到了 "--times" 参数，尝试获取它的下一个参数作为值
                    break
            else:
                self.restart_times = 0  # 没有break则次数初始化为0
            if info == "":
                info = "智能体重启"
            self.restart_times += 1
            print(info, f"当前重启次数为：{self.restart_times}次")
            os.execl(
                sys.executable,
                sys.executable,
                *(sys.argv + ["--times", f"{self.restart_times}"]),
            )  # 通过sys.executable获取当前进程，然后重新执行

    @classmethod
    def ros_spin_once(cls, sleep_time=0.5) -> None:
        """该函数改版自rospy.spin，实现了python中ros的单次spin的spinonce函数"""
        if not hasattr(cls.ros_spin_once, "first"):
            cls.ros_spin_once.__dict__["first"] = False
            if not rospy.core.is_initialized():
                raise rospy.exceptions.ROSInitException(
                    "client code must call rospy.init_node() first"
                )
            rospy.core.logdebug(
                "node[%s, %s] entering spin(), pid[%s]",
                rospy.core.get_caller_id(),
                rospy.core.get_node_uri(),
                os.getpid(),
            )
        try:
            if not rospy.core.is_shutdown():
                rospy.rostime.wallsleep(sleep_time)
            else:
                exit("\r\nROS Core Shut Down")
        except KeyboardInterrupt:
            rospy.core.logdebug("keyboard interrupt, shutting down")
            rospy.core.signal_shutdown("keyboard interrupt")

    @classmethod
    def ToSerial(
        cls,
        data_cmd,
        port_vpid: str,
        baudrate=115200,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=None,
        find=False,
        from_hex=False,
    ):
        """
        通过串口发送有关数据；port_pid可以是端口号（Linux下比如是'/dev/ttyUSB0',Windows下比如是'COM8'），也可以是设备的vid和pid信息结合；
        如vid为6790，PID为29987，则写为'6790_29987'；
        find=true时查找完直接函数直接返回；
        """
        if not hasattr(cls.ToSerial, f"{port_vpid}"):
            from serial import Serial  # 要装pyserial
            import serial.tools.list_ports
            import atexit

            if find is True:
                port_list = list(serial.tools.list_ports.comports())
                device_list = [port.device for port in port_list]
                vid_list = [port.vid for port in port_list]
                pid_list = [port.pid for port in port_list]
                description_list = [port.description for port in port_list]
                print("共检测到这些设备：", device_list)
                print("设备的VID为：", vid_list)
                print("设备的PID为：", pid_list)
                print("设备的descriptio为：", description_list)
                print(type(pid_list[0]))
                return

            # 继承串口类
            class MySerial(Serial):
                """
                继承串口类，实现了可以在初始化时不用配置任何信息，通过新增加的comset函数进行配置，然后通过connect进行鲁棒的连接。
                初始化和comset可以都不指定端口号，而是在connect时再指定。
                """

                # 配置串口参数（可以不包括端口号）
                def ComSet(
                    self, baudrate, bytesize, parity, stopbits, timeout, port=None
                ):
                    d = {
                        "baudrate": baudrate,
                        "bytesize": bytesize,
                        "parity": parity,
                        "stopbits": stopbits,
                        "timeout": timeout,
                    }
                    self.apply_settings(d)
                    self.port = port

                # 选择端口号并尝试连接
                def Connect(self, rs_port=None):
                    if self.is_open:
                        print(f"已成功连接串口{self.port}")
                        return
                    if rs_port is None:
                        if self.port is not None:
                            rs_port = self.port
                        else:
                            raise Exception("缺少串口号，请检查串口配置")
                    # 没连接到串口则一直连接
                    while True:
                        port_list = list(serial.tools.list_ports.comports())
                        if len(port_list) == 0:
                            exit("没有可用串口")
                        else:
                            port_list = [port.device for port in port_list]
                            port_list = ";".join(
                                list(map(str, port_list))
                            )  # 将字符串列表转换为一整个字符串
                            if rs_port in port_list:  # 判断选用的端口是否在其中
                                self.port = rs_port
                                self.open()  # 开启串口
                                if self.is_open:
                                    print(f"成功连接串口{rs_port}")
                                    return
                                else:
                                    print(f"已找到目标端口号{rs_port}，但无法连接")
                            else:
                                print(
                                    "未找到目标端口号{}。找到的所有端口为：{}".format(
                                        rs_port, port_list
                                    )
                                )

            # 串口初始化与连接
            if "_" in port_vpid:
                vid, pid = port_vpid.split("_")
                port_list = list(serial.tools.list_ports.comports())
                device_list = [port.device for port in port_list]
                vid_list = [port.vid for port in port_list]
                pid_list = [port.pid for port in port_list]
                if (int(pid) in pid_list) and (int(vid) in vid_list):
                    port = device_list[pid_list.index(int(pid))]
                else:
                    exit("错误：pid和vid与现连接所有设备不匹配")
            ser = MySerial(port, baudrate, bytesize, parity, stopbits, timeout)
            # ser.Connect()
            cls.ToSerial.__dict__[f"{port_vpid}"] = ser
            atexit.register(ser.close)  # 程序退出时关闭串口
        # 发送数据
        if data_cmd is not None:
            if from_hex:
                data_cmd = bytes.fromhex(data_cmd)
            cls.ToSerial.__dict__[f"{port_vpid}"].write(data_cmd)

    @staticmethod
    def quaternion_about_axis(angel: float, axis: tuple):
        """aixs表示要绕着旋转的轴(x,y,z)，angel表示绕轴旋转的角度（deg）"""
        return transformations.quaternion_about_axis(angel, axis)

    @staticmethod
    def quaternion_to_euler(quaternion):
        """将四元数转换为欧拉角"""
        return transformations.euler_from_quaternion(quaternion)

    @staticmethod
    def euler_to_quaternion(euler: Union[list, tuple]):
        """将欧拉角(rpy,对应sxyz)转换为四元数NDArray[float64](方便四元数运算)"""
        return transformations.quaternion_from_euler(*euler)

    @staticmethod
    def to_pose_type(pose) -> Pose:
        """将各种类型的pose统一转换为Pose类型"""
        if isinstance(pose, (tuple, list)):
            target_pose = Pose()
            target_pose.position.x = pose[0]
            target_pose.position.y = pose[1]
            target_pose.position.z = pose[2]
            if len(pose) == 7:
                target_pose.orientation.x = pose[3]
                target_pose.orientation.y = pose[4]
                target_pose.orientation.z = pose[5]
                target_pose.orientation.w = pose[6]
            elif len(pose) == 6:
                orientation = transformations.quaternion_from_euler(*pose[3:])
                target_pose.orientation.x = orientation[0]
                target_pose.orientation.y = orientation[1]
                target_pose.orientation.z = orientation[2]
                target_pose.orientation.w = orientation[3]
            else:
                raise Exception("错误的位姿元素数量")
        elif isinstance(pose, Pose):
            target_pose = pose
        elif isinstance(pose, PoseStamped):
            target_pose = pose.pose
        else:
            raise Exception("错误的位姿类型")
        return target_pose

    def to_pose_stamped(self, pose) -> PoseStamped:
        """将各种类型的pose统一转换为PoseStamped类型"""
        if isinstance(pose, PoseStamped):
            pose_stamped = pose
        else:
            pose_stamped = PoseStamped()
            pose_stamped.pose = self.to_pose_type(pose)
            pose_stamped.header = self.target_pose.header
            pose_stamped.header.stamp = rospy.Time.now()
        return pose_stamped

    @staticmethod
    def to_tuple7_pose(pose) -> tuple:
        """将各种类型的pose统一转换为tuple7类型(x,y,z,x,y,z,w)"""
        if isinstance(pose, (list, tuple)):
            if len(pose) == 7:
                tuple_pose = tuple(pose)
            elif len(pose) == 6:
                orientation = transformations.quaternion_from_euler(*pose[3:])
                tuple_pose = (
                    pose[0],
                    pose[1],
                    pose[2],
                    orientation[0],
                    orientation[1],
                    orientation[2],
                    orientation[3],
                )
        elif isinstance(pose, Pose):
            tuple_pose = (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        elif isinstance(pose, PoseStamped):
            tuple_pose = (
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
        else:
            raise Exception("错误的位姿类型")
        return tuple_pose

    @staticmethod
    def to_list7_pose(pose) -> list:
        """将各种类型的pose统一转换为list7类型[x,y,z,x,y,z,w]"""
        if isinstance(pose, (list, tuple)):
            if len(pose) == 7:
                list_pose = list(pose)
            elif len(pose) == 6:
                orientation = transformations.quaternion_from_euler(*pose[3:])
                list_pose = [
                    pose[0],
                    pose[1],
                    pose[2],
                    orientation[0],
                    orientation[1],
                    orientation[2],
                    orientation[3],
                ]
        elif isinstance(pose, Pose):
            list_pose = [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        elif isinstance(pose, PoseStamped):
            list_pose = [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ]
        else:
            raise Exception("错误的位姿类型")
        return list_pose

    def loginfo(self, info="", sleep_f=0, sleep_b=0):
        """可带节点名的消息打印"""
        if sleep_f > 0:
            rospy.sleep(sleep_f)
        if self.node_name is None:
            print(info)
        else:
            rospy.loginfo("[{}] {}".format(self.node_name, info))
        if sleep_b > 0:
            rospy.sleep(sleep_b)

    def logwarn(self, warn=""):
        """可带节点名的警告打印"""
        if self.node_name is None:
            print("WARN:", warn)
        else:
            rospy.logwarn("[{}] {}".format(self.node_name, warn))

    def __ros_init(self, node_name):
        """初始化ROS节点"""
        if node_name == "":
            self.node_name = self.product_name + "_node"
        else:
            self.node_name = node_name
        if (self.node_name is not None) and not rospy.core.is_initialized():
            rospy.init_node(node_name)
            self.loginfo("Initializing......")
        elif self.node_name is None:
            print("节点名为None，不使用ROS")

    def set_control_modes(self, control_modes):
        """设置控制模式"""
        self.__control_modes = control_modes
        self.set_plan_mode(control_modes[0])  # 设置规划模式
        self.set_interpolate_mode(control_modes[1])  # 设置插值模式
        self.set_target_mode(control_modes[2])  # 设置控制对象模式
        if len(control_modes) == 4:
            self.immediate_excute(control_modes[3])  # 设置快速执行

    def get_control_modes(self):
        """获得当前控制模式"""
        return self.__control_modes

    def __control_init(self, control_mode, gripper, node_name):
        """控制初始化"""
        if control_mode != (-1, -1, -1) and node_name is None:
            control_mode = (-1, -1, -1)
            print("非ROS模式下仅支持Normal模式，已强制修改为control_mode=(-1,-1,-1)")
        self.__control_modes = control_mode
        self.start_base_mode = 1
        self.target_pose = PoseStamped()
        # 初始化basic控制接口
        if self.node_name is None:
            self.set_basic_controller("airbot_play")
        else:
            self.set_basic_controller("moveit")
        # 初始化内置夹爪
        self.gripper_control = self._GripperControl(gripper)()
        # 设置是否等待(默认为True)
        self.set_wait_mode(True)
        self.set_move_speed(1)  # 初始化无规划模式下的移动速度
        # # 设置控制模式
        # self.set_control_modes(control_mode)  # TODO:后续通过测试将下面四个替换成这个函数
        # 规划模式设置
        self.set_plan_mode(control_mode[0])
        # 是否不允许执行
        self.set_interpolate_mode(control_mode[1])
        # 设置控制对象模式
        self.set_target_mode(control_mode[2])
        # 设置快速执行
        if len(control_mode) == 4:
            self.immediate_excute(control_mode[3])
        # 初始化target变量
        self.__update_last_target(True)

    def __airbot_play_init(self, group_name, name_space):
        """初始化airbot_play接口"""
        self.basic_c = None  # TODO

    def __moveit_init(self, group_name, name_space):
        """初始化moveit接口"""
        while True:  # 防止同一个launch文件启动时，MoveIt!接口还没启动好，导致报错退出
            try:
                basic_c = MoveItBasicController(
                    group_name,
                    robot_description=name_space + "/robot_description",
                    ns=name_space,
                    wait_for_servers=10,
                )
            except Exception as e:
                (
                    self.loginfo(
                        f"未找到MoveIt!Group或未连接到Server，重新尝试中……{e}"
                    ),
                    rospy.sleep(0.4),
                )
            else:
                self.basic_c = basic_c
                self.reference_mode = 0  # 默认参考坐标系为base_link
                break

    def __eef_check(self):
        eef_link = self.basic_c.get_end_effector_link()
        if eef_link == "":  # 根据是否具备末端坐标系来判断是否是纯关节空间规划
            self.only_joint_space = True
        else:
            self.only_joint_space = False
            if eef_link != self.links_name[-1]:
                print(
                    f"WARM: 末端坐标系一般为坐标系列表最末尾元素。当前列表最末端元素为：{self.links_name[-1]}，当前末端坐标系设置为：{eef_link}"
                )
                time.sleep(1)

    def __preset_pose_init(self):
        """初始化预置的pose(默认关节角度；也可以存储Pose或Posestamped格式"""
        self.preset_pose = {
            "Home": (0, -0.025, 0.025, 0, 0, 0),
            "Standby": (0, -0.419, 0.314, 0.297, 0, 0),
            "Forward": (0, -1.57, 0.73, 0.85, 0, 0),
            "n": (0, -1.61754, 1.2831, -1.24363, 0, 0),
            "RPY0": (0, -0.1467, 0.1935, 1.524, 0, math.pi),
            "One": (0, -math.pi / 2, math.pi, 0, 0, 0),
            "Gravity": (0, -math.pi, math.pi, 0, 0, 0),
        }

    def __start_pose_init_(self, preset_pose, sleep_time=2):
        """到预置的位置(传入None会保持当前位置)"""
        if preset_pose == 0:
            self.go_to_named_or_joint_target("Home", sleep_time=sleep_time)
        elif preset_pose == 1:
            self.go_to_named_or_joint_target("Standby", sleep_time=sleep_time)
        elif preset_pose == 2:
            self.go_to_named_or_joint_target("Forward", sleep_time=sleep_time)
        elif preset_pose == 3:
            self.go_to_named_or_joint_target("n", sleep_time=sleep_time)
        elif preset_pose == 4:
            self.go_to_named_or_joint_target("One", sleep_time=sleep_time)
        elif preset_pose == 5:
            self.go_to_named_or_joint_target("Gravity", sleep_time=sleep_time)

    def __pub_arm_state(self, duration):
        """发布arm的状态信息"""
        if duration > 0 and self.node_name is not None:

            def state_publisher():
                while True:
                    self.update_current_state()
                    rospy.set_param(
                        self.basic_c.get_name_space() + "/current_arm_state",
                        f"xyz:{self.current_xyz},rpy:{self.current_rpy},xyzw:{self.current_xyzw}",
                    )
                    time.sleep(duration)

            Thread(target=state_publisher, daemon=True).start()

    def __set_start_state_to_last_joints_target(self) -> None:
        """设定起始状态为上次的目标"""
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.joints_name
        joint_state.position = self.last_joint_target
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        self.basic_c.set_start_state(moveit_robot_state)

    def __set_start_state_to_current_state(self) -> None:
        """设定起始状态为当前的状态"""
        self.basic_c.set_start_state_to_current_state()

    def __update_last_target(self, first=False) -> None:
        """更新目标位置，一般在初始化以及在成功规划执行后调用"""
        if first:  # 首次时将last初始化为current
            self.last_target_pose = self.update_current_state(get_pose=True)
            self.last_xyz = copy(self.current_xyz)
            self.last_rpy = copy(self.current_rpy)
            self.last_joint_target = copy(self.current_joints)
            self.last_xyzw = copy(self.current_xyzw)
            # 将初始化位置也作为一个当前目标
            self.target_pose = copy(self.last_target_pose)
            self.target_pose_position = copy(self.current_xyz)
            self.target_pose_euler = copy(self.current_rpy)
            self.target_pose_orientation = copy(self.current_xyzw)
            # print(f'初始化位姿为:\n  xyz{self.current_xyz}\n  rpy{self.current_rpy}\n  xyzw{self.current_xyzw}')
            # print(f"初始化关节角为{self.last_joint_target}")
        else:
            self.last_target_pose = deepcopy(self.target_pose)
            self.last_xyz = copy(self.target_pose_position)
            self.last_rpy = copy(self.target_pose_euler)
            self.last_xyzw = copy(self.target_pose_orientation)
            self.last_joint_target = copy(self.new_joint_target)

    def __set_pose_target_once(
        self,
        approximate=False,
        log=True,
        check=False,
        target: Optional[PoseStamped] = None,
    ) -> bool:
        """通过basic接口设定目标pose"""
        # 不论目标是一个或多个，有用的只有最后一个
        if target is None:
            target = self.target_pose
        try:
            self.basic_c.set_joint_value_target(
                target, self.basic_c.get_end_effector_link(), approximate
            )
        except Exception as e:  # 失败
            if log:
                self.loginfo(f"目标姿态逆解失败：姿态错误或超出关节限制范围")
            return False
        else:  # 成功
            if not check:
                self.new_joint_target = self.basic_c.get_joint_value_target()
            return True  # moveit仅会记录一个最新的target

    def __set_pose_target(
        self,
        approximate=False,
        retry_times: int = 20,
        return_enable=False,
        check=False,
        target=None,
    ) -> bool:
        cnt = 0
        half_times = retry_times / 2
        while cnt < retry_times:
            if cnt < half_times:
                if self.__set_pose_target_once(
                    approximate, log=False, check=check, target=target
                ):
                    return True  # 精确求解
                else:
                    cnt += 1
            elif self.__set_pose_target_once(
                True, check=check, log=False, target=target
            ):
                return True  # 不精确求解
            else:
                cnt += 1
        else:
            if return_enable:
                if not check:
                    self.loginfo(
                        f"{retry_times}次求逆解失败，请检查目标姿态{target}是否超出合理范围。"
                    )
                return False
            else:
                raise Exception(
                    f"{retry_times}次求逆解失败，请检查目标姿态{target}是否超出合理范围。"
                )

    def __set_start_state(self, start_base, wait_flag):
        # 轨迹起点设定
        if (
            start_base == 0 or self.wait_flag == False
        ):  # 若上次的等待标志不为true，则起始位置强制设置为当前位置，否则将会无法规划
            self.__set_start_state_to_current_state()
            self.wait_flag = wait_flag
            start_joints = self.current_joints
        else:
            self.__set_start_state_to_last_joints_target()
            start_joints = self.last_joint_target
        return start_joints

    def __plan_and_execute(
        self, disable_plan, start_joints, wait_flag, sleep_time, return_enable
    ):
        """最终的规划执行"""
        # 不进行规划，直接通过反解得到的关节空间目标执行（速度会较快，但大幅度姿态变换时容易发生机械臂自身的干涉碰撞问题）
        if disable_plan or self.__plan_mode >= 0:
            self.__no_plan_excute([list(start_joints), list(self.new_joint_target)])
            self.__update_last_target()  # 执行成功才更新last
            self.basic_c.clear_pose_targets()  # It is always good to clear your targets after planning with poses
            if sleep_time > 0:
                rospy.sleep(sleep_time)
            return True
        # 工作空间规划
        else:
            plan_success, traj, self.__last_plan_time, error_code = (
                self.basic_c.plan()
            )  # error_code只在成功时有用好像，所以是没啥用的
            self.basic_c.clear_pose_targets()  # It is always good to clear your targets after planning with poses
            # 规划成功执行
            if plan_success:
                self.basic_c.stop()  # 避免多余动作Stop the current execution, if any
                self.basic_c.execute(
                    traj, wait=wait_flag
                )  # 因为上面调用了stop，故理论上无需等待，否则stop将不起作用必然将等待完成；但是不等待机械臂会明显表现出卡顿，因为plan频率过低。
                self.__update_last_target()  # 执行成功才更新last
                if sleep_time > 0:
                    rospy.sleep(sleep_time)
                return True  # plan成功退出函数
            # 规划失败处理
            else:
                if return_enable:
                    self.loginfo(f"{self.node_name}规划失败")
                    return False  # 如果没有启用loop则直接退出，否则继续循环直到plan成功
                else:
                    raise Exception(f"{self.node_name}规划失败")

    def __wait_excute(self, wait_flag: Union[bool, float]):
        """暂不支持
        wait_flag=True时根据当前关节状态和目标状态判断是否执行完。
        wait_flag为大于0的浮点数时，wait_flag可设置为以秒为单位的延时值。
        wait_flag=False或0时，不进行延时。
        """
        if wait_flag == True:
            self.loginfo("动作执行中......")
            while True:
                self.update_current_state()
                if (
                    np.max(
                        np.abs(
                            np.array(self.current_joints)
                            - np.array(self.new_joint_target)
                        )
                    )
                    <= self.__wait_tolrerance
                ):
                    break
                else:
                    rospy.sleep(0.01)
            self.loginfo("执行完毕")
        elif wait_flag:
            rospy.sleep(wait_flag)

    def __no_plan_excute(self, targets: List[list]):
        """直接根据逆解执行"""
        if self.__plan_mode == 1:
            self.__plan_mode = -1  # 单次不规划
        elif self.__plan_mode < 0:
            cp = self.__plan_mode
            self.set_plan_mode(1)
            self.__plan_mode = cp  # 恢复
        len__ = len(targets)
        target_cp = deepcopy(targets)
        target_cp[0].append(0)
        for i in range(len__ - 1):
            time_ = float(
                np.abs(np.array(targets[i]) - np.array(targets[i + 1])).max()
                / self.const_move_speed
            )  # 给定速度，得到两个关节目标的时间间隔 #TODO：这种方式不太正确，不同关节应该有相同的时间点，而速度不同，不能统一速度。
            target_cp[i + 1].append(time_)
        rospy.set_param(self.basic_c.get_name_space() + "/joint_target", target_cp)  #

    def _gravity_test(self):
        """测试重力影响"""
        self.go_to_named_or_joint_target("Gravity", sleep_time=5)
        self.log_arm_info(current_joint=True, target_joint=True)


class AirbotPlayConfig(object):
    """方便进行参数变量配置的类"""

    # 名称
    product_name = "airbot_play"
    product_name_upper_first = "AirbotPlay"
    # 夹爪配置
    gripper_two_finger_sim = (0, 1)
    gripper_two_finger_real = (0, 0)
    gripper_suck_sim = (2, 1)
    gripper_dog_sim = (1, 1)
    gripper_dog_real = (1, 0)
    gripper_hand = (5, 0)
    # 控制配置
    normal = (-1, -1, -1)  # 轨迹关节角度起点目标值为上次目标值，有插值执行，非fake
    current_normal = (-2, -1, -1)  # 轨迹关节角度起点目标值为当前值，有插值执行，非fake
    once_no_plan = (1, -1, -1)  # 临时无规划,有插值执行
    always_no_plan = (0, -1, -1)  # 全局无规划，有插值执行
    # fake控制
    fake = (0, 0, 0, 0)  # 全局无规划无插值快速fake执行
    fake_normal = (-1, -1, 0)  # 正常的fake执行
    fake_with_interpolate = (0, -1, 0)
    fake_with_plan = (-1, 0, 0)
    fake_with_plan_immediate = (-1, 0, 0, 0)


try:
    from std_msgs.msg import Bool, Float64
except:
    pass


class GripperSimpleControl(object):
    """简单的夹爪控制类"""

    def __init__(self) -> None:
        if rospy.get_param("/airbot_play/sim_player_nums", default=0):
            print("使用仿真")
            self.__sim_type = "sim"
            self.gripper_pub = rospy.Publisher("/gripper_state", Bool, queue_size=10)
        else:
            print("使用实机")
            self.__sim_type = "real"
            self.__real_gripper = RoboticArmAgent._GripperControl((0, 0))()

    def control(self, cmd):
        if self.__sim_type == "sim":
            if cmd in [1, True]:
                rospy.set_param("/pick_place_flag", "pick")
                self.__pub_gripper_cmd(True)
                self.__grasped = True
            elif cmd in [0, False]:
                rospy.set_param("/pick_place_flag", "place")
                self.__pub_gripper_cmd(False)
                self.__grasped = False
        elif self.__sim_type == "real":
            if cmd in [1, True]:
                self.__real_gripper(1)
                self.__grasped = True
            elif cmd in [0, False]:
                self.__real_gripper(0)
                self.__grasped = False

    def get_status(self) -> bool:
        return self.__grasped

    def __pub_gripper_cmd(self, cmd: bool, nums=20, freq=0.005):
        """发布夹爪控制命令"""
        for _ in range(nums):
            b = Bool()
            b.data = cmd
            self.gripper_pub.publish(cmd)
            rospy.sleep(freq)


class ChooseGripper(object):
    def __init__(self, sim_type, gripper_type) -> None:
        # 根据不同仿真类型定义不同的夹爪控制函数
        if sim_type in ["gazebo", "gazebo_i"]:
            from moveit_commander import MoveGroupCommander

            if sim_type == "gazebo_i":
                namespace = "/airbot_play"
            else:
                namespace = ""
            robot_description = f"{namespace}/robot_description"
            airbot_play_gripper = MoveGroupCommander(
                "airbot_play_gripper", robot_description, namespace
            )
            airbot_play_gripper.set_max_acceleration_scaling_factor(0.1)
            airbot_play_gripper.set_max_acceleration_scaling_factor(0.1)
            if gripper_type == "inspire":
                SCALE_CLOSE = 0.7
                delta = 0.04
                CLOSE = (-SCALE_CLOSE, SCALE_CLOSE, SCALE_CLOSE, -SCALE_CLOSE + delta)
            elif gripper_type == "gripper":
                # SCALE_CLOSE = 0.034
                SCALE_CLOSE = 0.026
                CLOSE = (SCALE_CLOSE, -SCALE_CLOSE)
            else:
                raise NotImplementedError("未实现的夹爪类型")

            def gripper_control(pp, sleep_time=0):
                if pp:  # pp=1, pick
                    airbot_play_gripper.go(CLOSE)
                    another_wait = 1.5
                else:
                    airbot_play_gripper.set_named_target("Home")
                    airbot_play_gripper.go(wait=True)
                    another_wait = 0
                rospy.sleep(sleep_time + another_wait)
        elif sim_type == "isaac":

            def gripper_control(pp, sleep_time=0):
                if pp:
                    rospy.set_param("/pick_place_flag", "pick")
                else:
                    rospy.set_param("/pick_place_flag", "place")
                rospy.sleep(sleep_time)
        elif gripper_type == "gripper_bool":
            gripper_pub = rospy.Publisher(
                "/airbot_play/gripper/state_cmd", Bool, queue_size=5
            )
            cmd = Bool(False)

            def cmd_pub_thread():
                rate = rospy.Rate(200)
                while not rospy.is_shutdown():
                    gripper_pub.publish(cmd)
                    rate.sleep()

            Thread(target=cmd_pub_thread, daemon=True).start()

            def gripper_control(pp, sleep_time=0):
                if pp:
                    cmd.data = True
                else:
                    cmd.data = False
                rospy.sleep(sleep_time)
        elif gripper_type == "gripper":
            gripper_pub = rospy.Publisher(
                "/airbot_play/gripper/set_position", Float64, queue_size=5
            )

            # cmd = Float64(0)
            # def cmd_pub_thread():
            #     rate = rospy.Rate(200)
            #     while not rospy.is_shutdown():
            #         gripper_pub.publish(cmd)
            #         rate.sleep()
            # Thread(target=cmd_pub_thread, daemon=True).start()
            def gripper_control(pp, sleep_time=0):
                pp = 1 - pp
                duration = 1.5
                current_pos = rospy.wait_for_message(
                    "/airbot_play/gripper/position", Float64
                )
                pos_error = pp - current_pos.data
                duration *= abs(pos_error)
                nums = int(duration * 50)
                if nums == 0:
                    nums = 1
                delta = pos_error / nums
                for _ in range(nums):
                    current_pos.data += delta
                    # print(current_pos.data)
                    gripper_pub.publish(current_pos)
                    rospy.sleep(0.02)
                # vel = 1.5
                # frequency = 200
                # # 根据当前位置、目标位置和速度、控制频率进行插值
                # pos_error = pp - current_pos.data
                # duration = abs(pos_error) / vel
                # nums = int(duration * frequency)
                # pos_delta = pos_error / nums
                # for _ in range(nums):
                #     current_pos.data += pos_delta
                #     gripper_pub.publish(current_pos)
                #     rospy.sleep(1 / frequency)
                # # 最后再持续发送一段时间结尾值
                # for _ in range(50):
                #     gripper_pub.publish(Float64(pp))
                #     rospy.sleep(0.005)
                for _ in range(50):
                    current_pos.data = pp
                    gripper_pub.publish(current_pos)
                    rospy.sleep(0.005)
                rospy.sleep(sleep_time)
        elif sim_type in ["real", None]:
            # 这里真机暂时还是使用内置的因时夹爪的控制函数
            gripper_control = False
        self._gripper = gripper_control

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._gripper
