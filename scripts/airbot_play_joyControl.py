#!/usr/bin/env python3
import rospy
from airbot_play_control.control import RoboticArmAgent,AirbotPlayConfig
import numpy as np
from sensor_msgs.msg import Joy
from IPython import embed

class joyControl:
    def __init__(self):
        self.acc_max = 1
        self.vel_max = 1
        self.NODE_NAME = 'airbot_play_joy_move'
        rospy.init_node(self.NODE_NAME)
        self.joytopic='Joy'
        self.arm = RoboticArmAgent(control_mode=AirbotPlayConfig.normal,init_pose=None,node_name=self.NODE_NAME,gripper=AirbotPlayConfig.gripper_two_finger_real)

    def subjoy(self):
        rospy.Subscriber(self.joytopic, Joy, self.callback)
        rospy.spin()

    def callback(self, msg):
        if msg.buttons[10]:# back
            self.arm.immediate_excute(-1),self.arm.set_plan_mode(-1)  # 恢复正常控制
            self.arm.go_to_named_or_joint_target('Home')
        if msg.buttons[0]:# back
            self.arm.immediate_excute(-1),self.arm.set_plan_mode(-1)
            position = [0.5073,-0.0012,0.0896]
            rot = [-0.722,-0.007,-0.692,0.005]
            self.arm.set_and_go_to_pose_target(position, rot, '0',sleep_time=2)  
        elif msg.buttons[1]:
            self.arm.gripper_control(0,sleep_time=1.5)
        elif msg.buttons[3]:
            self.arm.gripper_control(1,force=1000,always=True,sleep_time=1.5)
        elif (msg.axes[2] or msg.axes[3] or msg.axes[4] or msg.axes[5]):### axes[2]和axes[3]的值域都是[-1,1],axes[0]和axes[1]是控制底盘的，剩下的按键定义我画在纸上了
            # print("计算一个摇杆和移动快慢增量的比例,连续控制机械臂,就是怎么套用key_control()那部分")
            acc_limit = 1.0
            vel_limit = 1.0
            continuous = True
            kp = 0.001
            if continuous:  # 连续控制
                self.arm.immediate_excute(0),self.arm.set_plan_mode(0)
            if msg.axes[2]:
                self.arm.go_to_shift_single_axis_target(0,msg.axes[2]*kp,acc_limit=acc_limit,vel_limit=vel_limit,wait_flag=False,return_enable=True,no_plan=continuous)
            if msg.axes[3]:
                self.arm.go_to_shift_single_axis_target(1,msg.axes[3]*kp,acc_limit=acc_limit,vel_limit=vel_limit,wait_flag=False,return_enable=True,no_plan=continuous)
            if msg.axes[4]:
                self.arm.go_to_shift_single_axis_target(2,msg.axes[4]*kp,acc_limit=acc_limit,vel_limit=vel_limit,wait_flag=False,return_enable=True,no_plan=continuous)
            if msg.axes[5]:
                self.arm.go_to_shift_single_axis_target(3,msg.axes[5]*kp,acc_limit=acc_limit,vel_limit=vel_limit,wait_flag=False,return_enable=True,no_plan=continuous)            

if __name__ == '__main__':
    joyContr=joyControl()
    joyContr.subjoy()
