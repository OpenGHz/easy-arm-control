#!/usr/bin/env python3

import math
import rospy
from actionlib import SimpleActionServer

from scipy.interpolate   import CubicSpline,make_interp_spline,interp1d
from control_msgs.msg    import FollowJointTrajectoryGoal
from control_msgs.msg    import FollowJointTrajectoryResult
from control_msgs.msg    import FollowJointTrajectoryFeedback
from control_msgs.msg    import FollowJointTrajectoryAction
from trajectory_msgs.msg import JointTrajectoryPoint
from typing import List,Union

import matplotlib.pyplot as plt
import numpy as np
import functools
from sensor_msgs.msg import JointState

class MoveItAction(object):
	# TODO：这两个类目前还没有怎么用起来
    _feedback = FollowJointTrajectoryFeedback()
    _result = FollowJointTrajectoryResult()
    CONTROL_MODE = ['torque','position']  # 控制模式选择
	# Action initialisation
    def __init__(self,name,control_mode,interpolation=5,plot=False):
        """
            name：action服务器名。
            interpolation支持：1为线性插值；2-5为n次样条插值。
            control_mode：力控或位控（CONTROL_MODE = ['torque','position']  # 控制模式选择）
        """
        self.product_name = "airbot_play"
        self.product_name_space = "/" + self.product_name  # self.product_name_space + "

        self.drag = rospy.get_param("~drag", default=False)
        self.target_mode = -1

        self.interpolation = interpolation
        self.t_delta = 0.005  # 单位:s
        self.plot = plot
        self.arm_joint_nums = 6
        self.gripper_joint_nums = 0
        self.all_joint_nums = self.arm_joint_nums + self.gripper_joint_nums
        self.joints_name = [f"joint{i}" for i in range(1,self.all_joint_nums)]
        self.init_pose = [0,-0.025,0.025,0,0,0]  # 初始状态（单位为rad，防止零位干涉）

        self.new_action = False
        self.times = 0; self.max_times = 100

        self.cmd_publisher = rospy.Publisher(self.product_name_space + '/joint_cmd', JointState, queue_size=10)
        self.cmd = JointState()
        self.cmd.header.stamp = rospy.Time.now()
        self.cmd.header.frame_id = self.product_name
        self.cmd.name = self.joints_name
        # self.cmd.mode = control_mode
        # self.cmd.kp = [100 for _ in range(self.arm_joint_nums)]
        # self.cmd.kd = [10 for _ in range(self.arm_joint_nums)]
        self.cmd.position = self.init_pose
        self.cmd.velocity = [0.8 for _ in range(self.arm_joint_nums)]  # 初始化速度值23度/s，从而控制电机缓慢移动到0位
        self.cmd.effort = [0 for _ in range(self.arm_joint_nums)]
        # self.cmd.id = [id for id in range(1,self.arm_joint_nums+1)]

        rospy.Timer(rospy.Duration(1. / 200.),self.control_continue)
        rospy.Timer(rospy.Duration(1. / 50.), self.send_joint_cmd)

        self._action_server = SimpleActionServer(name,FollowJointTrajectoryAction, execute_cb=self.execute_cb, auto_start = False)
        self._action_server.start()

    def execute_cb(self,goal_handle:Union[FollowJointTrajectoryGoal,list]):
        """ Action Callback """
        # 避免函数被重复运行
        if not hasattr(self.execute_cb,'running'):
            self.execute_cb.__dict__['running'] = False
        if self.execute_cb.__dict__['running']: return
        else: self.execute_cb.__dict__['running'] = True

        # 不同类型处理（获得统一的points变量）
        if isinstance(goal_handle,list):  # 包含(self.arm_joint_nums+1)元素
            target_num = len(goal_handle)
            points = [JointTrajectoryPoint() for _ in range(target_num)]
            for i in range(target_num):
                points[i].time_from_start = rospy.Duration(goal_handle[i][self.arm_joint_nums])
                points[i].positions = goal_handle[i][:self.arm_joint_nums]
        else: points:List[JointTrajectoryPoint] = goal_handle.trajectory.points  # 获得目标途径上的所有轨迹点（无任何外在约束的情况下一般只有两个点）

        # 构造初始矩阵
        times = len(points)
        time_array = np.zeros(times)  # 关节位置的时间信息列表
        joints_matrix = np.zeros((times,self.arm_joint_nums))  # 存储关节位置插值数据用的列表
        joints_vel_matrix = joints_matrix.copy()
        joints_acc_matrix = joints_matrix.copy()
        # 分别获得各个点的时间信息和位置信息，放到两个列表中
        for time,point in enumerate(points):  # 单位为rad；每个point对应了某个时间点
            time_array[time] = point.time_from_start.to_sec()  # 不同关节具有相同的时间点
            joints_matrix[time] = point.positions  # point.positions[j]存储了关节j在当前时间的角度值；joints_matrix每一行存储了某个时间点所有关节的角度目标
            joints_vel_matrix[time] = point.velocities
            joints_acc_matrix[time] = point.accelerations

        # 是否插值和插值类型获取
        self.interpolation = rospy.get_param(self.product_name_space + '/interpolate_type',default=self.interpolation)
        interpolate_mode = rospy.get_param(self.product_name_space + "/no_interpolate", default=-1)
        immediate_excute = rospy.get_param(self.product_name_space + "/immediate_excute", default=-1)
        if immediate_excute == -1:  # 进行时间插值
            # 轨迹插值
            execute_time_array = Interpolate.time_clip(0,time_array[times-1],self.t_delta)  # 将轨迹首尾点的时间以5ms来分段，与rospy.Rate(200)对应上
            new_times = len(execute_time_array)
            if interpolate_mode != -1:
                if interpolate_mode == 1: rospy.set_param(self.product_name_space + "/no_interpolate",-1)  # 仅一次，然后恢复正常插值
                interpolate = functools.partial(Interpolate.no_interpolate,t=time_array,t_i=execute_time_array,plot=self.plot)
            elif isinstance(self.interpolation,int):
                if self.interpolation == 1:  # 线性插值
                    interpolate = functools.partial(Interpolate.linear_interpolate,t=time_array,t_i=execute_time_array,plot=self.plot)
                elif self.interpolation == 3:  # 三次B样条插值
                    interpolate = functools.partial(Interpolate.cubic_spline,t=time_array,t_i=execute_time_array,plot=self.plot)
                elif 1 < self.interpolation <= 5:  # 五次B样条插值
                    interpolate = functools.partial(Interpolate.spline,t=time_array,t_i=execute_time_array,k=self.interpolation,plot=self.plot)
                else: exit('暂不支持该类型的插值方式')
            else: exit('暂不支持该类型的插值方式')
            joints_matrix_new = np.apply_along_axis(interpolate,axis=0,arr=joints_matrix)  # axis=0表示对各个列施加函数
            speed_matrix = np.apply_along_axis(Interpolate.linear_speed_calculate,axis=0,arr=joints_matrix_new,t=self.t_delta)  # 速度计算
        else:  # 直接发送目标(不发送起始目标)
            if immediate_excute == 1: rospy.set_param(self.product_name_space + "/immediate_excute",-1)  # 仅一次，然后恢复正常插值
            joints_matrix_new = joints_matrix
            new_times = times
            speed_matrix = np.apply_along_axis(Interpolate.linear_speed_calculate,axis=0,arr=joints_matrix_new,t=time_array)
        joints_list = joints_matrix_new.tolist()

        # 速度限幅
        vel_limit = list(rospy.get_param(self.product_name_space + "/vel_limit", default=(0,0)))
        if vel_limit[0] == 0: vel_limit[0] = 2.593
        if vel_limit[1] == 0: vel_limit[1] = 2*math.pi
        # rospy.set_param(self.product_name_space + "/min_vel",(0,0))  # （仅修改一次，然后自动调整为默认值）
        speed_min_limit_matrix = np.abs(speed_matrix) + vel_limit[0]  # 对速度求绝对值，因为底层仅接收正速度；限制最小速度
        speed_list:list = np.where(speed_min_limit_matrix>vel_limit[1],vel_limit[1],speed_min_limit_matrix).tolist()  # 限制最大速度
        speed_list.insert(0,list(speed_list[0]))  # 发送第一个点的期望速度设置为与第二个点相同

        # 删减第一个目标值，以实现快速执行
        if immediate_excute != -1:
            speed_list = np.delete(speed_list,0,axis=0)
            joints_list = np.delete(joints_list,0,axis=0)
            new_times -= 1

        plan_mode = rospy.get_param(self.product_name_space + "/plan_mode",default=-1)

        test_mode = rospy.get_param(self.product_name_space + '/test_mode',default=0)
        if test_mode != 0: plt.ion()
        for i in range(6):
            j_num = i
            plt.figure(i+1)
            plt.clf()
            if test_mode == 1:
                # 绘制规划路点
                plt.plot(time_array,joints_matrix[:,j_num],'ro',label='original_cmd',markersize=6)
                plt.plot(time_array,joints_vel_matrix[:,j_num],'g-',label='original_vel_cmd',markersize=3)
                plt.plot(time_array,joints_vel_matrix[:,j_num],'go',label='original_vel_cmd',markersize=6)
                plt.plot(time_array,joints_acc_matrix[:,j_num],'b-',label='original_acc_cmd',markersize=3)
                plt.plot(time_array,joints_acc_matrix[:,j_num],'bo',label='original_acc_cmd',markersize=6)
                # 绘制样条插值的位置指令结果
                plt.plot(execute_time_array,joints_matrix_new[:,j_num],'y-',label='interpolated_cmd',markersize=1)
                # # 绘制关节反馈值
                # plt.plot(arr,self.jt_m[:,j_num],'bo',label='joint_states',markersize=1)
                # 标题
                plt.title('original and interpolated pos')
            if test_mode != 0:
                # 添加图例和标签
                plt.legend(loc='best')
                plt.xlabel('t')
                plt.ylabel('y')
                # manager = plt.get_current_fig_manager()
                # manager.full_screen_toggle()
                # 显示图形
                plt.show(block=False)  # 非block并不会清除原来显示的图片，而是等下次图片来覆盖（前提是使用了plt.clf()）
                plt.pause(0.001)

        # 遍历所有插值时间发送cmd
        self.new_action = True
        while self.continue_ok: pass
        r = rospy.Rate(200)  # 200Hz，5ms控制频率
        self.target_mode = rospy.get_param(self.product_name_space + '/target_mode',default=-1)  # 控制的对象确定（是否fake）
        for i in range(new_times):  # 遍历每一个插值点
            # 若有新的action中断发生，则立刻终止当前处理，转而下一个处理
            if plan_mode == -1 and self._action_server.is_preempt_requested():
                self._action_server.set_preempted()
                self.execute_cb.__dict__['running'] = False
                self.new_action = False
                return
            # 控制命令发布（只发6个关节的，即cmd长度为6）
            self.cmd.velocity = speed_list[i]
            self.cmd.position = joints_list[i]
            self.cmd.header.stamp = rospy.Time.now()
            if self.target_mode == -1:
                self.cmd_publisher.publish(self.cmd)
            else: self.fake_publish()  # 发布fake命令
            r.sleep()
        self.new_action = False
        self.times = 0
	    # 设置结果通知MoveIt执行完毕
        if plan_mode == -1:
            self._result.error_code = 0
            self._action_server.set_succeeded(self._result)
        elif plan_mode == 1:
            rospy.set_param(self.product_name_space + "/plan_mode",-1)
        self.execute_cb.__dict__['running'] = False

    def control_continue(self,event):
        """ 接收到控制消息后，发送完仍继续发送一段时间最后的目标值，防止丢包造成位置误差增大 """
        self.continue_ok = False
        if not self.new_action:
            self.cmd.header.stamp = rospy.Time.now()
            if self.drag: self.cmd.effort = [1 for _ in range(self.arm_joint_nums)]
            if self.target_mode == -1:
                self.cmd_publisher.publish(self.cmd)
                self.continue_ok = True
            else: self.fake_publish()
            if self.times >= self.max_times:
                self.times = 0
                if self.target_mode == -1:  # 关节状态是需要一直发送的，因此self.new_action不要改成False
                    self.new_action = True
                self.max_times = 20  # 初始化为100，因为初始化的时候由于启动不同步问题容易丢包；之后都是20；
            else: self.times += 1
        else:  # 不发送阶段检测param
            self.target_mode = rospy.get_param(self.product_name_space + '/target_mode',default=-1)
            if self.target_mode == 0: self.new_action = False  # 表明需要发送关节状态（并且关节状态是需要一直发送的）

    def send_joint_cmd(self,event):
        """ 定时检查是否采用不经过Moveit规划进行控制（但任然用到了逆解得到的关节角度值，通过paramself.product_name_space + '/joint_target'传递） """
        joint_target = rospy.get_param(self.product_name_space + '/joint_target',0)
        if joint_target != 0:
            rospy.set_param(self.product_name_space + '/joint_target',0)
            self.execute_cb(joint_target)

    def test(self,joints_list,time_array,speed_list,times,enable=False):
        """ 打印一些参数 """
        if enable:
            loginfo(f'起止点为：{joints_list[0]} {joints_list[1]}')
            loginfo(f'时间间隔：{time_array}')
            loginfo(f'速度值为：{speed_list}')
            loginfo(f'点数量为：{times}')

    def fake_publish(self,joints_name=None):
        """ 将控制命令发布转换为关节状态发布 """
        if not hasattr(self.fake_publish,'first'):
            self.fake_publish.__dict__['first'] = False
            if joints_name is None: joints_name = self.joints_name
            from sensor_msgs.msg import JointState
            self.fake_pub = rospy.Publisher(self.product_name_space + "/joint_states", JointState, queue_size=2)
            self.state_eq_target = JointState()
            self.state_eq_target.name = joints_name
            self.state_eq_target.header.frame_id = self.product_name
        self.state_eq_target.position = self.cmd.position
        # self.state_eq_target.velocity = self.cmd.velocity
        # self.state_eq_target.effort   = self.cmd.effort
        self.state_eq_target.header.stamp = self.cmd.header.stamp
        self.fake_pub.publish(self.state_eq_target)

class Interpolate(object):
    """ 轨迹插值 """
    @staticmethod
    def time_clip(start,end,interval,unit='s',end_control=False):
        """ms级的时间细化
            unit表示所给三个时间的单位，s或ms两种。
            end_control决定是否根据插分后的最后一个值与差分前的最后值相等进行末尾时间控制
        """
        if unit == 's': precision = 0.001
        elif unit == 'ms': precision = 1
        time_line = (np.array([start,end,interval])/precision).astype('int32')
        time_clipped = np.arange(time_line[0],time_line[1],step=time_line[2])
        if end_control:
            if time_clipped[-1] != time_line[1]:
                time_clipped = np.append(time_clipped,time_line[1])
        else: time_clipped = np.append(time_clipped,time_line[1])
        time_clipped = time_clipped.astype('float64')
        time_clipped*=precision
        return time_clipped

    @classmethod
    def no_interpolate(cls,y,t,t_i,unit='s',sort=False,plot=False):
        """ 无插值 """
        # 对时间序列进行排序，以保证时间单调递增
        if sort:
            idx = np.argsort(t)
            t,y = t[idx],y[idx]
        # 对插值后的时间序列进行扩展
        if isinstance(t_i,(float,int)):
            flag = True
            t_interp = cls.time_clip(t[0],t[-1],t_i,unit)
        else:
            flag = False
            t_interp = t_i
        # 无插值
        stage_num = len(t)
        def stage(t_):
            for i in range(stage_num-1):
                if t[i] <= t_ <= t[i+1]:
                    return y[i+1]
        y_interp = []
        for t_ in t_interp:
            y_interp.append(stage(t_))
        # 是否绘制曲线
        if plot: cls.plot(t,y,t_interp,y_interp)
        # 返回插值后的时间序列和函数值
        if flag: return t_interp, y_interp
        else: return y_interp

    @classmethod
    def linear_interpolate(cls,y,t,t_i,unit='s',sort=False,plot=False):
        # 对时间序列进行排序，以保证时间单调递增
        if sort:
            idx = np.argsort(t)
            t,y = t[idx],y[idx]
        # 对插值后的时间序列进行扩展
        if isinstance(t_i,(float,int)):
            flag = True
            t_interp = cls.time_clip(t[0],t[-1],t_i,unit)
        else:
            flag = False
            t_interp = t_i
        # 使用LinearNDInterpolator进行线性插值
        y_interp = interp1d(t,y)(t_interp)
        if plot: cls.plot(t,y,t_interp,y_interp)
        # 返回插值后的时间序列和函数值
        if flag: return t_interp, y_interp
        else: return y_interp

    @classmethod
    def cubic_spline(cls,y,t,t_i,unit='s',bc_type='clamped',sort=False,plot=False):
        """
        使用 UnivariateSpline 实现 1-5次样条插值，常用3次和5次
        参数:
            t: 时间序列，一个一维数组
            y: 时间序列对应的函数值，一个一维数组
            ti:
            unit: 时间单位是s还是ms
            bc_type: 'not-a-knot','natural'，'clamped'
        返回值:
            返回一个二元组，包含插值后的时间序列和对应的函数值
        """
        # 对时间序列进行排序，以保证时间单调递增
        if sort:
            idx = np.argsort(t)
            t,y = t[idx],y[idx]
        # 对插值后的时间序列进行扩展
        if isinstance(t_i,(float,int)):
            flag = True
            t_interp = cls.time_clip(t[0],t[-1],t_i,unit)
        else:
            flag = False
            t_interp = t_i
        # 使用CubicSpline进行3次样条插值
        cnt = 1
        while True:
            try: y_interp = CubicSpline(t,y,bc_type=bc_type)(t_interp)
            except: t[-cnt] += 0.0003/cnt  #  Expect x to not have duplicates or x must be strictly increasing sequence.
            else: break
            cnt += 1
        if plot: cls.plot(t,y,t_interp,y_interp)
        # 返回插值后的时间序列和函数值
        if flag: return t_interp, y_interp
        else: return y_interp

    @classmethod
    def spline(cls,y,t:np.ndarray,t_i,k=5,unit='s',sort=False,plot=False):
        """
        使用 make_interp_spline 实现 1-5次样条插值，常用3次和5次
        参数:
            t: 时间序列，一个一维数组
            y: 时间序列对应的函数值，一个一维数组
        返回值:
            当t_i为间隔时，返回一个二元组，包含插值后的时间序列和对应的函数值
            当t_i为时间序列时，仅返回插值后的函数值
        """
        # 对时间序列进行排序，以保证时间单调递增
        if sort:
            idx = np.argsort(t)
            t,y = t[idx],y[idx]
        # 对插值后的时间序列进行扩展
        if isinstance(t_i,(float,int)):  # 指定频率
            flag = True
            t_interp = cls.time_clip(t[0],t[-1],t_i,unit)
        else:  # 指定已经细化的时间点
            flag = False
            t_interp = t_i
        # 使用make_interp_spline进行样条插值
        cnt = 1
        while True:
            try: y_interp = make_interp_spline(t,y,k=k,bc_type=(((1, 0), (2, 0)),((1, 0), (2, 0))))(t_interp)
            except: t[-cnt] += 0.0003/cnt  #  Expect x to not have duplicates or x must be strictly increasing sequence.
            else: break
            cnt += 1
        if plot: cls.plot(t,y,t_interp,y_interp)
        # 返回插值后的时间序列和函数值
        if flag: return t_interp, y_interp
        else: return y_interp

    @staticmethod
    def linear_speed_calculate(y:np.ndarray,t):
        """ 计算曲线两点间的直线变化率 """
        y_cp = y.copy()
        y1 = np.delete(y,0)
        y2 = np.delete(y_cp,-1)
        if isinstance(t,(np.ndarray,list,tuple)):
            return (y1-y2)/(t[1:]-t[:-1])
        else: return (y1-y2)/t

    @staticmethod
    def interval_limit(y_i:np.ndarray,min_delta=0.001,max_delta=3):
        """ 控制插值后y方向的间隔 """
        end_index = y_i.shape[0] - 1
        i = 1
        # 中间元素限幅
        while i < end_index:
            if math.fabs(y_i[i] - y_i[i-1]) < min_delta:
                y_i = np.delete(y_i,i)
                end_index -= 1  # 删除中间元素后有效长度-1（即有效长度始终记录的是0-原末尾元素的长度）
                y_i = np.append(y_i,y_i[end_index])
            else: i += 1  # 未删除元素计数+1
        # 末尾（原）两元素限幅
        if math.fabs(y_i[end_index] - y_i[end_index-1]) < min_delta:
            y_i[end_index-1] = y_i[end_index]
        return y_i

    @classmethod
    def way_points_interpolate(cls,way_points:np.ndarray,time_points:np.ndarray,freq:float,k=5) -> list:
        """ 路点插值 """
        execute_time_array = cls.time_clip(0,time_points[-1],1/freq)
        interpolate = functools.partial(Interpolate.spline,t=time_points,t_i=execute_time_array,k=k)
        joints_matrix_new:np.ndarray = np.apply_along_axis(interpolate,axis=0,arr=way_points)
        return joints_matrix_new.tolist()

    @staticmethod
    def plot(t,y,t_interp,y_interp,pause=0,clear=True,ion=False,block=False):
        """
        绘制样条插值的结果:
            主要参数:
                t: 时间序列，一个一维数组
                y: 时间序列对应的函数值，一个一维数组
                t_interp: 插值后的时间序列，一个一维数组
                y_interp: 插值后的时间序列对应的函数值，一个一维数组
            辅助参数:
                block: 图片显示后是否阻塞
                pause: 图片显示后的延迟时间,block为False时生效
                clear: 本次图片是否覆盖上次图片
                ion: 是否开启交互模式
            默认是“不阻塞+无等待+覆盖+无交互”，用于实时显示最新的插值结果
        """
        if ion: plt.ion()  # 开启交互模式
        if clear: plt.clf()
        # 绘制原始数据点
        plt.plot(t,y,'ro',label='original',markersize=3)
        # 绘制样条插值的结果
        plt.plot(t_interp,y_interp,'g-',label='interpolated',markersize=3)
        # 添加图例和标签
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title('Interpolation Result')
        # 显示图形
        plt.show(block=block)  # 非block并不会清除原来显示的图片，而是等下次图片来覆盖（前提是使用了plt.clf()）
        if pause > 0 and not block: plt.pause(pause)


if __name__ == '__main__':

    NODE_NAME = "follow_joint_trajectory_server"

    def loginfo(msg: str):
        rospy.loginfo("[{}] {}".format(NODE_NAME, msg))

    rospy.init_node(NODE_NAME)
    loginfo("Initializing {} node.".format(NODE_NAME))
    action_name = rospy.get_param('~action_name',default='arm_position_controller/follow_joint_trajectory')
    interpolation_type = rospy.get_param('~interpolation_type',default=5)
    MoveItAction(action_name,MoveItAction.CONTROL_MODE[1],interpolation=interpolation_type,plot=False)

    rospy.loginfo("Ready to follow joint trajectory.")
    rospy.spin()