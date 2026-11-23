# easy-arm-control

基于 MoveIt 1（`moveit_commander`）的机械臂上层控制库：位姿 / 关节 / 笛卡尔运动、键盘与手柄遥操作，以及一个 `FollowJointTrajectory` action 服务器。

> 本仓库由 `airbot_play_control_upper` 迁移、改名而来（原 GitLab 仓库已无法访问）。
> 控制逻辑只通过 ROS / MoveIt 通信，不依赖任何机械臂驱动或 SDK；但**默认配置仍是 AIRBOT Play**：命名空间 `/airbot_play`、规划组 `airbot_play_arm` / `airbot_play_gripper`、话题 `/airbot_play/joint_cmd`、`/airbot_play/gripper/*` 等。用于其他机械臂时，需要提供同名的 MoveIt 配置和话题。

| 层级 | 名称 |
|------|------|
| GitHub 仓库 / pip 分发名 | `easy-arm-control` |
| Python 模块 | `easy_arm_control` |
| catkin 包 | `easy_arm_control` |

## 目录结构

```
├── pkgs/                          # pip 包（pyproject.toml）
│   ├── easy_arm_control/control.py    # RoboticArmAgent 等核心实现
│   └── airbot_play_control/           # 旧模块名的兼容别名（已弃用）
├── scripts/                       # ROS 节点脚本（rosrun easy_arm_control <脚本>）
├── demo/                          # 示例 notebook 与 hsv_finder.py
├── tests/                         # 冒烟测试（需要 ROS 环境）
└── tools/check_ast_equivalence.py # 校验两个提交之间的 Python 代码是否等价
```

## 安装

### 方式一：作为搭积木工程的一部分（推荐）

[`airbot_play_blocks_building`](https://github.com/OpenGHz/airbot_play_blocks_building) 以 git 子模块的形式包含本仓库，用 pixi 一键安装 ROS Noetic、MoveIt 和本包，不需要手动操作。

### 方式二：放进已有的 ROS Noetic 工作空间

```bash
cd <your_ws>/src
git clone https://github.com/OpenGHz/easy-arm-control.git easy_arm_control  # 目录名 = 包名
pip install -e easy_arm_control/pkgs
cd .. && catkin build && source devel/setup.bash
```

运行依赖：`rospy`、`moveit_commander`、`geometry_msgs` / `sensor_msgs` / `moveit_msgs` 等由 ROS 环境提供；`numpy`、`pyserial`、`robotics-tools` 由 pip 安装。支持 Python 3.8（apt 版 Noetic）和 Python 3.12（RoboStack）。

## 快速开始

先启动机械臂的 MoveIt（`move_group` 运行在 `/airbot_play` 命名空间下），然后：

**键盘遥操作**

```bash
rosrun easy_arm_control airbot_play_keyboard.py --init_pose configs/control/real.yaml
```

`--init_pose` 可以是 `forward`、`pick`、配置文件路径（`.yaml`，也支持 `.json`），或逗号分隔的 `x,y,z,roll,pitch,yaw`。启动后会打印按键说明（`?` 键可再次查看）。

**Python API**（摘自 `demo/simple_api_example.ipynb`）

```python
import rospy
from easy_arm_control.control import RoboticArmAgent, GripperSimpleControl

NODE_NAME = "my_arm_control"
rospy.init_node(NODE_NAME)
arm = RoboticArmAgent(node_name=NODE_NAME)
gripper = GripperSimpleControl()

print(arm.get_current_joint_values())  # 关节角
print(arm.get_current_pose())          # 末端位姿（相对 base_link）

arm.set_and_go_to_pose_target([0.3472, 0.0796, -0.231, 0, 0.7, 0, 0.714])  # xyz + 四元数
arm.go_to_shift_single_axis_target(2, 0.02)  # 沿 z 轴增量移动 2 cm
gripper.control(1)  # 闭合；0 为张开
```

更多用法见 `demo/` 下的 notebook。

## 脚本

| 脚本 | 用途 | 备注 |
|------|------|------|
| `airbot_play_keyboard.py` | 键盘遥操作 | |
| `airbot_play_joyControl.py` | 手柄遥操作（订阅 `Joy`） | 额外依赖 `IPython` |
| `airbot_play_follow_joint_trajectory_server.py` | MoveIt 的 `FollowJointTrajectory` action 服务器 | `moveit1_interface` 中 `use_basic:=false` 时使用 |
| `airbot_play_pickplace_calib.py` | 搭积木的 Pick/Place 标定 | 依赖搭积木工程中的 `airbot_play_pick_place` 模块 |
| `nothing_detected.py` | 视觉未检测到目标时，按上一次的方向持续发布 `/target_TF` | |
| `demo/hsv_finder.py` | 用滑动条确定物块颜色的 HSV 阈值 | |

脚本名沿用原名（保留 `airbot_play_` 前缀），以兼容已有文档和命令。

## 从 `airbot_play_control` / `airbot_play_control_upper` 迁移

| 原来 | 现在 |
|------|------|
| `from airbot_play_control.control import RoboticArmAgent` | `from easy_arm_control.control import RoboticArmAgent` |
| `rosrun airbot_play_control_upper <脚本>` | `rosrun easy_arm_control <脚本>` |
| launch 文件中的 `pkg="airbot_play_control_upper"` | `pkg="easy_arm_control"` |
| `pip install ./pkgs`（`setup.py`，分发名 `airbot_play_control` 1.1） | 同样的命令（`pyproject.toml`，分发名 `easy-arm-control` 2.0.0） |

类名、函数名、参数和默认值都没有变化。旧模块名 `airbot_play_control` 仍可导入，返回的是同一个模块对象，只会发出一条 `DeprecationWarning`。catkin 包名则没有兼容层，`rosrun` 和 launch 文件需要更新。

## 开发

```bash
pip install pre-commit
pre-commit install          # 之后每次 git commit 自动执行
pre-commit run --all-files  # 手动对全部文件执行
```

格式化完全由 `.pre-commit-config.yaml` 决定：ruff-format（配置见 `ruff.toml`，语法保持兼容 Python 3.8）、nbstripout、行尾 / 换行 / 文件末尾修正、JSON / XML / YAML 检查。ruff lint 只启用会导致运行错误的规则（语法错误、未定义名称等），不会自动修改代码。

测试需要在 ROS Noetic 环境中执行（CI 在 `ros:noetic-ros-base` 容器里运行）：

```bash
python3 -m pytest -v tests
```

### 等价性校验

`tools/check_ast_equivalence.py` 比较两个提交中所有 `.py` 文件和 notebook 代码单元的抽象语法树（AST）。注释、空行、引号、换行等排版变化不会体现在 AST 中，而任何逻辑改动都会被发现。本仓库前三个提交可以这样复核：

```bash
git log --oneline --reverse | head -3
python3 tools/check_ast_equivalence.py <导入提交> <格式化提交>
python3 tools/check_ast_equivalence.py <格式化提交> <改名提交> --rename airbot_play_control=easy_arm_control
```

两次结果都应为 `EQUIVALENT`。唯一做的规范化是 docstring 首尾空白（格式化工具会调整 docstring 缩进；代码中没有读取 `__doc__` 的地方，工具也会检查这一点）。

## 已知问题

以下问题来自原代码。本仓库只做了格式化和改名，没有修改逻辑，因此保留原样。`ruff check --select E4,E7,E9,F` 的结果：

| 规则 | 数量 | 说明 |
|------|------|------|
| F401 | 5 | 导入了但未使用 |
| E722 | 4 | 裸 `except:` |
| E401 | 3 | 一行导入多个模块 |
| E712 | 3 | 与 `True` / `False` 用 `==` 比较 |
| F541 | 2 | 没有占位符的 f-string |
| F841 | 2 | 赋值后未使用的局部变量 |

另外：

- `control.py` 在模块顶层继承了 `MoveGroupCommander`，所以没有 ROS / MoveIt 的环境中无法导入本包（顶部的 `try/except ImportError` 只会打印提示）。
- 夹爪、命名空间、规划组等默认值写死为 AIRBOT Play，参数化留到以后的版本。

## 来源与许可证

- 来源：`gitlab.com/OpenGHz/airbot_play_control_upper` 分支 `main`，提交 `40317fe`（2025-03-25，"support mujoco mode"），从 `airbot_play_block_building:1.4.0` 镜像中恢复。原提交历史没有保留。
- 许可证：[MIT](LICENSE)。
