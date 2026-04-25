"""
robot — 机器人控制与仿真接口包

包含以下子模块：
  abb_robotstudio_interface: ABB RobotStudio TCP/IP 通信接口
  abb_rapid/               : ABB RAPID 控制器代码（abb_server.mod）
"""

__all__ = [
    "AbbRobotBase",
    "AbbRobotStub",
    "AbbRobotStudioSim",
    "create_robot_interface",
]
