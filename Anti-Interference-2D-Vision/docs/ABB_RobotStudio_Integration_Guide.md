# 视觉算法与 ABB RobotStudio 仿真接口集成指南
# Vision Algorithm & ABB RobotStudio Integration Guide

:Author: RussellCooper
:Date: 2026-03-27

本文档详细介绍了如何将本项目的高反光工件视觉识别算法（基于 FLARE 与 EdgeVision-C）与 **ABB RobotStudio** 仿真软件进行无缝集成。通过建立 TCP Socket 通信，实现从视觉检测、位姿计算到机器人引导抓取的端到端闭环仿真联调。

This document details how to seamlessly integrate the highly reflective workpiece vision recognition algorithm (based on FLARE and EdgeVision-C) with the **ABB RobotStudio** simulation software. By establishing TCP Socket communication, it achieves an end-to-end closed-loop simulation debugging process from vision detection and pose calculation to robot-guided picking.

---

## 1. 架构概述 (Architecture Overview)

系统的集成架构分为三层：**视觉算法层 (Python)**、**通信接口层 (Python Socket)** 与 **机器人控制层 (RAPID)**。

The integration architecture is divided into three layers: **Vision Algorithm Layer (Python)**, **Communication Interface Layer (Python Socket)**, and **Robot Control Layer (RAPID)**.

| 层级 (Layer) | 组件 (Component) | 职责 (Responsibility) |
|-------------|------------------|-----------------------|
| 视觉层 (Vision) | `main_pipeline.py` | 图像采集、HDR融合、FLARE特征提取、亚像素定位 |
| 接口层 (Interface) | `abb_robotstudio_interface.py` | 坐标系转换、JSON指令封装、TCP通信管理、状态机同步 |
| 控制层 (Control) | `abb_server.mod` (RobotStudio) | 解析JSON指令、执行 `MoveL`/`MoveJ`、反馈机器人位姿 |

---

## 2. 核心集成步骤 (Core Integration Steps)

### 2.1 坐标系标定与映射 (Coordinate System Calibration & Mapping)

视觉算法输出的像素坐标 $(u, v)$ 必须转换为机器人的工件坐标系 (WObj) 或基坐标系 (Base Frame)。本系统采用**眼在手外 (Eye-to-Hand)** 标定模型。

The pixel coordinates $(u, v)$ output by the vision algorithm must be converted to the robot's Work Object (WObj) or Base Frame. This system adopts the **Eye-to-Hand** calibration model.

在 `localization_and_calibration.py` 中，我们通过 PnP (Perspective-n-Point) 算法计算相机外参：

```python
# 计算相机到机器人基座的齐次变换矩阵
# Calculate homogeneous transformation matrix from camera to robot base
T_cam_to_base = compute_extrinsic_matrix(image_points, robot_tcp_points)
```

在实际集成中，视觉系统将识别到的目标三维坐标 $(X_c, Y_c, Z_c)$ 乘以该变换矩阵，得到机器人的目标坐标 $(X_r, Y_r, Z_r)$，然后通过 TCP 发送给 RobotStudio。

### 2.2 Python 端接口设计 (Python Interface Design)

为了保证代码的鲁棒性与可移植性，我们在 `abb_robotstudio_interface.py` 中实现了**工厂模式**，提供三层抽象：

1. **`AbbRobotStub`**：纯 Python 内存模拟，无需开启任何仿真软件，用于算法逻辑验证。
2. **`AbbRobotStudioSim`**：真实的 TCP Socket 客户端，连接到 RobotStudio 虚拟控制器。
3. **`AbbRobotEGM`**（2026-04-23 已实现）：基于 ABB Externally Guided Motion 的实时控制接口，UDP 协议直连真实 ABB 机器人控制器（端口 6510）。

**集成代码示例 (Integration Code Example):**

```python
from abb_robotstudio_interface import create_robot_interface

# 自动模式：优先尝试连接 RobotStudio，若连接失败则自动降级为 Stub 模拟器
robot = create_robot_interface(host="127.0.0.1", port=10000, mode="auto")

with robot:
    # 1. 机器人回零点 (Robot goes to home position)
    robot.home()
    
    # 2. 视觉算法处理获取目标位姿 (Vision algorithm processing)
    target_pose = vision_pipeline.process_frame(camera_image)
    
    # 3. 发送运动指令引导机器人抓取 (Send motion command to guide robot)
    robot.send_target(
        x_mm=target_pose.x, 
        y_mm=target_pose.y, 
        z_mm=target_pose.z,
        rx_deg=0.0, ry_deg=180.0, rz_deg=target_pose.angle,
        speed_mm_s=150.0, 
        zone="fine"
    )
    
    # 4. 等待运动完成 (Wait for motion to complete)
    robot.wait_done(timeout_s=10.0)
```

### 2.3 RobotStudio 端 RAPID 服务端 (RobotStudio RAPID Server)

在 RobotStudio 中，虚拟控制器运行 `abb_rapid/abb_server.mod`。该程序作为一个 TCP Server，监听 `10000` 端口。

通信协议采用**单行 JSON 格式 (Single-line JSON)**，以换行符 `\n` 作为消息边界。

**RAPID 核心逻辑解析 (RAPID Core Logic Analysis):**

```rapid
! 解析接收到的 JSON 指令
cmd := ExtractJsonString(recvBuf, "cmd");

TEST cmd
CASE "MoveL":
    ! 提取坐标 (Extract coordinates)
    x := ExtractJsonNum(recvBuf, "x");
    y := ExtractJsonNum(recvBuf, "y");
    ...
    ! 转换为四元数并构建目标点 (Convert to quaternion and build target)
    target := [[x, y, z], EulerToQuat(rx, ry, rz), ...];
    
    ! 执行直线运动 (Execute linear motion)
    MoveL target, v100, fine, tVisionGripper \WObj:=wobj_conveyor;
    
    ! 返回完成状态 (Return completion status)
    SendResponse "{""status"":""accepted""}";
```

---

## 3. 典型联调工作流 (Typical Debugging Workflow)

要完成一次完整的视觉与仿真集成联调，请遵循以下步骤：

### 步骤 1: 启动 RobotStudio 仿真 (Start RobotStudio Simulation)
1. 打开 ABB RobotStudio，新建空工作站。
2. 导入机器人模型（推荐 IRB 2600 或 IRB 1200）。
3. 从 `abb_rapid/abb_server.mod` 加载模块到 `T_ROB1` 任务。
4. 点击 `RAPID -> Run` 启动程序。此时 FlexPendant 会提示 `TCP 服务端已启动，等待连接...`。

### 步骤 2: 启动视觉算法 (Start Vision Algorithm)
在 Ubuntu/Linux 环境下，运行集成脚本：

```bash
# 运行完整的视觉引导分拣演示
python3 abb_robotstudio_interface.py --mode robotstudio --host <Windows_IP> --demo sorting
```
*注意：如果 RobotStudio 运行在 Windows 宿主机，而 Python 运行在 WSL/VM 中，请将 `<Windows_IP>` 替换为宿主机的实际 IP 地址，并确保 Windows 防火墙放行了 10000 端口。*

### 步骤 3: 观察闭环行为 (Observe Closed-loop Behavior)
- **Python 端**：将打印出视觉检测到的目标坐标，以及发送给机器人的 JSON 指令。
- **RobotStudio 端**：机器人将按照视觉算法计算出的轨迹进行移动（接近 -> 抓取 -> 提升 -> 放置）。

---

## 4. 真实机器人接口：ABB EGM（2026-04-23 更新）
## (Real Robot Interface: ABB EGM - Updated 2026-04-23)

### 4.1 AbbRobotEGM 概述

`AbbRobotEGM` 类实现了 UDP EGM（Externally Guided Motion）协议，可直连真实 ABB 机器人控制器，支持 Protobuf 消息格式和简化二进制回退格式。

**通信参数：**

| 参数 | 值 |
|------|-----|
| 协议 | UDP |
| 端口 | 6510 |
| 消息格式 | Protobuf（优先）/ 二进制回退 |
| 反馈机制 | `wait_done()` 轮询 |

### 4.2 AbbRobotEGM 使用示例

```python
from abb_robotstudio_interface import create_robot_interface

# 创建 EGM 接口连接到真实 ABB 机器人
robot = create_robot_interface(host="192.168.1.100", port=6510, mode="egm")

with robot:
    # 1. 机器人回零点
    robot.home()

    # 2. 视觉算法处理
    target_pose = vision_pipeline.process_frame(camera_image)

    # 3. 发送 EGM 运动指令
    robot.send_target(
        x_mm=target_pose.x,
        y_mm=target_pose.y,
        z_mm=target_pose.z,
        rx_deg=0.0, ry_deg=180.0, rz_deg=target_pose.angle,
        speed_mm_s=150.0
    )

    # 4. 等待运动完成（EGM 反馈轮询）
    robot.wait_done(timeout_s=10.0)
```

### 4.3 EGM 与 RobotStudio 仿真对比

| 特性 | EGM（真实机器人） | RobotStudio 仿真 |
|------|-------------------|------------------|
| 协议 | UDP | TCP |
| 端口 | 6510 | 10000 |
| 延迟 | < 5ms | 取决于 PC 性能 |
| 反馈 | 实时位置/速度 | JSON 状态 |
| 安全性 | 需急停装置 | 无风险 |

### 4.4 EGM 配置要求

- ABB 机器人控制器需安装 **EGM（Externally Guided Motion）** 选项
- 控制器网络配置：静态 IP 或 DHCP（推荐静态）
- 机器人需配置正确的工具坐标系 (tooldata) 和工件坐标系 (wobjdata)

---

## 5. 高级应用：全国复杂场景数据生成
## (Advanced: National Complex Scene Data Generation)

为了确保视觉算法在真实工厂环境中的鲁棒性，本项目集成了 `synth_national_scenes.py` 生成器。该生成器可模拟全国 8 种典型的大型金属高光面场景（如船厂、钢厂、桥梁、高铁等）及 7 种复杂光照条件。

在进行 RobotStudio 仿真前，建议使用该生成器生成特定场景的数据来微调 FLARE 模型：

```bash
# 生成 500 张船厂强侧光与焊接弧光场景的训练图像
python3 synth_national_scenes.py --n 500 --scene SHIPYARD --output ./dataset
```

*(详情请参考仓库 `docs/all_scenes_grid.png` 查看 8 种场景的生成图集示例。)*

---

## 6. 故障排除 (Troubleshooting)

| 现象 (Symptom) | 可能原因 (Possible Cause) | 解决方案 (Solution) |
|---------------|---------------------------|---------------------|
| Python 提示 `ConnectionRefusedError` | RobotStudio 未运行或端口被防火墙拦截 | 检查 RAPID 程序是否在运行状态；在 Windows 防火墙中添加入站规则开放 TCP 10000 端口 |
| 机器人运动方向与视觉相反 | 手眼标定矩阵的坐标轴定义不一致 | 检查 `localization_and_calibration.py` 中的外参计算，确认 Z 轴方向是否匹配 |
| 机器人报错 `ERR_ROBLIMIT` (超出轴限位) | 视觉给出的坐标超出了机器人的工作空间 | 在 Python 端增加坐标范围断言 (Assertion)；检查 `abb_server.mod` 中的工具坐标系 (tooldata) 定义 |

---
*@copyright Licensed under the Apache License, Version 2.0. Commercial use please contact RussellCooper.*
