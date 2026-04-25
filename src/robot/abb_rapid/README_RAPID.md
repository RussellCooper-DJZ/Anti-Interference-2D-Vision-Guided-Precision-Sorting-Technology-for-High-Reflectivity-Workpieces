# ABB RobotStudio 仿真接入指南

**@author RussellCooper**

本目录包含在 ABB RobotStudio 仿真软件中运行的 RAPID 服务端程序，
配合 Python 端 `abb_robotstudio_interface.py` 实现视觉引导仿真联调。

---

## 快速开始

### 1. RobotStudio 端配置

#### 1.1 新建仿真站

1. 打开 ABB RobotStudio（建议 2023.1 或更高版本）
2. 新建空工作站 → 添加机器人模型（推荐 **IRB 2600-12/1.65** 或 **IRB 1200-7/0.7**）
3. 添加虚拟控制器（Virtual Controller），RobotWare 版本选 **7.x**

#### 1.2 加载 RAPID 程序

1. 在 Controller 面板中展开 `RAPID → T_ROB1 → Program Modules`
2. 右键 → `Load Module` → 选择本目录中的 `abb_server.mod`
3. 确认模块已加载，主程序为 `AbbVisionServer/main`

#### 1.3 配置 TCP 通信

RobotStudio 虚拟控制器默认允许本机 TCP 通信，无需额外配置。

| 参数 | 值 |
|------|----|
| 监听 IP | `0.0.0.0`（所有接口）|
| 监听端口 | `10000` |
| 协议 | TCP，JSON over newline |
| 超时 | 30 秒 |

> **注意**：如需从另一台机器连接，需在 Windows 防火墙中开放 10000 端口。

#### 1.4 运行仿真

1. 点击 `RAPID → Run` 启动主程序
2. 观察 FlexPendant 模拟器输出：
   ```
   === ABB Vision Server v1.0 | @author RussellCooper ===
   监听端口: 10000
   已回 Home 位置，等待视觉系统连接...
   TCP 服务端已启动，等待连接...
   ```

---

### 2. Python 端连接

#### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

#### 2.2 运行演示

```bash
# 自动模式（优先连接 RobotStudio，不可达则退化到模拟桩）
python3 abb_robotstudio_interface.py --mode auto --demo all

# 强制连接 RobotStudio（需先启动仿真）
python3 abb_robotstudio_interface.py --mode robotstudio --host 127.0.0.1 --port 10000 --demo sorting

# 纯模拟桩（无需任何软件）
python3 abb_robotstudio_interface.py --mode stub --demo all
```

#### 2.3 在主流水线中使用

```python
from abb_robotstudio_interface import create_robot_interface, RobotStudioDemoController

# 自动选择接口
robot = create_robot_interface(host="127.0.0.1", port=10000, mode="auto")

with robot:
    robot.home()
    # 发送视觉检测结果
    robot.send_target(x_mm=450.0, y_mm=120.0, z_mm=300.0,
                      rx_deg=0.0, ry_deg=180.0, rz_deg=45.0,
                      speed_mm_s=150.0, zone="z10")
    robot.wait_done(timeout_s=15.0)
```

---

## 通信协议参考

所有消息为单行 JSON，以 `\n` 结尾。

### 客户端 → 服务端（指令）

| 指令 | 说明 | 必填字段 |
|------|------|----------|
| `handshake` | 握手 | `client`, `version` |
| `MoveL` | 直线运动 | `x`, `y`, `z`, `rx`, `ry`, `rz`, `speed`, `zone` |
| `MoveJ` | 关节运动 | 同 MoveL |
| `home` | 回零点 | — |
| `get_status` | 查询状态 | — |
| `get_joints` | 查询关节角 | — |
| `get_tcp` | 查询 TCP 位姿 | — |
| `set_speed_override` | 设置速度倍率 | `percent`（0~100）|
| `ping` | 心跳 | — |
| `disconnect` | 断开 | — |

### 服务端 → 客户端（响应）

```json
// 握手响应
{"status": "ok", "controller": "RobotStudio-Sim", "version": "1.0", "author": "RussellCooper"}

// 运动指令响应（立即返回，运动异步执行）
{"status": "accepted"}

// 状态查询响应
{
  "status": "ok",
  "is_moving": false,
  "error_code": 0,
  "speed_override": 100,
  "current_pos": [450.0, 120.0, 300.0, 0.0, 180.0, 45.0]
}

// 关节角响应
{"status": "ok", "joints": [0.0, -30.0, 30.0, 0.0, 60.0, 0.0]}

// 心跳响应
{"status": "pong"}
```

---

## 演示模式说明

| 演示模式 | 命令 | 说明 |
|----------|------|------|
| `sorting` | `--demo sorting` | 视觉引导分拣（拾取→放置循环）|
| `weld` | `--demo weld` | 焊缝跟踪扫描演示 |
| `calibration` | `--demo calibration` | 手眼标定采集序列 |
| `all` | `--demo all` | 依次运行全部演示 |

---

## 机器人型号适配

`abb_server.mod` 默认使用 IRB 2600 工作空间参数。如需适配其他型号：

| 型号 | 工作半径 | 建议修改 |
|------|----------|----------|
| IRB 1200-7/0.7 | 700 mm | 缩小 `SCAN_POS` 的 x/z 值 |
| IRB 2600-12/1.65 | 1650 mm | 默认参数适用 |
| IRB 4600-45/2.05 | 2050 mm | 可适当增大工作空间 |

---

*@author RussellCooper — 本文件随仓库以 Apache 2.0 协议开源。*
