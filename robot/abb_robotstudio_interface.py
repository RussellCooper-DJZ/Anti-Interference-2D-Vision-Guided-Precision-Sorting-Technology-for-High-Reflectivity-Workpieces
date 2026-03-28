"""
abb_robotstudio_interface.py — ABB RobotStudio 仿真软件通信接口
:Author: RussellCooper

本模块提供三个层次的 ABB 机器人通信接口，接口完全统一，
上层代码无需修改即可在三种模式之间切换：

  1. AbbRobotStub         — 纯 Python 模拟桩（无需任何软件）
  2. AbbRobotStudioSim    — 通过 TCP Socket 接入 ABB RobotStudio 仿真
  3. AbbRobotEGM          — 通过 EGM 协议接入真实 ABB 机器人（接口预留）

ABB RobotStudio 接入方式：
  - 在 RobotStudio 中加载 abb_server.mod（本仓库提供）
  - RAPID 程序在虚拟控制器上监听 TCP 端口 10000
  - 本模块作为客户端发送 JSON 格式的目标位姿
  - 支持：MoveL / MoveJ / 速度/区域配置 / 状态查询 / 事件回调

协议格式（JSON over TCP，以 \\n 分隔）：
  发送：{"cmd": "MoveL", "x": 500.0, "y": 0.0, "z": 400.0,
         "rx": 0.0, "ry": 180.0, "rz": 0.0,
         "speed": 100, "zone": "z10"}
  接收：{"status": "ok", "pos": [500.0, 0.0, 400.0, 0.0, 180.0, 0.0]}

快速开始::

    from abb_robotstudio_interface import create_robot_interface

    # 自动选择接口（优先仿真，退化到模拟桩）
    robot = create_robot_interface(host="127.0.0.1", port=10000)
    with robot:
        robot.home()
        robot.send_target(500.0, 0.0, 400.0, rx_deg=0.0, ry_deg=180.0)
        robot.wait_done()
"""

import json
import math
import queue
import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

__all__ = [
    "AbbRobotBase",
    "AbbRobotStub",
    "AbbRobotStudioSim",
    "RobotStudioDemoController",
    "create_robot_interface"
]



# ============================================================
# 1. 基类接口（所有实现必须继承此类）
# ============================================================

class AbbRobotBase:
    """ABB 机器人通信基类，定义统一接口。"""

    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def send_target(
        self,
        x_mm: float, y_mm: float, z_mm: float,
        rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0,
        speed_mm_s: float = 100.0,
        zone: str = "z10",
        move_type: str = "MoveL",
    ) -> bool:
        raise NotImplementedError

    def get_status(self) -> Dict:
        raise NotImplementedError

    def wait_done(self, timeout_s: float = 30.0) -> bool:
        raise NotImplementedError

    def home(self) -> bool:
        raise NotImplementedError

    def set_event_callback(self, callback: Callable[[str, Dict], None]):
        """注册事件回调（状态变化、到位、错误等）。"""
        self._event_callback = callback

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# ============================================================
# 2. 纯模拟桩（无需任何外部软件）
# ============================================================

class AbbRobotStub(AbbRobotBase):
    """
    ABB 机器人纯 Python 模拟桩。

    完全在内存中模拟机器人运动，无需 RobotStudio 或真实机器人。
    适用于：算法开发、CI 测试、离线演示。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._connected = False
        self._move_count = 0
        self._current_pos = [0.0, 0.0, 500.0, 0.0, 180.0, 0.0]
        self._is_moving = False
        self._event_callback: Optional[Callable] = None
        self._move_log: List[Dict] = []

    def connect(self) -> bool:
        self._connected = True
        self._log("已连接（纯模拟桩模式）")
        return True

    def disconnect(self):
        self._connected = False
        self._log("已断开连接")

    def send_target(
        self,
        x_mm: float, y_mm: float, z_mm: float,
        rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0,
        speed_mm_s: float = 100.0,
        zone: str = "z10",
        move_type: str = "MoveL",
    ) -> bool:
        if not self._connected:
            return False
        self._move_count += 1
        target = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        record = {
            'seq':       self._move_count,
            'move_type': move_type,
            'target':    target,
            'speed':     speed_mm_s,
            'zone':      zone,
            'timestamp': time.time(),
        }
        self._move_log.append(record)
        dist = math.sqrt(sum((a - b) ** 2
                             for a, b in zip(self._current_pos[:3], target[:3])))
        self._current_pos = target
        self._log(
            f"#{self._move_count:04d} {move_type}  "
            f"({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm  "
            f"rot=({rx_deg:.1f}, {ry_deg:.1f}, {rz_deg:.1f})°  "
            f"v={speed_mm_s:.0f}mm/s  zone={zone}  dist={dist:.1f}mm"
        )
        if self._event_callback:
            self._event_callback("move_done", record)
        return True

    def get_status(self) -> Dict:
        return {
            'mode':        'stub',
            'connected':   self._connected,
            'move_count':  self._move_count,
            'current_pos': self._current_pos,
            'is_moving':   False,
            'error_code':  0,
        }

    def wait_done(self, timeout_s: float = 30.0) -> bool:
        return True

    def home(self) -> bool:
        return self.send_target(0.0, 0.0, 500.0, 0.0, 180.0, 0.0,
                                speed_mm_s=200.0, zone="z0")

    def get_move_log(self) -> List[Dict]:
        return list(self._move_log)

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ABB-Stub] {msg}")


# ============================================================
# 3. ABB RobotStudio TCP 仿真接口
# ============================================================

class AbbRobotStudioSim(AbbRobotBase):
    """
    ABB RobotStudio 仿真接口（TCP Socket 通信）。

    通过 TCP 与 RobotStudio 虚拟控制器中运行的 RAPID 服务端通信。
    RAPID 服务端代码见本仓库 abb_server.mod。

    通信协议：
      - 每条消息为一行 JSON，以 '\\n' 结尾
      - 客户端（本模块）发送指令，服务端（RAPID）返回响应
      - 心跳机制：每 5 秒发送 {"cmd": "ping"}，服务端回 {"status": "pong"}

    Args:
        host:         RobotStudio 所在主机 IP（本机仿真用 127.0.0.1）
        port:         RAPID 服务端监听端口（默认 10000）
        timeout_s:    连接/接收超时（秒）
        verbose:      是否打印详细日志
        auto_fallback: 连接失败时是否自动退化到模拟桩
    """

    PROTOCOL_VERSION = "1.0"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10000,
        timeout_s: float = 5.0,
        verbose: bool = True,
        auto_fallback: bool = True,
    ):
        self.host          = host
        self.port          = port
        self.timeout_s     = timeout_s
        self.verbose       = verbose
        self.auto_fallback = auto_fallback

        self._sock:     Optional[socket.socket] = None
        self._connected = False
        self._move_count = 0
        self._current_pos = [0.0, 0.0, 500.0, 0.0, 180.0, 0.0]
        self._is_moving   = False
        self._event_callback: Optional[Callable] = None
        self._recv_buffer = ""

        # 心跳线程
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()

        # 待确认的移动请求队列
        self._ack_queue: queue.Queue = queue.Queue()

    # ---- 连接管理 ----

    def connect(self) -> bool:
        """连接到 RobotStudio 仿真服务端。"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.timeout_s)
            self._sock.connect((self.host, self.port))
            self._connected = True
            self._log(f"已连接到 RobotStudio @ {self.host}:{self.port}")

            # 握手
            resp = self._send_cmd({"cmd": "handshake",
                                   "client": "ShipVisionPipeline",
                                   "version": self.PROTOCOL_VERSION})
            if resp and resp.get("status") == "ok":
                self._log(f"握手成功，控制器: {resp.get('controller', 'unknown')}")
            else:
                self._log(f"握手响应异常: {resp}")

            # 启动心跳线程
            self._stop_heartbeat.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()
            return True

        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            self._log(f"连接失败: {e}")
            self._connected = False
            self._sock = None
            return False

    def disconnect(self):
        """断开连接。"""
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        if self._sock:
            try:
                self._send_cmd({"cmd": "disconnect"})
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._connected = False
        self._log("已断开连接")

    # ---- 运动指令 ----

    def send_target(
        self,
        x_mm: float, y_mm: float, z_mm: float,
        rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0,
        speed_mm_s: float = 100.0,
        zone: str = "z10",
        move_type: str = "MoveL",
    ) -> bool:
        """
        发送运动目标到 RobotStudio。

        Args:
            x_mm, y_mm, z_mm:    目标位置（机器人基坐标系，mm）
            rx_deg, ry_deg, rz_deg: 目标姿态（欧拉角 ZYX，度）
            speed_mm_s:          运动速度（mm/s），映射到 ABB v 数据
            zone:                到位区域（z0/z5/z10/z50/fine）
            move_type:           运动类型（MoveL=直线，MoveJ=关节）

        Returns:
            True = 指令已被 RAPID 服务端接受
        """
        if not self._connected:
            self._log("错误：未连接")
            return False

        # 速度映射（mm/s → ABB v 数据名称）
        speed_name = self._map_speed(speed_mm_s)

        cmd = {
            "cmd":       move_type,
            "x":         round(x_mm,   3),
            "y":         round(y_mm,   3),
            "z":         round(z_mm,   3),
            "rx":        round(rx_deg, 3),
            "ry":        round(ry_deg, 3),
            "rz":        round(rz_deg, 3),
            "speed":     speed_name,
            "zone":      zone,
            "seq":       self._move_count + 1,
        }

        resp = self._send_cmd(cmd)
        if resp and resp.get("status") in ("ok", "accepted"):
            self._move_count += 1
            self._current_pos = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            self._log(
                f"#{self._move_count:04d} {move_type}  "
                f"({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm  "
                f"rot=({rx_deg:.1f}, {ry_deg:.1f}, {rz_deg:.1f})°  "
                f"v={speed_mm_s:.0f}mm/s  zone={zone}"
            )
            if self._event_callback:
                self._event_callback("move_sent", cmd)
            return True
        else:
            self._log(f"指令被拒绝: {resp}")
            return False

    def get_status(self) -> Dict:
        """查询机器人当前状态。"""
        if not self._connected:
            return {'mode': 'robotstudio', 'connected': False}
        resp = self._send_cmd({"cmd": "get_status"})
        if resp:
            resp['mode'] = 'robotstudio'
            resp['move_count'] = self._move_count
            return resp
        return {
            'mode':        'robotstudio',
            'connected':   self._connected,
            'move_count':  self._move_count,
            'current_pos': self._current_pos,
            'is_moving':   self._is_moving,
            'error_code':  -1,
        }

    def wait_done(self, timeout_s: float = 30.0) -> bool:
        """
        等待机器人到达目标位置。

        轮询 RAPID 服务端的 is_moving 状态，直到运动完成或超时。
        """
        if not self._connected:
            return False
        t_start = time.time()
        while time.time() - t_start < timeout_s:
            resp = self._send_cmd({"cmd": "get_status"})
            if resp:
                is_moving = resp.get("is_moving", False)
                self._is_moving = is_moving
                if not is_moving:
                    return True
            time.sleep(0.05)
        self._log(f"wait_done 超时（{timeout_s}s）")
        return False

    def home(self) -> bool:
        """回零点（Home 位置）。"""
        resp = self._send_cmd({"cmd": "home"})
        return resp is not None and resp.get("status") == "ok"

    def set_speed_override(self, percent: int) -> bool:
        """
        设置速度倍率（0~100%）。

        对应 RAPID 中的 SpeedRefresh 或 VelSet 指令。
        """
        resp = self._send_cmd({"cmd": "set_speed_override", "percent": percent})
        return resp is not None and resp.get("status") == "ok"

    def set_tool(self, tool_name: str) -> bool:
        """切换工具坐标系（对应 RAPID tooldata）。"""
        resp = self._send_cmd({"cmd": "set_tool", "tool": tool_name})
        return resp is not None and resp.get("status") == "ok"

    def set_wobj(self, wobj_name: str) -> bool:
        """切换工件坐标系（对应 RAPID wobjdata）。"""
        resp = self._send_cmd({"cmd": "set_wobj", "wobj": wobj_name})
        return resp is not None and resp.get("status") == "ok"

    def get_joint_angles(self) -> Optional[List[float]]:
        """获取当前关节角度（度）。"""
        resp = self._send_cmd({"cmd": "get_joints"})
        if resp and "joints" in resp:
            return resp["joints"]
        return None

    def get_tcp_pose(self) -> Optional[List[float]]:
        """获取当前 TCP 位姿 [x, y, z, rx, ry, rz]（mm, 度）。"""
        resp = self._send_cmd({"cmd": "get_tcp"})
        if resp and "tcp" in resp:
            return resp["tcp"]
        return None

    # ---- 内部通信 ----

    def _send_cmd(self, cmd: Dict) -> Optional[Dict]:
        """
        发送 JSON 指令并等待响应（同步，带超时）。

        Returns:
            解析后的响应字典，失败返回 None
        """
        if not self._sock:
            return None
        try:
            msg = json.dumps(cmd, ensure_ascii=False) + "\n"
            self._sock.sendall(msg.encode("utf-8"))
            # 接收响应
            resp_str = self._recv_line()
            if resp_str:
                return json.loads(resp_str)
            return None
        except (socket.timeout, ConnectionResetError, BrokenPipeError,
                json.JSONDecodeError, OSError) as e:
            self._log(f"通信错误: {e}")
            self._connected = False
            return None

    def _recv_line(self) -> Optional[str]:
        """接收一行 JSON 响应（以 '\\n' 结尾）。"""
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if "\n" in self._recv_buffer:
                line, self._recv_buffer = self._recv_buffer.split("\n", 1)
                return line.strip()
            try:
                chunk = self._sock.recv(4096).decode("utf-8")
                if not chunk:
                    return None
                self._recv_buffer += chunk
            except socket.timeout:
                continue
        return None

    def _heartbeat_loop(self):
        """心跳线程：每 5 秒发送 ping，检测连接存活。"""
        while not self._stop_heartbeat.wait(timeout=5.0):
            if not self._connected or not self._sock:
                break
            resp = self._send_cmd({"cmd": "ping"})
            if resp is None or resp.get("status") != "pong":
                self._log("心跳超时，连接可能已断开")
                self._connected = False
                if self._event_callback:
                    self._event_callback("disconnected", {})
                break

    @staticmethod
    def _map_speed(speed_mm_s: float) -> str:
        """将速度（mm/s）映射到 ABB RAPID v 数据名称。"""
        # ABB 标准速度数据：v5/v10/v20/v50/v100/v200/v300/v500/v1000/v2000
        speed_table = [
            (5,    "v5"),   (10,   "v10"),  (20,   "v20"),
            (50,   "v50"),  (100,  "v100"), (200,  "v200"),
            (300,  "v300"), (500,  "v500"), (1000, "v1000"),
            (2000, "v2000"),
        ]
        for threshold, name in speed_table:
            if speed_mm_s <= threshold:
                return name
        return "v2000"

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ABB-RobotStudio] {msg}")


# ============================================================
# 4. 仿真演示控制器
# ============================================================

class RobotStudioDemoController:
    """
    ABB RobotStudio 仿真演示控制器。

    提供高层演示接口：
      - run_vision_demo()    : 视觉引导分拣演示
      - run_path_demo()      : 路径规划演示（焊缝跟踪）
      - run_calibration_demo(): 手眼标定演示序列
    """

    # 标准演示位置（机器人基坐标系，mm）
    HOME_POS       = (0.0,    0.0,    500.0,  0.0, 180.0, 0.0)
    SCAN_POS       = (400.0,  0.0,    350.0,  0.0, 180.0, 0.0)
    PICKUP_ABOVE   = (400.0,  200.0,  300.0,  0.0, 180.0, 0.0)
    PICKUP_DOWN    = (400.0,  200.0,  150.0,  0.0, 180.0, 0.0)
    PLACE_ABOVE    = (-300.0, 200.0,  300.0,  0.0, 180.0, 0.0)
    PLACE_DOWN     = (-300.0, 200.0,  150.0,  0.0, 180.0, 0.0)

    def __init__(self, robot: AbbRobotBase, verbose: bool = True):
        self.robot   = robot
        self.verbose = verbose
        self._demo_log: List[Dict] = []

    def _move(self, pos: Tuple, speed: float = 150.0,
              zone: str = "z10", move_type: str = "MoveL") -> bool:
        ok = self.robot.send_target(*pos, speed_mm_s=speed,
                                    zone=zone, move_type=move_type)
        if ok:
            self._demo_log.append({
                'pos': pos, 'speed': speed,
                'zone': zone, 'time': time.time()
            })
        return ok

    def run_vision_demo(
        self,
        detections: List[Dict],
        plane_z_mm: float = 150.0,
        approach_height_mm: float = 100.0,
    ) -> Dict:
        """
        视觉引导分拣演示。

        根据视觉检测结果，依次对每个目标执行：
          扫描位 → 接近位 → 抓取位 → 提升 → 放置位 → 回扫描位

        Args:
            detections:         SubpixelLocalizer 输出（含 position_robot_mm）
            plane_z_mm:         工件平面高度（mm）
            approach_height_mm: 接近高度（mm，工件上方）

        Returns:
            演示统计信息
        """
        self._log(f"=== 视觉引导分拣演示 | 目标数: {len(detections)} ===")
        t_start = time.time()
        success_count = 0

        # 回扫描位
        self._move(self.SCAN_POS, speed=300.0, zone="z50")
        self.robot.wait_done(timeout_s=10.0)

        for i, det in enumerate(detections):
            pos = det.get('position_robot_mm')
            if pos is None:
                self._log(f"  目标 [{i}] 无坐标，跳过")
                continue

            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            rz = det.get('orientation_deg', 0.0)
            ftype = det.get('feature_type', 'unknown')

            self._log(f"\n  目标 [{i}] 类型={ftype}  "
                      f"位置=({x:.1f}, {y:.1f}, {z:.1f}) mm  "
                      f"方向={rz:.1f}°")

            # 1. 接近位（目标正上方）
            approach_z = plane_z_mm + approach_height_mm
            self._log(f"    → 移动到接近位 z={approach_z:.0f}mm")
            self._move((x, y, approach_z, 0.0, 180.0, rz),
                       speed=200.0, zone="z10")
            self.robot.wait_done(timeout_s=15.0)

            # 2. 抓取位（工件表面）
            self._log(f"    → 下降到抓取位 z={plane_z_mm:.0f}mm")
            self._move((x, y, plane_z_mm, 0.0, 180.0, rz),
                       speed=50.0, zone="fine")
            self.robot.wait_done(timeout_s=15.0)

            # 3. 提升
            self._log(f"    → 提升到 z={approach_z:.0f}mm")
            self._move((x, y, approach_z, 0.0, 180.0, rz),
                       speed=100.0, zone="z10")
            self.robot.wait_done(timeout_s=10.0)

            # 4. 移动到放置位上方
            self._log(f"    → 移动到放置位")
            self._move(self.PLACE_ABOVE, speed=300.0, zone="z50")
            self.robot.wait_done(timeout_s=15.0)

            # 5. 放置
            self._move(self.PLACE_DOWN, speed=50.0, zone="fine")
            self.robot.wait_done(timeout_s=10.0)

            # 6. 提升离开
            self._move(self.PLACE_ABOVE, speed=150.0, zone="z10")
            self.robot.wait_done(timeout_s=10.0)

            success_count += 1
            self._log(f"  目标 [{i}] 完成 ✓")

        # 回 Home
        self._log("\n  → 回 Home 位置")
        self._move(self.HOME_POS, speed=300.0, zone="z50")
        self.robot.wait_done(timeout_s=15.0)

        elapsed = time.time() - t_start
        result = {
            'total':    len(detections),
            'success':  success_count,
            'elapsed_s': elapsed,
            'avg_cycle_s': elapsed / max(success_count, 1),
        }
        self._log(f"\n=== 演示完成 | 成功 {success_count}/{len(detections)} | "
                  f"总耗时 {elapsed:.1f}s ===")
        return result

    def run_weld_seam_demo(
        self,
        weld_lines: List[Dict],
        scan_height_mm: float = 200.0,
        scan_speed_mm_s: float = 80.0,
    ) -> Dict:
        """
        焊缝跟踪演示（沿检测到的焊缝路径运动）。

        Args:
            weld_lines:      detect_weld_lines() 输出
            scan_height_mm:  扫描高度（mm）
            scan_speed_mm_s: 扫描速度（mm/s）
        """
        self._log(f"=== 焊缝跟踪演示 | 焊缝数: {len(weld_lines)} ===")
        t_start = time.time()

        self._move(self.HOME_POS, speed=300.0, zone="z50")
        self.robot.wait_done(timeout_s=10.0)

        for i, line in enumerate(weld_lines):
            p1_px, p2_px = line['endpoints']
            angle = line['angle_deg']
            length = line['length_px']

            self._log(f"\n  焊缝 [{i}] 角度={angle:.1f}°  "
                      f"长度={length:.0f}px")

            # 简化：将像素坐标映射到机器人坐标（实际需手眼标定）
            # 此处使用固定比例（1px ≈ 0.5mm）作为演示
            scale = 0.5
            x1 = float(p1_px[0]) * scale - 200.0
            y1 = float(p1_px[1]) * scale - 150.0
            x2 = float(p2_px[0]) * scale - 200.0
            y2 = float(p2_px[1]) * scale - 150.0

            # 移动到焊缝起点上方
            self._move((x1, y1, scan_height_mm + 50, 0.0, 180.0, angle),
                       speed=200.0, zone="z20")
            self.robot.wait_done(timeout_s=10.0)

            # 下降到扫描高度
            self._move((x1, y1, scan_height_mm, 0.0, 180.0, angle),
                       speed=50.0, zone="fine")
            self.robot.wait_done(timeout_s=10.0)

            # 沿焊缝扫描（直线运动）
            self._log(f"    → 沿焊缝扫描...")
            self._move((x2, y2, scan_height_mm, 0.0, 180.0, angle),
                       speed=scan_speed_mm_s, zone="fine")
            self.robot.wait_done(timeout_s=30.0)

            # 提升
            self._move((x2, y2, scan_height_mm + 50, 0.0, 180.0, angle),
                       speed=150.0, zone="z10")
            self.robot.wait_done(timeout_s=10.0)
            self._log(f"  焊缝 [{i}] 完成 ✓")

        self._move(self.HOME_POS, speed=300.0, zone="z50")
        self.robot.wait_done(timeout_s=10.0)

        elapsed = time.time() - t_start
        result = {
            'total':     len(weld_lines),
            'elapsed_s': elapsed,
        }
        self._log(f"\n=== 焊缝演示完成 | 总耗时 {elapsed:.1f}s ===")
        return result

    def run_calibration_sequence(
        self,
        n_poses: int = 20,
        workspace_mm: float = 200.0,
    ) -> List[Dict]:
        """
        手眼标定采集序列演示。

        在工作空间内随机采样 n_poses 个姿态，
        每个姿态停留 1 秒（供视觉系统采集标定图像）。

        Args:
            n_poses:      采集姿态数量（建议 15~30）
            workspace_mm: 工作空间范围（mm）

        Returns:
            采集的姿态列表
        """
        import random
        self._log(f"=== 手眼标定序列 | 采集 {n_poses} 个姿态 ===")
        poses = []
        rng = random.Random(42)

        self._move(self.HOME_POS, speed=300.0, zone="z50")
        self.robot.wait_done(timeout_s=10.0)

        for i in range(n_poses):
            # 随机姿态（在工作空间内）
            x  = rng.uniform(-workspace_mm, workspace_mm) + 300.0
            y  = rng.uniform(-workspace_mm, workspace_mm)
            z  = rng.uniform(200.0, 400.0)
            rx = rng.uniform(-20.0, 20.0)
            ry = rng.uniform(160.0, 200.0)
            rz = rng.uniform(-30.0, 30.0)

            self._log(f"  姿态 [{i+1:02d}/{n_poses}]: "
                      f"({x:.0f}, {y:.0f}, {z:.0f}) mm")
            self._move((x, y, z, rx, ry, rz), speed=150.0, zone="fine")
            self.robot.wait_done(timeout_s=15.0)

            poses.append({
                'seq':  i + 1,
                'pos':  [x, y, z, rx, ry, rz],
                'time': time.time(),
            })
            time.sleep(0.5)  # 等待视觉系统采集

        self._move(self.HOME_POS, speed=300.0, zone="z50")
        self.robot.wait_done(timeout_s=10.0)
        self._log(f"=== 标定序列完成 | 采集 {len(poses)} 个姿态 ===")
        return poses

    def _log(self, msg: str):
        if self.verbose:
            print(f"[DemoController] {msg}")


# ============================================================
# 5. 工厂函数（自动选择接口）
# ============================================================

def create_robot_interface(
    host: str = "127.0.0.1",
    port: int = 10000,
    mode: str = "auto",
    verbose: bool = True,
) -> AbbRobotBase:
    """
    工厂函数：自动选择最合适的机器人接口。

    Args:
        host:    RobotStudio 主机 IP
        port:    RAPID 服务端端口
        mode:    'auto'（自动）/ 'robotstudio'（强制仿真）/ 'stub'（强制模拟桩）
        verbose: 是否打印日志

    Returns:
        AbbRobotBase 实例（已连接）

    用法::

        robot = create_robot_interface()
        with robot:
            robot.home()
            robot.send_target(500, 0, 400)
    """
    if mode == "stub":
        robot = AbbRobotStub(verbose=verbose)
        robot.connect()
        return robot

    if mode == "robotstudio" or mode == "auto":
        sim = AbbRobotStudioSim(host=host, port=port,
                                verbose=verbose, auto_fallback=True)
        if sim.connect():
            return sim
        elif mode == "auto":
            print(f"[create_robot_interface] RobotStudio 不可达，退化到模拟桩")
            stub = AbbRobotStub(verbose=verbose)
            stub.connect()
            return stub
        else:
            raise ConnectionError(f"无法连接到 RobotStudio @ {host}:{port}")

    raise ValueError(f"未知 mode: {mode}")


# ============================================================
# 6. 命令行演示入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ABB RobotStudio 接口演示")
    parser.add_argument("--mode",  type=str, default="stub",
                        choices=["stub", "robotstudio", "auto"])
    parser.add_argument("--host",  type=str, default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=10000)
    parser.add_argument("--demo",  type=str, default="sorting",
                        choices=["sorting", "weld", "calibration", "all"])
    args = parser.parse_args()

    print("=" * 60)
    print("  ABB RobotStudio 仿真接口演示")
    print(f"  模式: {args.mode}  主机: {args.host}:{args.port}")
    print("=" * 60)

    # 创建接口
    robot = create_robot_interface(
        host=args.mode == "stub" and "127.0.0.1" or args.host,
        port=args.port,
        mode=args.mode,
        verbose=True,
    )
    demo = RobotStudioDemoController(robot, verbose=True)

    # 生成模拟检测结果
    from synth_dataset_generator import synthesize_one_sample, set_seed
    from localization_and_calibration import SubpixelLocalizer, CoordinateTransformer, detect_glare_regions
    import cv2 as _cv2
    import numpy as _np

    set_seed(42)
    sample = synthesize_one_sample(h=512, w=512)
    image  = sample['image']
    mask   = sample['mask']
    edge   = sample['edge']

    gray  = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
    glare = detect_glare_regions(image)
    loc   = SubpixelLocalizer(min_area=100)
    dets  = loc.localize(mask, edge, intensity_image=gray, glare_mask=glare)

    # 添加模拟机器人坐标
    K  = _np.array([[800,0,256],[0,800,256],[0,0,1]], dtype=_np.float64)
    T  = _np.eye(4)
    T[0,3] = 300.0; T[1,3] = 0.0; T[2,3] = 800.0
    transformer = CoordinateTransformer(K, _np.zeros((1,5)), T)
    dets = transformer.transform_localization_results(dets, plane_d=800.0)

    lines = loc.detect_weld_lines(edge)

    print(f"\n检测到 {len(dets)} 个目标，{len(lines)} 条焊缝")

    # 运行演示
    if args.demo in ("sorting", "all"):
        demo.run_vision_demo(dets, plane_z_mm=150.0)

    if args.demo in ("weld", "all"):
        demo.run_weld_seam_demo(lines[:3])

    if args.demo in ("calibration", "all"):
        demo.run_calibration_sequence(n_poses=5)

    # 打印状态
    status = robot.get_status()
    print(f"\n最终状态: {status}")
    robot.disconnect()
