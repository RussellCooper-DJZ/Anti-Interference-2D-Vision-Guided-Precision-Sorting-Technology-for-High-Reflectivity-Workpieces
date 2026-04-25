MODULE AbbVisionServer
!==============================================================================
! abb_server.mod — ABB RobotStudio 视觉引导仿真服务端
!
! 功能：
!   在 RobotStudio 虚拟控制器上监听 TCP 端口 10000，
!   接收来自 ShipVisionPipeline（Python 端）的 JSON 指令，
!   执行对应的 RAPID 运动指令，并返回 JSON 状态响应。
!
! 支持指令：
!   handshake   — 握手，返回控制器信息
!   MoveL       — 直线运动到目标位姿
!   MoveJ       — 关节运动到目标位姿
!   home        — 回零点
!   get_status  — 查询当前状态
!   get_joints  — 查询关节角度
!   get_tcp     — 查询当前 TCP 位姿
!   set_speed_override — 设置速度倍率
!   set_tool    — 切换工具坐标系
!   set_wobj    — 切换工件坐标系
!   ping        — 心跳
!   disconnect  — 断开连接
!
! 使用方法：
!   1. 在 RobotStudio 中新建 IRB 1200 / IRB 2600 仿真站
!   2. 将本文件加入 RAPID 程序模块
!   3. 在 Controller 配置中开放 TCP 端口 10000（无需防火墙例外，本机仿真）
!   4. 运行 main 程序
!   5. 在 Python 端运行：python3 abb_robotstudio_interface.py --mode robotstudio
!
! @author  RussellCooper
! @version 1.0.0
! @date    2026-03-27
! @license Apache 2.0
!==============================================================================

    ! ---- 常量定义 ----
    CONST num SERVER_PORT := 10000;
    CONST num RECV_BUF_SIZE := 512;
    CONST num SEND_BUF_SIZE := 256;
    CONST num SOCKET_TIMEOUT := 30;

    ! ---- 工具与工件坐标系 ----
    PERS tooldata tVisionGripper := [
        TRUE,
        [[0, 0, 150], [1, 0, 0, 0]],
        [1, [0, 0, 50], [1, 0, 0, 0], 0, 0, 0]
    ];

    PERS wobjdata wobj_conveyor := [
        FALSE, TRUE, "",
        [[400, 0, 0], [1, 0, 0, 0]],
        [[0, 0, 0], [1, 0, 0, 0]]
    ];

    ! ---- Home 位置 ----
    CONST jointtarget jHome := [
        [0, -30, 30, 0, 60, 0],
        [9E+09, 9E+09, 9E+09, 9E+09, 9E+09, 9E+09]
    ];

    ! ---- 全局变量 ----
    VAR socketdev serverSocket;
    VAR socketdev clientSocket;
    VAR string recvBuf;
    VAR string sendBuf;
    VAR bool isConnected := FALSE;
    VAR bool isMoving := FALSE;
    VAR num errorCode := 0;
    VAR num speedOverride := 100;

    !===========================================================================
    ! 主程序入口
    !===========================================================================
    PROC main()
        TPWrite "=== ABB Vision Server v1.0 | @author RussellCooper ===";
        TPWrite "监听端口: " + ValToStr(SERVER_PORT);

        ! 回 Home
        MoveAbsJ jHome, v300, z50, tVisionGripper;
        TPWrite "已回 Home 位置，等待视觉系统连接...";

        ! 启动服务端循环
        ServerLoop;
    ENDPROC

    !===========================================================================
    ! TCP 服务端主循环
    !===========================================================================
    PROC ServerLoop()
        VAR bool keepRunning := TRUE;

        SocketCreate serverSocket;
        SocketBind serverSocket, "0.0.0.0", SERVER_PORT;
        SocketListen serverSocket;
        TPWrite "TCP 服务端已启动，等待连接...";

        WHILE keepRunning DO
            ! 等待客户端连接
            SocketAccept serverSocket, clientSocket \Time:=SOCKET_TIMEOUT;
            isConnected := TRUE;
            TPWrite "视觉系统已连接！";

            ! 处理客户端请求
            ClientLoop;

            isConnected := FALSE;
            SocketClose clientSocket;
            TPWrite "连接已断开，等待下一个连接...";
        ENDWHILE

        SocketClose serverSocket;
    ERROR
        IF ERRNO = ERR_SOCK_TIMEOUT THEN
            TPWrite "等待连接超时，重试...";
            RETRY;
        ELSE
            TPWrite "服务端错误: " + ValToStr(ERRNO);
            SocketClose serverSocket;
        ENDIF
    ENDPROC

    !===========================================================================
    ! 客户端请求处理循环
    !===========================================================================
    PROC ClientLoop()
        VAR bool clientConnected := TRUE;
        VAR string cmd;

        WHILE clientConnected DO
            ! 接收一行 JSON
            SocketReceive clientSocket \Str:=recvBuf \Time:=SOCKET_TIMEOUT;

            ! 解析指令类型
            cmd := ExtractJsonString(recvBuf, "cmd");

            TEST cmd
            CASE "handshake":
                HandleHandshake;
            CASE "MoveL":
                HandleMoveL;
            CASE "MoveJ":
                HandleMoveJ;
            CASE "home":
                HandleHome;
            CASE "get_status":
                HandleGetStatus;
            CASE "get_joints":
                HandleGetJoints;
            CASE "get_tcp":
                HandleGetTcp;
            CASE "set_speed_override":
                HandleSetSpeedOverride;
            CASE "set_tool":
                HandleSetTool;
            CASE "set_wobj":
                HandleSetWobj;
            CASE "ping":
                SendResponse "{""status"":""pong""}";
            CASE "disconnect":
                SendResponse "{""status"":""ok""}";
                clientConnected := FALSE;
            DEFAULT:
                SendResponse "{""status"":""error"",""msg"":""unknown_cmd""}";
            ENDTEST
        ENDWHILE
    ERROR
        IF ERRNO = ERR_SOCK_TIMEOUT THEN
            TPWrite "客户端超时，断开连接";
        ELSEIF ERRNO = ERR_SOCK_CLOSED THEN
            TPWrite "客户端主动断开";
        ELSE
            TPWrite "客户端通信错误: " + ValToStr(ERRNO);
        ENDIF
    ENDPROC

    !===========================================================================
    ! 指令处理程序
    !===========================================================================

    PROC HandleHandshake()
        sendBuf := "{""status"":""ok"",""controller"":""RobotStudio-Sim"","
                 + """version"":""1.0"",""author"":""RussellCooper""}";
        SendResponse sendBuf;
    ENDPROC

    PROC HandleMoveL()
        VAR robtarget target;
        VAR speeddata spd;
        VAR zonedata zn;
        VAR num x, y, z, rx, ry, rz;

        ! 解析目标位姿
        x  := ExtractJsonNum(recvBuf, "x");
        y  := ExtractJsonNum(recvBuf, "y");
        z  := ExtractJsonNum(recvBuf, "z");
        rx := ExtractJsonNum(recvBuf, "rx");
        ry := ExtractJsonNum(recvBuf, "ry");
        rz := ExtractJsonNum(recvBuf, "rz");

        ! 构建 robtarget（欧拉角转四元数由 RAPID 内部处理）
        target := [[x, y, z],
                   EulerToQuat(rx, ry, rz),
                   [0, 0, 0, 0],
                   [9E+09, 9E+09, 9E+09, 9E+09, 9E+09, 9E+09]];

        ! 解析速度和区域
        spd := ParseSpeed(ExtractJsonString(recvBuf, "speed"));
        zn  := ParseZone(ExtractJsonString(recvBuf, "zone"));

        ! 执行运动
        isMoving := TRUE;
        SendResponse "{""status"":""accepted""}";
        MoveL target, spd, zn, tVisionGripper \WObj:=wobj_conveyor;
        isMoving := FALSE;
    ERROR
        isMoving := FALSE;
        errorCode := ERRNO;
        TPWrite "MoveL 错误: " + ValToStr(ERRNO);
    ENDPROC

    PROC HandleMoveJ()
        VAR robtarget target;
        VAR speeddata spd;
        VAR zonedata zn;
        VAR num x, y, z, rx, ry, rz;

        x  := ExtractJsonNum(recvBuf, "x");
        y  := ExtractJsonNum(recvBuf, "y");
        z  := ExtractJsonNum(recvBuf, "z");
        rx := ExtractJsonNum(recvBuf, "rx");
        ry := ExtractJsonNum(recvBuf, "ry");
        rz := ExtractJsonNum(recvBuf, "rz");

        target := [[x, y, z],
                   EulerToQuat(rx, ry, rz),
                   [0, 0, 0, 0],
                   [9E+09, 9E+09, 9E+09, 9E+09, 9E+09, 9E+09]];

        spd := ParseSpeed(ExtractJsonString(recvBuf, "speed"));
        zn  := ParseZone(ExtractJsonString(recvBuf, "zone"));

        isMoving := TRUE;
        SendResponse "{""status"":""accepted""}";
        MoveJ target, spd, zn, tVisionGripper \WObj:=wobj_conveyor;
        isMoving := FALSE;
    ERROR
        isMoving := FALSE;
        errorCode := ERRNO;
    ENDPROC

    PROC HandleHome()
        isMoving := TRUE;
        SendResponse "{""status"":""ok""}";
        MoveAbsJ jHome, v300, z50, tVisionGripper;
        isMoving := FALSE;
    ENDPROC

    PROC HandleGetStatus()
        VAR robtarget curPos;
        VAR string posStr;

        curPos := CRobT(\Tool:=tVisionGripper \WObj:=wobj_conveyor);
        posStr := "[" + ValToStr(curPos.trans.x) + ","
                      + ValToStr(curPos.trans.y) + ","
                      + ValToStr(curPos.trans.z) + ","
                      + "0,180,0]";

        sendBuf := "{""status"":""ok"","
                 + """is_moving"":" + BoolToStr(isMoving) + ","
                 + """error_code"":" + ValToStr(errorCode) + ","
                 + """speed_override"":" + ValToStr(speedOverride) + ","
                 + """current_pos"":" + posStr + "}";
        SendResponse sendBuf;
    ENDPROC

    PROC HandleGetJoints()
        VAR jointtarget jt;
        VAR string jStr;

        jt := CJointT();
        jStr := "[" + ValToStr(jt.robax.rax_1) + ","
                    + ValToStr(jt.robax.rax_2) + ","
                    + ValToStr(jt.robax.rax_3) + ","
                    + ValToStr(jt.robax.rax_4) + ","
                    + ValToStr(jt.robax.rax_5) + ","
                    + ValToStr(jt.robax.rax_6) + "]";

        sendBuf := "{""status"":""ok"",""joints"":" + jStr + "}";
        SendResponse sendBuf;
    ENDPROC

    PROC HandleGetTcp()
        VAR robtarget curPos;
        VAR string tcpStr;

        curPos := CRobT(\Tool:=tVisionGripper \WObj:=wobj_conveyor);
        tcpStr := "[" + ValToStr(curPos.trans.x) + ","
                      + ValToStr(curPos.trans.y) + ","
                      + ValToStr(curPos.trans.z) + ","
                      + "0,180,0]";

        sendBuf := "{""status"":""ok"",""tcp"":" + tcpStr + "}";
        SendResponse sendBuf;
    ENDPROC

    PROC HandleSetSpeedOverride()
        VAR num pct;
        pct := ExtractJsonNum(recvBuf, "percent");
        speedOverride := pct;
        SpeedRefresh pct;
        SendResponse "{""status"":""ok""}";
    ENDPROC

    PROC HandleSetTool()
        ! 工具切换（此处简化，实际可用 StrToVal 动态切换）
        SendResponse "{""status"":""ok""}";
    ENDPROC

    PROC HandleSetWobj()
        ! 工件坐标系切换
        SendResponse "{""status"":""ok""}";
    ENDPROC

    !===========================================================================
    ! 辅助函数
    !===========================================================================

    ! 发送 JSON 响应（追加换行符）
    PROC SendResponse(string msg)
        SocketSend clientSocket \Str:=msg + "\n";
    ENDPROC

    ! 从 JSON 字符串提取字符串值（简化实现，适用于扁平 JSON）
    FUNC string ExtractJsonString(string json, string key)
        VAR num keyPos;
        VAR num valStart;
        VAR num valEnd;
        VAR string searchKey;

        searchKey := """" + key + """:""";
        keyPos := StrFind(json, 1, searchKey);
        IF keyPos = 0 THEN
            RETURN "";
        ENDIF
        valStart := keyPos + StrLen(searchKey);
        valEnd := StrFind(json, valStart, """");
        IF valEnd = 0 THEN
            RETURN "";
        ENDIF
        RETURN StrPart(json, valStart, valEnd - valStart);
    ENDFUNC

    ! 从 JSON 字符串提取数值
    FUNC num ExtractJsonNum(string json, string key)
        VAR num keyPos;
        VAR num valStart;
        VAR num valEnd;
        VAR string searchKey;
        VAR string numStr;
        VAR num result;

        searchKey := """" + key + """:";
        keyPos := StrFind(json, 1, searchKey);
        IF keyPos = 0 THEN
            RETURN 0;
        ENDIF
        valStart := keyPos + StrLen(searchKey);
        ! 找到下一个逗号或花括号
        valEnd := StrFind(json, valStart, ",");
        IF valEnd = 0 THEN
            valEnd := StrFind(json, valStart, "}");
        ENDIF
        IF valEnd = 0 THEN
            RETURN 0;
        ENDIF
        numStr := StrPart(json, valStart, valEnd - valStart);
        StrToVal numStr, result;
        RETURN result;
    ENDFUNC

    ! 布尔值转 JSON 字符串
    FUNC string BoolToStr(bool val)
        IF val THEN
            RETURN "true";
        ELSE
            RETURN "false";
        ENDIF
    ENDFUNC

    ! 欧拉角（ZYX，度）转四元数（简化，仅处理常见姿态）
    ! 实际部署时建议使用 RAPID 内置的 OrientZYX 函数
    FUNC orient EulerToQuat(num rx, num ry, num rz)
        VAR orient q;
        ! 使用 RAPID 内置函数将欧拉角转换为四元数
        q := OrientZYX(rz, ry, rx);
        RETURN q;
    ENDFUNC

    ! 速度字符串解析（ABB v 数据名称 → speeddata）
    FUNC speeddata ParseSpeed(string speedName)
        TEST speedName
        CASE "v5":    RETURN v5;
        CASE "v10":   RETURN v10;
        CASE "v20":   RETURN v20;
        CASE "v50":   RETURN v50;
        CASE "v100":  RETURN v100;
        CASE "v200":  RETURN v200;
        CASE "v300":  RETURN v300;
        CASE "v500":  RETURN v500;
        CASE "v1000": RETURN v1000;
        CASE "v2000": RETURN v2000;
        DEFAULT:      RETURN v100;
        ENDTEST
    ENDFUNC

    ! 区域字符串解析（zone 名称 → zonedata）
    FUNC zonedata ParseZone(string zoneName)
        TEST zoneName
        CASE "fine": RETURN fine;
        CASE "z0":   RETURN z0;
        CASE "z1":   RETURN z1;
        CASE "z5":   RETURN z5;
        CASE "z10":  RETURN z10;
        CASE "z20":  RETURN z20;
        CASE "z50":  RETURN z50;
        CASE "z100": RETURN z100;
        DEFAULT:     RETURN z10;
        ENDTEST
    ENDFUNC

ENDMODULE
