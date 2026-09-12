#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMC 系列运动控制卡 Python SDK
基于 ctypes 封装 MCDLL_NET.dll，提供 Python 友好的接口

硬件: NMC1400 (4轴网络总线运动控制卡，16 DI / 16 DO)
DLL:  MCDLL_NET.dll
手册: NMCxxxx_编程手册.pdf / NMC1400_硬件手册.pdf

核心功能: 电机驱动（点位运动、JOG、回零、轴参数配置）

【重要：底层约定，改动前务必阅读】
1. MCDLL_NET.dll 是 **32 位 (x86) + __stdcall** 的库，导出名无 @N 修饰。
   因此必须用 ctypes.WinDLL（stdcall）加载，不能用 ctypes.CDLL（cdecl），
   否则每次调用后栈指针会被多弹出一次，导致栈错乱/崩溃。
   同时程序必须用 32 位 Python 运行（见 build_32.bat / main.spec）。
2. 本模块中各函数的参数类型均已按 DLL 反汇编结果核对（与编程手册一致）：
   - MCF_Uniaxial_Net       的距离参数是 **int32**（不是 double！）
   - MCF_Set_Axis_Profile_Net / MCF_JOG_Net / MCF_Search_Home_Set_Net 的速度、
     加速度、加加速度是 **double**；
   - MCF_Get_Input_Net      输出两个 uint32（DI00-DI31 / DI32-DI47），不是 uint64；
   - MCF_Set_Output_Net / MCF_Get_Output_Net 是 **uint32**，不是 uint64；
   - MCF_Get_Link_State_Net 带 1 个站号参数。
3. 返回值：0=成功；负数=错误码（见 ERRCODE_DESC）；正数=轴状态/缓冲区状态
   （见 AXIS_STATE_DESC，仅 MCF_JOG_Net / MCF_Uniaxial_Net 等运动函数会返回）。
"""

import ctypes
import os
import sys
from typing import Optional, Tuple, List

from core.log_manager import log_info, log_error, log_warning

# stdcall 加载器：Windows 下用 WinDLL，其它平台退化为 CDLL
_WinDLL = ctypes.WinDLL if hasattr(ctypes, "WinDLL") else ctypes.CDLL


# ============================================================================
# 异常定义
# ============================================================================
class NMCError(Exception):
    """NMC SDK 异常基类"""
    pass


class NMCConnectionError(NMCError):
    """连接相关异常"""
    pass


class NMCParamError(NMCError):
    """参数错误异常"""
    pass


class NMCRuntimeError(NMCError):
    """运行时错误异常"""
    pass


# ============================================================================
# 返回值定义
# ============================================================================
SUCCESS = 0

# 轴状态 / 缓冲区状态 (正数，取自编程手册"函数返回值"表)
SYS_STATE_IDLE_AXIS               = 0    # 轴空闲（MCF_Get_Axis_State_Net 返回 0）
AXIS_BUSY                         = 1    # 正在执行
AXIS_STOP_AT_EMG_IMD              = 2    # EMG 立即紧急停止
AXIS_STOP_AT_EMG_DEC              = 3    # EMG 减速紧急停止
AXIS_STOP_AT_ALM_IMD              = 4    # ALM 立即停止
AXIS_STOP_AT_ALM_DEC              = 5    # ALM 减速停止
AXIS_STOP_AT_SERVO_IMD            = 6    # 伺服使能立即停止
AXIS_STOP_AT_SERVO_DEC            = 7    # 伺服使能减速停止
AXIS_STOP_AT_POS_ERROR_IMD        = 8    # 指令编码器误差立即停止
AXIS_STOP_AT_POS_ERROR_DEC        = 9    # 指令编码器误差减速停止
AXIS_STOP_AT_INDEX_IMD            = 10   # Index 立即停止
AXIS_STOP_AT_INDEX_DEC            = 11   # Index 减速停止
AXIS_STOP_AT_HOME_IMD             = 12   # 原点立即停止
AXIS_STOP_AT_HOME_DEC             = 13   # 原点减速停止
AXIS_STOP_AT_ELP_IMD              = 14   # 正硬限位立即停止
AXIS_STOP_AT_ELP_DEC              = 15   # 正硬限位减速停止
AXIS_STOP_AT_ELN_IMD              = 16   # 负硬限位立即停止
AXIS_STOP_AT_ELN_DEC              = 17   # 负硬限位减速停止
AXIS_STOP_AT_SOFT_ELP_IMD         = 18   # 正软限位立即停止
AXIS_STOP_AT_SOFT_ELP_DEC         = 19   # 正软限位减速停止
AXIS_STOP_AT_SOFT_ELN_IMD         = 20   # 负软限位立即停止
AXIS_STOP_AT_SOFT_ELN_DEC         = 21   # 负软限位减速停止
AXIS_STOP_AT_CMD_IMD              = 22   # 命令立即停止
AXIS_STOP_AT_CMD_DEC              = 23   # 命令减速停止
AXIS_STOP_AT_OTHER_IMD            = 24   # 其它原因立即停止
AXIS_STOP_AT_LINK_IMD             = 25   # 网络通讯中断立即停止
AXIS_STOP_AT_UNKNOWN_IMD          = 26   # 未知原因立即停止
AXIS_STOP_AT_UNKNOWN_DEC          = 27   # 未知原因减速停止
AXIS_STOP_AT_IO_DEC               = 28   # 外部 IO 减速停止
BUFFER_EXECUTING                  = 29   # 缓冲区正在执行
BUFFER_STOPPED                    = 30   # 缓冲区停止
HOME_ERROR                        = 31   # 回零错误
HOME_IN_PROGRESS                  = 32   # 正在回零点

AXIS_STATE_DESC = {
    SYS_STATE_IDLE_AXIS: "空闲",
    AXIS_BUSY: "正在执行",
    AXIS_STOP_AT_EMG_IMD: "EMG立即紧急停止",
    AXIS_STOP_AT_EMG_DEC: "EMG减速紧急停止",
    AXIS_STOP_AT_ALM_IMD: "ALM立即停止",
    AXIS_STOP_AT_ALM_DEC: "ALM减速停止",
    AXIS_STOP_AT_SERVO_IMD: "伺服使能立即停止",
    AXIS_STOP_AT_SERVO_DEC: "伺服使能减速停止",
    AXIS_STOP_AT_POS_ERROR_IMD: "指令编码器误差立即停止",
    AXIS_STOP_AT_POS_ERROR_DEC: "指令编码器误差减速停止",
    AXIS_STOP_AT_INDEX_IMD: "Index立即停止",
    AXIS_STOP_AT_INDEX_DEC: "Index减速停止",
    AXIS_STOP_AT_HOME_IMD: "原点立即停止",
    AXIS_STOP_AT_HOME_DEC: "原点减速停止",
    AXIS_STOP_AT_ELP_IMD: "正硬限位立即停止",
    AXIS_STOP_AT_ELP_DEC: "正硬限位减速停止",
    AXIS_STOP_AT_ELN_IMD: "负硬限位立即停止",
    AXIS_STOP_AT_ELN_DEC: "负硬限位减速停止",
    AXIS_STOP_AT_SOFT_ELP_IMD: "正软限位立即停止",
    AXIS_STOP_AT_SOFT_ELP_DEC: "正软限位减速停止",
    AXIS_STOP_AT_SOFT_ELN_IMD: "负软限位立即停止",
    AXIS_STOP_AT_SOFT_ELN_DEC: "负软限位减速停止",
    AXIS_STOP_AT_CMD_IMD: "命令立即停止",
    AXIS_STOP_AT_CMD_DEC: "命令减速停止",
    AXIS_STOP_AT_OTHER_IMD: "其它原因立即停止",
    AXIS_STOP_AT_LINK_IMD: "网络通讯中断立即停止",
    AXIS_STOP_AT_UNKNOWN_IMD: "未知原因立即停止",
    AXIS_STOP_AT_UNKNOWN_DEC: "未知原因减速停止",
    AXIS_STOP_AT_IO_DEC: "外部IO减速停止",
    BUFFER_EXECUTING: "缓冲区正在执行",
    BUFFER_STOPPED: "缓冲区停止",
    HOME_ERROR: "回零错误",
    HOME_IN_PROGRESS: "正在回零点",
}


# 错误码 (负数)
# 缓冲区参数错误
ERR_BUFFER_NO_END_BUFFER         = -1   #缓冲区未结束
ERR_BUFFER_NO_PROFILE            = -2   #缓冲区无配置
ERR_BUFFER_NO_START              = -3   #缓冲区无开始标志
ERR_BUFFER_INTER_NUMBER          = -4   #缓冲区中断数量错误
ERR_BUFFER_SPACE_ENOUGH          = -5   #缓冲区空间不足
ERR_BUFFER_NUMBER                = -6   #缓冲区数量错误
ERR_BUFFER_MALLOC_FAIL           = -7   #缓冲区分配失败
#输入输出参数错误
ERR_INPUT_NUMBER                 = -8   #输入参数错误
ERR_OUTPUT_NUMBER                = -9   #输出参数错误
#坐标系参数错误
ERR_COORDINATE_NUMBER            = -10  #坐标系参数错误 
#指令参数错误
ERR_ARC_RADIUS                   = -11  #圆弧半径错误
ERR_PROFILE_CALCULATE            = -12  #配置计算错误
ERR_SET_PROFILE_ERR              = -13  #配置设置错误
ERR_NOT_SET_PROFILE              = -14  #未设置配置
ERR_AXIS_INTER_NUMBER            = -15  #轴中断数量错误
ERR_AXIS_NUMBER                  = -16  #轴数量错误
#控制卡系统错误
ERR_LINK_BREAK                   = -17  #链接断开
ERR_OPEN_STATION_FAIL            = -18  #打开站点失败
ERR_BEYOND_STATION_NUMBER        = -19  #超出站点数量
ERR_BEYOND_STATION_LENGTH        = -20  #超出站点长度
ERR_BEYOND_STATION_TYPE          = -21  #超出站点类型
ERR_CAPTURE_EMPTY                = -22  # 捕获为空
 
# 错误码 -> 中文描述映射
ERRCODE_DESC = {
    ERR_BUFFER_NO_END_BUFFER:    "缓冲区未结束",
    ERR_BUFFER_NO_PROFILE:       "缓冲区无配置",
    ERR_BUFFER_NO_START:         "缓冲区无开始标志",
    ERR_BUFFER_INTER_NUMBER:     "缓冲区中断数量错误",
    ERR_BUFFER_SPACE_ENOUGH:     "缓冲区空间不足",
    ERR_BUFFER_NUMBER:           "缓冲区数量错误",
    ERR_BUFFER_MALLOC_FAIL:      "缓冲区分配失败",
    ERR_INPUT_NUMBER:            "输入参数错误",
    ERR_OUTPUT_NUMBER:           "输出参数错误",
    ERR_COORDINATE_NUMBER:       "坐标系参数错误",
    ERR_ARC_RADIUS:              "圆弧半径错误",
    ERR_PROFILE_CALCULATE:       "配置计算错误",
    ERR_SET_PROFILE_ERR:         "配置设置错误",
    ERR_NOT_SET_PROFILE:         "未设置配置",
    ERR_AXIS_INTER_NUMBER:       "轴中断数量错误",
    ERR_AXIS_NUMBER:             "轴数量错误",
    ERR_LINK_BREAK:              "链接断开",
    ERR_OPEN_STATION_FAIL:       "打开站点失败",
    ERR_BEYOND_STATION_NUMBER:   "超出站点数量",
    ERR_BEYOND_STATION_LENGTH:   "超出站点长度",
    ERR_BEYOND_STATION_TYPE:     "超出站点类型",
    ERR_CAPTURE_EMPTY:           "捕获为空",
}

# 控制器整体状态（本 SDK 自定义，由 4 个轴状态归纳而来）
#   SYS_STATE_IDLE    所有轴都空闲（轴状态 0）
#   SYS_STATE_MOVING  有轴正在执行（轴状态 1）
#   SYS_STATE_STOPPED 有轴因异常/限位/急停等原因停止（轴状态 2~32）
SYS_STATE_IDLE             = 0
SYS_STATE_MOVING           = 1
SYS_STATE_STOPPED          = 2

SYS_STATE_DESC = {
    SYS_STATE_IDLE: "空闲",
    SYS_STATE_MOVING: "运动中",
    SYS_STATE_STOPPED: "停止",
}


# ============================================================================
# 宏常量定义
# ============================================================================

# 轴编号 (0-23)
Axis_1 = 0
Axis_2 = 1
Axis_3 = 2
Axis_4 = 3
Axis_5 = 4
Axis_6 = 5
Axis_7 = 6
Axis_8 = 7
Axis_9 = 8
Axis_10 = 9
Axis_11 = 10
Axis_12 = 11
Axis_13 = 12
Axis_14 = 13
Axis_15 = 14
Axis_16 = 15
Axis_17 = 16
Axis_18 = 17
Axis_19 = 18
Axis_20 = 19
Axis_21 = 20
Axis_22 = 21
Axis_23 = 22
Axis_24 = 23

# 位置模式
Position_Absolute = 0
Position_Opposite = 1

# 运动曲线
Profile_T = 0
Profile_S = 1

# 脉冲模式
Pulse_Dir_H = 0
Pulse_Dir_L = 1
Pulse_CW_CCW = 2
Pulse_CCW_CW = 3
Pulse_AB = 4
Pulse_BA = 5

# 伺服使能
Servo_Close = 0
Servo_Open = 1

# 停止模式
Axis_Stop_IMD = 0
Axis_Stop_DEC = 1
Stop_Abrupt = Axis_Stop_IMD    # 急停(立即停止)
Stop_Smooth = Axis_Stop_DEC    # 平滑停止(减速停止)

# 回零模式（详见编程手册 第6章 回原点模式选择参考表）
Home_Mode_1 = 1                # 模式1: 负限位开关 + Z相
Home_Mode_3 = 3                # 模式3: 安装原点开关，正方向，原点负边外侧，负方向偏移
Home_Mode_17 = 17              # 模式17: 安装负限位开关，负方向找负限位正边
Home_Mode_18 = 18              # 模式18: 安装正限位开关，正方向找正极限负边
Home_Mode_19 = 19              # 模式19: 安装原点开关，正方向，原点负边外侧，负方向偏移
Home_Mode_21 = 21              # 模式21: 安装原点开关，负方向，原点正边外侧，正方向偏移
Home_Mode_33 = 33              # 模式33: 负方向 Index
Home_Mode_34 = 34              # 模式34: 正方向 Index

# 回零状态 (MCF_Search_Home_Get_State_Net 的输出)
HOME_STATE_SUCCESS = 0         # 回零成功
HOME_STATE_ERROR = 31          # 回零错误
HOME_STATE_HOMING = 32         # 正在回零点

# 级联模式
Switch_State_Series = 0
Switch_State_Parallel = 1

# 站类型（MCF_Open_Net 的 Station_Type 参数，详见编程手册 1.2）
Station_Type_24I16O = 0
Station_Type_48I32O = 1
Station_Type_4D = 2     # NMC1200R/NMC1400/NMC3400/NMC3401/NMC3201 (2/4轴)
Station_Type_8D = 3
Station_Type_24D = 4
Station_Type_4DM = 5

STATION_TYPE_NAMES = {
    0: "NIO0808R/NIO1616R/NIO2416 (8/16/24 输入, 8/16/16 输出)",
    1: "NIO4832/NIO3232/NIO4000 (48/32/40 输入, 32/32/0 输出)",
    2: "NMC1200R/NMC1400/NMC3400/NMC3401/NMC3201 (2/4 轴运动控制卡)",
    3: "NMC5800/NMC5600R/NMC1800/NMC1600R (6/8 轴运动控制卡)",
    4: "NMC5120R/NMC5160 (12/16 轴运动控制卡)",
    5: "LMC3400/LMC3100 (网络运动控制激光卡)",
    6: "EIO0840 (8 路编码器 24 输入 16 输出模块)",
    7: "NAD0804/NAD0402/NAD0808/NAD0400/NAD0004/NAD0002 (模拟量模块)",
    8: "NIO0040 (40 路输出模块)",
}

# 本项目所用控制卡：NMC1400 —— 4 轴网络总线运动控制卡，16 路 DI / 16 路 DO
NMC1400_STATION_TYPE = Station_Type_4D
NMC1400_AXIS_COUNT = 4
NMC1400_INPUT_COUNT = 16
NMC1400_OUTPUT_COUNT = 16


# ============================================================================
# 工具函数
# ============================================================================
def get_error_message(ret: int) -> str:
    """获取返回值对应的错误描述"""
    if ret == 0:
        return "成功"
    if ret > 0:
        return AXIS_STATE_DESC.get(ret, f"未知状态码: {ret}")
    return ERRCODE_DESC.get(ret, f"未知错误码: {ret}")


def check_result(ret: int, func_name: str = "") -> int:
    """检查函数返回值，非0则抛出异常"""
    if ret != 0:
        msg = get_error_message(ret)
        if func_name:
            msg = f"[{func_name}] {msg}"
        if ret < 0:
            raise NMCRuntimeError(msg)
    return ret


# ============================================================================
# NMC SDK 主类
# ============================================================================
class NMCSDK:
    """
    NMC 运动控制卡 SDK 封装类

    通过 ctypes 加载 MCDLL_NET.dll，封装核心电机驱动 API 函数。

    核心功能:
        - 控制卡连接/断开
        - 轴参数配置 (脉冲模式、位置、编码器、速度)
        - 伺服使能/报警
        - 软件限位
        - 回零
        - JOG 点动
        - 单轴点位运动
        - 轴状态监测

    用法:
        sdk = NMCSDK()
        sdk.load_dll()
        sdk.set_switch_state(Switch_State_Series)
        station_num, station_types = sdk.open_net(0)
        # ... 执行运动控制 ...
        sdk.close_net()
    """

    DLL_FILENAME = "MCDLL_NET.dll"

    def __init__(self, dll_path: Optional[str] = None):
        self._dll: Optional[ctypes.CDLL] = None
        self._dll_path = dll_path or self.DLL_FILENAME
        self._is_open = False
        self._connected = False  # 连接状态（与 _is_open 同步管理）
        self._station_count = 0
        self._station_types: List[int] = []
        self._station_numbers: List[int] = []
        self._last_open_ret = None  # 最近一次 MCF_Open_Net 的返回值（便于诊断）

    # ---- DLL 加载 ----

    def load_dll(self) -> None:
        """加载 DLL 文件"""
        if self._dll is not None:
            return

        # 使用绝对路径确保 DLL 加载成功
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()

        search_paths = [
            self._dll_path,
            os.path.join(script_dir, self._dll_path),
            os.path.join(cwd, self._dll_path),
        ]

        found_path = None
        for path in search_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                found_path = abs_path
                break

        if found_path is None:
            raise NMCConnectionError(
                f"找不到 DLL 文件: {self.DLL_FILENAME}。"
                f"请确保 DLL 文件与程序在同一目录下。"
            )

        try:
            self._dll_path = found_path
            # 关键：MCDLL_NET.dll 是 __stdcall（32 位），必须用 WinDLL 加载。
            # 用 CDLL(cdecl) 调用会让栈指针每次调用后错位，表现为随机崩溃。
            self._dll = _WinDLL(self._dll_path)
        except OSError as e:
            raise NMCConnectionError(
                f"加载 DLL 失败: {e}\n"
                f"提示：MCDLL_NET.dll 为 32 位库，请使用 32 位 Python 运行/打包。"
            )

        self._setup_function_prototypes()

    def is_loaded(self) -> bool:
        return self._dll is not None

    def is_open(self) -> bool:
        return self._is_open

    def get_station_info(self) -> Tuple[int, List[int], List[int]]:
        return self._station_count, self._station_numbers, self._station_types

    # ---- 函数原型设置 ----

    def _setup_function_prototypes(self) -> None:
        """设置所有 DLL 函数的参数类型和返回类型"""
        dll = self._dll
        RET = ctypes.c_int16

        # ========== 初始化 ==========
        dll.MCF_Set_Switch_State_Net.argtypes = [ctypes.c_uint16]
        dll.MCF_Set_Switch_State_Net.restype = RET

        dll.MCF_Open_Net.argtypes = [
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        dll.MCF_Open_Net.restype = RET

        dll.MCF_Get_Open_Net.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        dll.MCF_Get_Open_Net.restype = RET

        dll.MCF_Close_Net.argtypes = []
        dll.MCF_Close_Net.restype = RET

        dll.MCF_Set_Link_TimeOut_Net.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint16,
        ]
        dll.MCF_Set_Link_TimeOut_Net.restype = RET

        dll.MCF_Get_Link_State_Net.argtypes = [ctypes.c_uint16]
        dll.MCF_Get_Link_State_Net.restype = RET

        dll.MCF_Get_Link_TimeOut_Net.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16]
        dll.MCF_Get_Link_TimeOut_Net.restype = RET

        # ========== 专用 I/O (伺服使能/报警/限位/原点) ==========
        dll.MCF_Set_Servo_Enable_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Set_Servo_Enable_Net.restype = RET

        dll.MCF_Get_Servo_Enable_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Servo_Enable_Net.restype = RET

        dll.MCF_Set_Servo_Alarm_Reset_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Set_Servo_Alarm_Reset_Net.restype = RET

        dll.MCF_Get_Servo_Alarm_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Servo_Alarm_Net.restype = RET

        dll.MCF_Get_Servo_INP_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Servo_INP_Net.restype = RET

        dll.MCF_Get_Z_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Z_Net.restype = RET

        dll.MCF_Get_Home_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Home_Net.restype = RET

        dll.MCF_Get_Positive_Limit_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Positive_Limit_Net.restype = RET

        dll.MCF_Get_Negative_Limit_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Negative_Limit_Net.restype = RET

        # ========== 轴参数 ==========
        dll.MCF_Set_Pulse_Mode_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint16]
        dll.MCF_Set_Pulse_Mode_Net.restype = RET

        dll.MCF_Get_Pulse_Mode_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16]
        dll.MCF_Get_Pulse_Mode_Net.restype = RET

        # 位置/编码器均为 32 位有符号整数（编程手册：长度、位置单位 脉冲）
        dll.MCF_Set_Position_Net.argtypes = [ctypes.c_uint16, ctypes.c_int32, ctypes.c_uint16]
        dll.MCF_Set_Position_Net.restype = RET

        dll.MCF_Get_Position_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint16]
        dll.MCF_Get_Position_Net.restype = RET

        dll.MCF_Set_Encoder_Net.argtypes = [ctypes.c_uint16, ctypes.c_int32, ctypes.c_uint16]
        dll.MCF_Set_Encoder_Net.restype = RET

        dll.MCF_Get_Encoder_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint16]
        dll.MCF_Get_Encoder_Net.restype = RET

        dll.MCF_Get_Vel_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_uint16]
        dll.MCF_Get_Vel_Net.restype = RET

        # ========== 运动停止触发 ==========
        dll.MCF_Set_EMG_Bit_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Set_EMG_Bit_Net.restype = RET

        dll.MCF_Set_Soft_Limit_Net.argtypes = [ctypes.c_uint16, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint16]
        dll.MCF_Set_Soft_Limit_Net.restype = RET

        dll.MCF_Get_Soft_Limit_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_uint16]
        dll.MCF_Get_Soft_Limit_Net.restype = RET

        dll.MCF_Set_Soft_Limit_Enable_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Set_Soft_Limit_Enable_Net.restype = RET

        dll.MCF_Get_Soft_Limit_Enable_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Soft_Limit_Enable_Net.restype = RET

        dll.MCF_Clear_Axis_State_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Clear_Axis_State_Net.restype = RET

        dll.MCF_Get_Axis_State_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_int16), ctypes.c_uint16]
        dll.MCF_Get_Axis_State_Net.restype = RET

        # ========== 回零 ==========
        dll.MCF_Search_Home_Set_Net.argtypes = [
            ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16,
            ctypes.c_uint16, ctypes.c_uint16,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_int32, ctypes.c_uint16, ctypes.c_uint16,
        ]
        dll.MCF_Search_Home_Set_Net.restype = RET

        dll.MCF_Search_Home_Start_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Search_Home_Start_Net.restype = RET

        dll.MCF_Search_Home_Stop_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Search_Home_Stop_Net.restype = RET

        dll.MCF_Search_Home_Stop_Time_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Search_Home_Stop_Time_Net.restype = RET

        dll.MCF_Search_Home_Keep_Position_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Search_Home_Keep_Position_Net.restype = RET

        dll.MCF_Search_Home_Keep_Encoder_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Search_Home_Keep_Encoder_Net.restype = RET

        dll.MCF_Search_Home_Get_State_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Search_Home_Get_State_Net.restype = RET

        # ========== 点位运动 ==========
        dll.MCF_JOG_Net.argtypes = [ctypes.c_uint16, ctypes.c_double, ctypes.c_double, ctypes.c_uint16]
        dll.MCF_JOG_Net.restype = RET

        dll.MCF_Set_Axis_Profile_Net.argtypes = [
            ctypes.c_uint16,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_uint16, ctypes.c_uint16,
        ]
        dll.MCF_Set_Axis_Profile_Net.restype = RET

        # MCF_Uniaxial_Net / MCF_Uniaxial_dDist_Change_Net 的 dDist 是 int32。
        # 这一点已通过反汇编 MCDLL_NET.dll 确认（函数内使用 fild dword ptr[ebp+0Ch]，
        # 且 stdcall 弹栈长度为 16 字节 = 4 个 4 字节参数）。
        # 若误传 c_double（8 字节），目标位置会变成随机值，务必不要改回 double。
        dll.MCF_Uniaxial_Net.argtypes = [ctypes.c_uint16, ctypes.c_int32, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Uniaxial_Net.restype = RET

        dll.MCF_Uniaxial_dDist_Change_Net.argtypes = [ctypes.c_uint16, ctypes.c_int32, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Uniaxial_dDist_Change_Net.restype = RET

        dll.MCF_Uniaxial_dMaxV_Change_Net.argtypes = [ctypes.c_uint16, ctypes.c_double, ctypes.c_uint16]
        dll.MCF_Uniaxial_dMaxV_Change_Net.restype = RET

        dll.MCF_Uniaxial_dMaxA_Change_Net.argtypes = [ctypes.c_uint16, ctypes.c_double, ctypes.c_uint16]
        dll.MCF_Uniaxial_dMaxA_Change_Net.restype = RET

        dll.MCF_Get_Axis_Profile_Net.argtypes = [
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_uint16,
        ]
        dll.MCF_Get_Axis_Profile_Net.restype = RET

        dll.MCF_Set_Axis_Stop_Profile_Net.argtypes = [
            ctypes.c_uint16, ctypes.c_double, ctypes.c_double,
            ctypes.c_uint16, ctypes.c_uint16,
        ]
        dll.MCF_Set_Axis_Stop_Profile_Net.restype = RET

        dll.MCF_Get_Axis_Stop_Profile_Net.argtypes = [
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16,
        ]
        dll.MCF_Get_Axis_Stop_Profile_Net.restype = RET

        dll.MCF_Axis_Stop_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Axis_Stop_Net.restype = RET

        # ========== 系统 ==========
        dll.MCF_Get_Version_Net.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16]
        dll.MCF_Get_Version_Net.restype = RET

        dll.MCF_Get_Serial_Number_Net.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint16]
        dll.MCF_Get_Serial_Number_Net.restype = RET

        dll.MCF_Get_Run_Time_Net.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16]
        dll.MCF_Get_Run_Time_Net.restype = RET

        # ========== 数字 I/O (第2章) ==========
        # MCF_Get_Input_Net 输出两个 uint32（DI00-DI31 / DI32-DI47），不是 uint64
        dll.MCF_Get_Input_Net.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16,
        ]
        dll.MCF_Get_Input_Net.restype = RET

        dll.MCF_Get_Input_Bit_Net.argtypes = [
            ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16,
        ]
        dll.MCF_Get_Input_Bit_Net.restype = RET

        # MCF_Set_Output_Net / MCF_Get_Output_Net 的电平字是 uint32，不是 uint64
        dll.MCF_Set_Output_Net.argtypes = [
            ctypes.c_uint32, ctypes.c_uint16,
        ]
        dll.MCF_Set_Output_Net.restype = RET

        dll.MCF_Get_Output_Net.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint16,
        ]
        dll.MCF_Get_Output_Net.restype = RET

        dll.MCF_Set_Output_Bit_Net.argtypes = [
            ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16,
        ]
        dll.MCF_Set_Output_Bit_Net.restype = RET

        dll.MCF_Get_Output_Bit_Net.argtypes = [
            ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16,
        ]
        dll.MCF_Get_Output_Bit_Net.restype = RET

        dll.MCF_Set_Output_Time_Bit_Net.argtypes = [
            ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16,
        ]
        dll.MCF_Set_Output_Time_Bit_Net.restype = RET

        dll.MCF_Get_Switch_State_Net.argtypes = [ctypes.c_uint16]
        dll.MCF_Get_Switch_State_Net.restype = RET

    # ========================================================================
    # 初始化函数
    # ========================================================================

    def set_switch_state(self, mode: int = 0) -> int:
        """设置级联模式: 0=串联, 1=并联"""
        return self._dll.MCF_Set_Switch_State_Net(ctypes.c_uint16(mode))

    def open_net(
        self,
        station_count: int = 1,
        station_numbers: Optional[List[int]] = None,
        station_types: Optional[List[int]] = None,
    ) -> int:
        """打开控制卡。

        对应手册 1.2：
            Connection_Number = 实际连接扩展模块的数量（大于 0）
            Station_Number    = 依次设置的站号数组（可自由分配，不可重复）
            Station_Type      = 对应的站点类型数组（NMC1400 = 2）

        示例:
            open_net(1)                          # 1 个站，站号 0，类型 2 (NMC1400)
            open_net(3, [0,1,2], [2,2,1])        # 3 个站，指定站号和类型

        Args:
            station_count: 扩展模块数量（即 Connection_Number）
            station_numbers: 站编号列表，默认 [0, 1, ..., station_count-1]
            station_types: 站类型列表，默认全部为 Station_Type_4D (2)

        Returns:
            int: 0=成功；负数=错误码（见 ERRCODE_DESC）
        """
        if station_numbers is None:
            station_numbers = list(range(station_count))
        if station_types is None:
            station_types = [NMC1400_STATION_TYPE] * station_count

        # 创建 ctypes 数组（DLL 会回填实际站号/站类型）
        num_array = (ctypes.c_uint16 * station_count)(*station_numbers)
        type_array = (ctypes.c_uint16 * station_count)(*station_types)

        ret = self._dll.MCF_Open_Net(
            ctypes.c_uint16(station_count),
            num_array,
            type_array,
        )

        self._last_open_ret = ret
        if ret == SUCCESS:
            self._is_open = True
            self._connected = True
            self._station_count = station_count
            self._station_numbers = [int(x) for x in num_array]
            self._station_types = [int(x) for x in type_array]
        return ret

    def get_open_net(self) -> Tuple[int, List[int], List[int]]:
        """读取打开参数: (连接号, 站编号列表, 站类型列表)"""
        connection_number = ctypes.c_uint16()
        max_stations = 32
        station_numbers = (ctypes.c_uint16 * max_stations)()
        station_types = (ctypes.c_uint16 * max_stations)()
        actual_count = ctypes.c_uint16(max_stations)

        ret = self._dll.MCF_Get_Open_Net(
            ctypes.byref(actual_count),
            station_numbers,
            station_types,
        )

        if ret < 0:
            raise NMCRuntimeError(f"读取打开参数失败: {get_error_message(ret)}")

        count = int(actual_count.value) or max_stations
        count = min(count, max_stations)
        numbers = [int(station_numbers[i]) for i in range(count)]
        types = [int(station_types[i]) for i in range(count)]
        return int(connection_number.value), numbers, types

    def close_net(self) -> int:
        """关闭控制卡"""
        ret = self._dll.MCF_Close_Net()
        if ret == 0:
            self._is_open = False
            self._connected = False
            self._station_count = 0
            self._station_numbers = []
            self._station_types = []
        return ret

    def set_link_timeout(self, time_1ms: int, timeout_output: int, station: int = 0) -> int:
        """链接超时紧急停止：超时 Time_1MS 毫秒后停止所有轴，并输出 DO TimeOut_Output

        手册 1.3.1：Time_1MS 范围 [0, 60000]
        """
        return self._dll.MCF_Set_Link_TimeOut_Net(
            ctypes.c_uint32(time_1ms),
            ctypes.c_uint32(timeout_output),
            ctypes.c_uint16(station),
        )

    def get_link_state(self, station: int = 0) -> int:
        """控制卡链接监测（手册 1.4）：0=链接正常；负数=错误码（如 -18 未打开站点）"""
        return self._dll.MCF_Get_Link_State_Net(ctypes.c_uint16(station))

    def get_link_timeout_count(self, station: int = 0) -> int:
        """读取历史链接中断次数（手册 1.3.1）。

        次数不为 0 说明控制卡曾经因为"链接超时"触发过紧急停止，
        排查"轴不走"时值得关注。
        """
        value = ctypes.c_uint32()
        ret = self._dll.MCF_Get_Link_TimeOut_Net(ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取链接中断次数失败: {get_error_message(ret)}")
        return value.value

    def get_switch_state(self, station: int = 0) -> int:
        """读取网络级联模式：0=串联, 1=并联"""
        return self._dll.MCF_Get_Switch_State_Net(ctypes.c_uint16(station))


    # ========================================================================
    # 专用 I/O (伺服使能/报警/限位/原点信号)
    # ========================================================================

    def set_servo_enable(self, axis: int, logic: int, station: int = 0) -> int:
        """设置伺服使能: 0=关闭, 1=开启"""
        return self._dll.MCF_Set_Servo_Enable_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(logic), ctypes.c_uint16(station))

    def get_servo_enable(self, axis: int, station: int = 0) -> int:
        """读取伺服使能状态"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Servo_Enable_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取伺服使能失败: {get_error_message(ret)}")
        return value.value

    def set_servo_alarm_reset(self, axis: int, logic: int, station: int = 0) -> int:
        """设置伺服报警复位"""
        return self._dll.MCF_Set_Servo_Alarm_Reset_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(logic), ctypes.c_uint16(station))

    def get_servo_alarm(self, axis: int, station: int = 0) -> int:
        """读取伺服报警状态: 0=无报警, 1=有报警"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Servo_Alarm_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取伺服报警失败: {get_error_message(ret)}")
        return value.value

    def get_servo_inp(self, axis: int, station: int = 0) -> int:
        """读取伺服 INP (定位完成): 0=未到位, 1=到位"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Servo_INP_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取INP失败: {get_error_message(ret)}")
        return value.value

    def get_z(self, axis: int, station: int = 0) -> int:
        """读取 Z 相信号"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Z_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取Z相失败: {get_error_message(ret)}")
        return value.value

    def get_home(self, axis: int, station: int = 0) -> int:
        """读取原点信号"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Home_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取原点失败: {get_error_message(ret)}")
        return value.value

    def get_positive_limit(self, axis: int, station: int = 0) -> int:
        """读取正限位信号"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Positive_Limit_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取正限位失败: {get_error_message(ret)}")
        return value.value

    def get_negative_limit(self, axis: int, station: int = 0) -> int:
        """读取负限位信号"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Negative_Limit_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取负限位失败: {get_error_message(ret)}")
        return value.value

    # ========================================================================
    # 轴参数
    # ========================================================================

    def set_pulse_mode(self, axis: int, pulse_mode: int, station: int = 0) -> int:
        """设置脉冲模式: 0=Pulse/Dir_H, 1=Pulse/Dir_L, 2=CW/CCW, 3=CCW/CW, 4=AB, 5=BA"""
        return self._dll.MCF_Set_Pulse_Mode_Net(
            ctypes.c_uint16(axis), ctypes.c_uint32(pulse_mode), ctypes.c_uint16(station))

    def get_pulse_mode(self, axis: int, station: int = 0) -> int:
        """读取脉冲模式"""
        value = ctypes.c_uint32()
        ret = self._dll.MCF_Get_Pulse_Mode_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取脉冲模式失败: {get_error_message(ret)}")
        return value.value

    def set_position(self, axis: int, position: int, station: int = 0) -> int:
        """设置轴规划位置（单位：脉冲）"""
        return self._dll.MCF_Set_Position_Net(
            ctypes.c_uint16(axis), ctypes.c_int32(int(position)), ctypes.c_uint16(station))

    def get_position(self, axis: int, station: int = 0) -> int:
        """读取轴规划位置（单位：脉冲）"""
        value = ctypes.c_int32()
        ret = self._dll.MCF_Get_Position_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取位置失败: {get_error_message(ret)}")
        return value.value

    def set_encoder(self, axis: int, encoder: int, station: int = 0) -> int:
        """设置轴编码器位置（单位：脉冲）"""
        return self._dll.MCF_Set_Encoder_Net(
            ctypes.c_uint16(axis), ctypes.c_int32(int(encoder)), ctypes.c_uint16(station))

    def get_encoder(self, axis: int, station: int = 0) -> int:
        """读取轴编码器位置（单位：脉冲）"""
        value = ctypes.c_int32()
        ret = self._dll.MCF_Get_Encoder_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取编码器失败: {get_error_message(ret)}")
        return value.value

    def get_velocity(self, axis: int, station: int = 0) -> Tuple[float, float]:
        """读取速度: (指令速度, 编码器速度)"""
        cmd_vel = ctypes.c_double()
        enc_vel = ctypes.c_double()
        ret = self._dll.MCF_Get_Vel_Net(
            ctypes.c_uint16(axis), ctypes.byref(cmd_vel), ctypes.byref(enc_vel), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取速度失败: {get_error_message(ret)}")
        return cmd_vel.value, enc_vel.value

    # ========================================================================
    # 运动停止触发
    # ========================================================================

    def set_emg_bit(self, emg_input: int, emg_mode: int, station: int = 0) -> int:
        """设置急停输入: mode: 0=关闭, 1=低电平立即, 2=低电平减速, 3=高电平立即, 4=高电平减速"""
        return self._dll.MCF_Set_EMG_Bit_Net(
            ctypes.c_uint16(emg_input), ctypes.c_uint16(emg_mode), ctypes.c_uint16(station))

    def set_soft_limit(self, axis: int, positive_pos: int, negative_pos: int, station: int = 0) -> int:
        """设置轴软件限位位置（正限位 / 负限位，单位：脉冲）"""
        return self._dll.MCF_Set_Soft_Limit_Net(
            ctypes.c_uint16(axis), ctypes.c_int32(int(positive_pos)),
            ctypes.c_int32(int(negative_pos)), ctypes.c_uint16(station))

    def get_soft_limit(self, axis: int, station: int = 0) -> tuple:
        """读取轴软件限位位置: (正限位, 负限位)"""
        pos = ctypes.c_int32()
        neg = ctypes.c_int32()
        ret = self._dll.MCF_Get_Soft_Limit_Net(
            ctypes.c_uint16(axis), ctypes.byref(pos), ctypes.byref(neg), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取软件限位失败: {get_error_message(ret)}")
        return pos.value, neg.value

    def set_soft_limit_enable(self, axis: int, enable: int, station: int = 0) -> int:
        """使能/禁用软件限位: 0=禁用, 1=使能"""
        return self._dll.MCF_Set_Soft_Limit_Enable_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(enable), ctypes.c_uint16(station))

    def get_soft_limit_enable(self, axis: int, station: int = 0) -> int:
        """读取软件限位使能状态: 0=禁用, 1=使能"""
        enable = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Soft_Limit_Enable_Net(
            ctypes.c_uint16(axis), ctypes.byref(enable), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取软件限位使能状态失败: {get_error_message(ret)}")
        return enable.value

    def clear_axis_state(self, axis: int, station: int = 0) -> int:
        """清除轴状态"""
        return self._dll.MCF_Clear_Axis_State_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(station))

    def get_axis_state(self, axis: int, station: int = 0) -> int:
        """查询轴状态（MCF_Get_Axis_State_Net）

        返回值：
            0  = 轴空闲（已停止，无异常）
            1  = 正在执行
            2~28 = 各类停止原因（急停/报警/限位/…，见 AXIS_STATE_DESC）
        """
        reason = ctypes.c_int16()
        ret = self._dll.MCF_Get_Axis_State_Net(
            ctypes.c_uint16(axis), ctypes.byref(reason), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取轴状态失败: {get_error_message(ret)}")
        return reason.value

    # ========================================================================
    # 回零
    # ========================================================================

    def search_home_set(self, axis: int, mode: int,
                        limit_logic: int, home_logic: int,
                        index_logic: int, h_dmaxv: float,
                        l_dmaxv: float, offset: int,
                        trigger_source: int, station: int = 0) -> int:
        """设置回零参数（手册 6.1，必须在 search_home_start 之前调用）

        Args:
            axis: 轴号
            mode: 回零方式 1~35，见编程手册"回原点模式选择参考表"
            limit_logic: 正负限位触发电平 0=低电平 1=高电平
            home_logic: 原点开关触发电平 0=低电平 1=高电平
            index_logic: Index(Z相) 触发电平 0=低电平 1=高电平
            h_dmaxv: 高速段速度
            l_dmaxv: 低速段速度
            offset: 偏移位置（按回零模式图示箭头方向偏移）
            trigger_source: 捕捉位置模式 0=指令位置 1=编码器位置
            station: 站号
        """
        return self._dll.MCF_Search_Home_Set_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(mode),
            ctypes.c_uint16(limit_logic), ctypes.c_uint16(home_logic), ctypes.c_uint16(index_logic),
            ctypes.c_double(h_dmaxv), ctypes.c_double(l_dmaxv),
            ctypes.c_int32(int(offset)), ctypes.c_uint16(trigger_source), ctypes.c_uint16(station))

    def search_home_stop_time(self, axis: int, stop_time_ms: int, station: int = 0) -> int:
        """设置回零碰撞原点缓停时间（手册 6.1，必须在 search_home_set 之前调用）"""
        return self._dll.MCF_Search_Home_Stop_Time_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(int(stop_time_ms)), ctypes.c_uint16(station))

    def search_home_keep_position(self, axis: int, station: int = 0) -> int:
        """设置回零完成后保持位置值"""
        return self._dll.MCF_Search_Home_Keep_Position_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(station))

    def search_home_keep_encoder(self, axis: int, station: int = 0) -> int:
        """设置回零完成后保持编码器值"""
        return self._dll.MCF_Search_Home_Keep_Encoder_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(station))

    def search_home_start(self, axis: int, station: int = 0) -> int:
        """开始回零"""
        return self._dll.MCF_Search_Home_Start_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(station))

    def search_home_stop(self, axis: int, station: int = 0) -> int:
        """停止回零"""
        return self._dll.MCF_Search_Home_Stop_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(station))

    def search_home_get_state(self, axis: int, station: int = 0) -> int:
        """获取回零状态: 0=回零成功, 31=回零错误, 32=正在回零点"""
        state = ctypes.c_uint16()
        ret = self._dll.MCF_Search_Home_Get_State_Net(
            ctypes.c_uint16(axis), ctypes.byref(state), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取回零状态失败: {get_error_message(ret)}")
        return state.value

    # ========================================================================
    # 点位运动
    # ========================================================================

    def jog(self, axis: int, dmaxv: float, dmaxa: float, station: int = 0) -> int:
        """JOG 速度控制: 正速度=正转, 负速度=反转; 速度为 0 表示停止速度环控制"""
        return self._dll.MCF_JOG_Net(
            ctypes.c_uint16(axis), ctypes.c_double(dmaxv), ctypes.c_double(dmaxa), ctypes.c_uint16(station))

    def set_axis_profile(self, axis: int, v_ini: float, v_max: float,
                         a_max: float, jerk: float, v_end: float,
                         profile: int, station: int = 0) -> int:
        """设置单轴曲线参数（手册 7.4.3）

        Args:
            v_ini: 启动速度 pulse/s，dMaxV > dV_ini >= 0
            v_max: 目标速度 pulse/s，> 0
            a_max: 加速度 pulse/s²，> 0
            jerk: 加加速度 pulse/s³，> 0
            v_end: 终止速度 pulse/s，dMaxV > dV_end >= 0
            profile: 0=Profile_T(T型曲线) 1=Profile_S(S型曲线)
        """
        return self._dll.MCF_Set_Axis_Profile_Net(
            ctypes.c_uint16(axis),
            ctypes.c_double(v_ini), ctypes.c_double(v_max), ctypes.c_double(a_max),
            ctypes.c_double(jerk), ctypes.c_double(v_end),
            ctypes.c_uint16(profile), ctypes.c_uint16(station))

    def get_axis_profile(self, axis: int, station: int = 0) -> dict:
        """读取单轴曲线参数（手册 7.4.3）"""
        v_ini = ctypes.c_double()
        v_max = ctypes.c_double()
        a_max = ctypes.c_double()
        jerk = ctypes.c_double()
        v_end = ctypes.c_double()
        profile = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Axis_Profile_Net(
            ctypes.c_uint16(axis), ctypes.byref(v_ini), ctypes.byref(v_max),
            ctypes.byref(a_max), ctypes.byref(jerk), ctypes.byref(v_end),
            ctypes.byref(profile), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取单轴曲线参数失败: {get_error_message(ret)}")
        return {
            "v_ini": v_ini.value, "v_max": v_max.value, "a_max": a_max.value,
            "jerk": jerk.value, "v_end": v_end.value, "profile": profile.value,
        }

    def set_axis_stop_profile(self, axis: int, a_max: float, jerk: float,
                              profile: int, station: int = 0) -> int:
        """设置单轴平滑停止曲线（减速停止前调用）"""
        return self._dll.MCF_Set_Axis_Stop_Profile_Net(
            ctypes.c_uint16(axis), ctypes.c_double(a_max), ctypes.c_double(jerk),
            ctypes.c_uint16(profile), ctypes.c_uint16(station))

    def get_axis_stop_profile(self, axis: int, station: int = 0) -> dict:
        """读取单轴平滑停止曲线"""
        a_max = ctypes.c_double()
        jerk = ctypes.c_double()
        profile = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Axis_Stop_Profile_Net(
            ctypes.c_uint16(axis), ctypes.byref(a_max), ctypes.byref(jerk),
            ctypes.byref(profile), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取停止曲线参数失败: {get_error_message(ret)}")
        return {"a_max": a_max.value, "jerk": jerk.value, "profile": profile.value}

    def uniaxial(self, axis: int, dist: int, position_mode: int, station: int = 0) -> int:
        """单轴点位运动（手册 7.5.1）

        Args:
            axis: 轴号
            dist: 目标位置，单位脉冲（**int32**：-2^31 ~ 2^31-1，不是浮点数）
            position_mode: 0=Position_Absolute 绝对位置, 1=Position_Opposite 相对位置
            station: 站号

        注意：需先调用 set_axis_profile 设置曲线参数。
        """
        return self._dll.MCF_Uniaxial_Net(
            ctypes.c_uint16(axis), ctypes.c_int32(int(dist)),
            ctypes.c_uint16(position_mode), ctypes.c_uint16(station))

    def uniaxial_change_dist(self, axis: int, dist: int, position_mode: int, station: int = 0) -> int:
        """运动过程中更改目标位置（仅单轴 T 型曲线运动中有效）"""
        return self._dll.MCF_Uniaxial_dDist_Change_Net(
            ctypes.c_uint16(axis), ctypes.c_int32(int(dist)),
            ctypes.c_uint16(position_mode), ctypes.c_uint16(station))

    def uniaxial_change_speed(self, axis: int, dmaxv: float, station: int = 0) -> int:
        """运动过程中更改目标速度（仅单轴 T 型曲线运动中有效）"""
        return self._dll.MCF_Uniaxial_dMaxV_Change_Net(
            ctypes.c_uint16(axis), ctypes.c_double(dmaxv), ctypes.c_uint16(station))

    def uniaxial_change_acc(self, axis: int, dmaxa: float, station: int = 0) -> int:
        """运动过程中更改加速度（仅单轴 T 型曲线运动中有效）"""
        return self._dll.MCF_Uniaxial_dMaxA_Change_Net(
            ctypes.c_uint16(axis), ctypes.c_double(dmaxa), ctypes.c_uint16(station))

    def axis_stop(self, axis: int, stop_mode: int, station: int = 0) -> int:
        """停止轴: stop_mode: 0=Axis_Stop_IMD 紧急停止, 1=Axis_Stop_DEC 减速停止"""
        return self._dll.MCF_Axis_Stop_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(stop_mode), ctypes.c_uint16(station))

    # ========================================================================
    # 系统
    # ========================================================================

    def get_version(self, station: int = 0) -> int:
        """获取固件版本"""
        version = ctypes.c_uint32()
        ret = self._dll.MCF_Get_Version_Net(ctypes.byref(version), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取版本失败: {get_error_message(ret)}")
        return version.value

    def get_serial_number(self, station: int = 0) -> int:
        """获取序列号"""
        serial = ctypes.c_int64()
        ret = self._dll.MCF_Get_Serial_Number_Net(ctypes.byref(serial), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取序列号失败: {get_error_message(ret)}")
        return serial.value

    def get_run_time(self, station: int = 0) -> int:
        """获取运行时间 (秒)"""
        run_time = ctypes.c_uint32()
        ret = self._dll.MCF_Get_Run_Time_Net(ctypes.byref(run_time), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取运行时间失败: {get_error_message(ret)}")
        return run_time.value

    # ========================================================================
    # 数字 I/O (第2章)
    # ========================================================================

    def get_input(self, station: int = 0) -> int:
        """读取全部数字输入（DI00-DI47），拼成一个 Python 整数返回。

        手册 2.4：MCF_Get_Input_Net 输出两个无符号长整型：
            All_Input_Logic1 -> DI00-DI31
            All_Input_Logic2 -> DI32-DI47
        注意本项目 NMC1400 只有 DI00-DI15（16 路输入）。
        """
        logic1 = ctypes.c_uint32()
        logic2 = ctypes.c_uint32()
        ret = self._dll.MCF_Get_Input_Net(
            ctypes.byref(logic1), ctypes.byref(logic2), ctypes.c_uint16(station))
        if ret != 0:
            return 0
        return (logic2.value << 32) | logic1.value

    def get_input_bit(self, bit_number: int, station: int = 0) -> int:
        """读取单个数字输入位（DI00-DI47）

        返回值：0=触点闭合(硬件灯亮)，1=触点打开(硬件灯灭)
        读取失败时抛出 NMCRuntimeError，避免把故障当成"低电平"。
        """
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Input_Bit_Net(
            ctypes.c_uint16(bit_number), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(
                f"读取输入点 DI{bit_number:02d} 失败: {get_error_message(ret)}")
        return value.value

    def set_output(self, all_output_logic: int, station: int = 0) -> int:
        """设置全部数字输出（DO00-DO31/DO47），参数为 32 位电平字"""
        return self._dll.MCF_Set_Output_Net(
            ctypes.c_uint32(all_output_logic & 0xFFFFFFFF), ctypes.c_uint16(station))

    def get_output(self, station: int = 0) -> int:
        """读取全部数字输出电平字（32 位）"""
        value = ctypes.c_uint32()
        ret = self._dll.MCF_Get_Output_Net(
            ctypes.byref(value), ctypes.c_uint16(station))
        if ret != 0:
            return 0
        return value.value

    def set_output_bit(self, bit_number: int, logic: int, station: int = 0) -> int:
        """按位设置数字输出（DO00-DO47）

        logic：0=触点闭合(硬件灯亮)，1=触点打开(硬件灯灭)
        """
        return self._dll.MCF_Set_Output_Bit_Net(
            ctypes.c_uint16(bit_number), ctypes.c_uint16(logic), ctypes.c_uint16(station))

    def get_output_bit(self, bit_number: int, station: int = 0) -> int:
        """按位读取数字输出（DO00-DO47）: 0=触点闭合, 1=触点打开"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Output_Bit_Net(
            ctypes.c_uint16(bit_number), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(
                f"读取输出点 DO{bit_number:02d} 失败: {get_error_message(ret)}")
        return value.value

    def set_output_time_bit(self, bit_number: int, logic: int,
                            hold_time_ms: int, station: int = 0) -> int:
        """按位输出并保持一段时间（单位 ms）"""
        return self._dll.MCF_Set_Output_Time_Bit_Net(
            ctypes.c_uint16(bit_number), ctypes.c_uint16(logic),
            ctypes.c_uint16(int(hold_time_ms)), ctypes.c_uint16(station))

    # ========================================================================
    # 便捷方法
    # ========================================================================

    def emergency_stop_all(self) -> None:
        """紧急停止所有轴 (立即停止 Axis_0 ~ Axis_3)"""
        for axis in range(4):
            try:
                self.axis_stop(axis, Axis_Stop_IMD)
            except Exception:
                pass

    def enable_all_servos(self, enable: bool = True, station: int = 0) -> None:
        """使能/关闭所有伺服"""
        logic = Servo_Open if enable else Servo_Close
        for axis in range(4):
            try:
                self.set_servo_enable(axis, logic, station)
            except Exception:
                pass

    def enable_axis2_servo(self, enable: bool = True, station: int = 0) -> None:
        """仅使能/关闭 Axis_2 (轴索引1)，其他轴保持不变"""
        logic = Servo_Open if enable else Servo_Close
        try:
            self.set_servo_enable(Axis_2, logic, station)
        except Exception as e:
            raise NMCRuntimeError(f"Axis_2 伺服{'使能' if enable else '关闭'}失败: {e}")

    def emergency_stop_axis2(self) -> None:
        """仅紧急停止 Axis_2 (轴索引1)，其他轴不受影响"""
        try:
            self.axis_stop(Axis_2, Axis_Stop_IMD)
        except Exception as e:
            raise NMCRuntimeError(f"Axis_2 紧急停止失败: {e}")

    def get_all_positions(self, station: int = 0) -> List[int]:
        """读取所有轴位置"""
        positions = []
        for axis in range(4):
            try:
                pos = self.get_position(axis, station)
                positions.append(pos)
            except Exception:
                positions.append(0)
        return positions

    def get_all_axis_states(self, station: int = 0) -> List[int]:
        """读取所有轴状态"""
        states = []
        for axis in range(4):
            try:
                state = self.get_axis_state(axis, station)
                states.append(state)
            except Exception:
                states.append(-1)
        return states

    def simplify_axis_state(self, state: int) -> int:
        """单轴原始状态码 → SYS_STATE_IDLE / SYS_STATE_MOVING / SYS_STATE_STOPPED"""
        if state == AXIS_BUSY:
            return SYS_STATE_MOVING
        if state == SYS_STATE_IDLE_AXIS:
            return SYS_STATE_IDLE
        return SYS_STATE_STOPPED   # 2~32（以及读取失败时的 -1）都算停止

    def get_state(self, station: int = 0) -> int:
        """综合 4 个轴判断控制器整体状态，返回 SYS_STATE_* 常量（int）。

        - 只要有任意轴处于停止/异常  → SYS_STATE_STOPPED
        - 否则只要有任意轴在运动     → SYS_STATE_MOVING
        - 否则四个轴都空闲           → SYS_STATE_IDLE
        """
        raw_states = self.get_all_axis_states(station)   # List[int]

        if any(self.simplify_axis_state(s) == SYS_STATE_STOPPED for s in raw_states):
            return SYS_STATE_STOPPED

        if any(self.simplify_axis_state(s) == SYS_STATE_MOVING for s in raw_states):
            return SYS_STATE_MOVING

        return SYS_STATE_IDLE
