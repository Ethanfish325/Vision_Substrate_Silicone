#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMC 系列运动控制卡 Python SDK
基于 ctypes 封装 MCDLL_NET.dll，提供 Python 友好的接口

硬件: NMC3401 (4轴运动控制卡)
DLL:  MCDLL_NET.dll

核心功能: 电机驱动（点位运动、JOG、回零、轴参数配置）
"""

import ctypes
import os
from typing import Optional, Tuple, List

from core.log_manager import log_info, log_error, log_warning


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

# 轴状态 (正数)
AXIS_BUSY = 1
AXIS_STOP_BY_EMG = 2
AXIS_STOP_BY_ALM = 3
AXIS_STOP_BY_POSITIVE_LIMIT = 4
AXIS_STOP_BY_NEGATIVE_LIMIT = 5
AXIS_STOP_BY_SOFT_POSITIVE = 6
AXIS_STOP_BY_SOFT_NEGATIVE = 7
AXIS_STOP_BY_EMG_BIT = 8
AXIS_STOP_BY_ALARM_TRIGGER = 9
AXIS_STOP_BY_ORIGIN = 10
AXIS_STOP_BY_Z_PHASE = 11
AXIS_STOP_BY_STOP_CMD = 12
AXIS_STOP_BY_PROFILE_END = 13
AXIS_STOP_BY_DEC_END = 14
AXIS_STOP_BY_GEAR_STOP = 15
AXIS_STOP_BY_PWM_STOP = 16
AXIS_STOP_BY_ERROR = 17
AXIS_STOP_BY_ORIGIN_DEC = 18
AXIS_STOP_BY_ORIGIN_BACK = 19
AXIS_STOP_BY_ORIGIN_LIMIT = 20
AXIS_STOP_BY_ORIGIN_DEV = 21
AXIS_STOP_BY_ORIGIN_TIME = 22
AXIS_STOP_BY_ORIGIN_OFFSET = 23
AXIS_STOP_BY_ORIGIN_ERR = 24
AXIS_STOP_BY_ORIGIN_ING = 25
AXIS_STOP_BY_ORIGIN_CFG_ERR = 26
AXIS_STOP_BY_ORIGIN_ABORT = 27
AXIS_STOP_BY_ORIGIN_OTHER = 28
BUFFER_EXECUTING = 29
BUFFER_STOPPED = 30
HOME_ERROR = 31
HOME_IN_PROGRESS = 32

# 错误码 (负数)
ERR_PARAM = -1
ERR_NO_AXIS = -2
ERR_AXIS_BUSY = -3
ERR_AXIS_STOP = -4
ERR_AXIS_EMG = -5
ERR_AXIS_ALM = -6
ERR_AXIS_LIMIT = -7
ERR_AXIS_ORIGIN = -8
ERR_AXIS_CFG = -9
ERR_AXIS_MODE = -10
ERR_BUFFER_FULL = -11
ERR_BUFFER_EMPTY = -12
ERR_BUFFER_MODE = -13
ERR_COORD_CFG = -14
ERR_COORD_BUSY = -15
ERR_COORD_STOP = -16
ERR_NO_STATION = -17
ERR_STATION_OFFLINE = -18
ERR_COMM_FAIL = -19
ERR_TIMEOUT = -20
ERR_UNKNOWN = -21
ERR_NOT_SUPPORT = -22

ERROR_MAP = {
    -1: "参数错误",
    -2: "轴不存在",
    -3: "轴忙",
    -4: "轴停止",
    -5: "轴急停",
    -6: "轴报警",
    -7: "轴限位",
    -8: "轴回零",
    -9: "轴配置错误",
    -10: "轴模式错误",
    -11: "缓冲区满",
    -12: "缓冲区空",
    -13: "缓冲区模式错误",
    -14: "坐标系配置错误",
    -15: "坐标系忙",
    -16: "坐标系停止",
    -17: "站不存在",
    -18: "站离线",
    -19: "通讯失败",
    -20: "超时",
    -21: "未知错误",
    -22: "不支持",
}

# 轴状态码定义（根据 NMC 编程手册 函数返回值 章节）
# 注意：这些是 MCF_JOG_Net / MCF_Get_Axis_State_Net 等函数的返回值
# 0 = 正常执行成功，>0 = 轴处于该状态（命令被拒绝）
AXIS_STATE_MAP = {
    0: "空闲",
    1: "执行中",
    2: "EMG立即紧急停止",
    3: "EMG减速紧急停止",
    4: "ALM立即停止",
    5: "ALM减速停止",
    6: "伺服使能立即停止",
    7: "伺服使能减速停止",
    8: "指令编码器误差立即停止",
    9: "指令编码器误差减速停止",
    10: "Index立即停止",
    11: "Index减速停止",
    12: "原点立即停止",
    13: "原点减速停止",
    14: "正硬限位立即停止",
    15: "正硬限位减速停止",
    16: "负硬限位立即停止",
    17: "负硬限位减速停止",
    18: "正软限位立即停止",
    19: "正软限位减速停止",
    20: "负软限位立即停止",
    21: "负软限位减速停止",
    22: "命令立即停止",
    23: "命令减速停止",
    24: "其它原因立即停止",
    25: "网络通讯中断立即停止",
    26: "未知原因立即停止",
    27: "未知原因减速停止",
    28: "外部IO减速停止",
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

# 回零模式
Home_Mode_1 = 1                # 模式1: 近门狗+Z相

# 级联模式
Switch_State_Series = 0
Switch_State_Parallel = 1

# 站类型
Station_Type_24I16O = 0
Station_Type_48I32O = 1
Station_Type_4D = 2     # NMC3401 (4轴)
Station_Type_8D = 3
Station_Type_24D = 4
Station_Type_4DM = 5

STATION_TYPE_NAMES = {
    0: "24输入/16输出 (NMC1200R/NMC1400/NMC3400)",
    1: "48输入/32输出",
    2: "4轴控制卡 (NMC3401)",
    3: "6/8轴控制卡 (NMC5800/NMC5600R)",
    4: "12/16轴控制卡 (NMC5120R/NMC5160)",
    5: "4轴DM控制卡",
}


# ============================================================================
# 工具函数
# ============================================================================
def get_error_message(ret: int) -> str:
    """获取返回值对应的错误描述"""
    if ret == 0:
        return "成功"
    if ret > 0:
        return AXIS_STATE_MAP.get(ret, f"未知状态码: {ret}")
    return ERROR_MAP.get(ret, f"未知错误码: {ret}")


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
            self._dll = ctypes.CDLL(self._dll_path)
        except OSError as e:
            raise NMCConnectionError(f"加载 DLL 失败: {e}")

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

        dll.MCF_Set_Position_Net.argtypes = [ctypes.c_uint16, ctypes.c_long, ctypes.c_uint16]
        dll.MCF_Set_Position_Net.restype = RET

        dll.MCF_Get_Position_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_long), ctypes.c_uint16]
        dll.MCF_Get_Position_Net.restype = RET

        dll.MCF_Set_Encoder_Net.argtypes = [ctypes.c_uint16, ctypes.c_long, ctypes.c_uint16]
        dll.MCF_Set_Encoder_Net.restype = RET

        dll.MCF_Get_Encoder_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_long), ctypes.c_uint16]
        dll.MCF_Get_Encoder_Net.restype = RET

        dll.MCF_Get_Vel_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_uint16]
        dll.MCF_Get_Vel_Net.restype = RET

        # ========== 运动停止触发 ==========
        dll.MCF_Set_EMG_Bit_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Set_EMG_Bit_Net.restype = RET

        dll.MCF_Set_Soft_Limit_Net.argtypes = [ctypes.c_uint16, ctypes.c_long, ctypes.c_long, ctypes.c_uint16]
        dll.MCF_Set_Soft_Limit_Net.restype = RET

        dll.MCF_Get_Soft_Limit_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long), ctypes.c_uint16]
        dll.MCF_Get_Soft_Limit_Net.restype = RET

        dll.MCF_Set_Soft_Limit_Enable_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Set_Soft_Limit_Enable_Net.restype = RET

        dll.MCF_Get_Soft_Limit_Enable_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Soft_Limit_Enable_Net.restype = RET

        dll.MCF_Clear_Axis_State_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Clear_Axis_State_Net.restype = RET

        dll.MCF_Get_Axis_State_Net.argtypes = [ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        dll.MCF_Get_Axis_State_Net.restype = RET

        # ========== 回零 ==========
        dll.MCF_Search_Home_Set_Net.argtypes = [
            ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16,
            ctypes.c_uint16, ctypes.c_uint16,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_long, ctypes.c_uint16, ctypes.c_uint16,
        ]
        dll.MCF_Search_Home_Set_Net.restype = RET

        dll.MCF_Search_Home_Start_Net.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Search_Home_Start_Net.restype = RET

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

        # 注意：MCF_Uniaxial_Net 的第二个参数在某些DLL版本中可能是 c_long
        # 这里同时保留 c_double 和 c_long 两个版本
        dll.MCF_Uniaxial_Net.argtypes = [ctypes.c_uint16, ctypes.c_double, ctypes.c_uint16, ctypes.c_uint16]
        dll.MCF_Uniaxial_Net.restype = RET

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
        dll.MCF_Get_Input_Net.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint16,
        ]
        dll.MCF_Get_Input_Net.restype = RET

        dll.MCF_Get_Input_Bit_Net.argtypes = [
            ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16,
        ]
        dll.MCF_Get_Input_Bit_Net.restype = RET

        dll.MCF_Set_Output_Net.argtypes = [
            ctypes.c_uint64, ctypes.c_uint16,
        ]
        dll.MCF_Set_Output_Net.restype = RET

        dll.MCF_Get_Output_Net.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint16,
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
    ) -> Tuple[int, List[int], List[int]]:
        """打开控制卡，返回 (站数量, 站编号列表, 站类型列表)

        根据手册:
            Connection_Number = 扩展模块数量
            Station_Number = 预分配的站编号数组 (函数会填充实际值)
            Station_Type = 预分配的站类型数组 (函数会填充实际值)

        示例:
            open_net(1)                          # 1个站，默认站号[0]，类型[2](NMC3401)
            open_net(3, [0,1,2], [2,2,1])        # 3个站，指定站号和类型

        Args:
            station_count: 扩展模块数量（即 Connection_Number）
            station_numbers: 预分配站编号列表，默认 [0, 1, ..., station_count-1]
            station_types: 预分配站类型列表，默认全部为 Station_Type_4D (2)
        """
        if station_numbers is None:
            station_numbers = list(range(station_count))
        if station_types is None:
            station_types = [Station_Type_4D] * station_count

        # 创建 ctypes 数组
        num_array = (ctypes.c_uint16 * station_count)(*station_numbers)
        type_array = (ctypes.c_uint16 * station_count)(*station_types)

        ret = self._dll.MCF_Open_Net(
            ctypes.c_uint16(station_count),
            num_array,
            type_array,
        )

        if ret < 0:
            raise NMCConnectionError(
                f"打开控制卡失败: {get_error_message(ret)}"
            )

        # 读取函数填充后的实际值
        self._station_count = station_count
        self._station_numbers = [int(num_array[i]) for i in range(station_count)]
        self._station_types = [int(type_array[i]) for i in range(station_count)]

        self._is_open = True
        self._connected = True
        return self._station_count, self._station_numbers, self._station_types

    def connect(
        self,
        ip: str = "192.168.1.200",
        port: int = 502,
        timeout_ms: int = 3000,
        station_count: int = 1,
    ) -> Tuple[int, List[int], List[int]]:
        """连接控制卡（封装 open_net，兼容旧版调用方式）

        Args:
            ip: 控制卡 IP 地址（仅用于日志记录，底层通过驱动通信）
            port: 端口号（仅用于日志记录）
            timeout_ms: 超时时间（毫秒）
            station_count: 扩展模块数量

        Returns:
            (站数量, 站编号列表, 站类型列表)
        """
        log_info(f"NMC 连接: {ip}:{port}, timeout={timeout_ms}ms")

        # 设置通信超时
        try:
            self._dll.MCF_Set_Link_TimeOut_Net(
                ctypes.c_uint32(timeout_ms),
                ctypes.c_uint32(timeout_ms),
                ctypes.c_uint16(0),
            )
        except Exception:
            pass

        return self.open_net(station_count=station_count)

    def get_open_net(self) -> Tuple[int, List[int], List[int]]:
        """读取打开参数: (连接号, 站编号列表, 站类型列表)"""
        connection_number = ctypes.c_uint16()
        max_stations = 32
        station_numbers = (ctypes.c_uint16 * max_stations)()
        station_types = (ctypes.c_uint16 * max_stations)()

        ret = self._dll.MCF_Get_Open_Net(
            ctypes.byref(connection_number),
            station_numbers,
            station_types,
        )

        if ret < 0:
            raise NMCRuntimeError(f"读取打开参数失败: {get_error_message(ret)}")

        numbers = []
        types = []
        for i in range(max_stations):
            if station_numbers[i] == 0 and station_types[i] == 0:
                break
            numbers.append(int(station_numbers[i]))
            types.append(int(station_types[i]))

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
        """设置链接超时"""
        return self._dll.MCF_Set_Link_TimeOut_Net(
            ctypes.c_uint32(time_1ms),
            ctypes.c_uint32(timeout_output),
            ctypes.c_uint16(station),
        )

    def get_link_state(self, station: int = 0) -> int:
        """获取链接状态: 0=断开, 1=连接"""
        return self._dll.MCF_Get_Link_State_Net(ctypes.c_uint16(station))

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
        """设置当前位置"""
        return self._dll.MCF_Set_Position_Net(
            ctypes.c_uint16(axis), ctypes.c_long(position), ctypes.c_uint16(station))

    def get_position(self, axis: int, station: int = 0) -> int:
        """读取当前位置"""
        value = ctypes.c_long()
        ret = self._dll.MCF_Get_Position_Net(
            ctypes.c_uint16(axis), ctypes.byref(value), ctypes.c_uint16(station))
        if ret < 0:
            raise NMCRuntimeError(f"读取位置失败: {get_error_message(ret)}")
        return value.value

    def set_encoder(self, axis: int, encoder: int, station: int = 0) -> int:
        """设置编码器值"""
        return self._dll.MCF_Set_Encoder_Net(
            ctypes.c_uint16(axis), ctypes.c_long(encoder), ctypes.c_uint16(station))

    def get_encoder(self, axis: int, station: int = 0) -> int:
        """读取编码器值"""
        value = ctypes.c_long()
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
        """设置软件限位位置"""
        return self._dll.MCF_Set_Soft_Limit_Net(
            ctypes.c_uint16(axis), ctypes.c_long(positive_pos), ctypes.c_long(negative_pos), ctypes.c_uint16(station))

    def get_soft_limit(self, axis: int, station: int = 0) -> tuple:
        """读取软件限位位置: (正限位, 负限位)"""
        pos = ctypes.c_long()
        neg = ctypes.c_long()
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
        """获取轴状态: 0=空闲, 1=执行中, 其他=停止原因"""
        reason = ctypes.c_uint16()
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
        """设置回零参数"""
        return self._dll.MCF_Search_Home_Set_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(mode),
            ctypes.c_uint16(limit_logic), ctypes.c_uint16(home_logic), ctypes.c_uint16(index_logic),
            ctypes.c_double(h_dmaxv), ctypes.c_double(l_dmaxv),
            ctypes.c_long(offset), ctypes.c_uint16(trigger_source), ctypes.c_uint16(station))

    def search_home_start(self, axis: int, station: int = 0) -> int:
        """开始回零"""
        return self._dll.MCF_Search_Home_Start_Net(
            ctypes.c_uint16(axis), ctypes.c_uint16(station))

    def search_home_get_state(self, axis: int, station: int = 0) -> int:
        """获取回零状态"""
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
        """JOG 运动: 正速度=正转, 负速度=反转"""
        return self._dll.MCF_JOG_Net(
            ctypes.c_uint16(axis), ctypes.c_double(dmaxv), ctypes.c_double(dmaxa), ctypes.c_uint16(station))

    def set_axis_profile(self, axis: int, v_ini: float, v_max: float,
                         a_max: float, jerk: float, v_end: float,
                         profile: int, station: int = 0) -> int:
        """设置轴速度参数: profile: 0=T曲线, 1=S曲线"""
        return self._dll.MCF_Set_Axis_Profile_Net(
            ctypes.c_uint16(axis),
            ctypes.c_double(v_ini), ctypes.c_double(v_max), ctypes.c_double(a_max),
            ctypes.c_double(jerk), ctypes.c_double(v_end),
            ctypes.c_uint16(profile), ctypes.c_uint16(station))

    def uniaxial(self, axis: int, dist: float, position_mode: int, station: int = 0) -> int:
        """单轴运动: position_mode: 0=绝对, 1=相对 (需先调用 set_axis_profile)"""
        return self._dll.MCF_Uniaxial_Net(
            ctypes.c_uint16(axis), ctypes.c_double(dist), ctypes.c_uint16(position_mode), ctypes.c_uint16(station))

    def uniaxial_int(self, axis: int, dist: int, position_mode: int, station: int = 0) -> int:
        """单轴运动（int版本）: 使用 c_long 传递距离值，兼容某些DLL版本"""
        return self._dll.MCF_Uniaxial_Net(
            ctypes.c_uint16(axis), ctypes.c_long(dist), ctypes.c_uint16(position_mode), ctypes.c_uint16(station))

    def uniaxial_long(self, axis: int, dist: int, position_mode: int, station: int = 0) -> int:
        """单轴运动（独立函数句柄，c_long版本）:
        使用独立的 ctypes 函数句柄，避免与主 argtypes(c_double) 冲突。
        适用于 DLL 第二个参数实际为 c_long 的版本。
        """
        _MCF_Uniaxial_Long = self._dll.MCF_Uniaxial_Net
        _MCF_Uniaxial_Long.argtypes = [ctypes.c_uint16, ctypes.c_long, ctypes.c_uint16, ctypes.c_uint16]
        _MCF_Uniaxial_Long.restype = ctypes.c_int16
        return _MCF_Uniaxial_Long(
            ctypes.c_uint16(axis), ctypes.c_long(dist), ctypes.c_uint16(position_mode), ctypes.c_uint16(station))

    def axis_stop(self, axis: int, stop_mode: int, station: int = 0) -> int:
        """停止轴: stop_mode: 0=立即停止, 1=减速停止"""
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
        """读取所有数字输入 (64位)"""
        value = ctypes.c_uint64()
        ret = self._dll.MCF_Get_Input_Net(
            ctypes.byref(value), ctypes.c_uint16(station))
        if ret != 0:
            # 非0可能表示错误或状态码，不抛出异常，返回0
            return 0
        return value.value

    def get_input_bit(self, bit_number: int, station: int = 0) -> int:
        """读取单个数字输入位: 0=低电平, 1=高电平"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Input_Bit_Net(
            ctypes.c_uint16(bit_number), ctypes.byref(value), ctypes.c_uint16(station))
        if ret != 0:
            return 0
        return value.value

    def set_output(self, all_output_logic: int, station: int = 0) -> int:
        """设置所有数字输出 (64位)"""
        return self._dll.MCF_Set_Output_Net(
            ctypes.c_uint64(all_output_logic), ctypes.c_uint16(station))

    def get_output(self, station: int = 0) -> int:
        """读取所有数字输出 (64位)"""
        value = ctypes.c_uint64()
        ret = self._dll.MCF_Get_Output_Net(
            ctypes.byref(value), ctypes.c_uint16(station))
        if ret != 0:
            return 0
        return value.value

    def set_output_bit(self, bit_number: int, logic: int, station: int = 0) -> int:
        """设置单个输出位: logic: 0=低电平, 1=高电平"""
        return self._dll.MCF_Set_Output_Bit_Net(
            ctypes.c_uint16(bit_number), ctypes.c_uint16(logic), ctypes.c_uint16(station))

    def get_output_bit(self, bit_number: int, station: int = 0) -> int:
        """读取单个输出位: 0=低电平, 1=高电平"""
        value = ctypes.c_uint16()
        ret = self._dll.MCF_Get_Output_Bit_Net(
            ctypes.c_uint16(bit_number), ctypes.byref(value), ctypes.c_uint16(station))
        if ret != 0:
            return 0
        return value.value

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