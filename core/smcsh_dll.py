# -*- coding: utf-8 -*-
"""
smcsh_dll.py
============
SMC6480 运动控制器动态链接库 (smcsh_mbs.dll) 的 ctypes 封装模块。

本模块负责加载 smcsh_mbs.dll，并封装与官方程序一致的底层函数：
  - 连接控制器相关函数（SMCOpen / SMCOpenEth / SMCOpenCom / SMCClose 等）
  - 运动参数设置函数（MSetting_* 系列）
  - 运动执行函数（Motion_* 系列）

官方程序在运动测试时实际调用的函数为 MSetting_* 与 Motion_* 格式
（例如 MSeting_SetStartSpeed、Motion_Pmove_Enter 等），与软件手册中
SMC* 前缀的命名不一致。本模块提供与官方一致的调用封装。

所有函数均返回错误码，错误码含义见软件手册 5.3 节"运动函数错误码说明"。
"""

import ctypes
import os
import struct
from ctypes import (
    c_char_p,
    c_double,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    POINTER,
)

# ---------------------------------------------------------------------------
# 常量定义（来自软件手册）
# ---------------------------------------------------------------------------

# 链接类型 (SMCOpen 的 type 参数)
SMC6X_CONNECTION_COM = 0   # 串口
SMC6X_CONNECTION_ETH = 1   # 以太网

# 控制器状态 (SMCGetState 返回值)
SYS_STATE_IDLE            = 1   # 待机
SYS_STATE_GRUNNING        = 3   # 运行
SYS_STATE_MANUALING       = 4   # 手动
SYS_STATE_PAUSE           = 5   # 暂停
SYS_STATE_GEDIT           = 6   # 程序编辑
SYS_STATE_SETTING         = 7   # 设置
SYS_STATE_TEST            = 8   # 测试
SYS_STATE_GFILEREVIEW     = 9   # gfile 浏览
SYS_STATE_UDISK           = 10  # U盘操作
SYS_STATE_GTEACHING       = 11  # 示教
SYS_STATE_CANNOT_CONNECT  = 50  # 链接不上

# 错误码（软件手册 5.3 节）
ERR_NOERR                = 0    # 成功
ERRCODE_UNKNOWN          = 1    # 未知错误
ERRCODE_PARAERR          = 2    # 参数错误
ERRCODE_TIMEOUT          = 3    # 操作超时
ERRCODE_CONTROLLERBUSY   = 4    # 控制卡状态忙
ERRCODE_CONNECT_TOOMANY  = 5    # 打开的客户端太多
ERRCODE_OS_ERR           = 6    # 操作系统错误
ERRCODE_CANNOT_OPEN_COM  = 7    # 无法打开串口
ERRCODE_CANNOT_CONNECTETH = 8   # 无法连接
ERRCODE_HANDLEERR        = 9    # 连接标识错误
ERRCODE_SENDERR          = 10   # 发送错误

# 错误码 -> 中文描述映射
ERRCODE_DESC = {
    ERR_NOERR:                 "成功",
    ERRCODE_UNKNOWN:           "未知错误",
    ERRCODE_PARAERR:           "参数错误",
    ERRCODE_TIMEOUT:           "操作超时",
    ERRCODE_CONTROLLERBUSY:    "控制卡状态忙",
    ERRCODE_CONNECT_TOOMANY:   "打开的客户端太多",
    ERRCODE_OS_ERR:            "操作系统错误",
    ERRCODE_CANNOT_OPEN_COM:   "无法打开串口",
    ERRCODE_CANNOT_CONNECTETH: "无法连接",
    ERRCODE_HANDLEERR:         "连接标识错误",
    ERRCODE_SENDERR:           "发送错误",
}

# 控制器状态 -> 中文描述映射
SYS_STATE_DESC = {
    SYS_STATE_IDLE:           "待机",
    SYS_STATE_GRUNNING:       "运行",
    SYS_STATE_MANUALING:      "手动",
    SYS_STATE_PAUSE:          "暂停",
    SYS_STATE_GEDIT:          "程序编辑",
    SYS_STATE_SETTING:        "设置",
    SYS_STATE_TEST:           "测试",
    SYS_STATE_GFILEREVIEW:    "gfile 浏览",
    SYS_STATE_UDISK:          "U盘操作",
    SYS_STATE_GTEACHING:      "示教",
    SYS_STATE_CANNOT_CONNECT: "链接不上",
}


class SMCHANDLE(ctypes.c_int32):
    """
    连接标识句柄类型。

    雷赛 SMC 系列 DLL 中 SMCHANDLE 定义为 int32（32 位整数句柄），
    而非指针。使用 c_int32 以确保句柄值被正确传递。
    """
    pass


class SMCSHDLL:
    """
    smcsh_mbs.dll 封装类。

    负责加载动态链接库并封装与官方程序一致的底层函数
    （连接、MSetting_* 运动参数、Motion_* 运动执行）。

    使用示例：
        dll = SMCSHDLL()
        handle = dll.open_eth("192.168.1.11")
        dll.set_start_speed(handle, 0, 800)
        dll.pmove_enter(handle, 0)
        dll.pmove_set_absolute(handle, 0, 0)
        dll.pmove_start(handle, 0)
        dll.close(handle)
    """

    def __init__(self, dll_path: str = "smcsh_mbs.dll"):
        """
        加载 smcsh_mbs.dll。

        优先基于本模块所在目录解析 DLL 路径，并转换为绝对路径，
        以避免因当前工作目录不同而导致的加载失败。同时会检测
        Python 与 DLL 的位数是否匹配（32 位 DLL 需使用 32 位 Python）。

        :param dll_path: 动态链接库路径，默认当前目录下的 smcsh_mbs.dll
        :raises OSError: 当无法加载 DLL 时抛出
        """
        self._dll_path = self._resolve_dll_path(dll_path)
        if not os.path.exists(self._dll_path):
            raise FileNotFoundError(
                f"未找到动态链接库: {self._dll_path}，请确认 smcsh_mbs.dll 位于程序目录下。"
            )

        # 检测 Python 与 DLL 位数是否匹配
        self._check_bitness(self._dll_path)

        # 使用 WinDLL 加载（stdcall 调用约定），传入绝对路径
        self._dll = ctypes.WinDLL(self._dll_path)

        # IO 端口函数（在 _bind_functions 中容错绑定，可能为 None）
        self._in_port_func = None
        self._out_port_func = None

        self._bind_functions()

    @staticmethod
    def _check_bitness(dll_path: str):
        """
        检测 Python 解释器与 DLL 的位数是否匹配。

        ctypes 只能加载与 Python 解释器位数一致的 DLL。
        若位数不匹配，抛出带清晰中文提示的异常。
        """
        python_bits = ctypes.sizeof(ctypes.c_void_p) * 8

        try:
            with open(dll_path, "rb") as f:
                # 读取 PE 头中的机器类型字段
                f.seek(0x3C)
                pe_offset = struct.unpack("<I", f.read(4))[0]
                f.seek(pe_offset + 4)
                machine = struct.unpack("<H", f.read(2))[0]
        except Exception:  # noqa: BLE001
            # 无法解析 PE 头时跳过检测，交由 WinDLL 处理
            return

        if machine == 0x8664:      # x64
            dll_bits = 64
        elif machine == 0x014C:    # x86
            dll_bits = 32
        else:
            return

        if python_bits != dll_bits:
            raise OSError(
                f"位数不匹配：当前 Python 为 {python_bits} 位，"
                f"而 smcsh_mbs.dll 为 {dll_bits} 位。\n"
                f"ctypes 无法加载位数不一致的 DLL。\n"
                f"请使用 {dll_bits} 位 Python 运行本程序。"
            )

    @staticmethod
    def _resolve_dll_path(dll_path: str) -> str:
        """
        将 DLL 路径解析为绝对路径。

        若传入的是相对路径，则依次在以下位置查找：
          1. smcsh_dll.py 所在目录（项目根目录）
          2. 项目根目录下的 libs/ 子目录
          3. 当前工作目录
        """
        if os.path.isabs(dll_path):
            return dll_path

        # 本模块所在目录（即项目根目录）
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. 项目根目录
        candidate = os.path.join(base_dir, dll_path)
        if os.path.exists(candidate):
            return candidate

        # 2. libs/ 子目录
        candidate = os.path.join(base_dir, "libs", dll_path)
        if os.path.exists(candidate):
            return candidate

        # 3. 回退到当前工作目录
        return os.path.abspath(dll_path)

    # ------------------------------------------------------------------
    # 内部：绑定 DLL 导出函数
    # ------------------------------------------------------------------
    def _bind_functions(self):
        dll = self._dll

        # ---- 连接控制器相关函数 ----
        # int32 SMCOpen(int32 type, char* pconnectstring, SMCHANDLE* phandle)
        dll.SMCOpen.restype = c_int32
        dll.SMCOpen.argtypes = [c_int32, c_char_p, POINTER(SMCHANDLE)]

        # int32 SMCOpenCom(uint32 comid, SMCHANDLE* phandle)
        dll.SMCOpenCom.restype = c_int32
        dll.SMCOpenCom.argtypes = [c_uint32, POINTER(SMCHANDLE)]

        # int32 SMCOpenEth(char* ipaddr, SMCHANDLE* phandle)
        dll.SMCOpenEth.restype = c_int32
        dll.SMCOpenEth.argtypes = [c_char_p, POINTER(SMCHANDLE)]

        # int32 SMCClose(SMCHANDLE handle)
        dll.SMCClose.restype = c_int32
        dll.SMCClose.argtypes = [SMCHANDLE]

        # int32 SMCSetTimeOut(SMCHANDLE handle, uint32 timems)
        dll.SMCSetTimeOut.restype = c_int32
        dll.SMCSetTimeOut.argtypes = [SMCHANDLE, c_uint32]

        # int32 SMCGetState(SMCHANDLE handle, uint8* pstate)
        dll.SMCGetState.restype = c_int32
        dll.SMCGetState.argtypes = [SMCHANDLE, POINTER(c_uint8)]

        # uint8 SMCGetAxises(SMCHANDLE handle)
        dll.SMCGetAxises.restype = c_uint8
        dll.SMCGetAxises.argtypes = [SMCHANDLE]

        # int32 SMCGetSoftwareVersion(SMCHANDLE handle, uint32* pVersion)
        dll.SMCGetSoftwareVersion.restype = c_int32
        dll.SMCGetSoftwareVersion.argtypes = [SMCHANDLE, POINTER(c_uint32)]

        # int32 SMCGetHardwareId(SMCHANDLE handle, uint16* pId)
        dll.SMCGetHardwareId.restype = c_int32
        dll.SMCGetHardwareId.argtypes = [SMCHANDLE, POINTER(c_uint16)]

        # char* SMCGetErrcodeDescription(int32 ierrcode)
        dll.SMCGetErrcodeDescription.restype = c_char_p
        dll.SMCGetErrcodeDescription.argtypes = [c_int32]

        # ---- 运动参数设置函数（MSetting_* 系列，与官方一致）----
        # 官方程序中除 S 段时间可用小数外，其余运动参数均为整数（int32）
        # 注意：MSetting_Get* 系列直接返回参数值（非错误码），签名只有 (handle, axis)
        # int32 MSetting_SetStartSpeed(SMCHANDLE, uint8, int32)
        dll.MSetting_SetStartSpeed.restype = c_int32
        dll.MSetting_SetStartSpeed.argtypes = [SMCHANDLE, c_uint8, c_int32]
        # int32 MSetting_GetStartSpeed(SMCHANDLE, uint8)  -> 直接返回速度值
        dll.MSetting_GetStartSpeed.restype = c_int32
        dll.MSetting_GetStartSpeed.argtypes = [SMCHANDLE, c_uint8]

        # int32 MSetting_SetMotionAxisSpeed(SMCHANDLE, uint8, int32)
        dll.MSetting_SetMotionAxisSpeed.restype = c_int32
        dll.MSetting_SetMotionAxisSpeed.argtypes = [SMCHANDLE, c_uint8, c_int32]
        # int32 MSetting_GetMotionAxisSpeed(SMCHANDLE, uint8)  -> 直接返回速度值
        dll.MSetting_GetMotionAxisSpeed.restype = c_int32
        dll.MSetting_GetMotionAxisSpeed.argtypes = [SMCHANDLE, c_uint8]

        # int32 MSetting_SetAcceleration(SMCHANDLE, uint8, int32)
        dll.MSetting_SetAcceleration.restype = c_int32
        dll.MSetting_SetAcceleration.argtypes = [SMCHANDLE, c_uint8, c_int32]
        # int32 MSetting_GetAcceleration(SMCHANDLE, uint8)  -> 直接返回加速度值
        dll.MSetting_GetAcceleration.restype = c_int32
        dll.MSetting_GetAcceleration.argtypes = [SMCHANDLE, c_uint8]

        # int32 MSetting_SetDeceleration(SMCHANDLE, uint8, int32)
        dll.MSetting_SetDeceleration.restype = c_int32
        dll.MSetting_SetDeceleration.argtypes = [SMCHANDLE, c_uint8, c_int32]
        # int32 MSetting_GetDeceleration(SMCHANDLE, uint8)  -> 直接返回减速度值
        dll.MSetting_GetDeceleration.restype = c_int32
        dll.MSetting_GetDeceleration.argtypes = [SMCHANDLE, c_uint8]

        # int32 MSetting_SetSCurveSet(SMCHANDLE, uint8, double)
        dll.MSetting_SetSCurveSet.restype = c_int32
        dll.MSetting_SetSCurveSet.argtypes = [SMCHANDLE, c_uint8, c_double]
        # int32 MSetting_GetSCurveSet(SMCHANDLE, uint8)  -> 直接返回 S 曲线值
        dll.MSetting_GetSCurveSet.restype = c_int32
        dll.MSetting_GetSCurveSet.argtypes = [SMCHANDLE, c_uint8]

        # ---- 回零参数设置函数（MSetting_* 系列，与官方一致）----
        # int32 MSetting_SetZeroSpeed(SMCHANDLE, uint8, int32)  -> 回零低速
        dll.MSetting_SetZeroSpeed.restype = c_int32
        dll.MSetting_SetZeroSpeed.argtypes = [SMCHANDLE, c_uint8, c_int32]
        # int32 MSetting_GetZeroSpeed(SMCHANDLE, uint8)  -> 直接返回回零低速
        dll.MSetting_GetZeroSpeed.restype = c_int32
        dll.MSetting_GetZeroSpeed.argtypes = [SMCHANDLE, c_uint8]

        # int32 MSetting_SetZeroDir(SMCHANDLE, uint8, int32)  -> 回零方向
        dll.MSetting_SetZeroDir.restype = c_int32
        dll.MSetting_SetZeroDir.argtypes = [SMCHANDLE, c_uint8, c_int32]
        # int32 MSetting_GetZeroDir(SMCHANDLE, uint8)  -> 直接返回回零方向
        dll.MSetting_GetZeroDir.restype = c_int32
        dll.MSetting_GetZeroDir.argtypes = [SMCHANDLE, c_uint8]

        # int32 MSetting_SetZeroMode(SMCHANDLE, uint8, int32)  -> 回零模式
        dll.MSetting_SetZeroMode.restype = c_int32
        dll.MSetting_SetZeroMode.argtypes = [SMCHANDLE, c_uint8, c_int32]
        # int32 MSetting_GetZeroMode(SMCHANDLE, uint8)  -> 直接返回回零模式
        dll.MSetting_GetZeroMode.restype = c_int32
        dll.MSetting_GetZeroMode.argtypes = [SMCHANDLE, c_uint8]

        # ---- 点位运动函数（Motion_Pmove_* 系列，与官方一致）----
        # int32 Motion_Pmove_Enter(SMCHANDLE, uint8)
        dll.Motion_Pmove_Enter.restype = c_int32
        dll.Motion_Pmove_Enter.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_Pmove_SetAbsolute(SMCHANDLE, uint8, int32)
        dll.Motion_Pmove_SetAbsolute.restype = c_int32
        dll.Motion_Pmove_SetAbsolute.argtypes = [SMCHANDLE, c_uint8, c_int32]

        # int32 Motion_Pmove_SetRelative(SMCHANDLE, uint8, int32)
        dll.Motion_Pmove_SetRelative.restype = c_int32
        dll.Motion_Pmove_SetRelative.argtypes = [SMCHANDLE, c_uint8, c_int32]

        # int32 Motion_Pmove_Start(SMCHANDLE, uint8)
        dll.Motion_Pmove_Start.restype = c_int32
        dll.Motion_Pmove_Start.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_Pmove_GetAbsolute(SMCHANDLE, uint8, double*)
        dll.Motion_Pmove_GetAbsolute.restype = c_int32
        dll.Motion_Pmove_GetAbsolute.argtypes = [SMCHANDLE, c_uint8, POINTER(c_double)]

        # int32 Motion_Pmove_GetRelative(SMCHANDLE, uint8, double*)
        dll.Motion_Pmove_GetRelative.restype = c_int32
        dll.Motion_Pmove_GetRelative.argtypes = [SMCHANDLE, c_uint8, POINTER(c_double)]

        # ---- 定速运动函数（Motion_Vmove_* 系列）----
        # int32 Motion_Vmove_Enter(SMCHANDLE, uint8)
        dll.Motion_Vmove_Enter.restype = c_int32
        dll.Motion_Vmove_Enter.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_Vmove_SetDir(SMCHANDLE, uint8, uint8)
        dll.Motion_Vmove_SetDir.restype = c_int32
        dll.Motion_Vmove_SetDir.argtypes = [SMCHANDLE, c_uint8, c_uint8]

        # int32 Motion_Vmove_SetSpeed(SMCHANDLE, uint8, int32)
        dll.Motion_Vmove_SetSpeed.restype = c_int32
        dll.Motion_Vmove_SetSpeed.argtypes = [SMCHANDLE, c_uint8, c_int32]

        # int32 Motion_Vmove_Start(SMCHANDLE, uint8)
        dll.Motion_Vmove_Start.restype = c_int32
        dll.Motion_Vmove_Start.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_Vmove_GetDir(SMCHANDLE, uint8, uint8*)
        dll.Motion_Vmove_GetDir.restype = c_int32
        dll.Motion_Vmove_GetDir.argtypes = [SMCHANDLE, c_uint8, POINTER(c_uint8)]

        # int32 Motion_Vmove_GetSpeed(SMCHANDLE, uint8, double*)
        dll.Motion_Vmove_GetSpeed.restype = c_int32
        dll.Motion_Vmove_GetSpeed.argtypes = [SMCHANDLE, c_uint8, POINTER(c_double)]

        # ---- 停止 / 状态函数（Motion_* 系列）----
        # int32 Motion_DeclStop(SMCHANDLE, uint8)
        dll.Motion_DeclStop.restype = c_int32
        dll.Motion_DeclStop.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_ImdStop(SMCHANDLE, uint8)
        dll.Motion_ImdStop.restype = c_int32
        dll.Motion_ImdStop.argtypes = [SMCHANDLE, c_uint8]

        # 注意：Motion_CheckDown 与 Motion_Home_IfHoming 均为"直接返回值"类型
        # （与 Motion_Get* 系列一致），而非"返回错误码 + 指针输出"。
        # 经 32 位 Python 实测确认：传入指针参数时指针不会被写入，返回值即状态值。
        # int32 Motion_CheckDown(SMCHANDLE, uint8)  -> 直接返回是否停止(0=否,1=是)
        dll.Motion_CheckDown.restype = c_int32
        dll.Motion_CheckDown.argtypes = [SMCHANDLE, c_uint8]

        # 注意：Motion_Get* 系列直接返回参数值（非错误码），签名只有 (handle, axis)
        # int32 Motion_GetPulsePositon(SMCHANDLE, uint8)  -> 直接返回规划位置
        dll.Motion_GetPulsePositon.restype = c_int32
        dll.Motion_GetPulsePositon.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_GetEncoderPositon(SMCHANDLE, uint8)  -> 直接返回编码器位置
        dll.Motion_GetEncoderPositon.restype = c_int32
        dll.Motion_GetEncoderPositon.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_GetAimPositon(SMCHANDLE, uint8)  -> 直接返回目标位置
        dll.Motion_GetAimPositon.restype = c_int32
        dll.Motion_GetAimPositon.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_SetPulsePositon(SMCHANDLE, uint8, int32)
        dll.Motion_SetPulsePositon.restype = c_int32
        dll.Motion_SetPulsePositon.argtypes = [SMCHANDLE, c_uint8, c_int32]

        # int32 Motion_GetCurSpeed(SMCHANDLE, uint8)  -> 直接返回当前速度
        dll.Motion_GetCurSpeed.restype = c_int32
        dll.Motion_GetCurSpeed.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_GetStopReason(SMCHANDLE, uint8)  -> 直接返回停止原因
        dll.Motion_GetStopReason.restype = c_int32
        dll.Motion_GetStopReason.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_Home_FindOrigin(SMCHANDLE, uint8)
        dll.Motion_Home_FindOrigin.restype = c_int32
        dll.Motion_Home_FindOrigin.argtypes = [SMCHANDLE, c_uint8]

        # int32 Motion_Home_IfHoming(SMCHANDLE, uint8)  -> 直接返回是否回零中(0=否,1=是)
        dll.Motion_Home_IfHoming.restype = c_int32
        dll.Motion_Home_IfHoming.argtypes = [SMCHANDLE, c_uint8]

        # 读取输入端口(DI) / 设置输出端口(DO) —— 容错绑定，尝试多个候选函数名
        # 不同版本的 smcsh_mbs.dll 函数命名可能不同（Motion_* / SMC_* 前缀）
        self._in_port_func = None
        self._out_port_func = None
        for name in ("SMCReadInBit", "SMCReadInPort", "SMCReadInpStates",
                     "Motion_GetInPort", "SMC_GetInPort",
                     "Motion_GetInPorts", "SMC_GetInPorts",
                     "Motion_ReadInPort", "SMC_ReadInPort"):
            func = getattr(dll, name, None)
            if func is not None:
                func.restype = c_int32
                func.argtypes = [SMCHANDLE, c_uint8, POINTER(c_uint8)]
                self._in_port_func = func
                break
        for name in ("SMCWriteOutBit", "SMCWriteOutPort",
                     "Motion_SetOutPort", "SMC_SetOutPort",
                     "Motion_WriteOutPort", "SMC_WriteOutPort"):
            func = getattr(dll, name, None)
            if func is not None:
                func.restype = c_int32
                func.argtypes = [SMCHANDLE, c_uint8, c_uint8]
                self._out_port_func = func
                break

    # ------------------------------------------------------------------
    # 对外：连接控制器相关函数
    # ------------------------------------------------------------------
    def open(self, conn_type: int, connect_string: str) -> tuple:
        """通用打开链接（SMCOpen）。返回 (错误码, SMCHANDLE 句柄)。"""
        handle = SMCHANDLE()
        err = self._dll.SMCOpen(
            c_int32(conn_type),
            c_char_p(connect_string.encode("utf-8")),
            ctypes.byref(handle),
        )
        return err, handle

    def open_com(self, com_id: int) -> tuple:
        """打开串口链接（SMCOpenCom）。返回 (错误码, SMCHANDLE 句柄)。"""
        handle = SMCHANDLE()
        err = self._dll.SMCOpenCom(c_uint32(com_id), ctypes.byref(handle))
        return err, handle

    def open_eth(self, ip_addr: str) -> tuple:
        """打开以太网链接（SMCOpenEth）。返回 (错误码, SMCHANDLE 句柄)。"""
        handle = SMCHANDLE()
        err = self._dll.SMCOpenEth(
            c_char_p(ip_addr.encode("utf-8")), ctypes.byref(handle)
        )
        return err, handle

    def close(self, handle) -> int:
        """关闭链接（SMCClose）。"""
        if not handle:
            return ERR_NOERR
        return self._dll.SMCClose(handle)

    def set_timeout(self, handle, timeout_ms: int) -> int:
        """设置命令应答最长延时（SMCSetTimeOut）。"""
        return self._dll.SMCSetTimeOut(handle, c_uint32(timeout_ms))

    def get_state(self, handle) -> tuple:
        """读取控制器状态（SMCGetState）。返回 (错误码, 状态码)。"""
        state = c_uint8()
        err = self._dll.SMCGetState(handle, ctypes.byref(state))
        return err, state.value

    def get_axises(self, handle) -> int:
        """读取控制器轴数（SMCGetAxises），出错返回 0。"""
        return self._dll.SMCGetAxises(handle)

    def get_software_version(self, handle) -> tuple:
        """读取软件版本（SMCGetSoftwareVersion），日期标识。"""
        ver = c_uint32()
        err = self._dll.SMCGetSoftwareVersion(handle, ctypes.byref(ver))
        return err, ver.value

    def get_hardware_id(self, handle) -> tuple:
        """读取硬件版本（SMCGetHardwareId）。"""
        hid = c_uint16()
        err = self._dll.SMCGetHardwareId(handle, ctypes.byref(hid))
        return err, hid.value

    def get_errcode_description(self, errcode: int) -> str:
        """读取错误码描述（SMCGetErrcodeDescription）。"""
        desc = self._dll.SMCGetErrcodeDescription(c_int32(errcode))
        if desc:
            return desc.decode("utf-8", errors="ignore")
        return ERRCODE_DESC.get(errcode, f"未知错误码({errcode})")

    # ------------------------------------------------------------------
    # 对外：运动参数设置函数（MSetting_* 系列，与官方一致）
    # ------------------------------------------------------------------
    def set_start_speed(self, handle, iaxis: int, speed: int) -> int:
        """设置启动速度（MSetting_SetStartSpeed），speed 为整数。"""
        return self._dll.MSetting_SetStartSpeed(handle, c_uint8(iaxis), c_int32(speed))

    def get_start_speed(self, handle, iaxis: int) -> int:
        """读取启动速度（MSetting_GetStartSpeed），直接返回速度值。"""
        return self._dll.MSetting_GetStartSpeed(handle, c_uint8(iaxis))

    def set_motion_axis_speed(self, handle, iaxis: int, speed: int) -> int:
        """设置最大速度（MSetting_SetMotionAxisSpeed），speed 为整数。"""
        return self._dll.MSetting_SetMotionAxisSpeed(handle, c_uint8(iaxis), c_int32(speed))

    def get_motion_axis_speed(self, handle, iaxis: int) -> int:
        """读取最大速度（MSetting_GetMotionAxisSpeed），直接返回速度值。"""
        return self._dll.MSetting_GetMotionAxisSpeed(handle, c_uint8(iaxis))

    def set_acceleration(self, handle, iaxis: int, acc: int) -> int:
        """设置加速度（MSetting_SetAcceleration），acc 为整数。"""
        return self._dll.MSetting_SetAcceleration(handle, c_uint8(iaxis), c_int32(acc))

    def get_acceleration(self, handle, iaxis: int) -> int:
        """读取加速度（MSetting_GetAcceleration），直接返回加速度值。"""
        return self._dll.MSetting_GetAcceleration(handle, c_uint8(iaxis))

    def set_deceleration(self, handle, iaxis: int, dec: int) -> int:
        """设置减速度（MSetting_SetDeceleration），dec 为整数。"""
        return self._dll.MSetting_SetDeceleration(handle, c_uint8(iaxis), c_int32(dec))

    def get_deceleration(self, handle, iaxis: int) -> int:
        """读取减速度（MSetting_GetDeceleration），直接返回减速度值。"""
        return self._dll.MSetting_GetDeceleration(handle, c_uint8(iaxis))

    def set_s_curve(self, handle, iaxis: int, s_curve: float) -> int:
        """设置 S 曲线（MSetting_SetSCurveSet）。"""
        return self._dll.MSetting_SetSCurveSet(handle, c_uint8(iaxis), c_double(s_curve))

    def get_s_curve(self, handle, iaxis: int) -> int:
        """读取 S 曲线（MSetting_GetSCurveSet），直接返回 S 曲线值。"""
        return self._dll.MSetting_GetSCurveSet(handle, c_uint8(iaxis))

    def set_zero_speed(self, handle, iaxis: int, speed: int) -> int:
        """设置回零低速（MSetting_SetZeroSpeed），speed 为整数。"""
        return self._dll.MSetting_SetZeroSpeed(handle, c_uint8(iaxis), c_int32(speed))

    def get_zero_speed(self, handle, iaxis: int) -> int:
        """读取回零低速（MSetting_GetZeroSpeed），直接返回速度值。"""
        return self._dll.MSetting_GetZeroSpeed(handle, c_uint8(iaxis))

    def set_zero_dir(self, handle, iaxis: int, direction: int) -> int:
        """设置回零方向（MSetting_SetZeroDir），direction 为整数。"""
        return self._dll.MSetting_SetZeroDir(handle, c_uint8(iaxis), c_int32(direction))

    def get_zero_dir(self, handle, iaxis: int) -> int:
        """读取回零方向（MSetting_GetZeroDir），直接返回方向值。"""
        return self._dll.MSetting_GetZeroDir(handle, c_uint8(iaxis))

    def set_zero_mode(self, handle, iaxis: int, mode: int) -> int:
        """设置回零模式（MSetting_SetZeroMode），mode 为整数。"""
        return self._dll.MSetting_SetZeroMode(handle, c_uint8(iaxis), c_int32(mode))

    def get_zero_mode(self, handle, iaxis: int) -> int:
        """读取回零模式（MSetting_GetZeroMode），直接返回模式值。"""
        return self._dll.MSetting_GetZeroMode(handle, c_uint8(iaxis))

    # ------------------------------------------------------------------
    # 对外：点位运动函数（Motion_Pmove_* 系列，与官方一致）
    # ------------------------------------------------------------------
    def pmove_enter(self, handle, iaxis: int) -> int:
        """进入点位运动模式（Motion_Pmove_Enter）。"""
        return self._dll.Motion_Pmove_Enter(handle, c_uint8(iaxis))

    def pmove_set_absolute(self, handle, iaxis: int, pos: int) -> int:
        """设置绝对目标位置（Motion_Pmove_SetAbsolute），pos 为整数。"""
        return self._dll.Motion_Pmove_SetAbsolute(handle, c_uint8(iaxis), c_int32(pos))

    def pmove_set_relative(self, handle, iaxis: int, dist: int) -> int:
        """设置相对移动距离（Motion_Pmove_SetRelative），dist 为整数。"""
        return self._dll.Motion_Pmove_SetRelative(handle, c_uint8(iaxis), c_int32(dist))

    def pmove_start(self, handle, iaxis: int) -> int:
        """启动点位运动（Motion_Pmove_Start）。"""
        return self._dll.Motion_Pmove_Start(handle, c_uint8(iaxis))

    def pmove_get_absolute(self, handle, iaxis: int) -> tuple:
        """读取绝对目标位置（Motion_Pmove_GetAbsolute）。返回 (错误码, 位置)。"""
        v = c_double()
        err = self._dll.Motion_Pmove_GetAbsolute(handle, c_uint8(iaxis), ctypes.byref(v))
        return err, v.value

    def pmove_get_relative(self, handle, iaxis: int) -> tuple:
        """读取相对移动距离（Motion_Pmove_GetRelative）。返回 (错误码, 距离)。"""
        v = c_double()
        err = self._dll.Motion_Pmove_GetRelative(handle, c_uint8(iaxis), ctypes.byref(v))
        return err, v.value

    # ------------------------------------------------------------------
    # 对外：定速运动函数（Motion_Vmove_* 系列）
    # ------------------------------------------------------------------
    def vmove_enter(self, handle, iaxis: int) -> int:
        """进入定速运动模式（Motion_Vmove_Enter）。"""
        return self._dll.Motion_Vmove_Enter(handle, c_uint8(iaxis))

    def vmove_set_dir(self, handle, iaxis: int, positive: int) -> int:
        """设置定速运动方向（Motion_Vmove_SetDir），positive 是否正向。"""
        return self._dll.Motion_Vmove_SetDir(handle, c_uint8(iaxis), c_uint8(positive))

    def vmove_set_speed(self, handle, iaxis: int, speed: int) -> int:
        """设置定速运动速度（Motion_Vmove_SetSpeed），speed 为整数。"""
        return self._dll.Motion_Vmove_SetSpeed(handle, c_uint8(iaxis), c_int32(speed))

    def vmove_start(self, handle, iaxis: int) -> int:
        """启动定速运动（Motion_Vmove_Start）。"""
        return self._dll.Motion_Vmove_Start(handle, c_uint8(iaxis))

    def vmove_get_dir(self, handle, iaxis: int) -> tuple:
        """读取定速运动方向（Motion_Vmove_GetDir）。返回 (错误码, 是否正向)。"""
        d = c_uint8()
        err = self._dll.Motion_Vmove_GetDir(handle, c_uint8(iaxis), ctypes.byref(d))
        return err, bool(d.value)

    def vmove_get_speed(self, handle, iaxis: int) -> tuple:
        """读取定速运动速度（Motion_Vmove_GetSpeed）。返回 (错误码, 速度)。"""
        v = c_double()
        err = self._dll.Motion_Vmove_GetSpeed(handle, c_uint8(iaxis), ctypes.byref(v))
        return err, v.value

    # ------------------------------------------------------------------
    # 对外：停止 / 状态函数（Motion_* 系列）
    # ------------------------------------------------------------------
    def decl_stop(self, handle, iaxis: int) -> int:
        """减速停止（Motion_DeclStop）。"""
        return self._dll.Motion_DeclStop(handle, c_uint8(iaxis))

    def imd_stop(self, handle, iaxis: int) -> int:
        """立即停止（Motion_ImdStop）。"""
        return self._dll.Motion_ImdStop(handle, c_uint8(iaxis))

    def check_down(self, handle, iaxis: int) -> int:
        """检查轴是否停止（Motion_CheckDown），直接返回是否停止(0=否,1=是)。"""
        return self._dll.Motion_CheckDown(handle, c_uint8(iaxis))

    def get_pulse_position(self, handle, iaxis: int) -> int:
        """读取当前坐标（脉冲）（Motion_GetPulsePositon），直接返回位置。"""
        return self._dll.Motion_GetPulsePositon(handle, c_uint8(iaxis))

    def get_encoder_position(self, handle, iaxis: int) -> int:
        """读取编码器位置（Motion_GetEncoderPositon），直接返回位置。"""
        return self._dll.Motion_GetEncoderPositon(handle, c_uint8(iaxis))

    def get_aim_position(self, handle, iaxis: int) -> int:
        """读取目标位置（Motion_GetAimPositon），直接返回位置。"""
        return self._dll.Motion_GetAimPositon(handle, c_uint8(iaxis))

    def set_pulse_position(self, handle, iaxis: int, position: int) -> int:
        """设置当前坐标（脉冲）（Motion_SetPulsePositon）。"""
        return self._dll.Motion_SetPulsePositon(handle, c_uint8(iaxis), c_int32(position))

    def get_cur_speed(self, handle, iaxis: int) -> int:
        """读取当前速度（Motion_GetCurSpeed），直接返回速度。"""
        return self._dll.Motion_GetCurSpeed(handle, c_uint8(iaxis))

    def get_stop_reason(self, handle, iaxis: int) -> int:
        """读取停止原因（Motion_GetStopReason），直接返回原因码。"""
        return self._dll.Motion_GetStopReason(handle, c_uint8(iaxis))

    def home_find_origin(self, handle, iaxis: int) -> int:
        """回零运动（Motion_Home_FindOrigin）。"""
        return self._dll.Motion_Home_FindOrigin(handle, c_uint8(iaxis))

    def home_if_homing(self, handle, iaxis: int) -> int:
        """检查是否回零中（Motion_Home_IfHoming），直接返回是否回零中(0=否,1=是)。"""
        return self._dll.Motion_Home_IfHoming(handle, c_uint8(iaxis))

    def get_in_port(self, handle, port: int) -> tuple:
        """读取单个输入端口（DI）状态。

        Args:
            handle: 连接句柄
            port: 输入端口号（0-based，如 IN1 对应端口 0）

        Returns:
            (错误码, 是否有效/高电平)

        Raises:
            AttributeError: 若 DLL 中未找到读取输入端口的函数
        """
        if self._in_port_func is None:
            raise AttributeError("DLL 中未找到读取输入端口(DI)的函数")
        value = c_uint8()
        err = self._in_port_func(handle, c_uint8(port), ctypes.byref(value))
        return err, bool(value.value)

    def set_out_port(self, handle, port: int, value: int) -> int:
        """设置单个输出端口（DO）状态。

        Args:
            handle: 连接句柄
            port: 输出端口号（0-based）
            value: 0=低电平/关，1=高电平/开

        Returns:
            错误码

        Raises:
            AttributeError: 若 DLL 中未找到设置输出端口的函数
        """
        if self._out_port_func is None:
            raise AttributeError("DLL 中未找到设置输出端口(DO)的函数")
        return self._out_port_func(handle, c_uint8(port), c_uint8(value))