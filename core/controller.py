# -*- coding: utf-8 -*-
"""
controller.py
=============
NMC1400 控制器功能模块。

本模块基于 NMCSDK 封装（MCDLL_NET.dll），提供面向业务层的控制器连接、
断开、状态查询以及运动控制等操作。界面层 (ui) 通过本模块与控制器交互，
不直接接触底层 DLL。

参考：NMCxxxx_编程手册.pdf（深圳市摩升泰科技有限公司）

【与旧 SMC6480 的对应关系】（上层调用代码无需改动）
    旧 SMC6480                          ->  新 NMC1400
    SMCOpenEth(ip)                      ->  MCF_Open_Net(站点数)   （网口自动发现，无需 IP）
    SMCClose(handle)                    ->  MCF_Close_Net()
    MSetting_SetStartSpeed/…Speed/Acc/Dec/S曲线
                                        ->  MCF_Set_Axis_Profile_Net(曲线参数)
    Motion_Pmove_Enter/SetAbsolute/Start
                                        ->  MCF_Uniaxial_Net(轴, 位置, 绝对模式)
    Motion_Pmove_SetRelative            ->  MCF_Uniaxial_Net(轴, 距离, 相对模式)
    Motion_Vmove_*                      ->  MCF_JOG_Net(轴, ±速度, 加速度)
    Motion_DeclStop / Motion_ImdStop    ->  MCF_Axis_Stop_Net(轴, 1 / 0)
    Motion_CheckDown                    ->  MCF_Get_Axis_State_Net(轴) == 0
    Motion_GetPulsePositon              ->  MCF_Get_Position_Net(轴)
    Motion_GetEncoderPositon            ->  MCF_Get_Encoder_Net(轴)
    Motion_SetPulsePositon              ->  MCF_Set_Position_Net(轴, 位置)
    Motion_GetCurSpeed                  ->  MCF_Get_Vel_Net(轴)
    Motion_Home_FindOrigin              ->  MCF_Search_Home_Start_Net(轴)
    Motion_Home_IfHoming                ->  MCF_Search_Home_Get_State_Net(轴) == 32
    SMCReadInBit / SMCWriteOutBit       ->  MCF_Get_Input_Bit_Net / MCF_Set_Output_Bit_Net
"""

import time

from core.nmc_sdk import (
    NMCSDK,
    NMCError,
    SYS_STATE_DESC,
    SYS_STATE_IDLE,
    SYS_STATE_IDLE_AXIS,
    SYS_STATE_MOVING,
    SYS_STATE_STOPPED,
    AXIS_STATE_DESC,
    AXIS_BUSY,
    ERRCODE_DESC,
    SUCCESS,
    get_error_message,
    # 常量
    NMC1400_AXIS_COUNT,
    NMC1400_STATION_TYPE,
    Axis_Stop_IMD,
    Axis_Stop_DEC,
    Position_Absolute,
    Position_Opposite,
    Profile_T,
    Profile_S,
    Servo_Close,
    Servo_Open,
    Switch_State_Series,
    HOME_STATE_SUCCESS,
    HOME_STATE_ERROR,
    HOME_STATE_HOMING,
    Home_Mode_3,
)
from core.log_manager import log_info, log_error, log_warning

# 轴状态锁存寄存器里，这些值属于"正常结束 / 我们主动发的停止命令"，不算异常：
#   0  = 正常执行完成
#   1  = 正在执行（由实时速度另行判断）
#   22 = 命令立即停止（imd_stop）
#   23 = 命令减速停止（decel_stop）
#   28 = 外部 IO 减速停止
NORMAL_STOP_CODES = frozenset({0, 1, 22, 23, 28})

# 单站点（现场仅接入一块 NMC1400 四轴控制卡）
DEFAULT_STATION = 0
DEFAULT_AXIS_COUNT = NMC1400_AXIS_COUNT

# 速度/加速度缺省值（脉冲、脉冲/s、脉冲/s²）
DEFAULT_ACC = 100000.0
DEFAULT_VMAX = 50000.0
INT32_MAX = 2 ** 31 - 1
INT32_MIN = -(2 ** 31)

# 回零参数缺省值（详见编程手册"回原点模式选择参考表"）。
# ⚠️ 现场必须按机械结构/开关安装情况确认 home_mode 与各触发电平后再启用自动回零，
#    模式选错会导致往错误方向找原点。
DEFAULT_HOME_PARAMS = {
    "home_mode": Home_Mode_3,    # 3: 安装原点开关，正方向找原点负边外侧（沿用旧 SMC 配置的数值）
    "limit_logic": 0,            # 正负限位触发电平 0=低电平
    "home_logic": 0,             # 原点开关触发电平 0=低电平
    "index_logic": 0,            # Index(Z相) 触发电平 0=低电平
    "high_speed": 10000.0,       # 高速段速度 pulse/s
    "low_speed": 1000.0,         # 低速段速度 pulse/s
    "offset": 0,                 # 回零偏移量 脉冲
    "trigger_source": 0,         # 0=指令位置 1=编码器位置
}


class ControllerError(Exception):
    """控制器操作异常。"""


class Controller:
    """
    NMC1400 控制器封装类。

    负责管理控制器的连接生命周期（连接 / 断开）、状态查询以及运动控制。
    使用示例：
        ctrl = Controller()
        ctrl.connect()
        print(ctrl.get_state_desc())
        ctrl.set_motion_params(0, 1000, 50000, 100000, 100000, 0)
        ctrl.pmove_abs(0, 1000)
        ctrl.disconnect()
    """

    # ------------------------------------------------------------------
    # 硬件约定（现场若有差异，只改这里）
    # ------------------------------------------------------------------
    # 指示灯：OUT3 红灯、OUT4 绿灯（二合一灯，低电平有效 → 写 0 点亮）
    LIGHT_RED_PORT = 3
    LIGHT_GREEN_PORT = 4

    # 伺服使能电平：手册 3.1 —— "0 表示使能端口输出低电平"，
    # 即 Servo_Close(0)=触点闭合=输出低电平=使能；Servo_Open(1)=触点断开=关闭使能。
    SERVO_ENABLE_LOGIC = Servo_Close      # 0
    SERVO_DISABLE_LOGIC = Servo_Open      # 1

    # DI 有效电平：手册 2.5 —— 读回 0=触点闭合(硬件灯亮，信号有效)，1=触点断开。
    # 而旧 SMC6480 读回的是原始电平，且上层工作流按"上升沿(0→1)=按下"判断，
    # 因此这里默认把 0 反相成 True（有效），保证上层逻辑不变。
    # 现场实测若相反（按下时读回 1），把这里改成 False 即可。
    INPUT_ACTIVE_LOW = True

    # 运动指令在"轴执行中(返回 1)"时的重试次数/间隔。
    # NMC1400 在轴处于"正在执行"状态时会拒绝新的运动命令并返回 1（实测）。
    MOTION_RETRY = 3
    MOTION_RETRY_INTERVAL = 0.1
    # 判定"假执行中"时观察位置/速度的采样间隔（秒）
    STUCK_CHECK_INTERVAL = 0.2
    # 判定轴是否在运动的速度阈值（pulse/s）。小于该值视为已停止。
    MOVE_VEL_THRESHOLD = 1.0

    def __init__(self, dll_path: str = "MCDLL_NET.dll"):
        """
        初始化控制器对象。

        :param dll_path: MCDLL_NET.dll 路径（默认使用与官方一致的 MCDLL_NET.dll）
        :raises ControllerError: 当 DLL 加载失败时抛出
        """
        try:
            self._dll = NMCSDK(dll_path)
            self._dll.load_dll()
            log_info(f"加载 {dll_path} 成功: {self._dll._dll_path}")
        except Exception as exc:  # noqa: BLE001
            raise ControllerError(f"加载 MCDLL_NET.dll 失败: {exc}") from exc

        self._handle = None
        self._connected = False
        self._conn_type = None
        self._conn_string = None
        self._station = DEFAULT_STATION

        # 各轴最近一次下发的加速度（JOG 时若未指定加速度则复用）
        self._axis_acc_cache = {}

        # 指示灯状态（二合一灯：OUT3 红灯、OUT4 绿灯，低电平有效）
        # 记录当前灯状态，避免不必要的重复写
        self._light_red_on = None
        self._light_green_on = None

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        """是否已连接。"""
        return self._connected and self._handle is not None

    @property
    def handle(self):
        """底层连接句柄（NMC1400 无句柄概念，连接成功时为 True）。"""
        return self._handle

    @property
    def conn_string(self):
        """当前连接描述（NMC1400 通过网口自动发现，显示站点信息）。"""
        return self._conn_string

    @property
    def station(self) -> int:
        """当前站点号。"""
        return self._station

    # ------------------------------------------------------------------
    # 连接 / 断开
    # ------------------------------------------------------------------
    def connect(self, timeout_ms: int = 0) -> bool:
        """
        打开（连接）运动控制卡。

        NMC1400 通过网口自动发现设备，不需要指定 IP；只需告诉 DLL
        挂了几块卡以及每块卡的站号/类型（手册 1.2）。

        :param timeout_ms: 链接超时时间（毫秒）。
            **默认 0 = 不设置**，保持控制卡原有配置。
            ⚠️ 不要随意开启：手册 1.3.1 的 `MCF_Set_Link_TimeOut_Net` 是
            "链接超时紧急停止所有轴"，实测一旦触发（程序退出 / 网线抖动 /
            电脑休眠都会触发），控制卡会**锁存**这个急停状态——
            此时运动指令照样返回 0、轴状态显示"正在执行"，但速度/位置恒为 0、
            任何运动都不执行，连厂家官方调试软件也点不动，
            **必须给控制卡断电重启才能恢复**。
            确需该安全功能时再显式传入，例如 connect(5000)。
        :return: 是否连接成功
        :raises ControllerError: 连接失败时抛出
        """
        if self.is_connected:
            raise ControllerError("控制器已连接，请先断开。")

        # 1.1 级联模式：0=串联（单卡默认串联）
        # 注意：该寄存器会保存在卡里，官方样例程序从不写它，这里也不写；
        #       只在打开控制卡之后读取并记录（打开前调用会返回 -18 未打开站点）。
        err = self._dll.open_net(
            1,
            [DEFAULT_STATION],
            [NMC1400_STATION_TYPE],
        )
        if err != SUCCESS:
            raise ControllerError(
                f"打开 NMC1400 控制卡失败: {get_error_message(err)} (错误码 {err})\n"
                f"请检查网线连接、控制卡供电，以及电脑网卡是否与控制卡同网段。"
            )

        self._connected = True
        self._handle = True
        self._conn_type = "NET"
        self._conn_string = f"NMC1400 (站号 {DEFAULT_STATION})"

        # 打开后读取网络级联模式（仅记录，便于排查串联/并联接法）
        try:
            switch_state = self._dll.get_switch_state(self._station)
            if switch_state in (0, 1):
                log_info(f"控制卡网络级联模式: {switch_state} "
                         f"({('串联' if switch_state == 0 else '并联')})")
            else:
                log_warning(f"读取网络级联模式失败: {get_error_message(switch_state)}")
        except Exception as exc:  # noqa: BLE001
            log_warning(f"读取网络级联模式失败（忽略）: {exc}")

        # 1.3.1 链接超时紧急停止：超时后停止所有轴（安全功能）。
        # 传 0 表示**不设置**（保持控制卡原有配置）——官方样例程序不调用该函数；
        # 如果现场发现"卡能读能写但轴就是不走"，可以试 connect(timeout_ms=0) 排除此项。
        if int(timeout_ms) > 0:
            try:
                self._dll.set_link_timeout(int(timeout_ms), 0, self._station)
                try:
                    count = self._dll.get_link_timeout_count(self._station)
                    if isinstance(count, int) and count > 0:
                        log_warning(f"控制卡历史链接中断次数: {count}")
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                log_warning(f"设置链接超时失败（忽略）: {exc}")
        else:
            log_info("未设置链接超时（保持控制卡原有配置）")

        log_info(
            f"NMC1400 控制卡已连接 "
            f"(站号 {DEFAULT_STATION}, 轴数 {self.get_axises()}, "
            f"链接超时 {timeout_ms if int(timeout_ms) > 0 else '未设置'})"
        )
        return True

    def disconnect(self) -> bool:
        """
        断开与控制器的连接。

        关闭前会先把"链接超时紧急停止"关掉（写 0 = 不设置）。
        否则一旦断链，控制卡会锁存这个急停状态，必须断电重启才能恢复
        （详见 connect() 的说明）。

        :return: 是否成功断开
        """
        if self._dll is not None and self._dll.is_open():
            # 先解除链接超时看门狗，避免断开瞬间触发卡上的锁存急停
            try:
                self._dll.set_link_timeout(0, 0, self._station)
                log_info("已解除链接超时看门狗")
            except Exception as exc:  # noqa: BLE001
                log_warning(f"解除链接超时看门狗失败（忽略）: {exc}")
            try:
                self._dll.close_net()
            except Exception as exc:  # noqa: BLE001
                log_error(f"关闭控制卡失败: {exc}")
        self._handle = None
        self._connected = False
        self._conn_type = None
        self._conn_string = None
        log_info("NMC1400 控制卡已断开")
        return True

    def check_link(self) -> bool:
        """链接监测（手册 1.4）：True=链接正常。"""
        if not self.is_connected:
            return False
        try:
            return self._dll.get_link_state(self._station) == SUCCESS
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def get_state(self) -> int:
        """
        读取控制器整体状态（由 4 个轴的**实时**状态归纳）。

        :return: SYS_STATE_IDLE / SYS_STATE_MOVING / SYS_STATE_STOPPED
        :raises ControllerError: 未连接或读取失败时抛出
        """
        self._check_connected()

        any_moving = False
        any_abnormal = False
        for iaxis in range(DEFAULT_AXIS_COUNT):
            status = self.get_axis_status(iaxis)
            if status["moving"]:
                any_moving = True
            elif status["raw_code"] not in NORMAL_STOP_CODES:
                any_abnormal = True

        if any_moving:
            return SYS_STATE_MOVING
        if any_abnormal:
            return SYS_STATE_STOPPED
        return SYS_STATE_IDLE

    def get_state_desc(self) -> str:
        """读取控制器状态的中文描述。"""
        state = self.get_state()
        return SYS_STATE_DESC.get(state, f"未知状态({state})")

    def get_axises(self) -> int:
        """
        读取控制器轴数。

        :return: 轴数（NMC1400 为 4 轴），未连接返回 0
        """
        if not self.is_connected:
            return 0
        return DEFAULT_AXIS_COUNT

    def get_version(self) -> int:
        """读取控制卡固件版本号。"""
        self._check_connected()
        return self._dll.get_version(self._station)

    def get_serial_number(self) -> int:
        """读取控制卡序列号。"""
        self._check_connected()
        return self._dll.get_serial_number(self._station)

    def get_run_time(self) -> int:
        """读取控制卡运行时间（秒）。"""
        self._check_connected()
        return self._dll.get_run_time(self._station)

    # ------------------------------------------------------------------
    # 运动参数设置（单轴曲线参数）
    # ------------------------------------------------------------------
    def set_motion_params(
        self,
        iaxis: int,
        start_speed: float,
        max_speed: float,
        acc: float,
        dec: float,
        s_curve: float = 0.0,
    ) -> None:
        """
        下发当前轴的运动曲线参数（对应旧 SMC 的 MSetting_* 系列）。

        NMC1400 用一个函数完成设置：
            MCF_Set_Axis_Profile_Net(轴, 启动速度, 目标速度, 加速度, 加加速度,
                                     终止速度, 曲线类型)
            MCF_Set_Axis_Stop_Profile_Net(轴, 减速度, 加加速度, 曲线类型)

        参数单位与旧卡一致：速度 pulse/s，加速度 pulse/s²。

        :param iaxis: 轴号（0-3）
        :param start_speed: 启动速度（pulse/s，需小于目标速度）
        :param max_speed: 目标速度（pulse/s）
        :param acc: 加速度（pulse/s²）
        :param dec: 减速度（pulse/s²，用于减速停止）
        :param s_curve: S 曲线过渡时间（秒）；>0 时用 S 形曲线，=0 用 T 形曲线
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()

        v_max = float(max_speed)
        if v_max <= 0:
            raise ControllerError(f"目标速度必须大于 0 (轴{iaxis}, 当前 {max_speed})")

        # 手册：dMaxV > dV_ini >= 0
        v_ini = float(start_speed)
        if v_ini < 0:
            v_ini = 0.0
        if v_ini >= v_max:
            v_ini = v_max * 0.1

        a_max = float(acc)
        if a_max <= 0:
            a_max = max(v_max * 10.0, 1000.0)

        d_max = float(dec)
        if d_max <= 0:
            d_max = a_max

        if s_curve and float(s_curve) > 0:
            # S 形曲线：加加速度 = 加速度 / 过渡时间（保证加速度在 s_curve 秒内升到最大）
            profile = Profile_S
            jerk = max(a_max / float(s_curve), 1.0)
        else:
            # T 形曲线：加加速度不参与规划（手册要求 > 0，给一个正值即可）
            profile = Profile_T
            jerk = max(a_max, 1.0)

        err = self._dll.set_axis_profile(
            iaxis, v_ini, v_max, a_max, jerk, 0.0, profile, self._station
        )
        if err != SUCCESS:
            raise ControllerError(
                f"设置运动曲线参数失败 (轴{iaxis}): {get_error_message(err)}"
            )

        err = self._dll.set_axis_stop_profile(iaxis, d_max, jerk, profile, self._station)
        if err != SUCCESS:
            log_warning(f"设置减速停止曲线失败 (轴{iaxis}): {get_error_message(err)}")

        self._axis_acc_cache[iaxis] = a_max
        log_info(
            f"轴{iaxis} 运动参数: 启动={v_ini:.0f} 目标={v_max:.0f} "
            f"加速={a_max:.0f} 减速={d_max:.0f} "
            f"曲线={'S' if profile == Profile_S else 'T'}"
        )

    def get_motion_params(self, iaxis: int) -> dict:
        """
        读取当前轴的运动参数。

        :param iaxis: 轴号（0-3）
        :return: dict，包含 start_speed / max_speed / acc / dec / s_curve / profile
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        profile = self._dll.get_axis_profile(iaxis, self._station)
        try:
            stop_profile = self._dll.get_axis_stop_profile(iaxis, self._station)
        except NMCError:
            stop_profile = {"a_max": profile["a_max"], "jerk": profile["jerk"],
                            "profile": profile["profile"]}
        return {
            "start_speed": profile["v_ini"],
            "max_speed": profile["v_max"],
            "acc": profile["a_max"],
            "dec": stop_profile["a_max"],
            "s_curve": 0.0,
            "jerk": profile["jerk"],
            "profile": profile["profile"],
        }

    # ------------------------------------------------------------------
    # 点位运动
    # ------------------------------------------------------------------
    def pmove_abs(self, iaxis: int, pos: float) -> None:
        """
        绝对定位。

        对应旧调用序列：Motion_Pmove_Enter + SetAbsolute + Start，
        新卡只需一次 MCF_Uniaxial_Net(轴, 位置, Position_Absolute)。

        :param iaxis: 轴号（0-3）
        :param pos: 绝对目标位置（脉冲，int32 范围）
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        target = self._to_int32(pos, iaxis, "绝对位置")

        # 目标就是当前位置时不下发运动指令。
        # 实测：给 NMC1400 下发"0 距离"点位运动后，该轴会被永久保持在
        # "正在执行(1)" 状态（Axis_Stop 也解不开），之后所有运动命令都被拒绝。
        current = self._dll.get_position(iaxis, self._station)
        if target == current:
            log_info(f"轴{iaxis} 已在目标位置 {target}，跳过下发运动指令")
            return

        self._run_motion_command(
            iaxis, f"绝对定位到 {target}",
            lambda: self._dll.uniaxial(iaxis, target, Position_Absolute, self._station))

    def pmove_rel(self, iaxis: int, dist: float) -> None:
        """
        相对定位（MCF_Uniaxial_Net + Position_Opposite）。

        :param iaxis: 轴号（0-3）
        :param dist: 相对移动距离（脉冲，int32 范围）
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        delta = self._to_int32(dist, iaxis, "相对距离")
        if delta == 0:
            log_info(f"轴{iaxis} 相对距离为 0，跳过下发运动指令")
            return

        self._run_motion_command(
            iaxis, f"相对移动 {delta}",
            lambda: self._dll.uniaxial(iaxis, delta, Position_Opposite, self._station))

    # ------------------------------------------------------------------
    # 定速运动（JOG）
    # ------------------------------------------------------------------
    def vmove(self, iaxis: int, positive: bool = True, speed: float = 0.0,
              acc: float = None) -> None:
        """
        定速（JOG）运动。

        对应旧调用序列：Motion_Vmove_Enter + SetDir + SetSpeed + Start，
        新卡只需一次 MCF_JOG_Net(轴, ±速度, 加速度)；速度为 0 表示停止速度环。

        :param iaxis: 轴号（0-3）
        :param positive: 是否正向移动
        :param speed: 速度（pulse/s）
        :param acc: 加速度（pulse/s²），None 表示复用最近一次下发的加速度
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        v = abs(float(speed))
        if v <= 0:
            self.imd_stop(iaxis)
            return

        a = float(acc) if acc and float(acc) > 0 else self._get_acc(iaxis)
        direction = 1 if positive else -1
        self._run_motion_command(
            iaxis, f"JOG 运动 (速度={v:.0f})",
            lambda: self._dll.jog(iaxis, direction * v, a, self._station))

    # ------------------------------------------------------------------
    # 停止 / 状态 / 坐标
    # ------------------------------------------------------------------
    def check_down(self, iaxis: int) -> bool:
        """
        检查轴是否已停止（对应旧 Motion_CheckDown）。

        ⚠️ 不能用 MCF_Get_Axis_State_Net 直接判断：
        该寄存器是**停止原因锁存寄存器**（手册 5.10），运动正常结束后不一定刷新，
        因此这里用实时速度来判断轴是否真的停了。

        :param iaxis: 轴号（0-3）
        :return: 是否已停止
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return not self.get_axis_status(iaxis)["moving"]

    def get_axis_status(self, iaxis: int) -> dict:
        """读取轴的**实时**状态（推荐用它做界面显示 / 到位判断）。

        NMC1400 的 `MCF_Get_Axis_State_Net` 读的是"停止原因锁存寄存器"：
        正常运动结束后它不一定会刷新（手册 5.10：需要清除函数、新动作或新报警才会刷新），
        所以直接用它判断"是否在运动"会得到过期结果。
        这里改为：
            是否在运动 → 看实时速度（命令速度 / 编码器速度）
            停止原因   → 锁存寄存器非 0/1 时给出中文原因

        :param iaxis: 轴号（0-3）
        :return: dict(code, raw_code, desc, moving, cmd_speed, enc_speed)
        """
        self._check_connected()
        raw = self._dll.get_axis_state(iaxis, self._station)
        cmd_speed, enc_speed = self._dll.get_velocity(iaxis, self._station)
        moving = (abs(cmd_speed) > self.MOVE_VEL_THRESHOLD
                  or abs(enc_speed) > self.MOVE_VEL_THRESHOLD)

        if moving:
            code = AXIS_BUSY
            desc = "运动中"
        elif raw == AXIS_BUSY:
            # 实测：正常运动结束后，这个锁存寄存器也会一直停在 1(正在执行)，
            # 直到调用 MCF_Clear_Axis_State_Net。速度已经是 0，所以按"空闲"显示
            # （原始码会写进日志，便于排查）。
            code = SYS_STATE_IDLE_AXIS
            desc = "空闲"
        else:
            code = raw
            desc = AXIS_STATE_DESC.get(raw, f"未知状态({raw})")

        return {
            "code": code,
            "raw_code": raw,
            "desc": desc,
            "moving": moving,
            "cmd_speed": cmd_speed,
            "enc_speed": enc_speed,
        }

    def get_axis_state(self, iaxis: int) -> int:
        """
        读取轴状态寄存器（锁存值）的原始码。

        ⚠️ 这是"停止原因锁存寄存器"，不保证实时；要判断是否在运动请用
        check_down() / get_axis_status()。

        :param iaxis: 轴号（0-3）
        :return: 0=空闲, 1=正在执行, 2~28=停止原因, 31=回零错误, 32=正在回零
        """
        self._check_connected()
        return self._dll.get_axis_state(iaxis, self._station)

    def get_axis_state_desc(self, iaxis: int) -> str:
        """读取轴状态的中文描述（实时状态，见 get_axis_status）。"""
        return self.get_axis_status(iaxis)["desc"]

    def decel_stop(self, iaxis: int) -> None:
        """减速停止（MCF_Axis_Stop_Net + Axis_Stop_DEC）。"""
        self._check_connected()
        err = self._dll.axis_stop(iaxis, Axis_Stop_DEC, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"减速停止失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def imd_stop(self, iaxis: int) -> None:
        """立即停止（MCF_Axis_Stop_Net + Axis_Stop_IMD）。"""
        self._check_connected()
        err = self._dll.axis_stop(iaxis, Axis_Stop_IMD, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"立即停止失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def stop_all(self) -> None:
        """立即停止所有轴（异常兜底，不抛异常）。"""
        if not self.is_connected:
            return
        for iaxis in range(DEFAULT_AXIS_COUNT):
            try:
                self._dll.axis_stop(iaxis, Axis_Stop_IMD, self._station)
            except Exception:  # noqa: BLE001
                pass

    def clear_axis_state(self, iaxis: int) -> None:
        """清除轴报警/停止原因寄存器（MCF_Clear_Axis_State_Net）。"""
        self._check_connected()
        err = self._dll.clear_axis_state(iaxis, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"清除轴状态失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def get_pulse_position(self, iaxis: int) -> int:
        """
        读取当前规划坐标（脉冲）。

        :param iaxis: 轴号（0-3）
        :return: 坐标，单位 脉冲
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return self._dll.get_position(iaxis, self._station)

    def get_encoder_position(self, iaxis: int) -> int:
        """
        读取编码器位置（脉冲）。

        :param iaxis: 轴号（0-3）
        :return: 编码器位置，单位 脉冲
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return self._dll.get_encoder(iaxis, self._station)

    def set_pulse_position(self, iaxis: int, position: int) -> None:
        """
        设置当前规划坐标（MCF_Set_Position_Net）。

        :param iaxis: 轴号（0-3）
        :param position: 坐标，单位 脉冲
        """
        self._check_connected()
        err = self._dll.set_position(iaxis, self._to_int32(position, iaxis, "坐标"),
                                     self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"设置坐标(脉冲)失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def set_encoder_position(self, iaxis: int, position: int) -> None:
        """设置编码器位置（脉冲）。"""
        self._check_connected()
        err = self._dll.set_encoder(iaxis, self._to_int32(position, iaxis, "编码器值"),
                                    self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"设置编码器位置失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def get_cur_speed(self, iaxis: int) -> float:
        """
        读取当前命令速度（pulse/s）。

        :param iaxis: 轴号（0-3）
        :return: 速度（指令速度）
        """
        self._check_connected()
        cmd_vel, _enc_vel = self._dll.get_velocity(iaxis, self._station)
        return cmd_vel

    def get_encoder_speed(self, iaxis: int) -> float:
        """读取当前编码器速度（pulse/s）。"""
        self._check_connected()
        _cmd_vel, enc_vel = self._dll.get_velocity(iaxis, self._station)
        return enc_vel

    # ------------------------------------------------------------------
    # 回零
    # ------------------------------------------------------------------
    def set_home_params(
        self,
        iaxis: int,
        home_mode: int = Home_Mode_3,
        limit_logic: int = 0,
        home_logic: int = 0,
        index_logic: int = 0,
        high_speed: float = 10000.0,
        low_speed: float = 1000.0,
        offset: int = 0,
        trigger_source: int = 0,
        stop_time_ms: int = 0,
    ) -> None:
        """设置回零参数（MCF_Search_Home_Set_Net，手册 6.1）。

        必须在 home_move() 之前调用；回零方式必须按现场机械/传感器安装情况，
        从编程手册"回原点模式选择参考表"中选择。

        :param iaxis: 轴号（0-3）
        :param home_mode: 回零方式 1~35（如 3/19/20=用原点开关，17/18=用限位开关，33/34=用 Index）
        :param limit_logic: 正负限位触发电平 0=低电平 1=高电平
        :param home_logic: 原点开关触发电平 0=低电平 1=高电平
        :param index_logic: Index(Z相) 触发电平 0=低电平 1=高电平
        :param high_speed: 高速段速度（脉冲/s）
        :param low_speed: 低速段速度（脉冲/s，精确定位用）
        :param offset: 回零完成后的偏移量（脉冲）
        :param trigger_source: 捕捉位置模式 0=指令位置 1=编码器位置
        :param stop_time_ms: 碰撞原点缓停时间（ms，0=急停）；需在设置回零参数前调用
        """
        self._check_connected()

        high_speed = float(high_speed)
        low_speed = float(low_speed)
        if high_speed <= 0:
            high_speed = max(low_speed, 10000.0)
        if low_speed <= 0:
            low_speed = max(high_speed / 10.0, 100.0)
        if low_speed > high_speed:
            low_speed = high_speed / 10.0

        if stop_time_ms and int(stop_time_ms) > 0:
            try:
                self._dll.search_home_stop_time(iaxis, int(stop_time_ms), self._station)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"设置回零缓停时间失败（忽略）: {exc}")

        err = self._dll.search_home_set(
            iaxis,
            int(home_mode),
            int(limit_logic),
            int(home_logic),
            int(index_logic),
            high_speed,
            low_speed,
            int(offset),
            int(trigger_source),
            self._station,
        )
        if err != SUCCESS:
            raise ControllerError(
                f"设置回零参数失败 (轴{iaxis}, 模式{home_mode}): {get_error_message(err)}"
            )
        log_info(
            f"轴{iaxis} 回零参数已设置: 模式={home_mode} 高速段={high_speed:.0f} "
            f"低速段={low_speed:.0f} 偏移={offset} 原点电平={'低' if home_logic == 0 else '高'}"
        )

    def home_move(self, iaxis: int) -> None:
        """
        启动回零（MCF_Search_Home_Start_Net）。

        :param iaxis: 轴号（0-3）
        """
        self._check_connected()
        err = self._dll.search_home_start(iaxis, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"回零运动失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def home_stop(self, iaxis: int) -> None:
        """停止回零（MCF_Search_Home_Stop_Net）。"""
        self._check_connected()
        err = self._dll.search_home_stop(iaxis, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"停止回零失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def get_home_state(self, iaxis: int) -> int:
        """
        读取回零状态：0=回零成功, 31=回零错误, 32=正在回零点。
        """
        self._check_connected()
        return self._dll.search_home_get_state(iaxis, self._station)

    def if_home_moving(self, iaxis: int) -> bool:
        """
        检查是否正在回零中（对应旧 Motion_Home_IfHoming）。

        :param iaxis: 轴号（0-3）
        :return: 是否回零中（状态 32）
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return self.get_home_state(iaxis) == HOME_STATE_HOMING

    def is_home_success(self, iaxis: int) -> bool:
        """回零是否已成功完成（状态 0）。"""
        return self.get_home_state(iaxis) == HOME_STATE_SUCCESS

    def is_home_error(self, iaxis: int) -> bool:
        """回零是否报错（状态 31）。"""
        return self.get_home_state(iaxis) == HOME_STATE_ERROR

    # ------------------------------------------------------------------
    # 伺服 / 限位
    # ------------------------------------------------------------------
    def set_servo_enable(self, iaxis: int, enable: bool) -> None:
        """伺服使能/关闭（手册 3.1：0=触点闭合=输出低电平=使能）。

        :param iaxis: 轴号（0-3）
        :param enable: True=使能，False=关闭使能
        """
        self._check_connected()
        logic = self.SERVO_ENABLE_LOGIC if enable else self.SERVO_DISABLE_LOGIC
        err = self._dll.set_servo_enable(iaxis, logic, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"伺服{'使能' if enable else '关闭'}失败 (轴{iaxis}): {get_error_message(err)}"
            )
        log_info(f"轴{iaxis} 伺服{'使能' if enable else '关闭'} (Servo_Logic={logic})")

    def get_servo_enable(self, iaxis: int) -> int:
        """读取伺服使能电平：0=触点闭合, 1=触点断开。"""
        self._check_connected()
        return self._dll.get_servo_enable(iaxis, self._station)

    def get_servo_alarm(self, iaxis: int) -> int:
        """读取伺服报警电平：0=触点闭合, 1=触点断开。"""
        self._check_connected()
        return self._dll.get_servo_alarm(iaxis, self._station)

    def reset_servo_alarm(self, iaxis: int) -> None:
        """伺服报警复位（输出复位电平）。"""
        self._check_connected()
        err = self._dll.set_servo_alarm_reset(iaxis, Servo_Close, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"伺服报警复位失败 (轴{iaxis}): {get_error_message(err)}"
            )

    def set_soft_limit(self, iaxis: int, positive: int, negative: int,
                       enable: bool = True) -> None:
        """设置软件限位并选择是否使能（MCF_Set_Soft_Limit[_Enable]_Net）。"""
        self._check_connected()
        err = self._dll.set_soft_limit(
            iaxis, self._to_int32(positive, iaxis, "正限位"),
            self._to_int32(negative, iaxis, "负限位"), self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"设置软件限位失败 (轴{iaxis}): {get_error_message(err)}"
            )
        err = self._dll.set_soft_limit_enable(iaxis, 1 if enable else 0, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"软件限位使能失败 (轴{iaxis}): {get_error_message(err)}"
            )
        log_info(f"轴{iaxis} 软件限位: 正={positive} 负={negative} 使能={enable}")

    def get_soft_limit(self, iaxis: int) -> tuple:
        """读取软件限位: (正限位, 负限位, 是否使能)"""
        self._check_connected()
        pos, neg = self._dll.get_soft_limit(iaxis, self._station)
        enable = self._dll.get_soft_limit_enable(iaxis, self._station)
        return pos, neg, bool(enable)

    # ------------------------------------------------------------------
    # 兼容方法（供旧代码 / demo 使用）
    # ------------------------------------------------------------------
    def p_move_pluses(self, iaxis: int, length: int, if_abs: bool = False) -> None:
        """
        定长运动（脉冲）兼容方法。

        :param iaxis: 轴号（0-3）
        :param length: 距离，单位 脉冲
        :param if_abs: 是否绝对移动
        """
        if if_abs:
            self.pmove_abs(iaxis, float(length))
        else:
            self.pmove_rel(iaxis, float(length))

    def get_position_pulses(self, iaxis: int) -> int:
        """读取当前坐标（脉冲）兼容方法。"""
        return self.get_pulse_position(iaxis)

    def set_position_pulses(self, iaxis: int, position: int) -> None:
        """设置当前坐标（脉冲）兼容方法。"""
        self.set_pulse_position(iaxis, position)

    # ------------------------------------------------------------------
    # IO 端口（DI/DO）
    # ------------------------------------------------------------------
    def read_in_port(self, port: int) -> bool:
        """读取单个输入端口（DI）状态。

        NMC1400 有 DI00-DI15 共 16 路输入，端口号 0-based（IN1 对应端口 0）。

        Args:
            port: 输入端口号（0-based）

        Returns:
            bool: 该输入端口是否"有效"（触点闭合/按钮按下）。
                  受 INPUT_ACTIVE_LOW 控制：True 表示读回 0 视为有效。

        Raises:
            ControllerError: 未连接或读取失败时抛出
        """
        self._check_connected()
        raw = self._dll.get_input_bit(port, self._station)
        return bool(raw == 0) if self.INPUT_ACTIVE_LOW else bool(raw == 1)

    def read_in_port_raw(self, port: int) -> int:
        """读取单个输入端口（DI）的原始电平：0=触点闭合(硬件灯亮), 1=触点断开。"""
        self._check_connected()
        return self._dll.get_input_bit(port, self._station)

    def read_all_in_ports(self) -> int:
        """读取全部数字输入的原始电平字（DI00-DI47 位图）。"""
        self._check_connected()
        return self._dll.get_input(self._station)

    def set_out_port(self, port: int, value: bool) -> None:
        """设置单个输出端口（DO）状态。

        NMC1400 有 DO00-DO15 共 16 路输出，端口号 0-based。

        Args:
            port: 输出端口号（0-based，如 OUT1 对应端口 0）
            value: True=写 1（触点断开/硬件灯灭），False=写 0（触点闭合/硬件灯亮）

        Raises:
            ControllerError: 未连接或设置失败时抛出
        """
        self._check_connected()
        err = self._dll.set_output_bit(port, 1 if value else 0, self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"设置输出端口失败 (OUT{port + 1}): {get_error_message(err)}"
            )

    def set_out_port_timed(self, port: int, value: bool, hold_ms: int) -> None:
        """设置单个输出端口并保持一段时间（ms）。"""
        self._check_connected()
        err = self._dll.set_output_time_bit(
            port, 1 if value else 0, int(hold_ms), self._station)
        if err != SUCCESS:
            raise ControllerError(
                f"设置输出端口(定时)失败 (OUT{port + 1}): {get_error_message(err)}"
            )

    # ------------------------------------------------------------------
    # 指示灯控制（二合一灯：OUT3 红灯、OUT4 绿灯，低电平有效）
    # ------------------------------------------------------------------
    # 硬件说明：
    #   - OUT3 对应红灯，OUT4 对应绿灯
    #   - 低电平有效：写 0 点亮，写 1 熄灭
    #   - 红灯 + 绿灯同时导通（都写 0）时，物理上显示黄灯
    #   - 灭灯时需要将两个位都写 1
    # 注意：set_out_port(port, value) 中 value=True 写 1（高电平/触点断开），
    #       value=False 写 0（低电平/触点闭合）。因此点亮灯需传 False，熄灭传 True。

    def set_light_state(self, red_on: bool, green_on: bool) -> None:
        """统一控制红/绿指示灯状态（二合一灯，低电平有效）。

        红灯 + 绿灯同时点亮时物理上显示黄灯。
        内部管理 OUT3/OUT4 的写入，避免不必要的重复写，并输出日志。

        Args:
            red_on: True=点亮红灯，False=熄灭红灯
            green_on: True=点亮绿灯，False=熄灭绿灯
        """
        if not self.is_connected:
            return

        # 低电平有效：点亮传 False（写 0），熄灭传 True（写 1）
        red_value = not red_on
        green_value = not green_on

        changed = []
        # 避免不必要的重复写（仅状态变化时写入）
        if red_value != self._light_red_on:
            try:
                self.set_out_port(self.LIGHT_RED_PORT, red_value)
                self._light_red_on = red_value
                changed.append(f"红={red_on}")
            except ControllerError as e:
                log_error(f"设置红灯失败: {e}")
        if green_value != self._light_green_on:
            try:
                self.set_out_port(self.LIGHT_GREEN_PORT, green_value)
                self._light_green_on = green_value
                changed.append(f"绿={green_on}")
            except ControllerError as e:
                log_error(f"设置绿灯失败: {e}")

        # 输出日志（仅状态变化时）
        if changed:
            if red_on and green_on:
                state_desc = "黄灯"
            elif red_on:
                state_desc = "红灯"
            elif green_on:
                state_desc = "绿灯"
            else:
                state_desc = "灭灯"
            log_info(f"指示灯状态: {state_desc} (红={red_on}, 绿={green_on})")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _check_connected(self):
        """检查是否已连接，未连接则抛出异常。"""
        if not self.is_connected:
            raise ControllerError("控制器未连接，请先连接控制卡。")

    def _run_motion_command(self, iaxis: int, action: str, sender) -> None:
        """下发运动命令，并按 NMC1400 的返回值语义处理结果。

        ⚠️ 运动函数的返回值与其它函数不同（实测 + 手册"函数返回值"表）：
            0       命令已下发成功
            1       轴正在执行 → **命令被拒绝**，需要等轴空闲后重试
            2~28    轴因急停/报警/限位等原因停止 → 命令被拒绝
            <0      DLL 错误码

        :param iaxis: 轴号
        :param action: 用于报错的描述
        :param sender: 无参 callable，内部调用具体的 MCF_*_Net 并返回 int
        :raises ControllerError: 轴始终无法接受命令时抛出
        """
        last = None
        total = self.MOTION_RETRY + 1
        for attempt in range(total):
            ret = sender()
            if ret == SUCCESS:
                # 命令被接受后立刻回读一次卡的内部状态，便于现场排查
                # （日志里能看到"命令已下发，但卡里速度/位置没动"这种关键信息）
                try:
                    raw = self._dll.get_axis_state(iaxis, self._station)
                    cmd_v, enc_v = self._dll.get_velocity(iaxis, self._station)
                    pos = self._dll.get_position(iaxis, self._station)
                    log_info(
                        f"{action}: 命令已下发(轴{iaxis})，控制卡状态="
                        f"{raw}({AXIS_STATE_DESC.get(raw, '未知')}) "
                        f"速度={cmd_v:.0f}/{enc_v:.0f} 规划位置={pos}"
                    )
                except Exception:  # noqa: BLE001
                    pass
                return
            last = ret
            if ret != AXIS_BUSY:
                # 其它正数 / 负数：命令确实没能执行
                raise ControllerError(
                    f"{action}失败 (轴{iaxis}): {get_error_message(ret)}"
                )
            # 轴报告"正在执行"：等它闲下来再试；最后一次尝试前先把"假执行中"复位
            time.sleep(self.MOTION_RETRY_INTERVAL)
            if attempt == total - 2:
                self._reset_stuck_axis(iaxis)

        raise ControllerError(
            f"{action}失败 (轴{iaxis}): 轴一直处于"
            f"「{AXIS_STATE_DESC.get(last, f'状态 {last}')}」状态，命令未生效。"
            f"请检查该轴是否卡在运动中，或触发了限位/报警。"
        )

    def _reset_stuck_axis(self, iaxis: int) -> bool:
        """把"假执行中"的轴复位。

        实测现象：给 NMC1400 下发"0 距离"点位运动后，轴会被永久保持在
        "正在执行(1)" 状态（此时位置、速度都完全不动），并且 Axis_Stop 也解不开，
        后续所有运动命令都会被拒绝（返回 1）。此种情况下只有
        MCF_Clear_Axis_State_Net 能把轴状态复位。

        为避免误伤"真的在运动"的轴，这里先观察一小段时间的位置/速度，
        确认完全没有动作才清除状态。

        :return: 是否已复位（或本来就空闲）
        """
        try:
            if self._dll.get_axis_state(iaxis, self._station) != AXIS_BUSY:
                return True

            p0 = self._dll.get_position(iaxis, self._station)
            v0 = self._dll.get_velocity(iaxis, self._station)
            time.sleep(self.STUCK_CHECK_INTERVAL)
            p1 = self._dll.get_position(iaxis, self._station)
            v1 = self._dll.get_velocity(iaxis, self._station)

            moving = (p0 != p1) or abs(v0[0]) > 1 or abs(v1[0]) > 1 or abs(v1[1]) > 1
            if moving:
                log_warning(f"轴{iaxis} 正在运动（位置/速度有变化），不清除其状态")
                return False

            err = self._dll.clear_axis_state(iaxis, self._station)
            log_warning(
                f"轴{iaxis} 报「正在执行」但位置({p0})与速度(0)均无变化，"
                f"已清除轴状态复位 (ret={err})"
            )
            return err == SUCCESS
        except Exception as exc:  # noqa: BLE001
            log_warning(f"复位轴{iaxis}状态失败: {exc}")
            return False

    def _get_acc(self, iaxis: int) -> float:
        """取该轴用于 JOG 的加速度：优先用缓存，其次读卡，最后用缺省值。"""
        acc = self._axis_acc_cache.get(iaxis)
        if acc and acc > 0:
            return float(acc)
        try:
            profile = self._dll.get_axis_profile(iaxis, self._station)
            acc = float(profile.get("a_max") or 0.0)
        except Exception:  # noqa: BLE001
            acc = 0.0
        if acc <= 0:
            acc = DEFAULT_ACC
        self._axis_acc_cache[iaxis] = acc
        return acc

    @staticmethod
    def _to_int32(value, iaxis: int, what: str) -> int:
        """把位置/距离转换成 NMC1400 要求的 int32（MCF_Uniaxial_Net 的距离是 32 位整数）。"""
        try:
            ivalue = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"{what}不是有效数值 (轴{iaxis}): {value!r}") from exc
        if ivalue > INT32_MAX or ivalue < INT32_MIN:
            raise ControllerError(
                f"{what}超出控制卡范围 (轴{iaxis}): {ivalue} "
                f"(允许 {INT32_MIN} ~ {INT32_MAX} 脉冲)"
            )
        return ivalue
