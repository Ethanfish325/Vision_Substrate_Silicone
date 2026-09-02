# -*- coding: utf-8 -*-
"""
controller.py
=============
SMC6480 控制器功能模块。

本模块基于 smcsh_dll 封装（smcsh_mbs.dll），提供面向业务层的控制器连接、
断开、状态查询以及运动控制等操作。界面层 (ui) 通过本模块与控制器交互，
不直接接触底层 DLL。

运动控制采用与官方程序一致的 MSetting_* 与 Motion_* 函数调用序列
（例如 MSetting_SetStartSpeed、Motion_Pmove_Enter 等）。

参考：雷赛运动 SMC-6480 控制器软件手册 第五章
"""

from core.smcsh_dll import (
    SMCSHDLL,
    SMC6X_CONNECTION_COM,
    SMC6X_CONNECTION_ETH,
    SYS_STATE_DESC,
    ERR_NOERR,
)
from core.log_manager import log_info, log_error


class ControllerError(Exception):
    """控制器操作异常。"""


class Controller:
    """
    SMC6480 控制器封装类。

    负责管理控制器的连接生命周期（连接 / 断开）、状态查询以及运动控制。
    使用示例：
        ctrl = Controller()
        ctrl.connect_eth("192.168.1.11")
        print(ctrl.get_state_desc())
        ctrl.set_motion_params(0, 800, 10000, 200000, 200000, 0)
        ctrl.pmove_abs(0, 1000)
        ctrl.disconnect()
    """

    def __init__(self, dll_path: str = "smcsh_mbs.dll"):
        """
        初始化控制器对象。

        :param dll_path: smcsh_mbs.dll 路径（默认使用与官方一致的 smcsh_mbs.dll）
        :raises ControllerError: 当 DLL 加载失败时抛出
        """
        try:
            self._dll = SMCSHDLL(dll_path)
        except Exception as exc:  # noqa: BLE001
            raise ControllerError(f"加载 smcsh_mbs.dll 失败: {exc}") from exc

        self._handle = None
        self._connected = False
        self._conn_type = None
        self._conn_string = None

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
        """底层连接句柄。"""
        return self._handle

    @property
    def conn_string(self):
        """当前连接字符串（IP 或 COM 口）。"""
        return self._conn_string

    # ------------------------------------------------------------------
    # 连接 / 断开
    # ------------------------------------------------------------------
    def connect_eth(self, ip_addr: str, timeout_ms: int = 1000) -> bool:
        """
        通过以太网连接控制器。

        :param ip_addr: 控制器 IP 地址，如 "192.168.1.11"
        :param timeout_ms: 命令应答超时时间（毫秒）
        :return: 是否连接成功
        :raises ControllerError: 连接失败时抛出
        """
        if self.is_connected:
            raise ControllerError("控制器已连接，请先断开。")

        err, handle = self._dll.open_eth(ip_addr)
        if err != ERR_NOERR:
            raise ControllerError(
                f"以太网连接失败 (IP={ip_addr}): {self._dll.get_errcode_description(err)}"
            )

        self._handle = handle
        self._connected = True
        self._conn_type = SMC6X_CONNECTION_ETH
        self._conn_string = ip_addr

        # 设置命令应答超时
        self._dll.set_timeout(handle, timeout_ms)
        return True

    def connect_com(self, com_id: int, timeout_ms: int = 1000) -> bool:
        """
        通过串口连接控制器。

        :param com_id: 串口号，1-255
        :param timeout_ms: 命令应答超时时间（毫秒）
        :return: 是否连接成功
        :raises ControllerError: 连接失败时抛出
        """
        if self.is_connected:
            raise ControllerError("控制器已连接，请先断开。")

        err, handle = self._dll.open_com(com_id)
        if err != ERR_NOERR:
            raise ControllerError(
                f"串口连接失败 (COM{com_id}): {self._dll.get_errcode_description(err)}"
            )

        self._handle = handle
        self._connected = True
        self._conn_type = SMC6X_CONNECTION_COM
        self._conn_string = f"COM{com_id}"

        self._dll.set_timeout(handle, timeout_ms)
        return True

    def disconnect(self) -> bool:
        """
        断开与控制器的连接。

        :return: 是否成功断开
        """
        if self._handle is not None:
            self._dll.close(self._handle)
        self._handle = None
        self._connected = False
        self._conn_type = None
        self._conn_string = None
        return True

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def get_state(self) -> int:
        """
        读取控制器状态码。

        :return: 状态码（见 SYS_STATE_* 常量）
        :raises ControllerError: 未连接或读取失败时抛出
        """
        self._check_connected()
        err, state = self._dll.get_state(self._handle)
        if err != ERR_NOERR:
            raise ControllerError(
                f"读取控制器状态失败: {self._dll.get_errcode_description(err)}"
            )
        return state

    def get_state_desc(self) -> str:
        """读取控制器状态的中文描述。"""
        state = self.get_state()
        return SYS_STATE_DESC.get(state, f"未知状态({state})")

    def get_axises(self) -> int:
        """
        读取控制器轴数。

        :return: 轴数，出错为 0
        """
        if not self.is_connected:
            return 0
        return self._dll.get_axises(self._handle)

    def get_software_version(self) -> str:
        """
        读取软件版本（日期标识）。

        :return: 版本字符串，如 "20200101"
        """
        self._check_connected()
        err, ver = self._dll.get_software_version(self._handle)
        if err != ERR_NOERR:
            raise ControllerError(
                f"读取软件版本失败: {self._dll.get_errcode_description(err)}"
            )
        return str(ver)

    def get_hardware_id(self) -> int:
        """读取硬件版本。"""
        self._check_connected()
        err, hid = self._dll.get_hardware_id(self._handle)
        if err != ERR_NOERR:
            raise ControllerError(
                f"读取硬件版本失败: {self._dll.get_errcode_description(err)}"
            )
        return hid

    # ------------------------------------------------------------------
    # 运动参数设置（MSetting_* 系列，与官方一致）
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
        下发当前轴的运动参数（与官方一致）。

        对应官方调用序列：
            MSetting_SetStartSpeed(handle, axis, start_speed)
            MSetting_SetMotionAxisSpeed(handle, axis, max_speed)
            MSetting_SetAcceleration(handle, axis, acc)
            MSetting_SetDeceleration(handle, axis, dec)
            MSetting_SetSCurveSet(handle, axis, s_curve)

        :param iaxis: 轴号（0-3）
        :param start_speed: 启动速度
        :param max_speed: 最大速度
        :param acc: 加速度
        :param dec: 减速度
        :param s_curve: S 曲线时间
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        # 注意：SMC6480 中除 S 曲线外，其余运动参数均为整数（int32）。
        # 界面层可能传入 float，这里统一转换为 int，避免 c_int32 类型错误。
        calls = [
            ("设置启动速度", self._dll.set_start_speed, int(start_speed)),
            ("设置最大速度", self._dll.set_motion_axis_speed, int(max_speed)),
            ("设置加速度", self._dll.set_acceleration, int(acc)),
            ("设置减速度", self._dll.set_deceleration, int(dec)),
            ("设置S曲线", self._dll.set_s_curve, s_curve),
        ]
        for name, func, value in calls:
            err = func(self._handle, iaxis, value)
            if err != ERR_NOERR:
                raise ControllerError(
                    f"{name}失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
                )

    def get_motion_params(self, iaxis: int) -> dict:
        """
        读取当前轴的运动参数（与官方一致）。

        :param iaxis: 轴号（0-3）
        :return: dict，包含 start_speed / max_speed / acc / dec / s_curve
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        result = {}
        # MSetting_Get* 系列直接返回参数值（非错误码）
        calls = [
            ("start_speed", self._dll.get_start_speed),
            ("max_speed", self._dll.get_motion_axis_speed),
            ("acc", self._dll.get_acceleration),
            ("dec", self._dll.get_deceleration),
            ("s_curve", self._dll.get_s_curve),
        ]
        for key, func in calls:
            result[key] = func(self._handle, iaxis)
        return result

    # ------------------------------------------------------------------
    # 点位运动（Motion_Pmove_* 系列，与官方一致）
    # ------------------------------------------------------------------
    def pmove_abs(self, iaxis: int, pos: float) -> None:
        """
        绝对定位（与官方一致）。

        对应官方调用序列：
            Motion_Pmove_Enter(handle, axis)
            Motion_Pmove_SetAbsolute(handle, axis, pos)
            Motion_Pmove_Start(handle, axis)

        :param iaxis: 轴号（0-3）
        :param pos: 绝对目标位置
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        err = self._dll.pmove_enter(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"进入点位运动失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )
        # 位置为整数（脉冲），界面层可能传入 float，这里统一转换为 int
        err = self._dll.pmove_set_absolute(self._handle, iaxis, int(pos))
        if err != ERR_NOERR:
            raise ControllerError(
                f"设置绝对位置失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )
        err = self._dll.pmove_start(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"启动点位运动失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )

    def pmove_rel(self, iaxis: int, dist: float) -> None:
        """
        相对定位（与官方一致）。

        对应官方调用序列：
            Motion_Pmove_Enter(handle, axis)
            Motion_Pmove_SetRelative(handle, axis, dist)
            Motion_Pmove_Start(handle, axis)

        :param iaxis: 轴号（0-3）
        :param dist: 相对移动距离
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        err = self._dll.pmove_enter(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"进入点位运动失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )
        # 距离为整数（脉冲），界面层可能传入 float，这里统一转换为 int
        err = self._dll.pmove_set_relative(self._handle, iaxis, int(dist))
        if err != ERR_NOERR:
            raise ControllerError(
                f"设置相对距离失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )
        err = self._dll.pmove_start(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"启动点位运动失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )

    # ------------------------------------------------------------------
    # 定速运动（Motion_Vmove_* 系列，与官方一致）
    # ------------------------------------------------------------------
    def vmove(self, iaxis: int, positive: bool = True, speed: float = 0.0) -> None:
        """
        定速运动（与官方一致）。

        对应官方调用序列：
            Motion_Vmove_Enter(handle, axis)
            Motion_Vmove_SetDir(handle, axis, positive)
            Motion_Vmove_SetSpeed(handle, axis, speed)
            Motion_Vmove_Start(handle, axis)

        :param iaxis: 轴号（0-3）
        :param positive: 是否正向移动
        :param speed: 定速运动速度
        :raises ControllerError: 未连接或调用失败时抛出
        """
        self._check_connected()
        err = self._dll.vmove_enter(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"进入定速运动失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )
        err = self._dll.vmove_set_dir(self._handle, iaxis, 1 if positive else 0)
        if err != ERR_NOERR:
            raise ControllerError(
                f"设置定速方向失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )
        # 速度为整数，界面层可能传入 float，这里统一转换为 int
        err = self._dll.vmove_set_speed(self._handle, iaxis, int(speed))
        if err != ERR_NOERR:
            raise ControllerError(
                f"设置定速速度失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )
        err = self._dll.vmove_start(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"启动定速运动失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )

    # ------------------------------------------------------------------
    # 停止 / 状态 / 坐标（Motion_* 系列，与官方一致）
    # ------------------------------------------------------------------
    def check_down(self, iaxis: int) -> bool:
        """
        检查轴是否停止（Motion_CheckDown）。

        注意：Motion_CheckDown 为"直接返回值"类型（0=未停止, 1=已停止），
        非"返回错误码 + 指针输出"类型。

        :param iaxis: 轴号（0-3）
        :return: 是否已停止
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return bool(self._dll.check_down(self._handle, iaxis))

    def decel_stop(self, iaxis: int) -> None:
        """减速停止（Motion_DeclStop）。"""
        self._check_connected()
        err = self._dll.decl_stop(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"减速停止失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )

    def imd_stop(self, iaxis: int) -> None:
        """立即停止（Motion_ImdStop）。"""
        self._check_connected()
        err = self._dll.imd_stop(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"立即停止失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )

    def get_pulse_position(self, iaxis: int) -> int:
        """
        读取当前坐标（脉冲）（Motion_GetPulsePositon），直接返回位置。

        :param iaxis: 轴号（0-3）
        :return: 坐标，单位 脉冲
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return self._dll.get_pulse_position(self._handle, iaxis)

    def get_encoder_position(self, iaxis: int) -> int:
        """
        读取编码器位置（Motion_GetEncoderPositon），直接返回位置。

        :param iaxis: 轴号（0-3）
        :return: 编码器位置，单位 脉冲
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return self._dll.get_encoder_position(self._handle, iaxis)

    def set_pulse_position(self, iaxis: int, position: int) -> None:
        """
        设置当前坐标（脉冲）（Motion_SetPulsePositon）。

        :param iaxis: 轴号（0-3）
        :param position: 坐标，单位 脉冲
        """
        self._check_connected()
        err = self._dll.set_pulse_position(self._handle, iaxis, position)
        if err != ERR_NOERR:
            raise ControllerError(
                f"设置坐标(脉冲)失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )

    def get_cur_speed(self, iaxis: int) -> int:
        """
        读取当前速度（Motion_GetCurSpeed），直接返回速度。

        :param iaxis: 轴号（0-3）
        :return: 速度
        """
        self._check_connected()
        return self._dll.get_cur_speed(self._handle, iaxis)

    def set_home_params(self, iaxis: int, start_speed: int = 100,
                        zero_speed: int = 100, acc: int = 1000,
                        dec: int = 1000, s_curve: float = 0.0,
                        zero_dir: int = 0, zero_mode: int = 3) -> None:
        """设置回零参数（与官方程序一致）。

        官方回零调用序列：
            MSetting_SetStartSpeed(handle, axis, start_speed)
            MSetting_SetZeroSpeed(handle, axis, zero_speed)   # 回零低速
            MSetting_SetAcceleration(handle, axis, acc)
            MSetting_SetDeceleration(handle, axis, dec)
            MSetting_SetSCurveSet(handle, axis, s_curve)
            MSetting_SetZeroDir(handle, axis, zero_dir)       # 回零方向
            MSetting_SetZeroMode(handle, axis, zero_mode)     # 回零模式
            Motion_Home_FindOrigin(handle, axis)

        :param iaxis: 轴号（0-3）
        :param start_speed: 启动速度
        :param zero_speed: 回零低速（回零时的速度）
        :param acc: 加速度
        :param dec: 减速度
        :param s_curve: S 曲线时间
        :param zero_dir: 回零方向
        :param zero_mode: 回零模式
        """
        self._check_connected()
        dll = self._dll
        handle = self._handle
        # 依次设置回零参数（与官方程序一致）
        dll.set_start_speed(handle, iaxis, start_speed)
        dll.set_zero_speed(handle, iaxis, zero_speed)
        dll.set_acceleration(handle, iaxis, acc)
        dll.set_deceleration(handle, iaxis, dec)
        dll.set_s_curve(handle, iaxis, s_curve)
        dll.set_zero_dir(handle, iaxis, zero_dir)
        dll.set_zero_mode(handle, iaxis, zero_mode)

    def home_move(self, iaxis: int) -> None:
        """
        回零运动（Motion_Home_FindOrigin）。

        :param iaxis: 轴号（0-3）
        """
        self._check_connected()
        err = self._dll.home_find_origin(self._handle, iaxis)
        if err != ERR_NOERR:
            raise ControllerError(
                f"回零运动失败 (轴{iaxis}): {self._dll.get_errcode_description(err)}"
            )

    def if_home_moving(self, iaxis: int) -> bool:
        """
        检查是否回零中（Motion_Home_IfHoming）。

        注意：Motion_Home_IfHoming 为"直接返回值"类型（0=不在回零, 1=回零中），
        非"返回错误码 + 指针输出"类型。

        :param iaxis: 轴号（0-3）
        :return: 是否回零中
        :raises ControllerError: 未连接时抛出
        """
        self._check_connected()
        return bool(self._dll.home_if_homing(self._handle, iaxis))

    # ------------------------------------------------------------------
    # 兼容方法（供旧代码 / demo 使用）
    # ------------------------------------------------------------------
    def p_move_pluses(self, iaxis: int, length: int, if_abs: bool = False) -> None:
        """
        定长运动（脉冲）兼容方法。

        内部转换为官方调用序列：
            - 绝对：Motion_Pmove_Enter -> Motion_Pmove_SetAbsolute -> Motion_Pmove_Start
            - 相对：Motion_Pmove_Enter -> Motion_Pmove_SetRelative -> Motion_Pmove_Start

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
        """读取单个输入端口（DI）状态（Motion_GetInPort）。

        Args:
            port: 输入端口号（0-based，如 IN1 对应端口 0）

        Returns:
            bool: 该输入端口是否为高电平（有效）

        Raises:
            ControllerError: 未连接或读取失败时抛出
        """
        self._check_connected()
        err, value = self._dll.get_in_port(self._handle, port)
        if err != ERR_NOERR:
            raise ControllerError(
                f"读取输入端口失败 (IN{port + 1}): {self._dll.get_errcode_description(err)}"
            )
        return value

    def set_out_port(self, port: int, value: bool) -> None:
        """设置单个输出端口（DO）状态（Motion_SetOutPort）。

        Args:
            port: 输出端口号（0-based，如 OUT1 对应端口 0）
            value: True=高电平/开，False=低电平/关

        Raises:
            ControllerError: 未连接或设置失败时抛出
        """
        self._check_connected()
        err = self._dll.set_out_port(self._handle, port, 1 if value else 0)
        if err != ERR_NOERR:
            raise ControllerError(
                f"设置输出端口失败 (OUT{port + 1}): {self._dll.get_errcode_description(err)}"
            )

    # ------------------------------------------------------------------
    # 指示灯控制（二合一灯：OUT3 红灯、OUT4 绿灯，低电平有效）
    # ------------------------------------------------------------------
    # 硬件说明：
    #   - OUT3 对应红灯，OUT4 对应绿灯
    #   - 低电平有效：写 0 点亮，写 1 熄灭
    #   - 红灯 + 绿灯同时导通（都写 0）时，物理上显示黄灯
    #   - 灭灯时需要将两个位都写 1
    # 端口号与官方 SMCWriteOutBit 一致（红灯=3，绿灯=4）：
    #   SMCWriteOutBit(handle, 3, 0) 点亮红灯
    #   SMCWriteOutBit(handle, 4, 0) 点亮绿灯
    # 注意：set_out_port(port, value) 中 value=True 写 1（高电平），
    #       value=False 写 0（低电平）。因此点亮灯需传 False，熄灭传 True。
    LIGHT_RED_PORT = 3    # OUT3 红灯（SMCWriteOutBit 端口 3）
    LIGHT_GREEN_PORT = 4  # OUT4 绿灯（SMCWriteOutBit 端口 4）

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

        # 避免不必要的重复写（仅状态变化时写入）
        if red_value != self._light_red_on:
            try:
                self.set_out_port(self.LIGHT_RED_PORT, red_value)
                self._light_red_on = red_value
            except ControllerError as e:
                log_error(f"设置红灯失败: {e}")
        if green_value != self._light_green_on:
            try:
                self.set_out_port(self.LIGHT_GREEN_PORT, green_value)
                self._light_green_on = green_value
            except ControllerError as e:
                log_error(f"设置绿灯失败: {e}")

        # 输出日志（仅状态变化时）
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
