# -*- coding: utf-8 -*-
"""
自动化检测工作流模块
=================
实现由 DI 信号触发的多位置自动化检测工作流。

工作流状态机:
    IDLE -> MONITORING -> MOVING -> CAPTURING -> TESTING
    -> (循环: MOVING -> CAPTURING -> TESTING 直到所有位置完成)
    -> RETURNING -> SHOW_RESULT -> MONITORING

依赖:
    - NMC SDK: 运动控制和 DI 读取
    - CameraManager: 相机拍照
    - VisionEngine: 视觉检测

使用方式:
    workflow = InspectionWorkflow(nmc_sdk, camera_mgr, vision_engine)
    workflow.load_product(product_config)
    workflow.state_changed.connect(on_state_changed)
    workflow.start_monitoring()
    # ... 工作流自动运行 ...
    workflow.stop_monitoring()
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import numpy as np

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from core.nmc_sdk import NMCSDK, Axis_Stop_IMD, Axis_Stop_DEC, Position_Absolute, Position_Opposite, Profile_S
from core.log_manager import log_info, log_error, log_warning
from core.serial_comm import SerialCommManager


# ============================================================================
# 配置
# ============================================================================

@dataclass
class WorkflowConfig:
    """工作流配置"""
    di_bit: int = 3                    # DI 输入位号（自动化流程触发）
    poll_interval_ms: int = 50         # DI 轮询间隔（毫秒）
    axis: int = 1                      # 运动轴号
    v_max: float = 50000.0             # 最大速度
    a_max: float = 100000.0            # 加速度
    origin_position: int = 0           # 原点位置
    move_timeout_ms: int = 10000       # 运动超时（毫秒）
    arrive_check_interval_ms: int = 50 # 到位检测轮询间隔
    # ---- 多DI监测配置 ----
    di_bit_ok: int = 0                 # D1 - OK确认输入位号（默认DI0）
    di_bit_ng: int = 1                 # D2 - NG确认输入位号（默认DI1）
    di_bit_reset: int = 2              # D3 - 复位回零输入位号（默认DI2）
    reset_speed: float = 20000.0       # 复位回零速度
    reset_acc: float = 10000.0         # 复位回零加速度


# ============================================================================
# 位置检测结果
# ============================================================================

@dataclass
class PositionResult:
    """单个位置的检测结果"""
    name: str                          # 位置名称
    position: int                      # 位置坐标
    passed: bool = False               # 是否通过
    message: str = ""                  # 结果消息
    annotated: Optional[np.ndarray] = None  # 标注图
    raw_image: Optional[np.ndarray] = None  # 原始图
    tool_results: list = field(default_factory=list)  # 工具检测结果
    elapsed_ms: float = 0.0            # 检测耗时


# ============================================================================
# 工作流管理器
# ============================================================================

class InspectionWorkflow(QObject):
    """自动化检测工作流管理器"""

    class State(Enum):
        IDLE = "空闲"
        MONITORING = "等待DI触发"
        WAITING = "等待工件放稳"
        MOVING = "移动中"
        SCANNING = "扫码中"       # 新增：移动到扫码位并扫描一维码
        CAPTURING = "拍照中"
        TESTING = "检测中"
        RETURNING = "退回原点"
        SHOW_RESULT = "显示结果"
        WAITING_FOR_CONFIRM = "等待确认"  # NG弹窗等待D1/D2确认
        ERROR = "错误"

    # ── 信号 ──

    state_changed = pyqtSignal(object)  # State 枚举
    """工作流状态变化信号"""

    position_result_ready = pyqtSignal(int, object)
    """单个位置检测完成信号 (位置索引, PositionResult)"""

    all_results_ready = pyqtSignal(bool, list)
    """所有位置检测完成信号 (最终OK/NG, List[PositionResult])"""

    error_occurred = pyqtSignal(str)
    """错误信号"""

    trigger_count_changed = pyqtSignal(int)
    """触发次数变化信号"""

    ok_count_changed = pyqtSignal(int)
    """OK次数变化信号"""

    ng_count_changed = pyqtSignal(int)
    """NG次数变化信号"""

    total_elapsed_changed = pyqtSignal(float)
    """一次检测总耗时变化信号 (秒)"""

    # ── NG 手工确认信号 ──

    ng_confirm_requested = pyqtSignal(object)
    """NG 手工确认请求信号 (List[PositionResult]) - 发射所有检测结果，等待 UI 层弹窗确认"""

    ng_confirm_closed = pyqtSignal()
    """NG 确认完成信号 - D1/D2 硬件按键或鼠标点击确认后发射，通知 UI 关闭弹窗"""

    reset_during_confirm = pyqtSignal()
    """D3复位时发射，通知UI关闭NG确认弹窗"""

    barcode_failed = pyqtSignal()
    """扫码失败信号 - 扫码超时或返回 NG 时发射，通知 UI 弹出提示"""

    def __init__(self, nmc_sdk: Optional[NMCSDK] = None,
                 camera_mgr=None, vision_engine=None,
                 config: Optional[WorkflowConfig] = None,
                 parent=None):
        """
        初始化工作流

        Args:
            nmc_sdk: NMC SDK 实例（可为 None，无控制卡时仅做模拟）
            camera_mgr: CameraManager 实例
            vision_engine: VisionEngine 实例
            config: 工作流配置
            parent: QObject 父对象
        """
        super().__init__(parent)
        self._nmc_sdk = nmc_sdk
        self._camera_mgr = camera_mgr
        self._vision_engine = vision_engine
        self._config = config or WorkflowConfig()

        # 产品配置
        self._product_config: Optional[Dict[str, Any]] = None
        self._pipelines: List = []  # 每个位置对应的 Pipeline

        # 状态
        self._state = self.State.IDLE
        self._running = False

        # DI 轮询
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_all_di)
        self._last_di_state = 1  # 默认高电平（未按下）
        self._last_di_states = {0: 1, 1: 1, 2: 1}  # 记录三个DI(D1/D2/D3)的上次状态，默认高电平

        # 到位检测
        self._arrive_timer = QTimer(self)
        self._arrive_timer.timeout.connect(self._check_arrival)
        self._arrive_start_time = 0

        # DI 触发后延时启动（等待工件放稳）
        self._start_delay_timer = QTimer(self)
        self._start_delay_timer.setSingleShot(True)
        self._start_delay_timer.timeout.connect(self._on_start_delay_elapsed)

        # 当前执行状态
        self._current_pos_index = 0
        self._results: List[PositionResult] = []
        self._move_target = 0

        # 统计
        self._trigger_count = 0
        self._ok_count = 0
        self._ng_count = 0

        # 一次检测的总耗时计时
        self._inspection_start_time: float = 0.0

        # ── 一维码扫码相关 ──
        self._serial_comm: Optional[SerialCommManager] = None
        """串口通信管理器（用于扫描头）"""
        self._barcode_data: Optional[str] = None
        """扫描到的一维码数据"""
        self._is_scan_move: bool = False
        """标记当前移动是否为扫码移动（到位后区分处理）"""
        # 扫码超时定时器
        self._scan_timer = QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.timeout.connect(self._on_scan_timeout)

    # ── 属性 ──

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def ok_count(self) -> int:
        return self._ok_count

    @property
    def ng_count(self) -> int:
        return self._ng_count

    @property
    def product_config(self) -> Optional[Dict]:
        return self._product_config

    # ── 产品配置加载 ──

    def load_product(self, product_config: Dict[str, Any]) -> bool:
        """加载产品配置

        Args:
            product_config: 产品配置字典

        Returns:
            bool: 是否加载成功
        """
        if self._running:
            log_warning("工作流运行中，无法加载产品配置")
            return False

        self._product_config = product_config
        positions = product_config.get("positions", [])

        # 加载每个位置的视觉方案
        from vision.pipeline import Pipeline
        from core.paths import SCHEME_DIR
        import os

        self._pipelines = []
        for pos in positions:
            scheme_name = pos.get("scheme", "")
            if scheme_name:
                scheme_path = os.path.join(SCHEME_DIR, f"{scheme_name}.json")
                if os.path.exists(scheme_path):
                    try:
                        with open(scheme_path, 'r', encoding='utf-8') as f:
                            import json
                            data = json.load(f)
                        pipeline = Pipeline.from_dict(data)
                        self._pipelines.append(pipeline)
                        log_info(f"加载方案 [{scheme_name}] 成功")
                    except Exception as e:
                        log_error(f"加载方案 [{scheme_name}] 失败: {e}")
                        self._pipelines.append(None)
                else:
                    log_warning(f"方案文件不存在: {scheme_path}")
                    self._pipelines.append(None)
            else:
                self._pipelines.append(None)

        # 更新配置
        motion = product_config.get("motion", {})
        self._config.axis = motion.get("axis", self._config.axis)
        self._config.v_max = motion.get("v_max", self._config.v_max)
        self._config.a_max = motion.get("a_max", self._config.a_max)
        self._config.origin_position = motion.get("origin_position", self._config.origin_position)
        self._config.move_timeout_ms = motion.get("move_timeout_s", 10) * 1000
        self._config.di_bit = product_config.get("di_bit", self._config.di_bit)
        self._config.poll_interval_ms = product_config.get("poll_interval_ms", self._config.poll_interval_ms)
        # 多DI监测配置
        self._config.di_bit_ok = product_config.get("di_bit_ok", self._config.di_bit_ok)
        self._config.di_bit_ng = product_config.get("di_bit_ng", self._config.di_bit_ng)
        self._config.di_bit_reset = product_config.get("di_bit_reset", self._config.di_bit_reset)
        self._config.reset_speed = product_config.get("reset_speed", self._config.reset_speed)
        self._config.reset_acc = product_config.get("reset_acc", self._config.reset_acc)

        log_info(f"产品配置已加载: {product_config.get('name', '未知')} "
                 f"({len(positions)}个位置)")
        return True

    # ── 生命周期控制 ──

    def start_monitoring(self):
        """开始监听 DI 信号"""
        if self._running:
            log_warning("工作流已在运行中")
            return

        if self._product_config is None:
            self.error_occurred.emit("未加载产品配置")
            return

        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            self.error_occurred.emit("NMC控制卡未连接")
            return

        self._running = True
        self._trigger_count = 0
        self._ok_count = 0
        self._ng_count = 0
        self._last_di_state = 0

        # 读取当前 DI 状态作为初始值
        try:
            self._last_di_state = self._nmc_sdk.get_input_bit(self._config.di_bit)
        except Exception:
            self._last_di_state = 0

        self._set_state(self.State.MONITORING)
        self._poll_timer.start(self._config.poll_interval_ms)
        log_info(f"开始监听 DI{self._config.di_bit} (轮询间隔: {self._config.poll_interval_ms}ms)")

    def stop_monitoring(self):
        """停止监听 DI 信号"""
        self._poll_timer.stop()
        self._arrive_timer.stop()
        self._start_delay_timer.stop()
        self._scan_timer.stop()
        self._running = False
        self._set_state(self.State.IDLE)
        log_info("停止监听 DI 信号")

    def emergency_stop(self):
        """紧急停止 - 仅停止当前配置的运动轴 (Axis_2)，其他轴不受影响"""
        self._poll_timer.stop()
        self._arrive_timer.stop()
        self._start_delay_timer.stop()
        self._scan_timer.stop()

        if self._nmc_sdk and self._nmc_sdk.is_open():
            try:
                # 仅停止当前配置的轴（Axis_2），其他轴不动
                self._nmc_sdk.axis_stop(self._config.axis, Axis_Stop_IMD)
                log_info(f"紧急停止轴{self._config.axis + 1} (其他轴不受影响)")
            except Exception as e:
                log_error(f"紧急停止失败: {e}")

        self._running = False
        self._set_state(self.State.IDLE)
        log_info("紧急停止")

    def reset_error(self):
        """复位错误状态"""
        if self._state == self.State.ERROR:
            self._running = False
            self._set_state(self.State.IDLE)
            log_info("错误已复位")

    # ── 状态管理 ──

    def _set_state(self, new_state: State):
        """安全切换状态"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            log_info(f"工作流: {old_state.value} -> {new_state.value}")
            self.state_changed.emit(new_state)

    # ── DI 轮询 ──

    def _poll_all_di(self):
        """轮询所有 DI 状态，根据当前工作流状态处理不同 DI 信号"""
        # 1. 始终检测 D3 复位信号（优先级最高）
        self._check_d3_reset()

        # 2. 根据当前状态检测其他 DI
        if self._state == self.State.MONITORING:
            self._check_di_trigger()
        elif self._state == self.State.WAITING_FOR_CONFIRM:
            self._check_di_confirm()

    def _check_d3_reset(self):
        """检测 D3 复位信号（下降沿：默认1，按下为0）"""
        try:
            current = self._nmc_sdk.get_input_bit(self._config.di_bit_reset)
            if current == 0 and self._last_di_states[2] == 1:
                log_info("D3 复位按键按下，执行复位回零")
                self._execute_reset()
            self._last_di_states[2] = current
        except Exception as e:
            log_error(f"读取 D3 复位 DI 失败: {e}")

    def _check_di_trigger(self):
        """检测原触发 DI 信号（下降沿：默认1，按下为0）"""
        try:
            current = self._nmc_sdk.get_input_bit(self._config.di_bit)
            # 检测下降沿: 1 -> 0（工件放入时 DI 从高变低）
            if current == 0 and self._last_di_state == 1:
                log_info(f"DI{self._config.di_bit} 下降沿触发 (工件放入)")
                self._on_di_triggered()
            self._last_di_state = current
        except Exception as e:
            log_error(f"读取 DI 失败: {e}")

    def _check_di_confirm(self):
        """检测 D1/D2 确认信号（下降沿：默认1，按下为0）"""
        try:
            # 检测 D1 (OK)
            d1 = self._nmc_sdk.get_input_bit(self._config.di_bit_ok)
            if d1 == 0 and self._last_di_states[0] == 1:
                log_info("D1 OK 按键按下")
                self.confirm_ng_result(True)
            self._last_di_states[0] = d1

            # 检测 D2 (NG)
            d2 = self._nmc_sdk.get_input_bit(self._config.di_bit_ng)
            if d2 == 0 and self._last_di_states[1] == 1:
                log_info("D2 NG 按键按下")
                self.confirm_ng_result(False)
            self._last_di_states[1] = d2
        except Exception as e:
            log_error(f"读取 D1/D2 确认 DI 失败: {e}")

    # ── D3 复位与归零 ──

    def _execute_reset(self):
        """执行 D3 复位操作 - 停止当前流程，AXI1 回到 0 坐标"""
        log_info("D3 复位按键触发，执行复位操作...")

        # 1) 停止所有定时器
        self._poll_timer.stop()
        self._arrive_timer.stop()
        self._start_delay_timer.stop()
        self._scan_timer.stop()

        # 2) 如果当前在 NG 确认等待状态，通知 UI 关闭弹窗
        if self._state == self.State.WAITING_FOR_CONFIRM:
            log_info("D3 复位：正在等待 NG 确认，通知关闭弹窗")
            self.reset_during_confirm.emit()

        # 3) 设置运行标志
        self._running = False
        self._set_state(self.State.IDLE)

        # 4) 先停止轴运动，再延时启动归零（确保轴完全停止）
        try:
            self._nmc_sdk.axis_stop(self._config.axis, Axis_Stop_IMD)
        except Exception as e:
            log_warning(f"D3 复位停止轴失败(可忽略): {e}")

        # 延时 100ms 后执行归零，等待轴完全停止
        QTimer.singleShot(100, self._move_to_zero)

    def _move_to_zero(self):
        """移动到 0 坐标（速度 20000，加速度 10000）"""
        try:
            axis = self._config.axis

            # 1) 确保轴已使能
            try:
                self._nmc_sdk.set_servo_enable(axis, 1)
            except Exception as e:
                log_warning(f"轴{axis + 1} 使能失败(可忽略): {e}")

            # 2) 清除轴状态
            try:
                self._nmc_sdk.clear_axis_state(axis)
            except Exception as e:
                log_warning(f"清除轴{axis + 1} 状态失败(可忽略): {e}")

            # 3) 读取当前位置，计算到 0 的相对位移
            try:
                current_pos = self._nmc_sdk.get_position(axis)
            except Exception:
                current_pos = 0
            diff = 0 - current_pos
            log_info(f"D3 复位: 轴{axis + 1} 当前位置: {current_pos}, 目标: 0, 相对位移: {diff}")

            # 4) 设置速度参数（速度 20000，加速度 10000）
            ret_profile = self._nmc_sdk.set_axis_profile(
                axis,
                0,                          # v_ini
                self._config.reset_speed,   # v_max = 20000
                self._config.reset_acc,     # a_max = 10000
                0,                          # jerk
                0,                          # v_end
                Profile_S                   # S曲线
            )
            log_info(f"D3 复位: set_axis_profile 返回值: {ret_profile}")

            # 5) 发送相对位置运动指令
            ret_move = self._nmc_sdk.uniaxial_long(axis, diff, Position_Opposite)
            log_info(f"D3 复位: uniaxial_long(相对, dist={diff}) 返回值: {ret_move}")

            if ret_move != 0:
                log_error(f"D3 复位: 轴{axis + 1} 运动失败，返回值: {ret_move}")
        except Exception as e:
            log_error(f"D3 复位运动失败: {e}")

    def _on_di_triggered(self):
        """DI 触发 - 延时后开始执行检测流程"""
        self._trigger_count += 1
        self.trigger_count_changed.emit(self._trigger_count)

        # 重置当前执行状态
        self._current_pos_index = 0
        self._results = []

        # 记录本次检测开始时间（用于计算总耗时）
        import time
        self._inspection_start_time = time.time()

        # 延时 1 秒再开始移动，等待工件放稳
        log_info("DI 触发，等待 1 秒后开始检测...")
        self._set_state(self.State.WAITING)
        self._start_delay_timer.start(1000)  # 1 秒

    def _on_start_delay_elapsed(self):
        """延时结束 - 判断是否需要先扫码，然后开始检测流程"""
        # 检查是否启用了扫码功能
        barcode_cfg = self._product_config.get("barcode_scan", {})
        if barcode_cfg.get("enabled", False):
            scan_pos = barcode_cfg.get("position", 0)
            log_info(f"延时结束，移动到扫码位 (坐标: {scan_pos})")
            self._move_to_scan_position(scan_pos)
        else:
            log_info("延时结束，开始执行检测流程")
            self._execute_current_position()

    # ── 位置执行 ──

    def _execute_current_position(self):
        """执行当前位置的检测"""
        positions = self._product_config.get("positions", [])
        if self._current_pos_index >= len(positions):
            # 所有位置已完成，退回原点
            self._return_to_origin()
            return

        pos = positions[self._current_pos_index]
        target = pos.get("position", 0)

        log_info(f"移动到位置 {self._current_pos_index + 1}: {pos.get('name', '')} (坐标: {target})")
        self._move_to(target)

    def _move_to(self, target_pos: int):
        """移动到目标位置"""
        self._set_state(self.State.MOVING)
        self._move_target = target_pos

        try:
            axis = self._config.axis

            # 1) 确保轴已使能
            try:
                self._nmc_sdk.set_servo_enable(axis, 1)
            except Exception as e:
                log_warning(f"轴{axis + 1} 使能失败(可忽略): {e}")

            # 2) 清除轴状态（清除可能的停止/报警状态）
            try:
                self._nmc_sdk.clear_axis_state(axis)
            except Exception as e:
                log_warning(f"清除轴{axis + 1} 状态失败(可忽略): {e}")

            # 3) 读取当前位置，计算相对位移（与 main_window.py 手动移动逻辑一致）
            try:
                current_pos = self._nmc_sdk.get_position(axis)
            except Exception:
                current_pos = 0
            diff = target_pos - current_pos
            log_info(f"轴{axis + 1} 当前位置: {current_pos}, 目标: {target_pos}, 相对位移: {diff}")

            # 4) 设置速度参数
            ret_profile = self._nmc_sdk.set_axis_profile(
                axis,
                0,                      # v_ini
                self._config.v_max,     # v_max
                self._config.a_max,     # a_max
                0,                      # jerk
                0,                      # v_end
                Profile_S               # S曲线
            )
            log_info(f"轴{axis + 1} set_axis_profile 返回值: {ret_profile}")

            # 5) 发送相对位置运动指令（使用 uniaxial_long 避免 c_double/c_long 冲突）
            ret_move = self._nmc_sdk.uniaxial_long(axis, diff, Position_Opposite)
            log_info(f"轴{axis + 1} uniaxial_long(相对, dist={diff}) 返回值: {ret_move}")

            if ret_move != 0:
                error_msg = f"轴{axis + 1} 移动失败，返回值: {ret_move}"
                log_error(error_msg)
                self._on_error(error_msg)
                return

            # 6) 启动到位检测
            import time
            self._arrive_start_time = int(time.time() * 1000)
            self._arrive_timer.start(self._config.arrive_check_interval_ms)

        except Exception as e:
            error_msg = f"运动失败: {e}"
            log_error(error_msg)
            self._on_error(error_msg)

    def _check_arrival(self):
        """检查轴是否到位"""
        try:
            state = self._nmc_sdk.get_axis_state(self._config.axis)
            if state == 0:  # 空闲 = 到位
                self._arrive_timer.stop()
                log_info(f"轴已到位 (位置: {self._move_target})")
                # 根据当前状态决定到位后的处理
                if self._state == self.State.RETURNING:
                    self._on_returned_to_origin()
                elif self._is_scan_move:
                    # 扫码移动到位
                    self._is_scan_move = False
                    self._on_arrived_at_scan_position()
                else:
                    self._on_arrived()
                return

            # 超时检查
            import time
            elapsed = int(time.time() * 1000) - self._arrive_start_time
            if elapsed > self._config.move_timeout_ms:
                self._arrive_timer.stop()
                error_msg = f"运动超时 ({self._config.move_timeout_ms}ms)"
                log_error(error_msg)
                self._nmc_sdk.axis_stop(self._config.axis, Axis_Stop_DEC)
                self._on_error(error_msg)

        except Exception as e:
            log_error(f"到位检测失败: {e}")

    def _on_arrived(self):
        """到位后的处理 - 拍照"""
        self._capture()

    # ── 拍照 ──

    def _capture(self):
        """拍照"""
        self._set_state(self.State.CAPTURING)

        if self._camera_mgr is None:
            self._on_error("相机管理器未初始化")
            return

        try:
            # 先设置相机参数（从产品配置读取）
            camera_cfg = self._product_config.get("camera", {})
            if camera_cfg:
                try:
                    if hasattr(self._camera_mgr, 'set_exposure_time'):
                        self._camera_mgr.set_exposure_time(camera_cfg.get("exposure_time", 18000))
                    if hasattr(self._camera_mgr, 'set_gain'):
                        self._camera_mgr.set_gain(camera_cfg.get("gain", 0))
                except Exception as e:
                    log_warning(f"设置相机参数失败: {e}")

            # 触发拍照
            if hasattr(self._camera_mgr, 'capture_once'):
                raw = self._camera_mgr.capture_once()
                # capture_once() 返回 (width, height, pixel_type, data) 元组
                # 需要转换为 OpenCV 图像
                if isinstance(raw, tuple) and len(raw) == 4:
                    from camera_manager import raw_to_opencv
                    width, height, pixel_type, frame_data = raw
                    image = raw_to_opencv(frame_data, width, height, pixel_type)
                else:
                    # 兼容：如果返回的已经是 numpy 数组
                    image = raw
            else:
                # 兼容：从实时流中获取当前帧
                image = getattr(self._camera_mgr, 'get_current_frame', lambda: None)()

            if image is None:
                self._on_error("拍照失败: 图像为空")
                return

            log_info(f"拍照成功 (位置 {self._current_pos_index + 1})")
            self._start_test(image)

        except Exception as e:
            self._on_error(f"拍照失败: {e}")

    # ── 检测 ──

    def _start_test(self, image: np.ndarray):
        """开始检测"""
        self._set_state(self.State.TESTING)

        # 获取当前位置对应的流水线
        pipeline = None
        if self._current_pos_index < len(self._pipelines):
            pipeline = self._pipelines[self._current_pos_index]

        if pipeline is None:
            # 没有流水线，直接标记为通过（占位）
            log_warning(f"位置 {self._current_pos_index + 1} 未设置视觉方案，标记为通过")
            self._on_test_completed(True, "未设置方案，默认通过", image, image, [])
            return

        # 设置流水线并执行检测
        self._vision_engine.set_pipeline(pipeline)
        try:
            passed, message, annotated = self._vision_engine.execute(
                image,
                scheme_name=self._product_config.get("name", "未知")
            )
            results = self._vision_engine.get_last_results()
            self._on_test_completed(passed, message, annotated, image, results)
        except Exception as e:
            log_error(f"检测异常: {e}")
            self._on_test_completed(False, f"检测异常: {e}", image, image, [])

    def _on_test_completed(self, passed: bool, message: str,
                           annotated: np.ndarray, raw_image: np.ndarray,
                           tool_results: list):
        """检测完成回调"""
        positions = self._product_config.get("positions", [])
        pos = positions[self._current_pos_index] if self._current_pos_index < len(positions) else {"name": f"位置{self._current_pos_index + 1}"}

        # 记录结果
        import time
        result = PositionResult(
            name=pos.get("name", f"位置{self._current_pos_index + 1}"),
            position=pos.get("position", 0),
            passed=passed,
            message=message,
            annotated=annotated,
            raw_image=raw_image,
            tool_results=tool_results,
            elapsed_ms=0.0  # TODO: 计算实际耗时
        )
        self._results.append(result)

        log_info(f"位置 {self._current_pos_index + 1} [{result.name}]: {'OK' if passed else 'NG'} | {message}")

        # 发射单个位置结果信号
        self.position_result_ready.emit(self._current_pos_index, result)

        # 移动到下一个位置
        self._current_pos_index += 1
        self._execute_current_position()

    # ── 退回原点 ──

    def _return_to_origin(self):
        """退回原点"""
        self._set_state(self.State.RETURNING)
        log_info(f"退回原点 (坐标: {self._config.origin_position})")

        # 先停止到位检测计时器，防止轴已到位时误触发 _on_arrived
        self._arrive_timer.stop()

        # 发送回原点运动指令
        self._move_target = self._config.origin_position
        try:
            axis = self._config.axis

            # 1) 确保轴已使能
            try:
                self._nmc_sdk.set_servo_enable(axis, 1)
            except Exception as e:
                log_warning(f"轴{axis + 1} 使能失败(可忽略): {e}")

            # 2) 清除轴状态（清除可能的停止/报警状态）
            try:
                self._nmc_sdk.clear_axis_state(axis)
            except Exception as e:
                log_warning(f"清除轴{axis + 1} 状态失败(可忽略): {e}")

            # 3) 读取当前位置，计算相对位移（与 _move_to 逻辑一致）
            try:
                current_pos = self._nmc_sdk.get_position(axis)
            except Exception:
                current_pos = 0
            diff = self._config.origin_position - current_pos
            log_info(f"轴{axis + 1} 当前位置: {current_pos}, 原点目标: {self._config.origin_position}, 相对位移: {diff}")

            # 4) 设置速度参数
            ret_profile = self._nmc_sdk.set_axis_profile(
                axis,
                0,                      # v_ini
                self._config.v_max,     # v_max
                self._config.a_max,     # a_max
                0,                      # jerk
                0,                      # v_end
                Profile_S               # S曲线
            )
            log_info(f"轴{axis + 1} set_axis_profile 返回值: {ret_profile}")

            # 5) 发送相对位置运动指令（使用 uniaxial_long 避免 c_double/c_long 冲突）
            ret_move = self._nmc_sdk.uniaxial_long(axis, diff, Position_Opposite)
            log_info(f"轴{axis + 1} uniaxial_long(相对, dist={diff}) 返回值: {ret_move}")

            if ret_move != 0:
                error_msg = f"轴{axis + 1} 退回原点失败，返回值: {ret_move}"
                log_error(error_msg)
                self._on_error(error_msg)
                return

            # 6) 启动到位检测（用于回原点）
            import time
            self._arrive_start_time = int(time.time() * 1000)
            self._arrive_timer.start(self._config.arrive_check_interval_ms)
        except Exception as e:
            log_error(f"退回原点运动失败: {e}")
            # 运动失败也直接显示结果
            self._show_final_result()

    def _on_returned_to_origin(self):
        """回到原点后的处理 - 显示结果"""
        self._show_final_result()

    # ── 显示结果 ──

    def _show_final_result(self):
        """显示最终结果"""
        self._set_state(self.State.SHOW_RESULT)

        # 计算本次检测总耗时
        import time
        total_elapsed = time.time() - self._inspection_start_time
        self.total_elapsed_changed.emit(total_elapsed)

        # 计算最终结果（所有位置都通过才算 OK）
        # 注意：空列表（如扫码失败时）应视为 NG
        if not self._results:
            all_passed = False
        else:
            all_passed = all(r.passed for r in self._results)

        if all_passed:
            # OK：保存数据、更新统计并继续
            self._save_ng_ok_data()  # 保存 OK 缩略图 + CSV 日志
            self._ok_count += 1
            self.ok_count_changed.emit(self._ok_count)
            # 发射最终结果信号
            self.all_results_ready.emit(True, self._results)
            log_info(f"最终结果: OK "
                     f"(触发: {self._trigger_count}, OK: {self._ok_count}, NG: {self._ng_count})"
                     f" | 总耗时: {total_elapsed:.2f}s")
            # 自动继续监听
            self._set_state(self.State.MONITORING)
        else:
            # NG：发射手工确认请求信号，等待 UI 层弹窗确认
            log_info(f"检测结果为 NG，请求手工确认... | 总耗时: {total_elapsed:.2f}s")
            self._set_state(self.State.WAITING_FOR_CONFIRM)  # 进入等待确认状态
            self.ng_confirm_requested.emit(self._results)

    def confirm_ng_result(self, confirmed_ok: bool):
        """
        NG 手工确认结果回调 - 由 UI 层在弹窗确认后调用。

        Args:
            confirmed_ok: True 表示操作员确认为 OK，False 表示确认为 NG
        """
        if self._state != self.State.WAITING_FOR_CONFIRM:
            log_warning(f"工作流状态不是 WAITING_FOR_CONFIRM，忽略确认回调 (当前: {self._state.value})")
            return

        if confirmed_ok:
            # 操作员确认为 OK：不保存错误图片
            self._ok_count += 1
            self.ok_count_changed.emit(self._ok_count)
            log_info("手工确认: OK")
            self._save_ng_ok_data()  # 可选：保存 OK 数据
            self.all_results_ready.emit(True, self._results)
        else:
            # 操作员确认为 NG：保存所有 NG 位置的错误图片和检测数据
            self._ng_count += 1
            self.ng_count_changed.emit(self._ng_count)
            log_info("手工确认: NG")
            self._save_ng_error_data()
            self.all_results_ready.emit(False, self._results)

        log_info(f"最终结果: {'OK' if confirmed_ok else 'NG'} "
                 f"(触发: {self._trigger_count}, OK: {self._ok_count}, NG: {self._ng_count})")

        # 通知 UI 关闭 NG 确认弹窗
        self.ng_confirm_closed.emit()

        # 自动继续监听
        self._set_state(self.State.MONITORING)
        
    def _save_ng_ok_data(self):
        """保存所有 OK 位置的检测数据（缩略图 + CSV 日志）

        目录结构:
            data/production data/
                YYYY-MM-DD/
                    OK/
                        {ID号}/
                            {ID号}_{HHMMSS}_thumbnail.jpg  # 缩略图
                        ok_log.csv
        """
        from core.result_storage import ResultStorage

        product_name = self._product_config.get("name", "未知产品") if self._product_config else "未知产品"
        barcode = self._barcode_data or "NO_BARCODE"

        storage = ResultStorage()
        for result in self._results:
            if result.annotated is not None:
                try:
                    storage.save_ok_data(
                        scheme_name=product_name,
                        product_id=barcode,
                        annotated_image=result.annotated,
                    )
                except Exception as e:
                    log_error(f"保存 OK 位置 [{result.name}] 数据失败: {e}")

    def _save_ng_error_data(self):
        """保存所有 NG 位置的错误数据（标注结果图 + CSV 日志）

        目录结构:
            data/production data/
                YYYY-MM-DD/
                    NG/
                        {ID号}/
                            {ID号}_{HHMMSS}_result.jpg   # 标注结果图
                        ng_log.csv
        """
        from core.result_storage import ResultStorage

        product_name = self._product_config.get("name", "未知产品") if self._product_config else "未知产品"
        barcode = self._barcode_data or "NO_BARCODE"

        storage = ResultStorage()
        for result in self._results:
            if not result.passed and result.annotated is not None:
                try:
                    storage.save_ng_data(
                        scheme_name=product_name,
                        product_id=barcode,
                        annotated_image=result.annotated,
                    )
                except Exception as e:
                    log_error(f"保存 NG 位置 [{result.name}] 错误数据失败: {e}")

    # ── 错误处理 ──

    def _on_error(self, error_msg: str):
        """错误处理"""
        log_error(error_msg)
        self.error_occurred.emit(error_msg)
        self._set_state(self.State.ERROR)

        # 尝试停止轴
        if self._nmc_sdk and self._nmc_sdk.is_open():
            try:
                self._nmc_sdk.axis_stop(self._config.axis, Axis_Stop_DEC)
            except Exception:
                pass

    # ── 资源清理 ──

    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        self._poll_timer.stop()
        self._arrive_timer.stop()
        self._scan_timer.stop()
        self._pipelines = []
        self._product_config = None
        self._results = []
        self._barcode_data = None
        # 断开串口数据接收信号
        if self._serial_comm is not None:
            try:
                self._serial_comm.data_received.disconnect(self._on_barcode_data_received)
            except (TypeError, RuntimeError):
                pass
            self._serial_comm = None
        log_info("工作流资源已清理")

    # ── 一维码扫码相关方法 ──

    def set_serial_comm(self, comm: Optional[SerialCommManager]):
        """设置串口通信管理器（用于扫描头）

        Args:
            comm: SerialCommManager 实例，为 None 时断开连接
        """
        # 断开旧连接
        if self._serial_comm is not None:
            try:
                self._serial_comm.data_received.disconnect(self._on_barcode_data_received)
            except (TypeError, RuntimeError):
                pass

        self._serial_comm = comm

        # 连接新信号
        if comm is not None:
            comm.data_received.connect(self._on_barcode_data_received)
            log_info("串口通信管理器已设置（用于一维码扫描）")

    def _move_to_scan_position(self, target_pos: int):
        """移动到扫码位

        Args:
            target_pos: 扫码位的轴坐标
        """
        self._set_state(self.State.MOVING)
        self._move_target = target_pos
        self._is_scan_move = True

        try:
            axis = self._config.axis

            # 1) 确保轴已使能
            try:
                self._nmc_sdk.set_servo_enable(axis, 1)
            except Exception as e:
                log_warning(f"轴{axis + 1} 使能失败(可忽略): {e}")

            # 2) 清除轴状态
            try:
                self._nmc_sdk.clear_axis_state(axis)
            except Exception as e:
                log_warning(f"清除轴{axis + 1} 状态失败(可忽略): {e}")

            # 3) 读取当前位置，计算相对位移
            try:
                current_pos = self._nmc_sdk.get_position(axis)
            except Exception:
                current_pos = 0
            diff = target_pos - current_pos
            log_info(f"扫码移动: 轴{axis + 1} 当前位置: {current_pos}, 目标: {target_pos}, 相对位移: {diff}")

            # 4) 设置速度参数
            ret_profile = self._nmc_sdk.set_axis_profile(
                axis,
                0,                      # v_ini
                self._config.v_max,     # v_max
                self._config.a_max,     # a_max
                0,                      # jerk
                0,                      # v_end
                Profile_S               # S曲线
            )
            log_info(f"扫码移动: set_axis_profile 返回值: {ret_profile}")

            # 5) 发送相对位置运动指令
            ret_move = self._nmc_sdk.uniaxial_long(axis, diff, Position_Opposite)
            log_info(f"扫码移动: uniaxial_long(相对, dist={diff}) 返回值: {ret_move}")

            if ret_move != 0:
                error_msg = f"扫码移动失败，返回值: {ret_move}"
                log_error(error_msg)
                self._on_error(error_msg)
                return

            # 6) 启动到位检测
            import time
            self._arrive_start_time = int(time.time() * 1000)
            self._arrive_timer.start(self._config.arrive_check_interval_ms)

        except Exception as e:
            error_msg = f"扫码移动失败: {e}"
            log_error(error_msg)
            self._on_error(error_msg)

    def _on_arrived_at_scan_position(self):
        """到达扫码位 - 发送扫描命令触发扫描头"""
        self._set_state(self.State.SCANNING)

        barcode_cfg = self._product_config.get("barcode_scan", {})
        command = barcode_cfg.get("command", "01 54 04")
        timeout_ms = barcode_cfg.get("timeout_ms", 5000)

        if self._serial_comm is not None and self._serial_comm.is_open:
            # 发送 HEX 扫描命令
            count = self._serial_comm.send_hex(command)
            if count > 0:
                log_info(f"已发送扫描命令: {command} ({count} 字节)")
                # 启动超时定时器
                self._scan_timer.start(timeout_ms)
                log_info(f"等待扫码返回 (超时: {timeout_ms}ms)...")
            else:
                log_error("发送扫描命令失败")
                self._on_barcode_failed()
        else:
            log_error("串口未打开，无法扫码")
            self._on_barcode_failed()

    def _on_barcode_data_received(self, data: bytes):
        """接收到串口数据 - 解析一维码

        仅在 SCANNING 状态下处理，解析 ASCII 文本并去除 \\r\\n。
        扫描头在没有条码时会返回 "NG"，此时视为扫码失败。

        Args:
            data: 收到的原始字节数据
        """
        if self._state != self.State.SCANNING:
            # 非扫码状态收到的串口数据，忽略
            return

        # 停止超时定时器
        self._scan_timer.stop()

        try:
            # 解析 ASCII 文本，去除首尾空白（包括 \\r\\n）
            raw_text = data.decode('ascii', errors='replace').strip()
            # 过滤掉不可见字符和控制字符，只保留可打印字符
            barcode = ''.join(c for c in raw_text if c.isprintable()).strip()

            # 判断是否为有效一维码：
            # 1. 非空
            # 2. 长度 >= 3
            # 3. 不是 "NG"（扫描头在没有条码时返回 "NG"）
            if (barcode
                    and len(barcode) >= 3
                    and barcode.upper() != "NG"):
                self._barcode_data = barcode
                log_info(f"扫码成功: {barcode}")
                # 扫码成功，开始执行检测位置
                self._current_pos_index = 0
                self._execute_current_position()
            else:
                log_warning(f"扫码失败: 返回={raw_text!r} (过滤后={barcode!r})")
                self._on_barcode_failed()
        except Exception as e:
            log_error(f"解析一维码失败: {e}")
            self._on_barcode_failed()

    def _on_scan_timeout(self):
        """扫码超时 - 未收到扫描头返回数据"""
        log_error("扫码超时：未收到一维码数据")
        self._on_barcode_failed()

    def _on_barcode_failed(self):
        """扫码失败处理 - 退回原点，发射扫码失败信号，不保存错误图片"""
        self._barcode_data = None
        log_info("扫码失败，退回原点，等待重新放入工件")

        # 注意：_trigger_count 已在 _on_di_triggered 中加过，这里不再重复加
        # 更新 NG 计数
        self._ng_count += 1
        self.ng_count_changed.emit(self._ng_count)

        # 发射扫码失败信号（UI 层弹出提示）
        self.barcode_failed.emit()

        # 不保存错误图片，直接退回原点并回到 MONITORING 状态
        # 扫码失败不需要人工确认 NG，直接继续监听
        self._return_to_origin_skip_result()

    def _return_to_origin_skip_result(self):
        """退回原点但不显示结果（用于扫码失败场景），到位后直接回到 MONITORING"""
        self._set_state(self.State.RETURNING)
        log_info(f"扫码失败: 退回原点 (坐标: {self._config.origin_position})")

        # 先停止到位检测计时器
        self._arrive_timer.stop()

        # 发送回原点运动指令
        self._move_target = self._config.origin_position
        try:
            axis = self._config.axis

            # 1) 确保轴已使能
            try:
                self._nmc_sdk.set_servo_enable(axis, 1)
            except Exception as e:
                log_warning(f"轴{axis + 1} 使能失败(可忽略): {e}")

            # 2) 清除轴状态
            try:
                self._nmc_sdk.clear_axis_state(axis)
            except Exception as e:
                log_warning(f"清除轴{axis + 1} 状态失败(可忽略): {e}")

            # 3) 读取当前位置，计算相对位移
            try:
                current_pos = self._nmc_sdk.get_position(axis)
            except Exception:
                current_pos = 0
            diff = self._config.origin_position - current_pos
            log_info(f"轴{axis + 1} 当前位置: {current_pos}, 原点目标: {self._config.origin_position}, 相对位移: {diff}")

            # 4) 设置速度参数
            ret_profile = self._nmc_sdk.set_axis_profile(
                axis,
                0,                      # v_ini
                self._config.v_max,     # v_max
                self._config.a_max,     # a_max
                0,                      # jerk
                0,                      # v_end
                Profile_S               # S曲线
            )
            log_info(f"set_axis_profile 返回值: {ret_profile}")

            # 5) 发送相对位置运动指令
            ret_move = self._nmc_sdk.uniaxial_long(axis, diff, Position_Opposite)
            log_info(f"uniaxial_long(相对, dist={diff}) 返回值: {ret_move}")

            if ret_move != 0:
                error_msg = f"退回原点失败，返回值: {ret_move}"
                log_error(error_msg)
                # 运动失败也直接回到 MONITORING
                self._set_state(self.State.MONITORING)
                return

            # 6) 启动到位检测（使用 _check_arrival_skip_result 处理）
            import time
            self._arrive_start_time = int(time.time() * 1000)
            # 临时替换到位回调，使用定时器单次检测
            self._arrive_timer.timeout.disconnect(self._check_arrival)
            self._arrive_timer.timeout.connect(self._check_arrival_skip_result)
            self._arrive_timer.start(self._config.arrive_check_interval_ms)
        except Exception as e:
            log_error(f"退回原点运动失败: {e}")
            # 运动失败也直接回到 MONITORING
            self._set_state(self.State.MONITORING)

    def _check_arrival_skip_result(self):
        """到位检测（扫码失败回原点专用）- 到位后直接回到 MONITORING"""
        try:
            state = self._nmc_sdk.get_axis_state(self._config.axis)
            if state == 0:  # 空闲 = 到位
                self._arrive_timer.stop()
                # 恢复原来的到位检测回调
                self._arrive_timer.timeout.disconnect(self._check_arrival_skip_result)
                self._arrive_timer.timeout.connect(self._check_arrival)
                log_info(f"扫码失败: 轴已退回原点")
                # 直接回到 MONITORING，不显示结果
                self._set_state(self.State.MONITORING)
                return

            # 超时检查
            import time
            elapsed = int(time.time() * 1000) - self._arrive_start_time
            if elapsed > self._config.move_timeout_ms:
                self._arrive_timer.stop()
                # 恢复原来的到位检测回调
                self._arrive_timer.timeout.disconnect(self._check_arrival_skip_result)
                self._arrive_timer.timeout.connect(self._check_arrival)
                log_error("扫码失败回原点超时")
                # 超时也直接回到 MONITORING
                self._set_state(self.State.MONITORING)
        except Exception as e:
            log_error(f"到位检测失败: {e}")
