# -*- coding: utf-8 -*-
"""
自动化检测工作流模块
=================
实现由  SMC 轴控制触发的多位置自动化检测工作流。

注意：本版本已移除 NMC 运动控制卡（轴控制）相关逻辑，仅保留：
    - 串口通信（一维码扫码）
    - 相机拍照
    - 视觉检测
    - 结果保存与 NG 手工确认

工作流状态机:
    IDLE -> WAITING -> SCANNING -> CAPTURING -> TESTING
    -> (循环: CAPTURING -> TESTING 直到所有位置完成)
    -> SHOW_RESULT -> MONITORING

    IDLE: 空闲状态，未加载产品配置或未开始监听
    WAITING: 等待触发（DI 上升沿或手动触发）
    SCANNING: 扫码中（可选）
    CAPTURING: 拍照中
    TESTING: 检测中
    SHOW_RESULT: 显示结果（OK/NG），等待人工确认或自动确认
    MONITORING: 监听状态，等待下一次触发

依赖:
    - CameraManager: 相机拍照
    - VisionEngine: 视觉检测
    - SerialCommManager: 一维码扫码（可选）

使用方式:
    workflow = InspectionWorkflow(camera_mgr, vision_engine)
    workflow.load_product(product_config)
    workflow.state_changed.connect(on_state_changed)
    workflow.start_inspection()
    # ... 工作流自动运行 ...
    workflow.stop_monitoring()
"""

from enum import Enum
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import cv2

from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal

from core.log_manager import log_info, log_error, log_warning
from core.serial_comm import SerialCommManager


# ============================================================================
# 配置
# ============================================================================

@dataclass
class WorkflowConfig:
    """工作流配置"""
    # 扫码超时（毫秒）
    scan_timeout_ms: int = 5000
    # 触发后延时（毫秒），等待工件放稳
    start_delay_ms: int = 1000


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
    qr_data: str = ""                  # 条码识别结果（板卡 SN）
    barcodes: list = field(default_factory=list)  # 识别到的条码详情列表
    save_timestamp: str = ""           # 数据保存时的时间戳（用于更新 XML）
    confirmed: bool = False            # 是否已人工确认（NG 逐点位确认用）


# ============================================================================
# 检测工作线程
# ============================================================================

class InspectionWorker(QThread):
    """检测工作线程：执行拍照 + 视觉检测（阻塞操作），避免阻塞主线程。

    主线程保持空闲，DI 轮询（QTimer）可持续运行，从而能随时响应停止等按钮。
    拍照与检测完成后通过信号把结果发回主线程，由主线程继续拼接、状态更新。
    """
    capture_done = pyqtSignal(object)          # 拍照完成，返回图像
    capture_failed = pyqtSignal(str)           # 拍照失败
    test_done = pyqtSignal(bool, str, object, object, list)  # 检测完成
    test_failed = pyqtSignal(str)              # 检测异常

    def __init__(self, camera_mgr, vision_engine, pipeline, scheme_name,
                 camera_cfg=None, parent=None):
        super().__init__(parent)
        self._camera_mgr = camera_mgr
        self._vision_engine = vision_engine
        self._pipeline = pipeline
        self._scheme_name = scheme_name
        self._camera_cfg = camera_cfg or {}

    def run(self):
        """线程主循环：拍照 → 检测 → 发信号。"""
        # 1. 拍照（阻塞）
        image = self._capture()
        if image is None:
            self.capture_failed.emit("拍照失败: 图像为空")
            return
        self.capture_done.emit(image)

        # 2. 检测（可能耗时）
        if self._pipeline is None:
            # 没有流水线，直接标记为通过（占位）
            self.test_done.emit(True, "未设置方案，默认通过", image, image, [])
            return

        try:
            self._vision_engine.set_pipeline(self._pipeline)
            passed, message, annotated = self._vision_engine.execute(
                image, scheme_name=self._scheme_name
            )
            results = self._vision_engine.get_last_results()
            self.test_done.emit(passed, message, annotated, image, results)
        except Exception as e:  # noqa: BLE001
            log_error(f"检测异常: {e}")
            self.test_failed.emit(str(e))

    def _capture(self):
        """拍照（阻塞操作，在工作线程执行）。"""
        if self._camera_mgr is None:
            return None
        try:
            # 设置相机参数
            if self._camera_cfg:
                try:
                    if hasattr(self._camera_mgr, 'set_exposure_time'):
                        self._camera_mgr.set_exposure_time(
                            self._camera_cfg.get("exposure_time", 18000))
                    if hasattr(self._camera_mgr, 'set_gain'):
                        self._camera_mgr.set_gain(self._camera_cfg.get("gain", 0))
                except Exception as e:  # noqa: BLE001
                    log_warning(f"设置相机参数失败: {e}")

            # 触发拍照
            if hasattr(self._camera_mgr, 'capture_once'):
                raw = self._camera_mgr.capture_once()
                if isinstance(raw, tuple) and len(raw) == 4:
                    from camera_manager import raw_to_opencv
                    width, height, pixel_type, frame_data = raw
                    return raw_to_opencv(frame_data, width, height, pixel_type)
                return raw
            # 兼容：从实时流中获取当前帧
            return getattr(self._camera_mgr, 'get_current_frame', lambda: None)()
        except Exception as e:  # noqa: BLE001
            log_error(f"拍照失败: {e}")
            return None


# ============================================================================
# 工作流管理器
# ============================================================================

class InspectionWorkflow(QObject):
    """自动化检测工作流管理器（无轴运动版本）"""

    class State(Enum):
        IDLE = "空闲"
        MONITORING = "等待触发"
        WAITING = "等待工件放稳"
        SCANNING = "扫码中"
        CAPTURING = "拍照中"
        TESTING = "检测中"
        SHOW_RESULT = "显示结果"
        WAITING_FOR_CONFIRM = "等待确认"  # NG弹窗等待D1/D2确认
        STOPPED = "已停止"  # 停止后等待复位
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

    ng_confirm_requested = pyqtSignal(int, object, int, int)
    """NG 手工确认请求信号 (位置索引, PositionResult, 当前第几个NG, NG总数) - 逐点位确认"""

    ng_confirm_closed = pyqtSignal()
    """NG 确认完成信号 - 确认后发射，通知 UI 关闭弹窗"""

    reset_during_confirm = pyqtSignal()
    """复位时发射，通知 UI 关闭 NG 确认弹窗"""

    barcode_failed = pyqtSignal()
    """扫码失败信号 - 扫码超时或返回 NG 时发射，通知 UI 弹出提示"""

    stitched_image_ready = pyqtSignal(object)
    """拼接整图更新信号 (np.ndarray) - 每测完一个点位拼入后发射，通知面板刷新"""

    takeout_confirm_requested = pyqtSignal()
    """取出确认请求信号 - 运动到结束位后发射，等待工人按下取出确认按钮"""

    reset_requested = pyqtSignal()
    """复位请求信号 - 硬件复位按钮按下时发射，通知主窗口执行自动回零"""

    home_completed = pyqtSignal()
    """回零完成信号 - 主窗口回零完成后发射，通知工作流继续执行（如再次跑到起始位）"""

    motion_state_changed = pyqtSignal(str)
    """运动状态变化信号 (描述文本)"""

    log_message = pyqtSignal(str)
    """执行日志信号 (文本) - 工作流核心步骤日志，供 UI 执行日志框显示"""

    def __init__(self,
                 camera_mgr=None, vision_engine=None,
                 config: Optional[WorkflowConfig] = None,
                 parent=None):
        """
        初始化工作流

        Args:
            camera_mgr: CameraManager 实例
            vision_engine: VisionEngine 实例
            config: 工作流配置
            parent: QObject 父对象
        """
        super().__init__(parent)
        self._camera_mgr = camera_mgr
        self._vision_engine = vision_engine
        self._config = config or WorkflowConfig()

        # 产品配置
        self._product_config: Optional[Dict[str, Any]] = None
        self._pipelines: List = []  # 每个位置对应的 Pipeline

        # 状态
        self._state = self.State.IDLE
        self._running = False
        # DI 轮询是否运行（与 _running 解耦：停止后仍可监听复位按钮）
        self._di_polling = False

        # 自动确认模式（自动测试用）：NG 直接判 NG，不弹窗人工确认
        self._auto_confirm = False

        # 触发后延时（等待工件放稳）
        self._start_delay_timer = QTimer(self)
        self._start_delay_timer.setSingleShot(True)
        self._start_delay_timer.timeout.connect(self._on_start_delay_elapsed)

        # 当前执行状态
        self._current_pos_index = 0
        self._results: List[PositionResult] = []

        # NG 逐点位确认状态
        self._pending_ng_confirm: List[int] = []  # 待确认的 NG 点位索引列表
        self._current_confirm_index = 0           # 当前确认到第几个 NG 点位

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
        # 扫码超时定时器
        self._scan_timer = QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.timeout.connect(self._on_scan_timeout)

        # ── 图像拼接相关 ──
        self._stitcher = None            # RigidStitcher 实例
        self._stitched_image = None      # 当前拼接整图（标注版）
        self._stitch_scale = 1.0         # 轴坐标 → 像素换算比例

        # ── 轴运动控制相关 ──
        self._controller = None          # Controller 实例（由主窗口注入）
        self._motion_enabled = False     # 是否启用轴运动
        self._motion_axis_x = 0          # X 轴号
        self._motion_axis_y = 1          # Y 轴号
        self._motion_poll_timer = QTimer(self)
        self._motion_poll_timer.timeout.connect(self._on_motion_poll)
        self._motion_target = None       # 当前运动目标 (x, y)
        self._motion_callback = None     # 运动到位后的回调
        self._motion_timeout_ms = 10000  # 运动超时
        self._motion_start_time = 0.0
        self._takeout_pending = False    # 是否等待取出确认
        # 测试结束后是否自动回零再回起始位（回零完成后再次跑到起始位等待下次测试）
        self._auto_home_after_test = False
        # 取出确认方式（由产品方案配置）："auto"=自动(取出传感器) / "manual"=手动(取出按键)
        self._takeout_mode = "manual"
        # 取出确认后延时回起始位的定时器（自动判定 2s，手动判定直接回）
        self._takeout_delay_timer = QTimer(self)
        self._takeout_delay_timer.setSingleShot(True)
        self._takeout_delay_timer.timeout.connect(self._on_takeout_delay_elapsed)

        # ── DI 触发相关 ──
        self._di_bit = 0                 # 触发用 DI 输入位（1-based，IN2=2）
        self._di_prev_state = False      # 上一次 DI 状态（用于检测上升沿）
        self._di_poll_timer = QTimer(self)
        self._di_poll_timer.timeout.connect(self._on_di_poll)
        self._di_poll_interval_ms = 50   # DI 轮询间隔

        # ── 多按钮 IO 配置（1-based IN 编号）──
        self._io_config = {}             # 产品配置中的 io 字段
        self._io_prev_states = {}        # 各按钮上一次状态 {功能名: bool}
        self._io_ports = {}              # 各按钮对应的端口号 {功能名: 端口号(0-based)}
        self._io_actions = {}            # 各按钮对应的动作 {功能名: 回调}
        self._io_out_ports = {}          # 输出端口 {功能名: 端口号(0-based)}

        # ── 检测工作线程 ──
        self._worker = None              # 当前检测工作线程（InspectionWorker）

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

        # ── 初始化图像拼接器（基于产品配置的 grid 与点位坐标）──
        self._init_stitcher(product_config)

        # ── 初始化轴运动参数 ──
        self._init_motion_config(product_config)

        # ── 初始化 DI 触发配置 ──
        self._di_bit = int(product_config.get("di_bit", 0))
        self._di_prev_state = False
        log_info(f"DI 触发位: IN{self._di_bit}")

        # ── 初始化多按钮 IO 配置 ──
        self._init_io_config(product_config)

        log_info(f"产品配置已加载: {product_config.get('name', '未知')} "
                 f"({len(positions)}个位置)")
        return True

    def _init_io_config(self, product_config: Dict[str, Any]):
        """读取产品配置中的 IO 配置，初始化各按钮的端口号与动作。

        io 字段格式（1-based IN 编号）:
            start: 启动按钮
            stop: 停止按钮
            reset: 复位按钮
            rejudge_ok: 复判OK按钮
            rejudge_ng: 复判NG按钮
            unload_sensor: 下料感应
            unload_btn: 下料按钮
            red_light: 红灯输出
            green_light: 绿灯输出
        """
        io = product_config.get("io", {}) or {}
        self._io_config = io
        self._io_prev_states = {}
        self._io_ports = {}

        # 读取取出确认方式（由产品方案配置）
        #   "auto"  = 自动判定：以取出传感器(unload_sensor)为准，拿出即取出，等待2s回起始位
        #   "manual" = 手动判定：以取出按键(unload_btn)为准，按下即取出，直接回起始位
        self._takeout_mode = str(product_config.get("takeout_mode", "manual")).lower()
        if self._takeout_mode not in ("auto", "manual"):
            log_warning(f"未知取出确认方式: {self._takeout_mode}，使用默认 manual")
            self._takeout_mode = "manual"
        log_info(f"取出确认方式: {'自动(取出传感器)' if self._takeout_mode == 'auto' else '手动(取出按键)'}")

        # 输入按钮（DI）→ 端口号（0-based）
        input_buttons = {
            "start": self.start_inspection,
            "stop": self.stop_monitoring,
            "reset": self.reset_error,
            "rejudge_ok": lambda: self.confirm_ng_result(True),
            "rejudge_ng": lambda: self.confirm_ng_result(False),
            "unload_sensor": self._on_unload_sensor,
            "unload_btn": self._on_unload_btn,
        }
        # 根据配置读取各输入按钮的端口号，并绑定对应动作
        for name, action in input_buttons.items():
            in_num = io.get(name)
            if in_num:
                port = max(0, int(in_num) - 1)  # 1-based → 0-based
                self._io_ports[name] = port
                # 初始化上一次状态为 False（低电平），并绑定动作回调
                self._io_prev_states[name] = False
                self._io_actions[name] = action

        # 输出端口（DO）→ 端口号（0-based）
        self._io_out_ports = {}
        for name in ("red_light", "green_light"):
            out_num = io.get(name)
            if out_num:
                self._io_out_ports[name] = max(0, int(out_num) - 1)

        log_info(f"IO 配置已加载: 输入按钮={list(self._io_ports.keys())}, "
                 f"输出端口={list(self._io_out_ports.keys())}")

    def _init_stitcher(self, product_config: Dict[str, Any]):
        """根据产品配置初始化拼接器。

        拼接采用行列网格紧密排列（_add_to_stitch 中按 row/col 计算像素坐标），
        因此 RigidStitcher 的 scale 固定为 1.0（坐标已是像素）。
        """
        from vision.stitch import RigidStitcher

        positions = product_config.get("positions", [])
        self._stitch_scale = 1.0
        self._stitcher = RigidStitcher(scale=1.0)
        self._stitched_image = None
        log_info(f"拼接器已初始化 (网格拼接, 点位数={len(positions)})")

    def _init_motion_config(self, product_config: Dict[str, Any]):
        """读取产品配置中的双轴运动参数。"""
        motion = product_config.get("motion", {}) or {}
        self._motion_axis_x = motion.get("x_axis", 0)
        self._motion_axis_y = motion.get("y_axis", 1)
        x_cfg = motion.get("x", {}) or {}
        y_cfg = motion.get("y", {}) or {}
        self._motion_timeout_ms = int(
            x_cfg.get("move_timeout_s", y_cfg.get("move_timeout_s", 10)) * 1000)
        log_info(f"运动参数: X轴={self._motion_axis_x}, Y轴={self._motion_axis_y}")

    # ── 生命周期控制 ──

    def start_monitoring(self):
        """开始监听（等待手动触发）"""

        # 若已在运行中，则不重复启动
        if self._running:
            log_warning("工作流已在运行中")
            return

        # 检查是否已加载产品配置
        if self._product_config is None:
            self.error_occurred.emit("未加载产品配置")
            return

        # 工作流设置为运行中，等待按下按键触发
        self._running = True
        self._di_polling = True
        self._trigger_count = 0
        self._ok_count = 0
        self._ng_count = 0


        self._set_state(self.State.MONITORING)
        log_info("开始监听（等待手动触发）")

        # 若已连接控制器，启动 DI 轮询触发
        if self._controller is not None and self._controller.is_connected:
            # 先读取各按钮当前状态作为初始状态，避免已是高电平时误判为上升沿
            for name, port in self._io_ports.items():
                try:
                    self._io_prev_states[name] = self._controller.read_in_port(port)
                except Exception as e:  # noqa: BLE001
                    log_warning(f"读取 IO [{name}] 初始状态失败: {e}")
                    self._io_prev_states[name] = False
            # 兼容旧配置：di_bit 初始状态
            if "start" not in self._io_ports and self._di_bit > 0:
                port = max(0, self._di_bit - 1)
                try:
                    self._di_prev_state = self._controller.read_in_port(port)
                except Exception as e:  # noqa: BLE001
                    log_warning(f"读取 DI 初始状态失败: {e}")
                    self._di_prev_state = False
            self._di_poll_timer.start(self._di_poll_interval_ms)
            log_info(f"DI 轮询已启动 (按钮={list(self._io_ports.keys())})")

    def stop_monitoring(self):
        """停止监听：停止所有动作，但继续监听 IO（等待复位）。

        按下 STOP 后：停止所有轴、检测线程、定时器，但保持 DI 轮询运行，
        以便能继续检测复位按钮。状态进入 STOPPED，等待复位回到监听态。
        _running 置为 False（表示不再运行/监听，允许重新加载产品），
        但 _di_polling 保持 True，DI 轮询继续运行以监听复位按钮。
        """
        self._start_delay_timer.stop()
        self._scan_timer.stop()
        self._motion_poll_timer.stop()
        self._takeout_delay_timer.stop()
        # 停止运行标志（允许重新加载产品），但保持 DI 轮询监听复位按钮
        self._running = False
        self._di_polling = True

        # 停止检测工作线程（若正在拍照/检测）
        self._stop_worker()

        # 停止所有轴运动（立即停止，轴停在原地）
        if self._controller is not None and self._controller.is_connected:
            for iaxis in (self._motion_axis_x, self._motion_axis_y):
                try:
                    self._controller.imd_stop(iaxis)
                    log_info(f"停止轴 {iaxis} 运动")
                except Exception as e:  # noqa: BLE001
                    log_warning(f"停止轴 {iaxis} 失败: {e}")

        self._motion_target = None
        self._motion_callback = None
        self._set_state(self.State.STOPPED)
        log_info("已停止（等待复位）")

    def _stop_worker(self):
        """停止当前检测工作线程。"""
        if self._worker is not None:
            try:
                if self._worker.isRunning():
                    self._worker.requestInterruption()
                    self._worker.wait(2000)
            except Exception as e:  # noqa: BLE001
                log_warning(f"停止检测线程失败: {e}")
            self._worker = None

    def emergency_stop(self):
        """紧急停止"""
        self._start_delay_timer.stop()
        self._scan_timer.stop()
        self._di_poll_timer.stop()
        self._takeout_delay_timer.stop()
        self._motion_poll_timer.stop()
        self._running = False
        self._di_polling = False
        # 停止检测工作线程
        self._stop_worker()
        # 停止所有轴运动
        if self._controller is not None and self._controller.is_connected:
            for iaxis in (self._motion_axis_x, self._motion_axis_y):
                try:
                    self._controller.imd_stop(iaxis)
                except Exception as e:  # noqa: BLE001
                    log_warning(f"停止轴 {iaxis} 失败: {e}")
        self._set_state(self.State.IDLE)
        log_info("紧急停止")

    def _on_di_poll(self):
        """DI 轮询：检测各输入按钮的上升沿（0→1），执行对应动作。"""
        if not self._di_polling:
            self._di_poll_timer.stop()
            return
        if self._controller is None or not self._controller.is_connected:
            self._di_poll_timer.stop()
            return

        # 轮询所有输入按钮
        for name, port in self._io_ports.items():
            try:
                current = self._controller.read_in_port(port)  # 读取当前状态
            except Exception as e:  # noqa: BLE001
                log_warning(f"读取 IO [{name}] 失败: {e}")
                continue

            prev = self._io_prev_states.get(name, False)
            # 检测上升沿（从低到高）
            if current and not prev:
                # 检测到按钮按下，执行对应动作，action 是一个回调函数
                action = self._io_actions.get(name)
                if action:
                    log_info(f"检测到按钮触发: {name} (IN{port + 1})")
                    try:
                        action()
                    except Exception as e:  # noqa: BLE001
                        log_error(f"执行按钮 [{name}] 动作失败: {e}")
            self._io_prev_states[name] = current

        # 兼容旧配置：若未配置 io.start，仍用 di_bit 触发
        if "start" not in self._io_ports and self._di_bit > 0:
            port = max(0, self._di_bit - 1)
            try:
                current = self._controller.read_in_port(port)
            except Exception as e:  # noqa: BLE001
                log_warning(f"读取 DI 输入失败: {e}")
                return
            if current and not self._di_prev_state:
                log_info(f"检测到 DI 触发 (IN{self._di_bit})")
                self.start_inspection()
            self._di_prev_state = current

    # ── 下料相关 ──

    def _on_unload_sensor(self):
        """取出传感器触发（自动判定取出确认）。

        仅在"已到结束位等待取出"（_takeout_pending=True）且取出方式为自动时生效。
        传感器检测到工件被拿出后，等待 2s 回起始位。
        """
        if not self._takeout_pending:
            return
        if self._takeout_mode != "auto":
            return
        log_info("取出传感器检测到工件已取出，等待 2s 回起始位")
        self.log_message.emit("取出传感器检测到工件已取出，等待 2s 回起始位")
        self._takeout_pending = False
        self._takeout_delay_timer.start(2000)

    def _on_unload_btn(self):
        """取出按键触发（手动判定取出确认）。

        仅在"已到结束位等待取出"（_takeout_pending=True）且取出方式为手动时生效。
        按下取出按键后直接回起始位。
        """
        if not self._takeout_pending:
            return
        if self._takeout_mode != "manual":
            return
        log_info("取出按键已按下，返回起始位")
        self.log_message.emit("取出按键已按下，返回起始位")
        self._takeout_pending = False
        self._move_to_start(callback=self._on_returned_to_start)

    def _on_takeout_delay_elapsed(self):
        """取出确认延时结束（自动判定 2s 后） - 返回起始位。"""
        log_info("取出确认延时结束，返回起始位")
        self.log_message.emit("取出确认延时结束，返回起始位")
        self._move_to_start(callback=self._on_returned_to_start)

    # ── 输出端口控制（红灯/绿灯）──

    def set_red_light(self, on: bool):
        """控制红灯亮灭。"""
        port = self._io_out_ports.get("red_light")
        if port is None or self._controller is None:
            return
        try:
            self._controller.set_out_port(port, on)
            log_info(f"红灯 {'亮' if on else '灭'}")
        except Exception as e:  # noqa: BLE001
            log_error(f"控制红灯失败: {e}")

    def set_green_light(self, on: bool):
        """控制绿灯亮灭。"""
        port = self._io_out_ports.get("green_light")
        if port is None or self._controller is None:
            return
        try:
            self._controller.set_out_port(port, on)
            log_info(f"绿灯 {'亮' if on else '灭'}")
        except Exception as e:  # noqa: BLE001
            log_error(f"控制绿灯失败: {e}")

    def reset_error(self):
        """复位：回到初始态，等待重新测试。

        - 从 ERROR 状态：复位回 IDLE
        - 从 STOPPED 状态（按下 STOP 后）：复位回 IDLE，等待重新启动
        复位后回到 IDLE（_running=False），允许重新加载产品方案；
        用户需重新点击「启动」才开始监听。
        """
        if self._state == self.State.ERROR:
            self._running = False
            self._di_polling = False
            self._set_state(self.State.IDLE)
            log_info("错误已复位")
        elif self._state == self.State.STOPPED:
            # 从停止状态复位回 IDLE，允许重新加载产品方案
            self._running = False
            self._di_polling = False
            self._set_state(self.State.IDLE)
            log_info("已复位，等待重新启动")

        # 无论什么状态，复位时都发射 reset_requested 信号，通知 UI 执行自动回零
        self.reset_requested.emit()

    # ── 手动触发 ──

    def set_auto_confirm(self, enabled: bool):
        """设置自动确认模式（自动测试用）。

        启用后，NG 结果直接判 NG，不弹窗人工确认；OK 结果自动确认取出。
        """
        self._auto_confirm = bool(enabled)

    def  start_inspection(self):
        """ 触发一次检测流程，三种触发方式都通过本函数触发检测流程
        1. 手动按下 START 按钮（IO 配置）
        2. 鼠标点击面板的触发按钮
        3. 自动测试中自动定时触发
        """
        if not self._running:
            log_warning("工作流未在监听状态，无法触发")
            return

        if self._state not in (self.State.MONITORING, self.State.IDLE):
            log_warning(f"当前状态 {self._state.value} 不允许触发")
            return

        # 触发计数
        self._trigger_count += 1
        self.trigger_count_changed.emit(self._trigger_count)

        # 重置当前执行状态
        self._current_pos_index = 0
        self._results = []
        self._barcode_data = None
        self._takeout_pending = False

        # 重置拼接器（清空上一轮点位图）
        if self._stitcher is not None:
            self._stitcher.reset()
            self._stitched_image = None

        # 记录本次检测开始时间（用于计算总耗时）
        import time
        self._inspection_start_time = time.time()

        # 延时后开始检测，等待工件放稳
        log_info(f"触发检测，等待 {self._config.start_delay_ms}ms 后开始...")
        self.log_message.emit(f"触发检测 #{self._trigger_count}，等待 {self._config.start_delay_ms}ms 后开始...")
        self._set_state(self.State.WAITING)
        # 等待默认1000ms后开始检测，避免工件放置不稳导致拍照模糊
        self._start_delay_timer.start(self._config.start_delay_ms)

    # ── 状态管理 ──

    def _set_state(self, new_state: State):
        """安全切换状态"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            log_info(f"工作流: {old_state.value} -> {new_state.value}")
            self.state_changed.emit(new_state)

    def _on_start_delay_elapsed(self):
        """延时结束 - 若启用轴运动先运动到起始位，然后开始检测流程"""
        # 若启用轴运动，先运动到起始位
        if self._motion_enabled and self._controller is not None:
            log_info("延时结束，运动到起始位")
            self.log_message.emit("延时结束，运动到起始位")
            self._move_to_start(callback=self._on_reached_start)
            return

        # 检查是否启用了扫码功能
        barcode_cfg = self._product_config.get("barcode_scan", {})
        if barcode_cfg.get("enabled", False):
            log_info("延时结束，开始扫码")
            self.log_message.emit("延时结束，开始扫码")
            self._start_scan()
        else:
            log_info("延时结束，开始执行检测流程")
            self.log_message.emit("延时结束，开始执行检测流程")
            self._execute_current_position()

    def _on_reached_start(self):
        """已运动到起始位，开始检测流程。"""
        barcode_cfg = self._product_config.get("barcode_scan", {})
        if barcode_cfg.get("enabled", False):
            log_info("已到起始位，开始扫码")
            self._start_scan()
        else:
            log_info("已到起始位，开始执行检测流程")
            self._execute_current_position()

    # ── 位置执行 ──

    def _execute_current_position(self):
        """执行当前位置的检测"""
        positions = self._product_config.get("positions", [])
        if self._current_pos_index >= len(positions):
            # 所有位置已完成，显示结果
            self._show_final_result()
            return

        pos = positions[self._current_pos_index]
        log_info(f"检测位置 {self._current_pos_index + 1}: {pos.get('name', '')}")
        self.log_message.emit(f"检测位置 {self._current_pos_index + 1}/{len(positions)}: {pos.get('name', '')}")

        # 若启用轴运动，先运动到该点位（行优先顺序由 positions 顺序保证）
        if self._motion_enabled and self._controller is not None:
            x = pos.get("x", 0)
            y = pos.get("y", 0)
            self._move_to(x, y, callback=self._capture)
        else:
            self._capture()

    # ── 拍照 ──

    def _capture(self):
        """拍照 + 检测（在工作线程执行，避免阻塞主线程导致 DI 轮询暂停）。"""
        self._set_state(self.State.CAPTURING)
        self.log_message.emit(f"拍照中... (位置 {self._current_pos_index + 1})")

        if self._camera_mgr is None:
            self._on_error("相机管理器未初始化")
            return

        # 获取当前位置对应的流水线
        pipeline = None
        if self._current_pos_index < len(self._pipelines):
            pipeline = self._pipelines[self._current_pos_index]

        # 启动工作线程执行拍照 + 检测
        camera_cfg = self._product_config.get("camera", {}) if self._product_config else {}
        scheme_name = self._product_config.get("name", "未知") if self._product_config else "未知"

        worker = InspectionWorker(
            camera_mgr=self._camera_mgr,
            vision_engine=self._vision_engine,
            pipeline=pipeline,
            scheme_name=scheme_name,
            camera_cfg=camera_cfg,
            parent=self,
        )
        worker.capture_failed.connect(self._on_worker_capture_failed)
        worker.test_done.connect(self._on_worker_test_done)
        worker.test_failed.connect(self._on_worker_test_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_worker_capture_failed(self, message: str):
        """工作线程拍照失败。"""
        self._on_error(message)

    def _on_worker_test_done(self, passed: bool, message: str,
                             annotated, raw_image, tool_results):
        """工作线程检测完成（主线程处理拼接、状态更新）。"""
        self._on_test_completed(passed, message, annotated, raw_image, tool_results)

    def _on_worker_test_failed(self, message: str):
        """工作线程检测异常。"""
        self._on_error(f"检测异常: {message}")

    def _on_test_completed(self, passed: bool, message: str,
                           annotated: np.ndarray, raw_image: np.ndarray,
                           tool_results: list):

        """回调函数处理停止状态，避免在停止后继续处理结果"""
        if self._state == self.State.STOPPED:
            log_info("检测完成，但工作流已停止，忽略结果")
            return
        """检测完成回调"""
        positions = self._product_config.get("positions", [])
        pos = positions[self._current_pos_index] if self._current_pos_index < len(positions) else {"name": f"位置{self._current_pos_index + 1}"}

        # 从检测结果中提取 QR 识别数据（作为该板卡 SN）
        qr_data = self._extract_qr_data(tool_results)

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
        # 附加 QR SN 到结果对象
        result.qr_data = qr_data
        # 提取条码详情（类型/内容/坐标/置信度）到结果对象，供结果面板展示
        result.barcodes = self._extract_barcodes(tool_results)
        self._results.append(result)

        log_info(f"位置 {self._current_pos_index + 1} [{result.name}]: {'OK' if passed else 'NG'} | {message}"
                 + (f" | SN={qr_data}" if qr_data else ""))
        self.log_message.emit(f"位置 {self._current_pos_index + 1} [{result.name}]: "
                              f"{'OK' if passed else 'NG'} | {message}"
                              + (f" | SN={qr_data}" if qr_data else ""))

        # 发射单个位置结果信号
        self.position_result_ready.emit(self._current_pos_index, result)

        # 将该点位标注图拼入整体图并刷新显示
        self._add_to_stitch(result, pos)

        # 立即保存该点位数据（照片 + XML），此时图像仍在内存中
        self._save_position_data(result, pos)

        # 拼入拼接器后立即释放该张板卡的原始图/标注图，降低内存占用
        # （拼接器已保存缩放后的图像，结果面板已通过信号拿到图像，后续不再需要）
        result.annotated = None
        result.raw_image = None
        result.tool_results = []

        # 移动到下一个位置
        self._current_pos_index += 1
        self._execute_current_position()

    def _extract_qr_data(self, tool_results: list) -> str:
        """从检测结果中提取 QR 识别数据（SN）。

        遍历所有工具结果，查找 QRCodeRecognize 的输出 data["qr_data"]。
        返回第一个非空识别结果。

        Args:
            tool_results: 工具检测结果列表

        Returns:
            str: 识别到的 SN，未识别到返回 ""
        """
        if not tool_results:
            return ""
        for r in tool_results:
            try:
                data = getattr(r, "data", None) or {}
                qr = data.get("qr_data", "")
                if qr:
                    return str(qr).strip()
            except Exception:  # noqa: BLE001
                continue
        return ""

    def _extract_barcodes(self, tool_results: list) -> list:
        """从检测结果中提取条码详情列表（类型/内容/坐标/置信度）。

        遍历所有工具结果，查找条码识别算子的输出 data["barcodes"]。

        Args:
            tool_results: 工具检测结果列表

        Returns:
            list: 条码详情列表，每个元素含 type/data/confidence/bbox
        """
        if not tool_results:
            return []
        for r in tool_results:
            try:
                data = getattr(r, "data", None) or {}
                barcodes = data.get("barcodes", [])
                if barcodes:
                    return list(barcodes)
            except Exception:  # noqa: BLE001
                continue
        return []

    def _add_to_stitch(self, result: PositionResult, pos: dict):
        """将单个点位标注图按行列网格拼入整体图，并发射刷新信号。

        拼接采用行列网格紧密排列（板卡之间无空隙），第一行在上、第二行在下，
        避免轴坐标间距与图像尺寸不匹配导致的空隙和顺序颠倒。

        Args:
            result: 该点位的检测结果
            pos: 该点位的配置（含 row/col）
        """
        if self._stitcher is None:
            return
        image = result.annotated if result.annotated is not None else result.raw_image
        if image is None:
            return

        # 将板卡图像缩放到合适尺寸
        image = self._scale_board_image(image)

        # 按行列网格计算位置（紧密排列，第一行在上）
        row = pos.get("row", 1)
        col = pos.get("col", 1)
        h, w = image.shape[:2]
        # 网格坐标：col 向右，row 向下（第一行在上）
        gx = (col - 1) * w
        gy = (row - 1) * h

        self._stitcher.add_board(
            name=result.name, x_axis=gx, y_axis=gy,
            image=image, passed=result.passed
        )
        # 生成带 OK/NG 标注的拼接整图
        self._stitched_image = self._stitcher.render_annotated()
        if self._stitched_image is not None:
            self.stitched_image_ready.emit(self._stitched_image)

    def _scale_board_image(self, image: np.ndarray) -> np.ndarray:
        """将板卡图像缩放到合适尺寸，控制拼接整图尺寸。

        轴坐标由 RigidStitcher 的 scale 统一换算为像素，这里只缩放板卡图像，
        使其最长边不超过 MAX_BOARD_SIZE，避免拼接整图过大导致内存溢出。

        Args:
            image: 原始板卡标注图

        Returns:
            缩放后的板卡图像
        """
        MAX_BOARD_SIZE = 800  # 单块板卡图像最长边上限（像素）

        h, w = image.shape[:2]
        if max(h, w) > MAX_BOARD_SIZE:
            s = MAX_BOARD_SIZE / max(h, w)
            image = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)
        return image

    # ── 显示结果 ──

    def _show_final_result(self):
        """显示最终结果"""
        self._set_state(self.State.SHOW_RESULT)
        # 标记本次测试结束后需要自动回零再回起始位（回零完成后再次跑到起始位等待下次测试）
        self._auto_home_after_test = True

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
            # OK：更新统计并继续（各点位数据已在检测完成时保存）
            self._ok_count += 1
            self.ok_count_changed.emit(self._ok_count)
            # 发射最终结果信号
            self.all_results_ready.emit(True, self._results)
            # 释放结果图像，降低内存占用
            self._release_result_images()
            log_info(f"最终结果: OK "
                     f"(触发: {self._trigger_count}, OK: {self._ok_count}, NG: {self._ng_count})"
                     f" | 总耗时: {total_elapsed:.2f}s")
            self.log_message.emit(f"最终结果: OK | 总耗时: {total_elapsed:.2f}s")
            # 若启用轴运动：运动到结束位，等待取出确认
            # 自动确认模式（自动测试）下：跳过取出确认，直接返回起始位
            if self._auto_confirm:
                if self._motion_enabled and self._controller is not None:
                    log_info("自动确认模式：OK 直接返回起始位")
                    self._move_to_start(callback=self._on_returned_to_start)
                else:
                    self._set_state(self.State.MONITORING)
            elif self._motion_enabled and self._controller is not None:
                self._move_to_end(callback=self._on_reached_end_ok)
            else:
                self._set_state(self.State.MONITORING)
        else:
            # NG：若为自动确认模式（自动测试），直接判 NG，不弹窗人工确认
            if self._auto_confirm:
                log_info(f"检测结果为 NG（自动确认模式，直接判 NG）| 总耗时: {total_elapsed:.2f}s")
                self.log_message.emit(f"最终结果: NG | 总耗时: {total_elapsed:.2f}s")
                self._handle_ng_auto()
            else:
                # 逐点位确认：收集所有 NG 点位，逐个弹窗让操作员确认
                log_info(f"检测结果为 NG，开始逐点位手工确认... | 总耗时: {total_elapsed:.2f}s")
                self.log_message.emit(f"最终结果: NG，开始逐点位手工确认 | 总耗时: {total_elapsed:.2f}s")
                self._pending_ng_confirm = [
                    i for i, r in enumerate(self._results) if not r.passed
                ]
                self._current_confirm_index = 0
                self._set_state(self.State.WAITING_FOR_CONFIRM)  # 进入等待确认状态
                self._request_next_ng_confirm()

    def _handle_ng_auto(self):
        """自动确认模式下的 NG 处理：直接判 NG，不弹窗人工确认。

        各点位数据已在检测完成时保存，此处仅更新统计并发射信号。
        """
        self._ng_count += 1
        self.ng_count_changed.emit(self._ng_count)
        log_info("自动确认: NG")
        self.all_results_ready.emit(False, self._results)
        # 释放结果图像，降低内存占用
        self._release_result_images()
        self.ng_confirm_closed.emit()
        # 运动控制：返回起始位
        if self._motion_enabled and self._controller is not None:
            self._move_to_start(callback=self._on_returned_to_start)
        else:
            self._set_state(self.State.MONITORING)

    def _on_reached_end_ok(self):
        """OK 流程：已运动到结束位，等待取出确认。

        根据产品方案配置的取出确认方式（takeout_mode）：
            - "auto"  = 自动判定：等待取出传感器(unload_sensor)检测到工件被拿出
            - "manual" = 手动判定：等待取出按键(unload_btn)被按下
        同时保留软件「取出确认」按钮（confirm_takeout）作为兜底。
        """
        self._takeout_pending = True
        if self._takeout_mode == "auto":
            self.motion_state_changed.emit("已到结束位，等待取出传感器检测")
            self.log_message.emit("已到结束位，等待取出传感器检测（自动判定）")
        else:
            self.motion_state_changed.emit("已到结束位，等待按下取出按键")
            self.log_message.emit("已到结束位，等待按下取出按键（手动判定）")
        self.takeout_confirm_requested.emit()

    def confirm_ng_result(self, confirmed_ok: bool):
        """
        NG 逐点位确认结果回调 - 由 UI 层在弹窗确认后调用。

        确认当前 NG 点位后，更新该点位最终判定与 XML，然后请求确认下一个 NG 点位；
        所有 NG 点位确认完成后，计算最终结果并继续流程。

        Args:
            confirmed_ok: True 表示操作员确认为 OK，False 表示确认为 NG
        """
        if self._state != self.State.WAITING_FOR_CONFIRM:
            log_warning(f"工作流状态不是 WAITING_FOR_CONFIRM，忽略确认回调 (当前: {self._state.value})")
            return

        if self._current_confirm_index >= len(self._pending_ng_confirm):
            log_warning("没有待确认的 NG 点位，忽略确认回调")
            return

        # 当前待确认的 NG 点位
        idx = self._pending_ng_confirm[self._current_confirm_index]
        result = self._results[idx]

        # 更新该点位最终判定
        result.passed = confirmed_ok
        result.confirmed = True
        log_info(f"点位 [{result.name}] 手工确认: {'OK' if confirmed_ok else 'NG'}")

        # 更新该点位 XML 判定
        self._update_position_xml(result)

        # 通知 UI 关闭当前确认弹窗
        self.ng_confirm_closed.emit()

        # 请求确认下一个 NG 点位
        self._current_confirm_index += 1
        self._request_next_ng_confirm()

    def _request_next_ng_confirm(self):
        """请求确认下一个 NG 点位；全部确认完成后进入最终结果处理。"""
        if self._current_confirm_index >= len(self._pending_ng_confirm):
            # 所有 NG 点位确认完成
            self._finish_ng_confirm()
            return

        idx = self._pending_ng_confirm[self._current_confirm_index]
        result = self._results[idx]
        current = self._current_confirm_index + 1
        total = len(self._pending_ng_confirm)
        log_info(f"请求确认 NG 点位 {current}/{total}: [{result.name}]")
        self.ng_confirm_requested.emit(idx, result, current, total)

    def _finish_ng_confirm(self):
        """所有 NG 点位确认完成，计算最终结果并继续流程。"""
        # 重新计算最终结果（确认后可能改变判定）
        all_passed = all(r.passed for r in self._results)

        if all_passed:
            self._ok_count += 1
            self.ok_count_changed.emit(self._ok_count)
            log_info("逐点位确认完成，最终结果: OK")
            self.log_message.emit("逐点位确认完成，最终结果: OK")
            self.all_results_ready.emit(True, self._results)
            # 运动控制：OK → 运动到结束位等待取出
            if self._motion_enabled and self._controller is not None:
                self._move_to_end(callback=self._on_reached_end_ok)
            else:
                self._set_state(self.State.MONITORING)
        else:
            self._ng_count += 1
            self.ng_count_changed.emit(self._ng_count)
            log_info("逐点位确认完成，最终结果: NG")
            self.log_message.emit("逐点位确认完成，最终结果: NG")
            self.all_results_ready.emit(False, self._results)
            # 运动控制：NG → 返回起始位
            if self._motion_enabled and self._controller is not None:
                self._move_to_start(callback=self._on_returned_to_start)
            else:
                self._set_state(self.State.MONITORING)

        # 释放结果图像，降低内存占用
        self._release_result_images()
        log_info(f"最终结果: {'OK' if all_passed else 'NG'} "
                 f"(触发: {self._trigger_count}, OK: {self._ok_count}, NG: {self._ng_count})")

    def _save_position_data(self, result: PositionResult, pos: dict):
        """保存单个点位（板卡）的数据：照片 + XML。

        每个点位即一块独立板卡，拥有独立 SN 号（QR 识别结果）。
        在检测完成、图像仍在内存中时立即调用，避免图像被释放后无法保存。

        Args:
            result: 该点位的检测结果
            pos: 该点位的配置
        """
        from core.result_storage import ResultStorage

        product_name = self._product_config.get("name", "未知产品") if self._product_config else "未知产品"

        image = result.annotated if result.annotated is not None else result.raw_image
        if image is None:
            log_warning(f"位置 [{result.name}] 无图像，跳过保存")
            return

        try:
            storage = ResultStorage()
            save_info = storage.save_position_data(
                scheme_name=product_name,
                sn=result.qr_data or "",
                position_name=result.name,
                position_index=self._current_pos_index + 1,
                annotated_image=image,
                passed=result.passed,
            )
            if save_info:
                # 记录保存时间戳，用于后续更新 XML
                result.save_timestamp = save_info.get('timestamp', '')
        except Exception as e:  # noqa: BLE001
            log_error(f"保存位置 [{result.name}] 数据失败: {e}")

    def _update_position_xml(self, result: PositionResult):
        """更新某点位（SN）的 XML 判定结果（NG 手工确认后调用）。

        Args:
            result: 该点位的检测结果（含最终判定）
        """
        from core.result_storage import ResultStorage

        try:
            storage = ResultStorage()
            storage.update_position_result(
                sn=result.qr_data or "",
                passed=result.passed,
                timestamp=result.save_timestamp,
            )
        except Exception as e:  # noqa: BLE001
            log_error(f"更新位置 [{result.name}] XML 判定失败: {e}")

    def _release_result_images(self):
        """释放所有检测结果中的原始图/标注图，降低内存占用。

        各点位数据已在检测完成时通过 _save_position_data 保存到磁盘，
        结果面板仅使用 name/passed/message/elapsed_ms 等元数据，不再需要图像。
        释放后仅保留 SN、结果等轻量数据，避免多张高分辨率图长期占用内存。
        """
        for result in self._results:
            try:
                result.annotated = None
                result.raw_image = None
                result.tool_results = []
            except Exception:  # noqa: BLE001
                pass
        # 提示 GC 回收（可选，不强制）
        try:
            import gc
            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        log_info("已释放检测结果中的图像数据，降低内存占用")

    # ── 轴运动控制 ──

    def set_controller(self, controller):
        """设置轴控制器实例（由主窗口注入）。

        Args:
            controller: Controller 实例，为 None 时禁用轴运动
        """
        self._controller = controller
        self._motion_enabled = controller is not None
        if controller is None:
            self._motion_poll_timer.stop()
        log_info(f"轴控制器已{'注入' if controller is not None else '移除'}，"
                 f"轴运动{'启用' if self._motion_enabled else '禁用'}")

    def _move_to(self, x: float, y: float, callback=None):
        """双轴运动到指定坐标（X、Y 同时运动）。

        Args:
            x: X 轴目标坐标
            y: Y 轴目标坐标
            callback: 运动到位后的回调
        """
        if self._controller is None:
            if callback:
                callback()
            return

        try:
            self._motion_target = (x, y)
            self._motion_callback = callback
            self._motion_start_time = time.time()
            self.motion_state_changed.emit(f"运动到 ({x}, {y})")
            self.log_message.emit(f"运动到 ({x}, {y})")

            # 设置双轴运动参数（start_speed 用较小值，max_speed 用 v_max）
            motion = self._product_config.get("motion", {}) or {}
            x_cfg = motion.get("x", {}) or {}
            y_cfg = motion.get("y", {}) or {}
            x_vmax = x_cfg.get("v_max", 50000)
            y_vmax = y_cfg.get("v_max", 50000)
            x_amax = x_cfg.get("a_max", 100000)
            y_amax = y_cfg.get("a_max", 100000)
            self._controller.set_motion_params(
                self._motion_axis_x,
                min(1000, x_vmax),   # 启动速度
                x_vmax,              # 最大速度
                x_amax,              # 加速度
                x_amax,              # 减速度
            )
            self._controller.set_motion_params(
                self._motion_axis_y,
                min(1000, y_vmax),   # 启动速度
                y_vmax,              # 最大速度
                y_amax,              # 加速度
                y_amax,              # 减速度
            )

            # 双轴绝对定位
            self._controller.pmove_abs(self._motion_axis_x, int(x))
            self._controller.pmove_abs(self._motion_axis_y, int(y))

            # 启动到位轮询
            self._motion_poll_timer.start(100)
        except Exception as e:
            log_error(f"运动到 ({x}, {y}) 失败: {e}")
            if callback:
                callback()

    def _on_motion_poll(self):
        """轮询检查双轴是否运动到位。

        使用位置检测（读取当前坐标与目标坐标比较），而非 check_down（停止检测），
        避免 Y 轴尚未开始运动时被误判为"已到位"。
        """
        if self._controller is None or self._motion_target is None:
            self._motion_poll_timer.stop()
            return

        target_x, target_y = self._motion_target
        tolerance = 50  # 到位容差（脉冲）

        try:
            cur_x = self._controller.get_pulse_position(self._motion_axis_x)
            cur_y = self._controller.get_pulse_position(self._motion_axis_y)
        except Exception as e:
            log_error(f"读取轴位置失败: {e}")
            self._motion_poll_timer.stop()
            self._motion_target = None
            cb = self._motion_callback
            self._motion_callback = None
            if cb:
                cb()
            return

        # 超时判断
        if time.time() - self._motion_start_time > self._motion_timeout_ms / 1000.0:
            log_warning("运动超时，继续执行")
            self._motion_poll_timer.stop()
            self._motion_target = None
            cb = self._motion_callback
            self._motion_callback = None
            if cb:
                cb()
            return

        # 双轴都到达目标位置（容差内）才算到位
        x_done = abs(cur_x - target_x) <= tolerance
        y_done = abs(cur_y - target_y) <= tolerance

        if x_done and y_done:
            self._motion_poll_timer.stop()
            self._motion_target = None
            cb = self._motion_callback
            self._motion_callback = None
            self.motion_state_changed.emit("运动到位")
            if cb:
                cb()

    def _move_to_start(self, callback=None):
        """运动到起始位。"""
        home = self._product_config.get("home", {}) or {}
        self._move_to(home.get("start_x", 0), home.get("start_y", 0), callback=callback)

    def _move_to_end(self, callback=None):
        """运动到结束位。"""
        home = self._product_config.get("home", {}) or {}
        self._move_to(home.get("end_x", 0), home.get("end_y", 0), callback=callback)

    def confirm_takeout(self):
        """工人按下取出确认按钮 - 返回起始位。"""
        if not self._takeout_pending:
            return
        self._takeout_pending = False
        log_info("取出确认，返回起始位")
        self._move_to_start(callback=self._on_returned_to_start)

    def _on_returned_to_start(self):
        """已返回起始位。

        若启用了"测试结束后自动回零"（_auto_home_after_test），
        则发射 reset_requested 触发主窗口回零，回零完成后由
        on_home_completed 再次跑到起始位等待下次测试；
        否则直接进入 MONITORING 等待下次触发。
        """
        self.motion_state_changed.emit("已返回起始位")
        if self._auto_home_after_test:
            log_info("测试结束，返回起始位，开始自动回零")
            self.log_message.emit("测试结束，返回起始位，开始自动回零")
            self.reset_requested.emit()
        else:
            self._set_state(self.State.MONITORING)

    def on_home_completed(self):
        """回零完成回调（由主窗口在回零完成后调用）。

        若设置了"测试结束后自动回零"，回零完成后再次跑到起始位等待下次测试。
        """
        if not self._auto_home_after_test:
            return
        self._auto_home_after_test = False
        log_info("回零完成，再次运动到起始位等待下次测试")
        self.log_message.emit("回零完成，再次运动到起始位等待下次测试")
        self._move_to_start(callback=self._on_returned_to_start)

    # ── 错误处理 ──

    def _on_error(self, error_msg: str):
        """错误处理"""
        log_error(error_msg)
        self.error_occurred.emit(error_msg)
        self._set_state(self.State.ERROR)

    # ── 资源清理 ──

    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        self._start_delay_timer.stop()
        self._scan_timer.stop()
        self._motion_poll_timer.stop()
        self._di_poll_timer.stop()
        self._pipelines = []
        self._product_config = None
        self._results = []
        self._barcode_data = None
        self._stitcher = None
        self._stitched_image = None
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

    def _start_scan(self):
        """开始扫码 - 发送扫描命令触发扫描头"""
        self._set_state(self.State.SCANNING)

        barcode_cfg = self._product_config.get("barcode_scan", {})
        command = barcode_cfg.get("command", "01 54 04")
        timeout_ms = barcode_cfg.get("timeout_ms", self._config.scan_timeout_ms)

        if self._serial_comm is not None and self._serial_comm.is_open:
            # 发送 HEX 扫描命令
            count = self._serial_comm.send_hex(command)
            if count > 0:
                log_info(f"已发送扫描命令: {command} ({count} 字节)")
                self.log_message.emit("扫码中...")
                # 启动超时定时器
                self._scan_timer.start(timeout_ms)
                log_info(f"等待扫码返回 (超时: {timeout_ms}ms)...")
            else:
                log_error("发送扫描命令失败")
                self.log_message.emit("发送扫描命令失败")
                self._on_barcode_failed()
        else:
            log_error("串口未打开，无法扫码")
            self.log_message.emit("串口未打开，无法扫码")
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
                self.log_message.emit(f"扫码成功: {barcode}")
                # 扫码成功，开始执行检测位置
                self._current_pos_index = 0
                self._execute_current_position()
            else:
                log_warning(f"扫码失败: 返回={raw_text!r} (过滤后={barcode!r})")
                self.log_message.emit(f"扫码失败: {barcode!r}")
                self._on_barcode_failed()
        except Exception as e:
            log_error(f"解析一维码失败: {e}")
            self.log_message.emit(f"解析一维码失败: {e}")
            self._on_barcode_failed()

    def _on_scan_timeout(self):
        """扫码超时 - 未收到扫描头返回数据"""
        log_error("扫码超时：未收到一维码数据")
        self.log_message.emit("扫码超时：未收到一维码数据")
        self._on_barcode_failed()

    def _on_barcode_failed(self):
        """扫码失败处理 - 发射扫码失败信号，不保存错误图片"""
        self._barcode_data = None
        log_info("扫码失败，等待重新触发")
        self.log_message.emit("扫码失败，等待重新触发")

        # 更新 NG 计数
        self._ng_count += 1
        self.ng_count_changed.emit(self._ng_count)

        # 发射扫码失败信号（UI 层弹出提示）
        self.barcode_failed.emit()

        # 不保存错误图片，直接回到 MONITORING 状态
        self._set_state(self.State.MONITORING)
