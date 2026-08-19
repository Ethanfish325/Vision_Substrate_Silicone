#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMC3401 运动控制卡对话框
======================
独立的运动控制卡控制窗口，通过菜单栏「通信 > 运动控制」打开。

基于 NMC_SDK_TEST/nmc_gui.py 重构为 QDialog，适配 Vision_Substrate_Silicone 架构。

功能:
    - 连接/断开 NMC3401 运动控制卡
    - IO 监测（数字输入/输出状态实时显示）
    - 轴参数配置（脉冲模式、命令位置、编码器位置、软限位）
    - 回零控制（三阶段速度控制）
    - 点位运动（JOG 点动、单轴运动、曲线参数设置）
"""

import time
from typing import Optional, List

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from core.nmc_sdk import (
    NMCSDK,
    Axis_1, Axis_2, Axis_3, Axis_4,
    Position_Absolute, Position_Opposite,
    Profile_T, Profile_S,
    Servo_Open, Servo_Close,
    Stop_Abrupt, Stop_Smooth,
    Home_Mode_1,
    NMCError, NMCConnectionError,
)
from core.log_manager import log_info, log_error, log_warning


# ============================================================================
# 常量定义
# ============================================================================

# 轴选择列表
AXIS_NAMES = ["Axis_1", "Axis_2", "Axis_3", "Axis_4"]
AXIS_VALUES = [Axis_1, Axis_2, Axis_3, Axis_4]

# 脉冲模式选项
PULSE_MODES = [
    ("脉冲+方向(正逻辑)", 0),
    ("脉冲+方向(负逻辑)", 1),
    ("CW/CCW(正逻辑)", 2),
    ("CW/CCW(负逻辑)", 3),
    ("AB相(4倍频)", 4),
]

# 回零模式
HOME_MODES = [
    ("模式1: 近门狗+Z相", 1),
    ("模式2: 近门狗+索引", 2),
    ("模式3: Z相", 3),
    ("模式4: 索引", 4),
    ("模式5: 正限位+Z相", 5),
    ("模式6: 负限位+Z相", 6),
]

# 停止模式
STOP_MODES = [
    ("急停(立即停止)", Stop_Abrupt),
    ("平滑停止", Stop_Smooth),
]


# ============================================================================
# LED 指示灯组件
# ============================================================================

class LedIndicator(QWidget):
    """LED 指示灯组件"""

    def __init__(self, text: str = "", color: str = "#00FF00", parent=None):
        super().__init__(parent)
        self._text = text
        self._color = color
        self._on = False
        self.setFixedSize(16, 16)

    def set_on(self, on: bool):
        self._on = on
        self.update()

    def set_color(self, color: str):
        self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(self._color) if self._on else QColor("#333333")
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawEllipse(1, 1, 14, 14)


# ============================================================================
# 日志处理器（适配 Vision_Substrate_Silicone 的 log_manager）
# ============================================================================

class _DialogLogHandler:
    """对话框内部日志处理器，将日志同时输出到 log_manager 和 UI 控件"""

    def __init__(self, log_edit: QTextEdit):
        self.log_edit = log_edit

    def info(self, msg: str):
        log_info(f"[NMC] {msg}")
        self._append(f"[INFO] {msg}", "#4fc3f7")

    def warn(self, msg: str):
        log_warning(f"[NMC] {msg}")
        self._append(f"[WARN] {msg}", "#ffa726")

    def error(self, msg: str):
        log_error(f"[NMC] {msg}")
        self._append(f"[ERROR] {msg}", "#ff5252")

    def _append(self, text: str, color: str):
        self.log_edit.append(f'<span style="color:{color};">{text}</span>')


# ============================================================================
# 轴位置实时显示面板
# ============================================================================

class AxisPositionPanel(QGroupBox):
    """轴位置实时显示面板"""

    def __init__(self, parent=None):
        super().__init__("轴位置", parent)
        self._labels: List[QLabel] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(4)

        headers = ["轴", "命令位置", "编码器位置", "速度", "状态"]
        for col, h in enumerate(headers):
            label = QLabel(h)
            label.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 15px;")
            layout.addWidget(label, 0, col)

        for i in range(4):
            axis_label = QLabel(f"Axis_{i + 1}")
            axis_label.setStyleSheet("color: #d4d4d4; font-weight: bold;")
            layout.addWidget(axis_label, i + 1, 0)

            for _ in range(4):
                label = QLabel("-")
                label.setStyleSheet("color: #c8c8c8; font-family: Consolas;")
                layout.addWidget(label, i + 1, len(self._labels) % 4 + 1)
                self._labels.append(label)

    def update_positions(self, sdk: NMCSDK):
        """刷新所有轴的位置信息"""
        for i, axis in enumerate(AXIS_VALUES):
            idx = i * 4
            try:
                pos = sdk.get_position(axis)
                self._labels[idx].setText(str(pos))
            except Exception:
                self._labels[idx].setText("--")

            try:
                enc = sdk.get_encoder(axis)
                self._labels[idx + 1].setText(str(enc))
            except Exception:
                self._labels[idx + 1].setText("--")

            try:
                vel = sdk.get_velocity(axis)
                self._labels[idx + 2].setText(f"{vel[0]:.1f}")
            except Exception:
                self._labels[idx + 2].setText("--")

            try:
                state = sdk.get_axis_state(axis)
                self._labels[idx + 3].setText(f"0x{state:08X}")
            except Exception:
                self._labels[idx + 3].setText("--")


# ============================================================================
# IO 监测面板
# ============================================================================

class IOMonitorPanel(QWidget):
    """运动卡 IO 口监测面板 — 显示数字输入/输出状态"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_leds: List[LedIndicator] = []
        self._output_leds: List[LedIndicator] = []
        self._input_labels: List[QLabel] = []
        self._output_labels: List[QLabel] = []
        self._special_io_labels: List[QLabel] = []
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        # ── 数字输入 ──
        input_group = QGroupBox("数字输入 (DI)")
        input_group.setStyleSheet(self._group_style())
        input_layout = QGridLayout(input_group)
        input_layout.setSpacing(4)

        for i in range(16):
            led = LedIndicator(color="#4CAF50")
            self._input_leds.append(led)
            label = QLabel(f"DI{i}")
            label.setStyleSheet("color: #d4d4d4; font-size: 14px;")
            self._input_labels.append(label)
            row = i // 8
            col = (i % 8) * 2
            input_layout.addWidget(led, row, col)
            input_layout.addWidget(label, row, col + 1)

        main_layout.addWidget(input_group)

        # ── 数字输出 ──
        output_group = QGroupBox("数字输出 (DO)")
        output_group.setStyleSheet(self._group_style())
        output_layout = QGridLayout(output_group)
        output_layout.setSpacing(4)

        for i in range(16):
            led = LedIndicator(color="#FF9800")
            self._output_leds.append(led)
            label = QLabel(f"DO{i}")
            label.setStyleSheet("color: #d4d4d4; font-size: 14px;")
            self._output_labels.append(label)
            row = i // 8
            col = (i % 8) * 2
            output_layout.addWidget(led, row, col)
            output_layout.addWidget(label, row, col + 1)

        main_layout.addWidget(output_group)

        # ── 特殊 IO 状态 ──
        special_group = QGroupBox("轴状态信号")
        special_group.setStyleSheet(self._group_style())
        special_layout = QGridLayout(special_group)
        special_layout.setSpacing(4)

        signals = ["Z相", "原点", "正限位", "负限位", "伺服报警", "伺服到位"]
        for i, name in enumerate(signals):
            label = QLabel(f"Axis_1 {name}:")
            label.setStyleSheet("color: #d4d4d4;")
            special_layout.addWidget(label, 0, i * 2)
            val_label = QLabel("-")
            val_label.setStyleSheet("color: #c8c8c8; font-family: Consolas;")
            special_layout.addWidget(val_label, 0, i * 2 + 1)
            self._special_io_labels.append(val_label)

        main_layout.addWidget(special_group)

    def _group_style(self):
        return """
            QGroupBox {
                font-weight: bold; font-size: 15px; border: 1px solid #444;
                border-radius: 4px; margin-top: 8px; padding-top: 14px;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
                color: #d4d4d4;
            }
        """

    def update_io(self, input_value: int, output_value: int):
        """更新 IO 显示"""
        for i in range(16):
            on_in = bool(input_value & (1 << i))
            self._input_leds[i].set_on(on_in)
            on_out = bool(output_value & (1 << i))
            self._output_leds[i].set_on(on_out)


# ============================================================================
# 主对话框
# ============================================================================

class NMCControlDialog(QDialog):
    """NMC3401 运动控制卡对话框"""

    # ── 信号 ──
    nmc_connected = pyqtSignal(bool)  # 连接状态变化
    motion_completed = pyqtSignal(int, bool)  # (axis, success) 运动完成信号（预留联动用）

    def __init__(self, parent=None, nmc_sdk: Optional[NMCSDK] = None):
        super().__init__(parent)
        self._external_sdk = nmc_sdk is not None
        self.sdk = nmc_sdk or NMCSDK()

        # 状态
        self._connected = False
        self._homing = False
        self._home_axis = Axis_1
        self._home_phase = 0
        self._home_phase2_speed = 0.0
        self._home_search_negative = True   # 当前搜索方向：True=负方向, False=正方向
        self._home_speed_level = 0          # 减速等级：0=原始速度, 1=1/5, 2=1/10, 3=1/20
        self._home_has_reversed = False     # 是否已经反向过（防止来回无限循环）
        self._home_monitor_axis = Axis_1    # 当前回零监测的轴

        # 定时器
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._on_timer_tick)

        self._home_monitor_timer = QTimer(self)
        self._home_monitor_timer.setInterval(50)
        self._home_monitor_timer.timeout.connect(self._on_home_monitor_tick)

        self._setup_ui()
        self._connect_signals()

        self.setWindowTitle("运动控制卡")
        self.setMinimumSize(900, 680)
        self.resize(960, 720)

        # 初始化时检测 SDK 是否已连接，自动更新 UI 状态
        if self.sdk._connected:
            self._update_connection_state(True)

    # ──────────────────────────────────────────────
    # UI 构建
    # ──────────────────────────────────────────────

    def _setup_ui(self):
        """构建 UI 布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        # ── 工具栏 ──
        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        # ── 日志显示 ──
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(100)
        self.log_edit.setStyleSheet("""
            QTextEdit {
                background-color: #0d0d0d; color: #c8c8c8;
                border: 1px solid #444; border-radius: 3px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 14px; padding: 4px;
            }
        """)
        main_layout.addWidget(self.log_edit)
        self.log = _DialogLogHandler(self.log_edit)

        # ── 标签页 ──
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444; border-radius: 4px;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d; color: #d4d4d4;
                padding: 6px 16px; border: 1px solid #444;
                border-bottom: none; border-top-left-radius: 4px;
                border-top-right-radius: 4px; margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e; color: #4fc3f7;
                border-bottom: 2px solid #4fc3f7;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
        """)

        self.tab_widget.addTab(self._create_init_tab(), "初始化")
        self.tab_widget.addTab(self._create_io_tab(), "IO监测")
        self.tab_widget.addTab(self._create_axis_tab(), "轴参数")
        self.tab_widget.addTab(self._create_home_tab(), "回零")
        self.tab_widget.addTab(self._create_motion_tab(), "点位运动")

        main_layout.addWidget(self.tab_widget, 1)

        # ── 状态栏 ──
        status_bar = QWidget()
        status_bar.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #444;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(4, 2, 4, 2)

        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color: #ff5252; font-size: 15px; font-weight: bold;")
        status_layout.addWidget(self.status_label, 1)

        self.sys_info_label = QLabel("")
        self.sys_info_label.setStyleSheet("color: #999; font-size: 14px;")
        status_layout.addWidget(self.sys_info_label)

        main_layout.addWidget(status_bar)

    # ──────────────────────────────────────────────
    # 工具栏
    # ──────────────────────────────────────────────

    def _create_toolbar(self) -> QToolBar:
        """创建工具栏"""
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #252526; border: 1px solid #444;
                border-radius: 4px; spacing: 6px; padding: 4px;
            }
            QToolButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 6px 14px; border: 1px solid #555;
                border-radius: 4px; font-size: 15px;
            }
            QToolButton:hover {
                background-color: #4a4a4a; border-color: #4A90D9;
            }
            QToolButton:disabled {
                background-color: #2d2d2d; color: #555; border-color: #3a3a3a;
            }
        """)

        self.btn_connect = QToolButton()
        self.btn_connect.setText("🔌 连接")
        self.btn_connect.setToolTip("连接 NMC3401 运动控制卡")
        self.btn_connect.clicked.connect(self._on_connect)
        toolbar.addWidget(self.btn_connect)

        self.btn_disconnect = QToolButton()
        self.btn_disconnect.setText("🔌 断开")
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.setToolTip("断开控制卡连接")
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        toolbar.addWidget(self.btn_disconnect)

        toolbar.addSeparator()

        self.btn_emergency = QToolButton()
        self.btn_emergency.setText("🛑 急停")
        self.btn_emergency.setEnabled(False)
        self.btn_emergency.setStyleSheet("""
            QToolButton {
                background-color: #C62828; color: #fff;
                padding: 6px 14px; border: 2px solid #EF5350;
                border-radius: 4px; font-size: 15px; font-weight: bold;
            }
            QToolButton:hover { background-color: #B71C1C; }
            QToolButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)
        self.btn_emergency.setToolTip("紧急停止所有轴")
        self.btn_emergency.clicked.connect(self._on_emergency_stop)
        toolbar.addWidget(self.btn_emergency)

        return toolbar

    # ──────────────────────────────────────────────
    # 初始化标签页
    # ──────────────────────────────────────────────

    def _create_init_tab(self) -> QWidget:
        """创建初始化标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # ── 连接参数 ──
        conn_group = QGroupBox("连接参数")
        conn_group.setStyleSheet(self._group_box_style())
        conn_layout = QGridLayout(conn_group)
        conn_layout.setSpacing(6)

        conn_layout.addWidget(QLabel("IP 地址:"), 0, 0)
        self.ip_edit = QLineEdit("192.168.1.100")
        self.ip_edit.setStyleSheet(self._line_edit_style())
        conn_layout.addWidget(self.ip_edit, 0, 1)

        conn_layout.addWidget(QLabel("端口:"), 0, 2)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(502)
        self.port_spin.setStyleSheet(self._spin_style())
        conn_layout.addWidget(self.port_spin, 0, 3)

        conn_layout.addWidget(QLabel("超时(ms):"), 1, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(100, 30000)
        self.timeout_spin.setValue(1000)
        self.timeout_spin.setSingleStep(100)
        self.timeout_spin.setStyleSheet(self._spin_style())
        conn_layout.addWidget(self.timeout_spin, 1, 1)

        layout.addWidget(conn_group)

        # ── 系统信息 ──
        sys_group = QGroupBox("系统信息")
        sys_group.setStyleSheet(self._group_box_style())
        sys_layout = QGridLayout(sys_group)
        sys_layout.setSpacing(4)

        info_items = [
            ("固件版本:", "sys_version"),
            ("序列号:", "sys_serial"),
            ("运行时间:", "sys_runtime"),
            ("连接状态:", "sys_conn_state"),
        ]
        self._sys_labels = {}
        for i, (label_text, key) in enumerate(info_items):
            label = QLabel(label_text)
            label.setStyleSheet("color: #d4d4d4;")
            sys_layout.addWidget(label, i, 0)
            val_label = QLabel("-")
            val_label.setStyleSheet("color: #c8c8c8; font-family: Consolas;")
            sys_layout.addWidget(val_label, i, 1)
            self._sys_labels[key] = val_label

        layout.addWidget(sys_group)

        # ── 轴使能控制 ──
        enable_group = QGroupBox("轴使能控制 (仅 Axis_2)")
        enable_group.setStyleSheet(self._group_box_style())
        enable_layout = QHBoxLayout(enable_group)
        enable_layout.setSpacing(8)

        self.btn_enable_all = QPushButton("Axis_2 使能")
        self.btn_enable_all.setStyleSheet(self._action_btn_style("#388E3C"))
        self.btn_enable_all.setEnabled(False)
        self.btn_enable_all.clicked.connect(lambda: self._on_enable_axis2(True))
        enable_layout.addWidget(self.btn_enable_all)

        self.btn_disable_all = QPushButton("Axis_2 关闭")
        self.btn_disable_all.setStyleSheet(self._action_btn_style("#C62828"))
        self.btn_disable_all.setEnabled(False)
        self.btn_disable_all.clicked.connect(lambda: self._on_enable_axis2(False))
        enable_layout.addWidget(self.btn_disable_all)

        enable_layout.addStretch()

        layout.addWidget(enable_group)

        # ── 轴位置显示 ──
        self.axis_pos_panel = AxisPositionPanel()
        self.axis_pos_panel.setStyleSheet(self._group_box_style())
        layout.addWidget(self.axis_pos_panel, 1)

        return widget

    # ──────────────────────────────────────────────
    # IO 监测标签页
    # ──────────────────────────────────────────────

    def _create_io_tab(self) -> QWidget:
        """创建 IO 监测标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.io_panel = IOMonitorPanel()
        layout.addWidget(self.io_panel)
        layout.addStretch()
        return widget

    # ──────────────────────────────────────────────
    # 轴参数标签页
    # ──────────────────────────────────────────────

    def _create_axis_tab(self) -> QWidget:
        """创建轴参数标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # ── 轴选择 ──
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("选择轴:"))
        self.cmb_axis = QComboBox()
        for i, name in enumerate(AXIS_NAMES):
            self.cmb_axis.addItem(name, AXIS_VALUES[i])
        self.cmb_axis.setStyleSheet(self._combo_style())
        self.cmb_axis.currentIndexChanged.connect(self._on_axis_changed)
        select_layout.addWidget(self.cmb_axis)
        select_layout.addStretch()
        layout.addLayout(select_layout)

        # ── 脉冲模式 ──
        pulse_group = QGroupBox("脉冲模式")
        pulse_group.setStyleSheet(self._group_box_style())
        pulse_layout = QHBoxLayout(pulse_group)
        self.cmb_pulse_mode = QComboBox()
        for name, val in PULSE_MODES:
            self.cmb_pulse_mode.addItem(name, val)
        self.cmb_pulse_mode.setStyleSheet(self._combo_style())
        pulse_layout.addWidget(self.cmb_pulse_mode)
        self.btn_set_pulse = QPushButton("设置")
        self.btn_set_pulse.setStyleSheet(self._small_btn_style())
        self.btn_set_pulse.clicked.connect(self._on_set_pulse_mode)
        pulse_layout.addWidget(self.btn_set_pulse)
        pulse_layout.addStretch()
        layout.addWidget(pulse_group)

        # ── 位置设置 ──
        pos_group = QGroupBox("位置设置")
        pos_group.setStyleSheet(self._group_box_style())
        pos_layout = QGridLayout(pos_group)
        pos_layout.setSpacing(6)

        pos_layout.addWidget(QLabel("命令位置:"), 0, 0)
        self.spin_cmd_pos = QSpinBox()
        self.spin_cmd_pos.setRange(-9999999, 9999999)
        self.spin_cmd_pos.setStyleSheet(self._spin_style())
        pos_layout.addWidget(self.spin_cmd_pos, 0, 1)
        self.btn_set_pos = QPushButton("设置")
        self.btn_set_pos.setStyleSheet(self._small_btn_style())
        self.btn_set_pos.clicked.connect(self._on_set_position)
        pos_layout.addWidget(self.btn_set_pos, 0, 2)

        pos_layout.addWidget(QLabel("编码器位置:"), 1, 0)
        self.spin_enc_pos = QSpinBox()
        self.spin_enc_pos.setRange(-9999999, 9999999)
        self.spin_enc_pos.setStyleSheet(self._spin_style())
        pos_layout.addWidget(self.spin_enc_pos, 1, 1)
        self.btn_set_enc = QPushButton("设置")
        self.btn_set_enc.setStyleSheet(self._small_btn_style())
        self.btn_set_enc.clicked.connect(self._on_set_encoder)
        pos_layout.addWidget(self.btn_set_enc, 1, 2)

        pos_layout.addWidget(QLabel("当前命令位置:"), 2, 0)
        self.lbl_cur_pos = QLabel("-")
        self.lbl_cur_pos.setStyleSheet("color: #4fc3f7; font-family: Consolas; font-size: 16px;")
        pos_layout.addWidget(self.lbl_cur_pos, 2, 1, 1, 2)

        pos_layout.addWidget(QLabel("当前编码器:"), 3, 0)
        self.lbl_cur_enc = QLabel("-")
        self.lbl_cur_enc.setStyleSheet("color: #ffa726; font-family: Consolas; font-size: 16px;")
        pos_layout.addWidget(self.lbl_cur_enc, 3, 1, 1, 2)

        layout.addWidget(pos_group)

        # ── 软限位 ──
        limit_group = QGroupBox("软件限位")
        limit_group.setStyleSheet(self._group_box_style())
        limit_layout = QGridLayout(limit_group)
        limit_layout.setSpacing(6)

        limit_layout.addWidget(QLabel("正限位:"), 0, 0)
        self.spin_limit_p = QSpinBox()
        self.spin_limit_p.setRange(-9999999, 9999999)
        self.spin_limit_p.setValue(100000)
        self.spin_limit_p.setStyleSheet(self._spin_style())
        limit_layout.addWidget(self.spin_limit_p, 0, 1)

        limit_layout.addWidget(QLabel("负限位:"), 0, 2)
        self.spin_limit_n = QSpinBox()
        self.spin_limit_n.setRange(-9999999, 9999999)
        self.spin_limit_n.setValue(-100000)
        self.spin_limit_n.setStyleSheet(self._spin_style())
        limit_layout.addWidget(self.spin_limit_n, 0, 3)

        self.btn_set_limit = QPushButton("设置并启用")
        self.btn_set_limit.setStyleSheet(self._small_btn_style())
        self.btn_set_limit.clicked.connect(self._on_set_soft_limit)
        limit_layout.addWidget(self.btn_set_limit, 0, 4)

        limit_layout.addWidget(QLabel("当前正限位:"), 1, 0)
        self.lbl_cur_limit_p = QLabel("-")
        self.lbl_cur_limit_p.setStyleSheet("color: #c8c8c8; font-family: Consolas;")
        limit_layout.addWidget(self.lbl_cur_limit_p, 1, 1)

        limit_layout.addWidget(QLabel("当前负限位:"), 1, 2)
        self.lbl_cur_limit_n = QLabel("-")
        self.lbl_cur_limit_n.setStyleSheet("color: #c8c8c8; font-family: Consolas;")
        limit_layout.addWidget(self.lbl_cur_limit_n, 1, 3)

        layout.addWidget(limit_group)

        # ── 轴状态显示 ──
        state_group = QGroupBox("轴状态")
        state_group.setStyleSheet(self._group_box_style())
        state_layout = QHBoxLayout(state_group)
        self.lbl_axis_state = QLabel("状态: -")
        self.lbl_axis_state.setStyleSheet("color: #c8c8c8; font-family: Consolas; font-size: 16px;")
        state_layout.addWidget(self.lbl_axis_state)
        state_layout.addStretch()
        layout.addWidget(state_group)

        layout.addStretch()
        return widget

    # ──────────────────────────────────────────────
    # 回零标签页
    # ──────────────────────────────────────────────

    def _create_home_tab(self) -> QWidget:
        """创建回零标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # ── 回零参数 ──
        param_group = QGroupBox("回零参数")
        param_group.setStyleSheet(self._group_box_style())
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(6)

        param_layout.addWidget(QLabel("选择轴:"), 0, 0)
        self.cmb_home_axis = QComboBox()
        for i, name in enumerate(AXIS_NAMES):
            self.cmb_home_axis.addItem(name, AXIS_VALUES[i])
        self.cmb_home_axis.setStyleSheet(self._combo_style())
        param_layout.addWidget(self.cmb_home_axis, 0, 1)

        param_layout.addWidget(QLabel("回零模式:"), 0, 2)
        self.cmb_home_mode = QComboBox()
        for name, val in HOME_MODES:
            self.cmb_home_mode.addItem(name, val)
        self.cmb_home_mode.setStyleSheet(self._combo_style())
        param_layout.addWidget(self.cmb_home_mode, 0, 3)

        param_layout.addWidget(QLabel("高速(mm/min):"), 1, 0)
        self.spin_home_high = QDoubleSpinBox()
        self.spin_home_high.setRange(1, 100000)
        self.spin_home_high.setValue(10000)
        self.spin_home_high.setStyleSheet(self._spin_style())
        param_layout.addWidget(self.spin_home_high, 1, 1)

        param_layout.addWidget(QLabel("低速(mm/min):"), 1, 2)
        self.spin_home_low = QDoubleSpinBox()
        self.spin_home_low.setRange(1, 100000)
        self.spin_home_low.setValue(1000)
        self.spin_home_low.setStyleSheet(self._spin_style())
        param_layout.addWidget(self.spin_home_low, 1, 3)

        param_layout.addWidget(QLabel("加速度(mm/s²):"), 2, 0)
        self.spin_home_acc = QDoubleSpinBox()
        self.spin_home_acc.setRange(1, 100000)
        self.spin_home_acc.setValue(1000)
        self.spin_home_acc.setStyleSheet(self._spin_style())
        param_layout.addWidget(self.spin_home_acc, 2, 1)

        param_layout.addWidget(QLabel("偏移(pulse):"), 2, 2)
        self.spin_home_offset = QSpinBox()
        self.spin_home_offset.setRange(-9999999, 9999999)
        self.spin_home_offset.setValue(0)
        self.spin_home_offset.setStyleSheet(self._spin_style())
        param_layout.addWidget(self.spin_home_offset, 2, 3)

        layout.addWidget(param_group)

        # ── 控制按钮 ──
        btn_layout = QHBoxLayout()
        self.btn_home_start = QPushButton("▶ 开始回零")
        self.btn_home_start.setStyleSheet(self._action_btn_style("#1976D2"))
        self.btn_home_start.setMinimumHeight(40)
        self.btn_home_start.clicked.connect(self._on_home_start)
        btn_layout.addWidget(self.btn_home_start)

        self.btn_home_stop = QPushButton("⏹ 停止回零")
        self.btn_home_stop.setStyleSheet(self._action_btn_style("#C62828"))
        self.btn_home_stop.setMinimumHeight(40)
        self.btn_home_stop.setEnabled(False)
        self.btn_home_stop.clicked.connect(self._on_home_stop)
        btn_layout.addWidget(self.btn_home_stop)

        layout.addLayout(btn_layout)

        # ── 回零状态 ──
        status_group = QGroupBox("回零状态")
        status_group.setStyleSheet(self._group_box_style())
        status_layout = QVBoxLayout(status_group)

        self.lbl_home_state = QLabel("就绪")
        self.lbl_home_state.setStyleSheet("color: #d4d4d4; font-size: 16px; font-weight: bold;")
        status_layout.addWidget(self.lbl_home_state)

        self.home_progress = QProgressBar()
        self.home_progress.setRange(0, 100)
        self.home_progress.setValue(0)
        self.home_progress.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d; border: 1px solid #444;
                border-radius: 3px; text-align: center; color: #d4d4d4;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7; border-radius: 2px;
            }
        """)
        status_layout.addWidget(self.home_progress)

        layout.addWidget(status_group)

        layout.addStretch()
        return widget

    # =====================================================================
    #  运动控制 Tab
    # =====================================================================
    def _create_motion_tab(self) -> QWidget:
        """创建运动控制标签页 — JOG + 单轴运动 + 曲线参数"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # ── JOG 控制 ──
        jog_group = QGroupBox("JOG 点动")
        jog_group.setStyleSheet(self._group_box_style())
        jog_layout = QVBoxLayout(jog_group)

        # 轴选择
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("轴选择:"))
        self.cmb_motion_axis = QComboBox()
        for i, name in enumerate(AXIS_NAMES):
            self.cmb_motion_axis.addItem(name, AXIS_VALUES[i])
        self.cmb_motion_axis.setStyleSheet(self._combo_style())
        axis_row.addWidget(self.cmb_motion_axis)
        axis_row.addStretch()
        jog_layout.addLayout(axis_row)

        # 速度 / 加速度
        param_grid = QGridLayout()
        param_grid.addWidget(QLabel("速度 (pulse/s):"), 0, 0)
        self.spin_jog_speed = QSpinBox()
        self.spin_jog_speed.setRange(100, 500000)
        self.spin_jog_speed.setValue(50000)
        self.spin_jog_speed.setSuffix(" pps")
        self.spin_jog_speed.setStyleSheet(self._spin_style())
        param_grid.addWidget(self.spin_jog_speed, 0, 1)

        param_grid.addWidget(QLabel("加速度 (pulse/s²):"), 0, 2)
        self.spin_jog_acc = QSpinBox()
        self.spin_jog_acc.setRange(100, 500000)
        self.spin_jog_acc.setValue(100000)
        self.spin_jog_acc.setSuffix(" pps²")
        self.spin_jog_acc.setStyleSheet(self._spin_style())
        param_grid.addWidget(self.spin_jog_acc, 0, 3)

        jog_layout.addLayout(param_grid)

        # JOG 按钮行
        btn_row = QHBoxLayout()
        self.btn_jog_fwd = QPushButton("▶ JOG+")
        self.btn_jog_fwd.setStyleSheet(self._action_btn_style("#4caf50"))
        self.btn_jog_fwd.setMinimumHeight(40)
        self.btn_jog_fwd.setToolTip("按住鼠标左键持续正转，松开停止")
        self.btn_jog_fwd.pressed.connect(self._on_jog_forward)
        self.btn_jog_fwd.released.connect(self._on_jog_stop)
        btn_row.addWidget(self.btn_jog_fwd)

        self.btn_jog_rev = QPushButton("◀ JOG-")
        self.btn_jog_rev.setStyleSheet(self._action_btn_style("#ff9800"))
        self.btn_jog_rev.setMinimumHeight(40)
        self.btn_jog_rev.setToolTip("按住鼠标左键持续反转，松开停止")
        self.btn_jog_rev.pressed.connect(self._on_jog_backward)
        self.btn_jog_rev.released.connect(self._on_jog_stop)
        btn_row.addWidget(self.btn_jog_rev)

        self.btn_jog_stop = QPushButton("■ 停止")
        self.btn_jog_stop.setStyleSheet(self._action_btn_style("#f44336"))
        self.btn_jog_stop.setMinimumHeight(40)
        self.btn_jog_stop.clicked.connect(self._on_jog_stop)
        btn_row.addWidget(self.btn_jog_stop)

        jog_layout.addLayout(btn_row)
        layout.addWidget(jog_group)

        # ── 单轴运动 ──
        motion_group = QGroupBox("单轴运动")
        motion_group.setStyleSheet(self._group_box_style())
        motion_layout = QVBoxLayout(motion_group)

        profile_grid = QGridLayout()
        profile_grid.addWidget(QLabel("起始速度:"), 0, 0)
        self.spin_single_vini = QSpinBox()
        self.spin_single_vini.setRange(0, 500000)
        self.spin_single_vini.setValue(1000)
        self.spin_single_vini.setSuffix(" pps")
        self.spin_single_vini.setStyleSheet(self._spin_style())
        profile_grid.addWidget(self.spin_single_vini, 0, 1)

        profile_grid.addWidget(QLabel("最大速度:"), 0, 2)
        self.spin_single_vmax = QSpinBox()
        self.spin_single_vmax.setRange(100, 500000)
        self.spin_single_vmax.setValue(50000)
        self.spin_single_vmax.setSuffix(" pps")
        self.spin_single_vmax.setStyleSheet(self._spin_style())
        profile_grid.addWidget(self.spin_single_vmax, 0, 3)

        profile_grid.addWidget(QLabel("加速度:"), 1, 0)
        self.spin_single_acc = QSpinBox()
        self.spin_single_acc.setRange(100, 500000)
        self.spin_single_acc.setValue(100000)
        self.spin_single_acc.setSuffix(" pps²")
        self.spin_single_acc.setStyleSheet(self._spin_style())
        profile_grid.addWidget(self.spin_single_acc, 1, 1)

        profile_grid.addWidget(QLabel("减速度:"), 1, 2)
        self.spin_single_dec = QSpinBox()
        self.spin_single_dec.setRange(100, 500000)
        self.spin_single_dec.setValue(100000)
        self.spin_single_dec.setSuffix(" pps²")
        self.spin_single_dec.setStyleSheet(self._spin_style())
        profile_grid.addWidget(self.spin_single_dec, 1, 3)

        profile_grid.addWidget(QLabel("运动距离:"), 2, 0)
        self.spin_single_dist = QSpinBox()
        self.spin_single_dist.setRange(-10000000, 10000000)
        self.spin_single_dist.setValue(50000)
        self.spin_single_dist.setSuffix(" pulse")
        self.spin_single_dist.setStyleSheet(self._spin_style())
        profile_grid.addWidget(self.spin_single_dist, 2, 1)

        profile_grid.addWidget(QLabel("位置模式:"), 2, 2)
        self.cmb_single_pos_mode = QComboBox()
        self.cmb_single_pos_mode.addItem("绝对位置", Position_Absolute)
        self.cmb_single_pos_mode.addItem("相对位置", Position_Opposite)
        self.cmb_single_pos_mode.setStyleSheet(self._combo_style())
        profile_grid.addWidget(self.cmb_single_pos_mode, 2, 3)

        profile_grid.addWidget(QLabel("曲线:"), 3, 0)
        self.cmb_single_profile = QComboBox()
        self.cmb_single_profile.addItem("T型曲线", Profile_T)
        self.cmb_single_profile.addItem("S型曲线", Profile_S)
        self.cmb_single_profile.setStyleSheet(self._combo_style())
        profile_grid.addWidget(self.cmb_single_profile, 3, 1)

        motion_layout.addLayout(profile_grid)

        # 运动按钮行
        motion_btn_row = QHBoxLayout()
        self.btn_set_profile = QPushButton("设置曲线参数")
        self.btn_set_profile.setStyleSheet(self._action_btn_style("#2196f3"))
        self.btn_set_profile.clicked.connect(self._on_set_profile)
        motion_btn_row.addWidget(self.btn_set_profile)

        self.btn_start_motion = QPushButton("▶ 启动运动")
        self.btn_start_motion.setStyleSheet(self._action_btn_style("#4caf50"))
        self.btn_start_motion.clicked.connect(self._on_start_motion)
        motion_btn_row.addWidget(self.btn_start_motion)

        self.btn_stop_motion = QPushButton("■ 停止运动")
        self.btn_stop_motion.setStyleSheet(self._action_btn_style("#f44336"))
        self.btn_stop_motion.clicked.connect(lambda: self._on_stop_motion(
            self.cmb_motion_axis.currentData()))
        motion_btn_row.addWidget(self.btn_stop_motion)

        motion_layout.addLayout(motion_btn_row)
        layout.addWidget(motion_group)

        layout.addStretch()
        return widget

    # =====================================================================
    #  信号连接
    # =====================================================================
    def _connect_signals(self):
        """连接控件信号"""
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        self.btn_emergency.clicked.connect(self._on_emergency_stop)
        self.cmb_axis.currentIndexChanged.connect(self._on_axis_changed)

    # =====================================================================
    #  连接 / 断开 / 急停
    # =====================================================================
    def _on_connect(self):
        """连接控制卡"""
        ip = self.ip_edit.text().strip()
        port = self.port_spin.value()
        timeout = self.timeout_spin.value()
        if not ip:
            QMessageBox.warning(self, "警告", "请输入控制卡 IP 地址")
            return

        try:
            if not self.sdk.is_loaded():
                self.sdk.load_dll()
                self.log.info("DLL 加载成功")
            self.sdk.connect(ip, port, timeout)
            connected_cards = self.sdk.get_open_net()
            if connected_cards[0] > 0:
                self._update_connection_state(True)
                self.log.info(f"连接成功: {ip}:{port}")
                self._auto_setup_soft_limits()
                self._refresh_sys_info()
            else:
                QMessageBox.warning(self, "连接失败", "未检测到控制卡")
        except Exception as e:
            QMessageBox.critical(self, "连接错误", str(e))
            self.log.error(f"连接失败: {e}")

    def _auto_setup_soft_limits(self):
        """自动设置 Axis_2 的软限位（仅控制 Axis_2，其他轴不动）"""
        if not self.sdk._connected:
            return
        # 仅设置 Axis_2 (索引1) 的软限位
        axis = Axis_2
        pos_limit = 2000
        neg_limit = -60000
        try:
            self.sdk.set_soft_limit(axis, pos_limit, neg_limit)
            self.sdk.set_soft_limit_enable(axis, 1)
            self.log.info(f"Axis_2 软限位已自动设置: [{neg_limit}, {pos_limit}]")
        except Exception as e:
            self.log.warning(f"Axis_2 自动设置软限位失败: {e}")

    def _on_disconnect(self):
        """断开控制卡连接"""
        try:
            self._refresh_timer.stop()
            self.sdk.close_net()
            self.log.info("控制卡已断开")
        except Exception as e:
            self.log.warning(f"断开时异常: {e}")
        self._update_connection_state(False)

    def _on_emergency_stop(self):
        """紧急停止 Axis_2（其他轴不受影响）"""
        if not self.sdk._connected:
            return
        try:
            self.log.warning("⚠ 紧急停止 Axis_2!")
            self.sdk.emergency_stop_axis2()
        except Exception as e:
            self.log.error(f"Axis_2 紧急停止失败: {e}")

    # =====================================================================
    #  定时器回调
    # =====================================================================
    def _on_timer_tick(self):
        """定时器事件 - 刷新状态"""
        if not self.sdk._connected:
            return
        try:
            self._refresh_axis_params()
            self.axis_pos_panel.update_positions(self.sdk)
            self._refresh_io()
        except Exception:
            pass

    def _update_connection_state(self, connected: bool):
        """更新连接状态显示（不再直接修改 sdk._connected，由 SDK 自身管理）"""
        self._connected = connected
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.btn_emergency.setEnabled(connected)
        self.btn_enable_all.setEnabled(connected)
        self.btn_disable_all.setEnabled(connected)
        self.ip_edit.setEnabled(not connected)
        self.port_spin.setEnabled(not connected)
        self.timeout_spin.setEnabled(not connected)

        # 启用/禁用各标签页
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, connected)
        # 但始终启用初始化页
        self.tab_widget.setTabEnabled(0, True)

        if connected:
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet("color: #4caf50; font-size: 15px; font-weight: bold;")
            self._refresh_timer.start(100)
        else:
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet("color: #ff5252; font-size: 15px; font-weight: bold;")
            self._refresh_timer.stop()

    # =====================================================================
    #  IO 刷新
    # =====================================================================
    def _refresh_io(self):
        """刷新 IO 显示"""
        if not self.sdk._connected:
            return
        try:
            input_val = self.sdk.get_input()
            output_val = self.sdk.get_output()
            self.io_panel.update_io(input_val, output_val)
        except Exception:
            pass

    # =====================================================================
    #  轴参数
    # =====================================================================
    def _on_axis_changed(self):
        """轴选择切换时，更新软限位默认值"""
        if not self.sdk._connected:
            return
        axis = self.cmb_axis.currentData()
        if axis is None:
            return
        try:
            if axis in (AXIS_VALUES[0], AXIS_VALUES[1], AXIS_VALUES[2]):
                self.spin_limit_p.setValue(2000)
                self.spin_limit_n.setValue(-60000)
            else:
                self.spin_limit_p.setValue(1000000)
                self.spin_limit_n.setValue(-1000000)
        except Exception:
            pass
        self._refresh_axis_params()

    def _refresh_axis_params(self):
        """定时刷新轴参数"""
        if not self.sdk._connected:
            return
        axis = self.cmb_axis.currentData()
        if axis is None:
            return
        try:
            pos = self.sdk.get_position(axis)
            enc = self.sdk.get_encoder(axis)
            vel, _ = self.sdk.get_velocity(axis)
            state = self.sdk.get_axis_state(axis)

            self.lbl_cur_pos.setText(str(pos))
            self.lbl_cur_enc.setText(str(enc))

            state_name = self._axis_state_text(state)
            self.lbl_axis_state.setText(f"状态: {state_name} (0x{state:08X})")
        except Exception:
            pass

    def _axis_state_text(self, state: int) -> str:
        """将轴状态值转换为可读文本"""
        state_map = {
            0: "空闲", 1: "加速", 2: "匀速", 3: "减速",
            4: "停止中", 5: "急停", 6: "正限位", 7: "负限位",
            8: "报警", 9: "回零中", 10: "回零完成",
            11: "JOG中", 12: "运动完成",
        }
        return state_map.get(state, f"未知({state})")

    # =====================================================================
    #  脉冲模式 / 位置 / 编码器 / 软限位
    # =====================================================================
    def _on_set_pulse_mode(self):
        """设置脉冲模式"""
        if not self.sdk._connected:
            return
        axis = self.cmb_axis.currentData()
        mode = self.cmb_pulse_mode.currentData()
        if axis is None or mode is None:
            return
        try:
            self.sdk.set_pulse_mode(axis, mode)
            self.log.info(f"Axis_{axis} 脉冲模式已设置")
        except Exception as e:
            self.log.error(f"设置脉冲模式失败: {e}")

    def _on_set_position(self):
        """设置命令位置"""
        if not self.sdk._connected:
            return
        axis = self.cmb_axis.currentData()
        pos = self.spin_cmd_pos.value()
        if axis is None:
            return
        try:
            self.sdk.set_position(axis, pos)
            self.log.info(f"Axis_{axis} 命令位置已设置为 {pos}")
        except Exception as e:
            self.log.error(f"设置命令位置失败: {e}")

    def _on_set_encoder(self):
        """设置编码器位置"""
        if not self.sdk._connected:
            return
        axis = self.cmb_axis.currentData()
        enc = self.spin_enc_pos.value()
        if axis is None:
            return
        try:
            self.sdk.set_encoder(axis, enc)
            self.log.info(f"Axis_{axis} 编码器位置已设置为 {enc}")
        except Exception as e:
            self.log.error(f"设置编码器位置失败: {e}")

    def _on_set_soft_limit(self):
        """设置软件限位并自动使能"""
        if not self.sdk._connected:
            return
        axis = self.cmb_axis.currentData()
        if axis is None:
            return
        pos_limit = self.spin_limit_p.value()
        neg_limit = self.spin_limit_n.value()
        try:
            self.sdk.set_soft_limit(axis, pos_limit, neg_limit)
            self.sdk.set_soft_limit_enable(axis, 1)
            self.log.info(f"Axis_{axis} 软限位已设置: [{neg_limit}, {pos_limit}] 已使能")
        except Exception as e:
            self.log.error(f"设置软限位失败: {e}")

    # =====================================================================
    #  全部使能
    # =====================================================================
    def _on_enable_axis2(self, enable: bool):
        """仅使能/关闭 Axis_2，其他轴保持不变"""
        if not self.sdk._connected:
            return
        try:
            self.sdk.enable_axis2_servo(enable)
            self.log.info(f"Axis_2 伺服{'已使能' if enable else '已关闭'} (其他轴保持不变)")
        except Exception as e:
            self.log.error(f"Axis_2 伺服{'使能' if enable else '关闭'}失败: {e}")

    # =====================================================================
    #  回零
    # =====================================================================
    def _on_home_start(self):
        """开始回零 - 双向搜索原点信号（三阶段速度控制）

        回零流程：
        阶段1 - 搜索：先向负方向 JOG 搜索原点
        阶段2 - 减速逼近：检测到原点后逐步降速，信号消失后进入阶段3
        阶段3 - 精定位：反向极低速寻找原点边缘，检测到后停止设为零点

        受软限位保护，不会超出限位范围。
        回零前自动设置并使能软限位。
        """
        if not self.sdk._connected:
            QMessageBox.warning(self, "警告", "请先连接控制卡")
            return

        axis = self.cmb_home_axis.currentData()
        if axis is None:
            return

        # 如果正在回零，先停止
        if self._homing:
            self._on_home_stop()

        low_speed = int(self.spin_home_low.value())
        acc = int(self.spin_home_acc.value())

        try:
            # 读取当前位置
            current_pos = self.sdk.get_position(axis)
            self.log.info(f"Axis_{axis} 当前位置: {current_pos}")

            # 读取原点信号状态 (0=OFF=检测到, 1=ON=未检测到)
            home_signal = self.sdk.get_home(axis)
            home_detected = (home_signal == 0)
            self.log.info(f"Axis_{axis} 原点信号: {'检测到(OFF)' if home_detected else '未检测到(ON)'}")

            # === 设置软限位（确保限位保护生效）===
            # Axis_1=0(第1轴), Axis_2=1(第2轴/X), Axis_3=2(第3轴/Y)
            if axis in (Axis_1, Axis_2, Axis_3):
                pos_limit = 2000
                neg_limit = -60000
            else:  # 第4轴/Z轴及以上
                pos_limit = 1000000
                neg_limit = -1000000

            self.log.info(f"Axis_{axis} 设置软限位: +{pos_limit}, {neg_limit}")
            ret = self.sdk.set_soft_limit(axis, pos_limit, neg_limit)
            self.log.info(f"Axis_{axis} 设置软限位位置 (返回值: {ret})")
            ret = self.sdk.set_soft_limit_enable(axis, 1)
            self.log.info(f"Axis_{axis} 使能软限位 (返回值: {ret})")

            # 如果已经检测到原点信号，先向负方向离开原点，再回零
            if home_detected:
                self.log.info(f"Axis_{axis} 已在原点位置，向负方向离开再回零")
                # 用 JOG 负方向离开原点区域（负方向没有传感器覆盖）
                self.sdk.jog(axis, float(-low_speed), float(acc))
                # 等待原点信号消失（最多等待 3 秒）
                wait_start = time.time()
                while time.time() - wait_start < 3.0:
                    sig = self.sdk.get_home(axis)
                    if sig != 0:  # 信号消失（ON=未检测到）
                        break
                    time.sleep(0.05)
                # 信号消失后停止
                self.sdk.axis_stop(axis, Stop_Abrupt)
                time.sleep(0.2)
                self.sdk.clear_axis_state(axis)
                leave_pos = self.sdk.get_position(axis)
                self.log.info(f"Axis_{axis} 原点信号消失位置: {leave_pos}")
                # 再向负方向多走一点，确保完全离开
                self.sdk.jog(axis, float(-low_speed), float(acc))
                time.sleep(0.2)
                self.sdk.axis_stop(axis, Stop_Abrupt)
                time.sleep(0.2)
                self.sdk.clear_axis_state(axis)
                safe_pos = self.sdk.get_position(axis)
                self.log.info(f"Axis_{axis} 已退回到安全位置: {safe_pos}")

            # 根据是否离开过原点决定搜索方向
            if home_detected:
                # 已离开原点，现在在负方向，向正方向搜索
                self.log.info(f"Axis_{axis} 开始回零: 向正方向搜索原点, speed={low_speed}")
                ret = self.sdk.jog(axis, float(low_speed), float(acc))
                self.log.info(f"Axis_{axis} 启动JOG正方向 (返回值: {ret})")
                self._home_search_negative = False
            else:
                # 正常情况，向负方向搜索
                self.log.info(f"Axis_{axis} 开始回零: 向负方向搜索原点, speed={low_speed}")
                ret = self.sdk.jog(axis, float(-low_speed), float(acc))
                self.log.info(f"Axis_{axis} 启动JOG负方向 (返回值: {ret})")
                self._home_search_negative = True

            # 回零状态机阶段: 1=搜索阶段, 2=减速逼近阶段, 3=精定位阶段
            self._home_phase = 1
            self._home_speed_level = 0
            self._home_has_reversed = False
            self._homing = True
            self._home_axis = axis
            self._home_monitor_axis = axis

            # 更新UI状态
            self.lbl_home_state.setText("回零状态: 正在搜索原点 (负方向)...")
            self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.btn_home_start.setEnabled(False)
            self.btn_home_stop.setEnabled(True)
            self.home_progress.setValue(0)

            # 启动回零监测定时器 (50ms)
            self._home_monitor_timer.start(50)
            self.log.info(f"Axis_{axis} 回零开始")

        except Exception as e:
            self.log.error(f"回零启动失败: {e}")
            QMessageBox.critical(self, "回零错误", f"启动回零失败: {e}")

    def _on_home_stop(self):
        """停止回零运动"""
        if not self.sdk._connected:
            return
        try:
            axis = self.cmb_home_axis.currentData()
            # 停止轴运动
            self.sdk.axis_stop(axis, Stop_Smooth)
            self._home_monitor_timer.stop()
            self.btn_home_start.setEnabled(True)
            self.btn_home_stop.setEnabled(False)
            # 重置回零阶段
            self._home_phase = 1
            self._home_speed_level = 0
            self._homing = False
            pos = self.sdk.get_position(axis)
            self.lbl_home_state.setText("回零状态: 已手动停止")
            self.lbl_home_state.setStyleSheet("color: #f44336;")
            self.home_progress.setValue(0)
            self.log.info(f"Axis_{axis} 回零已手动停止，位置: {pos}")
        except Exception as e:
            self.log.error(f"停止回零失败: {e}")

    def _on_home_monitor_tick(self):
        """回零监测定时器 - 三阶段速度控制（不停止轴，平滑降速）

        核心思路：全程只使用 jog() 改变速度/方向，不使用 axis_stop 或 set_axis_profile。
        利用 jog() 的加速度参数实现平滑的速度切换，避免轴进入停止状态。

        回零流程（三阶段状态机）：
        阶段1 - 搜索阶段：
          1a. 检测到原点信号（OFF=0）→ 进入阶段2（不改变方向，只降速）
          1b. 轴仍在运动中 → 继续等待
          1c. 轴停止（碰到限位）→ 如果还没反向过，切换方向继续搜索
          1d. 两个方向都搜索完还没找到原点 → 回零失败

        阶段2 - 减速逼近阶段（保持原方向，逐步降速）：
          2a. 原点信号消失（ON=1）→ 进入阶段3（反向极低速精定位）
          2b. 原点信号仍在 → 逐步降速（1/5 → 1/10 → 1/20）
          2c. 轴停止（碰到限位）→ 回零失败

        阶段3 - 精定位阶段（反向极低速寻找原点边缘）：
          3a. 检测到原点信号 → jog(0, high_acc) 停止 → 设为零点 ✓
          3b. 轴停止（碰到限位）→ 回零失败
        """
        if not self.sdk._connected:
            self._home_monitor_timer.stop()
            self.btn_home_start.setEnabled(True)
            self.btn_home_stop.setEnabled(False)
            return
        try:
            axis = self._home_monitor_axis
            low_speed = int(self.spin_home_low.value())
            acc = int(self.spin_home_acc.value())

            # 读取当前位置
            pos = self.sdk.get_position(axis)

            # 读取原点信号状态 (0=OFF=检测到, 1=ON=未检测到)
            home_signal = self.sdk.get_home(axis)
            home_detected = (home_signal == 0)

            # 读取轴状态
            axis_state = self.sdk.get_axis_state(axis)
            is_moving = (axis_state == 1)

            # 高加速度值，确保速度切换迅速
            HIGH_ACC = 100000

            # ============================================================
            # 阶段3：精定位阶段（反向极低速寻找原点边缘）
            # ============================================================
            if self._home_phase == 3:
                # 3a. 检测到原点信号 → 停止 → 设为零点 ✓
                if home_detected:
                    self.log.info(f"Axis_{axis} 阶段3: 检测到原点信号! 停止轴并设为零点")
                    # 用 jog(0, high_acc) 让轴自然减速停止（不调用 axis_stop）
                    ret = self.sdk.jog(axis, 0.0, float(HIGH_ACC))
                    self.log.info(f"Axis_{axis} 阶段3: jog停止 (返回值: {ret})")
                    time.sleep(0.3)

                    self._home_monitor_timer.stop()
                    self.btn_home_start.setEnabled(True)
                    self.btn_home_stop.setEnabled(False)

                    stop_pos = self.sdk.get_position(axis)

                    try:
                        self.sdk.set_position(axis, 0)
                        self.sdk.set_encoder(axis, 0)
                        self.log.info(f"Axis_{axis} 已将原点边缘位置 {stop_pos} 设为0点")
                    except Exception as e:
                        self.log.error(f"Axis_{axis} 设为零点失败: {e}")

                    self.lbl_home_state.setText(f"Axis_{axis} 回零完成 ✓")
                    self.lbl_home_state.setStyleSheet("color: #4caf50; font-weight: bold;")
                    self.home_progress.setValue(100)
                    self.log.info(f"Axis_{axis} 回零成功! 原点边缘位置: {stop_pos} → 已设为0点")
                    self._homing = False
                    self._home_phase = 0
                    self._home_speed_level = 0
                    return

                # 3b. 轴停止（碰到限位）→ 回零失败
                if not is_moving:
                    self._home_monitor_timer.stop()
                    self.btn_home_start.setEnabled(True)
                    self.btn_home_stop.setEnabled(False)
                    self.lbl_home_state.setText(f"Axis_{axis} 回零失败 ✗")
                    self.lbl_home_state.setStyleSheet("color: #f44336; font-weight: bold;")
                    self.home_progress.setValue(0)
                    self.log.error(f"Axis_{axis} 回零失败: 精定位阶段轴停止 (state={axis_state})")
                    self._homing = False
                    self._home_phase = 0
                    self._home_speed_level = 0
                    return

                # 继续等待
                self.lbl_home_state.setText("回零状态: 精定位中，等待原点信号...")
                self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")
                return

            # ============================================================
            # 阶段2：减速逼近阶段（保持原方向，逐步降速）
            # ============================================================
            if self._home_phase == 2:
                # 速度递减等级
                speed_levels = [
                    low_speed,           # level 0: 原始速度（阶段1使用）
                    max(low_speed // 5, 1),      # level 1: 1/5
                    max(low_speed // 10, 1),     # level 2: 1/10
                    max(low_speed // 20, 1),     # level 3: 1/20
                ]

                # 2c. 先检查轴是否已停止（碰到限位）→ 回零失败
                if not is_moving:
                    self._home_monitor_timer.stop()
                    self.btn_home_start.setEnabled(True)
                    self.btn_home_stop.setEnabled(False)
                    self.lbl_home_state.setText(f"Axis_{axis} 回零失败 ✗")
                    self.lbl_home_state.setStyleSheet("color: #f44336; font-weight: bold;")
                    self.home_progress.setValue(0)
                    self.log.error(f"Axis_{axis} 回零失败: 减速逼近阶段轴停止 (state={axis_state})")
                    self._homing = False
                    self._home_phase = 0
                    self._home_speed_level = 0
                    return

                # 2a. 原点信号已消失 → 进入阶段3（反向极低速精定位）
                if not home_detected:
                    self.log.info(f"Axis_{axis} 阶段2: 原点信号消失! 进入精定位阶段")
                    self._home_phase = 3
                    self._home_speed_level = 0
                    # 反向极低速运动，寻找原点信号边缘
                    creep_direction = -1 if self._home_search_negative else 1
                    creep_speed = max(low_speed // 50, 10)  # 极低速，至少10
                    ret = self.sdk.jog(axis, float(creep_direction * creep_speed), float(HIGH_ACC))
                    self.log.info(f"Axis_{axis} 阶段3: 反向精定位 speed={creep_speed} (返回值: {ret})")

                    self.lbl_home_state.setText("回零状态: 精定位中...")
                    self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")
                    return

                # 2b. 原点信号仍在 → 逐步降速
                if self._home_speed_level < len(speed_levels) - 1:
                    self._home_speed_level += 1
                    new_speed = speed_levels[self._home_speed_level]
                    # 反方向，只降速
                    direction = 1 if self._home_search_negative else -1
                    self.log.info(f"Axis_{axis} 阶段2: 降速到 level={self._home_speed_level}, speed={new_speed}")
                    ret = self.sdk.jog(axis, float(direction * new_speed), float(HIGH_ACC))
                    self.log.info(f"Axis_{axis} 阶段2: jog降速 (返回值: {ret})")

                    self.lbl_home_state.setText(f"回零状态: 减速逼近 ({self._home_speed_level}/3)...")
                    self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")
                    return
                else:
                    # 已经是最低档位了，但信号还在 → 继续等待信号消失
                    self.lbl_home_state.setText("回零状态: 最低速逼近中，等待信号消失...")
                    self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")
                    return

            # ============================================================
            # 阶段1：搜索阶段（高速搜索原点信号）
            # ============================================================

            # === 条件1a：检测到原点信号（OFF=0）→ 进入阶段2（不改变方向，只降速）===
            if home_detected:
                self.log.info(f"Axis_{axis} 阶段1: 检测到原点信号! 进入减速逼近阶段")
                self._home_phase = 2
                # 阶段1→2转换：不改变方向，只降速到 1/5
                self._home_speed_level = 1
                new_speed = max(low_speed // 5, 1)
                # 反方向，只降速
                direction = 1 if self._home_search_negative else -1
                ret = self.sdk.jog(axis, float(direction * new_speed), float(HIGH_ACC))
                self.log.info(f"Axis_{axis} 阶段2: 降速到 {new_speed}, acc={HIGH_ACC} (返回值: {ret})")

                self.lbl_home_state.setText("回零状态: 减速逼近 (1/3)...")
                self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")
                return

            # === 条件1b：轴仍在运动中 ===
            if is_moving:
                direction = "负方向" if self._home_search_negative else "正方向"
                self.lbl_home_state.setText(f"回零状态: 搜索原点 ({direction})...")
                self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")
                return

            # === 条件1c：轴已停止（碰到限位）===
            self.log.warn(f"Axis_{axis} 轴停止: state={axis_state}，位置: {pos}")

            # 清除轴停止状态
            self.sdk.clear_axis_state(axis)
            time.sleep(0.05)

            # 如果已经反向过一次，说明两个方向都搜索完了 → 回零失败
            if self._home_has_reversed:
                self._home_monitor_timer.stop()
                self.btn_home_start.setEnabled(True)
                self.btn_home_stop.setEnabled(False)
                self.lbl_home_state.setText(f"Axis_{axis} 回零失败 ✗")
                self.lbl_home_state.setStyleSheet("color: #f44336; font-weight: bold;")
                self.home_progress.setValue(0)
                self.log.error(f"Axis_{axis} 回零失败: 两个方向均未找到原点信号")
                self._homing = False
                self._home_phase = 0
                return

            # 还没反向过 → 切换方向继续搜索
            self._home_has_reversed = True
            self._home_search_negative = not self._home_search_negative

            # clear_axis_state 可能会清除软限位使能，需要重新使能
            self.sdk.set_soft_limit_enable(axis, 1)
            time.sleep(0.02)

            if self._home_search_negative:
                self.log.info(f"Axis_{axis} 碰到正限位，切换向负方向搜索")
                ret = self.sdk.jog(axis, float(-low_speed), float(acc))
                self.lbl_home_state.setText("回零状态: 搜索原点 (负方向)...")
            else:
                self.log.info(f"Axis_{axis} 碰到负限位，切换向正方向搜索")
                ret = self.sdk.jog(axis, float(low_speed), float(acc))
                self.lbl_home_state.setText("回零状态: 搜索原点 (正方向)...")

            self.log.info(f"Axis_{axis} 切换方向 (返回值: {ret})")
            self.lbl_home_state.setStyleSheet("color: #ff9800; font-weight: bold;")

        except Exception as e:
            self.log.error(f"_on_home_monitor_tick 异常: {e}")

    # =====================================================================
    #  JOG 控制
    # =====================================================================
    def _on_jog_forward(self):
        """JOG 正转"""
        if not self.sdk._connected:
            return
        axis = self.cmb_motion_axis.currentData()
        if axis is None:
            return
        speed = self.spin_jog_speed.value()
        acc = self.spin_jog_acc.value()
        try:
            self.sdk.jog(axis, float(speed), float(acc))
            self.log.info(f"Axis_{axis} JOG+ 速度={speed}")
        except Exception as e:
            self.log.error(f"JOG+ 失败: {e}")

    def _on_jog_backward(self):
        """JOG 反转"""
        if not self.sdk._connected:
            return
        axis = self.cmb_motion_axis.currentData()
        if axis is None:
            return
        speed = self.spin_jog_speed.value()
        acc = self.spin_jog_acc.value()
        try:
            self.sdk.jog(axis, float(-speed), float(acc))
            self.log.info(f"Axis_{axis} JOG- 速度={speed}")
        except Exception as e:
            self.log.error(f"JOG- 失败: {e}")

    def _on_jog_stop(self):
        """JOG 停止"""
        if not self.sdk._connected:
            return
        axis = self.cmb_motion_axis.currentData()
        if axis is None:
            return
        try:
            self.sdk.axis_stop(axis, Stop_Smooth)
        except Exception:
            pass

    # =====================================================================
    #  单轴运动
    # =====================================================================
    def _on_set_profile(self):
        """设置单轴曲线参数"""
        if not self.sdk._connected:
            return
        axis = self.cmb_motion_axis.currentData()
        if axis is None:
            return
        vini = self.spin_single_vini.value()
        vmax = self.spin_single_vmax.value()
        acc = self.spin_single_acc.value()
        dec = self.spin_single_dec.value()
        profile = self.cmb_single_profile.currentData()
        try:
            self.sdk.set_axis_profile(axis, vini, vmax, acc, dec, 0, profile)
            profile_name = "T型" if profile == Profile_T else "S型"
            self.log.info(f"Axis_{axis} 设置{profile_name}曲线参数")
        except Exception as e:
            self.log.error(f"设置曲线失败: {e}")

    def _on_start_motion(self):
        """启动单轴运动"""
        if not self.sdk._connected:
            return
        axis = self.cmb_motion_axis.currentData()
        if axis is None:
            return
        dist = self.spin_single_dist.value()
        pos_mode = self.cmb_single_pos_mode.currentData()
        try:
            self.sdk.uniaxial(axis, float(dist), pos_mode)
            mode_name = "绝对" if pos_mode == Position_Absolute else "相对"
            self.log.info(f"Axis_{axis} {mode_name}运动: {dist}")
        except Exception as e:
            self.log.error(f"启动运动失败: {e}")

    def _on_stop_motion(self, axis: int):
        """停止轴运动"""
        if not self.sdk._connected:
            return
        try:
            self.sdk.axis_stop(axis, Stop_Smooth)
            self.log.info(f"Axis_{axis} 运动已停止")
        except Exception as e:
            self.log.error(f"停止运动失败: {e}")

    # =====================================================================
    #  系统信息
    # =====================================================================
    def _refresh_sys_info(self):
        """刷新系统信息"""
        if not self.sdk._connected:
            return
        try:
            ver = self.sdk.get_version()
            serial = self.sdk.get_serial_number()
            run_time = self.sdk.get_run_time()
            self._sys_labels["sys_version"].setText(f"0x{ver:08X}")
            self._sys_labels["sys_serial"].setText(str(serial))
            hours = run_time // 3600
            mins = (run_time % 3600) // 60
            secs = run_time % 60
            self._sys_labels["sys_runtime"].setText(f"{hours:02d}:{mins:02d}:{secs:02d}")
            self._sys_labels["sys_conn_state"].setText("已连接")
        except Exception as e:
            self.log.warning(f"刷新系统信息异常: {e}")

    # =====================================================================
    #  样式辅助方法
    # =====================================================================
    def _group_box_style(self) -> str:
        return (
            "QGroupBox {"
            "  font-weight: bold; border: 1px solid #555; border-radius: 4px;"
            "  margin-top: 8px; padding-top: 16px; color: #d4d4d4;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin; left: 10px; padding: 0 4px;"
            "}"
        )

    def _combo_style(self) -> str:
        return (
            "QComboBox {"
            "  background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
            "  border-radius: 3px; padding: 2px 6px; min-width: 80px;"
            "}"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView {"
            "  background-color: #3c3c3c; color: #d4d4d4; selection-background-color: #264f78;"
            "}"
        )

    def _spin_style(self) -> str:
        return (
            "QSpinBox, QDoubleSpinBox {"
            "  background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
            "  border-radius: 3px; padding: 2px 4px;"
            "}"
        )

    def _line_edit_style(self) -> str:
        return (
            "QLineEdit {"
            "  background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
            "  border-radius: 3px; padding: 2px 4px;"
            "}"
        )

    def _small_btn_style(self, color: str = "#2196f3") -> str:
        return (
            f"QPushButton {{"
            f"  background-color: {color}; color: white; border: none;"
            f"  border-radius: 3px; padding: 4px 12px; font-size: 14px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {self._darken_color(color)}; }}"
            f"QPushButton:pressed {{ background-color: {self._darken_color(color, 0.7)}; }}"
            f"QPushButton:disabled {{ background-color: #555; color: #888; }}"
        )

    def _action_btn_style(self, color: str = "#2196f3") -> str:
        return (
            f"QPushButton {{"
            f"  background-color: {color}; color: white; border: none;"
            f"  border-radius: 4px; padding: 6px 16px; font-size: 15px; font-weight: bold;"
            f"}}"
            f"QPushButton:hover {{ background-color: {self._darken_color(color)}; }}"
            f"QPushButton:pressed {{ background-color: {self._darken_color(color, 0.7)}; }}"
            f"QPushButton:disabled {{ background-color: #555; color: #888; }}"
        )

    @staticmethod
    def _darken_color(hex_color: str, factor: float = 0.85) -> str:
        """使颜色变暗"""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02X}{g:02X}{b:02X}"

    # =====================================================================
    #  窗口事件
    # =====================================================================
    def closeEvent(self, event):
        """窗口关闭事件 - 只停止定时器，不断开 NMC 连接"""
        if self._homing:
            self._on_home_stop()
        # 停止刷新定时器（窗口关闭后无需刷新 UI）
        self._refresh_timer.stop()
        # 不再断开 NMC 连接：如果 SDK 是外部传入的共享实例，断开会影响其他组件
        # 如果 SDK 是内部创建的（_external_sdk=False），也不断开，保持与初始化逻辑一致
        self.nmc_connected.emit(False)
        event.accept()

    def reject(self):
        """对话框关闭（ESC键）"""
        self.close()
