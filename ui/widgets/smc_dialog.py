# -*- coding: utf-8 -*-
"""
SMC6480 四轴运动控制卡轴控制面板
================================
设计模式右侧「轴控制」标签页使用的轴控制面板。

功能:
    - 连接状态显示 + 手动重连
    - 多轴切换（Axis0~2 = XYZ 三轴）
    - 位置/状态显示（命令位置、编码器位置、速度、轴状态）
    - 运动控制（JOG 点动、绝对/相对定位、停止）
    - 回零控制（硬件回零、软件回零、设为零点）
    - 伺服使能、软限位
"""

import time
from typing import Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from core.controller import Controller, ControllerError
from core.log_manager import log_info, log_error, log_warning


# 默认控制器 IP（写死，用户可自行修改电脑静态 IP）
DEFAULT_SMC_IP = "192.168.1.11"

# 轴名称映射（Axis0~2 = XYZ）
AXIS_NAMES = ["X", "Y", "Z"]


class SMCAxisControlPanel(QWidget):
    """SMC6480 轴控制面板（可内嵌到设计模式标签页）。"""

    # 连接状态变化信号（True=已连接, False=未连接）
    # 用于通知主窗口同步共享的 Controller 实例引用
    connection_changed = pyqtSignal(bool)

    def __init__(self, parent=None, controller: Optional[Controller] = None):
        super().__init__(parent)
        # 支持外部传入共享的 Controller 实例（与主窗口共用）
        self._external_controller = controller is not None
        self._controller = controller

        # 当前选中轴（0-based）
        self._current_axis = 0

        # 回零状态
        self._homing = False

        self._setup_ui()
        self._connect_signals()

        # 定时刷新位置/状态显示（200ms）
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_status)
        self._refresh_timer.start(200)

        # 初始刷新一次连接状态
        self._update_connection_ui()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def controller(self) -> Optional[Controller]:
        """当前控制器实例。"""
        return self._controller

    def set_controller(self, controller: Optional[Controller]):
        """设置/更新控制器实例（由主窗口在连接后调用）。"""
        self._controller = controller
        self._update_connection_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── 连接状态区 ──
        conn_group = QGroupBox("控制器连接")
        conn_group.setStyleSheet(self._group_style())
        conn_layout = QVBoxLayout(conn_group)
        conn_layout.setSpacing(4)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("连接状态:"))
        self._conn_state_label = QLabel("未连接")
        self._conn_state_label.setStyleSheet(self._label_style("#f44336"))
        status_row.addWidget(self._conn_state_label, 1)

        status_row.addWidget(QLabel("控制器状态:"))
        self._ctrl_state_label = QLabel("--")
        self._ctrl_state_label.setStyleSheet(self._label_style("#d4d4d4"))
        status_row.addWidget(self._ctrl_state_label, 1)
        conn_layout.addLayout(status_row)

        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP 地址:"))
        self._ip_edit = QLineEdit(DEFAULT_SMC_IP)
        self._ip_edit.setStyleSheet(self._edit_style())
        self._ip_edit.setFixedWidth(140)
        ip_row.addWidget(self._ip_edit)

        self._btn_connect = QPushButton("连接")
        self._btn_connect.setStyleSheet(self._btn_style("#1565C0", "#1976D2"))
        ip_row.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("断开")
        self._btn_disconnect.setStyleSheet(self._btn_style("#E65100", "#BF360C"))
        ip_row.addWidget(self._btn_disconnect)
        ip_row.addStretch()
        conn_layout.addLayout(ip_row)

        layout.addWidget(conn_group)

        # ── 轴选择 + 状态显示区 ──  
        axis_group = QGroupBox("轴控制")
        axis_group.setStyleSheet(self._group_style())
        axis_layout = QVBoxLayout(axis_group)
        axis_layout.setSpacing(4)

        # 轴选择行
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("当前轴:"))
        self._axis_combo = QComboBox()
        for i, name in enumerate(AXIS_NAMES):
            self._axis_combo.addItem(f"Axis{i} ({name})")
        self._axis_combo.setStyleSheet(self._combo_style())
        self._axis_combo.setFixedWidth(120)
        sel_row.addWidget(self._axis_combo)
        sel_row.addStretch()
        axis_layout.addLayout(sel_row)

        # 状态显示网格
        status_grid = QGridLayout()
        status_grid.setSpacing(2)

        status_grid.addWidget(QLabel("命令位置:"), 0, 0)
        self._pos_label = QLabel("--")
        self._pos_label.setStyleSheet(self._label_style("#4fc3f7"))
        status_grid.addWidget(self._pos_label, 0, 1)

        status_grid.addWidget(QLabel("编码器:"), 0, 2)
        self._enc_label = QLabel("--")
        self._enc_label.setStyleSheet(self._label_style("#ff9800"))
        status_grid.addWidget(self._enc_label, 0, 3)

        status_grid.addWidget(QLabel("速度:"), 1, 0)
        self._vel_label = QLabel("--")
        self._vel_label.setStyleSheet(self._label_style("#d4d4d4"))
        status_grid.addWidget(self._vel_label, 1, 1)

        status_grid.addWidget(QLabel("轴状态:"), 1, 2)
        self._axis_state_label = QLabel("--")
        self._axis_state_label.setStyleSheet(self._label_style("#d4d4d4"))
        status_grid.addWidget(self._axis_state_label, 1, 3)

        axis_layout.addLayout(status_grid)

        # 运动参数行
        param_row = QHBoxLayout()
        param_row.setSpacing(4)
        param_row.addWidget(QLabel("速度:"))
        self._speed_edit = QLineEdit("2000")
        self._speed_edit.setStyleSheet(self._edit_style())
        self._speed_edit.setFixedWidth(60)
        param_row.addWidget(self._speed_edit)
        param_row.addWidget(QLabel("加速度:"))
        self._acc_edit = QLineEdit("2000")
        self._acc_edit.setStyleSheet(self._edit_style())
        self._acc_edit.setFixedWidth(60)
        param_row.addWidget(self._acc_edit)
        param_row.addWidget(QLabel("目标位置:"))
        self._target_edit = QLineEdit("0")
        self._target_edit.setStyleSheet(self._edit_style())
        self._target_edit.setFixedWidth(80)
        param_row.addWidget(self._target_edit)
        param_row.addStretch()
        axis_layout.addLayout(param_row)

        # 运动控制按钮行
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)

        self._btn_jog_plus = QPushButton("JOG +")
        self._btn_jog_plus.setStyleSheet(self._btn_style("#2E7D32", "#388E3C"))
        ctrl_row.addWidget(self._btn_jog_plus)

        self._btn_jog_minus = QPushButton("JOG -")
        self._btn_jog_minus.setStyleSheet(self._btn_style("#C62828", "#D32F2F"))
        ctrl_row.addWidget(self._btn_jog_minus)

        self._btn_move_abs = QPushButton("绝对定位")
        self._btn_move_abs.setStyleSheet(self._btn_style("#1565C0", "#1976D2"))
        ctrl_row.addWidget(self._btn_move_abs)

        self._btn_move_rel = QPushButton("相对定位")
        self._btn_move_rel.setStyleSheet(self._btn_style("#1565C0", "#1976D2"))
        ctrl_row.addWidget(self._btn_move_rel)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setStyleSheet(self._btn_style("#E65100", "#BF360C"))
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addStretch()
        axis_layout.addLayout(ctrl_row)

        layout.addWidget(axis_group)

        # ── 回零控制区 ──
        home_group = QGroupBox("回零控制")
        home_group.setStyleSheet(self._group_style())
        home_layout = QVBoxLayout(home_group)
        home_layout.setSpacing(4)

        home_status_row = QHBoxLayout()
        home_status_row.addWidget(QLabel("回零状态:"))
        self._home_state_label = QLabel("就绪")
        self._home_state_label.setStyleSheet(self._label_style("#d4d4d4"))
        home_status_row.addWidget(self._home_state_label, 1)
        home_layout.addLayout(home_status_row)

        home_btn_row = QHBoxLayout()
        home_btn_row.setSpacing(4)

        self._btn_home_hw = QPushButton("硬件回零")
        self._btn_home_hw.setStyleSheet(self._btn_style("#1a3a5c", "#2a4a7c"))
        home_btn_row.addWidget(self._btn_home_hw)

        self._btn_home_sw = QPushButton("软件回零")
        self._btn_home_sw.setStyleSheet(self._btn_style("#1a3a5c", "#2a4a7c"))
        home_btn_row.addWidget(self._btn_home_sw)

        self._btn_home_stop = QPushButton("停止回零")
        self._btn_home_stop.setStyleSheet(self._btn_style("#E65100", "#BF360C"))
        home_btn_row.addWidget(self._btn_home_stop)

        self._btn_set_zero = QPushButton("设为零点")
        self._btn_set_zero.setStyleSheet(self._btn_style("#3c3c3c", "#4a4a4a"))
        home_btn_row.addWidget(self._btn_set_zero)
        home_btn_row.addStretch()
        home_layout.addLayout(home_btn_row)

        layout.addWidget(home_group)

        # ── 伺服 / 限位区 ──
        servo_group = QGroupBox("伺服 / 限位")
        servo_group.setStyleSheet(self._group_style())
        servo_layout = QHBoxLayout(servo_group)
        servo_layout.setSpacing(4)

        self._btn_servo_on = QPushButton("伺服使能")
        self._btn_servo_on.setStyleSheet(self._btn_style("#2E7D32", "#388E3C"))
        servo_layout.addWidget(self._btn_servo_on)

        self._btn_servo_off = QPushButton("伺服关闭")
        self._btn_servo_off.setStyleSheet(self._btn_style("#C62828", "#D32F2F"))
        servo_layout.addWidget(self._btn_servo_off)

        servo_layout.addWidget(QLabel("正限位:"))
        self._pos_limit_edit = QLineEdit("2000")
        self._pos_limit_edit.setStyleSheet(self._edit_style())
        self._pos_limit_edit.setFixedWidth(70)
        servo_layout.addWidget(self._pos_limit_edit)

        servo_layout.addWidget(QLabel("负限位:"))
        self._neg_limit_edit = QLineEdit("-60000")
        self._neg_limit_edit.setStyleSheet(self._edit_style())
        self._neg_limit_edit.setFixedWidth(70)
        servo_layout.addWidget(self._neg_limit_edit)

        self._btn_set_limit = QPushButton("设置限位")
        self._btn_set_limit.setStyleSheet(self._btn_style("#3c3c3c", "#4a4a4a"))
        servo_layout.addWidget(self._btn_set_limit)
        servo_layout.addStretch()

        layout.addWidget(servo_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # 样式辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _group_style() -> str:
        return """
            QGroupBox {
                font-weight: bold; 
                font-size: 13px;    
                border: 1px 
                solid #444;
                border-radius: 4px; 
                margin-top: 6px; 
                padding-top: 10px; 
                color: #d4d4d4;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            font-size: 13px; font-weight: bold; color: {color};
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 1px 6px;
        """

    @staticmethod
    def _edit_style() -> str:
        return """
            QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
        """

    @staticmethod
    def _combo_style() -> str:
        return """
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4;
                selection-background-color: #1a3a5c;
            }
        """

    @staticmethod
    def _btn_style(bg: str, hover: str) -> str:
        return f"""
            QPushButton {{
                background-color: {bg}; color: #fff; padding: 3px 8px;
                border: 1px solid #555; border-radius: 3px; font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }}
        """

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self._btn_connect.clicked.connect(self._on_connect)
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)

        # JOG 使用 pressed/released 实现按压-保持
        self._btn_jog_plus.pressed.connect(self._on_jog_plus)
        self._btn_jog_plus.released.connect(self._on_jog_stop)
        self._btn_jog_minus.pressed.connect(self._on_jog_minus)
        self._btn_jog_minus.released.connect(self._on_jog_stop)

        self._btn_move_abs.clicked.connect(self._on_move_abs)
        self._btn_move_rel.clicked.connect(self._on_move_rel)
        self._btn_stop.clicked.connect(self._on_stop)

        self._btn_home_hw.clicked.connect(self._on_home_hw)
        self._btn_home_sw.clicked.connect(self._on_home_sw)
        self._btn_home_stop.clicked.connect(self._on_home_stop)
        self._btn_set_zero.clicked.connect(self._on_set_zero)

        self._btn_servo_on.clicked.connect(self._on_servo_on)
        self._btn_servo_off.clicked.connect(self._on_servo_off)
        self._btn_set_limit.clicked.connect(self._on_set_limit)

    # ------------------------------------------------------------------
    # 连接 / 断开
    # ------------------------------------------------------------------
    def _on_connect(self):
        """手动连接控制器。"""
        ip = self._ip_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "提示", "请输入控制器 IP 地址")
            return
        try:
            if self._controller is None:
                self._controller = Controller()
            self._controller.connect_eth(ip)
            log_info(f"SMC6480 连接成功: {ip}")
            self._update_connection_ui()
            self.connection_changed.emit(True)
        except ControllerError as e:
            log_error(f"SMC6480 连接失败: {e}")
            QMessageBox.warning(self, "连接失败", str(e))
            self._update_connection_ui()

    def _on_disconnect(self):
        """断开控制器连接。"""
        if self._controller is not None:
            try:
                self._controller.disconnect()
                log_info("SMC6480 已断开")
            except Exception as e:
                log_error(f"断开失败: {e}")
        self._update_connection_ui()
        self.connection_changed.emit(False)

    def _update_connection_ui(self):
        """根据连接状态更新 UI。"""
        connected = self._controller is not None and self._controller.is_connected
        if connected:
            self._conn_state_label.setText("已连接")
            self._conn_state_label.setStyleSheet(self._label_style("#4caf50"))
            self._btn_connect.setEnabled(False)
            self._btn_disconnect.setEnabled(True)
            # 读取控制器状态
            try:
                desc = self._controller.get_state_desc()
                self._ctrl_state_label.setText(desc)
            except Exception:
                self._ctrl_state_label.setText("--")
        else:
            self._conn_state_label.setText("未连接")
            self._conn_state_label.setStyleSheet(self._label_style("#f44336"))
            self._ctrl_state_label.setText("--")
            self._btn_connect.setEnabled(True)
            self._btn_disconnect.setEnabled(False)
            # 未连接时清空位置/状态显示
            self._pos_label.setText("--")
            self._enc_label.setText("--")
            self._vel_label.setText("--")
            self._axis_state_label.setText("--")

    # ------------------------------------------------------------------
    # 轴切换
    # ------------------------------------------------------------------
    def _on_axis_changed(self, index: int):
        """切换当前轴。"""
        self._current_axis = index
        self._refresh_status()

    # ------------------------------------------------------------------
    # 定时刷新
    # ------------------------------------------------------------------
    def _refresh_status(self):
        """定时刷新位置/状态显示。"""
        if self._controller is None or not self._controller.is_connected:
            return
        axis = self._current_axis
        try:
            pos = self._controller.get_pulse_position(axis)
            self._pos_label.setText(str(pos))
        except Exception:
            self._pos_label.setText("--")

        try:
            enc = self._controller.get_encoder_position(axis)
            self._enc_label.setText(str(enc))
        except Exception:
            self._enc_label.setText("--")

        try:
            vel = self._controller.get_cur_speed(axis)
            self._vel_label.setText(str(vel))
        except Exception:
            self._vel_label.setText("--")

        try:
            down = self._controller.check_down(axis)
            self._axis_state_label.setText("停止" if down else "运动中")
            self._axis_state_label.setStyleSheet(
                self._label_style("#4caf50" if down else "#ff9800")
            )
        except Exception:
            self._axis_state_label.setText("--")

        # 回零状态轮询：若正在回零中，用 Motion_Home_IfHoming 判断是否已完成
        if self._homing:
            try:
                still_homing = self._controller.if_home_moving(axis)
            except Exception:
                # 查询失败时保持"回零中"状态，等待下次轮询
                return
            if not still_homing:
                # 回零流程已结束（轴已停止、位置归零）
                self._homing = False
                self._home_state_label.setText("已回零")
                self._home_state_label.setStyleSheet(self._label_style("#4caf50"))
                log_info(f"Axis{axis} 回零完成")

    # ------------------------------------------------------------------
    # 运动控制
    # ------------------------------------------------------------------
    def _get_params(self):
        """读取速度/加速度/目标位置输入。返回 (speed, acc, target) 或 None。"""
        try:
            speed = float(self._speed_edit.text().strip())
            acc = float(self._acc_edit.text().strip())
            target = float(self._target_edit.text().strip())
            return speed, acc, target
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的数值")
            return None

    def _check_connected(self) -> bool:
        """检查是否已连接，未连接则提示。"""
        if self._controller is None or not self._controller.is_connected:
            QMessageBox.warning(self, "提示", "控制器未连接，请先连接")
            return False
        return True

    def _on_jog_plus(self):
        """JOG 正向移动（按压触发）。"""
        if not self._check_connected():
            return
        params = self._get_params()
        if params is None:
            return
        speed, acc, _ = params
        try:
            self._controller.vmove(self._current_axis, positive=True, speed=speed)
            log_info(f"Axis{self._current_axis} JOG+ (速度: {speed})")
        except ControllerError as e:
            log_error(f"JOG+ 失败: {e}")

    def _on_jog_minus(self):
        """JOG 反向移动（按压触发）。"""
        if not self._check_connected():
            return
        params = self._get_params()
        if params is None:
            return
        speed, acc, _ = params
        try:
            self._controller.vmove(self._current_axis, positive=False, speed=speed)
            log_info(f"Axis{self._current_axis} JOG- (速度: {speed})")
        except ControllerError as e:
            log_error(f"JOG- 失败: {e}")

    def _on_jog_stop(self):
        """JOG 停止（释放按钮触发）。"""
        if self._controller is None or not self._controller.is_connected:
            return
        try:
            self._controller.decel_stop(self._current_axis)
            log_info(f"Axis{self._current_axis} JOG 停止")
        except ControllerError as e:
            log_error(f"JOG 停止失败: {e}")

    def _on_move_abs(self):
        """绝对定位。"""
        if not self._check_connected():
            return
        params = self._get_params()
        if params is None:
            return
        speed, acc, target = params
        try:
            self._controller.set_motion_params(self._current_axis, 0, speed, acc, acc, 0)
            self._controller.pmove_abs(self._current_axis, target)
            log_info(f"Axis{self._current_axis} 绝对定位到 {target}")
        except ControllerError as e:
            log_error(f"绝对定位失败: {e}")
            QMessageBox.warning(self, "定位失败", str(e))

    def _on_move_rel(self):
        """相对定位。"""
        if not self._check_connected():
            return
        params = self._get_params()
        if params is None:
            return
        speed, acc, target = params
        try:
            self._controller.set_motion_params(self._current_axis, 0, speed, acc, acc, 0)
            self._controller.pmove_rel(self._current_axis, target)
            log_info(f"Axis{self._current_axis} 相对移动 {target}")
        except ControllerError as e:
            log_error(f"相对定位失败: {e}")
            QMessageBox.warning(self, "定位失败", str(e))

    def _on_stop(self):
        """停止轴。"""
        if self._controller is None or not self._controller.is_connected:
            return
        try:
            self._controller.imd_stop(self._current_axis)
            log_info(f"Axis{self._current_axis} 已停止")
        except ControllerError as e:
            log_error(f"停止失败: {e}")

    # ------------------------------------------------------------------
    # 回零控制
    # ------------------------------------------------------------------
    def _on_home_hw(self):
        """硬件回零（Motion_Home_FindOrigin，依赖原点信号）。"""
        if not self._check_connected():
            return
        try:
            self._controller.home_move(self._current_axis)
            self._homing = True
            self._home_state_label.setText("硬件回零中...")
            self._home_state_label.setStyleSheet(self._label_style("#ff9800"))
            log_info(f"Axis{self._current_axis} 硬件回零开始")
        except ControllerError as e:
            log_error(f"硬件回零失败: {e}")
            QMessageBox.warning(self, "回零失败", str(e))

    def _on_home_sw(self):
        """软件回零（移动到 0 坐标）。"""
        if not self._check_connected():
            return
        try:
            # 使用当前速度参数移动到 0 坐标
            params = self._get_params()
            speed = params[0] if params else 10000
            acc = params[1] if params else 10000
            self._controller.set_motion_params(self._current_axis, 0, speed, acc, acc, 0)
            self._controller.pmove_abs(self._current_axis, 0)
            self._homing = True
            self._home_state_label.setText("软件回零中...")
            self._home_state_label.setStyleSheet(self._label_style("#ff9800"))
            log_info(f"Axis{self._current_axis} 软件回零（移动到 0）")
        except ControllerError as e:
            log_error(f"软件回零失败: {e}")
            QMessageBox.warning(self, "回零失败", str(e))

    def _on_home_stop(self):
        """停止回零。"""
        if self._controller is None or not self._controller.is_connected:
            return
        try:
            self._controller.imd_stop(self._current_axis)
            self._homing = False
            self._home_state_label.setText("已停止")
            self._home_state_label.setStyleSheet(self._label_style("#f44336"))
            log_info(f"Axis{self._current_axis} 回零已停止")
        except ControllerError as e:
            log_error(f"停止回零失败: {e}")

    def _on_set_zero(self):
        """将当前位置设为零点。"""
        if not self._check_connected():
            return
        try:
            pos = self._controller.get_pulse_position(self._current_axis)
            self._controller.set_pulse_position(self._current_axis, 0)
            self._home_state_label.setText(f"已设为零点 (原位置: {pos})")
            self._home_state_label.setStyleSheet(self._label_style("#4caf50"))
            log_info(f"Axis{self._current_axis} 当前位置 {pos} 已设为零点")
        except ControllerError as e:
            log_error(f"设为零点失败: {e}")

    # ------------------------------------------------------------------
    # 伺服 / 限位
    # ------------------------------------------------------------------
    def _on_servo_on(self):
        """伺服使能。"""
        if not self._check_connected():
            return
        try:
            # 通过设置当前位置为当前值来触发使能（SMC 无独立使能接口时）
            # 此处调用 set_pulse_position 保持当前位置，作为使能占位
            pos = self._controller.get_pulse_position(self._current_axis)
            self._controller.set_pulse_position(self._current_axis, pos)
            log_info(f"Axis{self._current_axis} 伺服使能")
        except ControllerError as e:
            log_error(f"伺服使能失败: {e}")

    def _on_servo_off(self):
        """伺服关闭。"""
        if not self._check_connected():
            return
        try:
            self._controller.imd_stop(self._current_axis)
            log_info(f"Axis{self._current_axis} 伺服关闭")
        except ControllerError as e:
            log_error(f"伺服关闭失败: {e}")

    def _on_set_limit(self):
        """设置软限位。"""
        if not self._check_connected():
            return
        try:
            pos_limit = int(self._pos_limit_edit.text().strip())
            neg_limit = int(self._neg_limit_edit.text().strip())
            # SMC6480 软限位通过 set_pulse_position 无法直接设置，
            # 此处预留接口，记录日志
            log_info(f"Axis{self._current_axis} 设置软限位: +{pos_limit}, {neg_limit}")
            QMessageBox.information(self, "提示", "软限位设置接口已预留（需根据 SMC6480 手册补充）")
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的限位数值")

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def cleanup(self):
        """停止定时器并清理资源。"""
        self._refresh_timer.stop()
        # 仅当控制器由本面板创建时才断开（外部共享的不在此断开）
        if self._controller is not None and not self._external_controller:
            try:
                self._controller.disconnect()
            except Exception:
                pass
