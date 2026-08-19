# -*- coding: utf-8 -*-
"""
串口通信对话框
==============
独立的串口通信窗口，通过菜单栏「通信 > 串口通信」打开。

功能:
    - 端口扫描与选择
    - 串口参数配置（波特率/数据位/校验位/停止位/流控制）
    - 打开/关闭串口连接
    - 文本/HEX 两种模式发送数据
    - 实时接收数据显示（支持 HEX 显示模式）
    - 自动滚动
    - 收发字节统计
    - 配置持久化
"""

from typing import Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QTextCursor, QFont

from core.serial_comm import (
    SerialCommManager,
    list_ports,
    BAUDRATES,
    PARITY_NAMES,
    STOP_BITS_VALUES,
    FLOW_CONTROL_NAMES,
    DEFAULT_BAUDRATE,
    DEFAULT_PARITY,
    DEFAULT_STOP_BITS,
    DEFAULT_FLOW_CONTROL,
)
from core.log_manager import log_info, log_error
from core.config_manager import ConfigManager


class SerialDialog(QDialog):
    """串口通信对话框。"""

    # 配置前缀
    CONFIG_PREFIX = "serial_comm"

    def __init__(self, parent=None, comm_mgr: Optional[SerialCommManager] = None):
        super().__init__(parent)
        self._external_comm = comm_mgr is not None  # 标记是否为外部传入的 comm_mgr
        self._comm_mgr = comm_mgr or SerialCommManager()
        self._config_mgr = ConfigManager()

        self._setup_ui()
        self._connect_signals()
        self._load_config()
        self._refresh_ports()

        self.setWindowTitle("串口通信")
        self.setMinimumSize(720, 600)
        self.resize(800, 680)

    # ──────────────────────────────────────────────
    # UI 构建
    # ──────────────────────────────────────────────

    def _setup_ui(self):
        """构建 UI 布局。"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ========== 连接设置区域 ==========
        conn_group = QGroupBox("连接设置")
        conn_group.setStyleSheet(self._group_box_style())
        conn_layout = QGridLayout(conn_group)
        conn_layout.setSpacing(6)

        # 端口选择
        conn_layout.addWidget(QLabel("端口:"), 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(200)
        self.port_combo.setStyleSheet(self._combo_style())
        conn_layout.addWidget(self.port_combo, 0, 1)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setStyleSheet(self._btn_style())
        self.refresh_btn.setToolTip("扫描可用串口")
        conn_layout.addWidget(self.refresh_btn, 0, 2)

        # 波特率
        conn_layout.addWidget(QLabel("波特率:"), 0, 3)
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems([str(b) for b in BAUDRATES])
        self.baudrate_combo.setCurrentText(str(DEFAULT_BAUDRATE))
        self.baudrate_combo.setEditable(True)
        self.baudrate_combo.setStyleSheet(self._combo_style())
        conn_layout.addWidget(self.baudrate_combo, 0, 4)

        # 数据位
        conn_layout.addWidget(QLabel("数据位:"), 1, 0)
        self.databits_combo = QComboBox()
        self.databits_combo.addItems(["5", "6", "7", "8"])
        self.databits_combo.setCurrentText("8")
        self.databits_combo.setStyleSheet(self._combo_style())
        conn_layout.addWidget(self.databits_combo, 1, 1)

        # 校验位
        conn_layout.addWidget(QLabel("校验位:"), 1, 2)
        self.parity_combo = QComboBox()
        self.parity_combo.addItems(PARITY_NAMES)
        self.parity_combo.setCurrentText(DEFAULT_PARITY)
        self.parity_combo.setStyleSheet(self._combo_style())
        conn_layout.addWidget(self.parity_combo, 1, 3)

        # 停止位
        conn_layout.addWidget(QLabel("停止位:"), 1, 4)
        self.stopbits_combo = QComboBox()
        self.stopbits_combo.addItems([str(s) for s in STOP_BITS_VALUES])
        self.stopbits_combo.setCurrentText(str(DEFAULT_STOP_BITS))
        self.stopbits_combo.setStyleSheet(self._combo_style())
        conn_layout.addWidget(self.stopbits_combo, 1, 5)

        # 流控制
        conn_layout.addWidget(QLabel("流控制:"), 2, 0)
        self.flowctl_combo = QComboBox()
        self.flowctl_combo.addItems(FLOW_CONTROL_NAMES)
        self.flowctl_combo.setCurrentText(DEFAULT_FLOW_CONTROL)
        self.flowctl_combo.setStyleSheet(self._combo_style())
        conn_layout.addWidget(self.flowctl_combo, 2, 1)

        # 打开/关闭按钮
        self.open_btn = QPushButton("打开串口")
        self.open_btn.setMinimumHeight(36)
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #388E3C; color: #fff; font-size: 18px;
                font-weight: bold; padding: 6px 24px;
                border: 2px solid #4CAF50; border-radius: 6px;
            }
            QPushButton:hover { background-color: #2E7D32; border-color: #66BB6A; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)
        conn_layout.addWidget(self.open_btn, 2, 2, 1, 2)

        # 连接状态指示
        self.conn_status_label = QLabel("● 未连接")
        self.conn_status_label.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 16px;")
        conn_layout.addWidget(self.conn_status_label, 2, 4, 1, 2)

        main_layout.addWidget(conn_group)

        # ========== 发送区域 ==========
        send_group = QGroupBox("发送区")
        send_group.setStyleSheet(self._group_box_style())
        send_layout = QVBoxLayout(send_group)
        send_layout.setSpacing(4)

        send_toolbar = QHBoxLayout()
        self.hex_send_cb = QCheckBox("HEX 发送")
        self.hex_send_cb.setStyleSheet("color: #d4d4d4;")
        send_toolbar.addWidget(self.hex_send_cb)

        self.append_newline_cb = QCheckBox("添加换行")
        self.append_newline_cb.setStyleSheet("color: #d4d4d4;")
        self.append_newline_cb.setToolTip("发送时自动添加 \\r\\n")
        send_toolbar.addWidget(self.append_newline_cb)

        send_toolbar.addStretch()

        self.send_btn = QPushButton("发送")
        self.send_btn.setMinimumHeight(30)
        self.send_btn.setEnabled(False)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2; color: #fff; font-weight: bold;
                padding: 4px 20px; border: 1px solid #2196F3; border-radius: 4px;
            }
            QPushButton:hover { background-color: #1565C0; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)
        send_toolbar.addWidget(self.send_btn)

        self.clear_send_btn = QPushButton("清空")
        self.clear_send_btn.setStyleSheet(self._small_btn_style())
        send_toolbar.addWidget(self.clear_send_btn)

        send_layout.addLayout(send_toolbar)

        self.send_edit = QTextEdit()
        self.send_edit.setPlaceholderText("请输入要发送的数据...")
        self.send_edit.setMaximumHeight(80)
        self.send_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a; color: #c8c8c8;
                border: 1px solid #444; border-radius: 3px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 16px; padding: 4px;
            }
        """)
        send_layout.addWidget(self.send_edit)

        main_layout.addWidget(send_group)

        # ========== 接收区域 ==========
        recv_group = QGroupBox("接收区")
        recv_group.setStyleSheet(self._group_box_style())
        recv_layout = QVBoxLayout(recv_group)
        recv_layout.setSpacing(4)

        recv_toolbar = QHBoxLayout()
        self.hex_display_cb = QCheckBox("HEX 显示")
        self.hex_display_cb.setStyleSheet("color: #d4d4d4;")
        recv_toolbar.addWidget(self.hex_display_cb)

        self.auto_scroll_cb = QCheckBox("自动滚动")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.setStyleSheet("color: #d4d4d4;")
        recv_toolbar.addWidget(self.auto_scroll_cb)

        recv_toolbar.addStretch()

        self.clear_recv_btn = QPushButton("清空接收")
        self.clear_recv_btn.setStyleSheet(self._small_btn_style())
        recv_toolbar.addWidget(self.clear_recv_btn)

        recv_layout.addLayout(recv_toolbar)

        self.recv_view = QTextEdit()
        self.recv_view.setReadOnly(True)
        self.recv_view.setStyleSheet("""
            QTextEdit {
                background-color: #0d0d0d; color: #c8c8c8;
                border: 1px solid #444; border-radius: 3px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 16px; padding: 4px;
            }
        """)
        recv_layout.addWidget(self.recv_view, 1)

        main_layout.addWidget(recv_group, 1)

        # ========== 状态栏 ==========
        status_bar = QWidget()
        status_bar.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #444;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(4, 2, 4, 2)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #999; font-size: 15px;")
        status_layout.addWidget(self.status_label, 1)

        self.rx_label = QLabel("RX: 0")
        self.rx_label.setStyleSheet("color: #4fc3f7; font-size: 15px; padding: 0 8px;")
        status_layout.addWidget(self.rx_label)

        self.tx_label = QLabel("TX: 0")
        self.tx_label.setStyleSheet("color: #ffa726; font-size: 15px; padding: 0 8px;")
        status_layout.addWidget(self.tx_label)

        main_layout.addWidget(status_bar)

    # ── 样式辅助 ──

    def _group_box_style(self):
        return """
            QGroupBox {
                font-weight: bold; font-size: 16px; border: 1px solid #444;
                border-radius: 4px; margin-top: 10px; padding-top: 16px;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
                color: #d4d4d4;
            }
        """

    def _combo_style(self):
        return """
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 4px 8px; min-height: 24px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4;
                selection-background-color: #1a3a5c;
            }
        """

    def _btn_style(self):
        return """
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 4px 14px; border: 1px solid #555;
                border-radius: 3px; min-height: 24px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #4A90D9; }
        """

    def _small_btn_style(self):
        return """
            QPushButton {
                background-color: #3c3c3c; color: #b0b0b0;
                padding: 3px 12px; border: 1px solid #555;
                border-radius: 3px; font-size: 15px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """

    # ──────────────────────────────────────────────
    # 信号连接
    # ──────────────────────────────────────────────

    def _connect_signals(self):
        """连接信号与槽。"""
        # 按钮
        self.refresh_btn.clicked.connect(self._refresh_ports)
        self.open_btn.clicked.connect(self._toggle_connection)
        self.send_btn.clicked.connect(self._send_data)
        self.clear_send_btn.clicked.connect(lambda: self.send_edit.clear())
        self.clear_recv_btn.clicked.connect(lambda: self.recv_view.clear())

        # 快捷键：Enter 发送
        self.send_edit.installEventFilter(self)

        # 串口管理器信号
        self._comm_mgr.data_received.connect(self._on_data_received)
        self._comm_mgr.connection_changed.connect(self._on_connection_changed)
        self._comm_mgr.error_occurred.connect(self._on_error)
        self._comm_mgr.rx_count_changed.connect(
            lambda c: self.rx_label.setText(f"RX: {c}"))
        self._comm_mgr.tx_count_changed.connect(
            lambda c: self.tx_label.setText(f"TX: {c}"))

    def eventFilter(self, obj, event):
        """事件过滤器：在发送编辑框中按 Ctrl+Enter 发送。"""
        if obj is self.send_edit and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() & Qt.ControlModifier:
                self._send_data()
                return True
        return super().eventFilter(obj, event)

    # ──────────────────────────────────────────────
    # 端口扫描
    # ──────────────────────────────────────────────

    def _refresh_ports(self):
        """刷新可用串口列表。"""
        current_port = self.port_combo.currentText()
        self.port_combo.clear()

        ports = list_ports()
        if ports:
            for p in ports:
                label = f"{p['device']} - {p['description']}"
                self.port_combo.addItem(label, p["device"])
            self.status_label.setText(f"扫描到 {len(ports)} 个串口")
            log_info(f"串口扫描: 发现 {len(ports)} 个设备")
        else:
            self.port_combo.addItem("未发现串口设备", None)
            self.status_label.setText("未发现串口设备")
            log_info("串口扫描: 未发现设备")

        # 恢复之前选择的端口
        if current_port:
            idx = self.port_combo.findText(current_port, Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)

    # ──────────────────────────────────────────────
    # 连接管理
    # ──────────────────────────────────────────────

    def _toggle_connection(self):
        """切换串口连接状态。"""
        if self._comm_mgr.is_open:
            self._comm_mgr.close()
        else:
            self._do_open()

    def _do_open(self):
        """打开串口连接。"""
        # 获取端口
        port_data = self.port_combo.currentData()
        if not port_data:
            QMessageBox.warning(self, "提示", "请先选择一个有效的串口端口")
            return

        # 获取参数
        try:
            baudrate = int(self.baudrate_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "提示", "波特率格式错误")
            return

        bytesize = int(self.databits_combo.currentText())
        parity = self.parity_combo.currentText()
        stopbits = float(self.stopbits_combo.currentText())
        flow_control = self.flowctl_combo.currentText()

        # 配置并打开
        self._comm_mgr.set_config(
            port=port_data,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            flow_control=flow_control,
        )

        self.status_label.setText(f"正在打开 {port_data}...")
        QApplication.processEvents()

        if self._comm_mgr.open():
            self.status_label.setText(f"已连接 {port_data} ({baudrate} bps)")
            log_info(f"串口已打开: {port_data} @ {baudrate}")
            self._save_config()
        else:
            self.status_label.setText("打开串口失败")

    def _on_connection_changed(self, connected: bool):
        """连接状态变化回调。"""
        if connected:
            self.open_btn.setText("关闭串口")
            self.open_btn.setStyleSheet("""
                QPushButton {
                    background-color: #C62828; color: #fff; font-size: 18px;
                    font-weight: bold; padding: 6px 24px;
                    border: 2px solid #EF5350; border-radius: 6px;
                }
                QPushButton:hover { background-color: #B71C1C; border-color: #E57373; }
            """)
            self.conn_status_label.setText("● 已连接")
            self.conn_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 16px;")
            self.send_btn.setEnabled(True)

            # 禁用参数修改
            self.port_combo.setEnabled(False)
            self.baudrate_combo.setEnabled(False)
            self.databits_combo.setEnabled(False)
            self.parity_combo.setEnabled(False)
            self.stopbits_combo.setEnabled(False)
            self.flowctl_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
        else:
            self.open_btn.setText("打开串口")
            self.open_btn.setStyleSheet("""
                QPushButton {
                    background-color: #388E3C; color: #fff; font-size: 18px;
                    font-weight: bold; padding: 6px 24px;
                    border: 2px solid #4CAF50; border-radius: 6px;
                }
                QPushButton:hover { background-color: #2E7D32; border-color: #66BB6A; }
            """)
            self.conn_status_label.setText("● 未连接")
            self.conn_status_label.setStyleSheet("color: #ff5252; font-weight: bold; font-size: 16px;")
            self.send_btn.setEnabled(False)

            # 启用参数修改
            self.port_combo.setEnabled(True)
            self.baudrate_combo.setEnabled(True)
            self.databits_combo.setEnabled(True)
            self.parity_combo.setEnabled(True)
            self.stopbits_combo.setEnabled(True)
            self.flowctl_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)

    # ──────────────────────────────────────────────
    # 数据发送
    # ──────────────────────────────────────────────

    def _send_data(self):
        """发送数据。"""
        text = self.send_edit.toPlainText()
        if not text.strip():
            return

        if self.hex_send_cb.isChecked():
            # HEX 模式发送
            count = self._comm_mgr.send_hex(text)
            if count > 0:
                self.status_label.setText(f"已发送 {count} 字节 (HEX)")
        else:
            # 文本模式发送
            count = self._comm_mgr.send_text(
                text,
                append_newline=self.append_newline_cb.isChecked(),
            )
            if count > 0:
                mode = " +换行" if self.append_newline_cb.isChecked() else ""
                self.status_label.setText(f"已发送 {count} 字节 (文本{mode})")

    # ──────────────────────────────────────────────
    # 数据接收
    # ──────────────────────────────────────────────

    def _on_data_received(self, data: bytes):
        """接收到串口数据。"""
        if self.hex_display_cb.isChecked():
            # HEX 显示模式
            hex_str = " ".join(f"{b:02X}" for b in data)
            self._append_to_recv(hex_str + " ")
        else:
            # 文本显示模式
            try:
                text = data.decode("utf-8", errors="replace")
                # 替换控制字符为可读形式
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                self._append_to_recv(text)
            except Exception:
                hex_str = " ".join(f"{b:02X}" for b in data)
                self._append_to_recv(hex_str + " ")

    def _append_to_recv(self, text: str):
        """追加文本到接收区。"""
        self.recv_view.moveCursor(QTextCursor.End)
        self.recv_view.insertPlainText(text)
        if self.auto_scroll_cb.isChecked():
            self.recv_view.moveCursor(QTextCursor.End)

    # ──────────────────────────────────────────────
    # 错误处理
    # ──────────────────────────────────────────────

    def _on_error(self, error_msg: str):
        """错误回调。"""
        self.status_label.setText(f"错误: {error_msg}")
        log_error(f"串口通信错误: {error_msg}")

    # ──────────────────────────────────────────────
    # 配置持久化
    # ──────────────────────────────────────────────

    def _load_config(self):
        """加载保存的配置。"""
        try:
            prefix = self.CONFIG_PREFIX

            # 恢复参数
            baudrate = self._config_mgr.get(f"{prefix}.baudrate")
            if baudrate is not None:
                self.baudrate_combo.setCurrentText(str(baudrate))

            bytesize = self._config_mgr.get(f"{prefix}.bytesize")
            if bytesize is not None:
                self.databits_combo.setCurrentText(str(bytesize))

            parity = self._config_mgr.get(f"{prefix}.parity")
            if parity is not None:
                idx = self.parity_combo.findText(parity)
                if idx >= 0:
                    self.parity_combo.setCurrentIndex(idx)

            stopbits = self._config_mgr.get(f"{prefix}.stopbits")
            if stopbits is not None:
                self.stopbits_combo.setCurrentText(str(stopbits))

            flow_control = self._config_mgr.get(f"{prefix}.flow_control")
            if flow_control is not None:
                idx = self.flowctl_combo.findText(flow_control)
                if idx >= 0:
                    self.flowctl_combo.setCurrentIndex(idx)

            # 恢复选项
            hex_send = self._config_mgr.get(f"{prefix}.hex_send")
            if hex_send is not None:
                self.hex_send_cb.setChecked(hex_send)

            hex_display = self._config_mgr.get(f"{prefix}.hex_display")
            if hex_display is not None:
                self.hex_display_cb.setChecked(hex_display)

            auto_scroll = self._config_mgr.get(f"{prefix}.auto_scroll")
            if auto_scroll is not None:
                self.auto_scroll_cb.setChecked(auto_scroll)

            append_newline = self._config_mgr.get(f"{prefix}.append_newline")
            if append_newline is not None:
                self.append_newline_cb.setChecked(append_newline)

        except Exception as e:
            log_error(f"加载串口配置失败: {e}")

    def _save_config(self):
        """保存当前配置。"""
        try:
            prefix = self.CONFIG_PREFIX
            self._config_mgr.set(f"{prefix}.port", self._comm_mgr.port)
            self._config_mgr.set(f"{prefix}.baudrate", self._comm_mgr.baudrate)
            self._config_mgr.set(f"{prefix}.bytesize", int(self.databits_combo.currentText()))
            self._config_mgr.set(f"{prefix}.parity", self.parity_combo.currentText())
            self._config_mgr.set(f"{prefix}.stopbits", float(self.stopbits_combo.currentText()))
            self._config_mgr.set(f"{prefix}.flow_control", self.flowctl_combo.currentText())
            self._config_mgr.set(f"{prefix}.hex_send", self.hex_send_cb.isChecked())
            self._config_mgr.set(f"{prefix}.hex_display", self.hex_display_cb.isChecked())
            self._config_mgr.set(f"{prefix}.auto_scroll", self.auto_scroll_cb.isChecked())
            self._config_mgr.set(f"{prefix}.append_newline", self.append_newline_cb.isChecked())
            self._config_mgr.save()
        except Exception as e:
            log_error(f"保存串口配置失败: {e}")

    # ──────────────────────────────────────────────
    # 窗口事件
    # ──────────────────────────────────────────────

    def closeEvent(self, event):
        """窗口关闭时自动断开串口连接。"""
        if not self._external_comm:
            self._comm_mgr.cleanup()
        log_info("串口通信窗口关闭")
        super().closeEvent(event)

    def reject(self):
        """点击关闭按钮或按 ESC 时。"""
        if not self._external_comm:
            self._comm_mgr.cleanup()
        log_info("串口通信窗口关闭")
        super().reject()
