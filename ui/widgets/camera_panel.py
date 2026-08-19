# -*- coding: utf-8 -*-
"""
相机控制面板
============
提供相机设备选择、打开/关闭、实时预览、参数调节（曝光/增益/帧率）、
触发模式切换、拍照等功能。
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage

from camera_manager import CameraManager
from core.log_manager import log_info, log_error
from .zoomable_label import ZoomableLabel


class CameraPanel(QWidget):
    """
    相机控制面板。

    信号:
        frame_received(int, int, int, bytes): 实时帧数据信号
        capture_completed(int, int, int, bytes): 拍照完成信号
        status_message(str): 状态消息信号
    """

    frame_received = pyqtSignal(int, int, int, bytes)
    capture_completed = pyqtSignal(int, int, int, bytes)
    status_message = pyqtSignal(str)

    # 参数调节步长
    EXPOSURE_STEP = 500.0      # 微秒
    GAIN_STEP = 1.0            # dB
    FRAME_RATE_STEP = 1.0      # fps

    def __init__(self, parent=None, camera_mgr=None):
        super().__init__(parent)
        self.cam_mgr = camera_mgr if camera_mgr is not None else CameraManager()
        self._device_data = {}
        self._is_capturing = False

        # 参数范围缓存
        self._exposure_min = 0.0
        self._exposure_max = 100000.0
        self._gain_min = 0.0
        self._gain_max = 24.0

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """构建 UI 布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ========== 顶部控制栏 ==========
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(4)

        lbl_camera = QLabel("相机:")
        lbl_camera.setStyleSheet("color: #d4d4d4; font-weight: bold;")

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(200)
        self.device_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 2px 6px; min-height: 20px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4;
                selection-background-color: #1a3a5c;
            }
        """)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 2px 10px; border: 1px solid #555;
                border-radius: 3px; min-height: 20px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """)

        self.open_btn = QPushButton("打开")
        self.open_btn.setEnabled(False)
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a3a5c; color: #4A90D9;
                padding: 2px 10px; border: 1px solid #2a5a8c;
                border-radius: 3px; min-height: 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2a4a7c; }
            QPushButton:disabled { background-color: #3c3c3c; color: #666; border-color: #555; }
        """)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setEnabled(False)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #5c1a1a; color: #D94A4A;
                padding: 2px 10px; border: 1px solid #8c2a2a;
                border-radius: 3px; min-height: 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #7c2a2a; }
            QPushButton:disabled { background-color: #3c3c3c; color: #666; border-color: #555; }
        """)

        self.capture_btn = QPushButton("📷 拍照")
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2; color: #fff; font-weight: bold;
                padding: 3px 14px; border: none; border-radius: 3px;
                font-size: 13px; min-height: 22px;
            }
            QPushButton:hover { background-color: #1565C0; }
            QPushButton:disabled { background-color: #3c3c3c; color: #666; }
        """)

        ctrl_layout.addWidget(lbl_camera)
        ctrl_layout.addWidget(self.device_combo, 1)
        ctrl_layout.addWidget(self.refresh_btn)
        ctrl_layout.addWidget(self.open_btn)
        ctrl_layout.addWidget(self.close_btn)
        ctrl_layout.addWidget(self.capture_btn)

        # ========== 主区域（显示 + 参数面板） ==========
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：图像显示
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(4)

        display_title = QLabel("实时画面")
        display_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #d4d4d4;")

        self.display_label = ZoomableLabel("相机未打开")
        self.display_label.setMinimumSize(400, 300)
        self.display_label.setStyleSheet("""
            ZoomableLabel {
                background-color: #0d0d0d; border: 1px solid #444;
                border-radius: 3px;
            }
        """)

        display_layout.addWidget(display_title)
        display_layout.addWidget(self.display_label, 1)

        # 右侧：参数面板
        param_widget = QWidget()
        param_widget.setMinimumWidth(200)
        param_widget.setMaximumWidth(260)
        param_layout = QVBoxLayout(param_widget)
        param_layout.setContentsMargins(4, 0, 0, 0)
        param_layout.setSpacing(4)

        # -- 触发模式 --
        trigger_group = QGroupBox("触发模式")
        trigger_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 13px; border: 1px solid #444;
                border-radius: 3px; margin-top: 6px; padding-top: 10px;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
                color: #d4d4d4;
            }
        """)
        trigger_layout = QVBoxLayout(trigger_group)
        trigger_layout.setContentsMargins(4, 4, 4, 4)
        trigger_layout.setSpacing(4)

        self.trigger_combo = QComboBox()
        self.trigger_combo.addItems(["连续采集", "触发模式（软触发）"])
        self.trigger_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4;
                selection-background-color: #1a3a5c;
            }
        """)

        self.trigger_btn = QPushButton("发送软触发")
        self.trigger_btn.setEnabled(False)
        self.trigger_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 3px 8px; border: 1px solid #555;
                border-radius: 3px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; }
        """)

        trigger_layout.addWidget(self.trigger_combo)
        trigger_layout.addWidget(self.trigger_btn)

        # -- 曝光时间 --
        exp_group = QGroupBox("曝光时间")
        exp_group.setStyleSheet(trigger_group.styleSheet())
        exp_layout = QVBoxLayout(exp_group)
        exp_layout.setContentsMargins(4, 4, 4, 4)
        exp_layout.setSpacing(2)

        exp_slider_layout = QHBoxLayout()
        self.exp_slider = QSlider(Qt.Horizontal)
        self.exp_slider.setRange(0, 1000)
        self.exp_slider.setValue(500)
        self.exp_value_label = QLabel("-- us")
        self.exp_value_label.setMinimumWidth(80)
        self.exp_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.exp_value_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")

        exp_slider_layout.addWidget(self.exp_slider, 1)
        exp_slider_layout.addWidget(self.exp_value_label)

        exp_btn_layout = QHBoxLayout()
        self.exp_minus_btn = QPushButton("--")
        self.exp_plus_btn = QPushButton("++")
        for btn in [self.exp_minus_btn, self.exp_plus_btn]:
            btn.setFixedWidth(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4;
                    border: 1px solid #555; border-radius: 3px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #4a4a4a; }
                QPushButton:disabled { background-color: #2d2d2d; color: #555; }
            """)
        self.exp_auto_check = QCheckBox("自动曝光")
        self.exp_auto_check.setStyleSheet("color: #d4d4d4;")
        self.exp_auto_check.setChecked(False)  # 默认关闭自动曝光，使用固定值

        exp_btn_layout.addWidget(self.exp_minus_btn)
        exp_btn_layout.addWidget(self.exp_plus_btn)
        exp_btn_layout.addWidget(self.exp_auto_check)
        exp_btn_layout.addStretch()

        exp_layout.addLayout(exp_slider_layout)
        exp_layout.addLayout(exp_btn_layout)

        # -- 增益 --
        gain_group = QGroupBox("增益")
        gain_group.setStyleSheet(trigger_group.styleSheet())
        gain_layout = QVBoxLayout(gain_group)
        gain_layout.setContentsMargins(4, 4, 4, 4)
        gain_layout.setSpacing(2)

        gain_slider_layout = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 240)  # 0.0 ~ 24.0 dB
        self.gain_slider.setValue(0)
        self.gain_value_label = QLabel("-- dB")
        self.gain_value_label.setMinimumWidth(80)
        self.gain_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.gain_value_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")

        gain_slider_layout.addWidget(self.gain_slider, 1)
        gain_slider_layout.addWidget(self.gain_value_label)

        gain_btn_layout = QHBoxLayout()
        self.gain_minus_btn = QPushButton("--")
        self.gain_plus_btn = QPushButton("++")
        for btn in [self.gain_minus_btn, self.gain_plus_btn]:
            btn.setFixedWidth(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4;
                    border: 1px solid #555; border-radius: 3px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #4a4a4a; }
                QPushButton:disabled { background-color: #2d2d2d; color: #555; }
            """)
        self.gain_auto_check = QCheckBox("自动增益")
        self.gain_auto_check.setStyleSheet("color: #d4d4d4;")
        self.gain_auto_check.setChecked(False)  # 默认关闭自动增益，使用固定值

        gain_btn_layout.addWidget(self.gain_minus_btn)
        gain_btn_layout.addWidget(self.gain_plus_btn)
        gain_btn_layout.addWidget(self.gain_auto_check)
        gain_btn_layout.addStretch()

        gain_layout.addLayout(gain_slider_layout)
        gain_layout.addLayout(gain_btn_layout)

        # -- 帧率 --
        fps_group = QGroupBox("帧率")
        fps_group.setStyleSheet(trigger_group.styleSheet())
        fps_layout = QVBoxLayout(fps_group)
        fps_layout.setContentsMargins(4, 4, 4, 4)
        fps_layout.setSpacing(2)

        fps_slider_layout = QHBoxLayout()
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(0, 300)  # 0 ~ 30.0 fps
        self.fps_slider.setValue(0)
        self.fps_value_label = QLabel("-- fps")
        self.fps_value_label.setMinimumWidth(80)
        self.fps_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.fps_value_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")

        fps_slider_layout.addWidget(self.fps_slider, 1)
        fps_slider_layout.addWidget(self.fps_value_label)

        fps_btn_layout = QHBoxLayout()
        self.fps_minus_btn = QPushButton("--")
        self.fps_plus_btn = QPushButton("++")
        for btn in [self.fps_minus_btn, self.fps_plus_btn]:
            btn.setFixedWidth(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4;
                    border: 1px solid #555; border-radius: 3px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #4a4a4a; }
                QPushButton:disabled { background-color: #2d2d2d; color: #555; }
            """)

        fps_btn_layout.addWidget(self.fps_minus_btn)
        fps_btn_layout.addWidget(self.fps_plus_btn)
        fps_btn_layout.addStretch()

        fps_layout.addLayout(fps_slider_layout)
        fps_layout.addLayout(fps_btn_layout)

        # -- 白平衡 --
        wb_group = QGroupBox("白平衡")
        wb_group.setStyleSheet(trigger_group.styleSheet())
        wb_layout = QVBoxLayout(wb_group)
        wb_layout.setContentsMargins(4, 4, 4, 4)
        wb_layout.setSpacing(2)

        # 白平衡预设按钮
        wb_preset_layout = QHBoxLayout()
        self.wb_preset_combo = QComboBox()
        self.wb_preset_combo.addItems(["自定义", "日光 (6500K)", "荧光灯 (4000K)", "白炽灯 (2800K)"])
        self.wb_preset_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4;
                selection-background-color: #1a3a5c;
            }
        """)
        self.wb_apply_btn = QPushButton("应用")
        self.wb_apply_btn.setFixedWidth(50)
        self.wb_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                font-weight: bold; padding: 4px 8px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; }
        """)
        wb_preset_layout.addWidget(QLabel("预设:"), 0)
        wb_preset_layout.addWidget(self.wb_preset_combo, 1)
        wb_preset_layout.addWidget(self.wb_apply_btn)

        # R 通道滑块
        r_layout = QHBoxLayout()
        r_label = QLabel("R:")
        r_label.setStyleSheet("color: #ef5350; font-weight: bold;")
        self.wb_r_slider = QSlider(Qt.Horizontal)
        self.wb_r_slider.setRange(50, 300)  # 0.5 ~ 3.0
        self.wb_r_slider.setValue(150)      # 1.5
        self.wb_r_value = QLabel("1.50")
        self.wb_r_value.setMinimumWidth(40)
        self.wb_r_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wb_r_value.setStyleSheet("color: #ef5350; font-weight: bold;")
        r_layout.addWidget(r_label)
        r_layout.addWidget(self.wb_r_slider, 1)
        r_layout.addWidget(self.wb_r_value)

        # G 通道滑块
        g_layout = QHBoxLayout()
        g_label = QLabel("G:")
        g_label.setStyleSheet("color: #66bb6a; font-weight: bold;")
        self.wb_g_slider = QSlider(Qt.Horizontal)
        self.wb_g_slider.setRange(50, 300)  # 0.5 ~ 3.0
        self.wb_g_slider.setValue(100)      # 1.0
        self.wb_g_value = QLabel("1.00")
        self.wb_g_value.setMinimumWidth(40)
        self.wb_g_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wb_g_value.setStyleSheet("color: #66bb6a; font-weight: bold;")
        g_layout.addWidget(g_label)
        g_layout.addWidget(self.wb_g_slider, 1)
        g_layout.addWidget(self.wb_g_value)

        # B 通道滑块
        b_layout = QHBoxLayout()
        b_label = QLabel("B:")
        b_label.setStyleSheet("color: #42a5f5; font-weight: bold;")
        self.wb_b_slider = QSlider(Qt.Horizontal)
        self.wb_b_slider.setRange(50, 300)  # 0.5 ~ 3.0
        self.wb_b_slider.setValue(180)      # 1.8
        self.wb_b_value = QLabel("1.80")
        self.wb_b_value.setMinimumWidth(40)
        self.wb_b_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wb_b_value.setStyleSheet("color: #42a5f5; font-weight: bold;")
        b_layout.addWidget(b_label)
        b_layout.addWidget(self.wb_b_slider, 1)
        b_layout.addWidget(self.wb_b_value)

        wb_layout.addLayout(wb_preset_layout)
        wb_layout.addLayout(r_layout)
        wb_layout.addLayout(g_layout)
        wb_layout.addLayout(b_layout)

        # 组装参数面板
        param_layout.addWidget(trigger_group)
        param_layout.addWidget(exp_group)
        param_layout.addWidget(gain_group)
        param_layout.addWidget(fps_group)
        param_layout.addWidget(wb_group)
        param_layout.addStretch()

        splitter.addWidget(display_widget)
        splitter.addWidget(param_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # ========== 底部状态栏 ==========
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1e1e1e; color: #999;
                border-top: 1px solid #444; padding: 2px 8px;
            }
        """)
        self.status_bar.showMessage("就绪")

        # 组装主布局
        main_layout.addLayout(ctrl_layout)
        main_layout.addWidget(splitter, 1)
        main_layout.addWidget(self.status_bar)

        # 白平衡控件初始状态：禁用（相机未打开时不可调节）
        self.wb_r_slider.setEnabled(False)
        self.wb_g_slider.setEnabled(False)
        self.wb_b_slider.setEnabled(False)
        self.wb_preset_combo.setEnabled(False)
        self.wb_apply_btn.setEnabled(False)

    def _connect_signals(self):
        """连接信号与槽"""
        self.refresh_btn.clicked.connect(self.enumerate_devices)
        self.open_btn.clicked.connect(self.open_camera)
        self.close_btn.clicked.connect(self.close_camera)
        self.capture_btn.clicked.connect(self.capture_once)

        # 触发模式
        self.trigger_combo.currentIndexChanged.connect(self._on_trigger_mode_changed)
        self.trigger_btn.clicked.connect(self._on_soft_trigger)

        # 曝光
        self.exp_slider.valueChanged.connect(self._on_exp_slider_changed)
        self.exp_minus_btn.clicked.connect(lambda: self._adjust_exposure(-self.EXPOSURE_STEP))
        self.exp_plus_btn.clicked.connect(lambda: self._adjust_exposure(self.EXPOSURE_STEP))
        self.exp_auto_check.stateChanged.connect(self._on_exp_auto_changed)

        # 增益
        self.gain_slider.valueChanged.connect(self._on_gain_slider_changed)
        self.gain_minus_btn.clicked.connect(lambda: self._adjust_gain(-self.GAIN_STEP))
        self.gain_plus_btn.clicked.connect(lambda: self._adjust_gain(self.GAIN_STEP))
        self.gain_auto_check.stateChanged.connect(self._on_gain_auto_changed)

        # 帧率
        self.fps_slider.valueChanged.connect(self._on_fps_slider_changed)
        self.fps_minus_btn.clicked.connect(lambda: self._adjust_frame_rate(-self.FRAME_RATE_STEP))
        self.fps_plus_btn.clicked.connect(lambda: self._adjust_frame_rate(self.FRAME_RATE_STEP))

        # 白平衡
        self.wb_r_slider.valueChanged.connect(self._on_wb_r_changed)
        self.wb_g_slider.valueChanged.connect(self._on_wb_g_changed)
        self.wb_b_slider.valueChanged.connect(self._on_wb_b_changed)
        self.wb_apply_btn.clicked.connect(self._on_wb_apply_preset)

    # ========== 设备枚举 ==========

    def enumerate_devices(self, callback=None):
        """异步枚举相机设备

        Args:
            callback: 可选的回调函数，枚举完成后调用，接收 (devices, success) 参数
        """
        # 兼容按钮信号传递的布尔参数（QPushButton.clicked 会传递 checked 状态）
        if isinstance(callback, bool):
            callback = None

        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("搜索中...")
        self.status_bar.showMessage("正在搜索相机...")
        self.status_message.emit("正在搜索相机...")

        class EnumThread(QThread):
            finished = pyqtSignal(object)

            def __init__(self, cam_mgr):
                super().__init__()
                self.cam_mgr = cam_mgr

            def run(self):
                devices = self.cam_mgr.enumerate_devices(timeout_ms=200)
                self.finished.emit(devices)

        self._enum_thread = EnumThread(self.cam_mgr)
        self._enum_thread.finished.connect(lambda devices: self._on_enum_finished(devices, callback))
        self._enum_thread.start()

    def _on_enum_finished(self, devices, callback=None):
        try:
            self.device_combo.clear()
            self._device_data = {}
            for device in devices:
                idx = device.get("index", 0)
                name = device.get("name", "未知")
                model = device.get("model", "")
                serial = device.get("serial", "")
                ip = device.get("ip", "")
                dev_info = device.get("_raw_info")
                if dev_info is None:
                    continue

                # 显示名称: 型号 + 序列号
                display_name = f"{model} ({serial})" if serial else name
                if ip and ip != "N/A":
                    display_name += f" [{ip}]"

                self.device_combo.addItem(display_name, idx)
                self._device_data[idx] = {
                    "dev_info": dev_info,
                    "info": device,
                }

            self.open_btn.setEnabled(len(devices) > 0)
            msg = f"发现 {len(devices)} 个相机"
            self.status_bar.showMessage(msg)
            self.status_message.emit(msg)

            # 调用回调
            if callback is not None:
                callback(devices, len(devices) > 0)
        except RuntimeError:
            if callback is not None:
                callback([], False)
        finally:
            try:
                self.refresh_btn.setEnabled(True)
                self.refresh_btn.setText("刷新")
            except RuntimeError:
                pass

    # ========== 打开/关闭 ==========

    def open_camera(self):
        if self.cam_mgr.is_open:
            QMessageBox.warning(self, "提示", "相机已打开")
            return

        idx = self.device_combo.currentData()
        if idx is None:
            QMessageBox.warning(self, "提示", "请先刷新并选择相机")
            return

        device_entry = self._device_data.get(idx)
        if device_entry is None:
            QMessageBox.warning(self, "提示", "设备信息无效，请重新刷新")
            return

        dev_info = device_entry["dev_info"]

        try:
            success = self.cam_mgr.open_camera(dev_info)
            if not success:
                QMessageBox.critical(self, "错误", "打开相机失败，请检查相机连接")
                return

            # 不启动实时取流，仅在需要时拍照（capture_once）
            # 避免后台取流线程持续占用CPU和主线程资源

            # 更新 UI 状态
            self.open_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
            self.trigger_combo.setEnabled(True)
            self.trigger_btn.setEnabled(self.cam_mgr.is_trigger_mode)
            self.display_label.clear_pixmap()
            self.display_label.setText("相机已就绪，点击「拍照」获取图像")

            # 关闭自动曝光和自动增益，使用固定参数
            self.cam_mgr.set_enum_param("ExposureAuto", 0)  # GxAutoEntry.OFF
            self.cam_mgr.set_enum_param("GainAuto", 0)       # GxAutoEntry.OFF

            # 设置固定曝光时间和增益
            self.cam_mgr.set_exposure_time(18000.0)           # 曝光时间 18000 µs
            self.cam_mgr.set_gain(0.0)                        # 增益 0 dB

            # 设置白平衡系数（关闭自动白平衡，手动设置 R/G/B 三通道独立系数）
            # 画面偏绿校正：降低 G 通道（1.0），适当提升 R（1.5）和 B（1.8）
            self._set_white_balance(ratio_r=1.5, ratio_g=1.0, ratio_b=1.8)

            # 读取当前参数并更新 UI
            self._refresh_params()

            # 根据自动曝光/增益状态更新滑块可用性
            self.exp_slider.setEnabled(not self.exp_auto_check.isChecked())
            self.exp_minus_btn.setEnabled(not self.exp_auto_check.isChecked())
            self.exp_plus_btn.setEnabled(not self.exp_auto_check.isChecked())
            self.gain_slider.setEnabled(not self.gain_auto_check.isChecked())
            self.gain_minus_btn.setEnabled(not self.gain_auto_check.isChecked())
            self.gain_plus_btn.setEnabled(not self.gain_auto_check.isChecked())

            # 启用白平衡控件
            self.wb_r_slider.setEnabled(True)
            self.wb_g_slider.setEnabled(True)
            self.wb_b_slider.setEnabled(True)
            self.wb_preset_combo.setEnabled(True)
            self.wb_apply_btn.setEnabled(True)

            msg = "相机已打开"
            self.status_bar.showMessage(msg)
            self.status_message.emit(msg)
            log_info("相机已打开")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            log_error(f"打开相机失败: {e}")

    def close_camera(self):
        self.cam_mgr.close_camera()

        if not self.isVisible() and not self.isEnabled():
            return

        try:
            self.open_btn.setEnabled(True)
            self.close_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)
            self.trigger_combo.setEnabled(False)
            self.trigger_btn.setEnabled(False)
            self.display_label.clear_pixmap()
            self.display_label.setText("相机未打开")

            # 重置参数显示
            self.exp_value_label.setText("-- us")
            self.gain_value_label.setText("-- dB")
            self.fps_value_label.setText("-- fps")

            # 禁用白平衡控件
            self.wb_r_slider.setEnabled(False)
            self.wb_g_slider.setEnabled(False)
            self.wb_b_slider.setEnabled(False)
            self.wb_preset_combo.setEnabled(False)
            self.wb_apply_btn.setEnabled(False)

            msg = "相机已关闭"
            self.status_bar.showMessage(msg)
            self.status_message.emit(msg)
            log_info("相机已关闭")
        except RuntimeError:
            pass

    # ========== 白平衡设置 ==========

    def _set_white_balance(self, ratio_r: float = 1.5, ratio_g: float = 1.0, ratio_b: float = 1.8):
        """
        设置白平衡系数（关闭自动白平衡，手动设置 R/G/B 三通道独立系数）。

        Daheng SDK 通过 BalanceRatioSelector 选择通道（Red=0, Green=1, Blue=2），
        然后通过 BalanceRatio 设置对应通道的系数。

        画面偏绿的原因通常是 G 通道增益相对 R/B 过高。
        默认值 ratio_g=1.0（降低绿色），ratio_r=1.5（提升红色），ratio_b=1.8（提升蓝色），
        以校正偏绿问题。用户可根据实际画面微调。

        Args:
            ratio_r: Red 通道系数（默认 1.5）
            ratio_g: Green 通道系数（默认 1.0，降低绿色以校正偏绿）
            ratio_b: Blue 通道系数（默认 1.8，提升蓝色以校正偏绿）
        """
        if not self.cam_mgr.is_open:
            return

        try:
            # 关闭自动白平衡
            self.cam_mgr.set_enum_param("BalanceWhiteAuto", 0)  # GxAutoEntry.OFF

            # 分别设置 R、G、B 三个通道的独立系数
            from gxipy.gxidef import GxBalanceRatioSelectorEntry

            # Red 通道
            self.cam_mgr.set_enum_param("BalanceRatioSelector", GxBalanceRatioSelectorEntry.RED)
            self.cam_mgr.set_float_param("BalanceRatio", ratio_r)

            # Green 通道（降低绿色以校正偏绿）
            self.cam_mgr.set_enum_param("BalanceRatioSelector", GxBalanceRatioSelectorEntry.GREEN)
            self.cam_mgr.set_float_param("BalanceRatio", ratio_g)

            # Blue 通道（提升蓝色以校正偏绿）
            self.cam_mgr.set_enum_param("BalanceRatioSelector", GxBalanceRatioSelectorEntry.BLUE)
            self.cam_mgr.set_float_param("BalanceRatio", ratio_b)

            log_info(f"白平衡系数已设置: R={ratio_r:.2f}, G={ratio_g:.2f}, B={ratio_b:.2f}")
        except Exception as e:
            log_error(f"设置白平衡失败: {e}")

    # ========== 参数刷新 ==========

    def _refresh_params(self):
        """从相机读取当前参数并更新 UI"""
        if not self.cam_mgr.is_open:
            return

        # 曝光时间
        exp = self.cam_mgr.get_exposure_time()
        if exp is not None:
            self.exp_slider.blockSignals(True)
            # 映射到滑块范围（假设范围 0~100000us）
            slider_val = int(exp / 100.0) if exp < 100000 else 1000
            self.exp_slider.setValue(min(slider_val, 1000))
            self.exp_slider.blockSignals(False)
            self.exp_value_label.setText(f"{exp:.0f} us")

        # 增益
        gain = self.cam_mgr.get_gain()
        if gain is not None:
            self.gain_slider.blockSignals(True)
            slider_val = int(gain * 10)  # 0.1dB 精度
            self.gain_slider.setValue(min(slider_val, 240))
            self.gain_slider.blockSignals(False)
            self.gain_value_label.setText(f"{gain:.1f} dB")

        # 帧率
        fps = self.cam_mgr.get_frame_rate()
        if fps is not None:
            self.fps_slider.blockSignals(True)
            slider_val = int(fps * 10)  # 0.1fps 精度
            self.fps_slider.setValue(min(slider_val, 300))
            self.fps_slider.blockSignals(False)
            self.fps_value_label.setText(f"{fps:.1f} fps")

        # 触发模式
        self.trigger_combo.blockSignals(True)
        self.trigger_combo.setCurrentIndex(1 if self.cam_mgr.is_trigger_mode else 0)
        self.trigger_combo.blockSignals(False)
        self.trigger_btn.setEnabled(self.cam_mgr.is_trigger_mode)

    # ========== 触发模式 ==========

    def _on_trigger_mode_changed(self, index):
        if not self.cam_mgr.is_open:
            return
        enable = (index == 1)  # 1 = 触发模式
        self.cam_mgr.set_trigger_mode(enable)
        self.trigger_btn.setEnabled(enable)
        mode_str = "触发模式" if enable else "连续采集"
        self.status_bar.showMessage(f"已切换至: {mode_str}")
        self.status_message.emit(f"已切换至: {mode_str}")

    def _on_soft_trigger(self):
        if self.cam_mgr.is_trigger_mode:
            if self.cam_mgr.trigger_once():
                self.status_bar.showMessage("软触发信号已发送")
                self.status_message.emit("软触发信号已发送")
            else:
                self.status_bar.showMessage("软触发失败")
        else:
            self.status_bar.showMessage("当前为连续模式，无需触发")

    # ========== 曝光控制 ==========

    def _on_exp_slider_changed(self, value):
        if not self.cam_mgr.is_open:
            return
        # 滑块值 0~1000 映射到曝光时间
        exp_us = value * 100.0  # 0 ~ 100000 us
        if self.cam_mgr.set_exposure_time(exp_us):
            self.exp_value_label.setText(f"{exp_us:.0f} us")

    def _adjust_exposure(self, delta_us):
        if not self.cam_mgr.is_open:
            return
        current = self.cam_mgr.get_exposure_time()
        if current is None:
            return
        new_val = max(10.0, current + delta_us)
        if self.cam_mgr.set_exposure_time(new_val):
            self._refresh_params()
            self.status_bar.showMessage(f"曝光时间: {new_val:.0f} us")

    def _on_exp_auto_changed(self, state):
        if not self.cam_mgr.is_open:
            return
        if state == Qt.Checked:
            self.cam_mgr.set_enum_param("ExposureAuto", 1)  # 1 = 连续自动
            self.exp_slider.setEnabled(False)
            self.exp_minus_btn.setEnabled(False)
            self.exp_plus_btn.setEnabled(False)
            self.status_bar.showMessage("已开启自动曝光")
        else:
            self.cam_mgr.set_enum_param("ExposureAuto", 0)  # 0 = 关闭
            self.exp_slider.setEnabled(True)
            self.exp_minus_btn.setEnabled(True)
            self.exp_plus_btn.setEnabled(True)
            self.status_bar.showMessage("已关闭自动曝光")
            self._refresh_params()

    # ========== 增益控制 ==========

    def _on_gain_slider_changed(self, value):
        if not self.cam_mgr.is_open:
            return
        gain_db = value / 10.0  # 0.0 ~ 24.0 dB
        if self.cam_mgr.set_gain(gain_db):
            self.gain_value_label.setText(f"{gain_db:.1f} dB")

    def _adjust_gain(self, delta_db):
        if not self.cam_mgr.is_open:
            return
        current = self.cam_mgr.get_gain()
        if current is None:
            return
        new_val = max(0.0, current + delta_db)
        if self.cam_mgr.set_gain(new_val):
            self._refresh_params()
            self.status_bar.showMessage(f"增益: {new_val:.1f} dB")

    def _on_gain_auto_changed(self, state):
        if not self.cam_mgr.is_open:
            return
        if state == Qt.Checked:
            self.cam_mgr.set_enum_param("GainAuto", 1)
            self.gain_slider.setEnabled(False)
            self.gain_minus_btn.setEnabled(False)
            self.gain_plus_btn.setEnabled(False)
            self.status_bar.showMessage("已开启自动增益")
        else:
            self.cam_mgr.set_enum_param("GainAuto", 0)
            self.gain_slider.setEnabled(True)
            self.gain_minus_btn.setEnabled(True)
            self.gain_plus_btn.setEnabled(True)
            self.status_bar.showMessage("已关闭自动增益")
            self._refresh_params()

    # ========== 帧率控制 ==========

    def _on_fps_slider_changed(self, value):
        if not self.cam_mgr.is_open:
            return
        fps = value / 10.0  # 0.0 ~ 30.0 fps
        if fps > 0 and self.cam_mgr.set_frame_rate(fps):
            self.fps_value_label.setText(f"{fps:.1f} fps")

    def _adjust_frame_rate(self, delta_fps):
        if not self.cam_mgr.is_open:
            return
        current = self.cam_mgr.get_frame_rate()
        if current is None:
            return
        new_val = max(1.0, current + delta_fps)
        if self.cam_mgr.set_frame_rate(new_val):
            self._refresh_params()
            self.status_bar.showMessage(f"帧率: {new_val:.1f} fps")

    # ========== 白平衡控制 ==========

    def _apply_white_balance(self):
        """将当前滑块值应用到相机"""
        if not self.cam_mgr.is_open:
            return
        r = self.wb_r_slider.value() / 100.0
        g = self.wb_g_slider.value() / 100.0
        b = self.wb_b_slider.value() / 100.0
        self._set_white_balance(ratio_r=r, ratio_g=g, ratio_b=b)
        self.status_bar.showMessage(f"白平衡: R={r:.2f}, G={g:.2f}, B={b:.2f}")

    def _on_wb_r_changed(self, value):
        """Red 通道滑块变化"""
        val = value / 100.0
        self.wb_r_value.setText(f"{val:.2f}")
        self.wb_preset_combo.setCurrentIndex(0)  # 切换回"自定义"
        self._apply_white_balance()

    def _on_wb_g_changed(self, value):
        """Green 通道滑块变化"""
        val = value / 100.0
        self.wb_g_value.setText(f"{val:.2f}")
        self.wb_preset_combo.setCurrentIndex(0)
        self._apply_white_balance()

    def _on_wb_b_changed(self, value):
        """Blue 通道滑块变化"""
        val = value / 100.0
        self.wb_b_value.setText(f"{val:.2f}")
        self.wb_preset_combo.setCurrentIndex(0)
        self._apply_white_balance()

    def _on_wb_apply_preset(self):
        """应用白平衡预设"""
        idx = self.wb_preset_combo.currentIndex()
        if idx == 0:
            return  # 自定义，不做改变
        presets = {
            1: (1.8, 1.0, 1.2),   # 日光 6500K: 偏蓝，提升 R，降低 B
            2: (1.5, 1.0, 1.6),   # 荧光灯 4000K: 偏绿，降低 G
            3: (1.2, 1.0, 2.0),   # 白炽灯 2800K: 偏黄/红，提升 B
        }
        if idx in presets:
            r, g, b = presets[idx]
            self.wb_r_slider.blockSignals(True)
            self.wb_g_slider.blockSignals(True)
            self.wb_b_slider.blockSignals(True)
            self.wb_r_slider.setValue(int(r * 100))
            self.wb_g_slider.setValue(int(g * 100))
            self.wb_b_slider.setValue(int(b * 100))
            self.wb_r_slider.blockSignals(False)
            self.wb_g_slider.blockSignals(False)
            self.wb_b_slider.blockSignals(False)
            self.wb_r_value.setText(f"{r:.2f}")
            self.wb_g_value.setText(f"{g:.2f}")
            self.wb_b_value.setText(f"{b:.2f}")
            self._apply_white_balance()
            self.status_bar.showMessage(f"白平衡预设: {self.wb_preset_combo.currentText()}")

    # ========== 帧回调与拍照 ==========

    def _on_frame_received(self, width, height, pixel_type, img_bytes):
        """实时帧回调"""
        self.frame_received.emit(width, height, pixel_type, img_bytes)
        # 使用 ZoomableLabel 显示（支持缩放）
        qimg = CameraManager.convert_to_qimage(width, height, pixel_type, img_bytes)
        if qimg is not None:
            self.display_label.set_image(qimg)

    def capture_once(self):
        """拍照"""
        if self._is_capturing:
            return

        # 检查控件是否已被删除（防止相机初始化失败后的对象生命周期问题）
        try:
            self.capture_btn.isWidgetType()
        except RuntimeError:
            return

        self._is_capturing = True
        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("拍照中...")
        self.status_bar.showMessage("拍照中...")
        self.status_message.emit("拍照中...")

        class CaptureThread(QThread):
            finished = pyqtSignal(object)

            def __init__(self, cam_mgr):
                super().__init__()
                self.cam_mgr = cam_mgr

            def run(self):
                result = self.cam_mgr.capture_once(timeout_ms=3000)
                self.finished.emit(result)

        self._capture_thread = CaptureThread(self.cam_mgr)
        self._capture_thread.finished.connect(self._on_capture_finished)
        self._capture_thread.start()

    def _on_capture_finished(self, result):
        self._is_capturing = False
        self.capture_btn.setEnabled(True)
        self.capture_btn.setText("📷 拍照")

        if result is not None:
            width, height, pixel_type, img_bytes = result
            # 使用 ZoomableLabel 显示（支持缩放）
            qimg = CameraManager.convert_to_qimage(width, height, pixel_type, img_bytes)
            if qimg is not None:
                self.display_label.set_image(qimg)
            self.capture_completed.emit(width, height, pixel_type, img_bytes)
            msg = f"拍照完成: {width}x{height}"
            self.status_bar.showMessage(msg)
            self.status_message.emit(msg)
            log_info(msg)
        else:
            QMessageBox.warning(self, "拍照失败", "获取图像超时或失败，请重试")
            self.status_bar.showMessage("拍照失败")
            self.status_message.emit("拍照失败")
            log_error("拍照失败")

    # ========== 自动连接 ==========

    def auto_connect_camera(self):
        """自动枚举并连接第一个可用的相机设备"""
        self.status_bar.showMessage("正在自动搜索相机...")
        self.status_message.emit("正在自动搜索相机...")
        self.enumerate_devices(callback=self._on_auto_enum_finished)

    def _on_auto_enum_finished(self, devices, found):
        if found and len(devices) > 0:
            # 自动选择第一个设备
            self.device_combo.setCurrentIndex(0)
            # 自动打开相机
            self.open_camera()
        else:
            msg = "未发现相机设备，请手动连接"
            self.status_bar.showMessage(msg)
            self.status_message.emit(msg)
            log_info("自动连接相机: 未发现设备")

    # ========== 公共接口 ==========

    def is_camera_open(self) -> bool:
        return self.cam_mgr.is_open
