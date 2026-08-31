# -*- coding: utf-8 -*-
"""
自动化检测面板
============
提供自动化检测模式的 UI 界面，包含动态网格布局显示多位置检测结果、
产品选择、状态监控、统计信息和执行日志。

与 InspectionWorkflow 通过信号连接，实时更新 UI。
"""

from typing import Optional, List, Dict, Any
import time

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QGroupBox, QScrollArea, QFrame,
    QSplitter, QSizePolicy, QTextEdit, QMessageBox
)

from core.product_manager import list_products, load_product
from core.inspection_workflow import InspectionWorkflow, PositionResult
from core.log_manager import log_info, log_error


# ============================================================================
# 单个位置结果展示控件
# ============================================================================

class PositionResultWidget(QFrame):
    """单个位置的检测结果展示控件"""

    def __init__(self, position_name: str, parent=None):
        super().__init__(parent)
        self._position_name = position_name
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
            }
        """)
        self.setMinimumSize(200, 160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)

        # 标题栏：位置名称 + 结果状态
        title_bar = QHBoxLayout()
        title_bar.setSpacing(4)

        self._name_label = QLabel(self._position_name)
        self._name_label.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #d4d4d4;
            border: none; background: transparent;
        """)

        self._result_label = QLabel("等待检测")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setStyleSheet("""
            font-size: 12px; font-weight: bold; color: #666;
            background-color: #2d2d2d; border: 1px solid #444;
            border-radius: 3px; padding: 1px 8px;
            min-width: 40px;
        """)

        title_bar.addWidget(self._name_label)
        title_bar.addStretch()
        title_bar.addWidget(self._result_label)
        layout.addLayout(title_bar)

        # 图像显示区
        self._image_label = QLabel("等待检测...")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(180, 120)
        self._image_label.setStyleSheet("""
            QLabel {
                background-color: #0d0d0d; border: 1px solid #333;
                border-radius: 3px; color: #555;
                font-size: 13px;
            }
        """)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._image_label, 1)

        # 消息栏
        self._message_label = QLabel("")
        self._message_label.setStyleSheet("""
            font-size: 11px; color: #999;
            border: none; background: transparent;
        """)
        self._message_label.setWordWrap(True)
        layout.addWidget(self._message_label)

    def show_result(self, result: PositionResult):
        """显示检测结果"""
        # 更新结果标签
        if result.passed:
            self._result_label.setText("OK")
            self._result_label.setStyleSheet("""
                font-size: 12px; font-weight: bold; color: #66BB6A;
                background-color: #1a3a1a; border: 1px solid #4CAF50;
                border-radius: 3px; padding: 1px 8px;
                min-width: 40px;
            """)
            self.setStyleSheet("""
                QFrame {
                    background-color: #1e2a1e;
                    border: 1px solid #2E7D32;
                    border-radius: 4px;
                }
            """)
        else:
            self._result_label.setText("NG")
            self._result_label.setStyleSheet("""
                font-size: 12px; font-weight: bold; color: #EF5350;
                background-color: #2a1a1a; border: 1px solid #C62828;
                border-radius: 3px; padding: 1px 8px;
                min-width: 40px;
            """)
            self.setStyleSheet("""
                QFrame {
                    background-color: #2a1e1e;
                    border: 1px solid #C62828;
                    border-radius: 4px;
                }
            """)

        # 显示标注图
        if result.annotated is not None:
            self._display_image(result.annotated)
        elif result.raw_image is not None:
            self._display_image(result.raw_image)

        # 更新消息（含条码详情）
        msg = result.message or ""
        if getattr(result, "barcodes", None):
            parts = []
            for bc in result.barcodes:
                btype = bc.get("type", "")
                bdata = bc.get("data", "")
                conf = bc.get("confidence", 0)
                parts.append(f"{btype}:{bdata} (置信度{conf:.2f})")
            if parts:
                msg = msg + "\n" + " | ".join(parts)
        self._message_label.setText(msg)

    def show_waiting(self):
        """显示等待状态"""
        self._result_label.setText("等待检测")
        self._result_label.setStyleSheet("""
            font-size: 12px; font-weight: bold; color: #666;
            background-color: #2d2d2d; border: 1px solid #444;
            border-radius: 3px; padding: 1px 8px;
            min-width: 40px;
        """)
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
            }
        """)
        self._image_label.setText("等待检测...")
        self._message_label.setText("")

    def show_capturing(self):
        """显示拍照中状态"""
        self._image_label.setText("📷 拍照中...")
        self._image_label.setStyleSheet("""
            QLabel {
                background-color: #0d0d0d; border: 1px solid #4A90D9;
                border-radius: 3px; color: #4A90D9;
                font-size: 14px;
            }
        """)

    def show_testing(self):
        """显示检测中状态"""
        self._image_label.setText("🔍 检测中...")
        self._image_label.setStyleSheet("""
            QLabel {
                background-color: #0d0d0d; border: 1px solid #FFA000;
                border-radius: 3px; color: #FFA000;
                font-size: 14px;
            }
        """)

    def _display_image(self, cv_img: np.ndarray):
        """显示 OpenCV 图像"""
        try:
            height, width = cv_img.shape[:2]
            if len(cv_img.shape) == 2:
                bytes_per_line = width
                q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                bytes_per_line = 3 * width
                q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_BGR888)

            pixmap = QPixmap.fromImage(q_img)
            scaled = pixmap.scaled(
                self._image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)
            self._image_label.setStyleSheet("""
                QLabel {
                    background-color: #0d0d0d; border: 1px solid #333;
                    border-radius: 4px;
                }
            """)
        except Exception as e:
            self._image_label.setText(f"图像显示错误: {e}")

    def resizeEvent(self, event):
        """窗口大小变化时重新缩放图像"""
        super().resizeEvent(event)
        if self._image_label.pixmap() is not None:
            pixmap = self._image_label.pixmap()
            scaled = pixmap.scaled(
                self._image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)


# ============================================================================
# 自动化检测面板
# ============================================================================

class InspectionPanel(QWidget):
    """自动化检测面板"""

    # 信号
    start_requested = pyqtSignal()
    """请求开始监听 DI 信号"""
    stop_requested = pyqtSignal()
    """请求停止监听"""
    reset_requested = pyqtSignal()
    """请求复位错误"""
    home_requested = pyqtSignal()
    """请求手动触发完整自动回零（安全顺序：先 Z 轴抬起，再 X/Y 轴）"""
    product_changed = pyqtSignal(str)
    """产品切换信号 (产品名称)"""

    def __init__(self, workflow: InspectionWorkflow, parent=None):
        super().__init__(parent)
        self._workflow = workflow
        self._ng_dialog = None  # NG 确认对话框引用，用于实体按键关闭

        self._setup_ui()
        self._connect_signals()
        self._refresh_product_list()

    def _setup_ui(self):
        """构建 UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(4)

        # ── 顶部控制栏 ──
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #2d2d2d; border: 1px solid #444; border-radius: 3px;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 3, 8, 3)
        top_layout.setSpacing(8)

        # 产品选择
        product_label = QLabel("产品型号:")
        product_label.setStyleSheet("font-size: 13px; color: #d4d4d4; font-weight: bold; border: none;")

        self._product_combo = QComboBox()
        self._product_combo.setMinimumWidth(120)
        self._product_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555;
                padding: 2px 6px; border-radius: 3px; font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4;
                selection-background-color: #1a3a5c;
            }
        """)

        top_layout.addWidget(product_label)
        top_layout.addWidget(self._product_combo)

        self._btn_reload=QPushButton("⟳ 更新产品方案")
        self._btn_reload.setMinimumHeight(28)
        self._btn_reload.setStyleSheet("""
            QPushButton {
                background-color: #1565C8; color: #fff; font-size: 12px;
                font-weight: bold; padding: 2px 8px;
                border: 1px solid #42A5F5; border-radius: 3px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)
        self._btn_reload.setToolTip("重新加载当前产品方案")
        top_layout.addWidget(self._btn_reload)

        # 状态显示
        state_label = QLabel("状态:")
        state_label.setStyleSheet("font-size: 13px; color: #d4d4d4; font-weight: bold; border: none;")

        self._state_display = QLabel("空闲")
        self._state_display.setAlignment(Qt.AlignCenter)
        self._state_display.setMinimumWidth(100)
        self._state_display.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #666;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 2px 8px;
        """)

        top_layout.addWidget(state_label)
        top_layout.addWidget(self._state_display)

        # 最终结果
        result_label = QLabel("结果:")
        result_label.setStyleSheet("font-size: 13px; color: #d4d4d4; font-weight: bold; border: none;")

        self._final_result_label = QLabel("--")
        self._final_result_label.setAlignment(Qt.AlignCenter)
        self._final_result_label.setMinimumWidth(60)
        self._final_result_label.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #666;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 4px; padding: 2px 10px;
        """)

        top_layout.addWidget(result_label)
        top_layout.addWidget(self._final_result_label)

        # 回零状态指示（未回零 / 回零中 / 已回零）
        home_label = QLabel("回零:")
        home_label.setStyleSheet("font-size: 13px; color: #d4d4d4; font-weight: bold; border: none;")

        self._home_state_label = QLabel("未回零")
        self._home_state_label.setAlignment(Qt.AlignCenter)
        self._home_state_label.setMinimumWidth(70)
        self._home_state_label.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #f44336;
            background-color: #2a1a1a; border: 1px solid #C62828;
            border-radius: 3px; padding: 2px 8px;
        """)

        top_layout.addWidget(home_label)
        top_layout.addWidget(self._home_state_label)
        top_layout.addStretch()

        # 控制按钮
        self._btn_start = QPushButton("▶ 启动监听")
        self._btn_start.setMinimumHeight(28)
        self._btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; color: #fff; font-size: 13px;
                font-weight: bold; padding: 2px 10px;
                border: 1px solid #4CAF50; border-radius: 3px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)

        self._btn_stop = QPushButton("⏹ 停止")
        self._btn_stop.setMinimumHeight(28)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #C62828; color: #fff; font-size: 13px;
                font-weight: bold; padding: 2px 10px;
                border: 1px solid #EF5350; border-radius: 3px;
            }
            QPushButton:hover { background-color: #D32F2F; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)

        self._btn_reset = QPushButton("↺ 复位")
        self._btn_reset.setMinimumHeight(28)
        self._btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #E65100; color: #fff; font-size: 13px;
                font-weight: bold; padding: 2px 10px;
                border: 1px solid #FF6D00; border-radius: 3px;
            }
            QPushButton:hover { background-color: #BF360C; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)

        # 手动触发按钮（替代原 DI 触发）
        self._btn_trigger = QPushButton("⚡ 手动触发")
        self._btn_trigger.setMinimumHeight(28)
        self._btn_trigger.setEnabled(False)
        self._btn_trigger.setStyleSheet("""
            QPushButton {
                background-color: #1565C0; color: #fff; font-size: 13px;
                font-weight: bold; padding: 2px 10px;
                border: 1px solid #1976D2; border-radius: 3px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)
        self._btn_trigger.setToolTip("手动触发一次检测流程")

        # 取出确认按钮（OK 流程运动到结束位后，工人按下返回起始位）
        self._btn_takeout = QPushButton("✅ 取出确认")
        self._btn_takeout.setMinimumHeight(28)
        self._btn_takeout.setEnabled(False)
        self._btn_takeout.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; color: #fff; font-size: 13px;
                font-weight: bold; padding: 2px 10px;
                border: 1px solid #4CAF50; border-radius: 3px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)
        self._btn_takeout.setToolTip("运动到结束位后，工人取出板卡并按下此按钮返回起始位")

        top_layout.addWidget(self._btn_start)
        top_layout.addWidget(self._btn_stop)
        top_layout.addWidget(self._btn_reset)
        top_layout.addWidget(self._btn_trigger)
        top_layout.addWidget(self._btn_takeout)

        main_layout.addWidget(top_bar)

        # ── 统计信息栏 ──
        stats_bar = QWidget()
        stats_bar.setStyleSheet("background-color: #252525; border: 1px solid #444; border-radius: 3px;")
        stats_layout = QHBoxLayout(stats_bar)
        stats_layout.setContentsMargins(8, 2, 8, 2)
        stats_layout.setSpacing(12)

        self._trigger_count_label = QLabel("触发: 0")
        self._trigger_count_label.setStyleSheet("font-size: 13px; color: #4fc3f7; font-weight: bold; border: none;")

        self._ok_count_label = QLabel("OK: 0")
        self._ok_count_label.setStyleSheet("font-size: 13px; color: #66BB6A; font-weight: bold; border: none;")

        self._ng_count_label = QLabel("NG: 0")
        self._ng_count_label.setStyleSheet("font-size: 13px; color: #EF5350; font-weight: bold; border: none;")

        # 一次检测总耗时显示
        self._total_elapsed_label = QLabel("耗时: --")
        self._total_elapsed_label.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #CE93D8;
            border: none;
        """)

        self._product_name_label = QLabel("当前产品: 未选择")
        self._product_name_label.setStyleSheet("font-size: 13px; color: #d4d4d4; font-weight: bold; border: none;")

        stats_layout.addWidget(self._trigger_count_label)
        stats_layout.addWidget(self._ok_count_label)
        stats_layout.addWidget(self._ng_count_label)
        stats_layout.addWidget(self._total_elapsed_label)
        stats_layout.addStretch()
        stats_layout.addWidget(self._product_name_label)

        main_layout.addWidget(stats_bar)

        # ── 中间区域: 拼接整图 + 日志 ──
        middle_splitter = QSplitter(Qt.Horizontal)

        # 左侧: 拼接整图显示区
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 6px; background: #2d2d2d; }
            QScrollBar::handle:vertical { background: #555; border-radius: 3px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        # 拼接整图显示控件（可缩放）
        from .widgets.zoomable_label import ZoomableLabel
        self._stitch_label = ZoomableLabel()
        self._stitch_label.setAlignment(Qt.AlignCenter)
        self._stitch_label.setStyleSheet("""
            QLabel {
                background-color: #0d0d0d; border: 1px solid #333;
                border-radius: 4px; color: #555; font-size: 16px;
            }
        """)
        self._stitch_label.setText("等待检测...\n（拼接整图将在此显示）")

        scroll_area.setWidget(self._stitch_label)

        # 右侧: 日志面板
        right_panel = QWidget()
        right_panel.setMinimumWidth(180)
        right_panel.setMaximumWidth(250)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(2)

        log_title = QLabel("执行日志")
        log_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #d4d4d4; padding: 1px 0;")

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #444; border-radius: 3px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 11px;
            }
        """)

        right_layout.addWidget(log_title)
        right_layout.addWidget(self._log_text, 1)

        middle_splitter.addWidget(scroll_area)
        middle_splitter.addWidget(right_panel)
        middle_splitter.setStretchFactor(0, 4)
        middle_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(middle_splitter, 1)

        # ── 连接按钮信号 ──
        self._btn_start.clicked.connect(self._on_start_clicked)
        self._btn_stop.clicked.connect(self._on_stop_clicked)
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        self._btn_trigger.clicked.connect(self._on_trigger_clicked)
        self._btn_takeout.clicked.connect(self._on_takeout_clicked)
        self._product_combo.currentTextChanged.connect(self._on_product_changed)
        self._btn_reload.clicked.connect(self._on_reload_clicked)

    def _connect_signals(self):
        """连接工作流信号"""
        if self._workflow is None:
            return

        self._workflow.state_changed.connect(self._on_state_changed)
        self._workflow.position_result_ready.connect(self._on_position_result)
        self._workflow.all_results_ready.connect(self._on_all_results)
        self._workflow.error_occurred.connect(self._on_error)
        self._workflow.trigger_count_changed.connect(self._on_trigger_count_changed)
        self._workflow.ok_count_changed.connect(self._on_ok_count_changed)
        self._workflow.ng_count_changed.connect(self._on_ng_count_changed)
        self._workflow.ng_confirm_requested.connect(self._on_ng_confirm_requested)
        self._workflow.ng_confirm_closed.connect(self._on_ng_confirm_closed)
        self._workflow.reset_during_confirm.connect(self._on_reset_during_confirm)
        self._workflow.total_elapsed_changed.connect(self._on_total_elapsed_changed)
        self._workflow.barcode_failed.connect(self._on_barcode_failed)
        self._workflow.stitched_image_ready.connect(self._on_stitched_image_ready)
        self._workflow.takeout_confirm_requested.connect(self._on_takeout_confirm_requested)
        self._workflow.motion_state_changed.connect(self._on_motion_state_changed)

    def _refresh_product_list(self):
        """刷新产品列表"""
        self._product_combo.blockSignals(True)
        self._product_combo.clear()
        try:
            products = list_products()
            for name in products:
                self._product_combo.addItem(name)
        except Exception as e:
            log_error(f"加载产品列表失败: {e}")
        self._product_combo.blockSignals(False)

        # 如果有产品，默认选择第一个
        if self._product_combo.count() > 0:
            self._product_combo.setCurrentIndex(0)

    def _rebuild_position_grid(self, positions: List[Dict]):
        """产品切换时重置拼接整图显示（主区域显示拼接整图）。"""
        self._reset_stitch_display()

    def _reset_stitch_display(self):
        """重置拼接整图显示区。"""
        if hasattr(self, '_stitch_label'):
            self._stitch_label.clear_pixmap()
            self._stitch_label.setText("等待检测...\n（拼接整图将在此显示）")

    # ── 按钮事件 ──

    def _on_start_clicked(self):
        """启动按钮点击"""
        if self._workflow is None:
            self._append_log("错误: 工作流未初始化")
            return

        # 检查是否已加载产品
        if self._workflow.product_config is None:
            QMessageBox.warning(self, "提示", "请先选择产品型号")
            return
        self._btn_reload.setEnabled(False)
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_trigger.setEnabled(True)
        self._product_combo.setEnabled(False)
        self._final_result_label.setText("--")
        self._final_result_label.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #666;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 4px; padding: 2px 10px;
        """)

        # 重置拼接整图显示
        self._reset_stitch_display()

        self._append_log("启动自动化检测...")
        self.start_requested.emit()
        self._workflow.start_monitoring()

    def _on_stop_clicked(self):
        """停止按钮点击"""
        if self._workflow:
            self._workflow.stop_monitoring()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_trigger.setEnabled(False)
        self._product_combo.setEnabled(True)
        self._btn_reload.setEnabled(True)
        self._append_log("已停止自动化检测")
        self.stop_requested.emit()

    def _on_trigger_clicked(self):
        """手动触发按钮点击 - 触发一次检测流程"""
        if self._workflow is None:
            self._append_log("错误: 工作流未初始化")
            return
        if self._workflow.product_config is None:
            QMessageBox.warning(self, "提示", "请先选择产品型号")
            return
        self._append_log("手动触发检测...")
        self._workflow.start_inspection()

    def _on_reload_clicked(self):
        """重新加载当前产品方案"""
        product_name = self._product_combo.currentText()
        if not product_name:
            return
        self._on_product_changed(product_name)
        self._append_log(f"已重新加载产品方案: {product_name}")
        
    def _on_reset_clicked(self):
        """复位按钮点击

        复位错误状态，并触发一次完整自动回零（安全顺序：先 Z 轴抬起，再 X/Y 轴）。
        回零过程中的防重复触发 / 急停处理由主窗口的 _start_home_sequence 负责。
        """
        if self._workflow:
            self._workflow.reset_error()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_trigger.setEnabled(False)
        self._product_combo.setEnabled(True)
        self._state_display.setText("空闲")
        self._state_display.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #666;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 2px 8px;
        """)
        self._append_log("已复位，触发自动回零...")
        self.reset_requested.emit()
        # 触发完整自动回零（主窗口处理）
        self.home_requested.emit()

    def _on_product_changed(self, product_name: str):
        """产品切换"""
        if not product_name:
            return

        try:
            config = load_product(product_name)
            if config is None:
                self._append_log(f"加载产品 [{product_name}] 失败")
                return

            # 加载到工作流
            if self._workflow:
                if not self._workflow.load_product(config):
                    self._append_log(f"产品 [{product_name}] 加载失败（工作流可能正在运行）")
                    return

            # 重建位置网格
            positions = config.get("positions", [])
            self._rebuild_position_grid(positions)

            # 更新产品名称显示
            self._product_name_label.setText(f"当前产品: {product_name}")

            self._append_log(f"已加载产品: {product_name} ({len(positions)}个位置)")
            self.product_changed.emit(product_name)

        except Exception as e:
            self._append_log(f"加载产品失败: {e}")
            log_error(f"加载产品 [{product_name}] 失败: {e}")

    # ── 工作流信号回调 ──

    def _on_state_changed(self, state):
        """工作流状态变化"""
        state_names = {
            InspectionWorkflow.State.IDLE: "空闲",
            InspectionWorkflow.State.MONITORING: "等待触发",
            InspectionWorkflow.State.WAITING: "等待工件放稳",
            InspectionWorkflow.State.SCANNING: "扫码中",
            InspectionWorkflow.State.CAPTURING: "拍照中",
            InspectionWorkflow.State.TESTING: "检测中",
            InspectionWorkflow.State.SHOW_RESULT: "显示结果",
            InspectionWorkflow.State.WAITING_FOR_CONFIRM: "等待确认",
            InspectionWorkflow.State.ERROR: "错误",
        }

        name = state_names.get(state, str(state))
        self._state_display.setText(name)

        # 根据状态改变颜色
        if state == InspectionWorkflow.State.ERROR:
            color = "#EF5350"
            bg = "#2a1a1a"
            border = "#C62828"
        elif state == InspectionWorkflow.State.MONITORING:
            color = "#4fc3f7"
            bg = "#1a2a3a"
            border = "#4A90D9"
        elif state in (InspectionWorkflow.State.SCANNING,
                       InspectionWorkflow.State.CAPTURING,
                       InspectionWorkflow.State.TESTING):
            color = "#FFA000"
            bg = "#2a2a1a"
            border = "#FF8F00"
        elif state == InspectionWorkflow.State.SHOW_RESULT:
            color = "#66BB6A"
            bg = "#1a2a1a"
            border = "#4CAF50"
        else:
            color = "#666"
            bg = "#1e1e1e"
            border = "#444"

        self._state_display.setStyleSheet(f"""
            font-size: 14px; font-weight: bold; color: {color};
            background-color: {bg}; border: 1px solid {border};
            border-radius: 3px; padding: 2px 8px;
        """)

        # 更新日志
        self._append_log(f"状态: {name}")

    def _on_position_result(self, index: int, result: PositionResult):
        """单个位置检测完成"""
        self._append_log(f"位置 [{result.name}]: {'OK' if result.passed else 'NG'} - {result.message}")

    def _on_stitched_image_ready(self, stitched_image):
        """拼接整图更新 - 刷新主区域显示。"""
        if stitched_image is None:
            return
        try:
            height, width = stitched_image.shape[:2]
            if len(stitched_image.shape) == 2:
                bytes_per_line = width
                q_img = QImage(stitched_image.data, width, height,
                               bytes_per_line, QImage.Format_Grayscale8)
            else:
                bytes_per_line = 3 * width
                q_img = QImage(stitched_image.data, width, height,
                               bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self._stitch_label.set_pixmap(pixmap)
        except Exception as e:
            self._append_log(f"拼接图显示错误: {e}")

    def _on_takeout_confirm_requested(self):
        """运动到结束位后，请求工人取出确认。"""
        self._btn_takeout.setEnabled(True)
        self._append_log("已运动到结束位，请取出板卡并按下「取出确认」")

    def _on_motion_state_changed(self, desc: str):
        """运动状态变化。"""
        self._append_log(f"运动: {desc}")

    def _on_takeout_clicked(self):
        """取出确认按钮点击 - 通知工作流返回起始位。"""
        if self._workflow is not None:
            self._workflow.confirm_takeout()
        self._btn_takeout.setEnabled(False)
        self._append_log("取出确认，返回起始位")

    def _on_all_results(self, final_ok: bool, results: List[PositionResult]):
        """所有位置检测完成"""
        if final_ok:
            self._final_result_label.setText("OK")
            self._final_result_label.setStyleSheet("""
                font-size: 16px; font-weight: bold; color: #66BB6A;
                background-color: #1a3a1a; border: 1px solid #4CAF50;
                border-radius: 4px; padding: 2px 10px;
            """)
        else:
            self._final_result_label.setText("NG")
            self._final_result_label.setStyleSheet("""
                font-size: 16px; font-weight: bold; color: #EF5350;
                background-color: #2a1a1a; border: 1px solid #C62828;
                border-radius: 4px; padding: 2px 10px;
            """)

        total = len(results)
        ok_count = sum(1 for r in results if r.passed)
        ng_count = total - ok_count
        self._append_log(f"最终结果: {'OK' if final_ok else 'NG'} ({ok_count}/{total} 通过)")

    def _on_ng_confirm_requested(self, results):
        """
        NG 手工确认请求 - 弹出对话框让操作员确认 OK/NG。
        操作员可以查看所有位置的检测结果后做出最终判断。
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QSizePolicy
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap, QImage

        # 构建确认对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("⚠️ NG 检测结果 - 请手工确认")
        dialog.setMinimumSize(600, 400)
        dialog.setStyleSheet("""
            QDialog { background-color: #2d2d2d; }
            QLabel { color: #d4d4d4; font-size: 13px; }
            QPushButton {
                font-size: 15px; font-weight: bold; padding: 8px 24px;
                border-radius: 6px; min-width: 100px;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 8, 12, 8)

        # 标题
        title_label = QLabel("⚠️ 检测结果为 NG，请确认最终判定结果")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 16px; font-weight: bold; color: #EF5350;
            background-color: #2a1a1a; border: 1px solid #C62828;
            border-radius: 6px; padding: 8px;
        """)
        layout.addWidget(title_label)

        # 统计信息
        total = len(results)
        ok_count = sum(1 for r in results if r.passed)
        ng_count = total - ok_count
        stats_label = QLabel(f"总位置: {total}  |  OK: {ok_count}  |  NG: {ng_count}")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_label.setStyleSheet("""
            font-size: 14px; color: #d4d4d4; padding: 4px;
            background-color: #252525; border: 1px solid #444;
            border-radius: 3px;
        """)
        layout.addWidget(stats_label)

        # 滚动区域：显示每个位置的结果详情
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #444; border-radius: 3px;
                          background-color: #1e1e1e; }
            QScrollBar:vertical { width: 6px; background: #2d2d2d; }
            QScrollBar::handle:vertical { background: #555; border-radius: 3px; }
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(4)
        scroll_layout.setContentsMargins(4, 4, 4, 4)

        for i, r in enumerate(results):
            # 每个位置的结果卡片
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {'#1a2a1a' if r.passed else '#2a1a1a'};
                    border: 1px solid {'#2E7D32' if r.passed else '#C62828'};
                    border-radius: 4px;
                }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(2)
            card_layout.setContentsMargins(6, 3, 6, 3)

            # 位置名称 + 结果
            header = QHBoxLayout()
            name_label = QLabel(f"📍 位置 {i + 1}: {r.name}")
            name_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #d4d4d4; border: none;")
            result_text = "OK" if r.passed else "NG"
            result_color = "#66BB6A" if r.passed else "#EF5350"
            result_label = QLabel(result_text)
            result_label.setAlignment(Qt.AlignCenter)
            result_label.setStyleSheet(f"""
                font-size: 14px; font-weight: bold; color: {result_color};
                background-color: #1e1e1e; border: 1px solid {result_color};
                border-radius: 3px; padding: 1px 10px; min-width: 40px;
            """)
            header.addWidget(name_label)
            header.addStretch()
            header.addWidget(result_label)
            card_layout.addLayout(header)

            # 消息
            if r.message:
                msg_label = QLabel(f"消息: {r.message}")
                msg_label.setStyleSheet("font-size: 11px; color: #999; border: none;")
                msg_label.setWordWrap(True)
                card_layout.addWidget(msg_label)

            # 耗时
            time_label = QLabel(f"耗时: {r.elapsed_ms:.1f}ms")
            time_label.setStyleSheet("font-size: 11px; color: #888; border: none;")
            card_layout.addWidget(time_label)

            scroll_layout.addWidget(card)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        btn_ok = QPushButton("✓ 确认为 OK")
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; color: #fff;
                border: 1px solid #4CAF50;
            }
            QPushButton:hover { background-color: #388E3C; }
        """)
        btn_ok.setMinimumHeight(36)

        btn_ng = QPushButton("✗ 确认为 NG")
        btn_ng.setStyleSheet("""
            QPushButton {
                background-color: #C62828; color: #fff;
                border: 1px solid #EF5350;
            }
            QPushButton:hover { background-color: #D32F2F; }
        """)
        btn_ng.setMinimumHeight(36)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_ng)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 连接按钮信号
        def _on_confirm_ok():
            dialog.accept()
            if self._workflow:
                self._workflow.confirm_ng_result(True)

        def _on_confirm_ng():
            dialog.accept()
            if self._workflow:
                self._workflow.confirm_ng_result(False)

        btn_ok.clicked.connect(_on_confirm_ok)
        btn_ng.clicked.connect(_on_confirm_ng)

        # 保存对话框引用，以便实体按键（D1/D2）或复位（D3）信号能关闭它
        self._ng_dialog = dialog
        self._append_log("⚠️ NG 检测结果，等待手工确认...")
        dialog.exec_()
        self._ng_dialog = None  # 对话框关闭后清除引用

    def _on_ng_confirm_closed(self):
        """NG 确认完成 - 由工作流在 D1/D2 实体按键确认后发射，关闭弹窗"""
        if self._ng_dialog is not None:
            self._ng_dialog.accept()
            self._ng_dialog = None
            self._append_log("实体按键确认，关闭 NG 确认弹窗")

    def _on_reset_during_confirm(self):
        """D3 复位时关闭 NG 确认弹窗"""
        if self._ng_dialog is not None:
            self._ng_dialog.reject()  # 使用 reject() 而非 accept()，表示非正常关闭
            self._ng_dialog = None
            self._append_log("D3 复位，关闭 NG 确认弹窗")

    def _on_error(self, error_msg: str):
        """错误发生"""
        self._append_log(f"错误: {error_msg}")
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_trigger.setEnabled(False)

    def _on_trigger_count_changed(self, count: int):
        """触发次数更新"""
        self._trigger_count_label.setText(f"触发: {count}")

    def _on_ok_count_changed(self, count: int):
        """OK次数更新"""
        self._ok_count_label.setText(f"OK: {count}")

    def _on_ng_count_changed(self, count: int):
        """NG次数更新"""
        self._ng_count_label.setText(f"NG: {count}")

    def _on_total_elapsed_changed(self, elapsed_seconds: float):
        """一次检测总耗时更新"""
        if elapsed_seconds < 60:
            self._total_elapsed_label.setText(f"耗时: {elapsed_seconds:.2f}s")
        else:
            minutes = int(elapsed_seconds // 60)
            seconds = elapsed_seconds % 60
            self._total_elapsed_label.setText(f"耗时: {minutes}m {seconds:.1f}s")

    def _on_barcode_failed(self):
        """扫码失败 - 弹出提示弹窗，通知操作员重新放入工件"""
        self._append_log("⚠️ 扫码失败，请重新放入工件")
        msg = QMessageBox(self)
        msg.setWindowTitle("扫码失败")
        msg.setText("⚠️ 未扫描到有效一维码\n\n请重新放入工件，确保条码对准扫描头")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.button(QMessageBox.Ok).setText("确定")
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #d4d4d4;
                font-size: 15px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #d4d4d4;
                padding: 6px 24px;
                border: 1px solid #555;
                border-radius: 3px;
                font-size: 14px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #4A90D9;
            }
        """)
        msg.exec_()

    # ── 日志 ──

    def _append_log(self, message: str):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{timestamp}] {message}")
        # 自动滚动到底部
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ── 外部接口 ──

    def set_workflow(self, workflow: InspectionWorkflow):
        """设置工作流实例"""
        self._workflow = workflow
        self._connect_signals()

    def refresh_products(self):
        """刷新产品列表（外部调用，如 Engineer 模式新增产品后）"""
        current = self._product_combo.currentText()
        self._refresh_product_list()
        # 尝试恢复之前的选择
        index = self._product_combo.findText(current)
        if index >= 0:
            self._product_combo.setCurrentIndex(index)

    def get_current_product(self) -> str:
        """获取当前选择的产品名称"""
        return self._product_combo.currentText()

    def set_home_state(self, state: str):
        """更新回零状态指示（未回零 / 回零中 / 已回零）。

        由主窗口在回零状态变化时调用。

        Args:
            state: "未回零" / "回零中" / "已回零"
        """
        if not hasattr(self, '_home_state_label'):
            return
        self._home_state_label.setText(state)
        if state == "已回零":
            style = """
                font-size: 13px; font-weight: bold; color: #66BB6A;
                background-color: #1a3a1a; border: 1px solid #4CAF50;
                border-radius: 3px; padding: 2px 8px;
            """
        elif state == "回零中":
            style = """
                font-size: 13px; font-weight: bold; color: #FFA000;
                background-color: #2a2a1a; border: 1px solid #FF8F00;
                border-radius: 3px; padding: 2px 8px;
            """
        else:  # 未回零
            style = """
                font-size: 13px; font-weight: bold; color: #f44336;
                background-color: #2a1a1a; border: 1px solid #C62828;
                border-radius: 3px; padding: 2px 8px;
            """
        self._home_state_label.setStyleSheet(style)
