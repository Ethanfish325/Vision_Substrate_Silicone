# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class ResultPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("检测结果")
        title.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #d4d4d4;
            padding: 3px 8px; background-color: #1e1e1e;
            border-bottom: 1px solid #444;
        """)
        title.setFixedHeight(24)

        self.status_indicator = QLabel("等待检测...")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setMinimumHeight(40)
        self.status_indicator.setStyleSheet("""
            font-size: 20px; font-weight: bold; color: #666;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 4px; padding: 4px;
        """)

        # 总测试时间显示
        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #4fc3f7;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 2px 6px;
        """)
        self.time_label.setFixedHeight(22)

        layout.addWidget(title)
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.time_label)

    def show_result(self, passed, message, annotated_image=None, tool_results=None):
        if passed:
            self.status_indicator.setText("✓ OK")
            self.status_indicator.setStyleSheet("""
                font-size: 24px; font-weight: bold; color: #66BB6A;
                background-color: #1a3a1a; border: 2px solid #4CAF50;
                border-radius: 4px; padding: 4px;
            """)
        else:
            self.status_indicator.setText("✗ NG")
            self.status_indicator.setStyleSheet("""
                font-size: 24px; font-weight: bold; color: #EF5350;
                background-color: #3a1a1a; border: 2px solid #EF5350;
                border-radius: 4px; padding: 4px;
            """)

        # 显示总测试时间
        if tool_results and "total_elapsed_ms" in tool_results:
            total_ms = tool_results["total_elapsed_ms"]
            self.time_label.setText(f"⏱ 总耗时: {total_ms:.0f}ms")
        else:
            self.time_label.setText("")

    def clear(self):
        self.status_indicator.setText("等待检测...")
        self.status_indicator.setStyleSheet("""
            font-size: 20px; font-weight: bold; color: #666;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 4px; padding: 4px;
        """)
        self.time_label.setText("")
