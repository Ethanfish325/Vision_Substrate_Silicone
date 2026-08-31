# -*- coding: utf-8 -*-
"""自动测试设置对话框。

设置自动测试次数与间隔，显示测试进度与 OK/NG 统计。
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QPushButton, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt


class AutoTestDialog(QDialog):
    """自动测试对话框。"""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self.setWindowTitle("自动测试")
        self.setMinimumSize(480, 420)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 参数设置
        param_group = QGroupBox("测试参数")
        param_layout = QVBoxLayout(param_group)

        # 次数
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("测试次数:"))
        self._spin_count = QSpinBox()
        self._spin_count.setRange(1, 10000)
        self._spin_count.setValue(10)
        count_row.addWidget(self._spin_count, 1)
        param_layout.addLayout(count_row)

        # 间隔
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("间隔(ms):"))
        self._spin_interval = QSpinBox()
        self._spin_interval.setRange(100, 60000)
        self._spin_interval.setValue(2000)
        self._spin_interval.setSingleStep(100)
        interval_row.addWidget(self._spin_interval, 1)
        param_layout.addLayout(interval_row)

        layout.addWidget(param_group)

        # 进度与统计
        stats_group = QGroupBox("测试进度")
        stats_layout = QVBoxLayout(stats_group)

        self._progress_label = QLabel("进度: 0 / 0")
        self._progress_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4fc3f7;")
        stats_layout.addWidget(self._progress_label)

        self._result_label = QLabel("OK: 0    NG: 0")
        self._result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #8bc34a;")
        stats_layout.addWidget(self._result_label)

        layout.addWidget(stats_group)

        # 日志
        log_group = QGroupBox("测试日志")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        log_layout.addWidget(self._log_view)
        layout.addWidget(log_group, 1)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_start = QPushButton("▶ 开始")
        self._btn_start.setStyleSheet("""
            QPushButton { background-color: #1976D2; color: #fff; padding: 6px 20px;
                         border: none; border-radius: 3px; font-weight: bold; }
            QPushButton:hover { background-color: #1565C0; }
        """)
        self._btn_stop = QPushButton("■ 停止")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet("""
            QPushButton { background-color: #5c1a1a; color: #D94A4A; padding: 6px 20px;
                         border: 1px solid #8c2a2a; border-radius: 3px; font-weight: bold; }
        """)
        self._btn_close = QPushButton("关闭")
        self._btn_close.setStyleSheet("""
            QPushButton { background-color: #3c3c3c; color: #d4d4d4; padding: 6px 20px;
                         border: 1px solid #555; border-radius: 3px; }
        """)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_close.clicked.connect(self.close)

    def _connect_signals(self):
        if self._manager is None:
            return
        self._manager.progress_changed.connect(self._on_progress)
        self._manager.result_updated.connect(self._on_result)
        self._manager.finished.connect(self._on_finished)
        self._manager.log_message.connect(self._append_log)

    def _on_start(self):
        if self._manager is None:
            return
        total = self._spin_count.value()
        interval = self._spin_interval.value()
        # 更新管理器参数
        self._manager._total = total
        self._manager._interval_ms = interval
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._spin_count.setEnabled(False)
        self._spin_interval.setEnabled(False)
        self._manager.start()

    def _on_stop(self):
        if self._manager is not None:
            self._manager.stop()

    def _on_progress(self, current, total):
        self._progress_label.setText(f"进度: {current} / {total}")

    def _on_result(self, ok, ng):
        self._result_label.setText(f"OK: {ok}    NG: {ng}")

    def _on_finished(self, ok, ng):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._spin_count.setEnabled(True)
        self._spin_interval.setEnabled(True)
        self._append_log(f"══════ 自动测试完成: OK={ok}, NG={ng} ══════")

    def _append_log(self, msg: str):
        self._log_view.append(msg)
