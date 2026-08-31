# -*- coding: utf-8 -*-
"""自动测试管理器。

自动循环触发检测流程 N 次，自动处理 OK/NG 确认，统计结果并保存到文件。
用于可靠性测试：验证检测流程在连续运行下的稳定性。
"""

import os
import time
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from core.log_manager import log_info, log_error


class AutoTestManager(QObject):
    """自动测试管理器。

    信号:
        progress_changed(int, int): 当前完成次数, 总次数
        result_updated(int, int): OK 数, NG 数
        finished(int, int): 最终 OK 数, NG 数
        log_message(str): 日志消息
    """

    progress_changed = pyqtSignal(int, int)
    result_updated = pyqtSignal(int, int)
    finished = pyqtSignal(int, int)
    log_message = pyqtSignal(str)

    def __init__(self, workflow, total: int = 10, interval_ms: int = 2000,
                 save_path: str = "data/auto_test_results.csv", parent=None):
        super().__init__(parent)
        self._workflow = workflow
        self._total = max(1, int(total))
        self._interval_ms = max(100, int(interval_ms))
        self._save_path = save_path

        self._running = False
        self._current = 0
        self._ok_count = 0
        self._ng_count = 0
        self._waiting_confirm = False

        # 间隔定时器（一次检测完成后，间隔后触发下一次）
        self._interval_timer = QTimer(self)
        self._interval_timer.setSingleShot(True)
        self._interval_timer.timeout.connect(self._trigger_next)

        # 连接工作流信号
        if self._workflow is not None:
            self._workflow.all_results_ready.connect(self._on_all_results)
            self._workflow.state_changed.connect(self._on_state_changed)

    # ── 公共接口 ──

    def start(self):
        """开始自动测试。"""
        if self._running:
            return
        if self._workflow is None:
            self._emit_log("工作流未初始化，无法开始自动测试")
            return
        if not self._workflow.is_running:
            self._emit_log("工作流未在监听状态，请先启动监听")
            return

        self._running = True
        self._current = 0
        self._ok_count = 0
        self._ng_count = 0
        self._waiting_confirm = False
        # 启用工作流自动确认模式（NG 直接判 NG，不弹窗）
        if hasattr(self._workflow, "set_auto_confirm"):
            self._workflow.set_auto_confirm(True)
        self._emit_log(f"自动测试开始：共 {self._total} 次，间隔 {self._interval_ms}ms")
        self.progress_changed.emit(0, self._total)
        self.result_updated.emit(0, 0)

        # 触发第一次
        self._trigger_next()

    def stop(self):
        """停止自动测试。"""
        if not self._running:
            return
        self._running = False
        self._interval_timer.stop()
        # 关闭工作流自动确认模式
        if hasattr(self._workflow, "set_auto_confirm"):
            self._workflow.set_auto_confirm(False)
        self._emit_log(f"自动测试已停止（已完成 {self._current}/{self._total} 次）")
        self._save_results()
        self.finished.emit(self._ok_count, self._ng_count)

    def is_running(self) -> bool:
        return self._running

    # ── 内部逻辑 ──

    def _trigger_next(self):
        """触发下一次检测。"""
        if not self._running:
            return
        if self._current >= self._total:
            self._finish()
            return

        # 检查工作流状态是否允许触发
        state = self._workflow.state
        from core.inspection_workflow import InspectionWorkflow
        if state not in (InspectionWorkflow.State.MONITORING,
                         InspectionWorkflow.State.IDLE):
            self._emit_log(f"工作流状态 {state.value} 不允许触发，等待...")
            # 稍后重试
            self._interval_timer.start(500)
            return

        self._current += 1
        self._waiting_confirm = False
        self._emit_log(f"── 自动测试第 {self._current}/{self._total} 次 ──")
        self.progress_changed.emit(self._current, self._total)
        try:
            self._workflow.start_inspection()
        except Exception as e:  # noqa: BLE001
            self._emit_log(f"触发检测异常: {e}")
            self._interval_timer.start(self._interval_ms)

    def _on_all_results(self, ok: bool, results):
        """一次检测完成。"""
        if not self._running:
            return
        if ok:
            self._ok_count += 1
            self._emit_log(f"第 {self._current} 次: OK")
        else:
            self._ng_count += 1
            self._emit_log(f"第 {self._current} 次: NG")
        self.result_updated.emit(self._ok_count, self._ng_count)

        # 自动确认，使工作流回到可再次触发状态
        self._waiting_confirm = True
        try:
            if ok:
                # OK：若等待取出确认，自动确认取出
                if hasattr(self._workflow, "_takeout_pending") and self._workflow._takeout_pending:
                    self._workflow.confirm_takeout()
                else:
                    # 无轴运动时，工作流已回到 MONITORING，直接安排下一次
                    self._schedule_next()
            else:
                # NG：工作流在自动确认模式下已直接判 NG 并返回起始位，
                # 等待状态回到 MONITORING 后触发下一次（由 _on_state_changed 处理）
                pass
        except Exception as e:  # noqa: BLE001
            self._emit_log(f"自动确认异常: {e}")
            self._schedule_next()

    def _on_state_changed(self, state):
        """工作流状态变化。"""
        if not self._running:
            return
        from core.inspection_workflow import InspectionWorkflow
        # 回到 MONITORING 状态，说明一次检测已完全结束，可触发下一次
        if state == InspectionWorkflow.State.MONITORING and self._waiting_confirm:
            self._waiting_confirm = False
            self._schedule_next()

    def _schedule_next(self):
        """安排下一次检测（间隔后触发）。"""
        if not self._running:
            return
        if self._current >= self._total:
            self._finish()
            return
        self._interval_timer.start(self._interval_ms)

    def _finish(self):
        """自动测试完成。"""
        self._running = False
        # 关闭工作流自动确认模式
        if hasattr(self._workflow, "set_auto_confirm"):
            self._workflow.set_auto_confirm(False)
        self._emit_log(f"自动测试完成：OK={self._ok_count}, NG={self._ng_count}, "
                       f"共 {self._current} 次")
        self._save_results()
        self.finished.emit(self._ok_count, self._ng_count)

    def _save_results(self):
        """保存测试结果到 CSV 文件。"""
        try:
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = "时间,总次数,OK数,NG数\n"
            row = f"{now},{self._current},{self._ok_count},{self._ng_count}\n"
            # 若文件不存在则写表头
            if not os.path.exists(self._save_path):
                with open(self._save_path, "w", encoding="utf-8") as f:
                    f.write(header)
            with open(self._save_path, "a", encoding="utf-8") as f:
                f.write(row)
            self._emit_log(f"测试结果已保存: {self._save_path}")
        except Exception as e:  # noqa: BLE001
            log_error(f"保存测试结果失败: {e}")

    def _emit_log(self, msg: str):
        log_info(msg)
        self.log_message.emit(msg)
