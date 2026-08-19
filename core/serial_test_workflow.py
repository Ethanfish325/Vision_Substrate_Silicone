# -*- coding: utf-8 -*-
"""
串口自动测试工作流模块
=====================
实现由串口数据触发的自动化测试工作流。

工作流状态机:
    IDLE -> WAITING_TRIGGER -> CAPTURING -> TESTING -> SENDING_RESULT -> WAITING_TRIGGER

设计模式:
    - 状态机: 管理工作流的生命周期，防止重复触发
    - 策略模式: TriggerParser（触发解析）、ResultSender（结果发送）均可扩展

使用方式:
    workflow = SerialTestWorkflow(comm_mgr, config)
    workflow.state_changed.connect(on_state_changed)
    workflow.capture_requested.connect(on_capture)
    workflow.test_requested.connect(on_test)
    workflow.start()
    # ... 在拍照完成后调用 workflow.on_capture_completed(image)
    # ... 在检测完成后调用 workflow.on_test_completed(passed, message)
    workflow.stop()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

from core.serial_comm import SerialCommManager
from core.log_manager import log_info, log_error, log_warning


# ──────────────────────────────────────────────
# 策略模式：触发数据解析器
# ──────────────────────────────────────────────

class TriggerParser(ABC):
    """触发数据解析器基类 - 策略模式"""

    @abstractmethod
    def parse(self, data: bytes) -> Optional[dict]:
        """
        解析串口数据，判断是否为有效的触发信号。

        Args:
            data: 收到的原始字节数据

        Returns:
            Optional[dict]: 如果匹配触发条件，返回解析结果字典；
                           如果不匹配，返回 None
        """
        pass


class AnyDataTriggerParser(TriggerParser):
    """任意数据触发 - 收到任何非空数据即触发"""

    def parse(self, data: bytes) -> Optional[dict]:
        if data and len(data) > 0:
            return {"trigger": True, "raw": data}
        return None


# 预留：后续可在此添加特定协议解析器
# class ModbusTriggerParser(TriggerParser): ...
# class CustomProtocolParser(TriggerParser): ...


# ──────────────────────────────────────────────
# 策略模式：结果发送器
# ──────────────────────────────────────────────

class ResultSender(ABC):
    """结果发送器基类 - 策略模式"""

    @abstractmethod
    def build_result(self, passed: bool) -> bytes:
        """
        根据检测结果构建要发送的字节数据。

        Args:
            passed: 检测是否通过

        Returns:
            bytes: 要发送的字节数据
        """
        pass


class SimpleTextResultSender(ResultSender):
    """简单文本结果 - 发送 'OK(P)\\n' 或 'NG(F)\\n'"""

    def build_result(self, passed: bool) -> bytes:
        return b"P\n" if passed else b"F\n"


class HexResultSender(ResultSender):
    """HEX 格式结果 - 发送 0x01(OK) 或 0x02(NG)"""

    def build_result(self, passed: bool) -> bytes:
        return b"\x01" if passed else b"\x02"


# 预留：后续可在此添加自定义格式发送器


# ──────────────────────────────────────────────
# 工作流配置
# ──────────────────────────────────────────────

@dataclass
class WorkflowConfig:
    """串口自动测试工作流配置"""

    # 触发解析器
    trigger_parser: TriggerParser = field(default_factory=AnyDataTriggerParser)

    # 结果发送器
    result_sender: ResultSender = field(default_factory=SimpleTextResultSender)

    # 拍照最大重试次数
    max_capture_retries: int = 3

    # 发送结果最大重试次数
    max_send_retries: int = 3


# ──────────────────────────────────────────────
# 工作流管理器
# ──────────────────────────────────────────────

class SerialTestWorkflow(QObject):
    """
    串口触发测试工作流管理器。

    工作流状态机:
        IDLE -> WAITING_TRIGGER -> CAPTURING -> TESTING -> SENDING_RESULT -> WAITING_TRIGGER

    信号:
        state_changed(State): 状态变化时发射
        capture_requested(): 请求拍照（UI 层需响应此信号并调用 on_capture_completed）
        test_requested(np.ndarray): 请求检测（传递图像，UI 层需响应并调用 on_test_completed）
        error_occurred(str): 发生错误时发射
    """

    class State(Enum):
        IDLE = "空闲"
        WAITING_TRIGGER = "等待触发"
        CAPTURING = "拍照中"
        TESTING = "检测中"
        SENDING_RESULT = "发送结果"

    # ── 信号定义 ──

    state_changed = pyqtSignal(object)  # State 枚举
    capture_requested = pyqtSignal()
    test_requested = pyqtSignal(object)  # np.ndarray
    error_occurred = pyqtSignal(str)

    # ── 统计信息信号 ──

    trigger_count_changed = pyqtSignal(int)
    ok_count_changed = pyqtSignal(int)
    ng_count_changed = pyqtSignal(int)

    def __init__(self, comm_mgr: SerialCommManager,
                 config: Optional[WorkflowConfig] = None,
                 parent=None):
        """
        初始化工作流管理器。

        Args:
            comm_mgr: SerialCommManager 实例，用于发送结果
            config: 工作流配置，为 None 时使用默认配置
            parent: QObject 父对象
        """
        super().__init__(parent)
        self._comm_mgr = comm_mgr
        self._config = config or WorkflowConfig()

        # 状态
        self._state = self.State.IDLE
        self._running = False

        # 重试计数
        self._capture_retries = 0
        self._send_retries = 0

        # 统计
        self._trigger_count = 0
        self._ok_count = 0
        self._ng_count = 0

        # 当前触发数据（保留原始数据以备扩展）
        self._current_trigger_data: Optional[bytes] = None

        # 连接串口数据接收信号
        self._comm_mgr.data_received.connect(self._on_serial_data_received)

    # ── 属性 ──

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def config(self) -> WorkflowConfig:
        return self._config

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def ok_count(self) -> int:
        return self._ok_count

    @property
    def ng_count(self) -> int:
        return self._ng_count

    # ── 状态管理 ──

    def _set_state(self, new_state: State):
        """安全地切换状态并发射信号。"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            log_info(f"自动测试工作流: {old_state.value} -> {new_state.value}")
            self.state_changed.emit(new_state)

    # ── 生命周期控制 ──

    def start(self):
        """启动工作流，进入等待触发状态。"""
        if self._running:
            log_warning("自动测试工作流已在运行中")
            return

        if not self._comm_mgr.is_open:
            self.error_occurred.emit("串口未打开，无法启动自动测试")
            return

        self._running = True
        self._capture_retries = 0
        self._send_retries = 0
        self._trigger_count = 0
        self._ok_count = 0
        self._ng_count = 0

        self._set_state(self.State.WAITING_TRIGGER)
        log_info("自动测试工作流已启动")

    def stop(self):
        """停止工作流，回到空闲状态。"""
        if not self._running:
            return

        self._running = False
        self._capture_retries = 0
        self._send_retries = 0
        self._current_trigger_data = None

        self._set_state(self.State.IDLE)
        log_info("自动测试工作流已停止")

    # ── 串口数据接收 ──

    def _on_serial_data_received(self, data: bytes):
        """串口收到数据时的回调。"""
        if not self._running:
            return

        # 仅在等待触发状态处理触发信号
        if self._state != self.State.WAITING_TRIGGER:
            return

        # 使用 TriggerParser 解析数据
        result = self._config.trigger_parser.parse(data)
        if result is None:
            # 数据不匹配触发条件，忽略
            return

        # 匹配成功，开始拍照测试流程
        self._current_trigger_data = data
        self._trigger_count += 1
        self.trigger_count_changed.emit(self._trigger_count)

        log_info(f"收到触发信号 (第{self._trigger_count}次): {data.hex()}")
        self._start_capture()

    # ── 拍照流程 ──

    def _start_capture(self):
        """开始拍照流程。"""
        self._set_state(self.State.CAPTURING)
        self._capture_retries = 0
        self.capture_requested.emit()

    def on_capture_completed(self, image: Optional[np.ndarray]):
        """
        拍照完成回调 - 由 UI 层在拍照完成后调用。

        Args:
            image: 拍照得到的图像，为 None 表示拍照失败
        """
        if self._state != self.State.CAPTURING:
            log_warning(f"工作流状态不是 CAPTURING，忽略拍照完成回调 (当前: {self._state.value})")
            return

        if image is None:
            # 拍照失败，重试
            self._capture_retries += 1
            if self._capture_retries < self._config.max_capture_retries:
                log_warning(f"拍照失败，第{self._capture_retries}次重试...")
                self.capture_requested.emit()
                return
            else:
                log_error(f"拍照失败，已重试{self._config.max_capture_retries}次，放弃")
                self.error_occurred.emit(f"拍照失败（已重试{self._config.max_capture_retries}次）")
                self._set_state(self.State.WAITING_TRIGGER)
                return

        # 拍照成功，进入检测流程
        self._capture_retries = 0
        self._start_test(image)

    # ── 检测流程 ──

    def _start_test(self, image: np.ndarray):
        """开始检测流程。"""
        self._set_state(self.State.TESTING)
        self.test_requested.emit(image)

    def on_test_completed(self, passed: bool, message: str):
        """
        检测完成回调 - 由 UI 层在检测完成后调用。

        Args:
            passed: 检测是否通过
            message: 检测结果消息
        """
        if self._state != self.State.TESTING:
            log_warning(f"工作流状态不是 TESTING，忽略检测完成回调 (当前: {self._state.value})")
            return

        # 更新统计
        if passed:
            self._ok_count += 1
            self.ok_count_changed.emit(self._ok_count)
        else:
            self._ng_count += 1
            self.ng_count_changed.emit(self._ng_count)

        log_info(f"检测完成: {'OK' if passed else 'NG'} | {message}")

        # 进入发送结果流程
        self._send_result(passed)

    # ── 发送结果流程 ──

    def _send_result(self, passed: bool):
        """发送检测结果到下位机。"""
        self._set_state(self.State.SENDING_RESULT)
        self._send_retries = 0

        # 使用 ResultSender 构建结果数据
        result_bytes = self._config.result_sender.build_result(passed)

        # 发送
        count = self._comm_mgr.send(result_bytes)
        if count > 0:
            log_info(f"结果已发送: {result_bytes.hex()} ({count} 字节)")
            self._on_result_sent(True)
        else:
            # 发送失败，重试
            self._send_retries += 1
            if self._send_retries < self._config.max_send_retries:
                log_warning(f"发送结果失败，第{self._send_retries}次重试...")
                count = self._comm_mgr.send(result_bytes)
                if count > 0:
                    self._on_result_sent(True)
                    return
            log_error(f"发送结果失败，已重试{self._config.max_send_retries}次，放弃")
            self.error_occurred.emit(f"发送结果失败（已重试{self._config.max_send_retries}次）")
            self._on_result_sent(False)

    def _on_result_sent(self, success: bool):
        """结果发送完成后的处理。"""
        if success:
            log_info("结果发送成功")
        else:
            log_warning("结果发送失败")

        # 回到等待触发状态
        self._current_trigger_data = None
        self._send_retries = 0
        self._set_state(self.State.WAITING_TRIGGER)

    # ── 资源清理 ──

    def cleanup(self):
        """清理资源。"""
        self.stop()
        try:
            self._comm_mgr.data_received.disconnect(self._on_serial_data_received)
        except (TypeError, RuntimeError):
            pass
