# -*- coding: utf-8 -*-
"""
串口通信核心模块
================
封装串口通信的所有底层操作，提供清晰的 API 供 UI 层调用。

功能:
    - list_ports() - 扫描系统可用串口
    - SerialCommManager - 串口通信管理器类（打开/关闭/发送/接收/异步读取）
"""

from typing import List, Optional, Dict, Any, Callable
from enum import Enum

import serial
import serial.tools.list_ports

from PyQt5.QtCore import QObject, QThread, pyqtSignal, QMutex, QMutexLocker


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def list_ports() -> List[Dict[str, Any]]:
    """
    扫描系统所有可用串口。

    Returns:
        List[Dict]: 每个元素包含:
            - device (str): 端口名，如 "COM1"
            - description (str): 端口描述
            - hwid (str): 硬件 ID
            - vid (int): 供应商 ID（可能为 None）
            - pid (int): 产品 ID（可能为 None）
            - serial_number (str): 序列号（可能为 None）
    """
    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append({
            "device": port.device,
            "description": port.description,
            "hwid": port.hwid,
            "vid": port.vid,
            "pid": port.pid,
            "serial_number": port.serial_number,
        })
    return ports


# ──────────────────────────────────────────────
# 常量定义
# ──────────────────────────────────────────────

BAUDRATES = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
DEFAULT_BAUDRATE = 9600

DATA_BITS = {
    5: serial.FIVEBITS,
    6: serial.SIXBITS,
    7: serial.SEVENBITS,
    8: serial.EIGHTBITS,
}
DEFAULT_DATA_BITS = 8

PARITY_MAP = {
    "None": serial.PARITY_NONE,
    "Even": serial.PARITY_EVEN,
    "Odd": serial.PARITY_ODD,
    "Mark": serial.PARITY_MARK,
    "Space": serial.PARITY_SPACE,
}
PARITY_NAMES = list(PARITY_MAP.keys())
DEFAULT_PARITY = "None"

STOP_BITS_MAP = {
    1: serial.STOPBITS_ONE,
    1.5: serial.STOPBITS_ONE_POINT_FIVE,
    2: serial.STOPBITS_TWO,
}
STOP_BITS_VALUES = [1, 1.5, 2]
DEFAULT_STOP_BITS = 1

FLOW_CONTROL_NAMES = ["None", "RTS/CTS", "XON/XOFF"]
DEFAULT_FLOW_CONTROL = "None"


# ──────────────────────────────────────────────
# 异步读取线程
# ──────────────────────────────────────────────

class SerialReaderThread(QThread):
    """后台串口读取线程，持续读取数据并通过信号发射。"""

    data_received = pyqtSignal(bytes)
    error_occurred = pyqtSignal(str)

    def __init__(self, serial_obj: serial.Serial, parent=None):
        super().__init__(parent)
        self._serial = serial_obj
        self._running = False
        self._mutex = QMutex()

    def run(self):
        self._running = True
        while self._running:
            try:
                if self._serial and self._serial.is_open:
                    # 等待数据到达（超时由 serial 的 timeout 参数控制）
                    if self._serial.in_waiting > 0:
                        data = self._serial.read(self._serial.in_waiting)
                        if data:
                            self.data_received.emit(data)
                    else:
                        # 没有数据时短暂休眠，避免 busy wait
                        self.msleep(10)
                else:
                    self.msleep(50)
            except serial.SerialException as e:
                if self._running:
                    self.error_occurred.emit(f"读取错误: {str(e)}")
                break
            except Exception as e:
                if self._running:
                    self.error_occurred.emit(f"读取异常: {str(e)}")
                break

    def stop(self):
        self._running = False
        self.wait(2000)

    @property
    def is_running(self) -> bool:
        return self._running


# ──────────────────────────────────────────────
# 串口通信管理器
# ──────────────────────────────────────────────

class SerialCommManager(QObject):
    """
    串口通信管理器。

    封装串口的完整生命周期管理，包括打开、关闭、发送、接收，
    以及通过后台线程异步读取数据。

    信号:
        data_received(bytes): 接收到数据时发射
        connection_changed(bool): 连接状态变化时发射（True=已连接）
        error_occurred(str): 发生错误时发射
        rx_count_changed(int): 接收字节数更新时发射
        tx_count_changed(int): 发送字节数更新时发射
    """

    data_received = pyqtSignal(bytes)
    connection_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    rx_count_changed = pyqtSignal(int)
    tx_count_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._serial: Optional[serial.Serial] = None
        self._reader: Optional[SerialReaderThread] = None
        self._rx_count = 0
        self._tx_count = 0

        # 当前配置
        self._port: str = ""
        self._baudrate: int = DEFAULT_BAUDRATE
        self._bytesize: int = DEFAULT_DATA_BITS
        self._parity: str = DEFAULT_PARITY
        self._stopbits: float = DEFAULT_STOP_BITS
        self._timeout: float = 0.1
        self._flow_control: str = DEFAULT_FLOW_CONTROL

    # ── 属性 ──

    @property
    def is_open(self) -> bool:
        """串口是否已打开。"""
        return self._serial is not None and self._serial.is_open

    @property
    def port(self) -> str:
        return self._port

    @property
    def baudrate(self) -> int:
        return self._baudrate

    @property
    def rx_count(self) -> int:
        return self._rx_count

    @property
    def tx_count(self) -> int:
        return self._tx_count

    @property
    def settings(self) -> Dict[str, Any]:
        """获取当前串口参数配置。"""
        return {
            "port": self._port,
            "baudrate": self._baudrate,
            "bytesize": self._bytesize,
            "parity": self._parity,
            "stopbits": self._stopbits,
            "timeout": self._timeout,
            "flow_control": self._flow_control,
        }

    # ── 配置 ──

    def set_config(self, port: str, baudrate: int = DEFAULT_BAUDRATE,
                   bytesize: int = DEFAULT_DATA_BITS,
                   parity: str = DEFAULT_PARITY,
                   stopbits: float = DEFAULT_STOP_BITS,
                   timeout: float = 0.1,
                   flow_control: str = DEFAULT_FLOW_CONTROL):
        """
        设置串口参数（在 open() 之前调用）。

        Args:
            port: 端口名，如 "COM1"
            baudrate: 波特率
            bytesize: 数据位（5/6/7/8）
            parity: 校验位（None/Even/Odd/Mark/Space）
            stopbits: 停止位（1/1.5/2）
            timeout: 读取超时（秒）
            flow_control: 流控制（None/RTS/CTS/XON/XOFF）
        """
        self._port = port
        self._baudrate = baudrate
        self._bytesize = bytesize
        self._parity = parity
        self._stopbits = stopbits
        self._timeout = timeout
        self._flow_control = flow_control

    def load_config(self, config: Dict[str, Any]):
        """从字典加载配置。"""
        self._port = config.get("port", self._port)
        self._baudrate = config.get("baudrate", self._baudrate)
        self._bytesize = config.get("bytesize", self._bytesize)
        self._parity = config.get("parity", self._parity)
        self._stopbits = config.get("stopbits", self._stopbits)
        self._timeout = config.get("timeout", self._timeout)
        self._flow_control = config.get("flow_control", self._flow_control)

    # ── 连接管理 ──

    def open(self) -> bool:
        """
        打开串口连接。

        Returns:
            bool: 是否成功打开
        """
        if self.is_open:
            self.error_occurred.emit("串口已打开")
            return False

        if not self._port:
            self.error_occurred.emit("未指定串口端口")
            return False

        try:
            # 构建串口参数
            parity = PARITY_MAP.get(self._parity, serial.PARITY_NONE)
            stopbits = STOP_BITS_MAP.get(self._stopbits, serial.STOPBITS_ONE)
            bytesize = DATA_BITS.get(self._bytesize, serial.EIGHTBITS)

            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=self._timeout,
                write_timeout=1.0,
            )

            # 配置流控制
            if self._flow_control == "RTS/CTS":
                self._serial.rtscts = True
            elif self._flow_control == "XON/XOFF":
                self._serial.xonxoff = True

            # 重置计数器
            self._rx_count = 0
            self._tx_count = 0
            self.rx_count_changed.emit(0)
            self.tx_count_changed.emit(0)

            # 启动异步读取线程
            self._start_reader()

            self.connection_changed.emit(True)
            return True

        except serial.SerialException as e:
            self.error_occurred.emit(f"打开串口失败: {str(e)}")
            self._serial = None
            return False
        except Exception as e:
            self.error_occurred.emit(f"打开串口异常: {str(e)}")
            self._serial = None
            return False

    def close(self):
        """关闭串口连接。"""
        self._stop_reader()
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass
            self._serial = None
        self.connection_changed.emit(False)

    # ── 数据发送 ──

    def send(self, data: bytes) -> int:
        """
        发送数据。

        Args:
            data: 要发送的字节数据

        Returns:
            int: 实际发送的字节数
        """
        if not self.is_open:
            self.error_occurred.emit("串口未打开，无法发送")
            return 0

        try:
            count = self._serial.write(data)
            self._tx_count += count
            self.tx_count_changed.emit(self._tx_count)
            return count
        except serial.SerialException as e:
            self.error_occurred.emit(f"发送失败: {str(e)}")
            return 0
        except Exception as e:
            self.error_occurred.emit(f"发送异常: {str(e)}")
            return 0

    def send_text(self, text: str, encoding: str = "utf-8",
                  append_newline: bool = False) -> int:
        """
        发送文本数据。

        Args:
            text: 要发送的文本
            encoding: 编码方式
            append_newline: 是否自动添加换行符 (\\r\\n)

        Returns:
            int: 实际发送的字节数
        """
        data = text.encode(encoding)
        if append_newline:
            data += b"\r\n"
        return self.send(data)

    def send_hex(self, hex_str: str) -> int:
        """
        发送 HEX 格式数据（如 "AA BB CC"）。

        Args:
            hex_str: 十六进制字符串，空格分隔

        Returns:
            int: 实际发送的字节数
        """
        try:
            hex_str = hex_str.strip().replace(" ", "")
            data = bytes.fromhex(hex_str)
            return self.send(data)
        except ValueError as e:
            self.error_occurred.emit(f"HEX 格式错误: {str(e)}")
            return 0

    # ── 数据读取（同步） ──

    def read(self, size: int = 1) -> bytes:
        """
        同步读取指定字节数。

        Args:
            size: 要读取的字节数

        Returns:
            bytes: 读取到的数据
        """
        if not self.is_open:
            return b""
        try:
            data = self._serial.read(size)
            if data:
                self._rx_count += len(data)
                self.rx_count_changed.emit(self._rx_count)
            return data
        except Exception:
            return b""

    def read_all(self) -> bytes:
        """
        读取当前缓冲区中所有可用数据。

        Returns:
            bytes: 读取到的数据
        """
        if not self.is_open:
            return b""
        try:
            count = self._serial.in_waiting
            if count > 0:
                data = self._serial.read(count)
                self._rx_count += len(data)
                self.rx_count_changed.emit(self._rx_count)
                return data
            return b""
        except Exception:
            return b""

    def read_line(self) -> bytes:
        """
        读取一行数据（直到换行符 \\n）。

        Returns:
            bytes: 读取到的数据（含换行符）
        """
        if not self.is_open:
            return b""
        try:
            data = self._serial.readline()
            if data:
                self._rx_count += len(data)
                self.rx_count_changed.emit(self._rx_count)
            return data
        except Exception:
            return b""

    def read_until(self, expected: bytes = b"\n") -> bytes:
        """
        读取直到遇到指定字节序列。

        Args:
            expected: 期望的结束序列

        Returns:
            bytes: 读取到的数据（含结束序列）
        """
        if not self.is_open:
            return b""
        try:
            data = self._serial.read_until(expected)
            if data:
                self._rx_count += len(data)
                self.rx_count_changed.emit(self._rx_count)
            return data
        except Exception:
            return b""

    # ── 信号控制 ──

    def set_dtr(self, state: bool):
        """设置 DTR 信号。"""
        if self.is_open:
            try:
                self._serial.dtr = state
            except Exception:
                pass

    def set_rts(self, state: bool):
        """设置 RTS 信号。"""
        if self.is_open:
            try:
                self._serial.rts = state
            except Exception:
                pass

    # ── 内部方法 ──

    def _start_reader(self):
        """启动异步读取线程。"""
        self._stop_reader()
        if self._serial and self._serial.is_open:
            self._reader = SerialReaderThread(self._serial)
            self._reader.data_received.connect(self._on_reader_data)
            self._reader.error_occurred.connect(self._on_reader_error)
            self._reader.start()

    def _stop_reader(self):
        """停止异步读取线程。"""
        if self._reader is not None:
            self._reader.stop()
            try:
                self._reader.data_received.disconnect(self._on_reader_data)
                self._reader.error_occurred.disconnect(self._on_reader_error)
            except TypeError:
                pass
            self._reader = None

    def _on_reader_data(self, data: bytes):
        """读取线程收到数据时的回调。"""
        self._rx_count += len(data)
        self.rx_count_changed.emit(self._rx_count)
        self.data_received.emit(data)

    def _on_reader_error(self, error_msg: str):
        """读取线程发生错误时的回调。"""
        self.error_occurred.emit(error_msg)

    # ── 资源清理 ──

    def cleanup(self):
        """清理资源（关闭串口和读取线程）。"""
        self.close()
