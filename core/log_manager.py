# -*- coding: utf-8 -*-

import logging
import os
import time
import threading
import contextlib
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from core.paths import LOGS_DIR, PRODUCTION_DATA_DIR

# 默认日志限额：50 GB
_DEFAULT_MAX_LOG_SIZE = 50 * 1024 ** 3
# 清理目标比例：清理到最大限额的一半
_DEFAULT_CLEANUP_RATIO = 0.5
# 需要监控和清理的目录列表（只清理这些目录下的文件，不碰 schemes 等）
_CLEANUP_DIRS = [LOGS_DIR, PRODUCTION_DATA_DIR]

# ============================================================================
# 日志上下文：用于控制日志是否写入文件
# ============================================================================
# 线程局部变量，用于标记当前线程是否应抑制文件日志输出
_thread_local = threading.local()

def _is_file_logging_suppressed() -> bool:
    """检查当前线程是否抑制了文件日志输出"""
    return getattr(_thread_local, 'suppress_file_log', False)

@contextlib.contextmanager
def suppress_file_logging():
    """
    上下文管理器：在当前线程中临时抑制文件日志输出。
    
    用于设计模式等场景，避免将调试/测试过程中的日志写入文件，
    只保留自动化流程测试中的错误日志。
    
    使用方式:
        with suppress_file_logging():
            # 此范围内的日志不会写入文件
            vision_engine.execute(...)
    """
    old_value = getattr(_thread_local, 'suppress_file_log', False)
    _thread_local.suppress_file_log = True
    try:
        yield
    finally:
        _thread_local.suppress_file_log = old_value


def _get_dir_size(path: str) -> int:
    """递归计算目录总大小（字节）"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _get_dir_size(entry.path)
    except (PermissionError, OSError):
        pass
    return total


def _get_all_files_sorted(dirs: list) -> list:
    """获取多个目录下所有文件，按修改时间升序排列（最早的在前）"""
    files = []
    for d in dirs:
        try:
            for root, _, filenames in os.walk(d):
                for fname in filenames:
                    fpath = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                        files.append((fpath, mtime))
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError):
            continue
    files.sort(key=lambda x: x[1])  # 按修改时间升序
    return files


def _get_total_size(dirs: list) -> int:
    """计算多个目录的总大小"""
    total = 0
    for d in dirs:
        total += _get_dir_size(d)
    return total


def _cleanup_old_logs(max_size: int, cleanup_ratio: float):
    """
    当监控目录总大小超过 max_size 时，从最早的文件开始删除，
    直到总大小 <= max_size * cleanup_ratio。
    只删除 _CLEANUP_DIRS 列表中的文件。
    """
    current_size = _get_total_size(_CLEANUP_DIRS)
    if current_size <= max_size:
        return

    target_size = int(max_size * cleanup_ratio)
    files = _get_all_files_sorted(_CLEANUP_DIRS)

    for file_path, _ in files:
        if current_size <= target_size:
            break
        try:
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            current_size -= file_size
        except (PermissionError, OSError):
            continue


class _SuppressFileFilter(logging.Filter):
    """
    日志过滤器：当当前线程设置了 suppress_file_log 标志时，
    抑制日志写入文件（仅影响文件处理器，不影响控制台输出）。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 如果当前线程抑制了文件日志，则过滤掉（不写入文件）
        return not _is_file_logging_suppressed()


class _SizeCheckTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    继承 TimedRotatingFileHandler，在每次实际写入日志后检查并清理日志空间。
    清理操作在后台线程中执行，避免磁盘 I/O 阻塞日志写入。
    """

    def __init__(self, *args, **kwargs):
        self._max_log_size: int = kwargs.pop('max_log_size', _DEFAULT_MAX_LOG_SIZE)
        self._cleanup_ratio: float = kwargs.pop('cleanup_ratio', _DEFAULT_CLEANUP_RATIO)
        self._last_check_time: float = 0
        self._check_interval: float = 300.0  # 两次检查的最小间隔（秒），5分钟检查一次
        self._lock = threading.Lock()
        super().__init__(*args, **kwargs)
        # 添加抑制过滤器，使文件处理器支持线程级别的日志抑制
        self.addFilter(_SuppressFileFilter())

    def emit(self, record: logging.LogRecord):
        """写入日志记录，写入后检查是否需要清理"""
        super().emit(record)
        now = time.time()
        if now - self._last_check_time >= self._check_interval:
            self._last_check_time = now
            if self._lock.acquire(blocking=False):
                # 在后台线程执行清理，避免磁盘 I/O 阻塞日志写入
                t = threading.Thread(
                    target=self._do_cleanup,
                    daemon=True,
                )
                t.start()

    def _do_cleanup(self):
        """在后台线程执行日志清理"""
        try:
            _cleanup_old_logs(self._max_log_size, self._cleanup_ratio)
        except Exception:
            pass  # 清理失败不影响主流程
        finally:
            self._lock.release()


class LogManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._logger = None
        self._setup_logger()

    def _setup_logger(self):
        os.makedirs(LOGS_DIR, exist_ok=True)

        self._logger = logging.getLogger('Vision_Substrate_Silicone')
        self._logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 从配置读取日志限额参数
        try:
            from core.config_manager import ConfigManager
            cfg = ConfigManager()
            max_size_gb = cfg.get('system.log_max_size_gb', 50)
            cleanup_ratio = cfg.get('system.log_cleanup_ratio', 0.5)
            max_log_size = int(max_size_gb * 1024 ** 3)
        except Exception:
            max_log_size = _DEFAULT_MAX_LOG_SIZE
            cleanup_ratio = _DEFAULT_CLEANUP_RATIO

        log_file = os.path.join(LOGS_DIR, 'vision_system.log')
        file_handler = _SizeCheckTimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8',
            max_log_size=max_log_size,
            cleanup_ratio=cleanup_ratio,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self._logger

    @staticmethod
    def cleanup_now(max_size: Optional[int] = None, cleanup_ratio: Optional[float] = None):
        """
        手动触发日志和错误数据清理。
        清理范围包括 logs 和 errors 目录下的文件（按修改时间从早到晚删除）。
        可在外部（如定时任务、启动时）主动调用。

        Args:
            max_size: 最大字节数，默认 50GB
            cleanup_ratio: 清理目标比例，默认 0.5（清理到一半）
        """
        if max_size is None:
            max_size = _DEFAULT_MAX_LOG_SIZE
        if cleanup_ratio is None:
            cleanup_ratio = _DEFAULT_CLEANUP_RATIO
        _cleanup_old_logs(max_size, cleanup_ratio)


_log_manager = LogManager()


def init_logger():
    pass


def get_logger() -> logging.Logger:
    return _log_manager.get_logger()


def log_debug(msg: str):
    get_logger().debug(msg)


def log_info(msg: str):
    get_logger().info(msg)


def log_warning(msg: str):
    get_logger().warning(msg)


def log_error(msg: str):
    get_logger().error(msg)


def log_critical(msg: str):
    get_logger().critical(msg)


def log_exception(msg: str):
    get_logger().exception(msg)
