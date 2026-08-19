# -*- coding: utf-8 -*-
"""
相机管理模块 — Daheng (大恒) GalaxySDK 实现

对外接口（保持与旧版一致）：
    CameraManager          — 相机管理器（单例模式）
    CameraGrabbingThread   — 实时取流线程（QThread）
    raw_to_opencv()        — 原始帧数据转 OpenCV 图像
    is_mono_data()         — 判断是否为黑白像素格式
    is_color_data()        — 判断是否为彩色像素格式
    pixel_type_to_opencv() — 像素格式 → OpenCV 类型映射
"""

import os
import sys
import ctypes
import traceback
from typing import Optional, Tuple, List, Dict, Any, Callable
from enum import IntEnum

import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# ============================================================
# 日志工具（与项目现有日志风格保持一致）
# ============================================================
try:
    from core.log_manager import log_info, log_error, log_warning, log_debug
except ImportError:
    import logging
    _logger = logging.getLogger("CameraManager")
    def log_info(msg):    _logger.info(msg)
    def log_error(msg):   _logger.error(msg)
    def log_warning(msg): _logger.warning(msg)
    def log_debug(msg):   _logger.debug(msg)


# ============================================================
# 导入 Daheng gxipy 库
# ============================================================
try:
    import gxipy as gx
    from gxipy.gxidef import GxDeviceClassList, GxPixelFormatEntry, GxSwitchEntry
    from gxipy.gxidef import GxTriggerSourceEntry, GxAutoEntry, GxAcquisitionModeEntry
    from gxipy.gxidef import GxDSStreamBufferHandlingModeEntry
    DAHENG_AVAILABLE = True
except ImportError:
    DAHENG_AVAILABLE = False
    log_error("无法导入 gxipy 库，请确认 Daheng GalaxySDK Python 包已正确安装")


# ============================================================
# 像素格式常量（兼容旧版调用方使用的枚举值）
# ============================================================
# 这些值用于 pixel_type 参数传递，调用方（main_window.py）通过
# 这些值来判断像素格式。我们映射为 Daheng 的 GxPixelFormatEntry 值。
class PixelType:
    """像素类型枚举（兼容旧版接口）"""
    Mono8  = 0x01080001
    Mono10 = 0x01100003
    Mono12 = 0x01100005
    BayerRG8  = 0x01080009
    BayerRG10 = 0x0110000B
    BayerRG12 = 0x0110000D
    BayerGB8  = 0x01080010
    BayerGB10 = 0x01100012
    BayerGB12 = 0x01100014
    RGB8      = 0x02180014


# ============================================================
# 固定 Bayer 模式配置
# ============================================================
# 请根据你的相机实际 Bayer 模式修改此值。
# 可在 Galaxy Viewer 中查看相机的 Pixel Format 设置来确定正确的模式。
# 常用值：
#   cv2.COLOR_BAYER_BG2BGR  — Bayer BG（最常见）
#   cv2.COLOR_BAYER_GB2BGR  — Bayer GB
#   cv2.COLOR_BAYER_RG2BGR  — Bayer RG
#   cv2.COLOR_BAYER_GR2BGR  — Bayer GR
# 注意：OpenCV 的 Bayer 模式命名规则是"第二个像素的颜色"，
# 例如 BG2BGR 表示 2x2 块中 (0,0)=B, (0,1)=G, (1,0)=G, (1,1)=R
CAMERA_BAYER_PATTERN = cv2.COLOR_BAYER_GB2BGR

# ============================================================
# 图像后处理配置（提升画面质量，接近 Galaxy Viewer 效果）
# ============================================================

# Gamma 校正值（sRGB 标准 Gamma ≈ 2.2）
# 值越大，暗部提亮越明显。Galaxy Viewer 默认使用 0.45 的 Gamma 值，
# 对应 OpenCV 校正时使用 1/0.45 ≈ 2.22
CAMERA_GAMMA = 2.2

# 锐化强度（0.0 = 不锐化，建议 0.3 ~ 1.0）
# Galaxy Viewer 默认会应用轻度锐化以提升画面清晰度
CAMERA_SHARPEN_STRENGTH = 0.5

# 16bit → 8bit 转换方式
# "shift" : 直接右移 8 位（固定映射，亮度稳定，推荐）
# "norm"  : NORM_MINMAX 动态拉伸（每帧自适应，亮度可能跳动）
CAMERA_16BIT_CONVERSION = "shift"


# ============================================================
# 像素格式工具函数（兼容旧版接口）
# ============================================================

def _ensure_int_pixel_type(pixel_type) -> int:
    """确保 pixel_type 是 Python int 类型（处理 c_int 等情况）"""
    if isinstance(pixel_type, int):
        return pixel_type
    try:
        return int(pixel_type)
    except (TypeError, ValueError):
        return 0


def is_mono_data(pixel_type: int) -> bool:
    """判断是否为黑白（Mono）像素格式"""
    pt = _ensure_int_pixel_type(pixel_type)
    mono_codes = {
        0x01080001,  # Mono8
        0x01100003,  # Mono10
        0x01100005,  # Mono12
        0x01100007,  # Mono16
    }
    return pt in mono_codes


def is_color_data(pixel_type: int) -> bool:
    """判断是否为彩色（Bayer/RGB）像素格式"""
    pt = _ensure_int_pixel_type(pixel_type)
    # 所有 Bayer 格式（8bit/10bit/12bit/16bit）和 RGB 格式
    bayer_formats = {
        0x01080008, 0x01080009, 0x0108000A, 0x0108000B,  # Bayer8  GR/RG/GB/BG
        0x0110000C, 0x0110000D, 0x0110000E, 0x0110000F,  # Bayer10 GR/RG/GB/BG
        0x01100010, 0x01100011, 0x01100012, 0x01100013,  # Bayer12 GR/RG/GB/BG
        0x0110002E, 0x0110002F, 0x01100030, 0x01100031,  # Bayer16 GR/RG/GB/BG
    }
    rgb8_code = 0x02180014  # RGB8
    return pt in bayer_formats or pt == rgb8_code


def pixel_type_to_opencv(pixel_type: int) -> Tuple[Optional[int], bool]:
    """
    将 Daheng 像素格式映射为 OpenCV 类型。
    
    Returns:
        (cv2_type, is_color)
        cv2_type: OpenCV 枚举值（如 cv2.CV_8UC1），None 表示未知
        is_color: 是否为彩色格式
    """
    pt = _ensure_int_pixel_type(pixel_type)
    
    # Mono 格式
    if pt in (0x01080001,):  # Mono8
        return cv2.CV_8UC1, False
    if pt in (0x01100003, 0x01100005, 0x01100007):  # Mono10, Mono12, Mono16
        return cv2.CV_16UC1, False
    
    # Bayer 8bit 格式 — 数据是 8bit，直接 demosaic
    if pt in (0x01080008, 0x01080009, 0x0108000A, 0x0108000B):
        return cv2.CV_8UC1, True
    
    # Bayer 10/12/16bit 格式 — 数据是 16bit，需先缩放到 8bit 再 demosaic
    if pt in (0x0110000C, 0x0110000D, 0x0110000E, 0x0110000F,   # Bayer10 GR/RG/GB/BG
              0x01100010, 0x01100011, 0x01100012, 0x01100013,   # Bayer12 GR/RG/GB/BG
              0x0110002E, 0x0110002F, 0x01100030, 0x01100031):  # Bayer16 GR/RG/GB/BG
        return cv2.CV_16UC1, True
    
    # RGB8
    if pt == 0x02180014:
        return cv2.CV_8UC3, True
    
    log_warning(f"未知的像素格式: 0x{pt:08X}")
    return None, False


def _convert_16bit_to_8bit(img_16u: np.ndarray) -> np.ndarray:
    """
    将 16bit 图像转换为 8bit 图像。
    
    使用固定位右移方式（>> 8），避免 NORM_MINMAX 动态拉伸导致的
    亮度跳动问题。这种方式与 Galaxy Viewer 的默认行为更接近。
    
    Args:
        img_16u: 16bit numpy 数组 (np.uint16)
    
    Returns:
        8bit numpy 数组 (np.uint8)
    """
    if CAMERA_16BIT_CONVERSION == "norm":
        # 动态拉伸（保留全部动态范围，但亮度可能跳动）
        return cv2.normalize(img_16u, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        # 固定右移 8 位（默认，亮度稳定，与 Galaxy Viewer 行为一致）
        return (img_16u >> 8).astype(np.uint8)


def _apply_gamma(img_bgr: np.ndarray, gamma: float = CAMERA_GAMMA) -> np.ndarray:
    """
    应用 Gamma 校正，提亮暗部区域，增强对比度。
    
    Galaxy Viewer 默认应用 Gamma=0.45 校正，对应 OpenCV 实现中
    使用 1/0.45 ≈ 2.22 的 Gamma 值进行逆校正。
    
    Args:
        img_bgr: BGR 格式的 8bit 图像
        gamma: Gamma 值（默认 2.2，对应 sRGB 标准）
    
    Returns:
        Gamma 校正后的 BGR 图像
    """
    if gamma <= 0 or gamma == 1.0:
        return img_bgr
    # 归一化到 [0, 1] → 应用 Gamma → 还原到 [0, 255]
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
    return cv2.LUT(img_bgr, look_up_table)


def _apply_sharpen(img_bgr: np.ndarray, strength: float = CAMERA_SHARPEN_STRENGTH) -> np.ndarray:
    """
    应用轻度锐化，提升画面清晰度。
    
    使用 Unsharp Mask 方式：原图 + strength * (原图 - 高斯模糊)
    
    Args:
        img_bgr: BGR 格式的 8bit 图像
        strength: 锐化强度（0.0 = 不锐化，建议 0.3 ~ 1.0）
    
    Returns:
        锐化后的 BGR 图像
    """
    if strength <= 0:
        return img_bgr
    # 高斯模糊（核大小 3x3，sigma=1.0）
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 1.0)
    # 原图 + strength * (原图 - 模糊)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def raw_to_opencv(frame_data: bytes, width: int, height: int,
                  pixel_type: int) -> Optional[np.ndarray]:
    """
    将相机原始帧数据转换为 OpenCV BGR 图像。
    
    处理流程：
        1. 原始数据 → numpy 数组
        2. Bayer demosaic（彩色 Bayer 格式）
        3. 16bit → 8bit 转换（固定右移，避免亮度跳动）
        4. Gamma 校正（提亮暗部，增强对比度）
        5. 轻度锐化（提升清晰度）
    
    这是兼容旧版接口的全局函数，main_window.py 中直接调用。
    
    Args:
        frame_data: 原始帧字节数据
        width:  图像宽度
        height: 图像高度
        pixel_type: 像素格式枚举值
    
    Returns:
        BGR 格式的 np.ndarray，失败返回 None
    """
    try:
        if not frame_data or width <= 0 or height <= 0:
            return None
        
        cv_type, is_color = pixel_type_to_opencv(pixel_type)
        if cv_type is None:
            return None
        
        # 计算期望数据长度
        if cv_type == cv2.CV_8UC1:
            expected_len = width * height
            dtype = np.uint8
            channels = 1
        elif cv_type == cv2.CV_16UC1:
            expected_len = width * height * 2
            dtype = np.uint16
            channels = 1
        elif cv_type == cv2.CV_8UC3:
            expected_len = width * height * 3
            dtype = np.uint8
            channels = 3
        else:
            return None
        
        # 数据长度检查
        actual_len = len(frame_data)
        if actual_len < expected_len:
            log_warning(f"帧数据长度不足: 期望 {expected_len}, 实际 {actual_len}")
            # 尝试用已有数据填充
            pad_len = expected_len - actual_len
            frame_data = frame_data + b'\x00' * pad_len
        elif actual_len > expected_len:
            frame_data = frame_data[:expected_len]
        
        # 创建 numpy 数组
        img = np.frombuffer(frame_data, dtype=dtype).reshape(height, width)
        
        if is_color and channels == 1:
            # Bayer 格式 — 使用固定 Bayer 模式 demosaic（避免自动检测在暗画面下闪烁）
            if dtype == np.uint16:
                # 16bit Bayer → 缩放到 8bit → demosaic
                img_8u = _convert_16bit_to_8bit(img)
                img_bgr = cv2.cvtColor(img_8u, CAMERA_BAYER_PATTERN)
            else:
                # 8bit Bayer → 直接 demosaic
                img_bgr = cv2.cvtColor(img, CAMERA_BAYER_PATTERN)
            
            # 后处理：Gamma 校正 + 锐化（使画面接近 Galaxy Viewer 效果）
            img_bgr = _apply_gamma(img_bgr, CAMERA_GAMMA)
            img_bgr = _apply_sharpen(img_bgr, CAMERA_SHARPEN_STRENGTH)
            return img_bgr
        
        elif is_color and channels == 3:
            # 已经是 RGB8 — 转换为 BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # 后处理：Gamma 校正 + 锐化
            img_bgr = _apply_gamma(img_bgr, CAMERA_GAMMA)
            img_bgr = _apply_sharpen(img_bgr, CAMERA_SHARPEN_STRENGTH)
            return img_bgr
        
        else:
            # Mono 格式 — 堆叠为 3 通道灰度
            if img.dtype == np.uint16:
                # 16bit → 8bit 固定右移
                img_8u = _convert_16bit_to_8bit(img)
            else:
                img_8u = img
            img_bgr = cv2.merge([img_8u, img_8u, img_8u])
            # 后处理：Gamma 校正 + 锐化
            img_bgr = _apply_gamma(img_bgr, CAMERA_GAMMA)
            img_bgr = _apply_sharpen(img_bgr, CAMERA_SHARPEN_STRENGTH)
            return img_bgr
    
    except Exception as e:
        log_error(f"raw_to_opencv 转换失败: {e}")
        return None


def _auto_detect_bayer(bayer_img: np.ndarray) -> int:
    """
    自动检测 Bayer 排列模式。
    通过分析图像 2x2 块的颜色分量来推断 Bayer 模式。
    
    Returns:
        OpenCV Bayer 转换常量（如 cv2.COLOR_BAYER_BG2BGR）
    """
    h, w = bayer_img.shape
    if h < 4 or w < 4:
        return cv2.COLOR_BAYER_BG2BGR  # 默认
    
    # 取中心区域分析
    cy, cx = h // 2, w // 2
    # 检查 2x2 块的四角亮度差异
    block = bayer_img[cy-1:cy+1, cx-1:cx+1].astype(np.float32)
    
    # 计算水平和垂直方向的梯度
    grad_h = np.abs(bayer_img[cy-1:cy+1, cx-2:cx+2].astype(np.float32)).mean()
    grad_v = np.abs(bayer_img[cy-2:cy+2, cx-1:cx+1].astype(np.float32)).mean()
    
    # 通过分析自然图像的 Bayer 模式特征来推断
    # 大多数 Bayer 相机使用 BG 或 GB 模式
    # 检查 (0,0) 和 (1,1) 位置的差异
    diff_diag = abs(float(block[0, 0]) - float(block[1, 1]))
    diff_anti = abs(float(block[0, 1]) - float(block[1, 0]))
    
    if diff_diag < diff_anti:
        # 对角线相似 → BG 或 GR
        if grad_h > grad_v:
            return cv2.COLOR_BAYER_BG2BGR
        else:
            return cv2.COLOR_BAYER_GR2BGR
    else:
        # 反对角线相似 → GB 或 RG
        if grad_h > grad_v:
            return cv2.COLOR_BAYER_GB2BGR
        else:
            return cv2.COLOR_BAYER_RG2BGR


# ============================================================
# 相机取流线程
# ============================================================

class CameraGrabbingThread(QThread):
    """
    实时取流线程。
    使用 Daheng SDK 的 data_stream[0].get_image() 轮询模式。
    """
    frame_received = pyqtSignal(int, int, int, bytes)  # width, height, pixel_type, data
    
    def __init__(self, device):
        super().__init__()
        self._device = device          # gx.Device 对象
        self._running = False
        self._data_stream = None
    
    def run(self):
        """线程主循环"""
        self._running = True
        
        try:
            # 获取数据流对象
            if not self._device or not self._device.data_stream:
                log_error("取流线程: 设备没有数据流")
                self._running = False
                return
            
            self._data_stream = self._device.data_stream[0]
            
            # 设置采集缓冲区数量（优化性能）
            try:
                self._data_stream.set_acquisition_buffer_number(8)
            except Exception:
                pass
            
            # 开始采集
            self._device.stream_on()
            log_info("取流线程: 开始采集")
            
            while self._running:
                try:
                    # 获取图像（超时 1000ms）
                    raw_image = self._data_stream.get_image(1000)
                    if raw_image is None:
                        continue
                    
                    # 获取图像信息
                    width = raw_image.get_width()
                    height = raw_image.get_height()
                    pixel_format = raw_image.get_pixel_format()
                    frame_data = raw_image.get_data()
                    
                    if frame_data is None or width == 0 or height == 0:
                        continue
                    
                    # 将 Daheng 像素格式映射为兼容的 pixel_type 值
                    pixel_type = _daheng_pf_to_compat(pixel_format)
                    
                    # 发出信号
                    self.frame_received.emit(width, height, pixel_type, bytes(frame_data))
                    
                except Exception as e:
                    if self._running:
                        # 超时是正常情况，不打印错误
                        err_msg = str(e)
                        if "timeout" not in err_msg.lower() and "超时" not in err_msg:
                            log_debug(f"取流线程获取图像异常: {e}")
        
        except Exception as e:
            log_error(f"取流线程异常: {e}")
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        try:
            if self._device:
                self._device.stream_off()
        except Exception:
            pass
        log_info("取流线程: 已停止")
    
    def stop(self):
        """停止取流线程"""
        self._running = False
        self.wait(3000)  # 等待最多 3 秒


def _daheng_pf_to_compat(daheng_pf) -> int:
    """
    将 Daheng SDK 的像素格式值映射为兼容的 PixelType 枚举值。
    
    Daheng 的像素格式定义在 GxPixelFormatEntry 中，
    与 Hikvision 的 PixelType_header.py 定义一致（均为 GenICam 标准）。
    
    注意：daheng_pf 可能是 c_int 类型，需要转为 Python int。
    """
    # 确保返回 Python int（处理 c_int 等情况）
    return int(daheng_pf)


# ============================================================
# 相机管理器（单例）
# ============================================================

class CameraManager:
    """
    相机管理器 — Daheng GalaxySDK 实现。
    
    使用方式（与旧版一致）：
        CameraManager.initialize_sdk()   # 应用启动时
        mgr = CameraManager()
        devices = mgr.enumerate_devices()
        mgr.open_camera(devices[0])
        mgr.start_grabbing(callback)
        ...
        mgr.close_camera()
        CameraManager.finalize_sdk()     # 应用退出时
    """
    
    _sdk_initialized = False
    _device_manager = None  # gx.DeviceManager 单例
    
    def __init__(self):
        self._device = None          # 当前打开的 gx.Device 对象
        self._device_info = None     # 当前设备信息 dict
        self._is_open = False
        self._is_trigger_mode = False #当前是否为触发模式
        self._grabbing_thread = None
        self._data_stream = None
        self._stream_on = False      # 数据流是否已开启
    
    # ---- 属性 ----
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    @property
    def is_trigger_mode(self) -> bool:
        """当前是否为触发模式"""
        return self._is_trigger_mode
    
    @property
    def device(self):
        return self._device
    
    # ---- SDK 生命周期 ----
    
    @staticmethod
    def initialize_sdk():
        """
        初始化 Daheng SDK。
        注意：Daheng SDK 不需要显式初始化，DeviceManager 实例化时自动初始化。
        此方法仅做标记和检查。
        """
        if CameraManager._sdk_initialized:
            return
        
        if not DAHENG_AVAILABLE:
            log_error("Daheng gxipy 不可用，请检查安装")
            return
        
        try:
            # 实例化 DeviceManager（单例，自动初始化 SDK）
            if CameraManager._device_manager is None:
                CameraManager._device_manager = gx.DeviceManager()
            CameraManager._sdk_initialized = True
            log_info("Daheng SDK 初始化成功")
        except Exception as e:
            log_error(f"Daheng SDK 初始化失败: {e}")
    
    @staticmethod
    def finalize_sdk():
        """
        反初始化 Daheng SDK。
        Daheng SDK 不需要显式反初始化，但为了兼容旧版接口保留此方法。
        """
        if not CameraManager._sdk_initialized:
            return
        
        try:
            # 清理 DeviceManager 引用
            CameraManager._device_manager = None
            CameraManager._sdk_initialized = False
            log_info("Daheng SDK 已释放")
        except Exception as e:
            log_error(f"Daheng SDK 释放异常: {e}")
    
    @staticmethod
    def _ensure_sdk_initialized():
        """确保 SDK 已初始化"""
        if not CameraManager._sdk_initialized:
            CameraManager.initialize_sdk()
    
    # ---- 设备枚举 ----
    
    def enumerate_devices(self, timeout_ms: int = 200) -> List[Dict[str, Any]]:
        """
        枚举所有可用的相机设备。
        
        默认先尝试同网段快速搜索（update_device_list），
        如果未找到设备，自动切换为跨网段搜索（update_all_device_list），
        以解决相机 IP 与电脑网卡 IP 不在同一网段时搜索不到的问题。
        
        Args:
            timeout_ms: 枚举超时（毫秒）
        
        Returns:
            设备信息字典列表，每个字典包含:
                - 'index':       设备索引
                - 'sn':          序列号
                - 'name':        设备名称（型号）
                - 'vendor':      厂商名
                - 'display_name': 显示名称
                - 'device_class': 设备类型（GigE/U3V）
                - 'ip':          IP 地址（GigE 设备）
                - 'mac':         MAC 地址（GigE 设备）
                - '_raw_info':   原始设备信息
        """
        self._ensure_sdk_initialized()
        
        devices = []
        try:
            mgr = CameraManager._device_manager
            if mgr is None:
                return devices
            
            # 第一步：先尝试同网段快速搜索
            result = mgr.update_device_list(timeout_ms)
            if isinstance(result, tuple):
                num, dev_info_list = result
            else:
                num = result
                dev_info_list = mgr.get_device_info()
            
            # 第二步：同网段没找到，自动跨网段搜索
            if num <= 0 or not dev_info_list:
                log_info("同网段未发现设备，尝试跨网段搜索...")
                result = mgr.update_all_device_list(timeout_ms)
                if isinstance(result, tuple):
                    num, dev_info_list = result
                else:
                    num = result
                    dev_info_list = mgr.get_device_info()
            
            if num <= 0:
                log_info("未发现相机设备")
                return devices
            if not dev_info_list:
                return devices
            
            for i, info in enumerate(dev_info_list):
                try:
                    device_info = self._parse_device_info(info, i)
                    if device_info:
                        devices.append(device_info)
                except Exception as e:
                    log_debug(f"解析设备信息失败 (index={i}): {e}")
                    continue
            
            log_info(f"枚举到 {len(devices)} 个相机设备")
        
        except Exception as e:
            log_error(f"枚举设备失败: {e}")
        
        return devices
    
    def _parse_device_info(self, dev_info, index: int) -> Optional[Dict[str, Any]]:
        """
        解析 Daheng 设备信息为统一字典格式。
        
        dev_info 是 gxipy 返回的设备信息元组:
            (vendor, model, serial_number, device_class, mac, ip, user_id)
        """
        try:
            if isinstance(dev_info, dict):
                #新版gxipy返回的是dict(详情请在__get_device_info_list中查看)
                vendor = str(dev_info.get('vendor_name', ''))
                model  = str(dev_info.get('model_name', '')) 
                sn     = str(dev_info.get('sn', ''))
                dev_class = int(dev_info.get('device_class', 0))
                mac    = str(dev_info.get('mac', ''))
                ip     = str(dev_info.get('ip', ''))
                user_id = str(dev_info.get('user_id', ''))
            elif isinstance(dev_info, (tuple, list)):
                #旧版元组格式
                vendor = str(dev_info[0]) if len(dev_info) > 0 else ""
                model  = str(dev_info[1]) if len(dev_info) > 1 else ""
                sn     = str(dev_info[2]) if len(dev_info) > 2 else ""
                dev_class = int(dev_info[3]) if len(dev_info) > 3 else 0
                mac    = str(dev_info[4]) if len(dev_info) > 4 else ""
                ip     = str(dev_info[5]) if len(dev_info) > 5 else ""
                user_id = str(dev_info[6]) if len(dev_info) > 6 else ""
            else:
                # 如果是对象，尝试属性访问
                vendor = getattr(dev_info, 'vendor', '') or ''
                model  = getattr(dev_info, 'model', '') or ''
                sn     = getattr(dev_info, 'serial_number', '') or ''
                dev_class = getattr(dev_info, 'device_class', 0) or 0
                mac    = getattr(dev_info, 'mac', '') or ''
                ip     = getattr(dev_info, 'ip', '') or ''
                user_id = getattr(dev_info, 'user_id', '') or ''
            
            # 构建显示名称
            if user_id:
                display_name = f"{user_id} ({model})"
            else:
                display_name = f"{model} [{sn[:8]}]" if sn else model
            
            # 设备类型
            if dev_class == GxDeviceClassList.GEV:
                class_name = "GigE"
            elif dev_class == GxDeviceClassList.U3V:
                class_name = "U3V"
            elif dev_class == GxDeviceClassList.U2:
                class_name = "U2"
            else:
                class_name = f"Unknown({dev_class})"
            
            return {
                'index':        index,
                'sn':           sn,
                'name':         model,
                'vendor':       vendor,
                'display_name': display_name,
                'device_class': class_name,
                'ip':           ip,
                'mac':          mac,
                'user_id':      user_id,
                '_raw_info':    dev_info,
            }
        
        except Exception as e:
            log_debug(f"解析设备信息异常: {e}")
            return None
    
    # ---- 打开/关闭相机 ----
    
    def open_camera(self, dev_info) -> bool:
        """
        打开相机设备。
        
        Args:
            dev_info: 设备信息字典（来自 enumerate_devices）
        
        Returns:
            是否成功
        """
        if self._is_open:
            log_warning("相机已打开，请先关闭")
            return False
        
        try:
            self._ensure_sdk_initialized()
            
            mgr = CameraManager._device_manager
            if mgr is None:
                log_error("DeviceManager 未初始化")
                return False
            
            # 获取设备索引
            if isinstance(dev_info, dict):
                index = dev_info.get('index', 0)
            else:
                index = int(dev_info)
            
            # 打开设备
            self._device = mgr.open_device_by_index(index)
            if self._device is None:
                log_error("打开设备失败: 返回空")
                return False
            
            self._device_info = dev_info
            self._is_open = True
            
            # 如果是 GigE 设备，优化网络参数
            if isinstance(dev_info, dict) and dev_info.get('device_class') == 'GigE':
                self._optimize_gige()
            
            log_info(f"相机已打开: {dev_info.get('display_name', str(index))}")
            return True
        
        except Exception as e:
            log_error(f"打开相机失败: {e}")
            self._device = None
            self._is_open = False
            return False
    
    def _optimize_gige(self):
        """优化 GigE 相机网络参数"""
        try:
            if self._device is None:
                return
            
            # 设置 GigE 数据包大小（需先查询最佳值）
            try:
                # 尝试设置 packet size 为 9000（Jumbo Frame）
                packet_size_feature = getattr(self._device, 'GevSCPSPacketSize', None)
                if packet_size_feature is not None and packet_size_feature.is_writable():
                    # 先尝试获取最佳值
                    try:
                        # 部分相机支持自动获取最佳包大小
                        optimal = self._device.GevSCPSPacketSize.get()
                        if optimal > 0:
                            packet_size_feature.set(optimal)
                    except Exception:
                        packet_size_feature.set(9000)  # 默认 Jumbo Frame
            except Exception:
                pass
            
            # 设置帧传输模式为尽可能快
            try:
                # 设置流通道包延时为 0（最小延迟）
                delay_feature = getattr(self._device, 'GevSCPD', None)
                if delay_feature is not None and delay_feature.is_writable():
                    delay_feature.set(0)
            except Exception:
                pass
            
            # 设置帧传输包数量
            try:
                fw_feature = getattr(self._device, 'GevSCFW', None)
                if fw_feature is not None and fw_feature.is_writable():
                    fw_feature.set(4)  # 一次发送 4 个包
            except Exception:
                pass
            
            log_info("GigE 网络参数已优化")
        
        except Exception as e:
            log_debug(f"GigE 优化异常: {e}")
    
    def close_camera(self):
        """关闭相机"""
        if not self._is_open:
            return
        
        try:
            # 先停止取流
            self.stop_grabbing()
            
            # 如果 capture_once 开启了流，关闭它
            if self._stream_on:
                try:
                    self._device.stream_off()
                except Exception:
                    pass
                self._stream_on = False
            
            # 关闭设备
            if self._device is not None:
                self._device.close_device()
                self._device = None
            
            self._is_open = False
            self._device_info = None
            log_info("相机已关闭")
        
        except Exception as e:
            log_error(f"关闭相机异常: {e}")
            self._is_open = False
    
    # ---- 触发模式 ----
    
    def set_trigger_mode(self, enable: bool) -> bool:
        """
        设置触发模式。
        
        Args:
            enable: True=触发模式（软触发），False=连续采集模式
        
        Returns:
            是否成功
        """
        if not self._is_open or self._device is None:
            log_warning("相机未打开")
            return False
        
        try:
            # 设置采集模式
            acq_mode = getattr(self._device, 'AcquisitionMode', None)
            if acq_mode is not None and acq_mode.is_writable():
                acq_mode.set(GxAcquisitionModeEntry.CONTINUOUS)
            
            # 设置触发模式
            trigger_mode = getattr(self._device, 'TriggerMode', None)
            if trigger_mode is None or not trigger_mode.is_writable():
                log_warning("设备不支持 TriggerMode")
                return False
            
            if enable:
                # 开启触发模式
                trigger_mode.set(GxSwitchEntry.ON)
                self._is_trigger_mode = True
                
                # 设置触发源为软触发（Software）
                trigger_source = getattr(self._device, 'TriggerSource', None)
                if trigger_source is not None and trigger_source.is_writable():
                    trigger_source.set(GxTriggerSourceEntry.SOFTWARE)
                
                # 设置触发选择器为 FrameStart
                trigger_selector = getattr(self._device, 'TriggerSelector', None)
                if trigger_selector is not None and trigger_selector.is_writable():
                    from gxipy.gxidef import GxTriggerSelectorEntry
                    trigger_selector.set(GxTriggerSelectorEntry.FRAME_START)
                
                log_info("触发模式: 已开启（软触发）")
            else:
                self._is_trigger_mode = False
                # 关闭触发模式（连续采集）
                trigger_mode.set(GxSwitchEntry.OFF)
                log_info("触发模式: 已关闭（连续采集）")

            return True
        
        except Exception as e:
            log_error(f"设置触发模式失败: {e}")
            return False
    
    def trigger_once(self) -> bool:
        """
        执行一次软触发（仅在触发模式下有效）。
        
        Returns:
            是否成功
        """
        if not self._is_open or self._device is None:
            return False
        
        try:
            trigger_software = getattr(self._device, 'TriggerSoftware', None)
            if trigger_software is None:
                log_warning("设备不支持 TriggerSoftware")
                return False
            
            trigger_software.send_command()
            return True
        
        except Exception as e:
            log_error(f"软触发失败: {e}")
            return False
    
    # ---- 参数读写 ----
    
    def get_float_param(self, name: str) -> Optional[float]:
        """获取浮点参数"""
        if self._device is None:
            return None
        try:
            feature = getattr(self._device, name, None)
            if feature is None:
                return None
            # 检查参数是否可读，避免 "is not readable" 异常
            if hasattr(feature, 'is_readable') and not feature.is_readable():
                return None
            return float(feature.get())
        except Exception:
            return None
    
    def set_float_param(self, name: str, value: float) -> bool:
        """设置浮点参数"""
        if self._device is None:
            return False
        try:
            feature = getattr(self._device, name, None)
            if feature is None or not feature.is_writable():
                return False
            feature.set(value)
            return True
        except Exception as e:
            log_debug(f"设置浮点参数 {name}={value} 失败: {e}")
            return False
    
    def get_int_param(self, name: str) -> Optional[int]:
        """获取整数参数"""
        if self._device is None:
            return None
        try:
            feature = getattr(self._device, name, None)
            if feature is None:
                return None
            # 检查参数是否可读
            if hasattr(feature, 'is_readable') and not feature.is_readable():
                return None
            return int(feature.get())
        except Exception:
            return None
    
    def set_int_param(self, name: str, value: int) -> bool:
        """设置整数参数"""
        if self._device is None:
            return False
        try:
            feature = getattr(self._device, name, None)
            if feature is None or not feature.is_writable():
                return False
            feature.set(value)
            return True
        except Exception as e:
            log_debug(f"设置整数参数 {name}={value} 失败: {e}")
            return False
    
    def get_enum_param(self, name: str) -> Optional[int]:
        """获取枚举参数"""
        if self._device is None:
            return None
        try:
            feature = getattr(self._device, name, None)
            if feature is None:
                return None
            return int(feature.get())
        except Exception:
            return None
    
    def set_enum_param(self, name: str, value: int) -> bool:
        """设置枚举参数"""
        if self._device is None:
            return False
        try:
            feature = getattr(self._device, name, None)
            if feature is None or not feature.is_writable():
                return False
            feature.set(value)
            return True
        except Exception as e:
            log_debug(f"设置枚举参数 {name}={value} 失败: {e}")
            return False
    
    # ---- 常用参数快捷方法 ----
    
    def set_exposure_time(self, value_us: float) -> bool:
        """
        设置曝光时间。
        
        Args:
            value_us: 曝光时间（微秒）
        """
        return self.set_float_param('ExposureTime', value_us)
    
    def set_gain(self, value_db: float) -> bool:
        """
        设置增益。
        
        Args:
            value_db: 增益值（dB）
        """
        return self.set_float_param('Gain', value_db)
    
    def get_exposure_time(self) -> Optional[float]:
        """获取当前曝光时间（微秒）"""
        return self.get_float_param('ExposureTime')
    
    def get_gain(self) -> Optional[float]:
        """获取当前增益值（dB）"""
        return self.get_float_param('Gain')
    
    def get_frame_rate(self) -> Optional[float]:
        """获取当前帧率（fps）"""
        return self.get_float_param('AcquisitionFrameRate')

    def set_frame_rate(self, value_fps: float) -> bool:
        """
        设置帧率。
        
        Args:
            value_fps: 帧率（fps）
        """
        return self.set_float_param('AcquisitionFrameRate', value_fps)
    
    # ---- 图像采集 ----
    
    def start_grabbing(self, frame_callback: Callable = None) -> bool:
        """
        开始实时取流。
        
        Args:
            frame_callback: 帧回调函数，签名 callback(width, height, pixel_type, data)
        
        Returns:
            是否成功
        """
        if not self._is_open or self._device is None:
            log_warning("相机未打开")
            return False
        
        try:
            # 如果已有取流线程在运行，先停止
            self.stop_grabbing()
            
            # 创建并启动取流线程
            self._grabbing_thread = CameraGrabbingThread(self._device)
            
            if frame_callback is not None:
                self._grabbing_thread.frame_received.connect(frame_callback)
            
            self._grabbing_thread.start()
            log_info("实时取流已开始")
            return True
        
        except Exception as e:
            log_error(f"开始取流失败: {e}")
            return False
    
    def stop_grabbing(self):
        """停止实时取流"""
        if self._grabbing_thread is not None:
            try:
                self._grabbing_thread.stop()
            except Exception as e:
                log_debug(f"停止取流线程异常: {e}")
            self._grabbing_thread = None
            log_info("实时取流已停止")
    
    def capture_once(self, timeout_ms: int = 3000) -> Optional[Tuple[int, int, int, bytes]]:
        """
        单次拍照。
        每次拍照时临时开启数据流，拍照完成后立即关闭，避免持续占用资源。
        
        Args:
            timeout_ms: 超时时间（毫秒）
        
        Returns:
            (width, height, pixel_type, data) 元组，失败返回 None
        """
        if not self._is_open or self._device is None:
            log_warning("相机未打开")
            return None
        
        stream_started_here = False
        data_stream = None
        
        try:
            # 确保流已开启
            if self._device.data_stream:
                data_stream = self._device.data_stream[0]
            else:
                log_error("设备没有数据流")
                return None
            
            # 如果流未开启，临时开启它
            if not self._stream_on:
                self._device.stream_on()
                self._stream_on = True
                stream_started_here = True
            
            # 获取图像
            raw_image = data_stream.get_image(timeout_ms)
            if raw_image is None:
                log_warning("单次拍照超时")
                return None
            
            # 提取图像信息
            width = raw_image.get_width()
            height = raw_image.get_height()
            pixel_format = raw_image.get_pixel_format()
            frame_data = raw_image.get_data()
            
            if frame_data is None or width == 0 or height == 0:
                log_warning("单次拍照返回空数据")
                return None
            
            pixel_type = _daheng_pf_to_compat(pixel_format)
            
            log_info(f"单次拍照成功: {width}x{height}, pixel_type=0x{pixel_type:08X}")
            return (width, height, pixel_type, bytes(frame_data))
        
        except Exception as e:
            log_error(f"单次拍照异常: {e}")
            return None
        
        finally:
            # 如果是在本次调用中开启的流，拍照完成后立即关闭
            if stream_started_here:
                try:
                    self._device.stream_off()
                except Exception:
                    pass
                self._stream_on = False
    
    # ---- 图像显示工具 ----
    
    @staticmethod
    def convert_to_qimage(width: int, height: int, pixel_type: int,
                          frame_data: bytes) -> Optional[QImage]:
        """
        将相机帧数据转换为 QImage（兼容旧版接口）。
        
        Args:
            width:  图像宽度
            height: 图像高度
            pixel_type: 像素格式
            frame_data: 原始帧数据
        
        Returns:
            QImage 对象，失败返回 None
        """
        try:
            cv_img = raw_to_opencv(frame_data, width, height, pixel_type)
            if cv_img is None:
                return None
            
            h, w = cv_img.shape[:2]
            if cv_img.ndim == 2:
                # 灰度图
                return QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
            else:
                # BGR → RGB
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                return QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
        
        except Exception as e:
            log_error(f"convert_to_qimage 失败: {e}")
            return None
    
    @staticmethod
    def display_on_label(label, width: int, height: int, pixel_type: int,
                         frame_data: bytes):
        """
        将相机帧数据显示在 QLabel 上（兼容旧版接口）。
        
        Args:
            label:       QLabel 对象
            width:       图像宽度
            height:      图像高度
            pixel_type:  像素格式
            frame_data:  原始帧数据
        """
        try:
            qimage = CameraManager.convert_to_qimage(width, height, pixel_type, frame_data)
            if qimage is None:
                return
            
            pixmap = QPixmap.fromImage(qimage)
            label.setPixmap(pixmap)
        
        except Exception as e:
            log_error(f"display_on_label 失败: {e}")
