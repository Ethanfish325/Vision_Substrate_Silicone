# -*- coding: utf-8 -*-
"""颜色识别优化单元测试。

覆盖 ColorModel 序列化、ColorMatcher 三种匹配算法、
ColorSampler 采样、ColorLibrary 颜色库、向后兼容等。
"""

import os
import sys
import time

import numpy as np
import cv2
import pytest

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.color.color_model import ColorModel
from vision.color.color_matcher import ColorMatcher
from vision.color.color_sampler import ColorSampler
from vision.color.color_library import ColorLibrary


# ========== ColorModel 序列化 ==========

def test_color_model_roundtrip():
    """ColorModel 应能 to_dict / from_dict 无损往返。"""
    model = ColorModel(
        name="基板蓝", color_space="HSV", center=(110, 180, 150),
        match_mode="range", tolerance=(8, 40, 40),
        cluster_centers=[(110, 180, 150), (115, 200, 160)],
        normalize_illumination=True, adaptive_threshold=True, source="custom",
    )
    restored = ColorModel.from_dict(model.to_dict())
    assert restored.name == model.name
    assert restored.color_space == model.color_space
    assert restored.center == model.center
    assert restored.match_mode == model.match_mode
    assert restored.tolerance == model.tolerance
    assert restored.cluster_centers == model.cluster_centers
    assert restored.normalize_illumination is True
    assert restored.adaptive_threshold is True


def test_color_model_from_empty():
    """空 dict 应返回默认 ColorModel。"""
    model = ColorModel.from_dict(None)
    assert model.name == "未命名"
    assert model.match_mode == "range"


# ========== ColorMatcher 区间匹配 ==========

def test_hsv_red_wraparound():
    """红色跨 H=0 边界应正确匹配。"""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # BGR 纯红
    model = ColorModel(name="红", center=(0, 255, 255), tolerance=(10, 50, 50))
    mask = ColorMatcher.build_mask(img, model)
    assert mask.sum() > 0


def test_hsv_red_wraparound_high_hue():
    """H 接近 180 的红色（如 175）也应匹配红色模型。"""
    img = np.zeros((100, 100, 3), np.uint8)
    # 构造 H=175 的红色
    hsv = np.zeros((100, 100, 3), np.uint8)
    hsv[:, :] = (175, 255, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    model = ColorModel(name="红", center=(0, 255, 255), tolerance=(10, 50, 50))
    mask = ColorMatcher.build_mask(bgr, model)
    assert mask.sum() > 0


def test_range_match_blue():
    """蓝色应被蓝色模型匹配，不被红色模型匹配。"""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = (255, 0, 0)  # BGR 纯蓝
    blue = ColorModel(name="蓝", center=(115, 255, 255), tolerance=(15, 50, 50))
    red = ColorModel(name="红", center=(0, 255, 255), tolerance=(10, 50, 50))
    assert ColorMatcher.build_mask(img, blue).sum() > 0
    assert ColorMatcher.build_mask(img, red).sum() == 0


# ========== ColorMatcher 距离匹配 ==========

def test_distance_match():
    """距离匹配应容忍轻微色差。"""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # BGR 纯红
    # 红色在 HSV 中 H≈0, S=255, V=255
    model = ColorModel(name="t", center=(0, 255, 255),
                       match_mode="distance", distance_threshold=30.0)
    mask = ColorMatcher.build_mask(img, model)
    assert mask.sum() > 0


def test_distance_match_reject():
    """距离匹配应拒绝差异大的颜色。"""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # 红
    model = ColorModel(name="t", center=(255, 0, 0),  # 蓝
                       match_mode="distance", distance_threshold=30.0)
    mask = ColorMatcher.build_mask(img, model)
    assert mask.sum() == 0


# ========== ColorMatcher 聚类匹配 ==========

def test_cluster_match():
    """聚类匹配应对多个中心取并集。"""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # 红
    model = ColorModel(
        name="金属红", match_mode="cluster",
        cluster_centers=[(0, 255, 255), (10, 200, 200)],
        distance_threshold=40.0,
    )
    mask = ColorMatcher.build_mask(img, model)
    assert mask.sum() > 0


# ========== ColorSampler 采样 ==========

def test_sample_point():
    """点选采样应生成合理的 range 模型。"""
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # 红
    model = ColorSampler.sample_point(img, 100, 100, name="红")
    assert model.match_mode == "range"
    assert model.source == "custom"
    # 中心应为红色（HSV 中 H≈0）
    assert model.center[0] < 10 or model.center[0] > 170


def test_sample_roi_auto_tolerance():
    """框选采样应自动推断合理容差。"""
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (255, 0, 0)  # 蓝
    model = ColorSampler.sample_roi(img, 50, 50, 100, 100, name="蓝")
    assert model.match_mode == "range"
    assert model.center[0] > 90  # 蓝色 H 约 115


def test_sample_roi_cluster():
    """多峰颜色（如带高光）应生成 cluster 模型。"""
    img = np.zeros((200, 200, 3), np.uint8)
    # 一半红一半暗红，模拟多峰
    img[:100, :] = (0, 0, 255)
    img[100:, :] = (0, 0, 100)
    model = ColorSampler.sample_roi(img, 0, 0, 200, 200, name="多峰红")
    # 可能生成 range 或 cluster，但都应能匹配
    mask = ColorMatcher.build_mask(img, model)
    assert mask.sum() > 0


# ========== ColorLibrary 颜色库 ==========

def test_color_library_presets():
    """预设库应包含 8 种颜色。"""
    lib = ColorLibrary(path="data/color_library_test.json")
    assert len(lib.get_presets()) == 8


def test_color_library_add_remove(tmp_path):
    """自定义颜色应能添加、持久化、删除。"""
    path = str(tmp_path / "color_library.json")
    lib = ColorLibrary(path=path)
    model = ColorModel(name="测试色", center=(100, 100, 100))
    lib.add(model, persist=True)
    assert len(lib.get_custom()) == 1

    # 重新加载应保留
    lib2 = ColorLibrary(path=path)
    assert len(lib2.get_custom()) == 1
    assert lib2.get_custom()[0].name == "测试色"

    # 删除
    assert lib2.remove("测试色") is True
    assert len(lib2.get_custom()) == 0


def test_color_library_temporary():
    """临时集合不应持久化。"""
    lib = ColorLibrary(path="data/color_library_test.json")
    lib.add(ColorModel(name="临时色"), persist=False)
    assert len(lib.get_temporary()) == 1
    lib.clear_temporary()
    assert len(lib.get_temporary()) == 0


# ========== 光照归一化与自适应阈值 ==========

def test_normalize_illumination():
    """光照归一化不应改变图像尺寸。"""
    img = np.random.randint(0, 255, (100, 100, 3), np.uint8)
    out = ColorMatcher.normalize_illumination(img)
    assert out.shape == img.shape


def test_adaptive_threshold_mask():
    """自适应阈值应返回同尺寸掩膜。"""
    img = np.random.randint(0, 255, (100, 100, 3), np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    out = ColorMatcher.adaptive_threshold_mask(mask, img)
    assert out.shape == mask.shape


# ========== 性能基准 ==========

def test_performance_1080p():
    """1080p 单次匹配应 < 50ms（宽松阈值，避免 CI 波动）。"""
    img = np.random.randint(0, 255, (1920, 1080, 3), np.uint8)
    model = ColorModel(name="t", center=(100, 100, 100), tolerance=(10, 50, 50))
    t0 = time.time()
    for _ in range(5):
        ColorMatcher.build_mask(img, model)
    avg_ms = (time.time() - t0) / 5 * 1000
    assert avg_ms < 50, f"匹配耗时 {avg_ms:.1f}ms 超过 50ms"


# ========== 向后兼容（旧六参数） ==========

def test_legacy_backward_compat():
    """旧六参数方案应保持原行为。"""
    from vision.tools.recognize import ColorRecognition
    from vision.tools.base_tool import PipelineContext

    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # 红
    tool = ColorRecognition({
        "h_min": 0, "h_max": 10, "s_min": 50, "s_max": 255,
        "v_min": 50, "v_max": 255, "color_name": "红色",
    })
    context = PipelineContext(original_image=img, current_image=img)
    result = tool.process(context)
    assert result.success
    assert result.data["color_area"] > 0


def test_color_model_path():
    """新 ColorModel 路径应正常工作（双轨制优先）。"""
    from vision.tools.recognize import ColorRecognition
    from vision.tools.base_tool import PipelineContext

    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # 红
    model = ColorModel(name="自定义红", center=(0, 255, 255),
                       tolerance=(10, 50, 50), match_mode="range")
    tool = ColorRecognition({"color_model": model.to_dict()})
    context = PipelineContext(original_image=img, current_image=img)
    result = tool.process(context)
    assert result.success
    assert result.data["color_area"] > 0
    assert result.data["color_name"] == "自定义红"


def test_color_model_distance_path():
    """距离匹配路径应正常工作。"""
    from vision.tools.recognize import ColorRecognition
    from vision.tools.base_tool import PipelineContext

    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # 红
    model = ColorModel(name="距离红", center=(0, 255, 255),
                       match_mode="distance", distance_threshold=40.0)
    tool = ColorRecognition({"color_model": model.to_dict()})
    context = PipelineContext(original_image=img, current_image=img)
    result = tool.process(context)
    assert result.success
    assert result.data["color_area"] > 0


def test_color_model_illumination_path():
    """光照归一化 + 自适应阈值路径应正常工作。"""
    from vision.tools.recognize import ColorRecognition
    from vision.tools.base_tool import PipelineContext

    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # 红
    model = ColorModel(name="增强红", center=(0, 255, 255),
                       tolerance=(10, 50, 50), match_mode="range",
                       normalize_illumination=True, adaptive_threshold=True)
    tool = ColorRecognition({"color_model": model.to_dict()})
    context = PipelineContext(original_image=img, current_image=img)
    result = tool.process(context)
    assert result.success
    assert result.data["color_area"] > 0
