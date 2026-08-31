# -*- coding: utf-8 -*-
"""颜色识别核心模块。

提供颜色模型（ColorModel）、匹配算法（ColorMatcher）、
采样器（ColorSampler）与颜色库（ColorLibrary），
支持从图像上点选/框选任意颜色作为识别目标。
"""

from .color_model import ColorModel
from .color_matcher import ColorMatcher
from .color_sampler import ColorSampler
from .color_library import ColorLibrary

__all__ = [
    "ColorModel",
    "ColorMatcher",
    "ColorSampler",
    "ColorLibrary",
]
