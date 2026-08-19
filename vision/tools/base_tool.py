# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import cv2


@dataclass
class ToolResult:
    success: bool = True
    passed: bool = True
    processed_image: Optional[np.ndarray] = None
    overlay_image: Optional[np.ndarray] = None  # 在原图上标注检测结果，供工业操作员查看
    data: Dict[str, Any] = field(default_factory=dict)
    regions: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    tool_type: str = ""
    tool_name: str = ""
    elapsed_ms: float = 0.0


@dataclass
class PipelineContext:
    original_image: np.ndarray
    current_image: np.ndarray
    regions: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, 'ToolResult'] = field(default_factory=dict)
    _data: Dict[str, Any] = field(default_factory=dict)
    _images: Dict[str, np.ndarray] = field(default_factory=dict)

    def set_data(self, key: str, value: Any):
        self._data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set_image(self, key: str, image: np.ndarray):
        self._images[key] = image

    def get_image(self, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self._images.get(key, default)


class VisionTool(ABC):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = type(self).__name__
        self.params: Dict[str, Any] = params if params is not None else {}
        # 如果子类在类级别定义了 display_name，则保留子类的定义
        # 否则使用类名作为默认值
        cls_display = type(self).__dict__.get('display_name')
        if cls_display is not None:
            self.display_name = cls_display
        else:
            self.display_name = self.name
        # 当使用 ROI 区域作为输入源时，保存完整帧图像，供子类返回 processed_image
        self._full_frame_image: Optional[np.ndarray] = None

    @abstractmethod
    def process(self, context: PipelineContext) -> ToolResult:
        pass

    def _get_input_image(self, context: PipelineContext) -> np.ndarray:
        # 兼容旧方案文件：可能存的是 "input_source"（无下划线前缀）
        input_source = self.params.get("_input_source") or self.params.get("input_source", "current")

        # 保存完整帧，供子类在返回 processed_image 时使用
        # 注意：如果上游有灰度化步骤，current_image 可能是单通道
        # 这里统一转成3通道BGR，确保下游步骤不会因通道数问题报错
        full_frame = context.current_image.copy()
        if len(full_frame.shape) == 2 or (len(full_frame.shape) == 3 and full_frame.shape[2] == 1):
            full_frame = cv2.cvtColor(full_frame, cv2.COLOR_GRAY2BGR)
        self._full_frame_image = full_frame

        if input_source == "original":
            return context.original_image.copy()

        # 兼容中文 "区域:" 前缀（旧方案文件手动编辑可能使用中文）
        if input_source.startswith("区域:"):
            input_source = "region:" + input_source[3:]

        if input_source.startswith("region:"):
            region_name = input_source[7:]
            if region_name in context.regions:
                x, y, w, h = context.regions[region_name]
                # 从 current_image 裁剪 ROI 区域，保留上游预处理结果
                # 裁剪坐标不能超出图像边界
                img_h, img_w = context.current_image.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                if w > 0 and h > 0:
                    return context.current_image[y:y+h, x:x+w].copy()
                else:
                    return context.current_image.copy()
            else:
                return context.current_image.copy()
        else:
            return context.current_image.copy()

    def get_param_widgets(self, parent):
        return []

    def get_input_source_widgets(self, parent, context_info: Dict[str, List[str]]) -> list:
        from PyQt5.QtWidgets import QComboBox

        widgets = []

        combo = QComboBox(parent)
        combo.addItem("当前图像", "current")
        combo.addItem("原始图像", "original")

        for region_name in context_info.get("regions", []):
            combo.addItem(f"区域: {region_name}", f"region:{region_name}")

        current_source = self.params.get("_input_source", "current")
        index = combo.findData(current_source)
        if index >= 0:
            combo.setCurrentIndex(index)

        # 用户切换输入源时同步更新 params
        combo.currentIndexChanged.connect(
            lambda i: self.params.update({"_input_source": combo.itemData(i)}))

        widgets.append(("输入源:", combo))
        return widgets

    def to_dict(self) -> Dict[str, Any]:
        return self.params

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)
