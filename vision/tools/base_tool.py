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
    # ROI 随动附加信息: {region_name: (cx, cy, w, h, angle_deg)}
    # regions[name] 恒为轴对齐外接框(兼容下游解包);此表保存真正的
    # 旋转矩形中心/参考宽高/角度,供 base_tool 摆正裁剪与 ROI 框绘制。
    region_rot: Dict[str, Tuple] = field(default_factory=dict)

    def set_data(self, key: str, value: Any):
        self._data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set_image(self, key: str, image: np.ndarray):
        self._images[key] = image

    def get_image(self, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self._images.get(key, default)

    # ── 位置修正(定位)结果通道 ──
    # 由 PositionCorrect 步骤写入;MultiROI 依据它把参考 ROI 变换为当前图坐标。
    @property
    def location(self) -> Optional[Dict[str, Any]]:
        return self._data.get("location")

    @location.setter
    def location(self, value: Optional[Dict[str, Any]]):
        self._data["location"] = value


class VisionTool(ABC):
    # 能力标记：该算子是否支持"图像上取色"交互（仅 ColorRecognition 等
    # 颜色类算子声明；通用参数对话框据此按算子显示/隐藏取色区，避免
    # 取色 UI 与无关算子公用）。
    SUPPORTS_COLOR_PICK: bool = False

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

                # ── ROI 随动:该区域存在旋转信息(由 MultiROI 依据定位写入)──
                # 需要把旋转矩形区域局部摆正为水平,再交给下游算子。
                rot = context.region_rot.get(region_name)
                if rot is not None:
                    from vision.geometry_util import (crop_rotated_rect,
                                                      is_axis_aligned)
                    rcx, rcy, rw, rh, rang = rot
                    if not is_axis_aligned(rang):
                        img_hh, img_ww = context.current_image.shape[:2]
                        if not (rcx - rw / 2 > 0 and rcy - rh / 2 > 0
                                and rcx + rw / 2 < img_ww
                                and rcy + rh / 2 < img_hh):
                            # 旋转区域出界:回退外接框普通裁剪(宁可不摆正,
                            # 也不因越界返回黑图;出界本身会由边界校验兜底 NG)
                            x0 = max(0, min(x, img_ww - 1))
                            y0 = max(0, min(y, img_hh - 1))
                            x1 = min(img_ww, x + w)
                            y1 = min(img_hh, y + h)
                            if x1 > x0 and y1 > y0:
                                return context.current_image[y0:y1,
                                                             x0:x1].copy()
                            return context.current_image.copy()
                        return crop_rotated_rect(
                            context.current_image,
                            rcx, rcy, rw, rh, rang)
                    # 轴对齐旋转(0/90/180/270):angle≈0 直接裁剪;
                    # 90/180/270 走摆正,保证下游内容方向正确
                    if abs(rang) > 1e-3:
                        return crop_rotated_rect(
                            context.current_image,
                            rcx, rcy, rw, rh, rang)

                # 从 current_image 裁剪 ROI 区域，保留上游预处理结果
                # 裁剪坐标不能超出图像边界
                img_h, img_w = context.current_image.shape[:2]
                # 若 ROI 坐标明显超出当前图像范围（说明当前图像是已裁剪的局部图，
                # 而 ROI 坐标是针对完整相机图设计的），则回退到从原始图像裁剪 ROI，
                # 因为 ROI 坐标通常是基于原始完整相机图设计的。
                if x >= img_w or y >= img_h:
                    orig = context.original_image
                    if orig is not None:
                        orig_h, orig_w = orig.shape[:2]
                        # 若原始图像足够大且 ROI 有效，则从原始图像裁剪 ROI
                        if x < orig_w and y < orig_h:
                            ox = max(0, min(x, orig_w - 1))
                            oy = max(0, min(y, orig_h - 1))
                            ow = min(w, orig_w - ox)
                            oh = min(h, orig_h - oy)
                            if ow > 0 and oh > 0:
                                print(f"[DEBUG][{type(self).__name__}] ROI {region_name}({x},{y},{w},{h}) 超出当前图像({img_w}x{img_h})，回退从原始图像({orig_w}x{orig_h})裁剪")
                                return orig[oy:oy+oh, ox:ox+ow].copy()
                    print(f"[DEBUG][{type(self).__name__}] ROI {region_name}({x},{y},{w},{h}) 超出图像({img_w}x{img_h})，回退整图")
                    return context.current_image.copy()
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                # 裁剪区域过小（相对原始 ROI 明显缩小，说明 ROI 与图像尺寸不匹配）
                # 也回退到从原始图像裁剪
                if w <= 0 or h <= 0:
                    print(f"[DEBUG][{type(self).__name__}] ROI {region_name} 裁剪后无效({w}x{h})，回退整图")
                    return context.current_image.copy()
                if w < max(1, int(context.regions[region_name][2] * 0.1)) or \
                   h < max(1, int(context.regions[region_name][3] * 0.1)):
                    print(f"[DEBUG][{type(self).__name__}] ROI {region_name} 裁剪后过小({w}x{h})，回退整图")
                    return context.current_image.copy()
                print(f"[DEBUG][{type(self).__name__}] ROI {region_name}({x},{y},{w},{h}) 从图像({img_w}x{img_h})裁剪")
                return context.current_image[y:y+h, x:x+w].copy()
            else:
                print(f"[DEBUG][{type(self).__name__}] ROI {region_name} 不在 regions 中，使用整图")
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
