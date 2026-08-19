# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

from .base_tool import VisionTool, ToolResult, PipelineContext


class Grayscale(VisionTool):
    display_name = "灰度化"

    def __init__(self, params=None):
        super().__init__(params)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)

        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        return ToolResult(
            success=True,
            passed=True,
            processed_image=gray,
            data={"channels": 1},
            message="灰度化完成"
        )

    def get_param_widgets(self, parent):
        return []


class GaussianBlur(VisionTool):
    display_name = "高斯滤波"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("kernel_size", 5)
        self.params.setdefault("sigma", 1.0)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        ksize = int(self.params.get("kernel_size", 5))
        sigma = float(self.params.get("sigma", 1.0))

        # 自动修正：核大小必须为奇数
        if ksize % 2 == 0:
            ksize += 1
            self.params["kernel_size"] = ksize
        if ksize < 1:
            ksize = 1

        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        return ToolResult(
            success=True,
            passed=True,
            processed_image=blurred,
            data={"kernel_size": ksize, "sigma": sigma},
            message=f"高斯滤波完成 (核={ksize}, σ={sigma})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QHBoxLayout, QWidget, QLabel

        widgets = []

        kernel_spin = QSpinBox(parent)
        kernel_spin.setRange(1, 31)
        kernel_spin.setSingleStep(2)
        kernel_spin.setValue(int(self.params.get("kernel_size", 5)))
        kernel_spin.valueChanged.connect(
            lambda v: self.params.update({"kernel_size": v}))
        widgets.append(("核大小:", kernel_spin))

        sigma_spin = QDoubleSpinBox(parent)
        sigma_spin.setRange(0.1, 10.0)
        sigma_spin.setSingleStep(0.1)
        sigma_spin.setValue(float(self.params.get("sigma", 1.0)))
        sigma_spin.valueChanged.connect(
            lambda v: self.params.update({"sigma": v}))
        widgets.append(("标准差:", sigma_spin))

        return widgets


class HistEqualize(VisionTool):
    display_name = "直方图均衡化"

    def __init__(self, params=None):
        super().__init__(params)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)

        if len(img.shape) == 2:
            equalized = cv2.equalizeHist(img)
        else:
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        return ToolResult(
            success=True,
            passed=True,
            processed_image=equalized,
            data={},
            message="直方图均衡化完成"
        )

    def get_param_widgets(self, parent):
        return []


class Morphology(VisionTool):
    display_name = "形态学操作"

    # 结构元素形状映射
    SHAPE_MAP = {
        "矩形": cv2.MORPH_RECT,
        "椭圆": cv2.MORPH_ELLIPSE,
        "十字": cv2.MORPH_CROSS,
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("operation", "erode")
        self.params.setdefault("kernel_size", 3)
        self.params.setdefault("iterations", 1)
        self.params.setdefault("shape", "矩形")

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        op = self.params.get("operation", "erode")
        ksize = int(self.params.get("kernel_size", 3))
        iterations = int(self.params.get("iterations", 1))
        shape_name = self.params.get("shape", "矩形")

        # 自动修正：核大小必须为奇数
        if ksize % 2 == 0:
            ksize += 1
            self.params["kernel_size"] = ksize
        ksize = max(1, ksize)

        shape_cv = self.SHAPE_MAP.get(shape_name, cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(shape_cv, (ksize, ksize))

        if op == "erode":
            result = cv2.erode(img, kernel, iterations=iterations)
        elif op == "dilate":
            result = cv2.dilate(img, kernel, iterations=iterations)
        elif op == "open":
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif op == "close":
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            result = img.copy()

        return ToolResult(
            success=True,
            passed=True,
            processed_image=result,
            data={"operation": op, "kernel_size": ksize, "shape": shape_name, "iterations": iterations},
            message=f"形态学操作完成 ({op}, {shape_name}, 核={ksize}, 迭代={iterations})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QSpinBox, QHBoxLayout,
                                      QWidget, QLabel)

        widgets = []

        op_combo = QComboBox(parent)
        op_map = {"erode": "腐蚀", "dilate": "膨胀", "open": "开运算", "close": "闭运算"}
        for eng, cn in op_map.items():
            op_combo.addItem(cn, eng)
        current_op = self.params.get("operation", "erode")
        idx = op_combo.findData(current_op)
        if idx >= 0:
            op_combo.setCurrentIndex(idx)
        op_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"operation": op_combo.itemData(i)}))
        widgets.append(("操作:", op_combo))

        # 结构元素形状选择
        shape_combo = QComboBox(parent)
        for shape_name in self.SHAPE_MAP.keys():
            shape_combo.addItem(shape_name)
        current_shape = self.params.get("shape", "矩形")
        idx = shape_combo.findText(current_shape)
        if idx >= 0:
            shape_combo.setCurrentIndex(idx)
        shape_combo.currentTextChanged.connect(
            lambda v: self.params.update({"shape": v}))
        widgets.append(("形状:", shape_combo))

        kernel_spin = QSpinBox(parent)
        kernel_spin.setRange(1, 31)
        kernel_spin.setSingleStep(2)
        kernel_spin.setValue(int(self.params.get("kernel_size", 3)))
        kernel_spin.valueChanged.connect(
            lambda v: self.params.update({"kernel_size": v}))
        widgets.append(("核大小:", kernel_spin))

        iter_spin = QSpinBox(parent)
        iter_spin.setRange(1, 10)
        iter_spin.setValue(int(self.params.get("iterations", 1)))
        iter_spin.valueChanged.connect(
            lambda v: self.params.update({"iterations": v}))
        widgets.append(("迭代:", iter_spin))

        return widgets


class MultiROI(VisionTool):
    display_name = "多区域ROI"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("regions", [])
        self.params.setdefault("use_percentage", False)  # 是否使用百分比坐标

    def _normalize_regions(self, regions, img_shape=None):
        """Convert various region formats to {name: (x, y, w, h)} dict.
        
        支持百分比坐标：当 use_percentage=True 时，存储的坐标是百分比(0~100)，
        返回时根据图像实际尺寸转换为像素坐标。
        """
        if isinstance(regions, dict):
            return regions
        result = {}
        use_pct = self.params.get("use_percentage", False)
        h_img, w_img = img_shape[:2] if img_shape is not None else (1, 1)

        for r in regions:
            if isinstance(r, dict):
                name = r.get("name", "未命名")
                if not r.get("enabled", True):
                    continue
                if use_pct:
                    # 百分比坐标 -> 像素坐标
                    x = int(r.get("x", 0) / 100.0 * w_img)
                    y = int(r.get("y", 0) / 100.0 * h_img)
                    w = int(r.get("width", r.get("w", 100)) / 100.0 * w_img)
                    h = int(r.get("height", r.get("h", 100)) / 100.0 * h_img)
                else:
                    x = r.get("x", 0)
                    y = r.get("y", 0)
                    w = r.get("width", r.get("w", 100))
                    h = r.get("height", r.get("h", 100))
                result[name] = (x, y, w, h)
        return result

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        raw_regions = self.params.get("regions", [])
        regions = self._normalize_regions(raw_regions, img.shape)

        display = img.copy()
        # 在黑色背景上绘制标注，用于叠加到原图
        overlay = np.zeros_like(img)
        for name, (x, y, w, h) in regions.items():
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, name, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, name, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return ToolResult(
            success=True,
            passed=True,
            processed_image=display,
            overlay_image=overlay,
            regions=regions,
            data={"region_count": len(regions)},
            message=f"定义了 {len(regions)} 个ROI区域"
        )

    def get_param_widgets(self, parent):
        return []


class MedianBlur(VisionTool):
    display_name = "中值滤波"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("kernel_size", 5)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        ksize = int(self.params.get("kernel_size", 5))

        # 自动修正：核大小必须为奇数
        if ksize % 2 == 0:
            ksize += 1
            self.params["kernel_size"] = ksize
        if ksize < 1:
            ksize = 1

        blurred = cv2.medianBlur(img, ksize)

        return ToolResult(
            success=True,
            passed=True,
            processed_image=blurred,
            data={"kernel_size": ksize},
            message=f"中值滤波完成 (核={ksize})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import QSpinBox

        widgets = []

        kernel_spin = QSpinBox(parent)
        kernel_spin.setRange(1, 31)
        kernel_spin.setSingleStep(2)
        kernel_spin.setValue(int(self.params.get("kernel_size", 5)))
        kernel_spin.valueChanged.connect(
            lambda v: self.params.update({"kernel_size": v}))
        widgets.append(("核大小:", kernel_spin))

        return widgets


class Resize(VisionTool):
    display_name = "图像缩放"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("mode", "ratio")
        self.params.setdefault("width", 800)
        self.params.setdefault("height", 600)
        self.params.setdefault("percent", 50)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        mode = self.params.get("mode", "ratio")
        h, w = img.shape[:2]

        if mode == "fixed":
            target_w = int(self.params.get("width", 800))
            target_h = int(self.params.get("height", 600))
        elif mode == "percent":
            percent = float(self.params.get("percent", 50))
            target_w = int(w * percent / 100)
            target_h = int(h * percent / 100)
        else:
            target_w = int(self.params.get("width", 800))
            target_h = int(h * target_w / w)

        target_w = max(1, target_w)
        target_h = max(1, target_h)

        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 计算缩放比例，用于后续ROI坐标跟踪
        scale_x = target_w / w if w > 0 else 1.0
        scale_y = target_h / h if h > 0 else 1.0

        return ToolResult(
            success=True,
            passed=True,
            processed_image=resized,
            data={
                "original_size": f"{w}x{h}",
                "new_size": f"{target_w}x{target_h}",
                "scale_x": scale_x,
                "scale_y": scale_y,
                "_roi_scale": (scale_x, scale_y),  # 用于ROI坐标跟踪
            },
            message=f"缩放完成: {w}x{h} -> {target_w}x{target_h}"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QSpinBox, QDoubleSpinBox,
                                      QHBoxLayout, QWidget, QLabel,
                                      QStackedWidget)

        widgets = []

        mode_combo = QComboBox(parent)
        mode_combo.addItem("等比例缩放", "ratio")
        mode_combo.addItem("固定尺寸", "fixed")
        mode_combo.addItem("百分比", "percent")
        current_mode = self.params.get("mode", "ratio")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)

        # 用 QStackedWidget 根据 mode 切换显示不同参数
        stacked = QStackedWidget(parent)

        # page 0: ratio 模式 - 显示宽度（高度自动计算）
        ratio_widget = QWidget()
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(0, 0, 0, 0)
        width_spin = QSpinBox()
        width_spin.setRange(1, 10000)
        width_spin.setValue(int(self.params.get("width", 800)))
        width_spin.valueChanged.connect(
            lambda v: self.params.update({"width": v}))
        ratio_layout.addWidget(QLabel("目标宽度:"))
        ratio_layout.addWidget(width_spin)
        ratio_layout.addStretch()
        stacked.addWidget(ratio_widget)

        # page 1: fixed 模式 - 显示宽度和高度
        fixed_widget = QWidget()
        fixed_layout = QHBoxLayout(fixed_widget)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fw_spin = QSpinBox()
        fw_spin.setRange(1, 10000)
        fw_spin.setValue(int(self.params.get("width", 800)))
        fw_spin.valueChanged.connect(
            lambda v: self.params.update({"width": v}))
        fh_spin = QSpinBox()
        fh_spin.setRange(1, 10000)
        fh_spin.setValue(int(self.params.get("height", 600)))
        fh_spin.valueChanged.connect(
            lambda v: self.params.update({"height": v}))
        fixed_layout.addWidget(QLabel("宽度:"))
        fixed_layout.addWidget(fw_spin)
        fixed_layout.addWidget(QLabel("高度:"))
        fixed_layout.addWidget(fh_spin)
        fixed_layout.addStretch()
        stacked.addWidget(fixed_widget)

        # page 2: percent 模式 - 显示百分比
        pct_widget = QWidget()
        pct_layout = QHBoxLayout(pct_widget)
        pct_layout.setContentsMargins(0, 0, 0, 0)
        percent_spin = QDoubleSpinBox()
        percent_spin.setRange(1, 1000)
        percent_spin.setValue(float(self.params.get("percent", 50)))
        percent_spin.valueChanged.connect(
            lambda v: self.params.update({"percent": v}))
        pct_layout.addWidget(QLabel("百分比(%):"))
        pct_layout.addWidget(percent_spin)
        pct_layout.addStretch()
        stacked.addWidget(pct_widget)

        # 根据当前 mode 显示对应页面
        mode_map = {"ratio": 0, "fixed": 1, "percent": 2}
        stacked.setCurrentIndex(mode_map.get(current_mode, 0))

        def on_mode_changed(i):
            mode = mode_combo.itemData(i)
            self.params.update({"mode": mode})
            stacked.setCurrentIndex(mode_map.get(mode, 0))

        mode_combo.currentIndexChanged.connect(on_mode_changed)
        widgets.append(("模式:", mode_combo))
        widgets.append(("参数:", stacked))

        return widgets


class AdaptiveThreshold(VisionTool):
    display_name = "自适应阈值"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("method", "mean")
        self.params.setdefault("block_size", 11)
        self.params.setdefault("c", 2)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        method = self.params.get("method", "mean")
        block_size = int(self.params.get("block_size", 11))
        c = float(self.params.get("c", 2))

        # 自动修正：block_size 必须为奇数且 >= 3
        if block_size % 2 == 0:
            block_size += 1
            self.params["block_size"] = block_size
        if block_size < 3:
            block_size = 3

        if method == "gaussian":
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

        binary = cv2.adaptiveThreshold(gray, 255, adaptive_method,
                                       cv2.THRESH_BINARY, block_size, c)

        context.set_image('binary', binary)

        return ToolResult(
            success=True,
            passed=True,
            processed_image=binary,
            data={"method": method, "block_size": block_size},
            message=f"自适应阈值完成 ({method}, 块={block_size})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QSpinBox, QDoubleSpinBox,
                                      QHBoxLayout, QWidget, QLabel)

        widgets = []

        method_combo = QComboBox(parent)
        method_combo.addItem("均值法 (MEAN)", "mean")
        method_combo.addItem("高斯法 (GAUSSIAN)", "gaussian")
        current_method = self.params.get("method", "mean")
        idx = method_combo.findData(current_method)
        if idx >= 0:
            method_combo.setCurrentIndex(idx)
        method_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"method": method_combo.itemData(i)}))
        widgets.append(("方法:", method_combo))

        block_spin = QSpinBox(parent)
        block_spin.setRange(3, 99)
        block_spin.setSingleStep(2)
        block_spin.setValue(int(self.params.get("block_size", 11)))
        block_spin.valueChanged.connect(
            lambda v: self.params.update({"block_size": v}))
        widgets.append(("块大小:", block_spin))

        c_spin = QDoubleSpinBox(parent)
        c_spin.setRange(-10, 10)
        c_spin.setValue(float(self.params.get("c", 2)))
        c_spin.valueChanged.connect(
            lambda v: self.params.update({"c": v}))
        widgets.append(("常数C:", c_spin))

        return widgets
