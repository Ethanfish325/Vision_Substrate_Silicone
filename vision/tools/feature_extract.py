# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QGroupBox, QSlider, QStackedWidget
from PyQt5.QtCore import Qt

from .base_tool import VisionTool, ToolResult, PipelineContext


class CannyEdge(VisionTool):
    display_name = "Canny边缘检测"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault('auto_threshold', False)
        self.params.setdefault('low_threshold', 50)
        self.params.setdefault('high_threshold', 150)
        self.params.setdefault('aperture_size', 3)
        self.params.setdefault('l2_gradient', False)

    @staticmethod
    def _compute_auto_thresholds(gray: np.ndarray) -> Tuple[int, int]:
        """使用Otsu算法自动计算Canny双阈值。
        
        计算步骤：
        1. 对灰度图进行中值滤波降噪（核大小5x5）
        2. 使用Otsu算法计算最佳二值化阈值
        3. 低阈值 = Otsu阈值 * 0.5，高阈值 = Otsu阈值 * 1.5
        4. 将结果钳制在[0, 255]范围内
        """
        blurred = cv2.medianBlur(gray, 5)
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        low = int(np.clip(otsu_thresh * 0.5, 0, 255))
        high = int(np.clip(otsu_thresh * 1.5, 0, 255))
        # 确保 high > low，至少相差10
        if high - low < 10:
            high = min(low + 10, 255)
        return low, high

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        aperture = self.params['aperture_size']
        l2 = self.params['l2_gradient']

        if self.params.get('auto_threshold', False):
            low, high = self._compute_auto_thresholds(gray)
            threshold_info = f"自动(Otsu)"
        else:
            low = self.params['low_threshold']
            high = self.params['high_threshold']
            # 自动修正：低阈值 <= 高阈值
            if low > high:
                high = low
                self.params['high_threshold'] = high
            threshold_info = f"{low}/{high}"

        edges = cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=l2)

        context.set_image('edges', edges)

        return ToolResult(
            success=True, passed=True,
            processed_image=cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
            message=f"Canny边缘检测完成 (阈值:{threshold_info})"
        )

    def get_param_widgets(self, parent):
        widgets = []

        auto_cb = QCheckBox(parent)
        auto_cb.setChecked(self.params.get('auto_threshold', False))
        auto_cb.stateChanged.connect(
            lambda v: self.params.update({'auto_threshold': bool(v)}))
        widgets.append(("自动阈值", auto_cb))

        low_sb = QSpinBox(parent)
        low_sb.setRange(0, 255)
        low_sb.setValue(self.params['low_threshold'])
        low_sb.setEnabled(not self.params.get('auto_threshold', False))
        low_sb.valueChanged.connect(lambda v: self.params.update({'low_threshold': v}))
        widgets.append(("低阈值", low_sb))

        high_sb = QSpinBox(parent)
        high_sb.setRange(0, 255)
        high_sb.setValue(self.params['high_threshold'])
        high_sb.setEnabled(not self.params.get('auto_threshold', False))
        high_sb.valueChanged.connect(lambda v: self.params.update({'high_threshold': v}))
        widgets.append(("高阈值", high_sb))

        # 自动阈值复选框联动：勾选时禁用低/高阈值SpinBox，取消时启用
        def _on_auto_changed(state):
            enabled = not bool(state)
            low_sb.setEnabled(enabled)
            high_sb.setEnabled(enabled)
        auto_cb.stateChanged.connect(_on_auto_changed)

        aperture_cb = QComboBox(parent)
        aperture_cb.addItems(['3', '5', '7'])
        aperture_cb.setCurrentText(str(self.params['aperture_size']))
        aperture_cb.currentTextChanged.connect(lambda v: self.params.update({'aperture_size': int(v)}))
        widgets.append(("Sobel核大小", aperture_cb))

        l2_cb = QCheckBox(parent)
        l2_cb.setChecked(self.params['l2_gradient'])
        l2_cb.stateChanged.connect(lambda v: self.params.update({'l2_gradient': bool(v)}))
        widgets.append(("L2梯度", l2_cb))

        return widgets


class Threshold(VisionTool):
    display_name = "阈值分割"

    # 阈值类型名称 -> OpenCV常量 映射
    TYPE_MAP = {
        "二值化": cv2.THRESH_BINARY,
        "反二值化": cv2.THRESH_BINARY_INV,
        "截断": cv2.THRESH_TRUNC,
        "归零": cv2.THRESH_TOZERO,
        "反归零": cv2.THRESH_TOZERO_INV,
    }
    # OpenCV常量 -> 名称 反向映射
    TYPE_NAME_MAP = {v: k for k, v in TYPE_MAP.items()}

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault('threshold_value', 127)
        self.params.setdefault('max_value', 255)
        # 改用字符串存储类型名称，确保JSON序列化兼容
        self.params.setdefault('threshold_type_name', "二值化")
        # 自适应阈值模式（空字符串表示不使用自适应）
        self.params.setdefault('adaptive_method', "")
        self.params.setdefault('adaptive_block_size', 11)
        self.params.setdefault('adaptive_c', 2)

    def _get_threshold_type(self):
        """获取OpenCV阈值类型常量，兼容旧格式（直接存常量值）"""
        type_name = self.params.get('threshold_type_name')
        if type_name and type_name in self.TYPE_MAP:
            return self.TYPE_MAP[type_name]
        # 兼容旧格式：直接存储的OpenCV常量值
        old_type = self.params.get('threshold_type')
        if old_type is not None and old_type in self.TYPE_NAME_MAP:
            # 迁移到新格式
            self.params['threshold_type_name'] = self.TYPE_NAME_MAP[old_type]
            return old_type
        return cv2.THRESH_BINARY

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        adaptive_method = self.params.get('adaptive_method', "")
        
        if adaptive_method:
            # 自适应阈值模式
            block_size = int(self.params.get('adaptive_block_size', 11))
            c = float(self.params.get('adaptive_c', 2))
            
            # 自动修正：block_size 必须为奇数且 >= 3
            if block_size % 2 == 0:
                block_size += 1
                self.params['adaptive_block_size'] = block_size
            if block_size < 3:
                block_size = 3

            if adaptive_method == "gaussian":
                adaptive_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            else:
                adaptive_cv = cv2.ADAPTIVE_THRESH_MEAN_C

            binary = cv2.adaptiveThreshold(gray, 255, adaptive_cv,
                                           cv2.THRESH_BINARY, block_size, c)
            method_name = "高斯" if adaptive_method == "gaussian" else "均值"
            message = f"自适应阈值完成 ({method_name}, 块={block_size})"
        else:
            # 传统阈值模式（含Otsu/Triangle）
            thresh_val = int(self.params.get('threshold_value', 127))
            max_val = int(self.params.get('max_value', 255))
            thresh_type = self._get_threshold_type()
            
            # 自动修正：低阈值 <= 高阈值（仅对非Otsu/Triangle模式）
            if thresh_val > max_val:
                max_val = thresh_val
                self.params['max_value'] = max_val

            _, binary = cv2.threshold(gray, thresh_val, max_val, thresh_type)

            type_name = self.params.get('threshold_type_name', "二值化")
            message = f"阈值分割完成 (阈值:{thresh_val}, 类型:{type_name})"

        context.set_image('binary', binary)

        return ToolResult(
            success=True, passed=True,
            processed_image=cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
            data={"threshold_type": self.params.get('threshold_type_name', "二值化")},
            message=message
        )

    def get_param_widgets(self, parent):
        widgets = []

        # 阈值模式选择：传统 / 自适应
        mode_combo = QComboBox(parent)
        mode_combo.addItem("传统阈值", "standard")
        mode_combo.addItem("自适应阈值", "adaptive")
        current_adaptive = self.params.get('adaptive_method', "")
        mode_combo.setCurrentIndex(1 if current_adaptive else 0)
        widgets.append(("模式:", mode_combo))

        # 使用 QStackedWidget 切换传统/自适应参数
        stacked = QStackedWidget(parent)

        # Page 0: 传统阈值参数
        standard_widget = QWidget()
        standard_layout = QVBoxLayout(standard_widget)
        standard_layout.setContentsMargins(0, 0, 0, 0)

        thresh_sb = QSpinBox()
        thresh_sb.setRange(0, 255)
        thresh_sb.setValue(int(self.params.get('threshold_value', 127)))
        thresh_sb.valueChanged.connect(lambda v: self.params.update({'threshold_value': v}))
        standard_layout.addWidget(self._make_row("阈值:", thresh_sb))

        max_sb = QSpinBox()
        max_sb.setRange(0, 255)
        max_sb.setValue(int(self.params.get('max_value', 255)))
        max_sb.valueChanged.connect(lambda v: self.params.update({'max_value': v}))
        standard_layout.addWidget(self._make_row("最大值:", max_sb))

        type_cb = QComboBox()
        type_cb.addItems(list(self.TYPE_MAP.keys()))
        current_name = self.params.get('threshold_type_name', "二值化")
        idx = type_cb.findText(current_name)
        if idx >= 0:
            type_cb.setCurrentIndex(idx)
        type_cb.currentTextChanged.connect(
            lambda v: self.params.update({'threshold_type_name': v}))
        standard_layout.addWidget(self._make_row("阈值类型:", type_cb))
        standard_layout.addStretch()
        stacked.addWidget(standard_widget)

        # Page 1: 自适应阈值参数
        adaptive_widget = QWidget()
        adaptive_layout = QVBoxLayout(adaptive_widget)
        adaptive_layout.setContentsMargins(0, 0, 0, 0)

        method_cb = QComboBox()
        method_cb.addItem("均值法 (MEAN)", "mean")
        method_cb.addItem("高斯法 (GAUSSIAN)", "gaussian")
        current_method = self.params.get('adaptive_method', "mean")
        if current_method:
            idx = method_cb.findData(current_method)
            if idx >= 0:
                method_cb.setCurrentIndex(idx)
        method_cb.currentIndexChanged.connect(
            lambda i: self.params.update({'adaptive_method': method_cb.itemData(i)}))
        adaptive_layout.addWidget(self._make_row("方法:", method_cb))

        block_sb = QSpinBox()
        block_sb.setRange(3, 99)
        block_sb.setSingleStep(2)
        block_sb.setValue(int(self.params.get('adaptive_block_size', 11)))
        block_sb.valueChanged.connect(lambda v: self.params.update({'adaptive_block_size': v}))
        adaptive_layout.addWidget(self._make_row("块大小:", block_sb))

        c_spin = QDoubleSpinBox()
        c_spin.setRange(-10, 10)
        c_spin.setValue(float(self.params.get('adaptive_c', 2)))
        c_spin.valueChanged.connect(lambda v: self.params.update({'adaptive_c': v}))
        adaptive_layout.addWidget(self._make_row("常数C:", c_spin))
        adaptive_layout.addStretch()
        stacked.addWidget(adaptive_widget)

        def on_mode_changed(idx):
            mode = mode_combo.itemData(idx)
            stacked.setCurrentIndex(idx)
            if mode == "adaptive":
                # 切换到自适应模式时，设置默认方法
                if not self.params.get('adaptive_method', ""):
                    self.params['adaptive_method'] = "mean"
            else:
                # 切换到传统模式时，清除自适应方法标记
                self.params['adaptive_method'] = ""

        mode_combo.currentIndexChanged.connect(on_mode_changed)
        stacked.setCurrentIndex(1 if current_adaptive else 0)
        widgets.append(("参数:", stacked))

        return widgets

    def _make_row(self, label_text, widget):
        """创建带标签的水平布局行"""
        from PyQt5.QtWidgets import QHBoxLayout
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label_text))
        layout.addWidget(widget)
        layout.addStretch()
        return row


class ContourAnalysis(VisionTool):
    display_name = "轮廓分析"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault('mode', cv2.RETR_EXTERNAL)
        self.params.setdefault('method', cv2.CHAIN_APPROX_SIMPLE)
        self.params.setdefault('min_area', 10)
        self.params.setdefault('max_area', 100000)
        self.params.setdefault('sort_by', 'area')
        self.params.setdefault('sort_descending', True)
        self.params.setdefault('max_count', 10)
        self.params.setdefault('use_existing_binary', True)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if img.shape[-1] == 3:
            display = img.copy()
        else:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 尝试复用上游的二值图，避免重复阈值计算
        use_existing = self.params.get('use_existing_binary', True)
        binary = context.get_image('binary')
        if binary is None or not use_existing:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, self.params['mode'], self.params['method'])

        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.params['min_area'] <= area <= self.params['max_area']:
                filtered.append(cnt)

        sort_by = self.params['sort_by']
        descending = self.params['sort_descending']
        if sort_by == 'area':
            filtered.sort(key=cv2.contourArea, reverse=descending)
        elif sort_by == 'perimeter':
            filtered.sort(key=cv2.arcLength, reverse=descending)
        elif sort_by == 'x':
            filtered.sort(key=lambda c: cv2.boundingRect(c)[0], reverse=descending)
        elif sort_by == 'y':
            filtered.sort(key=lambda c: cv2.boundingRect(c)[1], reverse=descending)
        elif sort_by == 'width':
            filtered.sort(key=lambda c: cv2.boundingRect(c)[2], reverse=descending)
        elif sort_by == 'height':
            filtered.sort(key=lambda c: cv2.boundingRect(c)[3], reverse=descending)

        max_count = self.params['max_count']
        filtered = filtered[:max_count]

        cv2.drawContours(display, filtered, -1, (0, 255, 0), 2)

        overlay = np.zeros_like(img) if img.shape[-1] == 3 else np.zeros((*img.shape, 3), dtype=np.uint8)
        cv2.drawContours(overlay, filtered, -1, (0, 255, 0), 2)

        for i, cnt in enumerate(filtered):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            moments = cv2.moments(cnt)
            cx = int(moments['m10'] / (moments['m00'] + 1e-6))
            cy = int(moments['m01'] / (moments['m00'] + 1e-6))
            cv2.putText(display, f"#{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(overlay, f"#{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        result_data = {
            'contour_count': len(filtered),
            'contours': filtered,
            'areas': [cv2.contourArea(c) for c in filtered],
            'perimeters': [cv2.arcLength(c, True) for c in filtered],
        }

        context.set_data('contour_data', result_data)

        return ToolResult(
            success=True, passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={"contour_count": len(filtered)},
            message=f"找到 {len(filtered)} 个轮廓"
        )

    def get_param_widgets(self, parent):
        widgets = []
        mode_cb = QComboBox(parent)
        mode_map = {
            "只检测外轮廓": cv2.RETR_EXTERNAL,
            "列表": cv2.RETR_LIST,
            "树形": cv2.RETR_TREE,
        }
        current_mode = "只检测外轮廓"
        for name, val in mode_map.items():
            if val == self.params['mode']:
                current_mode = name
                break
        mode_cb.addItems(mode_map.keys())
        mode_cb.setCurrentText(current_mode)
        mode_cb.currentTextChanged.connect(lambda v: self.params.update({'mode': mode_map[v]}))
        widgets.append(("轮廓模式", mode_cb))

        method_cb = QComboBox(parent)
        method_map = {
            "简单": cv2.CHAIN_APPROX_SIMPLE,
            "无压缩": cv2.CHAIN_APPROX_NONE,
        }
        current_method = "简单"
        for name, val in method_map.items():
            if val == self.params['method']:
                current_method = name
                break
        method_cb.addItems(method_map.keys())
        method_cb.setCurrentText(current_method)
        method_cb.currentTextChanged.connect(lambda v: self.params.update({'method': method_map[v]}))
        widgets.append(("逼近方法", method_cb))

        min_area_sb = QSpinBox(parent)
        min_area_sb.setRange(1, 999999)
        min_area_sb.setValue(self.params['min_area'])
        min_area_sb.valueChanged.connect(lambda v: self.params.update({'min_area': v}))
        widgets.append(("最小面积", min_area_sb))

        max_area_sb = QSpinBox(parent)
        max_area_sb.setRange(1, 999999)
        max_area_sb.setValue(self.params['max_area'])
        max_area_sb.valueChanged.connect(lambda v: self.params.update({'max_area': v}))
        widgets.append(("最大面积", max_area_sb))

        sort_cb = QComboBox(parent)
        sort_cb.addItems(['area', 'perimeter', 'x', 'y', 'width', 'height'])
        sort_cb.setCurrentText(self.params['sort_by'])
        sort_cb.currentTextChanged.connect(lambda v: self.params.update({'sort_by': v}))
        widgets.append(("排序依据", sort_cb))

        desc_cb = QCheckBox(parent)
        desc_cb.setChecked(self.params['sort_descending'])
        desc_cb.stateChanged.connect(lambda v: self.params.update({'sort_descending': bool(v)}))
        widgets.append(("降序", desc_cb))

        max_sb = QSpinBox(parent)
        max_sb.setRange(1, 100)
        max_sb.setValue(self.params['max_count'])
        max_sb.valueChanged.connect(lambda v: self.params.update({'max_count': v}))
        widgets.append(("最大数量", max_sb))

        use_bin_cb = QCheckBox(parent)
        use_bin_cb.setChecked(self.params.get('use_existing_binary', True))
        use_bin_cb.stateChanged.connect(lambda v: self.params.update({'use_existing_binary': bool(v)}))
        widgets.append(("复用上游二值图", use_bin_cb))

        return widgets


class BlobDetection(VisionTool):
    display_name = "斑点检测"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault('min_area', 100)
        self.params.setdefault('max_area', 5000)
        self.params.setdefault('min_circularity', 0.1)
        self.params.setdefault('min_convexity', 0.5)
        self.params.setdefault('min_inertia_ratio', 0.1)
        self.params.setdefault('max_count', 50)
        self.params.setdefault('filter_by_color', False)
        self.params.setdefault('blob_color', 0)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if img.shape[-1] == 3:
            display = img.copy()
        else:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.params['min_area']
        params.maxArea = self.params['max_area']
        params.filterByCircularity = True
        params.minCircularity = self.params['min_circularity']
        params.filterByConvexity = True
        params.minConvexity = self.params['min_convexity']
        params.filterByInertia = True
        params.minInertiaRatio = self.params['min_inertia_ratio']

        if self.params['filter_by_color']:
            params.filterByColor = True
            params.blobColor = self.params['blob_color']

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        max_count = self.params['max_count']
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:max_count]

        cv2.drawKeypoints(display, keypoints, display, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        overlay = np.zeros_like(display)
        cv2.drawKeypoints(overlay, keypoints, overlay, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        blob_data = []
        for kp in keypoints:
            blob_data.append({
                'x': kp.pt[0],
                'y': kp.pt[1],
                'size': kp.size,
                'angle': kp.angle,
            })

        context.set_data('blob_data', blob_data)

        return ToolResult(
            success=True, passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={"blob_count": len(keypoints)},
            message=f"检测到 {len(keypoints)} 个斑点"
        )

    def get_param_widgets(self, parent):
        widgets = []
        min_area_sb = QSpinBox(parent)
        min_area_sb.setRange(1, 999999)
        min_area_sb.setValue(self.params['min_area'])
        min_area_sb.valueChanged.connect(lambda v: self.params.update({'min_area': v}))
        widgets.append(("最小面积", min_area_sb))

        max_area_sb = QSpinBox(parent)
        max_area_sb.setRange(1, 999999)
        max_area_sb.setValue(self.params['max_area'])
        max_area_sb.valueChanged.connect(lambda v: self.params.update({'max_area': v}))
        widgets.append(("最大面积", max_area_sb))

        min_circ = QDoubleSpinBox(parent)
        min_circ.setRange(0.0, 1.0)
        min_circ.setSingleStep(0.05)
        min_circ.setValue(self.params['min_circularity'])
        min_circ.valueChanged.connect(lambda v: self.params.update({'min_circularity': v}))
        widgets.append(("最小圆度", min_circ))

        min_conv = QDoubleSpinBox(parent)
        min_conv.setRange(0.0, 1.0)
        min_conv.setSingleStep(0.05)
        min_conv.setValue(self.params['min_convexity'])
        min_conv.valueChanged.connect(lambda v: self.params.update({'min_convexity': v}))
        widgets.append(("最小凸度", min_conv))

        min_inertia = QDoubleSpinBox(parent)
        min_inertia.setRange(0.0, 1.0)
        min_inertia.setSingleStep(0.05)
        min_inertia.setValue(self.params['min_inertia_ratio'])
        min_inertia.valueChanged.connect(lambda v: self.params.update({'min_inertia_ratio': v}))
        widgets.append(("最小惯性比", min_inertia))

        max_sb = QSpinBox(parent)
        max_sb.setRange(1, 200)
        max_sb.setValue(self.params['max_count'])
        max_sb.valueChanged.connect(lambda v: self.params.update({'max_count': v}))
        widgets.append(("最大数量", max_sb))

        return widgets


class ContourFilter(VisionTool):
    display_name = "轮廓筛选"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault('min_area', 10)
        self.params.setdefault('max_area', 100000)
        self.params.setdefault('min_perimeter', 0)
        self.params.setdefault('max_perimeter', 10000)
        self.params.setdefault('min_width', 0)
        self.params.setdefault('max_width', 10000)
        self.params.setdefault('min_height', 0)
        self.params.setdefault('max_height', 10000)
        self.params.setdefault('min_aspect_ratio', 0.0)
        self.params.setdefault('max_aspect_ratio', 100.0)
        self.params.setdefault('max_count', 20)
        self.params.setdefault('logic_operator', 'AND')  # AND / OR

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        contour_data = context.get_data('contour_data')
        if contour_data is None:
            return ToolResult(success=False, passed=False, message="请先运行轮廓分析工具")

        contours = contour_data.get('contours', [])
        if not contours:
            return ToolResult(success=False, passed=False, message="无轮廓可筛选")

        if img.shape[-1] == 3:
            display = img.copy()
        else:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        logic_op = self.params.get('logic_operator', 'AND')

        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0

            # 各条件判断
            cond_area = self.params['min_area'] <= area <= self.params['max_area']
            cond_perimeter = self.params['min_perimeter'] <= perimeter <= self.params['max_perimeter']
            cond_width = self.params['min_width'] <= w <= self.params['max_width']
            cond_height = self.params['min_height'] <= h <= self.params['max_height']
            cond_aspect = self.params['min_aspect_ratio'] <= aspect_ratio <= self.params['max_aspect_ratio']

            if logic_op == 'AND':
                if cond_area and cond_perimeter and cond_width and cond_height and cond_aspect:
                    filtered.append(cnt)
            else:  # OR
                if cond_area or cond_perimeter or cond_width or cond_height or cond_aspect:
                    filtered.append(cnt)

        max_count = self.params['max_count']
        filtered = filtered[:max_count]

        cv2.drawContours(display, filtered, -1, (0, 255, 255), 2)

        overlay = np.zeros_like(display)
        cv2.drawContours(overlay, filtered, -1, (0, 255, 255), 2)
        for i, cnt in enumerate(filtered):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(display, f"#{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(overlay, f"#{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        filtered_data = {
            'contour_count': len(filtered),
            'contours': filtered,
            'areas': [cv2.contourArea(c) for c in filtered],
        }
        context.set_data('filtered_contour_data', filtered_data)

        return ToolResult(
            success=True, passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={"contour_count": len(filtered)},
            message=f"筛选后剩余 {len(filtered)} 个轮廓"
        )

    def get_param_widgets(self, parent):
        widgets = []

        # 逻辑运算符选择
        logic_cb = QComboBox(parent)
        logic_cb.addItem("全部满足 (AND)", "AND")
        logic_cb.addItem("任一满足 (OR)", "OR")
        current_logic = self.params.get('logic_operator', 'AND')
        idx = logic_cb.findData(current_logic)
        if idx >= 0:
            logic_cb.setCurrentIndex(idx)
        logic_cb.currentTextChanged.connect(
            lambda v: self.params.update({'logic_operator': v}))
        widgets.append(("逻辑:", logic_cb))

        min_area_sb = QSpinBox(parent)
        min_area_sb.setRange(0, 999999)
        min_area_sb.setValue(self.params['min_area'])
        min_area_sb.valueChanged.connect(lambda v: self.params.update({'min_area': v}))
        widgets.append(("最小面积", min_area_sb))

        max_area_sb = QSpinBox(parent)
        max_area_sb.setRange(0, 999999)
        max_area_sb.setValue(self.params['max_area'])
        max_area_sb.valueChanged.connect(lambda v: self.params.update({'max_area': v}))
        widgets.append(("最大面积", max_area_sb))

        min_peri = QSpinBox(parent)
        min_peri.setRange(0, 99999)
        min_peri.setValue(self.params['min_perimeter'])
        min_peri.valueChanged.connect(lambda v: self.params.update({'min_perimeter': v}))
        widgets.append(("最小周长", min_peri))

        max_peri = QSpinBox(parent)
        max_peri.setRange(0, 99999)
        max_peri.setValue(self.params['max_perimeter'])
        max_peri.valueChanged.connect(lambda v: self.params.update({'max_perimeter': v}))
        widgets.append(("最大周长", max_peri))

        min_w = QSpinBox(parent)
        min_w.setRange(0, 99999)
        min_w.setValue(self.params['min_width'])
        min_w.valueChanged.connect(lambda v: self.params.update({'min_width': v}))
        widgets.append(("最小宽度", min_w))

        max_w = QSpinBox(parent)
        max_w.setRange(0, 99999)
        max_w.setValue(self.params['max_width'])
        max_w.valueChanged.connect(lambda v: self.params.update({'max_width': v}))
        widgets.append(("最大宽度", max_w))

        min_h = QSpinBox(parent)
        min_h.setRange(0, 99999)
        min_h.setValue(self.params['min_height'])
        min_h.valueChanged.connect(lambda v: self.params.update({'min_height': v}))
        widgets.append(("最小高度", min_h))

        max_h = QSpinBox(parent)
        max_h.setRange(0, 99999)
        max_h.setValue(self.params['max_height'])
        max_h.valueChanged.connect(lambda v: self.params.update({'max_height': v}))
        widgets.append(("最大高度", max_h))

        min_ar = QDoubleSpinBox(parent)
        min_ar.setRange(0.0, 1000.0)
        min_ar.setSingleStep(0.1)
        min_ar.setValue(self.params['min_aspect_ratio'])
        min_ar.valueChanged.connect(lambda v: self.params.update({'min_aspect_ratio': v}))
        widgets.append(("最小宽高比", min_ar))

        max_ar = QDoubleSpinBox(parent)
        max_ar.setRange(0.0, 1000.0)
        max_ar.setSingleStep(0.1)
        max_ar.setValue(self.params['max_aspect_ratio'])
        max_ar.valueChanged.connect(lambda v: self.params.update({'max_aspect_ratio': v}))
        widgets.append(("最大宽高比", max_ar))

        max_sb = QSpinBox(parent)
        max_sb.setRange(1, 100)
        max_sb.setValue(self.params['max_count'])
        max_sb.valueChanged.connect(lambda v: self.params.update({'max_count': v}))
        widgets.append(("最大数量", max_sb))

        return widgets


class LineDetection(VisionTool):
    display_name = "直线检测"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault('rho', 1)
        self.params.setdefault('theta', 1)
        self.params.setdefault('threshold', 100)
        self.params.setdefault('min_line_length', 50)
        self.params.setdefault('max_line_gap', 10)
        self.params.setdefault('canny_low', 50)
        self.params.setdefault('canny_high', 150)
        self.params.setdefault('auto_params', False)  # 自动参数估计

    def _estimate_params(self, img_shape):
        """根据图像尺寸自动估计霍夫变换参数"""
        h, w = img_shape[:2]
        diag = np.sqrt(w**2 + h**2)
        
        # 根据图像对角线长度估算参数
        # 阈值：图像越大，阈值越高（减少噪声）
        estimated_threshold = int(max(50, min(300, diag / 10)))
        estimated_rho = 1 if diag < 1000 else 2
        estimated_theta = 1 if diag < 1000 else 2
        # Canny阈值使用Otsu
        gray = None
        if len(img_shape) == 3:
            gray = cv2.cvtColor(np.zeros((h, w), dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        if gray is not None:
            low, high = CannyEdge._compute_auto_thresholds(gray)
        else:
            low, high = 50, 150
        
        return {
            'threshold': estimated_threshold,
            'rho': estimated_rho,
            'theta': estimated_theta,
            'canny_low': low,
            'canny_high': high,
        }

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 自动参数估计
        if self.params.get('auto_params', False):
            estimated = self._estimate_params(img.shape)
            threshold = estimated['threshold']
            rho = estimated['rho']
            theta = estimated['theta']
            canny_low = estimated['canny_low']
            canny_high = estimated['canny_high']
            param_info = "自动"
        else:
            threshold = int(self.params.get('threshold', 100))
            rho = float(self.params.get('rho', 1))
            theta = float(self.params.get('theta', 1))
            canny_low = int(self.params.get('canny_low', 50))
            canny_high = int(self.params.get('canny_high', 150))
            # 自动修正：低阈值 <= 高阈值
            if canny_low > canny_high:
                canny_high = canny_low
                self.params['canny_high'] = canny_high
            param_info = "手动"

        edges = cv2.Canny(gray, canny_low, canny_high)
        theta_rad = theta * np.pi / 180

        lines = cv2.HoughLinesP(edges, rho, theta_rad, threshold,
                                 minLineLength=int(self.params.get('min_line_length', 50)),
                                 maxLineGap=int(self.params.get('max_line_gap', 10)))

        display = img.copy()
        overlay = np.zeros_like(img)
        line_data = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                line_data.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "length": float(length),
                    "angle": float(angle),
                })

        return ToolResult(
            success=True, passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={
                "line_count": len(line_data),
                "lines": line_data,
                "params_mode": param_info,
            },
            message=f"检测到 {len(line_data)} 条直线 ({param_info})"
        )

    def get_param_widgets(self, parent):
        widgets = []

        auto_cb = QCheckBox(parent)
        auto_cb.setChecked(self.params.get('auto_params', False))
        auto_cb.stateChanged.connect(lambda v: self.params.update({'auto_params': bool(v)}))
        widgets.append(("自动参数", auto_cb))

        thresh_sb = QSpinBox(parent)
        thresh_sb.setRange(1, 1000)
        thresh_sb.setValue(int(self.params.get('threshold', 100)))
        thresh_sb.setEnabled(not self.params.get('auto_params', False))
        thresh_sb.valueChanged.connect(lambda v: self.params.update({'threshold': v}))
        widgets.append(("阈值", thresh_sb))

        rho_sb = QDoubleSpinBox(parent)
        rho_sb.setRange(0.1, 10.0)
        rho_sb.setSingleStep(0.1)
        rho_sb.setValue(float(self.params.get('rho', 1)))
        rho_sb.setEnabled(not self.params.get('auto_params', False))
        rho_sb.valueChanged.connect(lambda v: self.params.update({'rho': v}))
        widgets.append(("rho", rho_sb))

        theta_sb = QDoubleSpinBox(parent)
        theta_sb.setRange(0.1, 10.0)
        theta_sb.setSingleStep(0.1)
        theta_sb.setValue(float(self.params.get('theta', 1)))
        theta_sb.setEnabled(not self.params.get('auto_params', False))
        theta_sb.valueChanged.connect(lambda v: self.params.update({'theta': v}))
        widgets.append(("theta", theta_sb))

        min_len_sb = QSpinBox(parent)
        min_len_sb.setRange(1, 10000)
        min_len_sb.setValue(int(self.params.get('min_line_length', 50)))
        min_len_sb.valueChanged.connect(lambda v: self.params.update({'min_line_length': v}))
        widgets.append(("最小长度", min_len_sb))

        gap_sb = QSpinBox(parent)
        gap_sb.setRange(0, 1000)
        gap_sb.setValue(int(self.params.get('max_line_gap', 10)))
        gap_sb.valueChanged.connect(lambda v: self.params.update({'max_line_gap': v}))
        widgets.append(("最大间隔", gap_sb))

        canny_low_sb = QSpinBox(parent)
        canny_low_sb.setRange(0, 255)
        canny_low_sb.setValue(int(self.params.get('canny_low', 50)))
        canny_low_sb.setEnabled(not self.params.get('auto_params', False))
        canny_low_sb.valueChanged.connect(lambda v: self.params.update({'canny_low': v}))
        widgets.append(("Canny低阈值", canny_low_sb))

        canny_high_sb = QSpinBox(parent)
        canny_high_sb.setRange(0, 255)
        canny_high_sb.setValue(int(self.params.get('canny_high', 150)))
        canny_high_sb.setEnabled(not self.params.get('auto_params', False))
        canny_high_sb.valueChanged.connect(lambda v: self.params.update({'canny_high': v}))
        widgets.append(("Canny高阈值", canny_high_sb))

        # 自动参数复选框联动
        def _on_auto_changed(state):
            enabled = not bool(state)
            thresh_sb.setEnabled(enabled)
            rho_sb.setEnabled(enabled)
            theta_sb.setEnabled(enabled)
            canny_low_sb.setEnabled(enabled)
            canny_high_sb.setEnabled(enabled)
        auto_cb.stateChanged.connect(_on_auto_changed)

        return widgets


class RectangleDetection(VisionTool):
    display_name = "矩形检测"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("min_area", 100)
        self.params.setdefault("max_area", 1000000)
        self.params.setdefault("epsilon", 0.02)
        self.params.setdefault("min_aspect", 0.1)
        self.params.setdefault("max_aspect", 10.0)
        self.params.setdefault("max_count", 20)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 尝试复用上游二值图
        binary = context.get_image('binary')
        if binary is None:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        min_area = float(self.params.get("min_area", 100))
        max_area = float(self.params.get("max_area", 1000000))
        epsilon_ratio = float(self.params.get("epsilon", 0.02))
        min_aspect = float(self.params.get("min_aspect", 0.1))
        max_aspect = float(self.params.get("max_aspect", 10.0))
        max_count = int(self.params.get("max_count", 20))

        rectangles = []
        display = img.copy()
        overlay = np.zeros_like(img)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            epsilon = epsilon_ratio * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / h if h > 0 else 0

                if min_aspect <= aspect <= max_aspect:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    cv2.drawContours(display, [approx], -1, (0, 255, 0), 2)
                    cv2.drawContours(display, [box], -1, (255, 0, 0), 1)
                    cv2.putText(display, f"({w}x{h})", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)
                    cv2.drawContours(overlay, [box], -1, (255, 0, 0), 1)
                    cv2.putText(overlay, f"({w}x{h})", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    rectangles.append({
                        "area": area,
                        "width": w, "height": h,
                        "aspect": aspect,
                        "x": x, "y": y,
                        "angle": float(rect[2]),
                        "vertices": approx.tolist(),
                    })

                    if len(rectangles) >= max_count:
                        break

        return ToolResult(
            success=True, passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={
                "rect_count": len(rectangles),
                "rectangles": rectangles,
            },
            message=f"检测到 {len(rectangles)} 个矩形"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QDoubleSpinBox, QSpinBox, QHBoxLayout,
                                      QWidget, QLabel)

        widgets = []

        min_area = QDoubleSpinBox(parent)
        min_area.setRange(0, 10000000)
        min_area.setValue(float(self.params.get("min_area", 100)))
        min_area.valueChanged.connect(lambda v: self.params.update({"min_area": v}))
        widgets.append(("最小面积:", min_area))

        max_area = QDoubleSpinBox(parent)
        max_area.setRange(0, 10000000)
        max_area.setValue(float(self.params.get("max_area", 1000000)))
        max_area.valueChanged.connect(lambda v: self.params.update({"max_area": v}))
        widgets.append(("最大面积:", max_area))

        min_aspect = QDoubleSpinBox(parent)
        min_aspect.setRange(0, 1000)
        min_aspect.setValue(float(self.params.get("min_aspect", 0.1)))
        min_aspect.valueChanged.connect(lambda v: self.params.update({"min_aspect": v}))
        widgets.append(("最小宽高比:", min_aspect))

        max_aspect = QDoubleSpinBox(parent)
        max_aspect.setRange(0, 1000)
        max_aspect.setValue(float(self.params.get("max_aspect", 10.0)))
        max_aspect.valueChanged.connect(lambda v: self.params.update({"max_aspect": v}))
        widgets.append(("最大宽高比:", max_aspect))

        eps_spin = QDoubleSpinBox(parent)
        eps_spin.setRange(0.001, 0.1)
        eps_spin.setSingleStep(0.001)
        eps_spin.setDecimals(3)
        eps_spin.setValue(float(self.params.get("epsilon", 0.02)))
        eps_spin.valueChanged.connect(lambda v: self.params.update({"epsilon": v}))
        widgets.append(("逼近精度:", eps_spin))

        max_cnt = QSpinBox(parent)
        max_cnt.setRange(1, 200)
        max_cnt.setValue(int(self.params.get("max_count", 20)))
        max_cnt.valueChanged.connect(lambda v: self.params.update({"max_count": v}))
        widgets.append(("最大数量:", max_cnt))

        return widgets
