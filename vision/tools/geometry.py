# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from .base_tool import VisionTool, ToolResult, PipelineContext


class CircleDetection(VisionTool):
    display_name = "圆检测"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("dp", 1.2)
        self.params.setdefault("minDist", 20)
        self.params.setdefault("param1", 100)
        self.params.setdefault("param2", 30)
        self.params.setdefault("minRadius", 10)
        self.params.setdefault("maxRadius", 500)
        self.params.setdefault("auto_params", False)
        # 圆心距限制：过滤掉圆心距离过近的重复圆
        self.params.setdefault("center_distance_limit", 0)  # 0表示不启用

    def _estimate_params(self, img_shape):
        """根据图像尺寸自动估计霍夫圆检测参数"""
        h, w = img_shape[:2]
        diag = np.sqrt(w**2 + h**2)
        
        # dp: 小图像用较小值，大图像用较大值
        estimated_dp = 1.0 if diag < 500 else (1.5 if diag < 1500 else 2.0)
        # minDist: 基于图像对角线
        estimated_minDist = int(max(10, diag / 20))
        # param1 (Canny阈值): 基于图像尺寸
        estimated_param1 = int(max(50, min(200, diag / 10)))
        # param2 (圆心阈值): 基于图像尺寸
        estimated_param2 = int(max(20, min(100, diag / 30)))
        # 半径范围: 基于图像尺寸
        estimated_minRadius = int(max(5, diag / 50))
        estimated_maxRadius = int(min(10000, diag / 3))
        
        return {
            'dp': estimated_dp,
            'minDist': estimated_minDist,
            'param1': estimated_param1,
            'param2': estimated_param2,
            'minRadius': estimated_minRadius,
            'maxRadius': estimated_maxRadius,
        }

    def _filter_by_center_distance(self, circles):
        """根据圆心距过滤重复圆"""
        limit = float(self.params.get("center_distance_limit", 0))
        if limit <= 0 or not circles:
            return circles
        
        filtered = []
        for c in circles:
            x, y, r = c
            too_close = False
            for fx, fy, fr in filtered:
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                if dist < limit:
                    # 保留半径较大的圆
                    if r > fr:
                        filtered.remove([fx, fy, fr])
                    else:
                        too_close = True
                    break
            if not too_close:
                filtered.append([x, y, r])
        return filtered

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)

        if self.params.get('auto_params', False):
            estimated = self._estimate_params(img.shape)
            dp = estimated['dp']
            minDist = estimated['minDist']
            param1 = estimated['param1']
            param2 = estimated['param2']
            minRadius = estimated['minRadius']
            maxRadius = estimated['maxRadius']
            param_info = "自动"
        else:
            dp = float(self.params.get("dp", 1.2))
            minDist = float(self.params.get("minDist", 20))
            param1 = float(self.params.get("param1", 100))
            param2 = float(self.params.get("param2", 30))
            minRadius = int(self.params.get("minRadius", 10))
            maxRadius = int(self.params.get("maxRadius", 500))
            # 自动修正：minRadius <= maxRadius
            if minRadius > maxRadius:
                maxRadius = minRadius + 1
                self.params['maxRadius'] = maxRadius
            param_info = "手动"

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                    param1=param1, param2=param2,
                                    minRadius=minRadius, maxRadius=maxRadius)

        display = img.copy()
        overlay = np.zeros_like(img)
        circle_data = []
        if circles is not None:
            circles_list = np.round(circles[0]).astype("int").tolist()
            # 圆心距过滤
            circles_list = self._filter_by_center_distance(circles_list)
            
            for (x, y, r) in circles_list:
                cv2.circle(display, (x, y), r, (0, 255, 0), 2)
                cv2.circle(display, (x, y), 2, (0, 0, 255), 3)
                cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
                cv2.circle(overlay, (x, y), 2, (0, 0, 255), 3)
                circle_data.append({"x": int(x), "y": int(y), "radius": int(r)})

        return ToolResult(
            success=True,
            passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={
                "circle_count": len(circle_data),
                "circles": circle_data,
                "params_mode": param_info,
            },
            message=f"检测到 {len(circle_data)} 个圆 ({param_info})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QDoubleSpinBox, QSpinBox, QHBoxLayout,
                                      QWidget, QLabel, QCheckBox)

        widgets = []

        auto_cb = QCheckBox(parent)
        auto_cb.setChecked(self.params.get('auto_params', False))
        auto_cb.stateChanged.connect(lambda v: self.params.update({'auto_params': bool(v)}))
        widgets.append(("自动参数", auto_cb))

        dp_spin = QDoubleSpinBox(parent)
        dp_spin.setRange(1.0, 5.0)
        dp_spin.setSingleStep(0.1)
        dp_spin.setValue(float(self.params.get("dp", 1.2)))
        dp_spin.setEnabled(not self.params.get('auto_params', False))
        dp_spin.valueChanged.connect(lambda v: self.params.update({"dp": v}))
        widgets.append(("dp:", dp_spin))

        dist_spin = QSpinBox(parent)
        dist_spin.setRange(1, 1000)
        dist_spin.setValue(int(self.params.get("minDist", 20)))
        dist_spin.setEnabled(not self.params.get('auto_params', False))
        dist_spin.valueChanged.connect(lambda v: self.params.update({"minDist": v}))
        widgets.append(("最小距离:", dist_spin))

        p1_spin = QSpinBox(parent)
        p1_spin.setRange(1, 500)
        p1_spin.setValue(int(self.params.get("param1", 100)))
        p1_spin.setEnabled(not self.params.get('auto_params', False))
        p1_spin.valueChanged.connect(lambda v: self.params.update({"param1": v}))
        widgets.append(("Canny阈值:", p1_spin))

        p2_spin = QSpinBox(parent)
        p2_spin.setRange(1, 500)
        p2_spin.setValue(int(self.params.get("param2", 30)))
        p2_spin.setEnabled(not self.params.get('auto_params', False))
        p2_spin.valueChanged.connect(lambda v: self.params.update({"param2": v}))
        widgets.append(("圆心阈值:", p2_spin))

        min_r = QSpinBox(parent)
        min_r.setRange(1, 10000)
        min_r.setValue(int(self.params.get("minRadius", 10)))
        min_r.setEnabled(not self.params.get('auto_params', False))
        min_r.valueChanged.connect(lambda v: self.params.update({"minRadius": v}))
        widgets.append(("最小半径:", min_r))

        max_r = QSpinBox(parent)
        max_r.setRange(1, 10000)
        max_r.setValue(int(self.params.get("maxRadius", 500)))
        max_r.setEnabled(not self.params.get('auto_params', False))
        max_r.valueChanged.connect(lambda v: self.params.update({"maxRadius": v}))
        widgets.append(("最大半径:", max_r))

        # 圆心距限制
        center_dist = QSpinBox(parent)
        center_dist.setRange(0, 1000)
        center_dist.setValue(int(self.params.get("center_distance_limit", 0)))
        center_dist.setToolTip("0表示不启用；大于0时过滤掉圆心距离小于此值的重复圆")
        center_dist.valueChanged.connect(lambda v: self.params.update({"center_distance_limit": v}))
        widgets.append(("圆心距限制:", center_dist))

        # 自动参数复选框联动
        def _on_auto_changed(state):
            enabled = not bool(state)
            dp_spin.setEnabled(enabled)
            dist_spin.setEnabled(enabled)
            p1_spin.setEnabled(enabled)
            p2_spin.setEnabled(enabled)
            min_r.setEnabled(enabled)
            max_r.setEnabled(enabled)
        auto_cb.stateChanged.connect(_on_auto_changed)

        return widgets


class HoughLineDetection(VisionTool):
    display_name = "直线检测(霍夫)"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("threshold", 100)
        self.params.setdefault("rho", 1)
        self.params.setdefault("theta", 1)
        self.params.setdefault("canny_low", 50)
        self.params.setdefault("canny_high", 150)
        self.params.setdefault("auto_params", False)

    def _estimate_params(self, img_shape):
        """根据图像尺寸自动估计霍夫线检测参数"""
        h, w = img_shape[:2]
        diag = np.sqrt(w**2 + h**2)
        
        # 阈值：图像越大，阈值越高
        estimated_threshold = int(max(50, min(300, diag / 10)))
        # rho: 小图像用1，大图像用2
        estimated_rho = 1 if diag < 1000 else 2
        # theta: 小图像用1度，大图像用1度（角度精度不变）
        estimated_theta = 1
        
        return {
            'threshold': estimated_threshold,
            'rho': estimated_rho,
            'theta': estimated_theta,
        }

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if self.params.get('auto_params', False):
            estimated = self._estimate_params(img.shape)
            threshold = estimated['threshold']
            rho = estimated['rho']
            theta = estimated['theta']
            canny_low = 50
            canny_high = 150
            param_info = "自动"
        else:
            threshold = int(self.params.get("threshold", 100))
            rho = float(self.params.get("rho", 1))
            theta = float(self.params.get("theta", 1))
            canny_low = int(self.params.get("canny_low", 50))
            canny_high = int(self.params.get("canny_high", 150))
            # 自动修正：低阈值 <= 高阈值
            if canny_low > canny_high:
                canny_high = canny_low
                self.params['canny_high'] = canny_high
            param_info = "手动"

        edges = cv2.Canny(gray, canny_low, canny_high)
        theta_rad = theta * np.pi / 180

        lines = cv2.HoughLines(edges, rho, theta_rad, threshold)

        display = img.copy()
        overlay = np.zeros_like(img)
        line_data = []
        if lines is not None:
            for line in lines:
                rho_val, theta_val = line[0]
                a = np.cos(theta_val)
                b = np.sin(theta_val)
                x0 = a * rho_val
                y0 = b * rho_val
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                line_data.append({
                    "rho": float(rho_val),
                    "theta": float(theta_val),
                    "angle_deg": float(np.degrees(theta_val)),
                })

        return ToolResult(
            success=True,
            passed=True,
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
        from PyQt5.QtWidgets import (QSpinBox, QDoubleSpinBox, QHBoxLayout,
                                      QWidget, QLabel, QCheckBox)

        widgets = []

        auto_cb = QCheckBox(parent)
        auto_cb.setChecked(self.params.get('auto_params', False))
        auto_cb.stateChanged.connect(lambda v: self.params.update({'auto_params': bool(v)}))
        widgets.append(("自动参数", auto_cb))

        canny_low = QSpinBox(parent)
        canny_low.setRange(0, 255)
        canny_low.setValue(int(self.params.get("canny_low", 50)))
        canny_low.setEnabled(not self.params.get('auto_params', False))
        canny_low.valueChanged.connect(lambda v: self.params.update({"canny_low": v}))
        widgets.append(("Canny低阈值:", canny_low))

        canny_high = QSpinBox(parent)
        canny_high.setRange(0, 255)
        canny_high.setValue(int(self.params.get("canny_high", 150)))
        canny_high.setEnabled(not self.params.get('auto_params', False))
        canny_high.valueChanged.connect(lambda v: self.params.update({"canny_high": v}))
        widgets.append(("Canny高阈值:", canny_high))

        thresh_spin = QSpinBox(parent)
        thresh_spin.setRange(1, 1000)
        thresh_spin.setValue(int(self.params.get("threshold", 100)))
        thresh_spin.setEnabled(not self.params.get('auto_params', False))
        thresh_spin.valueChanged.connect(
            lambda v: self.params.update({"threshold": v}))
        widgets.append(("阈值:", thresh_spin))

        rho_spin = QDoubleSpinBox(parent)
        rho_spin.setRange(0.1, 10)
        rho_spin.setSingleStep(0.1)
        rho_spin.setValue(float(self.params.get("rho", 1)))
        rho_spin.setEnabled(not self.params.get('auto_params', False))
        rho_spin.valueChanged.connect(lambda v: self.params.update({"rho": v}))
        widgets.append(("rho:", rho_spin))

        theta_spin = QDoubleSpinBox(parent)
        theta_spin.setRange(0.1, 10)
        theta_spin.setSingleStep(0.1)
        theta_spin.setValue(float(self.params.get("theta", 1)))
        theta_spin.setEnabled(not self.params.get('auto_params', False))
        theta_spin.valueChanged.connect(lambda v: self.params.update({"theta": v}))
        widgets.append(("theta:", theta_spin))

        # 自动参数复选框联动
        def _on_auto_changed(state):
            enabled = not bool(state)
            canny_low.setEnabled(enabled)
            canny_high.setEnabled(enabled)
            thresh_spin.setEnabled(enabled)
            rho_spin.setEnabled(enabled)
            theta_spin.setEnabled(enabled)
        auto_cb.stateChanged.connect(_on_auto_changed)

        return widgets


class ContourRectDetection(VisionTool):
    display_name = "矩形检测(轮廓)"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("min_area", 100)
        self.params.setdefault("max_area", 1000000)
        self.params.setdefault("epsilon", 0.02)
        self.params.setdefault("min_aspect", 0.1)
        self.params.setdefault("max_aspect", 10.0)

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

        return ToolResult(
            success=True,
            passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={
                "rect_count": len(rectangles),
                "rectangles": rectangles,
            },
            message=f"检测到 {len(rectangles)} 个矩形"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QWidget, QLabel)

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

        return widgets


class SimpleBlobDetect(VisionTool):
    display_name = "Blob检测(简单)"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("min_area", 100)
        self.params.setdefault("max_area", 100000)
        self.params.setdefault("min_circularity", 0.1)
        self.params.setdefault("max_circularity", 1.0)
        self.params.setdefault("min_convexity", 0.1)
        self.params.setdefault("max_convexity", 1.0)
        self.params.setdefault("min_inertia_ratio", 0.1)
        self.params.setdefault("max_inertia_ratio", 1.0)
        self.params.setdefault("max_count", 50)
        self.params.setdefault("filter_by_color", False)
        self.params.setdefault("blob_color", 0)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        # SimpleBlobDetector 需要灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = float(self.params.get("min_area", 100))
        params.maxArea = float(self.params.get("max_area", 100000))

        params.filterByCircularity = True
        params.minCircularity = float(self.params.get("min_circularity", 0.1))
        params.maxCircularity = float(self.params.get("max_circularity", 1.0))

        params.filterByConvexity = True
        params.minConvexity = float(self.params.get("min_convexity", 0.1))
        params.maxConvexity = float(self.params.get("max_convexity", 1.0))

        params.filterByInertia = True
        params.minInertiaRatio = float(self.params.get("min_inertia_ratio", 0.1))
        params.maxInertiaRatio = float(self.params.get("max_inertia_ratio", 1.0))

        if self.params.get('filter_by_color', False):
            params.filterByColor = True
            params.blobColor = int(self.params.get('blob_color', 0))

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        max_count = int(self.params.get("max_count", 50))
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:max_count]

        display = img.copy()
        overlay = np.zeros_like(img)
        blob_data = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            cv2.circle(display, (x, y), r, (0, 255, 0), 2)
            cv2.circle(display, (x, y), 2, (0, 0, 255), -1)
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
            cv2.circle(overlay, (x, y), 2, (0, 0, 255), -1)
            blob_data.append({
                "x": x, "y": y,
                "radius": r,
                "size": kp.size,
                "response": kp.response,
            })

        return ToolResult(
            success=True,
            passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={
                "blob_count": len(blob_data),
                "blobs": blob_data,
            },
            message=f"检测到 {len(blob_data)} 个Blob"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QWidget, QLabel, QCheckBox, QSpinBox)

        widgets = []

        min_area = QDoubleSpinBox(parent)
        min_area.setRange(0, 1000000)
        min_area.setValue(float(self.params.get("min_area", 100)))
        min_area.valueChanged.connect(lambda v: self.params.update({"min_area": v}))
        widgets.append(("最小面积:", min_area))

        max_area = QDoubleSpinBox(parent)
        max_area.setRange(0, 10000000)
        max_area.setValue(float(self.params.get("max_area", 100000)))
        max_area.valueChanged.connect(lambda v: self.params.update({"max_area": v}))
        widgets.append(("最大面积:", max_area))

        min_circ = QDoubleSpinBox(parent)
        min_circ.setRange(0, 1)
        min_circ.setSingleStep(0.05)
        min_circ.setValue(float(self.params.get("min_circularity", 0.1)))
        min_circ.valueChanged.connect(lambda v: self.params.update({"min_circularity": v}))
        widgets.append(("最小圆度:", min_circ))

        max_circ = QDoubleSpinBox(parent)
        max_circ.setRange(0, 1)
        max_circ.setSingleStep(0.05)
        max_circ.setValue(float(self.params.get("max_circularity", 1.0)))
        max_circ.valueChanged.connect(lambda v: self.params.update({"max_circularity": v}))
        widgets.append(("最大圆度:", max_circ))

        min_conv = QDoubleSpinBox(parent)
        min_conv.setRange(0, 1)
        min_conv.setSingleStep(0.05)
        min_conv.setValue(float(self.params.get("min_convexity", 0.1)))
        min_conv.valueChanged.connect(lambda v: self.params.update({"min_convexity": v}))
        widgets.append(("最小凸度:", min_conv))

        max_conv = QDoubleSpinBox(parent)
        max_conv.setRange(0, 1)
        max_conv.setSingleStep(0.05)
        max_conv.setValue(float(self.params.get("max_convexity", 1.0)))
        max_conv.valueChanged.connect(lambda v: self.params.update({"max_convexity": v}))
        widgets.append(("最大凸度:", max_conv))

        min_inertia = QDoubleSpinBox(parent)
        min_inertia.setRange(0, 1)
        min_inertia.setSingleStep(0.05)
        min_inertia.setValue(float(self.params.get("min_inertia_ratio", 0.1)))
        min_inertia.valueChanged.connect(lambda v: self.params.update({"min_inertia_ratio": v}))
        widgets.append(("最小惯性比:", min_inertia))

        max_inertia = QDoubleSpinBox(parent)
        max_inertia.setRange(0, 1)
        max_inertia.setSingleStep(0.05)
        max_inertia.setValue(float(self.params.get("max_inertia_ratio", 1.0)))
        max_inertia.valueChanged.connect(lambda v: self.params.update({"max_inertia_ratio": v}))
        widgets.append(("最大惯性比:", max_inertia))

        max_cnt = QSpinBox(parent)
        max_cnt.setRange(1, 200)
        max_cnt.setValue(int(self.params.get("max_count", 50)))
        max_cnt.valueChanged.connect(lambda v: self.params.update({"max_count": v}))
        widgets.append(("最大数量:", max_cnt))

        return widgets
