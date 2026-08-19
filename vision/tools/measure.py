# -*- coding: utf-8 -*-
"""面积测量工具、圆检测、直线检测、矩形检测等几何特征的测量和分析。"""


from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from .base_tool import VisionTool, ToolResult, PipelineContext


class AreaMeasure(VisionTool):
    display_name = "面积测量"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("min_area", 10)
        self.params.setdefault("max_area", 1000000)
        self.params.setdefault("pass_min", 0)
        self.params.setdefault("pass_max", 1000000)

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

        min_area = float(self.params.get("min_area", 10))
        max_area = float(self.params.get("max_area", 1000000))
        pass_min = float(self.params.get("pass_min", 0))
        pass_max = float(self.params.get("pass_max", 1000000))

        areas = []
        display = img.copy()
        overlay = np.zeros_like(img)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                areas.append(area)
                cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(display, f"{area:.1f}", (cx-20, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(overlay, f"{area:.1f}", (cx-20, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        total_area = sum(areas)
        max_area_val = max(areas) if areas else 0

        passed = pass_min <= total_area <= pass_max

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=display,
            overlay_image=overlay,
            data={
                "total_area": total_area,
                "max_area": max_area_val,
                "count": len(areas),
                "areas": areas,
            },
            message=f"总面积={total_area:.1f}, 最大={max_area_val:.1f}, 数量={len(areas)}"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QWidget, QLabel)

        widgets = []

        min_area = QDoubleSpinBox(parent)
        min_area.setRange(0, 10000000)
        min_area.setValue(float(self.params.get("min_area", 10)))
        min_area.valueChanged.connect(lambda v: self.params.update({"min_area": v}))
        widgets.append(("最小面积:", min_area))

        max_area = QDoubleSpinBox(parent)
        max_area.setRange(0, 10000000)
        max_area.setValue(float(self.params.get("max_area", 1000000)))
        max_area.valueChanged.connect(lambda v: self.params.update({"max_area": v}))
        widgets.append(("最大面积:", max_area))

        pass_min = QDoubleSpinBox(parent)
        pass_min.setRange(0, 10000000)
        pass_min.setValue(float(self.params.get("pass_min", 0)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QDoubleSpinBox(parent)
        pass_max.setRange(0, 10000000)
        pass_max.setValue(float(self.params.get("pass_max", 1000000)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets


class DistanceMeasure(VisionTool):
    display_name = "距离测量"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("mode", "contour_center")
        self.params.setdefault("ref_x", 0)
        self.params.setdefault("ref_y", 0)
        self.params.setdefault("pass_min", 0)
        self.params.setdefault("pass_max", 1000)

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

        mode = self.params.get("mode", "contour_center")
        ref_x = float(self.params.get("ref_x", 0))
        ref_y = float(self.params.get("ref_y", 0))
        pass_min = float(self.params.get("pass_min", 0))
        pass_max = float(self.params.get("pass_max", 1000))

        display = img.copy()
        overlay = np.zeros_like(img)
        distances = []

        if mode == "contour_center":
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = np.sqrt((cx - ref_x)**2 + (cy - ref_y)**2)
                    distances.append(dist)
                    cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)
                    cv2.line(display, (int(ref_x), int(ref_y)), (cx, cy), (255, 0, 0), 1)
                    cv2.putText(display, f"{dist:.1f}", (cx, cy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
                    cv2.line(overlay, (int(ref_x), int(ref_y)), (cx, cy), (255, 0, 0), 1)
                    cv2.putText(overlay, f"{dist:.1f}", (cx, cy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        avg_dist = np.mean(distances) if distances else 0
        passed = pass_min <= avg_dist <= pass_max

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=display,
            overlay_image=overlay,
            data={
                "distance": float(avg_dist),
                "count": len(distances),
                "distances": [float(d) for d in distances],
            },
            message=f"平均距离={avg_dist:.1f}"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout,
                                      QWidget, QLabel)

        widgets = []

        mode_combo = QComboBox(parent)
        mode_combo.addItem("轮廓中心距离", "contour_center")
        current_mode = self.params.get("mode", "contour_center")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"mode": mode_combo.itemData(i)}))
        widgets.append(("模式:", mode_combo))

        ref_x = QDoubleSpinBox(parent)
        ref_x.setRange(0, 100000)
        ref_x.setValue(float(self.params.get("ref_x", 0)))
        ref_x.valueChanged.connect(lambda v: self.params.update({"ref_x": v}))
        widgets.append(("参考X:", ref_x))

        ref_y = QDoubleSpinBox(parent)
        ref_y.setRange(0, 100000)
        ref_y.setValue(float(self.params.get("ref_y", 0)))
        ref_y.valueChanged.connect(lambda v: self.params.update({"ref_y": v}))
        widgets.append(("参考Y:", ref_y))

        pass_min = QDoubleSpinBox(parent)
        pass_min.setRange(0, 100000)
        pass_min.setValue(float(self.params.get("pass_min", 0)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QDoubleSpinBox(parent)
        pass_max.setRange(0, 100000)
        pass_max.setValue(float(self.params.get("pass_max", 1000)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets


class PointMeasure(VisionTool):
    display_name = "点测量"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("mode", "contour_center")
        self.params.setdefault("max_corners", 10)
        self.params.setdefault("quality_level", 0.01)
        self.params.setdefault("min_distance", 10)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")
        mode = self.params.get("mode", "contour_center")

        display = img.copy()
        overlay = np.zeros_like(img)
        points = []

        if mode == "contour_center":
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
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append({"x": cx, "y": cy})
                    cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(display, f"({cx},{cy})", (cx+5, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(overlay, f"({cx},{cy})", (cx+5, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        elif mode == "corner":
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=int(self.params.get("max_corners", 10)),
                qualityLevel=float(self.params.get("quality_level", 0.01)),
                minDistance=int(self.params.get("min_distance", 10))
            )
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    x, y = int(x), int(y)
                    points.append({"x": x, "y": y})
                    cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
                    cv2.putText(display, f"({x},{y})", (x+5, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
                    cv2.putText(overlay, f"({x},{y})", (x+5, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        elif mode == "blob_center":
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            params = cv2.SimpleBlobDetector_Params()
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                points.append({"x": x, "y": y})
                cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(display, f"({x},{y})", (x+5, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(overlay, f"({x},{y})", (x+5, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return ToolResult(
            success=True,
            passed=True,
            processed_image=display,
            overlay_image=overlay,
            data={
                "point_count": len(points),
                "points": points,
            },
            message=f"检测到 {len(points)} 个点"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QSpinBox, QDoubleSpinBox,
                                      QHBoxLayout, QWidget, QLabel)

        widgets = []

        mode_combo = QComboBox(parent)
        mode_combo.addItem("轮廓中心", "contour_center")
        mode_combo.addItem("角点", "corner")
        mode_combo.addItem("Blob中心", "blob_center")
        current_mode = self.params.get("mode", "contour_center")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"mode": mode_combo.itemData(i)}))
        widgets.append(("模式:", mode_combo))

        max_corners = QSpinBox(parent)
        max_corners.setRange(1, 100)
        max_corners.setValue(int(self.params.get("max_corners", 10)))
        max_corners.valueChanged.connect(lambda v: self.params.update({"max_corners": v}))
        widgets.append(("最大角点:", max_corners))

        quality = QDoubleSpinBox(parent)
        quality.setRange(0.001, 1.0)
        quality.setSingleStep(0.01)
        quality.setDecimals(3)
        quality.setValue(float(self.params.get("quality_level", 0.01)))
        quality.valueChanged.connect(lambda v: self.params.update({"quality_level": v}))
        widgets.append(("质量等级:", quality))

        min_dist = QSpinBox(parent)
        min_dist.setRange(1, 100)
        min_dist.setValue(int(self.params.get("min_distance", 10)))
        min_dist.valueChanged.connect(lambda v: self.params.update({"min_distance": v}))
        widgets.append(("最小距离:", min_dist))

        return widgets


class LineMeasure(VisionTool):
    display_name = "线测量"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("mode", "hough")
        self.params.setdefault("threshold", 50)
        self.params.setdefault("min_length", 30)
        self.params.setdefault("max_gap", 10)
        self.params.setdefault("canny_low", 50)
        self.params.setdefault("canny_high", 150)
        self.params.setdefault("pass_min", 0)
        self.params.setdefault("pass_max", 10000)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")
        mode = self.params.get("mode", "hough")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        edges = cv2.Canny(gray, self.params["canny_low"], self.params["canny_high"])

        display = img.copy()
        overlay = np.zeros_like(img)
        lines_data = []

        if mode == "hough":
            threshold = int(self.params.get("threshold", 50))
            min_length = int(self.params.get("min_length", 30))
            max_gap = int(self.params.get("max_gap", 10))

            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                     threshold=threshold,
                                     minLineLength=min_length,
                                     maxLineGap=max_gap)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    lines_data.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "length": float(length),
                        "angle": float(angle),
                    })
                    cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, f"{length:.1f}", ((x1+x2)//2, (y1+y2)//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(overlay, f"{length:.1f}", ((x1+x2)//2, (y1+y2)//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        elif mode == "contour":
            binary = context.get_image('binary')
            if binary is None:
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 10:
                    continue
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                w, h = rect[1]
                length = max(w, h)
                angle = rect[2]
                lines_data.append({
                    "length": float(length),
                    "angle": float(angle),
                    "width": float(w),
                    "height": float(h),
                })
                cv2.drawContours(display, [box], -1, (0, 255, 0), 2)
                cv2.drawContours(overlay, [box], -1, (0, 255, 0), 2)

        avg_length = np.mean([d["length"] for d in lines_data]) if lines_data else 0
        pass_min = float(self.params.get("pass_min", 0))
        pass_max = float(self.params.get("pass_max", 10000))
        passed = pass_min <= avg_length <= pass_max

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=display,
            overlay_image=overlay,
            data={
                "line_count": len(lines_data),
                "avg_length": float(avg_length),
                "lines": lines_data,
            },
            message=f"检测到 {len(lines_data)} 条线段, 平均长度={avg_length:.1f}"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QSpinBox, QDoubleSpinBox,
                                      QHBoxLayout, QWidget, QLabel)

        widgets = []

        mode_combo = QComboBox(parent)
        mode_combo.addItem("霍夫变换", "hough")
        mode_combo.addItem("轮廓拟合", "contour")
        current_mode = self.params.get("mode", "hough")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"mode": mode_combo.itemData(i)}))
        widgets.append(("模式:", mode_combo))

        canny_low = QSpinBox(parent)
        canny_low.setRange(0, 255)
        canny_low.setValue(int(self.params.get("canny_low", 50)))
        canny_low.valueChanged.connect(lambda v: self.params.update({"canny_low": v}))
        widgets.append(("Canny低阈值:", canny_low))

        canny_high = QSpinBox(parent)
        canny_high.setRange(0, 255)
        canny_high.setValue(int(self.params.get("canny_high", 150)))
        canny_high.valueChanged.connect(lambda v: self.params.update({"canny_high": v}))
        widgets.append(("Canny高阈值:", canny_high))

        thresh_spin = QSpinBox(parent)
        thresh_spin.setRange(1, 500)
        thresh_spin.setValue(int(self.params.get("threshold", 50)))
        thresh_spin.valueChanged.connect(lambda v: self.params.update({"threshold": v}))
        widgets.append(("阈值:", thresh_spin))

        min_len = QSpinBox(parent)
        min_len.setRange(1, 1000)
        min_len.setValue(int(self.params.get("min_length", 30)))
        min_len.valueChanged.connect(lambda v: self.params.update({"min_length": v}))
        widgets.append(("最小长度:", min_len))

        pass_min = QDoubleSpinBox(parent)
        pass_min.setRange(0, 100000)
        pass_min.setValue(float(self.params.get("pass_min", 0)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QDoubleSpinBox(parent)
        pass_max.setRange(0, 100000)
        pass_max.setValue(float(self.params.get("pass_max", 10000)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets


class AngleMeasure(VisionTool):
    display_name = "角度测量"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("mode", "contour_angle")
        self.params.setdefault("p1_x", 0)
        self.params.setdefault("p1_y", 0)
        self.params.setdefault("p2_x", 100)
        self.params.setdefault("p2_y", 100)
        self.params.setdefault("p3_x", 200)
        self.params.setdefault("p3_y", 0)
        self.params.setdefault("pass_min", 0)
        self.params.setdefault("pass_max", 180)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")
        mode = self.params.get("mode", "contour_angle")

        display = img.copy()
        overlay = np.zeros_like(img)
        angle = 0.0

        if mode == "contour_angle":
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

            angles = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 10:
                    continue
                rect = cv2.minAreaRect(cnt)
                # minAreaRect 返回角度范围 [-90, 0)，归一化到 [0, 180)
                angle_val = rect[2]
                if angle_val < 0:
                    angle_val += 180
                angles.append(angle_val)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(display, [box], -1, (0, 255, 0), 2)
                cv2.drawContours(overlay, [box], -1, (0, 255, 0), 2)
                center = (int(rect[0][0]), int(rect[0][1]))
                cv2.putText(display, f"{angle_val:.1f}°", (center[0]-20, center[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(overlay, f"{angle_val:.1f}°", (center[0]-20, center[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            angle = np.mean(angles) if angles else 0

        elif mode == "three_point":
            p1 = (int(self.params.get("p1_x", 0)), int(self.params.get("p1_y", 0)))
            p2 = (int(self.params.get("p2_x", 100)), int(self.params.get("p2_y", 100)))
            p3 = (int(self.params.get("p3_x", 200)), int(self.params.get("p3_y", 0)))

            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            dot = v1[0] * v2[0] + v1[1] * v2[1]
            norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if norm1 * norm2 > 0:
                cos_angle = dot / (norm1 * norm2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = np.degrees(np.arccos(cos_angle))

            for pt, color, label in [(p1, (0, 0, 255), "P1"),
                                      (p2, (0, 255, 0), "P2"),
                                      (p3, (255, 0, 0), "P3")]:
                cv2.circle(display, pt, 5, color, -1)
                cv2.putText(display, label, (pt[0]+5, pt[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(overlay, pt, 5, color, -1)
                cv2.putText(overlay, label, (pt[0]+5, pt[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.line(display, p2, p1, (255, 255, 0), 1)
            cv2.line(display, p2, p3, (255, 255, 0), 1)
            cv2.line(overlay, p2, p1, (255, 255, 0), 1)
            cv2.line(overlay, p2, p3, (255, 255, 0), 1)
            cv2.putText(display, f"{angle:.1f}°", (p2[0]+10, p2[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(overlay, f"{angle:.1f}°", (p2[0]+10, p2[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        pass_min = float(self.params.get("pass_min", 0))
        pass_max = float(self.params.get("pass_max", 180))
        passed = pass_min <= angle <= pass_max

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=display,
            overlay_image=overlay,
            data={
                "angle": float(angle),
                "mode": mode,
            },
            message=f"角度={angle:.1f}°"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout,
                                      QWidget, QLabel)

        widgets = []

        mode_combo = QComboBox(parent)
        mode_combo.addItem("轮廓角度", "contour_angle")
        mode_combo.addItem("三点角度", "three_point")
        current_mode = self.params.get("mode", "contour_angle")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"mode": mode_combo.itemData(i)}))
        widgets.append(("模式:", mode_combo))

        p1x = QDoubleSpinBox(parent)
        p1x.setRange(0, 100000)
        p1x.setValue(float(self.params.get("p1_x", 0)))
        p1x.valueChanged.connect(lambda v: self.params.update({"p1_x": v}))
        widgets.append(("P1-X:", p1x))

        p1y = QDoubleSpinBox(parent)
        p1y.setRange(0, 100000)
        p1y.setValue(float(self.params.get("p1_y", 0)))
        p1y.valueChanged.connect(lambda v: self.params.update({"p1_y": v}))
        widgets.append(("P1-Y:", p1y))

        p2x = QDoubleSpinBox(parent)
        p2x.setRange(0, 100000)
        p2x.setValue(float(self.params.get("p2_x", 100)))
        p2x.valueChanged.connect(lambda v: self.params.update({"p2_x": v}))
        widgets.append(("P2-X:", p2x))

        p2y = QDoubleSpinBox(parent)
        p2y.setRange(0, 100000)
        p2y.setValue(float(self.params.get("p2_y", 100)))
        p2y.valueChanged.connect(lambda v: self.params.update({"p2_y": v}))
        widgets.append(("P2-Y:", p2y))

        p3x = QDoubleSpinBox(parent)
        p3x.setRange(0, 100000)
        p3x.setValue(float(self.params.get("p3_x", 200)))
        p3x.valueChanged.connect(lambda v: self.params.update({"p3_x": v}))
        widgets.append(("P3-X:", p3x))

        p3y = QDoubleSpinBox(parent)
        p3y.setRange(0, 100000)
        p3y.setValue(float(self.params.get("p3_y", 0)))
        p3y.valueChanged.connect(lambda v: self.params.update({"p3_y": v}))
        widgets.append(("P3-Y:", p3y))

        pass_min = QDoubleSpinBox(parent)
        pass_min.setRange(0, 360)
        pass_min.setValue(float(self.params.get("pass_min", 0)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QDoubleSpinBox(parent)
        pass_max.setRange(0, 360)
        pass_max.setValue(float(self.params.get("pass_max", 180)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets


class ObjectCount(VisionTool):
    display_name = "目标计数"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("min_area", 10)
        self.params.setdefault("max_area", 1000000)
        self.params.setdefault("pass_min", 1)
        self.params.setdefault("pass_max", 1000)

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

        min_area = float(self.params.get("min_area", 10))
        max_area = float(self.params.get("max_area", 1000000))
        pass_min = int(self.params.get("pass_min", 1))
        pass_max = int(self.params.get("pass_max", 1000))

        count = 0
        display = img.copy()
        overlay = np.zeros_like(img)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                count += 1
                cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(display, f"#{count}", (cx-10, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(overlay, f"#{count}", (cx-10, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        passed = pass_min <= count <= pass_max

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=display,
            overlay_image=overlay,
            data={
                "count": count,
                "min_area": min_area,
                "max_area": max_area,
            },
            message=f"目标数量={count} (范围:{pass_min}~{pass_max})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QDoubleSpinBox, QSpinBox, QHBoxLayout,
                                      QWidget, QLabel)

        widgets = []

        min_area = QDoubleSpinBox(parent)
        min_area.setRange(0, 10000000)
        min_area.setValue(float(self.params.get("min_area", 10)))
        min_area.valueChanged.connect(lambda v: self.params.update({"min_area": v}))
        widgets.append(("最小面积:", min_area))

        max_area = QDoubleSpinBox(parent)
        max_area.setRange(0, 10000000)
        max_area.setValue(float(self.params.get("max_area", 1000000)))
        max_area.valueChanged.connect(lambda v: self.params.update({"max_area": v}))
        widgets.append(("最大面积:", max_area))

        pass_min = QSpinBox(parent)
        pass_min.setRange(0, 100000)
        pass_min.setValue(int(self.params.get("pass_min", 1)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QSpinBox(parent)
        pass_max.setRange(0, 100000)
        pass_max.setValue(int(self.params.get("pass_max", 1000)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets


class BrightnessMeasure(VisionTool):
    display_name = "亮度测量"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("pass_min", 0)
        self.params.setdefault("pass_max", 255)

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        # 转为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 计算亮度指标
        mean_gray = float(np.mean(gray))
        std_gray = float(np.std(gray))
        min_gray = int(np.min(gray))
        max_gray = int(np.max(gray))

        # 合格判定（基于平均灰度值）
        pass_min = float(self.params.get("pass_min", 0))
        pass_max = float(self.params.get("pass_max", 255))
        passed = pass_min <= mean_gray <= pass_max

        # 构建显示图像
        display = img.copy()
        overlay = np.zeros_like(img)

        # 在图像上标注亮度数据
        info_lines = [
            f"Mean: {mean_gray:.2f}",
            f"Std:  {std_gray:.2f}",
            f"Min:  {min_gray}",
            f"Max:  {max_gray}",
        ]
        color = (0, 255, 0) if passed else (0, 0, 255)
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 30
            cv2.putText(display, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(overlay, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 使用完整帧作为 processed_image
        output_image = self._full_frame_image if self._full_frame_image is not None else img

        # ROI输入源时，将标注平移到完整帧坐标
        input_source = self.params.get("_input_source", "current")
        if input_source.startswith("region:") and self._full_frame_image is not None:
            overlay_full = np.zeros_like(self._full_frame_image)
            region_name = input_source[7:]
            if region_name in context.regions:
                rx, ry, rw, rh = context.regions[region_name]
                for i, line in enumerate(info_lines):
                    y_pos = 30 + i * 30
                    cv2.putText(overlay_full, line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            overlay = overlay_full

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=output_image,
            overlay_image=overlay,
            data={
                "mean_gray": mean_gray,
                "std_gray": std_gray,
                "min_gray": min_gray,
                "max_gray": max_gray,
            },
            message=f"亮度: Mean={mean_gray:.2f}, Std={std_gray:.2f}, "
                    f"Min={min_gray}, Max={max_gray}"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import QDoubleSpinBox, QHBoxLayout, QWidget, QLabel

        widgets = []

        pass_min = QDoubleSpinBox(parent)
        pass_min.setRange(0, 255)
        pass_min.setDecimals(1)
        pass_min.setSingleStep(1)
        pass_min.setValue(float(self.params.get("pass_min", 0)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QDoubleSpinBox(parent)
        pass_max.setRange(0, 255)
        pass_max.setDecimals(1)
        pass_max.setSingleStep(1)
        pass_max.setValue(float(self.params.get("pass_max", 255)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets
