# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from .base_tool import VisionTool, ToolResult, PipelineContext


class ColorRecognition(VisionTool):
    display_name = "颜色识别"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("h_min", 0)
        self.params.setdefault("s_min", 50)
        self.params.setdefault("v_min", 50)
        self.params.setdefault("h_max", 10)
        self.params.setdefault("s_max", 255)
        self.params.setdefault("v_max", 255)
        self.params.setdefault("color_name", "红色")
        self.params.setdefault("min_area", 100)
        self.params.setdefault("pass_min", 0)
        self.params.setdefault("pass_max", 100)
        # 色彩空间选择: "HSV" / "Lab"
        self.params.setdefault("color_space", "HSV")
        # 区域颜色占比分析
        self.params.setdefault("analyze_regions", False)

    # HSV颜色预设
    HSV_PRESETS = {
        "红色": ([0, 50, 50], [10, 255, 255]),
        "绿色": ([35, 50, 50], [85, 255, 255]),
        "蓝色": ([100, 50, 50], [130, 255, 255]),
        "黄色": ([20, 50, 50], [35, 255, 255]),
        "橙色": ([10, 50, 50], [25, 255, 255]),
        "紫色": ([130, 50, 50], [160, 255, 255]),
        "白色": ([0, 0, 200], [180, 30, 255]),
        "黑色": ([0, 0, 0], [180, 255, 50]),
    }

    # Lab颜色预设（近似值）
    LAB_PRESETS = {
        "红色": ([0, 140, 120], [255, 180, 200]),
        "绿色": ([0, 100, 100], [255, 140, 160]),
        "蓝色": ([0, 120, 100], [255, 160, 150]),
        "黄色": ([0, 100, 150], [255, 140, 200]),
        "橙色": ([0, 130, 140], [255, 170, 200]),
        "紫色": ([0, 120, 100], [255, 160, 150]),
        "白色": ([180, 0, 0], [255, 30, 30]),
        "黑色": ([0, 0, 0], [100, 30, 30]),
    }

    def _update_range_from_color(self):
        color_name = self.params.get("color_name", "红色")
        color_space = self.params.get("color_space", "HSV")
        
        if color_space == "Lab":
            presets = self.LAB_PRESETS
        else:
            presets = self.HSV_PRESETS
            
        if color_name in presets:
            lower, upper = presets[color_name]
            self.params["h_min"], self.params["s_min"], self.params["v_min"] = lower
            self.params["h_max"], self.params["s_max"], self.params["v_max"] = upper

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        # 如果输入是单通道灰度图，转换为3通道BGR（颜色识别需要3通道）
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        color_space = self.params.get("color_space", "HSV")
        
        if color_space == "Lab":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            channel_names = ("L", "a", "b")
            max_vals = (255, 255, 255)
        else:
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            channel_names = ("H", "S", "V")
            max_vals = (180, 255, 255)

        lower = np.array([
            int(self.params.get("h_min", 0)),
            int(self.params.get("s_min", 50)),
            int(self.params.get("v_min", 50))
        ])
        upper = np.array([
            int(self.params.get("h_max", 10)),
            int(self.params.get("s_max", 255)),
            int(self.params.get("v_max", 255))
        ])

        mask = cv2.inRange(converted, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        min_area = float(self.params.get("min_area", 100))
        color_area = np.sum(mask > 0)
        total_area = img.shape[0] * img.shape[1]
        area_ratio = (color_area / total_area) * 100 if total_area > 0 else 0

        display = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        valid_count = 0
        region_data = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                valid_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(display, f"#{valid_count}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                region_data.append({
                    "index": valid_count,
                    "area": float(area),
                    "x": int(x), "y": int(y),
                    "width": int(w), "height": int(h),
                    "area_ratio": float(area / total_area * 100) if total_area > 0 else 0,
                })

        pass_min = float(self.params.get("pass_min", 0))
        pass_max = float(self.params.get("pass_max", 100))
        passed = pass_min <= area_ratio <= pass_max

        result_data = {
            "color_area": int(color_area),
            "area_ratio": float(area_ratio),
            "valid_regions": valid_count,
            "color_name": self.params.get("color_name", "红色"),
            "color_space": color_space,
        }
        if self.params.get("analyze_regions", False):
            result_data["regions"] = region_data

        # 使用完整帧作为 processed_image，确保下游步骤能访问完整图像
        output_image = self._full_frame_image if self._full_frame_image is not None else img

        # 在完整帧的对应位置绘制 overlay 标注
        input_source = self.params.get("_input_source", "current")
        if input_source.startswith("region:") and self._full_frame_image is not None:
            overlay = np.zeros_like(self._full_frame_image)
            region_name = input_source[7:]
            if region_name in context.regions:
                rx, ry, rw, rh = context.regions[region_name]
                # 将 ROI 内的标注绘制到完整帧 overlay 的对应位置
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area >= min_area:
                        # 将轮廓坐标从 ROI 局部坐标转换为完整帧坐标
                        cnt_full = cnt.copy()
                        cnt_full[:, :, 0] += rx
                        cnt_full[:, :, 1] += ry
                        cv2.drawContours(overlay, [cnt_full], -1, (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(cnt_full)
                        cv2.putText(overlay, f"#{valid_count}", (x, y-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            overlay = np.zeros_like(img)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.putText(overlay, f"#{valid_count}", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=output_image,
            overlay_image=overlay,
            data=result_data,
            message=f"颜色区域占比={area_ratio:.1f}% ({color_space})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QSpinBox, QHBoxLayout,
                                      QWidget, QLabel, QSlider, QCheckBox)
        from PyQt5.QtCore import Qt

        widgets = []

        # 色彩空间选择
        space_combo = QComboBox(parent)
        space_combo.addItem("HSV", "HSV")
        space_combo.addItem("Lab", "Lab")
        current_space = self.params.get("color_space", "HSV")
        idx = space_combo.findData(current_space)
        if idx >= 0:
            space_combo.setCurrentIndex(idx)
        space_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"color_space": space_combo.itemData(i)}))
        widgets.append(("色彩空间:", space_combo))

        color_combo = QComboBox(parent)
        colors = ["红色", "绿色", "蓝色", "黄色", "橙色", "紫色", "白色", "黑色"]
        color_combo.addItems(colors)
        current_color = self.params.get("color_name", "红色")
        idx = color_combo.findText(current_color)
        if idx >= 0:
            color_combo.setCurrentIndex(idx)

        def on_color_changed(text):
            self.params["color_name"] = text
            self._update_range_from_color()

        color_combo.currentTextChanged.connect(on_color_changed)
        widgets.append(("颜色:", color_combo))

        def make_slider(label, key, default, min_v=0, max_v=255):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(int(self.params.get(key, default)))
            slider.valueChanged.connect(lambda v: self.params.update({key: v}))
            return slider

        # 根据色彩空间动态显示通道标签
        def get_channel_labels():
            cs = self.params.get("color_space", "HSV")
            if cs == "Lab":
                return ("L:", "a:", "b:")
            return ("H:", "S:", "V:")

        ch = get_channel_labels()
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(ch[0]))
        h_layout.addWidget(make_slider("H_min", "h_min", 0, 0, 255))
        h_layout.addWidget(make_slider("H_max", "h_max", 10, 0, 255))
        h_widget = QWidget()
        h_widget.setLayout(h_layout)
        widgets.append((f"{ch[0]}范围:", h_widget))

        s_layout = QHBoxLayout()
        s_layout.addWidget(QLabel(ch[1]))
        s_layout.addWidget(make_slider("S_min", "s_min", 50))
        s_layout.addWidget(make_slider("S_max", "s_max", 255))
        s_widget = QWidget()
        s_widget.setLayout(s_layout)
        widgets.append((f"{ch[1]}范围:", s_widget))

        v_layout = QHBoxLayout()
        v_layout.addWidget(QLabel(ch[2]))
        v_layout.addWidget(make_slider("V_min", "v_min", 50))
        v_layout.addWidget(make_slider("V_max", "v_max", 255))
        v_widget = QWidget()
        v_widget.setLayout(v_layout)
        widgets.append((f"{ch[2]}范围:", v_widget))

        # 区域占比分析复选框
        region_cb = QCheckBox(parent)
        region_cb.setChecked(self.params.get("analyze_regions", False))
        region_cb.stateChanged.connect(lambda v: self.params.update({"analyze_regions": bool(v)}))
        widgets.append(("区域分析:", region_cb))

        pass_min = QSpinBox(parent)
        pass_min.setRange(0, 100)
        pass_min.setSuffix("%")
        pass_min.setValue(int(self.params.get("pass_min", 0)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QSpinBox(parent)
        pass_max.setRange(0, 100)
        pass_max.setSuffix("%")
        pass_max.setValue(int(self.params.get("pass_max", 100)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets


class TemplateMatch(VisionTool):
    display_name = "模板匹配"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("mode", "standard")
        self.params.setdefault("method", "TM_CCOEFF_NORMED")
        self.params.setdefault("threshold", 0.8)
        self.params.setdefault("template_path", "")
        self.params.setdefault("template_data", None)
        self.params.setdefault("angle_start", -30)
        self.params.setdefault("angle_end", 30)
        self.params.setdefault("angle_step", 5)
        self.params.setdefault("feature_mode", "sift")
        self.params.setdefault("min_matches", 10)
        self.params.setdefault("nms_distance", 20)
        # 掩膜支持
        self.params.setdefault("use_mask", False)
        self.params.setdefault("mask_path", "")
        self._template_cache = None
        self._mask_cache = None

    def set_template(self, template_img):
        self._template_cache = template_img

    def set_mask(self, mask_img):
        self._mask_cache = mask_img

    def _non_max_suppression(self, locations, scores, h, w, min_distance):
        if not locations:
            return []

        indices = np.argsort(scores)[::-1]
        keep = []

        for i in indices:
            should_keep = True
            x1, y1 = locations[i]
            for j in keep:
                x2, y2 = locations[j]
                # 计算两个框的 IoU（交并比）
                # 框1: (x1, y1, x1+w, y1+h)
                # 框2: (x2, y2, x2+w, y2+h)
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w, x2 + w)
                inter_y2 = min(y1 + h, y2 + h)

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                box1_area = w * h
                box2_area = w * h
                union_area = box1_area + box2_area - inter_area

                iou = inter_area / union_area if union_area > 0 else 0

                # 同时检查中心点距离和 IoU，任一条件满足即认为重叠
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < min_distance or iou > 0.3:
                    should_keep = False
                    break
            if should_keep:
                keep.append(i)

        return [(locations[i][0], locations[i][1], scores[i]) for i in keep]

    def _rotate_template(self, template, angle):
        h, w = template.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(template, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        return rotated

    def _rotate_mask(self, mask, angle):
        """旋转掩膜，与模板旋转保持一致"""
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(mask, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        return rotated

    def _multi_angle_match(self, gray_img, template, method, threshold):
        angle_start = float(self.params.get("angle_start", -30))
        angle_end = float(self.params.get("angle_end", 30))
        angle_step = float(self.params.get("angle_step", 5))

        results = []
        score_curve = []  # 每个角度的最佳分数
        th, tw = template.shape[:2]

        use_mask = self.params.get("use_mask", False)
        mask = self._mask_cache

        for angle in np.arange(angle_start, angle_end + angle_step, angle_step):
            rotated = self._rotate_template(template, angle)

            if use_mask and mask is not None:
                rotated_mask = self._rotate_mask(mask, angle)
                result = cv2.matchTemplate(gray_img, rotated, method, mask=rotated_mask)
            else:
                result_mask = (rotated > 0).astype(np.uint8) * 255
                result = cv2.matchTemplate(gray_img, rotated, method, mask=result_mask)

            # 记录该角度的最佳分数
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                best_score = 1 - min_val
            else:
                best_score = max_val
            score_curve.append({"angle": float(angle), "score": float(best_score)})

            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                results.append((pt[0], pt[1], result[pt[1], pt[0]], angle))

        return results, score_curve

    def _feature_match_sift(self, gray_img):
        template = self._template_cache
        if template is None:
            return False, [], gray_img

        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        if len(gray_img.shape) == 3:
            img_gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = gray_img.copy()

        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(template_gray, None)
        kp2, des2 = sift.detectAndCompute(img_gray, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return False, [], gray_img

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        min_matches = int(self.params.get("min_matches", 10))
        if len(good_matches) < min_matches:
            return False, [], gray_img

        display = cv2.drawMatches(template_gray, kp1, img_gray, kp2,
                                   good_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return True, good_matches, display

    def _feature_match_orb(self, gray_img):
        template = self._template_cache
        if template is None:
            return False, [], gray_img

        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        if len(gray_img.shape) == 3:
            img_gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = gray_img.copy()

        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(img_gray, None)

        if des1 is None or des2 is None:
            return False, [], gray_img

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        min_matches = int(self.params.get("min_matches", 10))
        if len(matches) < min_matches:
            return False, [], gray_img

        good_matches = matches[:min_matches * 2]

        display = cv2.drawMatches(template_gray, kp1, img_gray, kp2,
                                   good_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return True, good_matches, display

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")
        mode = self.params.get("mode", "standard")

        template = self._template_cache
        score_curve = []  # 初始化，防止非rotation模式引用报错
        if template is None:
            # 尝试从 template_path 重新加载模板（兼容保存方案后重新运行的情况）
            template_path = self.params.get("template_path", "")
            if template_path:
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is not None:
                    self._template_cache = template

        if template is None:
            return ToolResult(
                success=False, passed=False,
                processed_image=img, data={},
                message="未设置模板图像"
            )

        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()

        th, tw = template_gray.shape[:2]
        display = img.copy()
        overlay = np.zeros_like(img)
        matches_data = []

        use_mask = self.params.get("use_mask", False)
        mask = self._mask_cache if use_mask else None

        if mode == "standard":
            method_map = {
                "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
                "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
                "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
            }
            method_name = self.params.get("method", "TM_CCOEFF_NORMED")
            method = method_map.get(method_name, cv2.TM_CCOEFF_NORMED)
            threshold = float(self.params.get("threshold", 0.8))
            nms_dist = int(self.params.get("nms_distance", 20))

            # 标准模式支持掩膜
            if use_mask and mask is not None:
                if len(mask.shape) == 3:
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask_gray = mask.copy()
                result = cv2.matchTemplate(gray_img, template_gray, method, mask=mask_gray)
            else:
                result = cv2.matchTemplate(gray_img, template_gray, method)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                locations = np.where(result <= (1 - threshold))
                scores = [1 - result[pt[1], pt[0]] for pt in zip(*locations[::-1])]
            else:
                locations = np.where(result >= threshold)
                scores = [result[pt[1], pt[0]] for pt in zip(*locations[::-1])]

            locations_list = list(zip(*locations[::-1])) if len(locations[0]) > 0 else []
            nms_results = self._non_max_suppression(locations_list, scores, th, tw, nms_dist)

            for x, y, score in nms_results:
                cv2.rectangle(display, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                cv2.putText(display, f"{score:.2f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(overlay, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                cv2.putText(overlay, f"{score:.2f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                matches_data.append({"x": int(x), "y": int(y),
                                      "width": int(tw), "height": int(th),
                                      "score": float(score)})

        elif mode == "rotation":
            threshold = float(self.params.get("threshold", 0.8))
            nms_dist = int(self.params.get("nms_distance", 20))

            results, score_curve = self._multi_angle_match(gray_img, template_gray,
                                                            cv2.TM_CCOEFF_NORMED, threshold)

            locations = [(int(x), int(y)) for x, y, s, a in results]
            scores = [float(s) for x, y, s, a in results]
            nms_results = self._non_max_suppression(locations, scores, th, tw, nms_dist)

            for x, y, score in nms_results:
                angle = 0
                for rx, ry, rs, ra in results:
                    if int(rx) == x and int(ry) == y and abs(rs - score) < 0.01:
                        angle = ra
                        break
                cv2.rectangle(display, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                cv2.putText(display, f"{score:.2f} {angle:.0f}°", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(overlay, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                cv2.putText(overlay, f"{score:.2f} {angle:.0f}°", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                matches_data.append({"x": int(x), "y": int(y),
                                      "width": int(tw), "height": int(th),
                                      "score": float(score), "angle": float(angle)})

        elif mode == "feature":
            feature_mode = self.params.get("feature_mode", "sift")
            if feature_mode == "sift":
                success, good_matches, display = self._feature_match_sift(gray_img)
            else:
                success, good_matches, display = self._feature_match_orb(gray_img)

            if not success:
                return ToolResult(
                    success=True, passed=False,
                    processed_image=img, data={"match_count": 0},
                    message="特征点匹配失败（匹配点不足）"
                )

            matches_data = [{"distance": m.distance} for m in good_matches]

        # 分数阈值判断：最高分 >= threshold 才算通过
        best_score = max([m["score"] for m in matches_data]) if matches_data else 0
        passed = best_score >= threshold

        result_data = {
            "match_count": len(matches_data),
            "matches": matches_data,
            "mode": mode,
            "best_score": float(best_score),
        }

        # 多角度模式下输出分数曲线
        if mode == "rotation" and score_curve:
            result_data["score_curve"] = score_curve

        # 使用完整帧作为 processed_image，确保下游步骤能访问完整图像
        output_image = self._full_frame_image if self._full_frame_image is not None else img

        # 在完整帧的对应位置绘制 overlay 标注
        input_source = self.params.get("_input_source", "current")
        if input_source.startswith("region:") and self._full_frame_image is not None:
            overlay_full = np.zeros_like(self._full_frame_image)
            region_name = input_source[7:]
            if region_name in context.regions:
                rx, ry, rw, rh = context.regions[region_name]
                # 将 overlay 上的标注从 ROI 局部坐标平移到完整帧坐标
                # 对于矩形标注，直接平移矩形左上角坐标
                h_roi, w_roi = img.shape[:2]
                # 重新在完整帧 overlay 上绘制
                for md in matches_data:
                    x0 = md["x"] + rx
                    y0 = md["y"] + ry
                    w0 = md.get("width", tw)
                    h0 = md.get("height", th)
                    score = md.get("score", 0)
                    cv2.rectangle(overlay_full, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                    cv2.putText(overlay_full, f"{score:.2f}", (x0, y0-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            overlay = overlay_full
        # feature 模式下 display 已被替换为特征匹配结果图，不覆盖 overlay

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=output_image,
            overlay_image=overlay,
            data=result_data,
            message=f"找到 {len(matches_data)} 个匹配 (得分={best_score:.3f})"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QSpinBox,
                                      QPushButton, QHBoxLayout, QWidget,
                                      QLabel, QFileDialog, QCheckBox)

        widgets = []

        mode_combo = QComboBox(parent)
        mode_combo.addItem("标准匹配", "standard")
        mode_combo.addItem("多角度匹配", "rotation")
        mode_combo.addItem("特征点匹配", "feature")
        current_mode = self.params.get("mode", "standard")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"mode": mode_combo.itemData(i)}))
        widgets.append(("模式:", mode_combo))

        method_combo = QComboBox(parent)
        method_combo.addItem("归一化相关系数", "TM_CCOEFF_NORMED")
        method_combo.addItem("归一化相关", "TM_CCORR_NORMED")
        method_combo.addItem("归一化平方差", "TM_SQDIFF_NORMED")
        current_method = self.params.get("method", "TM_CCOEFF_NORMED")
        idx = method_combo.findData(current_method)
        if idx >= 0:
            method_combo.setCurrentIndex(idx)
        method_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"method": method_combo.itemData(i)}))
        widgets.append(("方法:", method_combo))

        thresh_spin = QDoubleSpinBox(parent)
        thresh_spin.setRange(0, 1)
        thresh_spin.setSingleStep(0.05)
        thresh_spin.setValue(float(self.params.get("threshold", 0.8)))
        thresh_spin.valueChanged.connect(lambda v: self.params.update({"threshold": v}))
        widgets.append(("阈值:", thresh_spin))

        def choose_template():
            path, _ = QFileDialog.getOpenFileName(
                parent, "选择模板图像", "",
                "图片文件 (*.png *.jpg *.bmp);;所有文件 (*.*)")
            if path:
                self.params["template_path"] = path
                template_img = cv2.imread(path, cv2.IMREAD_COLOR)
                if template_img is not None:
                    self._template_cache = template_img

        btn = QPushButton("选择模板")
        btn.clicked.connect(choose_template)
        widgets.append(("模板:", btn))

        # 掩膜支持
        use_mask_cb = QCheckBox(parent)
        use_mask_cb.setChecked(self.params.get("use_mask", False))
        use_mask_cb.stateChanged.connect(lambda v: self.params.update({"use_mask": bool(v)}))
        widgets.append(("使用掩膜:", use_mask_cb))

        def choose_mask():
            path, _ = QFileDialog.getOpenFileName(
                parent, "选择掩膜图像", "",
                "图片文件 (*.png *.jpg *.bmp);;所有文件 (*.*)")
            if path:
                self.params["mask_path"] = path
                mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    self._mask_cache = mask_img

        mask_btn = QPushButton("选择掩膜")
        mask_btn.clicked.connect(choose_mask)
        widgets.append(("掩膜:", mask_btn))

        angle_start = QSpinBox(parent)
        angle_start.setRange(-180, 180)
        angle_start.setValue(int(self.params.get("angle_start", -30)))
        angle_start.valueChanged.connect(lambda v: self.params.update({"angle_start": v}))
        widgets.append(("起始角度:", angle_start))

        angle_end = QSpinBox(parent)
        angle_end.setRange(-180, 180)
        angle_end.setValue(int(self.params.get("angle_end", 30)))
        angle_end.valueChanged.connect(lambda v: self.params.update({"angle_end": v}))
        widgets.append(("结束角度:", angle_end))

        angle_step = QDoubleSpinBox(parent)
        angle_step.setRange(0.5, 30)
        angle_step.setSingleStep(0.5)
        angle_step.setValue(float(self.params.get("angle_step", 5)))
        angle_step.valueChanged.connect(lambda v: self.params.update({"angle_step": v}))
        widgets.append(("步长:", angle_step))

        feat_combo = QComboBox(parent)
        feat_combo.addItem("SIFT", "sift")
        feat_combo.addItem("ORB", "orb")
        current_feat = self.params.get("feature_mode", "sift")
        idx = feat_combo.findData(current_feat)
        if idx >= 0:
            feat_combo.setCurrentIndex(idx)
        feat_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"feature_mode": feat_combo.itemData(i)}))
        widgets.append(("特征模式:", feat_combo))

        min_match = QSpinBox(parent)
        min_match.setRange(1, 1000)
        min_match.setValue(int(self.params.get("min_matches", 10)))
        min_match.valueChanged.connect(lambda v: self.params.update({"min_matches": v}))
        widgets.append(("最小匹配数:", min_match))

        return widgets


class EdgeMatch(VisionTool):
    display_name = "边缘匹配"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("template_path", "")
        self.params.setdefault("template_data", None)
        self.params.setdefault("canny_low", 50)
        self.params.setdefault("canny_high", 150)
        self.params.setdefault("match_threshold", 0.3)
        self.params.setdefault("min_area", 100)
        self._template_edges = None
        self._template_contour = None

    def _load_template(self, path):
        try:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                return
            low = int(self.params.get("canny_low", 50))
            high = int(self.params.get("canny_high", 150))
            edges = cv2.Canny(template, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                self._template_contour = max(contours, key=cv2.contourArea)
                self._template_edges = edges
        except Exception as e:
            print(f"加载模板失败: {e}")

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        if self._template_contour is None:
            template_path = self.params.get("template_path", "")
            if template_path:
                self._load_template(template_path)

            if self._template_contour is None:
                return ToolResult(
                    success=False, passed=False,
                    processed_image=img, data={},
                    message="未加载模板"
                )

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        low = int(self.params.get("canny_low", 50))
        high = int(self.params.get("canny_high", 150))
        edges = cv2.Canny(gray, low, high)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        match_threshold = float(self.params.get("match_threshold", 0.3))
        min_area = float(self.params.get("min_area", 100))

        display = img.copy()
        overlay = np.zeros_like(img)
        matches = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            try:
                match_value = cv2.matchShapes(self._template_contour, contour,
                                               cv2.CONTOURS_MATCH_I1, 0)
            except Exception:
                continue

            if match_value < match_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(display, f"{match_value:.3f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(overlay, f"{match_value:.3f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                matches.append({
                    "x": int(x), "y": int(y),
                    "width": int(w), "height": int(h),
                    "match_value": float(match_value),
                })

        passed = len(matches) > 0

        # 使用完整帧作为 processed_image，确保下游步骤能访问完整图像
        output_image = self._full_frame_image if self._full_frame_image is not None else img

        # 在完整帧的对应位置绘制 overlay 标注
        input_source = self.params.get("_input_source", "current")
        if input_source.startswith("region:") and self._full_frame_image is not None:
            overlay_full = np.zeros_like(self._full_frame_image)
            region_name = input_source[7:]
            if region_name in context.regions:
                rx, ry, rw, rh = context.regions[region_name]
                # 将轮廓坐标从 ROI 局部坐标平移到完整帧坐标
                for m in matches:
                    x0 = m["x"] + rx
                    y0 = m["y"] + ry
                    w0 = m["width"]
                    h0 = m["height"]
                    cv2.rectangle(overlay_full, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 0), 1)
                    cv2.putText(overlay_full, f"{m['match_value']:.3f}", (x0, y0-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            overlay = overlay_full

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=output_image,
            overlay_image=overlay,
            data={
                "match_count": len(matches),
                "matches": matches,
            },
            message=f"找到 {len(matches)} 个边缘匹配"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QPushButton, QDoubleSpinBox, QSpinBox,
                                      QHBoxLayout, QWidget, QLabel, QFileDialog)

        widgets = []

        def choose_template():
            path, _ = QFileDialog.getOpenFileName(
                parent, "选择模板图像", "",
                "图片文件 (*.png *.jpg *.bmp);;所有文件 (*.*)")
            if path:
                self.params["template_path"] = path
                self._load_template(path)

        btn = QPushButton("选择模板")
        btn.clicked.connect(choose_template)
        widgets.append(("模板:", btn))

        canny_low = QSpinBox(parent)
        canny_low.setRange(0, 500)
        canny_low.setValue(int(self.params.get("canny_low", 50)))
        canny_low.valueChanged.connect(lambda v: self.params.update({"canny_low": v}))
        widgets.append(("Canny低阈值:", canny_low))

        canny_high = QSpinBox(parent)
        canny_high.setRange(0, 1000)
        canny_high.setValue(int(self.params.get("canny_high", 150)))
        canny_high.valueChanged.connect(lambda v: self.params.update({"canny_high": v}))
        widgets.append(("Canny高阈值:", canny_high))

        match_thresh = QDoubleSpinBox(parent)
        match_thresh.setRange(0, 1)
        match_thresh.setSingleStep(0.05)
        match_thresh.setValue(float(self.params.get("match_threshold", 0.3)))
        match_thresh.valueChanged.connect(
            lambda v: self.params.update({"match_threshold": v}))
        widgets.append(("匹配阈值:", match_thresh))

        return widgets


class FastMatch(VisionTool):
    display_name = "快速匹配"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("template_path", "")
        self.params.setdefault("template_data", None)
        self.params.setdefault("pyramid_levels", 3)
        self.params.setdefault("threshold", 0.7)
        self.params.setdefault("method", "TM_CCOEFF_NORMED")
        self._template_cache = None

    def set_template(self, template_img):
        self._template_cache = template_img

    def _build_pyramid(self, img, levels):
        pyramid = [img]
        for i in range(levels):
            if pyramid[-1].shape[0] > 10 and pyramid[-1].shape[1] > 10:
                down = cv2.pyrDown(pyramid[-1])
                pyramid.append(down)
            else:
                break
        return pyramid

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        template = self._template_cache
        if template is None:
            template_path = self.params.get("template_path", "")
            if template_path:
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is not None:
                    self._template_cache = template

            if template is None:
                return ToolResult(
                    success=False, passed=False,
                    processed_image=img, data={},
                    message="未设置模板图像"
                )

        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()

        levels = int(self.params.get("pyramid_levels", 3))
        threshold = float(self.params.get("threshold", 0.7))
        method_name = self.params.get("method", "TM_CCOEFF_NORMED")

        method_map = {
            "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        }
        method = method_map.get(method_name, cv2.TM_CCOEFF_NORMED)

        img_pyramid = self._build_pyramid(gray_img, levels)
        tmpl_pyramid = self._build_pyramid(template_gray, levels)

        best_score = -1
        best_location = None
        best_scale = 1.0

        for level in range(min(len(img_pyramid), len(tmpl_pyramid))):
            img_level = img_pyramid[level]
            tmpl_level = tmpl_pyramid[level]

            if img_level.shape[0] < tmpl_level.shape[0] or \
               img_level.shape[1] < tmpl_level.shape[1]:
                continue

            result = cv2.matchTemplate(img_level, tmpl_level, method)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                score = 1 - min_val
                location = min_loc
            else:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                score = max_val
                location = max_loc

            if score > best_score:
                best_score = score
                best_location = location
                best_scale = 2 ** level

        if best_location is not None:
            x = int(best_location[0] * best_scale)
            y = int(best_location[1] * best_scale)
            w = int(template_gray.shape[1] * best_scale)
            h = int(template_gray.shape[0] * best_scale)
        else:
            x, y, w, h = 0, 0, 0, 0

        display = img.copy()
        overlay = np.zeros_like(img)
        passed = best_score >= threshold

        if passed:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, f"{best_score:.2f}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, f"{best_score:.2f}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 使用完整帧作为 processed_image，确保下游步骤能访问完整图像
        output_image = self._full_frame_image if self._full_frame_image is not None else img

        # 在完整帧的对应位置绘制 overlay 标注
        input_source = self.params.get("_input_source", "current")
        if input_source.startswith("region:") and self._full_frame_image is not None and passed:
            overlay_full = np.zeros_like(self._full_frame_image)
            region_name = input_source[7:]
            if region_name in context.regions:
                rx, ry, rw, rh = context.regions[region_name]
                x0 = x + rx
                y0 = y + ry
                cv2.rectangle(overlay_full, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
                cv2.putText(overlay_full, f"{best_score:.2f}", (x0, y0-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            overlay = overlay_full

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=output_image,
            overlay_image=overlay,
            data={
                "score": float(best_score),
                "x": int(x), "y": int(y),
                "width": int(w), "height": int(h),
            },
            message=f"匹配得分={best_score:.3f}" + (" (通过)" if passed else " (未通过)")
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QSpinBox,
                                      QPushButton, QHBoxLayout, QWidget,
                                      QLabel, QFileDialog)

        widgets = []

        def choose_template():
            path, _ = QFileDialog.getOpenFileName(
                parent, "选择模板图像", "",
                "图片文件 (*.png *.jpg *.bmp);;所有文件 (*.*)")
            if path:
                self.params["template_path"] = path
                template_img = cv2.imread(path, cv2.IMREAD_COLOR)
                if template_img is not None:
                    self._template_cache = template_img

        btn = QPushButton("选择模板")
        btn.clicked.connect(choose_template)
        widgets.append(("模板:", btn))

        levels_spin = QSpinBox(parent)
        levels_spin.setRange(1, 10)
        levels_spin.setValue(int(self.params.get("pyramid_levels", 3)))
        levels_spin.valueChanged.connect(
            lambda v: self.params.update({"pyramid_levels": v}))
        widgets.append(("金字塔层数:", levels_spin))

        thresh_spin = QDoubleSpinBox(parent)
        thresh_spin.setRange(0, 1)
        thresh_spin.setSingleStep(0.05)
        thresh_spin.setValue(float(self.params.get("threshold", 0.7)))
        thresh_spin.valueChanged.connect(lambda v: self.params.update({"threshold": v}))
        widgets.append(("阈值:", thresh_spin))


