# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from .base_tool import VisionTool, ToolResult, PipelineContext
from core.log_manager import log_warning


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

        # 若模板比输入图像大，自动缩放模板到输入尺寸内（cv2.matchTemplate 要求模板 ≤ 输入）
        img_h, img_w = gray_img.shape[:2]
        th, tw = template_gray.shape[:2]
        if th > img_h or tw > img_w:
            scale = min(img_h / th, img_w / tw)
            new_w = max(1, int(tw * scale))
            new_h = max(1, int(th * scale))
            template_gray = cv2.resize(template_gray, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
            th, tw = template_gray.shape[:2]
            log_warning(f"模板大于输入图像，已自动缩放模板到 {tw}x{th}")

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
            is_sqdiff = method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

            # 多尺度匹配参数
            scale_min = float(self.params.get("scale_min", 0.5))
            scale_max = float(self.params.get("scale_max", 2.0))
            scale_step = float(self.params.get("scale_step", 0.1))

            # 多尺度搜索：遍历缩放比例，找出最佳匹配
            best_score = -1.0 if not is_sqdiff else 1.0
            best_loc = (0, 0)
            best_tw, best_th = tw, th
            best_scale = 1.0
            best_result = None

            th0, tw0 = template_gray.shape[:2]
            scale = scale_min
            while scale <= scale_max + 1e-6:
                sw = max(1, int(round(tw0 * scale)))
                sh = max(1, int(round(th0 * scale)))
                # 模板不能大于输入图像
                if sw > img_w or sh > img_h:
                    scale += scale_step
                    continue
                scaled_templ = cv2.resize(template_gray, (sw, sh),
                                          interpolation=cv2.INTER_AREA)
                if use_mask and mask is not None:
                    if len(mask.shape) == 3:
                        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    else:
                        mask_gray = mask.copy()
                    if mask_gray.shape[:2] != scaled_templ.shape[:2]:
                        mask_gray = cv2.resize(mask_gray, (sw, sh),
                                               interpolation=cv2.INTER_AREA)
                    result = cv2.matchTemplate(gray_img, scaled_templ, method, mask=mask_gray)
                else:
                    result = cv2.matchTemplate(gray_img, scaled_templ, method)

                if is_sqdiff:
                    min_val, _, min_loc, _ = cv2.minMaxLoc(result)
                    if min_val < best_score:
                        best_score = min_val
                        best_loc = min_loc
                        best_tw, best_th = sw, sh
                        best_scale = scale
                        best_result = result
                else:
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_score:
                        best_score = max_val
                        best_loc = max_loc
                        best_tw, best_th = sw, sh
                        best_scale = scale
                        best_result = result
                scale += scale_step

            # 用最佳缩放的结果提取匹配位置
            result = best_result
            tw, th = best_tw, best_th
            if result is not None:
                if is_sqdiff:
                    locations = np.where(result <= (1 - threshold))
                    scores = [1 - result[pt[1], pt[0]] for pt in zip(*locations[::-1])]
                else:
                    locations = np.where(result >= threshold)
                    scores = [result[pt[1], pt[0]] for pt in zip(*locations[::-1])]

                locations_list = list(zip(*locations[::-1])) if len(locations[0]) > 0 else []
                nms_results = self._non_max_suppression(locations_list, scores, th, tw, nms_dist)
            else:
                nms_results = []

            for x, y, score in nms_results:
                cv2.rectangle(display, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                cv2.putText(display, f"{score:.2f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(overlay, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                cv2.putText(overlay, f"{score:.2f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                matches_data.append({"x": int(x), "y": int(y),
                                      "width": int(tw), "height": int(th),
                                      "score": float(score),
                                      "scale": float(best_scale)})

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

        # 分数阈值判断
        if mode == "feature":
            # 特征点匹配：用匹配点数判断（>= min_matches 才算通过）
            min_matches = int(self.params.get("min_matches", 10))
            best_score = float(len(matches_data))
            passed = len(matches_data) >= min_matches
        else:
            # 标准/多角度匹配：最高分 >= threshold 才算通过
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
                                      QPushButton, QHBoxLayout, QVBoxLayout,
                                      QWidget, QLabel, QFileDialog, QCheckBox)

        # 返回一个容器 QWidget，内部根据所选模式动态显示/隐藏对应参数行
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        def make_row(label_text, widget):
            """创建一行 (标签 + 控件)，返回包裹的 QWidget 以便显示/隐藏。"""
            row_widget = QWidget(container)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(80)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            return row_widget

        # ---- 模式选择（始终显示） ----
        mode_combo = QComboBox(container)
        mode_combo.addItem("标准匹配", "standard")
        mode_combo.addItem("多角度匹配", "rotation")
        mode_combo.addItem("特征点匹配", "feature")
        current_mode = self.params.get("mode", "standard")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"mode": mode_combo.itemData(i)}))
        layout.addWidget(make_row("模式:", mode_combo))

        # ---- 模板选择（所有模式都需要，始终显示） ----
        def choose_template():
            path, _ = QFileDialog.getOpenFileName(
                parent, "选择模板图像", "",
                "图片文件 (*.png *.jpg *.bmp);;所有文件 (*.*)")
            if path:
                self.params["template_path"] = path
                template_img = cv2.imread(path, cv2.IMREAD_COLOR)
                if template_img is not None:
                    self._template_cache = template_img

        template_btn = QPushButton("选择模板")
        template_btn.clicked.connect(choose_template)
        layout.addWidget(make_row("模板:", template_btn))

        # ---- 标准匹配参数 ----
        method_combo = QComboBox(container)
        method_combo.addItem("归一化相关系数", "TM_CCOEFF_NORMED")
        method_combo.addItem("归一化相关", "TM_CCORR_NORMED")
        method_combo.addItem("归一化平方差", "TM_SQDIFF_NORMED")
        current_method = self.params.get("method", "TM_CCOEFF_NORMED")
        idx = method_combo.findData(current_method)
        if idx >= 0:
            method_combo.setCurrentIndex(idx)
        method_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"method": method_combo.itemData(i)}))
        row_method = make_row("方法:", method_combo)

        thresh_spin = QDoubleSpinBox(container)
        thresh_spin.setRange(0, 1)
        thresh_spin.setSingleStep(0.05)
        thresh_spin.setValue(float(self.params.get("threshold", 0.8)))
        thresh_spin.valueChanged.connect(lambda v: self.params.update({"threshold": v}))
        row_thresh = make_row("阈值:", thresh_spin)

        # 多尺度缩放范围
        scale_min_spin = QDoubleSpinBox(container)
        scale_min_spin.setRange(0.1, 5.0)
        scale_min_spin.setSingleStep(0.1)
        scale_min_spin.setValue(float(self.params.get("scale_min", 0.5)))
        scale_min_spin.valueChanged.connect(lambda v: self.params.update({"scale_min": v}))
        row_scale_min = make_row("最小缩放:", scale_min_spin)

        scale_max_spin = QDoubleSpinBox(container)
        scale_max_spin.setRange(0.1, 5.0)
        scale_max_spin.setSingleStep(0.1)
        scale_max_spin.setValue(float(self.params.get("scale_max", 2.0)))
        scale_max_spin.valueChanged.connect(lambda v: self.params.update({"scale_max": v}))
        row_scale_max = make_row("最大缩放:", scale_max_spin)

        scale_step_spin = QDoubleSpinBox(container)
        scale_step_spin.setRange(0.05, 1.0)
        scale_step_spin.setSingleStep(0.05)
        scale_step_spin.setValue(float(self.params.get("scale_step", 0.1)))
        scale_step_spin.valueChanged.connect(lambda v: self.params.update({"scale_step": v}))
        row_scale_step = make_row("缩放步长:", scale_step_spin)

        use_mask_cb = QCheckBox(container)
        use_mask_cb.setChecked(self.params.get("use_mask", False))
        use_mask_cb.stateChanged.connect(lambda v: self.params.update({"use_mask": bool(v)}))
        row_mask_cb = make_row("使用掩膜:", use_mask_cb)

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
        row_mask_btn = make_row("掩膜:", mask_btn)

        # ---- 多角度匹配参数 ----
        angle_start = QSpinBox(container)
        angle_start.setRange(-180, 180)
        angle_start.setValue(int(self.params.get("angle_start", -30)))
        angle_start.valueChanged.connect(lambda v: self.params.update({"angle_start": v}))
        row_angle_start = make_row("起始角度:", angle_start)

        angle_end = QSpinBox(container)
        angle_end.setRange(-180, 180)
        angle_end.setValue(int(self.params.get("angle_end", 30)))
        angle_end.valueChanged.connect(lambda v: self.params.update({"angle_end": v}))
        row_angle_end = make_row("结束角度:", angle_end)

        angle_step = QDoubleSpinBox(container)
        angle_step.setRange(0.5, 30)
        angle_step.setSingleStep(0.5)
        angle_step.setValue(float(self.params.get("angle_step", 5)))
        angle_step.valueChanged.connect(lambda v: self.params.update({"angle_step": v}))
        row_angle_step = make_row("步长:", angle_step)

        # ---- 特征点匹配参数 ----
        feat_combo = QComboBox(container)
        feat_combo.addItem("SIFT", "sift")
        feat_combo.addItem("ORB", "orb")
        current_feat = self.params.get("feature_mode", "sift")
        idx = feat_combo.findData(current_feat)
        if idx >= 0:
            feat_combo.setCurrentIndex(idx)
        feat_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"feature_mode": feat_combo.itemData(i)}))
        row_feat = make_row("特征模式:", feat_combo)

        min_match = QSpinBox(container)
        min_match.setRange(1, 1000)
        min_match.setValue(int(self.params.get("min_matches", 10)))
        min_match.valueChanged.connect(lambda v: self.params.update({"min_matches": v}))
        row_min_match = make_row("最小匹配数:", min_match)

        # 按模式组织参数行
        mode_rows = {
            "standard": [row_method, row_thresh, row_scale_min, row_scale_max,
                         row_scale_step, row_mask_cb, row_mask_btn],
            "rotation": [row_thresh, row_angle_start, row_angle_end, row_angle_step],
            "feature": [row_feat, row_min_match],
        }

        # 将各参数行加入布局
        for rows in mode_rows.values():
            for rw in rows:
                layout.addWidget(rw)

        # 连接预览信号（parent 为 ParamConfigDialog）
        if hasattr(parent, "_connect_auto_preview"):
            for w in [mode_combo, method_combo, thresh_spin, scale_min_spin,
                      scale_max_spin, scale_step_spin, use_mask_cb,
                      angle_start, angle_end, angle_step, feat_combo, min_match]:
                parent._connect_auto_preview(w)

        def _update_visibility():
            mode = mode_combo.itemData(mode_combo.currentIndex())
            for key, rows in mode_rows.items():
                visible = (key == mode)
                for rw in rows:
                    rw.setVisible(visible)

        mode_combo.currentIndexChanged.connect(lambda i: _update_visibility())
        _update_visibility()

        # 返回容器（渲染逻辑会将 QWidget 作为单独一行加入）
        return [(container, None)]


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


class QRCodeRecognize(VisionTool):
    """通用条码识别算子（二维码 + 一维码）。

    使用 OpenCV 的 QRCodeDetector 识别二维码，使用 pyzbar 识别一维码
    （Code 128 / Code 39 / EAN-13 / UPC-A 等）。自动区分二维码与一维码，
    将识别结果（如板卡 SN）写入 ToolResult.data["qr_data"]（第一个条码，向后兼容），
    并将所有条码详情写入 data["barcodes"]。

    参数:
        - require_pass: 是否将"识别到条码"作为通过条件（默认 True）
        - expected_prefix: 可选，期望的 SN 前缀（用于校验，可为空）
        - enable_1d: 是否启用一维码识别（默认 True）
        - barcode_formats: 可选的一维码格式集合（如 ["CODE_128", "CODE_39", "EAN_13", "UPC_A"]）
    """
    display_name = "条码识别"

    # 一维码格式常量（pyzbar 返回的类型名）
    BARCODE_FORMATS = {
        "CODE_128": "CODE128",
        "CODE_39": "CODE39",
        "EAN_13": "EAN13",
        "EAN_8": "EAN8",
        "UPC_A": "UPCA",
        "UPC_E": "UPCE",
        "ITF": "I25",
        "CODABAR": "CODABAR",
    }

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("require_pass", True)
        self.params.setdefault("expected_prefix", "")
        self.params.setdefault("enable_1d", True)
        self.params.setdefault("barcode_formats",
                               ["CODE_128", "CODE_39", "EAN_13", "UPC_A"])
        self._pyzbar_available = None

    def _check_pyzbar(self) -> bool:
        """检查 pyzbar 是否可用。"""
        if self._pyzbar_available is None:
            try:
                from pyzbar import pyzbar  # noqa: F401
                self._pyzbar_available = True
            except Exception:  # noqa: BLE001
                self._pyzbar_available = False
        return self._pyzbar_available

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False, message="无输入图像")

        # 转灰度（条码检测需要灰度图）
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 识别所有条码（二维码 + 一维码）
        barcodes = []  # 每个元素: {"type", "data", "confidence", "bbox"}
        barcodes.extend(self._decode_qr(gray))
        enable_1d = self.params.get("enable_1d", True)
        print(f"[DEBUG][QRCodeRecognize] enable_1d={enable_1d} params={self.params}")
        if enable_1d:
            barcodes.extend(self._decode_1d(gray))

        # 去重（按内容 + 位置）
        barcodes = self._deduplicate(barcodes)

        # 调试日志：输出算子实际收到的图像尺寸与识别结果
        print(f"[DEBUG][QRCodeRecognize] 输入图像 shape={img.shape} dtype={img.dtype} "
              f"input_source={self.params.get('_input_source') or self.params.get('input_source', 'current')} "
              f"识别到 {len(barcodes)} 个条码: {[b.get('data') for b in barcodes]}")
        # 保存算子收到的输入图像，便于排查 ROI 内容
        try:
            import os
            dbg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '_debug_roi')
            os.makedirs(dbg_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dbg_dir, 'operator_input.png'), img)
        except Exception:  # noqa: BLE001
            pass

        # 第一个条码内容（向后兼容 qr_data 字段）
        first_data = barcodes[0]["data"] if barcodes else ""

        expected_prefix = self.params.get("expected_prefix", "").strip()

        # 校验前缀（可选）：对第一个条码校验
        prefix_ok = True
        if expected_prefix and not first_data.startswith(expected_prefix):
            prefix_ok = False

        # 判定：识别到且（无前缀要求或前缀匹配）
        recognized = bool(first_data) and prefix_ok
        require_pass = self.params.get("require_pass", True)
        passed = recognized if require_pass else True

        # 绘制 overlay 标注（框出所有条码）
        # 注意：若使用 ROI 输入源，img 是 ROI 局部图像，需将标注偏移到完整帧坐标
        input_source = self.params.get("_input_source") or self.params.get("input_source", "current")
        if input_source.startswith("region:") and self._full_frame_image is not None:
            overlay = np.zeros_like(self._full_frame_image)
            region_name = input_source[7:]
            rx, ry = 0, 0
            if region_name in context.regions:
                rx, ry, _, _ = context.regions[region_name]
        else:
            overlay = np.zeros_like(img)
            rx, ry = 0, 0

        for bc in barcodes:
            x, y, w, h = bc["bbox"]
            # 偏移到完整帧坐标（ROI 模式）
            x += rx
            y += ry
            # 识别到条码统一用绿色标注（二维码/一维码）
            color = (0, 255, 0)
            # 使用识别点（圆点）标注，避免 bbox 位置不稳定导致识别框乱跳
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(overlay, (cx, cy), 6, color, -1)
            cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 1)
            label = f"{bc['type']}:{bc['data']}"
            cv2.putText(overlay, label, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        result_data = {
            "qr_data": first_data,
            "recognized": recognized,
            "barcodes": barcodes,
            "barcode_count": len(barcodes),
        }

        if recognized:
            message = f"条码识别成功: {first_data} ({len(barcodes)} 个)"
        elif first_data:
            message = f"条码前缀校验失败: {first_data}"
        else:
            message = "未识别到条码"

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=img,
            overlay_image=overlay,
            data=result_data,
            message=message
        )

    def _decode_qr(self, gray: np.ndarray) -> list:
        """识别二维码，返回条码列表。"""
        results = []
        data, points = self._try_decode(gray)
        if data:
            bbox = self._points_to_bbox(points)
            results.append({
                "type": "QR",
                "data": data,
                "confidence": 1.0,
                "bbox": bbox,
            })
        return results

    def _decode_1d(self, gray: np.ndarray) -> list:
        """识别一维码（使用 pyzbar），多策略提高识别率，返回条码列表。

        依次尝试：
            1. 原始灰度图
            2. 自适应阈值二值化（增强对比度）
            3. CLAHE 对比度增强
            4. 放大 2 倍（小一维码）
            5. 旋转 90°（垂直一维码）
        """
        pyzbar_ok = self._check_pyzbar()
        print(f"[DEBUG][_decode_1d] pyzbar可用={pyzbar_ok}")
        if not pyzbar_ok:
            return []
        try:
            from pyzbar import pyzbar
        except Exception as e:  # noqa: BLE001
            print(f"[DEBUG][_decode_1d] pyzbar导入失败: {e}")
            return []

        results = []

        def _collect(decoded, scale=1.0, rot=0, src_shape=None, tag=""):
            """收集识别结果，scale 为放大倍数，rot 为旋转角度。

            rot=90 表示图像顺时针旋转 90° 后识别，需将旋转后坐标逆变换回原图坐标。
            src_shape 为原图 (h, w)，用于旋转坐标逆变换。
            """
            for d in decoded:
                btype = d.type
                if not self._format_allowed(btype):
                    continue
                data = d.data.decode('utf-8', errors='replace') if d.data else ""
                rect = d.rect
                # 坐标缩放回原图（若放大过）
                left = int(rect.left / scale)
                top = int(rect.top / scale)
                width = int(rect.width / scale)
                height = int(rect.height / scale)
                # 旋转后坐标逆变换回原图坐标
                if rot == 90 and src_shape is not None:
                    src_h, src_w = src_shape
                    # 顺时针旋转90°: 原图(x,y) -> 旋转图(y, src_h-1-x)
                    # 逆变换: 原图 x = src_h-1-y_rot, 原图 y = x_rot
                    x0 = src_h - 1 - (top + height)
                    y0 = left
                    left, top = x0, y0
                    width, height = height, width
                elif rot == 270 and src_shape is not None:
                    src_h, src_w = src_shape
                    # 逆时针旋转90°(顺时针270°): 原图(x,y) -> 旋转图(src_w-1-y, x)
                    # 逆变换: 原图 x = src_w-1-y_rot, 原图 y = x_rot
                    x0 = src_w - 1 - (top + height)
                    y0 = left
                    left, top = x0, y0
                    width, height = height, width
                bbox = (left, top, width, height)
                print(f"[DEBUG][_decode_1d] 策略[{tag}] 识别到 {data} bbox={bbox} raw_rect={rect}")
                # 一维码可能返回极窄或 0 尺寸的矩形（小一维码/旋转后），
                # 给 bbox 设置最小宽度/高度，保证识别框可见且不丢失识别结果。
                min_w = max(20, int(width * 0.5))
                min_h = max(20, int(height * 0.5))
                if width < min_w:
                    left = max(0, left - (min_w - width) // 2)
                    width = min_w
                if height < min_h:
                    top = max(0, top - (min_h - height) // 2)
                    height = min_h
                bbox = (left, top, width, height)
                confidence = min(1.0, getattr(d, 'quality', 100) / 100.0)
                results.append({
                    "type": "1D",
                    "data": data,
                    "confidence": confidence,
                    "barcode_type": btype,
                    "bbox": bbox,
                })

        # 策略 1：原始灰度图
        try:
            _collect(pyzbar.decode(gray))
        except Exception:  # noqa: BLE001
            pass

        # 策略 2：自适应阈值二值化
        try:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 10)
            _collect(pyzbar.decode(binary))
        except Exception:  # noqa: BLE001
            pass

        # 策略 3：CLAHE 对比度增强
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _collect(pyzbar.decode(enhanced))
        except Exception:  # noqa: BLE001
            pass

        # 策略 4：多尺度放大（2x/3x/4x，小一维码），放大后同时尝试原始/自适应阈值/CLAHE
        # 放大策略的 bbox 宽高非 0（位置准确），优先使用
        try:
            h, w = gray.shape[:2]
            if max(h, w) < 800:
                for scale in (2, 3, 4):
                    up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                    _collect(pyzbar.decode(up), scale=scale, tag=f"4a_x{scale}")
                    # 放大后自适应阈值（小一维码放大后仍需二值化增强）
                    up_bin = cv2.adaptiveThreshold(
                        up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 51, 10)
                    _collect(pyzbar.decode(up_bin), scale=scale, tag=f"4b_x{scale}")
                    # 放大后 CLAHE
                    up_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(up)
                    _collect(pyzbar.decode(up_clahe), scale=scale, tag=f"4c_x{scale}")
                    if results:
                        break
        except Exception as e:  # noqa: BLE001
            print(f"[DEBUG][_decode_1d] 策略4异常: {e}")

        # 放大策略已识别成功（bbox 位置准确），直接返回，避免原始图/旋转策略产生错误 bbox
        if results:
            return results

        # 策略 5：旋转 90°（垂直一维码），旋转后同时尝试原始/自适应阈值
        try:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            _collect(pyzbar.decode(rotated), rot=90, src_shape=gray.shape[:2], tag="5a")
            rot_bin = cv2.adaptiveThreshold(
                rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 10)
            _collect(pyzbar.decode(rot_bin), rot=90, src_shape=gray.shape[:2], tag="5b")
        except Exception:  # noqa: BLE001
            pass

        # 策略 6：旋转 270°（垂直一维码，反向），旋转后同时尝试原始/自适应阈值
        try:
            rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            _collect(pyzbar.decode(rotated), rot=270, src_shape=gray.shape[:2], tag="6a")
            rot_bin = cv2.adaptiveThreshold(
                rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 10)
            _collect(pyzbar.decode(rot_bin), rot=270, src_shape=gray.shape[:2], tag="6b")
        except Exception:  # noqa: BLE001
            pass

        return results

    def _format_allowed(self, btype: str) -> bool:
        """判断一维码格式是否在允许集合内。

        btype 为 pyzbar 返回的类型名（如 CODE128），
        配置的 barcode_formats 使用标准名（如 CODE_128）。
        """
        formats = self.params.get("barcode_formats", [])
        if not formats:
            return True
        # 将配置的标准名映射为 pyzbar 类型名
        allowed_types = set()
        for fmt in formats:
            allowed_types.add(self.BARCODE_FORMATS.get(fmt, fmt))
        return btype in allowed_types

    def _points_to_bbox(self, points) -> tuple:
        """将条码角点转换为轴对齐矩形框 (x, y, w, h)。"""
        if points is None or len(points) == 0:
            return (0, 0, 0, 0)
        pts = points.reshape(-1, 2).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        return (int(x), int(y), int(w), int(h))

    def _deduplicate(self, barcodes: list) -> list:
        """按内容 + 位置去重（位置接近的条码合并）。

        多策略识别可能对同一个条码返回多个位置略有差异的结果，
        这里按内容 + 位置重叠度判断，位置重叠超过 50% 视为同一条码。
        对相同内容但位置差异大的条码（如旋转策略产生的错误 bbox），
        优先保留 bbox 宽高非 0 的条码（放大/原始图策略的 bbox 更准确）。
        """
        # 优先保留 bbox 宽高非 0 的条码（位置准确），宽高为 0 的排后面
        def _bbox_valid(bc):
            bbox = bc.get("bbox", (0, 0, 0, 0))
            return bbox[2] > 0 and bbox[3] > 0

        barcodes = sorted(barcodes, key=lambda bc: (0 if _bbox_valid(bc) else 1))

        result = []
        for bc in barcodes:
            data = bc.get("data", "")
            bbox = bc.get("bbox", (0, 0, 0, 0))
            # 检查是否与已有结果重叠
            duplicate = False
            for existing in result:
                if existing.get("data", "") != data:
                    continue
                if self._bbox_overlap(bbox, existing.get("bbox", (0, 0, 0, 0))):
                    duplicate = True
                    break
            if not duplicate:
                result.append(bc)
        # 对相同内容但位置差异大的条码，只保留第一个（位置最准确）
        seen_data = set()
        final = []
        for bc in result:
            data = bc.get("data", "")
            if data in seen_data:
                continue
            seen_data.add(data)
            final.append(bc)
        return final

    @staticmethod
    def _bbox_overlap(b1: tuple, b2: tuple) -> bool:
        """判断两个矩形框是否重叠（重叠面积占比 > 50%）。"""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return False
        # 交集
        ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter = ix * iy
        area1 = w1 * h1
        area2 = w2 * h2
        # 重叠占比（相对较小框）
        min_area = min(area1, area2)
        if min_area <= 0:
            return False
        return (inter / min_area) > 0.5

    def _try_decode(self, gray: np.ndarray):
        """多策略尝试解码二维码，返回 (data, points)。

        依次尝试：
            1. 原始灰度图（detectAndDecode）
            2. 自适应阈值二值化
            3. 放大 2 倍（小二维码）
            4. CLAHE 对比度增强
            5. 多码检测 detectAndDecodeMulti
        """
        detector = cv2.QRCodeDetector()

        # 策略 1：原始灰度图
        try:
            data, points, _ = detector.detectAndDecode(gray)
            if data:
                return data, points
        except Exception:  # noqa: BLE001
            pass

        # 策略 2：自适应阈值二值化（增强对比度）
        try:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 10)
            data, points, _ = detector.detectAndDecode(binary)
            if data:
                return data, points
        except Exception:  # noqa: BLE001
            pass

        # 策略 3：放大 2 倍（小二维码）
        try:
            h, w = gray.shape[:2]
            if max(h, w) < 800:
                up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                data, points, _ = detector.detectAndDecode(up)
                if data:
                    # 坐标缩放回原图
                    if points is not None and len(points) > 0:
                        points = points / 2.0
                    return data, points
        except Exception:  # noqa: BLE001
            pass

        # 策略 4：CLAHE 对比度增强
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            data, points, _ = detector.detectAndDecode(enhanced)
            if data:
                return data, points
        except Exception:  # noqa: BLE001
            pass

        # 策略 5：多码检测（更鲁棒）
        try:
            ok, decoded, points_arr, _ = detector.detectAndDecodeMulti(gray)
            if ok and decoded:
                for i, d in enumerate(decoded):
                    if d:
                        pts = points_arr[i] if points_arr is not None else None
                        return d, pts
        except Exception:  # noqa: BLE001
            pass

        return "", None

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QCheckBox, QLineEdit, QHBoxLayout,
                                      QWidget, QLabel, QComboBox)

        widgets = []

        require_cb = QCheckBox("识别到条码才判定通过")
        require_cb.setChecked(bool(self.params.get("require_pass", True)))
        require_cb.stateChanged.connect(
            lambda s: self.params.update({"require_pass": bool(s)}))
        widgets.append(("", require_cb))

        prefix_edit = QLineEdit(self.params.get("expected_prefix", ""))
        prefix_edit.setPlaceholderText("可选，SN 前缀校验")
        prefix_edit.textChanged.connect(
            lambda t: self.params.update({"expected_prefix": t}))
        widgets.append(("SN前缀:", prefix_edit))

        # 是否启用一维码识别
        enable_1d_cb = QCheckBox("启用一维码识别")
        enable_1d_cb.setChecked(bool(self.params.get("enable_1d", True)))
        enable_1d_cb.stateChanged.connect(
            lambda s: self.params.update({"enable_1d": bool(s)}))
        widgets.append(("", enable_1d_cb))

        # 一维码格式集合（可编辑，逗号分隔）
        formats_combo = QComboBox(parent)
        formats_combo.setEditable(True)
        current_formats = self.params.get("barcode_formats",
                                          ["CODE_128", "CODE_39", "EAN_13", "UPC_A"])
        formats_combo.addItem("全部格式", "")
        for fmt in self.BARCODE_FORMATS:
            formats_combo.addItem(fmt, fmt)
        if current_formats:
            formats_combo.setCurrentText(",".join(current_formats))
        formats_combo.currentTextChanged.connect(
            lambda t: self.params.update({"barcode_formats": self._parse_formats(t)}))
        widgets.append(("一维码格式:", formats_combo))

        return widgets

    def _parse_formats(self, text: str) -> list:
        """解析一维码格式配置（逗号分隔）。空表示全部格式。"""
        text = (text or "").strip()
        if not text or text == "全部格式":
            return []
        parts = [p.strip().upper() for p in text.split(",") if p.strip()]
        return [p for p in parts if p in self.BARCODE_FORMATS]


