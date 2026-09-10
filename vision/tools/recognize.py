# -*- coding: utf-8 -*-

import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from .base_tool import VisionTool, ToolResult, PipelineContext
from core.log_manager import log_warning


class ColorRecognition(VisionTool):
    display_name = "颜色识别"
    # 颜色类算子：参数配置对话框显示专属的"图像取色"交互
    SUPPORTS_COLOR_PICK: bool = True

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
        # 新颜色模型（双轨制：优先使用 color_model，否则回退旧六参数）
        self.params.setdefault("color_model", None)
        # 匹配方式: "range" / "distance" / "cluster"
        self.params.setdefault("match_mode", "range")
        # 距离匹配阈值
        self.params.setdefault("distance_threshold", 30.0)
        # 光照归一化 / 自适应阈值
        self.params.setdefault("normalize_illumination", False)
        self.params.setdefault("adaptive_threshold", False)
        # 颜色库实例（懒加载）
        self._color_library = None

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

        # 双轨制：优先使用新的 ColorModel，否则回退旧六参数区间
        model_dict = self.params.get("color_model")
        if model_dict:
            from vision.color.color_model import ColorModel
            from vision.color.color_matcher import ColorMatcher
            model = ColorModel.from_dict(model_dict)
            # 用参数面板的匹配方式/容差覆盖模型默认值（若用户调整过）
            model.match_mode = self.params.get("match_mode", model.match_mode)
            model.distance_threshold = float(
                self.params.get("distance_threshold", model.distance_threshold))
            model.normalize_illumination = bool(
                self.params.get("normalize_illumination", model.normalize_illumination))
            model.adaptive_threshold = bool(
                self.params.get("adaptive_threshold", model.adaptive_threshold))
            mask = ColorMatcher.build_mask(img, model)
            color_space = model.color_space
        else:
            if color_space == "Lab":
                converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            else:
                converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

        # 噪声抑制：形态学开闭运算
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        min_area = float(self.params.get("min_area", 100))
        color_area = np.sum(mask > 0)
        total_area = img.shape[0] * img.shape[1]
        area_ratio = (color_area / total_area) * 100 if total_area > 0 else 0

        # 诊断日志：对比预览与流水线运行时的输入差异
        try:
            input_source = self.params.get("_input_source") or self.params.get("input_source", "current")
            roi_info = ""
            if input_source.startswith("region:"):
                rname = input_source[7:]
                if rname in context.regions:
                    roi_info = f" ROI={context.regions[rname]}"
                else:
                    roi_info = f" ROI={rname}(未找到)"
            has_model = bool(self.params.get("color_model"))
            print(f"[ColorRecognition][诊断] 输入尺寸={img.shape[1]}x{img.shape[0]} "
                  f"输入源={input_source}{roi_info} color_model={has_model} "
                  f"匹配像素={color_area} 总像素={total_area} 占比={area_ratio:.2f}%")
        except Exception:  # noqa: BLE001
            pass

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

        # 颜色名称：优先使用 color_model 的名称，否则用 color_name 参数
        model_dict = self.params.get("color_model")
        if model_dict:
            color_name = model_dict.get("name", self.params.get("color_name", "红色"))
        else:
            color_name = self.params.get("color_name", "红色")

        result_data = {
            "color_area": int(color_area),
            "total_area": int(total_area),
            "area_ratio": float(area_ratio),
            "valid_regions": valid_count,
            "color_name": color_name,
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
                rx, ry, rw_, rh_ = context.regions[region_name]
                # ROI 随动:若该区域经 crop_rotated_rect 摆正裁剪(旋转矩形),
                # 轮廓坐标需按 +angle 绕区域中心逆旋转回投,不能只平移 bbox,
                # 否则识别描边相对实际区域会"旋转错位"(镜像 base_tool 分支:
                # 非轴对齐且区域出界时 base_tool 回退外接框普通裁剪 → 仅平移)。
                from vision.geometry_util import (is_axis_aligned,
                                                  crop_points_to_frame)
                rot = context.region_rot.get(region_name)
                rot_back = None
                if rot is not None:
                    rcx, rcy, rr_w, rr_h, rang = rot
                    axis = is_axis_aligned(rang)
                    need_rot = (not axis) or abs(rang) > 1e-3
                    if need_rot and not axis:
                        fh, fw = self._full_frame_image.shape[:2]
                        inside = (rcx - rr_w / 2 > 0 and rcy - rr_h / 2 > 0
                                  and rcx + rr_w / 2 < fw
                                  and rcy + rr_h / 2 < fh)
                        if not inside:
                            need_rot = False
                    if need_rot:
                        rot_back = (float(rcx), float(rcy), float(rang))
                # 将轮廓坐标从 ROI 局部坐标转换为完整帧坐标
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area >= min_area:
                        if rot_back is not None:
                            cnt_full = crop_points_to_frame(
                                cnt, img.shape[1], img.shape[0],
                                rot_back[0], rot_back[1], rot_back[2])
                        else:
                            cnt_full = cnt.copy()
                            cnt_full[:, :, 0] += rx
                            cnt_full[:, :, 1] += ry
                        cv2.drawContours(overlay, [cnt_full], -1,
                                         (0, 255, 0), 2)
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

    def _refresh_color_lib_combo(self):
        """刷新颜色库下拉列表（取色后调用，使新颜色出现在下拉中）。"""
        combo = getattr(self, "_color_lib_combo", None)
        if combo is None:
            return
        # 确保颜色库已初始化
        if getattr(self, "_color_library", None) is None:
            from vision.color.color_library import ColorLibrary
            self._color_library = ColorLibrary()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("-- 选择颜色库 --", None)
        for m in self._color_library.get_all():
            combo.addItem(f"[{m.source}] {m.name}", m.to_dict())
        # 若当前已有 color_model，选中对应项
        current = self.params.get("color_model")
        if current:
            name = current.get("name")
            for i in range(combo.count()):
                data = combo.itemData(i)
                if data and data.get("name") == name:
                    combo.setCurrentIndex(i)
                    break
        combo.blockSignals(False)

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QSpinBox, QDoubleSpinBox,
                                      QHBoxLayout, QWidget, QLabel, QSlider, QCheckBox)
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

        # 颜色库选择（预设 + 自定义 + 临时）
        from vision.color.color_library import ColorLibrary
        if self._color_library is None:
            self._color_library = ColorLibrary()
        lib_combo = QComboBox(parent)
        self._color_lib_combo = lib_combo  # 保存引用，供取色后刷新
        self._refresh_color_lib_combo()

        def on_lib_changed(idx):
            data = lib_combo.itemData(idx)
            if data is None:
                return
            self.params["color_model"] = data
            self.params["color_name"] = data.get("name", "自定义颜色")
            self.params["color_space"] = data.get("color_space", "HSV")
            self.params["match_mode"] = data.get("match_mode", "range")
            self.params["distance_threshold"] = data.get("distance_threshold", 30.0)
            self.params["normalize_illumination"] = data.get("normalize_illumination", False)
            self.params["adaptive_threshold"] = data.get("adaptive_threshold", False)

        lib_combo.currentIndexChanged.connect(on_lib_changed)
        widgets.append(("颜色库:", lib_combo))

        # 匹配方式
        mode_combo = QComboBox(parent)
        mode_combo.addItem("区间匹配", "range")
        mode_combo.addItem("距离匹配", "distance")
        mode_combo.addItem("聚类匹配", "cluster")
        current_mode = self.params.get("match_mode", "range")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"match_mode": mode_combo.itemData(i)}))
        widgets.append(("匹配方式:", mode_combo))

        # 距离阈值（distance/cluster 模式）
        dist_spin = QDoubleSpinBox(parent)
        dist_spin.setRange(1, 500)
        dist_spin.setSingleStep(5)
        dist_spin.setValue(float(self.params.get("distance_threshold", 30.0)))
        dist_spin.valueChanged.connect(
            lambda v: self.params.update({"distance_threshold": v}))
        widgets.append(("距离阈值:", dist_spin))

        # 光照归一化
        norm_cb = QCheckBox(parent)
        norm_cb.setChecked(self.params.get("normalize_illumination", False))
        norm_cb.stateChanged.connect(
            lambda v: self.params.update({"normalize_illumination": bool(v)}))
        widgets.append(("光照归一化:", norm_cb))

        # 自适应阈值
        adapt_cb = QCheckBox(parent)
        adapt_cb.setChecked(self.params.get("adaptive_threshold", False))
        adapt_cb.stateChanged.connect(
            lambda v: self.params.update({"adaptive_threshold": bool(v)}))
        widgets.append(("自适应阈值:", adapt_cb))

        # 颜色下拉（向后兼容，选择预设颜色）
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
        self.params.setdefault("enable_qr", True)
        self.params.setdefault("barcode_formats",
                               ["CODE_128", "CODE_39", "EAN_13", "UPC_A"])
        # 反色增强(浅色码/镭雕码:亮条码在深色板面上)
        self.params.setdefault("try_inverted", True)
        # 调试用:把算子实际收到的输入图写到 _debug_roi/operator_input.png
        # (识别失败时总会写,便于现场排查;成功时默认不写,避免产线每片写盘)
        self.params.setdefault("debug_dump_input", False)
        self._pyzbar_available = None
        self._qr_detector = None
        self._barcode_detector = None
        self._barcode_detector_tried = False
        self._last_variants = []      # 最近一次解码尝试的策略标签(调试用)

    # ---- 解码器(惰性创建,避免每次重建) ----

    def _get_qr_detector(self):
        if self._qr_detector is None:
            try:
                self._qr_detector = cv2.QRCodeDetector()
            except Exception:  # noqa: BLE001
                self._qr_detector = False
        return self._qr_detector or None

    def _get_barcode_detector(self):
        """OpenCV 一维码检测器(EAN/UPC/Code128 等,作为 pyzbar 的补充)。"""
        if not self._barcode_detector_tried:
            self._barcode_detector_tried = True
            try:
                if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
                    self._barcode_detector = cv2.barcode.BarcodeDetector()
            except Exception as e:  # noqa: BLE001
                print(f"[DEBUG][QRCodeRecognize] OpenCV 一维码检测器不可用: {e}")
                self._barcode_detector = None
        return self._barcode_detector

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

        # 识别所有条码（二维码 + 一维码）:统一多策略解码
        barcodes = self._decode_all(gray)
        # 去重（按内容 + 位置）
        barcodes = self._deduplicate(barcodes)

        # 调试日志：输出算子实际收到的图像尺寸与识别结果
        print(f"[DEBUG][QRCodeRecognize] 输入图像 shape={img.shape} dtype={img.dtype} "
              f"input_source={self.params.get('_input_source') or self.params.get('input_source', 'current')} "
              f"识别到 {len(barcodes)} 个条码: {[b.get('data') for b in barcodes]}")
        if not barcodes:
            print(f"[DEBUG][QRCodeRecognize] 未识别到条码,已尝试策略: {self._last_variants}")

        # 排查用:识别失败时(或显式开启 debug_dump_input)保存算子收到的输入图
        if (not barcodes) or self.params.get("debug_dump_input", False):
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
        # 注意：若使用 ROI 输入源，img 是 ROI 局部图像，需把标注回投到完整帧坐标。
        # 若该 ROI 是"位置修正随动"的旋转矩形(base_tool 已用 crop_rotated_rect
        # 摆正后交给本算子),回投必须做逆旋转,不能只平移 bbox 原点——否则标注
        # 会相对真实条码错位(与颜色识别描边同类问题)。
        input_source = self.params.get("_input_source") or self.params.get("input_source", "current")
        rot_back = None
        rx, ry = 0, 0
        if input_source.startswith("region:") and self._full_frame_image is not None:
            overlay = np.zeros_like(self._full_frame_image)
            region_name = input_source[7:]
            if region_name in context.regions:
                rx, ry, _, _ = context.regions[region_name]
            rot = context.region_rot.get(region_name)
            if rot is not None:
                from vision.geometry_util import (is_axis_aligned,
                                                  crop_points_to_frame)
                rcx, rcy, rw, rh, rang = rot
                axis = is_axis_aligned(rang)
                need_rot = (not axis) or abs(rang) > 1e-3
                if need_rot and not axis:
                    # 镜像 base_tool:非轴对齐且区域出界时回退为外接框普通裁剪
                    fh, fw = self._full_frame_image.shape[:2]
                    inside = (rcx - rw / 2 > 0 and rcy - rh / 2 > 0
                              and rcx + rw / 2 < fw and rcy + rh / 2 < fh)
                    if not inside:
                        need_rot = False
                if need_rot:
                    rot_back = (float(rcx), float(rcy), float(rang))
        else:
            overlay = np.zeros_like(img)

        for bc in barcodes:
            x, y, w, h = bc["bbox"]
            # 回投到完整帧坐标（ROI 模式）:旋转随动 ROI 需逆旋转
            if rot_back is not None:
                corners = np.array([[[x, y]], [[x + w, y]],
                                    [[x + w, y + h]], [[x, y + h]]],
                                   dtype=np.float32)
                back = crop_points_to_frame(corners, img.shape[1], img.shape[0],
                                            rot_back[0], rot_back[1],
                                            rot_back[2])
                pts = back.reshape(-1, 2)
                x, y, w, h = cv2.boundingRect(pts)
            else:
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
                        cv2.FONT_HERSHEY_SIMPLEX, 5, color, 3)

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

    # ==================================================================
    # 统一多策略解码(二维码 + 一维码)
    # ==================================================================

    # 单个解码变体的像素上限:放大后超过该值就不再试(保护耗时与内存)
    MAX_VARIANT_PIXELS = 6_000_000
    # 全面模式的像素上限(更保守:此阶段变体多,避免单片耗时过长)
    MAX_THOROUGH_VARIANT_PIXELS = 4_000_000
    # 整体解码时间预算(毫秒):超时即停止继续扩展策略,保证产线节拍可控
    DECODE_TIME_BUDGET_MS = 1200
    # 大图(超过该像素)启用"先粗定位条码区域、再局部放大"策略
    CANDIDATE_REGION_MIN_PIXELS = 1_200_000

    # ---- 预处理小工具 ----

    @staticmethod
    def _odd_at_least(value: int, limit: int = 0) -> int:
        """返回不小于 value 的奇数(自适应阈值的 blockSize 必须为奇数)。"""
        b = int(value)
        if b % 2 == 0:
            b += 1
        if limit > 0:
            max_odd = limit if limit % 2 == 1 else limit - 1
            if max_odd >= 3 and b > max_odd:
                b = max_odd
        return max(3, b)

    @staticmethod
    def _resize(gray: np.ndarray, scale: int) -> np.ndarray:
        return cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _bin_otsu(gray: np.ndarray) -> np.ndarray:
        return cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @staticmethod
    def _bin_adaptive(gray: np.ndarray, block: int, c: int = 10) -> np.ndarray:
        h, w = gray.shape[:2]
        b = QRCodeRecognize._odd_at_least(block, min(h, w))
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, b, c)

    @staticmethod
    def _clahe(gray: np.ndarray) -> np.ndarray:
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    @staticmethod
    def _unsharp(gray: np.ndarray, amount: float = 0.6) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)

    def _polarities(self, gray: np.ndarray):
        """返回 (前缀, 基准图) 列表:常规 + 反色(亮码/镭雕码在深底)。

        注意:反色必须"先取反得到新基准图,再套同一套增强"。
        实测"对增强结果取反"得到的仍是亮条码,解码器读不出。
        """
        out = [("", gray)]
        if self.params.get("try_inverted", True):
            out.append(("inv_", cv2.bitwise_not(gray)))
        return out

    # ---- 变体生成 ----

    def _build_variants(self, gray: np.ndarray, thorough: bool = False):
        """生成解码变体 (tag, image, scale)。

        快通道(命中即返回):两种极性的 raw + 自适应(31),再按像素预算加
        2x 放大 —— 多数清晰条码一轮即中,保证产线单片耗时可控。
        全面模式(仅快通道无结果时):补充 Otsu、自适应(15/51)、CLAHE、
        锐化,以及 2x/3x/4x 放大(大图改为"局部候选区域放大",见
        _candidate_variants,避免整图放大导致耗时爆炸)。

        说明:
          - pyzbar(zbar)内部会按 0/90/180/270 四个方向扫描,无需手工旋转;
          - 不再限制"图像小于 800px 才放大":真实 ROI 常更大、条码只占
            几十像素,必须放大才可能解出;
          - 实测同一张真实 1D 样本:raw 解不出,adap31/51 与 2x+CLAHE 能解出,
            故"多种自适应核 + 多尺度"缺一不可。
        """
        h, w = gray.shape[:2]
        px = int(h) * int(w)
        polarities = self._polarities(gray)

        if not thorough:
            for ptag, base in polarities:
                yield f"{ptag}raw", base, 1.0
                yield f"{ptag}adap31", self._bin_adaptive(base, 31), 1.0
            if px * 4 <= self.MAX_VARIANT_PIXELS:
                for ptag, base in polarities:
                    up = self._resize(base, 2)
                    yield f"{ptag}x2", up, 2.0
                    yield f"{ptag}x2_adap63", self._bin_adaptive(up, 63), 2.0
            return

        # ---- 全面模式:1x 增强(两种极性) ----
        for ptag, base in polarities:
            yield f"{ptag}otsu", self._bin_otsu(base), 1.0
            for bs in (15, 51):
                yield f"{ptag}adap{bs}", self._bin_adaptive(base, bs), 1.0
            yield f"{ptag}clahe", self._clahe(base), 1.0
            yield f"{ptag}unsharp", self._unsharp(base), 1.0

        # ---- 全面模式:小图整图多尺度放大 ----
        if px <= self.CANDIDATE_REGION_MIN_PIXELS:
            for scale in (2, 3, 4):
                if px * scale * scale > self.MAX_THOROUGH_VARIANT_PIXELS:
                    continue
                for ptag, base in polarities:
                    up = self._resize(base, scale)
                    yield f"{ptag}x{scale}", up, float(scale)
                    yield (f"{ptag}x{scale}_adap",
                           self._bin_adaptive(up, 31 * scale), float(scale))
                    yield f"{ptag}x{scale}_clahe", self._clahe(up), float(scale)

    def _candidate_variants(self, gray: np.ndarray):
        """大图策略:先粗定位"像条码"的区域,再对这些小区域局部放大解码。

        整图放大在 1MP 以上会非常慢;而条码在 ROI 里通常只占一小块,
        因此用形态学梯度找"边缘密集"的连通块 → 裁出候选区域 → 放大 2~4x
        解码,既快又对小条码更敏感。

        yield: (tag, image, scale, offset_x, offset_y)
        """
        h, w = gray.shape[:2]
        for idx, (x0, y0, x1, y1) in enumerate(self._find_barcode_regions(gray)):
            sub = gray[y0:y1, x0:x1]
            sh, sw = sub.shape[:2]
            if sh < 24 or sw < 24:
                continue
            for ptag, base in self._polarities(sub):
                # 候选区域很小,1x 先用最有效的自适应阈值
                yield (f"c{idx}{ptag}adap31",
                       self._bin_adaptive(base, 31), 1.0, x0, y0)
                # 局部放大 2x/3x 是最有价值的一档(小条码靠这个解出)
                for scale in (2, 3):
                    if (sw * scale) * (sh * scale) > self.MAX_THOROUGH_VARIANT_PIXELS:
                        continue
                    up = self._resize(base, scale)
                    yield (f"c{idx}{ptag}x{scale}", up, float(scale), x0, y0)
                    yield (f"c{idx}{ptag}x{scale}_adap",
                           self._bin_adaptive(up, 31 * scale), float(scale),
                           x0, y0)

    @staticmethod
    def _find_barcode_regions(gray: np.ndarray, max_regions: int = 3,
                              max_side: int = 640, min_side: int = 32):
        """粗定位疑似条码区域,返回 [(x0,y0,x1,y1)]（最多 max_regions 个）。

        原理:条码是"高频、方向一致的条纹",形态学梯度响应强且成片;
        用梯度 → Otsu → 闭运算 → 连通块,按面积取前几个。
        """
        try:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
            _, bw = cv2.threshold(grad, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = cv2.morphologyEx(
                bw, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
            n, _labels, stats, _cent = cv2.connectedComponentsWithStats(bw, 8)
            boxes = []
            for i in range(1, n):
                x, y, bw_, bh_, area = stats[i]
                if bw_ < min_side or bh_ < min_side:
                    continue
                boxes.append((int(area), int(x), int(y), int(bw_), int(bh_)))
            boxes.sort(reverse=True)
            out = []
            h, w = gray.shape[:2]
            for _area, x, y, bw_, bh_ in boxes:
                pad = int(0.15 * max(bw_, bh_))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(w, x + bw_ + pad), min(h, y + bh_ + pad)
                if (x1 - x0) > max_side or (y1 - y0) > max_side:
                    # 区域过大:说明整图都像高频内容(如噪声),不裁剪
                    continue
                out.append((x0, y0, x1, y1))
                if len(out) >= max_regions:
                    break
            return out
        except Exception:  # noqa: BLE001
            return []

    # ---- 解码主流程 ----

    def _decode_all(self, gray: np.ndarray) -> list:
        """统一多策略解码(二维码 + 一维码),返回条码列表。

        顺序:快通道(约 4~8 个变体,命中即返回)→ 全面模式(1x 增强 +
        多尺度/候选区域放大)。整个过程受时间预算约束,避免无码图像上
        无限尝试导致产线节拍失控。
        """
        results: list = []
        self._last_variants = []
        try:
            if float(gray.std()) < 3.0:
                # 近乎纯色(纯白/纯黑)→ 不可能有条码,直接返回
                return results
        except Exception:  # noqa: BLE001
            pass

        budget = float(self.params.get("decode_time_budget_ms",
                                       self.DECODE_TIME_BUDGET_MS)) / 1000.0
        t0 = time.perf_counter()
        tried = set()

        def _try_variant(tag, im, scale, ox=0, oy=0):
            if tag in tried:
                return False
            tried.add(tag)
            self._decode_variant(im, tag, scale, results, ox, oy)
            return bool(results)

        # ── 第一轮:快通道(整图,命中即返回;约 4~8 个变体) ──
        # 先跑整图:真实图上一轮解码只要几十毫秒,是命中率最高、代价最低的路径。
        phase1_deadline = t0 + budget * 0.5
        for tag, im, scale in self._build_variants(gray, thorough=False):
            if _try_variant(tag, im, scale):
                return results
            if time.perf_counter() > phase1_deadline:
                break

        # ── 第二轮:大图候选区域局部放大(取部分预算,避免挤掉后续整图策略) ──
        h, w = gray.shape[:2]
        large = int(h) * int(w) > self.CANDIDATE_REGION_MIN_PIXELS
        if large:
            phase2_deadline = t0 + budget * 0.8
            for tag, im, scale, ox, oy in self._candidate_variants(gray):
                if _try_variant(tag, im, scale, ox, oy):
                    return results
                if time.perf_counter() > phase2_deadline:
                    break

        # ── 第三轮:全面模式(1x 增强 + 多尺度) ──
        for tag, im, scale in self._build_variants(gray, thorough=True):
            if _try_variant(tag, im, scale):
                return results
            if time.perf_counter() - t0 > budget:
                print(f"[DEBUG][QRCodeRecognize] 解码超时({budget * 1000:.0f}ms),"
                      f"提前结束,已试 {len(tried)} 个策略")
                return results
        return results

    def _decode_variant(self, im: np.ndarray, tag: str, scale: float,
                        out: list, offset_x: int = 0,
                        offset_y: int = 0) -> None:
        """对单个变体依次尝试各解码器,结果追加到 out。

        offset_x/offset_y:该变体相对完整 ROI 的裁剪偏移(候选区域策略用)。
        """
        self._last_variants.append(tag)
        # 1) pyzbar(zbar):一次调用可同时返回二维码与一维码
        if self._check_pyzbar():
            try:
                from pyzbar import pyzbar
                for d in pyzbar.decode(im):
                    self._append_pyzbar(out, d, scale, offset_x, offset_y)
                if out:
                    return
            except Exception as e:  # noqa: BLE001
                print(f"[DEBUG][QRCodeRecognize] pyzbar 解码异常({tag}): {e}")
        # 2) OpenCV 二维码检测器(部分二维码比 zbar 更稳)
        if self.params.get("enable_qr", True):
            detector = self._get_qr_detector()
            if detector is not None:
                try:
                    data, points, _ = detector.detectAndDecode(im)
                    if data:
                        self._append_2d(out, data, points, scale, "QR",
                                        offset_x, offset_y)
                except Exception:  # noqa: BLE001
                    pass
        # 3) OpenCV 一维码检测器(补充 EAN/UPC/Code128)
        if self.params.get("enable_1d", True):
            bd = self._get_barcode_detector()
            if bd is not None:
                try:
                    ret = bd.detectAndDecodeWithType(im)
                    ok = ret[0]
                    info = ret[1] if len(ret) > 1 else None
                    types = ret[2] if len(ret) > 2 else None
                    quads = ret[3] if len(ret) > 3 else None
                    if ok:
                        infos = info if isinstance(info, (list, tuple)) else [info]
                        typs = (types if isinstance(types, (list, tuple))
                                else [types])
                        for i, txt in enumerate(infos):
                            txt = str(txt or "").strip()
                            if not txt:
                                continue
                            btype = str(typs[i]) if i < len(typs) else "1D"
                            quad = None
                            if quads is not None and len(quads) > i:
                                quad = quads[i]
                            self._append_1d(out, txt, btype, quad, scale,
                                            offset_x, offset_y)
                except Exception:  # noqa: BLE001
                    pass

    # ---- 结果整理 ----

    def _append_pyzbar(self, out: list, decoded, scale: float,
                       offset_x: int = 0, offset_y: int = 0) -> None:
        """把 pyzbar 的单条结果整理为内部结构。"""
        data = (decoded.data.decode("utf-8", errors="replace")
                if decoded.data else "")
        if not data:
            return
        btype = (getattr(decoded, "type", "") or "").upper()
        confidence = min(1.0, float(getattr(decoded, "quality", 100)) / 100.0)
        bbox = self._bbox_from_pyzbar(decoded, scale, offset_x, offset_y)
        if btype in ("QRCODE", "QR", "MICROQRCODE", "DATAMATRIX", "AZTEC",
                     "PDF417", "MAXICODE"):
            if not self.params.get("enable_qr", True):
                return
            typ = "QR" if "QR" in btype else "DM"
            out.append({"type": typ, "data": data, "confidence": confidence,
                        "barcode_type": btype, "bbox": bbox})
            return
        # 一维码:受"启用一维码 + 格式集合"过滤
        if not self.params.get("enable_1d", True):
            return
        if not self._format_allowed(btype):
            return
        out.append({"type": "1D", "data": data, "confidence": confidence,
                    "barcode_type": btype, "bbox": bbox})

    def _append_2d(self, out: list, data: str, points, scale: float,
                   typ: str, offset_x: int = 0, offset_y: int = 0) -> None:
        out.append({"type": typ, "data": data, "confidence": 1.0,
                    "barcode_type": "QRCODE" if typ == "QR" else typ,
                    "bbox": self._bbox_from_points(points, scale,
                                                   offset_x, offset_y)})

    def _append_1d(self, out: list, data: str, btype: str, quad,
                   scale: float, offset_x: int = 0, offset_y: int = 0) -> None:
        if not self._format_allowed(btype):
            return
        out.append({"type": "1D", "data": data, "confidence": 1.0,
                    "barcode_type": btype,
                    "bbox": self._bbox_from_points(quad, scale,
                                                   offset_x, offset_y)})

    def _bbox_from_pyzbar(self, decoded, scale: float,
                          offset_x: int = 0, offset_y: int = 0) -> tuple:
        """pyzbar 结果的 bbox:优先用四点多边形。

        旋转(如竖放)条码的 rect 会退化成 1px 宽/高,多边形才是真实位置,
        否则标注点与"条码区域"会对不上。
        """
        pts = getattr(decoded, "polygon", None)
        if pts:
            return self._bbox_from_points(pts, scale, offset_x, offset_y)
        r = decoded.rect
        return self._sanitize_bbox(int(r.left / scale) + offset_x,
                                   int(r.top / scale) + offset_y,
                                   int(r.width / scale), int(r.height / scale))

    def _bbox_from_points(self, points, scale: float,
                          offset_x: int = 0, offset_y: int = 0) -> tuple:
        if points is None:
            return (0, 0, 0, 0)
        try:
            arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            if arr.size == 0:
                return (0, 0, 0, 0)
            x, y, w, h = cv2.boundingRect(
                (arr / max(1e-6, float(scale))).astype(np.float32))
            return self._sanitize_bbox(int(x) + offset_x, int(y) + offset_y,
                                       int(w), int(h))
        except Exception:  # noqa: BLE001
            return (0, 0, 0, 0)

    @staticmethod
    def _sanitize_bbox(x: int, y: int, w: int, h: int) -> tuple:
        """保证标注框可见:过小的框补到最小尺寸(仅用于标注/日志)。"""
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        if w < 20:
            x = max(0, x - (20 - w) // 2)
            w = 20
        if h < 20:
            y = max(0, y - (20 - h) // 2)
            h = 20
        return (x, y, w, h)

    # ---- 兼容旧接口(内部统一走 _decode_all) ----

    def _decode_qr(self, gray: np.ndarray) -> list:
        """仅返回二维码/DataMatrix 结果(兼容旧接口)。"""
        return [b for b in self._decode_all(gray)
                if b.get("type") in ("QR", "DM")]

    def _decode_1d(self, gray: np.ndarray) -> list:
        """仅返回一维码结果(兼容旧接口)。"""
        return [b for b in self._decode_all(gray) if b.get("type") == "1D"]

    def _format_allowed(self, btype: str) -> bool:
        """判断一维码格式是否在允许集合内。

        btype 为解码器返回的类型名:pyzbar 用 CODE128,OpenCV 可能用
        CODE_128,因此这里先把传入类型名统一归一化后再比较,
        避免"配置了 CODE_128 却因命名差异把结果过滤掉"。
        配置的 barcode_formats 使用标准名（如 CODE_128）。
        """
        formats = self.params.get("barcode_formats", [])
        if not formats:
            return True
        norm = (btype or "").upper()
        allowed_types = set()
        for fmt in formats:
            f = str(fmt).upper()
            allowed_types.add(self.BARCODE_FORMATS.get(f, f))
        # 归一化传入类型(CODE_128 -> CODE128;EAN_13 -> EAN13 ...)
        norm = self.BARCODE_FORMATS.get(norm, norm)
        return norm in allowed_types

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

        # 是否启用二维码识别
        enable_qr_cb = QCheckBox("启用二维码识别")
        enable_qr_cb.setChecked(bool(self.params.get("enable_qr", True)))
        enable_qr_cb.stateChanged.connect(
            lambda s: self.params.update({"enable_qr": bool(s)}))
        widgets.append(("", enable_qr_cb))

        # 反色增强（浅色/镭雕码在深色板面上）
        inv_cb = QCheckBox("反色增强(浅色码/镭雕码)")
        inv_cb.setToolTip("亮条码/镭雕码落在深色板面上时,自动尝试反色后解码")
        inv_cb.setChecked(bool(self.params.get("try_inverted", True)))
        inv_cb.stateChanged.connect(
            lambda s: self.params.update({"try_inverted": bool(s)}))
        widgets.append(("", inv_cb))

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


