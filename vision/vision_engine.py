# -*- coding: utf-8 -*-
import os
import json
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

from .pipeline import Pipeline
from .tools.base_tool import ToolResult
from core.log_manager import log_error, log_info
from core.result_storage import ResultStorage


class VisionEngine:
    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._last_results: List[ToolResult] = []

    @property
    def pipeline(self) -> Optional[Pipeline]:
        return self._pipeline

    def set_pipeline(self, pipeline: Pipeline):
        self._pipeline = pipeline

    def set_pipeline_by_name(self, scheme_name: str) -> bool:
        """按方案名称加载并设置流水线

        Args:
            scheme_name: 方案名称（不含 .json 后缀）

        Returns:
            bool: 是否加载成功
        """
        from core.paths import SCHEME_DIR
        import os, json

        scheme_path = os.path.join(SCHEME_DIR, f"{scheme_name}.json")
        if not os.path.exists(scheme_path):
            log_error(f"方案文件不存在: {scheme_path}")
            return False

        try:
            with open(scheme_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pipeline = Pipeline.from_dict(data)
            self._pipeline = pipeline
            log_info(f"按名称加载方案 [{scheme_name}] 成功")
            return True
        except Exception as e:
            log_error(f"加载方案 [{scheme_name}] 失败: {e}")
            return False

    def clear_pipeline(self):
        self._pipeline = None

    def execute(self, cv_image: np.ndarray, scheme_name: str = "",
                product_id: str = "") -> Tuple[bool, str, np.ndarray]:
        if self._pipeline is None:
            return False, "未设置流水线", cv_image

        try:
            passed, results, current_image, step_roi_map = self._pipeline.execute(cv_image)
            self._last_results = results

            # 在原图上叠加所有工具的 overlay_image，生成标注结果图
            annotated = cv_image.copy()
            for r in results:
                if r.overlay_image is not None:
                    overlay = r.overlay_image
                    # 尺寸不一致时，将 overlay 缩放到与 annotated 一致
                    if overlay.shape[:2] != annotated.shape[:2]:
                        overlay = cv2.resize(
                            overlay,
                            (annotated.shape[1], annotated.shape[0])
                        )
                    # 通过 mask 只叠加非黑色部分（标注内容）
                    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    bg = cv2.bitwise_and(annotated, annotated, mask=mask_inv)
                    fg = cv2.bitwise_and(overlay, overlay, mask=mask)
                    annotated = cv2.add(bg, fg)

            # ── 根据每个算子的检测结果，在 ROI 框上显示 OK/NG 颜色 ──
            self._draw_roi_results(annotated, cv_image, step_roi_map)

            if passed:
                message = "检测通过 (OK)"
                log_info(f"检测OK | 方案={scheme_name} | 产品={product_id}")
            else:
                message = "检测失败 (NG)"
                log_info(f"检测NG | 方案={scheme_name} | 产品={product_id}")

            return passed, message, annotated

        except Exception as e:
            error_msg = f"检测执行异常: {str(e)}"
            log_error(error_msg)
            self._last_results = []
            return False, error_msg, cv_image

    def get_last_results(self) -> List[ToolResult]:
        return self._last_results

    def _draw_roi_results(self, annotated: np.ndarray, raw_image: np.ndarray,
                          step_roi_map: Dict[int, str] = None):
        """根据每个算子的检测结果，在 ROI 框上绘制 OK(绿)/NG(红) 颜色。

        遍历 Pipeline 中的所有步骤：
        1. 从 MultiROI 工具中提取 ROI 区域定义 {name: (x, y, w, h)}
        2. 通过 step_roi_map（由 Pipeline.execute() 构建）将步骤索引映射到 ROI 名称
        3. 从 results 中获取每个步骤的 passed 值，建立 {roi_name: passed} 映射
        4. 在 annotated 图像上绘制带颜色的 ROI 框

        Args:
            annotated: 标注图像（会被修改）
            raw_image: 原始图像，用于计算百分比坐标
            step_roi_map: {step_index: roi_name}，由 Pipeline.execute() 在运行时构建
        """
        if self._pipeline is None:
            return

        # 收集 ROI 区域信息 {roi_name: (x, y, w, h)}
        roi_regions = {}
        # 收集 ROI 对应的算子检测结果 {roi_name: passed(bool)}
        roi_results = {}

        results = self._last_results
        if step_roi_map is None:
            step_roi_map = {}

        # 第一步：从 MultiROI 工具中提取所有 ROI 区域定义
        for step in self._pipeline.steps:
            if not step.enabled:
                continue
            tool_type = type(step.tool).__name__
            if tool_type == "MultiROI":
                raw_regions = step.tool.params.get("regions", [])
                use_pct = step.tool.params.get("use_percentage", False)
                h_img, w_img = raw_image.shape[:2]

                for r in raw_regions:
                    if isinstance(r, dict) and r.get("enabled", True):
                        name = r.get("name", "未命名")
                        if use_pct:
                            x = int(r.get("x", 0) / 100.0 * w_img)
                            y = int(r.get("y", 0) / 100.0 * h_img)
                            w = int(r.get("width", r.get("w", 100)) / 100.0 * w_img)
                            h = int(r.get("height", r.get("h", 100)) / 100.0 * h_img)
                        else:
                            x = r.get("x", 0)
                            y = r.get("y", 0)
                            w = r.get("width", r.get("w", 100))
                            h = r.get("height", r.get("h", 100))
                        roi_regions[name] = (x, y, w, h)

        # 第二步：通过 step_roi_map 建立 ROI → 检测结果的映射
        # step_roi_map 由 Pipeline.execute() 在运行时构建，记录了每个步骤使用的 ROI 名称
        # 这种方式比仅依赖 _input_source 参数更可靠，因为有些步骤可能没有 _input_source 参数
        # （例如旧方案文件），但运行时仍然会通过 _get_input_image 从 ROI 裁剪图像。
        for step_idx, roi_name in step_roi_map.items():
            if step_idx < len(results):
                roi_results[roi_name] = results[step_idx].passed
            else:
                # 步骤因前面步骤失败导致 break 未执行到，视为 NG
                roi_results[roi_name] = False

        # 第三步：在 annotated 上绘制 ROI 结果
        for roi_name, (x, y, w, h) in roi_regions.items():
            if roi_name in roi_results:
                passed = roi_results[roi_name]
                color = (0, 255, 0) if passed else (0, 0, 255)  # 绿/红
                thickness = 3  # 加粗边框突出显示
                label = "OK" if passed else "NG"
            else:
                # 未被引用的 ROI：不绘制，避免检测后残留预览时的绿色框
                continue

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            # ROI 名称文字（放大）
            cv2.putText(annotated, roi_name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if label:
                # 在 ROI 框右上角添加 OK/NG 标签（放大）
                (label_w, _), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(annotated, label,
                            (x + w - label_w - 5, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def save_error_data(self, scheme_name, product_id,
                        annotated_image, custom_prefix=None):
        """
        保存检测失败的标注结果图到 production data 目录。

        目录结构:
            data/production data/
                YYYY-MM-DD/
                    {SN号}/
                        {SN号}_{HHMMSS}_pos{序号}_{位置名}_NG.jpg  # 标注结果图
                        {SN号}.xml                                   # MES 上传用 XML
                    未识别到SN号/
                        ...

        Args:
            scheme_name: 方案名称
            product_id: 产品ID（一维码数据）
            annotated_image: 标注图像
            custom_prefix: 自定义前缀（如一维码），用于文件夹和文件名
        """
        try:
            storage = ResultStorage()
            pid = custom_prefix or product_id
            storage.save_position_data(
                scheme_name=scheme_name,
                sn=pid,
                position_name="NG",
                position_index=1,
                annotated_image=annotated_image,
                passed=False,
            )
            log_info(f"NG 数据已保存 (product_id={pid})")
        except Exception as e:
            log_error(f"保存 NG 数据失败: {e}")
