# -*- coding: utf-8 -*-
"""
位置修正算子 (PositionCorrect)
================================
解决"产品摆放偏移/翻转导致固定 ROI 误判"的问题。

原理:
    在参考图(产品标准摆放)上框选一个稳定特征作为"基准模板"(如板角、
    丝印、mark),记录其参考位置与角度(=0°)。运行时对当前图像做一次
    轻量模板匹配(支持 0°/180° 翻转 / 小角度范围),求出特征当前中心
    与角度,然后构造仿射变换把整幅图像"校正回参考姿态":
        - 绕图像中心旋转 -matched_angle,再平移使特征中心回到参考中心。
    校正后的图像作为 processed_image 输出给下游,MultiROI 及后续算子
    仍按参考坐标框选即可自动对准产品,无需感知偏移。

参数(params):
    template_b64 : 基准模板图像(PNG, base64),由配置界面框选生成
    ref_x/ref_y  : 特征参考位置(模板在参考图中的左上角)
    ref_w/ref_h  : 模板尺寸
    rot_mode     : "0"(仅0°,最快) / "0and180"(0°与180°翻转)
                   / "range"(小角度范围)
    angle_min/max/step : range 模式角度搜索范围(度)
    threshold    : 匹配分数阈值(0~1),低于阈值判 NG

输出:
    ToolResult.passed      : 是否成功定位并校正
    data.dx/dy             : 特征中心相对参考中心的像素偏移(校正前)
    data.angle_deg         : 匹配到的最优角度
    data.score             : 匹配分数
    processed_image        : 校正后的图像(未定位成功时返回原图)
"""

import base64
import math
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .base_tool import VisionTool, ToolResult, PipelineContext
from core.log_manager import log_info, log_error, log_warning

# 若图像变化导致模板需要按比例缩放时的上限(与 TemplateMatch 约定一致)
MAX_TEMPLATE_LONG_SIDE = 600


def _encode_png_bgr(img_bgr: np.ndarray) -> str:
    """把 BGR 图编码为 PNG base64(用于存入方案 JSON)。"""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("模板图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_png_bgr(b64: str) -> Optional[np.ndarray]:
    """从 PNG base64 解码为 BGR 图。"""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log_error(f"解码模板失败: {e}")
        return None


def _rotate_gray(img: np.ndarray, angle: float,
                 border: int = 0) -> np.ndarray:
    """把灰度图绕中心旋转 angle 度(角度正=逆时针)。"""
    h, w = img.shape[:2]
    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    # 180° 用精确翻转,避免插值损耗
    if abs(angle % 360.0) < 1e-6:
        return img.copy()
    if abs((angle - 180.0) % 360.0) < 1e-6:
        return cv2.rotate(img, cv2.ROTATE_180)
    M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


class PositionCorrect(VisionTool):
    display_name = "位置修正"
    # 需要独立配置界面(在参考图上框选基准),不适用通用参数对话框
    SUPPORTS_CUSTOM_DIALOG: bool = True

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("template_b64", "")
        self.params.setdefault("ref_x", 0)
        self.params.setdefault("ref_y", 0)
        self.params.setdefault("ref_w", 0)
        self.params.setdefault("ref_h", 0)
        self.params.setdefault("rot_mode", "0and180")
        self.params.setdefault("angle_min", -10)
        self.params.setdefault("angle_max", 10)
        self.params.setdefault("angle_step", 2)
        self.params.setdefault("threshold", 0.7)
        self._template_cache: Optional[np.ndarray] = None

    # ── 模板读写 ──

    def set_template(self, template_bgr: np.ndarray, ref_x: int, ref_y: int):
        """配置界面调用:写入模板图与参考位置。"""
        self._template_cache = template_bgr
        self.params["template_b64"] = _encode_png_bgr(template_bgr)
        self.params["ref_x"] = int(ref_x)
        self.params["ref_y"] = int(ref_y)
        h, w = template_bgr.shape[:2]
        self.params["ref_w"] = int(w)
        self.params["ref_h"] = int(h)

    def _get_template(self) -> Optional[np.ndarray]:
        if self._template_cache is not None:
            return self._template_cache
        self._template_cache = _decode_png_bgr(self.params.get("template_b64", ""))
        return self._template_cache

    # ── 候选角度 ──

    def _candidate_angles(self) -> list:
        mode = self.params.get("rot_mode", "0and180")
        if mode == "0":
            return [0.0]
        if mode == "0and180":
            return [0.0, 180.0]
        amin = float(self.params.get("angle_min", -10))
        amax = float(self.params.get("angle_max", 10))
        astep = float(self.params.get("angle_step", 2))
        if astep <= 0:
            return [0.0]
        vals = [0.0]
        a = amin
        while a <= amax + 1e-6:
            if abs(a) > 1e-6:
                vals.append(round(float(a), 3))
            a += astep
        return sorted(set(vals))

    # ── 核心:定位 + 校正 ──

    def process(self, context: PipelineContext) -> ToolResult:
        img = self._get_input_image(context)
        if img is None:
            return ToolResult(success=False, passed=False,
                              message="无输入图像")

        template = self._get_template()
        if template is None or not self.params.get("ref_w"):
            return ToolResult(success=False, passed=False,
                              processed_image=img.copy(),
                              data={},
                              message="未配置基准模板(请双击本步骤框选基准)")

        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
        if len(template.shape) == 3:
            templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            templ_gray = template.copy()

        img_h, img_w = gray_img.shape[:2]
        th, tw = templ_gray.shape[:2]

        # 模板过大时按比例缩小(参考位置同比例修正)
        scale_t = 1.0
        if max(th, tw) > MAX_TEMPLATE_LONG_SIDE or th > img_h or tw > img_w:
            scale_t = min(MAX_TEMPLATE_LONG_SIDE / max(th, tw),
                          min(img_h / th, img_w / tw), 1.0)
            if scale_t <= 0:
                return ToolResult(success=False, passed=False,
                                  processed_image=img.copy(), data={},
                                  message="模板尺寸异常")
            new_w = max(1, int(round(tw * scale_t)))
            new_h = max(1, int(round(th * scale_t)))
            templ_gray = cv2.resize(templ_gray, (new_w, new_h),
                                    interpolation=cv2.INTER_AREA)
            th, tw = templ_gray.shape[:2]

        threshold = float(self.params.get("threshold", 0.7))
        best_score = -2.0
        best_angle = 0.0
        best_loc = (0, 0)
        best_rot = None

        try:
            for angle in self._candidate_angles():
                rot_t = _rotate_gray(templ_gray, angle)
                if angle in (0.0, 180.0) or abs(angle % 180.0) < 1e-6:
                    result = cv2.matchTemplate(gray_img, rot_t,
                                               cv2.TM_CCOEFF_NORMED)
                else:
                    # 小角度旋转的模板含黑边,用掩膜忽略黑边
                    mask = (rot_t > 0).astype(np.uint8) * 255
                    result = cv2.matchTemplate(gray_img, rot_t,
                                               cv2.TM_CCOEFF_NORMED,
                                               mask=mask)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = float(max_val)
                    best_angle = float(angle)
                    best_loc = max_loc
        except cv2.error as e:  # noqa: BLE001
            log_error(f"位置修正匹配失败: {e}")
            return ToolResult(success=False, passed=False,
                              processed_image=img.copy(), data={},
                              message=f"匹配失败: {e}")

        # 特征在当前图中的中心(校正到未缩放模板坐标系)
        cur_cx = best_loc[0] + tw / 2.0
        cur_cy = best_loc[1] + th / 2.0
        if scale_t != 1.0:
            cur_cx /= scale_t
            cur_cy /= scale_t

        # 参考中心
        ref_cx = self.params.get("ref_x", 0) + self.params.get("ref_w", tw) / 2.0
        ref_cy = self.params.get("ref_y", 0) + self.params.get("ref_h", th) / 2.0

        dx = cur_cx - ref_cx
        dy = cur_cy - ref_cy
        angle_deg = best_angle

        if best_score < threshold:
            log_warning(f"位置修正未找到基准: score={best_score:.3f} < "
                        f"{threshold:.2f} (dx={dx:.1f}, dy={dy:.1f})")
            return ToolResult(
                success=False, passed=False,
                processed_image=img.copy(),
                data={"matched": False, "score": float(best_score),
                      "angle_deg": angle_deg, "dx": float(dx), "dy": float(dy)},
                message=f"未找到定位基准 (score={best_score:.2f})"
            )

        # 构造校正变换: 绕图像中心旋转 -angle,再平移使特征中心回到参考中心
        center = ((img_w - 1) / 2.0, (img_h - 1) / 2.0)
        # 匹配中心按模板中心(带缩放)校正
        cur_cx_s = best_loc[0] + tw / 2.0
        cur_cy_s = best_loc[1] + th / 2.0

        M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        # 旋转后原 cur 中心的新位置
        rx = M[0, 0] * cur_cx_s + M[0, 1] * cur_cy_s + M[0, 2]
        ry = M[1, 0] * cur_cx_s + M[1, 1] * cur_cy_s + M[1, 2]
        # 平移量使旋转后的中心落到参考中心
        M[0, 2] += ref_cx - rx
        M[1, 2] += ref_cy - ry

        corrected = cv2.warpAffine(
            img, M, (img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0) if len(img.shape) == 3 else 0)

        # 归一化角度显示(-180,180]
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360

        log_info(f"位置修正: score={best_score:.3f} 角度={angle_deg:.1f}° "
                 f"dx={dx:.1f} dy={dy:.1f} (已校正)")
        return ToolResult(
            success=True,
            passed=True,
            processed_image=corrected,
            data={
                "matched": True,
                "score": float(best_score),
                "angle_deg": float(angle_deg),
                "dx": float(dx),
                "dy": float(dy),
            },
            message=f"位置修正: score={best_score:.2f} "
                    f"偏移=({dx:.0f},{dy:.0f})px 角度={angle_deg:.0f}°"
        )

    # ── UI 说明(具体框选界面见 position_correct_dialog.py)──

    def get_param_widgets(self, parent):
        return []
