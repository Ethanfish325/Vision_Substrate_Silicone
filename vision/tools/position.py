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

    # 模板最小纹理要求:标准差低于该值视为纯色/低纹理,拒绝使用。
    # 纯色模板会让 TM_CCOEFF_NORMED 分数虚高、匹配位置无意义(假成功)。
    MIN_TEMPLATE_STD: float = 8.0

    def template_is_valid(self) -> bool:
        """校验当前模板是否具有足够纹理(非纯色)。

        工程现场常见误操作:框选基准时框到无特征的板面/底色,
        导致模板为纯色 → 匹配 score≈1.0 但位置随机,校正无意义。
        这里用灰度标准差判定:太低则拒绝。
        """
        templ = self._get_template()
        if templ is None:
            return False
        gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY) \
            if len(templ.shape) == 3 else templ
        std = float(gray.std())
        return std >= self.MIN_TEMPLATE_STD

    def set_template(self, template_bgr: np.ndarray, ref_x: int, ref_y: int) -> bool:
        """配置界面调用:写入模板图与参考位置。

        返回是否成功;模板为纯色/低纹理时返回 False(不写入),避免后续
        匹配分数虚高、位置随机导致的"假成功"。

        现场提示:基准应框选产品上【稳定且独特】的特征(板角L形、丝印、
        mark、定位孔等),并让框略大于特征、包含一定对比度。
        """
        gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY) \
            if len(template_bgr.shape) == 3 else template_bgr
        if float(gray.std()) < self.MIN_TEMPLATE_STD:
            log_warning(
                f"基准模板纹理不足(std={float(gray.std()):.2f} < "
                f"{self.MIN_TEMPLATE_STD}),拒绝保存——请框选板上的稳定特征"
                f"(板角/丝印/mark),而不是无特征的板面/底色")
            return False
        self._template_cache = template_bgr
        self.params["template_b64"] = _encode_png_bgr(template_bgr)
        self.params["ref_x"] = int(ref_x)
        self.params["ref_y"] = int(ref_y)
        h, w = template_bgr.shape[:2]
        self.params["ref_w"] = int(w)
        self.params["ref_h"] = int(h)
        return True

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
        # 防御:加载的模板可能是早期保存的纯色/低纹理基准 → 拒绝并提示
        if not self.template_is_valid():
            return ToolResult(
                success=False, passed=False,
                processed_image=img.copy(), data={},
                message="基准模板无足够纹理(纯色板面/底色)。请重新双击本步骤,"
                        "框选板上的稳定特征(板角/丝印/mark)作为基准")

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

        # 角度搜索:旋转【整幅输入图】后与 0° 模板匹配(argmax)。
        # 注意:不旋转模板 + 掩膜——旋转模板产生的黑边会让掩膜匹配
        # 产生 NaN 分数,导致小角度搜索失效(曾实测 angle 恒为 0)。
        try:
            for angle in self._candidate_angles():
                if angle == 0.0:
                    search_img = gray_img
                elif abs(angle - 180.0) < 1e-6:
                    search_img = cv2.rotate(gray_img, cv2.ROTATE_180)
                else:
                    search_img = _rotate_gray(gray_img, angle)
                result = cv2.matchTemplate(search_img, templ_gray,
                                           cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if float(max_val) > best_score:
                    best_score = float(max_val)
                    best_angle = float(angle)
                    best_loc = max_loc
        except cv2.error as e:  # noqa: BLE001
            log_error(f"位置修正匹配失败: {e}")
            return ToolResult(success=False, passed=False,
                              processed_image=img.copy(), data={},
                              message=f"匹配失败: {e}")

        # 参考中心(始终用模板未缩放的原尺寸)
        ref_w = int(self.params.get("ref_w", 0))
        ref_h = int(self.params.get("ref_h", 0))
        ref_cx = self.params.get("ref_x", 0) + ref_w / 2.0
        ref_cy = self.params.get("ref_y", 0) + ref_h / 2.0

        # 校正所需的平移:把"旋转后图上模板中心"搬回"参考中心"。
        # 模板缩放仅影响匹配用模板,原图左上角 = best_loc(缩放围绕 0,0),
        # 模板中心按未缩放原尺寸 ref_w/ref_h 计算。
        loc_cx = best_loc[0] + ref_w / 2.0
        loc_cy = best_loc[1] + ref_h / 2.0
        tx = ref_cx - loc_cx
        ty = ref_cy - loc_cy

        # 平移/角度报告(语义:产品相对参考姿态的偏移;符号与校正矩阵相反)
        dx = -tx
        dy = -ty
        # 归一化角度显示(-180,180],取反:best_angle 是"校正旋转量",
        # 报告为"产品相对参考旋转了多少"
        angle_deg = -float(best_angle)
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360

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

        # 构造校正矩阵: 绕图像中心旋转 best_angle(校正量),再平移 (tx, ty)
        center = ((img_w - 1) / 2.0, (img_h - 1) / 2.0)
        M = cv2.getRotationMatrix2D(center, float(best_angle), 1.0)
        M[0, 2] += tx
        M[1, 2] += ty

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
