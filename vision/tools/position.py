# -*- coding: utf-8 -*-
"""
位置修正算子 (PositionCorrect)
================================
解决"产品摆放偏移/翻转导致固定 ROI 误判"的问题。

原理(ROI 随动,不旋转整图):
    在参考图(产品标准摆放)上框选一个稳定特征作为"基准模板"(如板角、
    丝印、mark),记录其参考位置与角度(=0°)。运行时对当前图像做一次
    轻量模板匹配(支持 0°/180° / 小角度 / 全角度搜索),求出产品相对
    参考姿态的旋转角度与平移,并把"参考坐标 → 当前图坐标"的仿射矩阵
    (M_ref2cur)写入 PipelineContext.location。

    后续 MultiROI 依据该矩阵把各参考 ROI 变换为当前图像上的实际区域
    (旋转 + 平移随动);裁剪时把旋转区域局部摆正,保证颜色/条码算子
    看到的是水平内容。整幅图像不做旋转(无黑边、无性能损失)。

参数(params):
    template_b64 : 基准模板图像(PNG, base64),由配置界面框选生成
    ref_x/ref_y  : 特征参考位置(模板在参考图中的左上角)
    ref_w/ref_h  : 模板尺寸
    rot_mode     : "0" / "0and180" / "range"(小角度) / "any"(全角度两级搜索)
    angle_min/max/step : range 模式角度搜索范围(度)
    threshold    : 匹配分数阈值(0~1),低于阈值判 NG

输出:
    ToolResult.passed      : 是否成功定位
    context.location       : 定位结果(矩阵/角度/锚点),供 MultiROI 随动
    data.dx/dy             : 特征中心相对参考中心的像素偏移
    data.angle_deg         : 产品相对参考的旋转角度
    data.score             : 匹配分数
    overlay_image          : 基准参考框(绿) + 当前匹配框(黄) + 偏移连线
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

    def _candidate_angles(self, fine: bool = False) -> list:
        """生成角度搜索候选。

        rot_mode:
            "0"        — 仅 0°
            "0and180"  — 0° 与 180°(最常见:正放/反放)
            "range"    — 小角度范围(angle_min..angle_max,默认 ±10°)
            "any"      — 全角度。两级搜索:
                          第一级 coarse(30° 步进覆盖全周)由外部调用 fine=False,
                          第二级在最优角附近 ±20° 内 2° 步进由外部调用 fine=True。
        """
        mode = self.params.get("rot_mode", "0and180")
        if mode == "0":
            return [0.0]
        if mode == "0and180":
            return [0.0, 180.0]
        if mode == "range":
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
        if mode == "any":
            # 全角度两级
            if fine:
                # 由调用方给出中心角,这里仅占位;实际细扫在 _match_any_angle 内
                return [0.0]
            # 粗扫:0..330 step 30(避免与 0° 重复)
            vals = [0.0]
            a = 30.0
            while a <= 360 - 1e-6:
                vals.append(a)
                a += 30.0
            return vals
        # 默认
        return [0.0]

    def _match_any_angle(self, gray_img: np.ndarray, templ_gray: np.ndarray,
                         threshold: float):
        """全角度两级搜索:先粗扫 30°,再在最优角附近 ±20° 细扫。

        Returns:
            (best_score, best_angle, best_loc)
        """
        best_score, best_angle, best_loc = -2.0, 0.0, (0, 0)
        # ── 第一级:粗扫 ──
        for angle in self._candidate_angles(fine=False):
            score, loc = self._match_at_angle(gray_img, templ_gray, angle)
            if score > best_score:
                best_score, best_angle, best_loc = score, angle, loc

        # ── 第二级:在粗扫最优角附近细扫 ──
        fine_angles = []
        center_angle = float(best_angle)
        a = center_angle - 20.0
        while a <= center_angle + 20.0 + 1e-6:
            fa = round(float(a) % 360.0, 3)
            if fa > 180:
                fa -= 360.0
            fine_angles.append(fa)
            a += 2.0
        for angle in sorted(set(fine_angles)):
            score, loc = self._match_at_angle(gray_img, templ_gray, angle)
            if score > best_score:
                best_score, best_angle, best_loc = score, angle, loc

        return best_score, best_angle, best_loc

    def _match_at_angle(self, gray_img: np.ndarray, templ_gray: np.ndarray,
                        angle: float):
        """在给定角度下旋转图像匹配 0° 模板,返回 (score, loc)。"""
        if abs(angle % 360.0) < 1e-6:
            search_img = gray_img
        elif abs((angle - 180.0) % 360.0) < 1e-6:
            search_img = cv2.rotate(gray_img, cv2.ROTATE_180)
        else:
            search_img = _rotate_gray(gray_img, angle)
        result = cv2.matchTemplate(search_img, templ_gray,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return float(max_val), max_loc

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
            if self.params.get("rot_mode", "0and180") == "any":
                best_score, best_angle, best_loc = self._match_any_angle(
                    gray_img, templ_gray, threshold)
            else:
                for angle in self._candidate_angles():
                    best_score_a, best_loc_a = self._match_at_angle(
                        gray_img, templ_gray, angle)
                    if best_score_a > best_score:
                        best_score = best_score_a
                        best_angle = float(angle)
                        best_loc = best_loc_a
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
            context.location = None
            return ToolResult(
                success=False, passed=False,
                processed_image=img.copy(),
                data={"matched": False, "score": float(best_score),
                      "angle_deg": angle_deg, "dx": float(dx), "dy": float(dy)},
                message=f"未找到定位基准 (score={best_score:.2f})"
            )

        # ── 输出"定位结果"(不再旋转整图,由下游按变换随动)──
        # 构造 参考→当前 的仿射矩阵 M_ref2cur:
        #   校正矩阵 M_corr 满足  M_corr(P_cur) ≈ P_ref(把当前图校正回参考姿态)
        #   其逆即 参考→当前: P_cur = M_ref2cur(P_ref)
        center = ((img_w - 1) / 2.0, (img_h - 1) / 2.0)
        M_corr = cv2.getRotationMatrix2D(center, float(best_angle), 1.0)
        M_corr[0, 2] += tx
        M_corr[1, 2] += ty
        # 求逆(2x3 -> 3x3 -> 逆 -> 2x3)
        M3 = np.vstack([M_corr, [0.0, 0.0, 1.0]])
        try:
            M3_inv = np.linalg.inv(M3)
        except np.linalg.LinAlgError:
            context.location = None
            return ToolResult(success=False, passed=False,
                              processed_image=img.copy(),
                              data={"matched": False, "score": float(best_score),
                                    "angle_deg": angle_deg, "dx": float(dx),
                                    "dy": float(dy)},
                              message="定位矩阵不可逆")
        M_ref2cur = M3_inv[:2, :]

        # 定位结果:供 MultiROI 把参考 ROI 坐标变换为当前图实际区域
        # 当前锚点 = M_ref2cur 作用于参考锚点(特征在当前图中的实际中心)
        ref_pt = np.array([ref_cx, ref_cy, 1.0])
        cur_pt = M_ref2cur @ ref_pt
        context.location = {
            "matched": True,
            "score": float(best_score),
            "angle_deg": float(angle_deg),      # 产品相对参考的旋转(逆时针+)
            "dx": float(dx),
            "dy": float(dy),
            "matrix": M_ref2cur.tolist(),        # 2x3 仿射: 参考坐标 -> 当前图坐标
            "ref_anchor_x": float(ref_cx),
            "ref_anchor_y": float(ref_cy),
            "cur_anchor_x": float(cur_pt[0]),
            "cur_anchor_y": float(cur_pt[1]),
        }

        # ── overlay:标注匹配结果 ──
        overlay = np.zeros_like(img)
        # 绿框:参考基准位置(产品无偏移时特征应在的位置,角度 0)
        self._draw_box(overlay, float(ref_cx), float(ref_cy),
                       ref_w + 6, ref_h + 6, 0.0,
                       (0, 255, 0), 2, label="基准")
        # 黄框:实际匹配位置(特征在当前图中的位置,带产品角度)
        self._draw_box(overlay, float(cur_pt[0]), float(cur_pt[1]),
                       ref_w, ref_h, float(angle_deg),
                       (0, 255, 255), 2, label="匹配")
        # 偏移连线(洋红)
        cv2.line(overlay,
                 (int(round(float(cur_pt[0]))), int(round(float(cur_pt[1])))),
                 (int(round(float(ref_cx))), int(round(float(ref_cy)))),
                 (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (int(round(float(cur_pt[0]))),
                             int(round(float(cur_pt[1])))), 4, (255, 0, 255), -1)

        log_info(f"位置修正: score={best_score:.3f} 角度={angle_deg:.1f}° "
                 f"dx={dx:.1f} dy={dy:.1f} (ROI 随动)")
        return ToolResult(
            success=True,
            passed=True,
            # 不再输出旋转后的整图——交给 MultiROI 做 ROI 随动
            processed_image=img.copy(),
            overlay_image=overlay,
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

    @staticmethod
    def _draw_box(img: np.ndarray, cx: float, cy: float,
                  w: float, h: float, angle_deg: float,
                  color, thickness: int, label: str = ""):
        """画(可能旋转的)矩形框与角标。"""
        from vision.geometry_util import draw_rotated_rect
        try:
            draw_rotated_rect(img, cx, cy, w, h, angle_deg,
                              color=color, thickness=thickness)
            if label:
                cv2.putText(img, label, (int(round(cx - w / 2)), int(round(cy - h / 2 - 6))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        except Exception as e:  # noqa: BLE001
            log_warning(f"绘制定位标注失败: {e}")

    # ── UI 说明(具体框选界面见 position_correct_dialog.py)──

    def get_param_widgets(self, parent):
        return []
