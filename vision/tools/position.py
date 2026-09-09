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

# any(全角度)模式防伪峰参数:
#   粗扫 30° 步进对"大面积均匀板面"可能出现伪峰:整图旋转后,板边/边角在
#   旋转图上形成的"伪 L"轮廓(粗形状相似但对比度/细节不同)分数可高达
#   0.93,超过真实但倾斜 7° 的粗扫分数 0.86——若只相信粗扫 argmax 会在
#   错误盆地细扫,漏掉真实角度。
#   对策:按粗扫分数从高到低探索前 ANY_BASIN_LIMIT 个盆地细扫,并对每个
#   细扫候选用"原图内容还原 + 亮度归一化逐像素距离"校验真伪:
#     ANY_BASIN_LIMIT   — 最多细扫的粗扫盆地数(控制最坏耗时)
#     ANY_VERIFY_ACCEPT_D — 内容距离 ≤ 该值 ≈ 像素级一致,直接接受
#     ANY_VERIFY_REL    — 相对判据:最小距离 ≤ 次小 × 该比例 → 显著最真实
#     ANY_VERIFY_ABS_MAX — 相对判据成立时最小距离的绝对上限
#   说明:TM_CCOEFF 只对"形状相关"敏感,伪 L 也能给 ~0.93 高分;逐像素
#   差异对细节错位更敏感,伪 L 还原区与模板距离明显更大(合成板实测 ~5 vs
#   真实 ~2),能可靠区分。校验仅为"粗筛",决策还会综合多盆地相对距离,
#   全部无法确认时退回细扫最高分(与原行为一致)。
ANY_BASIN_LIMIT = 3
ANY_VERIFY_ACCEPT_D = 4.5
ANY_VERIFY_REL = 0.6
ANY_VERIFY_ABS_MAX = 7.0


def _encode_png_bgr(img_bgr: np.ndarray) -> str:
    """把 BGR 图编码为 PNG base64(用于存入方案 JSON)。"""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("模板图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_jpg_bgr(img_bgr: np.ndarray, quality: int = 88) -> str:
    """把 BGR 参考图编码为 JPEG base64(体积更小,用于还原配置底图)。

    参考图仅用于配置时还原"框选底图",运行时不需要,因此用有损 JPEG
    即可(不影响模板本身,模板仍用无损 PNG 存储)。
    """
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("参考图编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_jpg_bgr(b64: str) -> Optional[np.ndarray]:
    """从 JPEG base64 解码为 BGR 图。"""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log_error(f"解码参考图失败: {e}")
        return None


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
        # 参考图(JPEG b64):仅配置期用作"框选底图"还原,运行时不用
        self.params.setdefault("ref_image_b64", "")
        self._template_cache: Optional[np.ndarray] = None

    # ── 参考图(配置底图)──

    def set_reference_image(self, image_bgr: np.ndarray):
        """保存参考图(标准摆放照片),供配置界面再次打开时还原框选底图。

        参考图仅用于配置:工程师在它上面框选基准;方案保存时随模板一起
        写入 JSON(JPEG 压缩,体积较小),运行时不需要它。
        """
        self.params["ref_image_b64"] = _encode_jpg_bgr(image_bgr)

    def get_reference_image(self) -> Optional[np.ndarray]:
        """取回参考图(若已保存)。"""
        return _decode_jpg_bgr(self.params.get("ref_image_b64", ""))

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
                         threshold: float,
                         verify_templ_gray: Optional[np.ndarray] = None):
        """全角度两级搜索:粗扫 30° 后细扫,并用原图内容校验防伪峰。

        Args:
            gray_img: 输入灰度图(全分辨率)
            templ_gray: 匹配用模板(可能已被缩放)
            threshold: 匹配分数阈值(仅与调用方一致,本方法内不使用)
            verify_templ_gray: 未缩放模板(尺寸 == 参考 ref_w×ref_h),
                用于候选真实性校验;为 None 时不校验(退化为纯两级搜索)。

        Returns:
            (best_score, best_angle, best_loc)
        """
        # ── 第一级:粗扫 ──
        coarse = []
        for angle in self._candidate_angles(fine=False):
            score, loc = self._match_at_angle(gray_img, templ_gray, angle)
            coarse.append((float(score), float(angle), loc))
        coarse.sort(key=lambda c: c[0], reverse=True)

        best_score, best_angle, best_loc = coarse[0]
        # ── 第二级:按粗扫分数从高到低探索盆地,细扫 + 原图内容校验 ──
        dists = []  # (内容距离, 候选):距离越小越像模板真实内容
        for _, angle, _ in coarse[:ANY_BASIN_LIMIT]:
            cand = self._fine_around(gray_img, templ_gray, angle)
            if cand[0] > best_score:
                best_score, best_angle, best_loc = cand
            if verify_templ_gray is not None:
                geom = self._geometry(gray_img.shape[1], gray_img.shape[0],
                                      cand[1], cand[2])
                d = self._content_distance(gray_img, verify_templ_gray,
                                           cand[1], cand[2], geom)
                if d is not None:
                    if d <= ANY_VERIFY_ACCEPT_D:
                        # 还原区与模板几乎像素级一致 → 真实候选,直接接受
                        return cand
                    dists.append((d, cand))
        # 无"像素级一致"候选:若某候选的内容距离显著小于其它候选,取它;
        # 否则退回细扫最高分(保持原两级搜索行为)
        if len(dists) >= 2:
            dists.sort(key=lambda x: x[0])
            d0, cand0 = dists[0]
            d1 = dists[1][0]
            if d0 <= ANY_VERIFY_ABS_MAX and d0 <= d1 * ANY_VERIFY_REL:
                return cand0
        return best_score, best_angle, best_loc

    def _fine_around(self, gray_img: np.ndarray, templ_gray: np.ndarray,
                     center_angle: float):
        """在 center_angle ±20° 内以 2° 步进细扫,返回 (score, angle, loc)。"""
        best = (-2.0, float(center_angle), (0, 0))
        seen = set()
        a = float(center_angle) - 20.0
        while a <= float(center_angle) + 20.0 + 1e-6:
            fa = round(float(a) % 360.0, 3)
            if fa > 180:
                fa -= 360.0
            if fa not in seen:
                seen.add(fa)
                score, loc = self._match_at_angle(gray_img, templ_gray, fa)
                if score > best[0]:
                    best = (score, fa, loc)
            a += 2.0
        return best

    def _geometry(self, img_w: int, img_h: int, best_angle: float,
                  best_loc) -> dict:
        """把匹配结果换算为报告量(dx/dy/角度)与中间几何量(纯计算,无副作用)。

        语义:best_angle 是"把当前图旋转该角度后模板即水平"的校正量;
        best_loc 是该旋转后图上模板左上角。参考模板中心 (ref_cx, ref_cy)
        在校正后图上应回到参考位置,因此平移量 tx/ty = 参考中心 - 匹配中心。
        """
        ref_w = int(self.params.get("ref_w", 0))
        ref_h = int(self.params.get("ref_h", 0))
        ref_cx = float(self.params.get("ref_x", 0)) + ref_w / 2.0
        ref_cy = float(self.params.get("ref_y", 0)) + ref_h / 2.0
        loc_cx = float(best_loc[0]) + ref_w / 2.0
        loc_cy = float(best_loc[1]) + ref_h / 2.0
        tx = ref_cx - loc_cx
        ty = ref_cy - loc_cy
        # 报告语义:产品相对参考的偏移(与校正量符号相反)
        dx, dy = -tx, -ty
        angle_deg = -float(best_angle)
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
        return {"ref_cx": ref_cx, "ref_cy": ref_cy,
                "loc_cx": loc_cx, "loc_cy": loc_cy,
                "tx": tx, "ty": ty, "dx": dx, "dy": dy,
                "angle_deg": float(angle_deg)}

    def _build_location_matrix(self, img_w: int, img_h: int,
                               best_angle: float, geom: dict):
        """构造 参考→当前 的仿射矩阵 M_ref2cur 并算出当前锚点。

        Returns:
            (M_ref2cur, cur_x, cur_y) 或 (None, None, None)(矩阵不可逆)
        """
        center = ((img_w - 1) / 2.0, (img_h - 1) / 2.0)
        M_corr = cv2.getRotationMatrix2D(center, float(best_angle), 1.0)
        M_corr[0, 2] += geom["tx"]
        M_corr[1, 2] += geom["ty"]
        M3 = np.vstack([M_corr, [0.0, 0.0, 1.0]])
        try:
            M3_inv = np.linalg.inv(M3)
        except np.linalg.LinAlgError:
            return None, None, None
        M_ref2cur = M3_inv[:2, :]
        cur_pt = M_ref2cur @ np.array([geom["ref_cx"], geom["ref_cy"], 1.0])
        return M_ref2cur, float(cur_pt[0]), float(cur_pt[1])

    def _content_distance(self, gray_img: np.ndarray,
                          templ_gray: np.ndarray,
                          best_angle: float, best_loc, geom: dict):
        """候选的"内容距离":候选几何在原图上还原的区域与模板的差异(越小越真实)。

        按与 process() 完全相同的几何换算得到模板区域在当前图上的位置/角度,
        裁剪摆正后做亮度归一化(把裁剪区均值/方差对齐到模板)再逐像素比较。
        真实匹配的还原区就是模板本身 → 距离很小(实测 ~2);
        旋转边框/边角形成的"伪 L"还原区细节错位 → 距离明显更大(实测 ~5)。

        Returns:
            内容距离(平均绝对灰度差,0~255),越小越像;
            无法校验(模板被缩放/区域出界/还原区近乎纯色)时返回 None。
        """
        try:
            from vision.geometry_util import crop_rotated_rect
            ref_w = int(self.params.get("ref_w", 0))
            ref_h = int(self.params.get("ref_h", 0))
            if ref_w <= 0 or ref_h <= 0:
                return None
            # 模板被缩放时尺寸 != 参考尺寸,无法按原尺寸还原比对 → 跳过
            if templ_gray.shape[1] != ref_w or templ_gray.shape[0] != ref_h:
                return None
            img_h, img_w = gray_img.shape[:2]
            M, cx, cy = self._build_location_matrix(img_w, img_h,
                                                    best_angle, geom)
            if M is None:
                return None
            crop = crop_rotated_rect(gray_img, cx, cy, ref_w, ref_h,
                                     geom["angle_deg"])
            if crop is None or crop.size == 0:
                return None
            if crop.shape[0] != ref_h or crop.shape[1] != ref_w:
                crop = cv2.resize(crop, (ref_w, ref_h),
                                  interpolation=cv2.INTER_LINEAR)
            crop_f = crop.reshape(-1).astype(np.float64)
            templ_f = templ_gray.reshape(-1).astype(np.float64)
            if crop.std() < 3.0:
                # 还原区近乎纯色(伪峰常见),无法做有意义的逐像素比较
                return None
            # 亮度归一化:均值/方差对齐到模板,抵消照明差异后再比细节
            crop_n = (crop_f - crop_f.mean()) * \
                (templ_f.std() / crop_f.std()) + templ_f.mean()
            return float(np.abs(crop_n - templ_f).mean())
        except Exception:  # noqa: BLE001 校验失败不应影响主流程
            return None

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
        # 未缩放模板(any 模式候选真实性校验用,尺寸 == 参考 ref_w×ref_h)
        templ_gray_full = templ_gray

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
                # 模板被缩放时锚点/校验几何近似,不做原图校验(退化为两级搜索)
                verify_templ = templ_gray_full if scale_t == 1.0 else None
                best_score, best_angle, best_loc = self._match_any_angle(
                    gray_img, templ_gray, threshold, verify_templ)
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

        # 匹配结果 → 报告量(dx/dy/角度)与中间几何量
        geom = self._geometry(img_w, img_h, float(best_angle), best_loc)
        dx, dy = geom["dx"], geom["dy"]
        angle_deg = geom["angle_deg"]

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
        M_ref2cur, cur_x, cur_y = self._build_location_matrix(
            img_w, img_h, float(best_angle), geom)
        if M_ref2cur is None:
            context.location = None
            return ToolResult(success=False, passed=False,
                              processed_image=img.copy(),
                              data={"matched": False, "score": float(best_score),
                                    "angle_deg": angle_deg, "dx": float(dx),
                                    "dy": float(dy)},
                              message="定位矩阵不可逆")

        # 定位结果:供 MultiROI 把参考 ROI 坐标变换为当前图实际区域
        context.location = {
            "matched": True,
            "score": float(best_score),
            "angle_deg": float(angle_deg),      # 产品相对参考的旋转(逆时针+)
            "dx": float(dx),
            "dy": float(dy),
            "matrix": M_ref2cur.tolist(),        # 2x3 仿射: 参考坐标 -> 当前图坐标
            "ref_anchor_x": float(geom["ref_cx"]),
            "ref_anchor_y": float(geom["ref_cy"]),
            "cur_anchor_x": float(cur_x),
            "cur_anchor_y": float(cur_y),
        }

        # ── overlay:标注匹配结果 ──
        overlay = np.zeros_like(img)
        ref_w = int(self.params.get("ref_w", 0))
        ref_h = int(self.params.get("ref_h", 0))
        ref_cx = float(geom["ref_cx"])
        ref_cy = float(geom["ref_cy"])
        # 绿框:参考基准位置(产品无偏移时特征应在的位置,角度 0)
        self._draw_box(overlay, ref_cx, ref_cy,
                       ref_w + 6, ref_h + 6, 0.0,
                       (0, 255, 0), 2, label="基准")
        # 黄框:实际匹配位置(特征在当前图中的位置,带产品角度)
        self._draw_box(overlay, float(cur_x), float(cur_y),
                       ref_w, ref_h, float(angle_deg),
                       (0, 255, 255), 2, label="匹配")
        # 偏移连线(洋红)
        cv2.line(overlay,
                 (int(round(float(cur_x))), int(round(float(cur_y)))),
                 (int(round(ref_cx)), int(round(ref_cy))),
                 (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (int(round(float(cur_x))),
                             int(round(float(cur_y)))), 4, (255, 0, 255), -1)

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
