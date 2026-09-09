# -*- coding: utf-8 -*-
"""
几何/坐标工具(位置修正与 ROI 随动共用)
========================================
定义"参考姿态 → 当前图"的变换表达,以及把任意角度(旋转)矩形区域
从图像上裁剪并摆正(转水平)的算法。

约定:
    - 图像坐标: (x, y, w, h),原点左上角;
    - 角度单位: 度,逆时针为正(与 OpenCV warpAffine 一致);
    - location: 定位结果 dict,由 PositionCorrect 写入 context:
        {
          "matched": bool,
          "angle_deg": float,   # 产品相对参考姿态旋转的角度(逆时针+)
          "dx": float,          # 产品相对参考姿态平移量
          "dy": float,
          # 旋转/平移基于的参考锚点(参考图上的特征中心)
          "ref_anchor_x": float, "ref_anchor_y": float,
          # 锚点(特征)当前在图像上的中心
          "cur_anchor_x": float, "cur_anchor_y": float,
          "score": float,
        }
    - 坐标变换: 参考图上的点 P_ref -> 当前图像点 P_cur:
        先绕 ref_anchor 旋转 angle_deg, 再平移到 cur_anchor 对齐。
        即 P_cur = R(P_ref - A_ref) + A_cur
      (旋转中心取特征锚点本身:产品以特征为不动点旋转+平移,
       对小角度/180°翻转的近似足够,这也是模板匹配的常规做法)
"""

from typing import Dict, Tuple, Optional, Any

import numpy as np
import cv2

# 角度容差:低于该值视为"无旋转",可走快速轴对齐路径
ANGLE_EPS = 1e-3


# ============================================================================
# 旋转裁剪(把任意角度的矩形区域转水平后裁出)
# ============================================================================

def crop_rotated_rect(image: np.ndarray,
                      cx: float, cy: float,
                      w: float, h: float,
                      angle_deg: float,
                      pad: int = 4) -> np.ndarray:
    """从图像中按旋转矩形(中心+宽高+角度)裁剪并摆正为 w×h 水平图。

    用于"ROI 随动":位置修正后 ROI 在图像中是旋转矩形,但下游算子
    (颜色/条码等)需要水平矩形输入,因此把该区域局部旋转回水平再裁。

    Args:
        image: 原图(BGR/灰度)
        cx, cy: 旋转矩形中心(图像坐标)
        w, h: 旋转矩形宽高(未旋转时的宽高)
        angle_deg: 矩形相对水平的旋转角(度)
        pad: 四周额外留白像素,避免旋转插值丢边

    Returns:
        w×h 水平摆正图(若区域出界,则返回可用部分,黑边填充)
    """
    img_h, img_w = image.shape[:2]

    # 规范化角度到 (-180, 180]
    a = float(angle_deg)
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360

    # 无旋转:直接裁
    if abs(a) < ANGLE_EPS:
        x0 = max(0, int(round(cx - w / 2)))
        y0 = max(0, int(round(cy - h / 2)))
        x1 = min(img_w, int(round(cx + w / 2)))
        y1 = min(img_h, int(round(cy + h / 2)))
        if x1 <= x0 or y1 <= y0:
            return _blank_like(image, w, h)
        return image[y0:y1, x0:x1].copy()

    # 旋转矩形外接正矩形(含 pad),从原图裁出——注意:外接块尺寸不一定能
    # 容纳旋转-摆正后的 w×h 内容(如 90° 时外接 40×200 而目标是 200×40),
    # 因此改用"对角半径正方形"作为旋转载体,保证任意角度摆正不截断。
    rad = np.deg2rad(a)
    half_diag = np.hypot(w, h) / 2.0 + pad
    side = int(np.ceil(half_diag * 2.0))   # 正方形边长(含余量)

    x0 = int(max(0, round(cx - side / 2)))
    y0 = int(max(0, round(cy - side / 2)))
    x1 = int(min(img_w, round(cx + side / 2)))
    y1 = int(min(img_h, round(cy + side / 2)))
    if x1 <= x0 or y1 <= y0:
        return _blank_like(image, w, h)
    crop = image[y0:y1, x0:x1].copy()

    # 以裁出块中心为旋转中心,把块整体反向旋转 -a,使矩形转水平
    ch, cw = crop.shape[:2]
    rot = cv2.getRotationMatrix2D(((cw - 1) / 2.0, (ch - 1) / 2.0), -a, 1.0)
    rotated = cv2.warpAffine(
        crop, rot, (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0) if len(image.shape) == 3 else 0)

    # 摆正后目标 w×h 仍位于块中心(旋转围绕块中心,中心不动)
    tx0 = int(round(cw / 2.0 - w / 2))
    ty0 = int(round(ch / 2.0 - h / 2))
    tx1 = int(round(cw / 2.0 + w / 2))
    ty1 = int(round(ch / 2.0 + h / 2))
    out = _blank_like(image, w, h)
    sx0 = max(0, tx0)
    sy0 = max(0, ty0)
    sx1 = min(cw, tx1)
    sy1 = min(ch, ty1)
    ox0 = sx0 - tx0
    oy0 = sy0 - ty0
    if sx1 > sx0 and sy1 > sy0:
        out[oy0:oy0 + (sy1 - sy0), ox0:ox0 + (sx1 - sx0)] = \
            rotated[sy0:sy1, sx0:sx1]
    return out


def _blank_like(image: np.ndarray, w: float, h: float) -> np.ndarray:
    """生成与 image 同通道数的空白图。"""
    iw = max(1, int(round(w)))
    ih = max(1, int(round(h)))
    if len(image.shape) == 3:
        return np.zeros((ih, iw, image.shape[2]), dtype=image.dtype)
    return np.zeros((ih, iw), dtype=image.dtype)


# ============================================================================
# 坐标变换(参考姿态 → 当前图)
# ============================================================================

def apply_location(point_x: float, point_y: float,
                  loc: Dict[str, Any]) -> Tuple[float, float]:
    """把参考姿态上的点 (x,y) 按 location 变换到当前图像坐标。

    变换模型: P_cur = R(angle)·(P_ref - A_ref) + A_cur
        即先绕参考锚点 A_ref 旋转 angle,再平移到当前锚点 A_cur。
    """
    ax = float(loc.get("ref_anchor_x", 0.0))
    ay = float(loc.get("ref_anchor_y", 0.0))
    bx = float(loc.get("cur_anchor_x", ax))
    by = float(loc.get("cur_anchor_y", ay))
    a = np.deg2rad(float(loc.get("angle_deg", 0.0)))
    cos_a, sin_a = np.cos(a), np.sin(a)
    dx = point_x - ax
    dy = point_y - ay
    rx = bx + dx * cos_a - dy * sin_a
    ry = by + dx * sin_a + dy * cos_a
    return float(rx), float(ry)


def region_center(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    return (x + w / 2.0, y + h / 2.0)


def make_region_from_center(cx: float, cy: float,
                            w: float, h: float) -> Tuple[int, int, int, int]:
    """由中心+宽高生成轴对齐 (x,y,w,h)(供无旋转快速路径)。"""
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return x, y, int(round(w)), int(round(h))


def is_axis_aligned(angle_deg: float, eps: float = 2.0) -> bool:
    """角度是否近似轴对齐(0/90/180/270)——可避免旋转裁剪。"""
    a = float(angle_deg) % 180.0
    if a > 90:
        a = 180.0 - a
    return a < eps or abs(a - 90.0) < eps


def draw_rotated_rect(image: np.ndarray,
                      cx: float, cy: float,
                      w: float, h: float,
                      angle_deg: float,
                      color=(0, 255, 255),
                      thickness: int = 2) -> None:
    """在图上画旋转矩形(用于标注随动 ROI 与匹配结果)。"""
    a = np.deg2rad(float(angle_deg))
    cos_a, sin_a = np.cos(a), np.sin(a)
    hw, hh = w / 2.0, h / 2.0
    corners = []
    for dxs, dys in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        px = cx + dxs * cos_a - dys * sin_a
        py = cy + dxs * sin_a + dys * cos_a
        corners.append((int(round(px)), int(round(py))))
    pts = np.array(corners, np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True,
                  color=color, thickness=thickness, lineType=cv2.LINE_AA)


def transform_point(matrix2x3, x: float, y: float) -> Tuple[float, float]:
    """把点 (x,y) 用 2x3 仿射矩阵变换(齐次坐标)。"""
    m = np.asarray(matrix2x3, dtype=np.float64)
    px = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    py = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return float(px), float(py)


def transform_region(matrix2x3, x: float, y: float,
                     w: float, h: float) -> Dict[str, Any]:
    """把参考姿态上的轴对齐矩形经仿射矩阵映射为当前图的旋转矩形。

    参考矩形四角 (x,y)-(x+w,y+h) 各自经矩阵映射后得到一平行四边形;
    由于映射为刚体(旋转+平移,无剪切),用中心 + 角度表达最稳。

    Returns:
        {
          "cx": float, "cy": float,          # 旋转矩形中心
          "w": float, "h": float,            # 参考宽高(未旋转)
          "angle_deg": float,                # 相对参考的旋转角(度)
          "bbox": (bx, by, bw, bh),          # 轴对齐外接框(显示/快速裁剪用)
        }
    """
    pts = []
    for corner_x, corner_y in [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]:
        pts.append(transform_point(matrix2x3, corner_x, corner_y))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # 中心 = 对角线中点(刚体变换后仍正确)
    cx = (pts[0][0] + pts[2][0]) / 2.0
    cy = (pts[0][1] + pts[2][1]) / 2.0
    # 旋转角 = 矩阵旋转分量。注意 OpenCV 旋转矩阵第二行为 [-sin, cos],
    # 因此 m[1,0] = -sin(a), 用 atan2(-m[1,0], m[0,0]) 才是内容实际旋转角。
    m = np.asarray(matrix2x3, dtype=np.float64)
    angle_deg = float(np.degrees(np.arctan2(-m[1, 0], m[0, 0])))
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg <= -180:
        angle_deg += 360
    bbox = (int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs) - min(xs))), int(round(max(ys) - min(ys))))
    return {
        "cx": float(cx), "cy": float(cy),
        "w": float(w), "h": float(h),
        "angle_deg": angle_deg,
        "bbox": bbox,
    }
