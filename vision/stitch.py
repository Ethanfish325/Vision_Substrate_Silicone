# -*- coding: utf-8 -*-
"""
图像拼接模块 — 基于轴坐标的刚性拼接（增量累积模式）
====================================================
将多个点位的检测后图像按各点位的 X/Y 轴坐标刚性拼接成一张"托盘总览图"。

特性:
    - 基于轴坐标刚性拼接，自动计算画布尺寸与偏移，无需人工填写拼接参数。
    - 拼接顺序为行优先（1.1 → 1.2 → 1.3 → 2.1 → …），由调用方保证输入顺序。
    - 支持逐点增量拼接：每测完一个点位即可调用 add_board() 拼入并刷新。
    - 支持在每个板卡区域左上角标注 OK/NG 状态。
    - 增量累积模式：每张板卡图像在拼入画布后立即释放，不长期保留，
      内存只占用 1 张拼接画布 + 当前 1 张板卡图，避免多张板卡图累积导致内存溢出。

坐标约定:
    - 轴坐标单位与图像像素的换算比例由 scale 参数控制（默认 1 像素 = 1 轴单位）。
    - 若轴坐标跨度远大于图像尺寸，可设置 scale 使拼接图适配显示。
"""

from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import cv2


class BoardPlacement:
    """单个板卡（点位）的拼接放置信息（仅元数据，不保留图像）。"""

    def __init__(self, name: str, x: int, y: int, w: int, h: int,
                 passed: bool = True, image: Optional[np.ndarray] = None):
        self.name = name          # 点位名称，如 "1.1"
        self.x = x                # 轴坐标 X（原始值，不修改）
        self.y = y                # 轴坐标 Y（原始值，不修改）
        self.w = w                # 板卡图像宽度（像素）
        self.h = h                # 板卡图像高度（像素）
        self.passed = passed      # 该板卡检测是否通过
        self.image = image        # 该板卡的检测后图像（标注图）；增量模式下拼入后置为 None 释放
        self.px = 0               # 在拼接画布中的左上角 x（像素）
        self.py = 0               # 在拼接画布中的左上角 y（像素）


class RigidStitcher:
    """基于轴坐标的刚性拼接器（增量累积模式）。

    用法:
        stitcher = RigidStitcher(scale=1.0)
        stitcher.add_board(name, x_axis, y_axis, image, passed)
        stitched = stitcher.render()          # 生成拼接整图
        stitched = stitcher.render_annotated()  # 生成带 OK/NG 标注的整图

    内存优化:
        add_board() 会将板卡图像 blit 到内部累积画布后立即释放该图像，
        因此不会长期保留所有板卡图像，避免多张高分辨率图累积导致内存溢出。
    """

    def __init__(self, scale: float = 1.0, overlap_mode: str = "overwrite"):
        """
        Args:
            scale: 轴坐标 → 像素的换算比例（像素 = 轴坐标 × scale）。
                   当轴坐标跨度远大于图像尺寸时，可设置 scale < 1 使拼接图适配显示。
            overlap_mode: 重叠区域处理方式：
                          "overwrite" - 后放置的图覆盖先放置的图（默认）
                          "average"   - 重叠区域取平均
        """
        self.scale = float(scale)
        self.overlap_mode = overlap_mode
        self._placements: List[BoardPlacement] = []
        self._canvas: Optional[np.ndarray] = None
        self._canvas_w = 0
        self._canvas_h = 0
        self._canvas_channels = 3

    # ------------------------------------------------------------------
    # 画布管理
    # ------------------------------------------------------------------
    def _ensure_canvas(self, need_w: int, need_h: int, channels: int):
        """确保画布至少为 need_w x need_h，不足时动态扩展（保留已有内容）。"""
        if self._canvas is None:
            self._canvas_w = max(1, need_w)
            self._canvas_h = max(1, need_h)
            self._canvas_channels = channels
            if channels == 3:
                self._canvas = np.zeros((self._canvas_h, self._canvas_w, 3), dtype=np.uint8)
            else:
                self._canvas = np.zeros((self._canvas_h, self._canvas_w), dtype=np.uint8)
            return

        if need_w <= self._canvas_w and need_h <= self._canvas_h:
            return

        new_w = max(self._canvas_w, need_w)
        new_h = max(self._canvas_h, need_h)
        if self._canvas_channels == 3:
            new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        else:
            new_canvas = np.zeros((new_h, new_w), dtype=np.uint8)
        new_canvas[:self._canvas_h, :self._canvas_w] = self._canvas
        self._canvas = new_canvas
        self._canvas_w = new_w
        self._canvas_h = new_h

    # ------------------------------------------------------------------
    # 添加板卡
    # ------------------------------------------------------------------
    def add_board(self, name: str, x_axis: float, y_axis: float,
                  image: np.ndarray, passed: bool = True):
        """添加一个板卡（点位）到拼接器，并立即拼入累积画布。

        Args:
            name: 点位名称（如 "1.1"）
            x_axis: 该点位的 X 轴坐标（scale=1.0 时即像素坐标）
            y_axis: 该点位的 Y 轴坐标（scale=1.0 时即像素坐标）
            image: 该点位的检测后图像（标注图）
            passed: 该板卡检测是否通过
        """
        if image is None:
            return
        h, w = image.shape[:2]
        channels = 3 if (len(image.shape) == 3 and image.shape[2] == 3) else 1

        # 像素位置（scale 换算）
        px = int(round(x_axis * self.scale))
        py = int(round(y_axis * self.scale))

        # 动态扩展画布
        self._ensure_canvas(px + w, py + h, channels)

        # 将图像 blit 到累积画布
        self._blit_at(self._canvas, image, px, py)

        # 绘制 OK/NG 标注（增量绘制到画布）
        color = (0, 255, 0) if passed else (0, 0, 255)
        label = "OK" if passed else "NG"
        cv2.rectangle(self._canvas, (px, py), (px + w, py + h), color, 2)
        cv2.putText(self._canvas, label, (px + 4, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(self._canvas, name, (px + 4, py + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 记录元数据（不保留图像，拼入后立即释放）
        self._placements.append(BoardPlacement(
            name=name, x=int(x_axis), y=int(y_axis), w=w, h=h,
            passed=passed, image=None
        ))

    def _blit_at(self, canvas: np.ndarray, img: np.ndarray, x0: int, y0: int):
        """将图像 img 绘制到画布 canvas 的 (x0, y0) 位置。"""
        h, w = img.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]

        # 计算有效区域（裁剪到画布内）
        x1 = min(x0 + w, canvas_w)
        y1 = min(y0 + h, canvas_h)
        if x0 >= x1 or y0 >= y1:
            return

        # 源图像对应区域
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = sx0 + (x1 - max(0, x0))
        sy1 = sy0 + (y1 - max(0, y0))

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)

        src_roi = img[sy0:sy1, sx0:sx1]
        dst_roi = canvas[dst_y0:dst_y0 + src_roi.shape[0],
                         dst_x0:dst_x0 + src_roi.shape[1]]

        if self.overlap_mode == "average":
            # 重叠区域取平均（需浮点运算）
            canvas_f = canvas.astype(np.float32)
            dst_f = canvas_f[dst_y0:dst_y0 + src_roi.shape[0],
                             dst_x0:dst_x0 + src_roi.shape[1]]
            mask = (src_roi > 0).astype(np.float32)
            dst_f = dst_f * (1 - mask) + src_roi.astype(np.float32) * mask
            canvas[dst_y0:dst_y0 + src_roi.shape[0],
                   dst_x0:dst_x0 + src_roi.shape[1]] = dst_f.astype(np.uint8)
        else:
            # 默认：后图覆盖前图
            canvas[dst_y0:dst_y0 + src_roi.shape[0],
                   dst_x0:dst_x0 + src_roi.shape[1]] = src_roi

    def reset(self):
        """清空所有板卡与累积画布。"""
        self._placements.clear()
        self._canvas = None
        self._canvas_w = 0
        self._canvas_h = 0

    @property
    def placements(self) -> List[BoardPlacement]:
        return self._placements

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return self._canvas_w, self._canvas_h

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        """返回当前累积拼接整图（含已增量绘制的 OK/NG 标注）。

        Returns:
            拼接后的图像，无板卡时返回 None。
        """
        if self._canvas is None or self._canvas_w <= 0 or self._canvas_h <= 0:
            return None
        return self._canvas.copy()

    def render_annotated(self) -> Optional[np.ndarray]:
        """返回带 OK/NG 标注的拼接整图（标注已在 add_board 时增量绘制）。

        Returns:
            拼接后的图像，无板卡时返回 None。
        """
        return self.render()


def build_stitcher_from_config(config: Dict[str, Any],
                               scale: float = 1.0) -> RigidStitcher:
    """根据产品配置构建拼接器（含所有点位坐标）。

    Args:
        config: 产品配置字典（含 grid 与 positions）
        scale: 轴坐标 → 像素换算比例

    Returns:
        已按配置初始化（含所有点位坐标占位）的 RigidStitcher。
        注意：此函数仅注册点位坐标，不包含图像；实际拼接时需调用 add_board。
    """
    stitcher = RigidStitcher(scale=scale)
    positions = config.get("positions", [])
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        name = pos.get("name", "")
        x = pos.get("x", 0)
        y = pos.get("y", 0)
        # 用占位图注册坐标，使布局可预先计算
        stitcher.add_board(name=name, x_axis=x, y_axis=y,
                           image=np.zeros((1, 1, 3), dtype=np.uint8),
                           passed=True)
    return stitcher
