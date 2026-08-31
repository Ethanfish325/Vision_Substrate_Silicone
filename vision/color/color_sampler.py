# -*- coding: utf-8 -*-
"""颜色采样器。

从图像上的点或 ROI 区域采样像素，统计生成 ColorModel。
支持自动推断容差、KMeans 聚类提取多峰颜色中心。
"""

from typing import List, Tuple

import numpy as np
import cv2

from .color_model import ColorModel
from .color_matcher import ColorMatcher


class ColorSampler:
    """颜色采样器。"""

    @staticmethod
    def _to_color_space(pixels_bgr: np.ndarray, color_space: str) -> np.ndarray:
        """将 BGR 像素数组转换为指定颜色空间。"""
        if color_space == "RGB":
            return pixels_bgr[:, ::-1]  # BGR -> RGB
        conv = ColorMatcher._CONV.get(color_space, cv2.COLOR_BGR2HSV)
        # 需要 3 通道图像才能 cvtColor
        img = pixels_bgr.reshape(1, -1, 3).astype(np.uint8)
        converted = cv2.cvtColor(img, conv)
        return converted.reshape(-1, 3)

    @staticmethod
    def auto_tolerance(pixels: np.ndarray, color_space: str = "HSV") -> Tuple[int, int, int]:
        """根据像素分布自动计算各通道容差（2~3 倍标准差）。

        Args:
            pixels: 颜色空间下的像素数组 (N, 3)
            color_space: 颜色空间，用于决定容差下限

        Returns:
            (tol0, tol1, tol2) 各通道 ± 容差
        """
        if pixels.shape[0] == 0:
            return (10, 50, 50)
        std = pixels.astype(np.float32).std(axis=0)
        # 2 倍标准差，至少保留最小容差
        tol = np.maximum(std * 2.0, [5, 20, 20])
        # HSV 的 H 通道范围 0~180，容差上限 60
        if color_space == "HSV":
            tol[0] = min(tol[0], 60)
        else:
            tol = np.minimum(tol, [40, 60, 60])
        return (int(round(tol[0])), int(round(tol[1])), int(round(tol[2])))

    @staticmethod
    def sample_point(image_bgr: np.ndarray, x: int, y: int,
                     radius: int = 3, color_space: str = "HSV",
                     name: str = "自定义颜色") -> ColorModel:
        """点选采样：取 (x,y) 邻域像素，生成 range 模型。

        Args:
            image_bgr: BGR 图像
            x, y: 采样点坐标
            radius: 邻域半径（取 (2r+1)x(2r+1) 窗口）
            color_space: 颜色空间
            name: 颜色名称

        Returns:
            ColorModel（range 模式，自动容差）
        """
        h, w = image_bgr.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(w, x + radius + 1), min(h, y + radius + 1)
        roi = image_bgr[y0:y1, x0:x1]
        pixels_bgr = roi.reshape(-1, 3)
        pixels = ColorSampler._to_color_space(pixels_bgr, color_space)
        center = tuple(int(v) for v in np.median(pixels, axis=0))
        tol = ColorSampler.auto_tolerance(pixels, color_space)
        return ColorModel(
            name=name, color_space=color_space, center=center,
            match_mode="range", tolerance=tol, source="custom",
        )

    @staticmethod
    def sample_roi(image_bgr: np.ndarray, x: int, y: int, w: int, h: int,
                   color_space: str = "HSV", name: str = "自定义颜色",
                   n_clusters: int = 3) -> ColorModel:
        """框选采样：对 ROI 内像素统计，生成 range 或 cluster 模型。

        若 ROI 内颜色分布集中（单峰），生成 range 模型；
        若分布分散（多峰，如带高光），生成 cluster 模型。

        Args:
            image_bgr: BGR 图像
            x, y, w, h: ROI 区域
            color_space: 颜色空间
            name: 颜色名称
            n_clusters: KMeans 聚类数上限

        Returns:
            ColorModel
        """
        img_h, img_w = image_bgr.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        roi = image_bgr[y:y + h, x:x + w]
        pixels_bgr = roi.reshape(-1, 3)
        pixels = ColorSampler._to_color_space(pixels_bgr, color_space)

        if pixels.shape[0] < 50:
            # 像素太少，退化为点采样
            return ColorSampler.sample_point(
                image_bgr, x + w // 2, y + h // 2, color_space=color_space, name=name)

        # KMeans 聚类，判断是否多峰
        k = min(n_clusters, max(1, pixels.shape[0] // 200))
        k = max(1, k)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(counts)[::-1]
        main_centers = [tuple(int(v) for v in centers[i]) for i in order]

        # 主簇占比
        main_ratio = counts[order[0]] / pixels.shape[0]

        if main_ratio >= 0.7 or k == 1:
            # 单峰：用主簇像素生成 range 模型
            main_mask = labels.flatten() == order[0]
            main_pixels = pixels[main_mask]
            center = tuple(int(v) for v in np.median(main_pixels, axis=0))
            tol = ColorSampler.auto_tolerance(main_pixels, color_space)
            return ColorModel(
                name=name, color_space=color_space, center=center,
                match_mode="range", tolerance=tol, source="custom",
            )

        # 多峰：生成 cluster 模型（取前 2 个主要簇）
        cluster_centers = main_centers[:2]
        # 用主簇像素标准差估计距离阈值
        main_mask = labels.flatten() == order[0]
        main_pixels = pixels[main_mask]
        std = main_pixels.astype(np.float32).std(axis=0)
        dist_threshold = float(np.sqrt(np.sum(std * std)) * 2.0 + 10.0)
        return ColorModel(
            name=name, color_space=color_space,
            center=cluster_centers[0], match_mode="cluster",
            cluster_centers=cluster_centers,
            distance_threshold=dist_threshold, source="custom",
        )
