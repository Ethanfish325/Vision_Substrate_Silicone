# -*- coding: utf-8 -*-
"""颜色匹配算法。

提供三种匹配方式：
    - range: 区间匹配（inRange），支持 HSV 色相环绕
    - distance: 距离匹配（欧氏距离阈值）
    - cluster: 聚类匹配（多中心取并集）

并包含光照归一化与自适应阈值等鲁棒性增强辅助函数。
"""

from typing import Optional

import numpy as np
import cv2

from .color_model import ColorModel


class ColorMatcher:
    """颜色匹配器。"""

    # 颜色空间 -> OpenCV 转换常量
    _CONV = {
        "HSV": cv2.COLOR_BGR2HSV,
        "Lab": cv2.COLOR_BGR2Lab,
        "RGB": cv2.COLOR_BGR2RGB,
    }

    @staticmethod
    def convert(image_bgr: np.ndarray, color_space: str) -> np.ndarray:
        """将 BGR 图像转换为指定颜色空间。"""
        conv = ColorMatcher._CONV.get(color_space, cv2.COLOR_BGR2HSV)
        return cv2.cvtColor(image_bgr, conv)

    @staticmethod
    def normalize_illumination(image_bgr: np.ndarray) -> np.ndarray:
        """光照归一化：灰度世界校正 + CLAHE 亮度均衡。

        灰度世界校正消除整体色偏；CLAHE 对亮度通道做对比度受限
        直方图均衡，抑制阴影/高光影响。
        """
        img = image_bgr.copy().astype(np.float32)

        # 灰度世界校正：各通道均值归一化到同一水平
        means = img.reshape(-1, 3).mean(axis=0)
        gray_mean = means.mean()
        if gray_mean > 0:
            scale = gray_mean / np.maximum(means, 1e-6)
            img = img * scale
        img = np.clip(img, 0, 255).astype(np.uint8)

        # CLAHE 亮度均衡（在 HSV 的 V 通道上）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _build_hsv_range_mask(hsv: np.ndarray, model: ColorModel) -> np.ndarray:
        """HSV 区间匹配，处理色相环绕。"""
        h_center, s_center, v_center = model.center
        h_tol, s_tol, v_tol = model.tolerance

        h_lo = (h_center - h_tol) % 180
        h_hi = (h_center + h_tol) % 180
        s_lo = max(0, s_center - s_tol)
        s_hi = min(255, s_center + s_tol)
        v_lo = max(0, v_center - v_tol)
        v_hi = min(255, v_center + v_tol)

        if h_lo <= h_hi:
            lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)

        # 跨 0 边界：两个区间取并集
        lower1 = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper1 = np.array([179, s_hi, v_hi], dtype=np.uint8)
        lower2 = np.array([0, s_lo, v_lo], dtype=np.uint8)
        upper2 = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)

    @staticmethod
    def _build_generic_range_mask(converted: np.ndarray, model: ColorModel) -> np.ndarray:
        """非 HSV 空间的区间匹配（Lab/RGB）。"""
        center = np.array(model.center, dtype=np.int32)
        tol = np.array(model.tolerance, dtype=np.int32)
        lower = np.clip(center - tol, 0, 255).astype(np.uint8)
        upper = np.clip(center + tol, 0, 255).astype(np.uint8)
        return cv2.inRange(converted, lower, upper)

    @staticmethod
    def match_range(image_bgr: np.ndarray, model: ColorModel) -> np.ndarray:
        """区间匹配。"""
        converted = ColorMatcher.convert(image_bgr, model.color_space)
        if model.color_space == "HSV":
            return ColorMatcher._build_hsv_range_mask(converted, model)
        return ColorMatcher._build_generic_range_mask(converted, model)

    @staticmethod
    def match_distance(image_bgr: np.ndarray, model: ColorModel) -> np.ndarray:
        """距离匹配：逐像素计算到颜色中心的欧氏距离，低于阈值则命中。"""
        converted = ColorMatcher.convert(image_bgr, model.color_space)
        center = np.array(model.center, dtype=np.float32)
        diff = converted.astype(np.float32) - center
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        return (dist <= model.distance_threshold).astype(np.uint8) * 255

    @staticmethod
    def match_cluster(image_bgr: np.ndarray, model: ColorModel) -> np.ndarray:
        """聚类匹配：对每个聚类中心做距离匹配后取并集。"""
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        if not model.cluster_centers:
            # 无聚类中心时退化为单中心距离匹配
            return ColorMatcher.match_distance(image_bgr, model)
        for center in model.cluster_centers:
            sub = ColorModel(
                name=model.name,
                color_space=model.color_space,
                center=tuple(int(v) for v in center),
                match_mode="distance",
                distance_threshold=model.distance_threshold,
            )
            mask = cv2.bitwise_or(mask, ColorMatcher.match_distance(image_bgr, sub))
        return mask

    @staticmethod
    def adaptive_threshold_mask(mask: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
        """自适应阈值：对掩膜候选区域用 Otsu 重新二值化，适应局部光照不均。

        仅对掩膜覆盖的像素做 Otsu 阈值，减少光照不均导致的漏检/误检。
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # 取掩膜覆盖的像素
        pixels = gray[mask > 0]
        if pixels.size == 0:
            return mask
        # Otsu 全局阈值
        thresh, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 用 Otsu 阈值重新生成掩膜（保留原掩膜区域内的像素）
        refined = np.zeros_like(mask)
        refined[(mask > 0) & (gray >= thresh)] = 255
        return refined

    @staticmethod
    def build_mask(image_bgr: np.ndarray, model: ColorModel) -> np.ndarray:
        """根据颜色模型生成二值掩膜（核心入口）。

        支持光照归一化与自适应阈值增强。
        """
        img = image_bgr
        if model.normalize_illumination:
            img = ColorMatcher.normalize_illumination(img)

        if model.match_mode == "distance":
            mask = ColorMatcher.match_distance(img, model)
        elif model.match_mode == "cluster":
            mask = ColorMatcher.match_cluster(img, model)
        else:
            mask = ColorMatcher.match_range(img, model)

        if model.adaptive_threshold:
            mask = ColorMatcher.adaptive_threshold_mask(mask, img)

        return mask

    @staticmethod
    def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """噪声抑制：中值滤波 + 开闭运算。

        开运算去孤立噪点，闭运算填补空洞。
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def region_stats(mask: np.ndarray, min_area: float = 0.0) -> list:
        """连通域分析，返回区域统计列表（比逐轮廓 Python 循环更快）。

        返回: [{"index", "area", "x", "y", "width", "height", "area_ratio"}, ...]
        """
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        total = mask.shape[0] * mask.shape[1]
        regions = []
        idx = 0
        for i in range(1, num):  # 0 是背景
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            idx += 1
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            regions.append({
                "index": idx,
                "area": float(area),
                "x": x, "y": y,
                "width": w, "height": h,
                "area_ratio": float(area / total * 100) if total > 0 else 0,
            })
        return regions
