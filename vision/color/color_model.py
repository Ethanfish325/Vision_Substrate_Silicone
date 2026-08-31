# -*- coding: utf-8 -*-
"""颜色模型数据类。

ColorModel 描述一个可识别的目标颜色，包含颜色空间、中心值、
匹配方式（区间/距离/聚类）、容差等参数，并支持与 dict 互转以便持久化。
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ColorModel:
    """目标颜色模型。

    属性:
        name: 颜色名称（如 "自定义-蓝"）
        color_space: 颜色空间，HSV / Lab / RGB
        center: 颜色中心（对应空间的三个通道值）
        match_mode: 匹配方式
            - "range": 区间匹配，用 tolerance 作为各通道 ± 容差
            - "distance": 距离匹配，用 distance_threshold 作为欧氏距离阈值
            - "cluster": 聚类匹配，对 cluster_centers 逐个距离匹配后取并集
        tolerance: range 模式各通道容差（±）
        distance_threshold: distance 模式欧氏距离阈值
        cluster_centers: cluster 模式聚类中心列表（多峰颜色）
        normalize_illumination: 是否启用光照归一化
        adaptive_threshold: 是否启用自适应阈值
        source: 来源，preset / custom / temporary
    """

    name: str = "未命名"
    color_space: str = "HSV"
    center: Tuple[int, int, int] = (0, 0, 0)
    match_mode: str = "range"
    tolerance: Tuple[int, int, int] = (10, 50, 50)
    distance_threshold: float = 30.0
    cluster_centers: List[Tuple[int, int, int]] = field(default_factory=list)
    normalize_illumination: bool = False
    adaptive_threshold: bool = False
    source: str = "custom"

    def to_dict(self) -> dict:
        """转换为可 JSON 序列化的 dict。"""
        return {
            "name": self.name,
            "color_space": self.color_space,
            "center": list(self.center),
            "match_mode": self.match_mode,
            "tolerance": list(self.tolerance),
            "distance_threshold": float(self.distance_threshold),
            "cluster_centers": [list(c) for c in self.cluster_centers],
            "normalize_illumination": bool(self.normalize_illumination),
            "adaptive_threshold": bool(self.adaptive_threshold),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColorModel":
        """从 dict 恢复 ColorModel。"""
        if not d:
            return cls()
        return cls(
            name=d.get("name", "未命名"),
            color_space=d.get("color_space", "HSV"),
            center=tuple(d.get("center", [0, 0, 0])),
            match_mode=d.get("match_mode", "range"),
            tolerance=tuple(d.get("tolerance", [10, 50, 50])),
            distance_threshold=float(d.get("distance_threshold", 30.0)),
            cluster_centers=[tuple(c) for c in d.get("cluster_centers", [])],
            normalize_illumination=bool(d.get("normalize_illumination", False)),
            adaptive_threshold=bool(d.get("adaptive_threshold", False)),
            source=d.get("source", "custom"),
        )

    def __repr__(self) -> str:
        return (f"ColorModel(name={self.name!r}, space={self.color_space}, "
                f"center={self.center}, mode={self.match_mode})")
