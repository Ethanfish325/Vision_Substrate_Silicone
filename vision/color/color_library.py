# -*- coding: utf-8 -*-
"""颜色库管理。

提供内置预设颜色库、用户自定义颜色库（JSON 持久化）与临时颜色集合。
"""

import os
import json
from typing import List

from .color_model import ColorModel


class ColorLibrary:
    """颜色库管理器。

    预设库（内置，不可修改） + 自定义库（JSON 持久化） + 临时集合（不持久化）。
    """

    # 内置预设颜色（HSV 区间）
    PRESETS: List[ColorModel] = [
        ColorModel(name="红色", color_space="HSV", center=(0, 255, 255),
                   tolerance=(10, 50, 50), match_mode="range", source="preset"),
        ColorModel(name="绿色", color_space="HSV", center=(60, 255, 255),
                   tolerance=(25, 50, 50), match_mode="range", source="preset"),
        ColorModel(name="蓝色", color_space="HSV", center=(115, 255, 255),
                   tolerance=(15, 50, 50), match_mode="range", source="preset"),
        ColorModel(name="黄色", color_space="HSV", center=(27, 255, 255),
                   tolerance=(8, 50, 50), match_mode="range", source="preset"),
        ColorModel(name="橙色", color_space="HSV", center=(17, 255, 255),
                   tolerance=(8, 50, 50), match_mode="range", source="preset"),
        ColorModel(name="紫色", color_space="HSV", center=(145, 255, 255),
                   tolerance=(15, 50, 50), match_mode="range", source="preset"),
        ColorModel(name="白色", color_space="HSV", center=(0, 15, 255),
                   tolerance=(180, 15, 55), match_mode="range", source="preset"),
        ColorModel(name="黑色", color_space="HSV", center=(0, 0, 25),
                   tolerance=(180, 255, 25), match_mode="range", source="preset"),
    ]

    def __init__(self, path: str = "data/color_library.json"):
        self.path = path
        self._custom: List[ColorModel] = []
        self._temporary: List[ColorModel] = []
        self._load()

    def _load(self):
        """从 JSON 文件加载自定义颜色库。"""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._custom = [ColorModel.from_dict(d) for d in data.get("custom", [])]
        except Exception as e:  # noqa: BLE001
            print(f"[ColorLibrary] 加载颜色库失败: {e}")

    def _save(self):
        """保存自定义颜色库到 JSON 文件。"""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            data = {"custom": [m.to_dict() for m in self._custom]}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:  # noqa: BLE001
            print(f"[ColorLibrary] 保存颜色库失败: {e}")

    def get_presets(self) -> List[ColorModel]:
        """返回内置预设颜色列表。"""
        return list(self.PRESETS)

    def get_custom(self) -> List[ColorModel]:
        """返回用户自定义颜色列表。"""
        return list(self._custom)

    def get_temporary(self) -> List[ColorModel]:
        """返回临时颜色集合（不持久化）。"""
        return list(self._temporary)

    def get_all(self) -> List[ColorModel]:
        """返回预设 + 自定义 + 临时全部颜色。"""
        return self.get_presets() + self.get_custom() + self.get_temporary()

    def add(self, model: ColorModel, persist: bool = True) -> None:
        """加入颜色库。

        Args:
            model: 颜色模型
            persist: True 加入自定义库并持久化；False 加入临时集合
        """
        if persist:
            # 同名覆盖
            self._custom = [m for m in self._custom if m.name != model.name]
            self._custom.append(model)
            self._save()
        else:
            self._temporary = [m for m in self._temporary if m.name != model.name]
            self._temporary.append(model)

    def remove(self, name: str) -> bool:
        """从自定义库删除指定名称的颜色。"""
        before = len(self._custom)
        self._custom = [m for m in self._custom if m.name != name]
        if len(self._custom) != before:
            self._save()
            return True
        return False

    def clear_temporary(self) -> None:
        """清空临时颜色集合。"""
        self._temporary = []
