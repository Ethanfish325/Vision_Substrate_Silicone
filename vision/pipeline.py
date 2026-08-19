# -*- coding: utf-8 -*-
import importlib
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from .tools.base_tool import VisionTool, ToolResult, PipelineContext

ALL_TOOLS: Dict[str, type] = {}

# 中文显示名称 -> 类名映射表（用于兼容旧格式方案文件）
# 旧版方案文件使用 display_name 作为 tool_type，需要映射到类名
CN_TO_EN: Dict[str, str] = {
    "多区域ROI": "MultiROI",
    "灰度化": "Grayscale",
    "高斯滤波": "GaussianBlur",
    "直方图均衡化": "HistEqualize",
    "形态学操作": "Morphology",
    "中值滤波": "MedianBlur",
    "缩放": "Resize",
    "自适应阈值": "AdaptiveThreshold",
    "Canny边缘检测": "CannyEdge",
    "阈值分割": "Threshold",
    "轮廓分析": "ContourAnalysis",
    "斑点检测": "BlobDetection",
    "Blob检测": "BlobDetection",
    "轮廓筛选": "ContourFilter",
    "直线检测": "LineDetection",
    "矩形检测": "RectangleDetection",
    "圆检测": "CircleDetection",
    "面积测量": "AreaMeasure",
    "距离测量": "DistanceMeasure",
    "点测量": "PointMeasure",
    "线测量": "LineMeasure",
    "角度测量": "AngleMeasure",
    "对象计数": "ObjectCount",
    "目标计数": "ObjectCount",
    "颜色识别": "ColorRecognition",
    "模板匹配": "TemplateMatch",
    "边缘匹配": "EdgeMatch",
    "快速匹配": "FastMatch",
    "亮度测量": "BrightnessMeasure",
    "坐标转换": "CoordinateTransform",
    "计算器": "Calculator",
    "数值计算": "Calculator",
    "逻辑判断": "LogicJudge",
    # geometry.py 重命名后的新类名
    "直线检测(霍夫)": "HoughLineDetection",
    "矩形检测(轮廓)": "ContourRectDetection",
    "Blob检测(简单)": "SimpleBlobDetect",
    # 兼容旧方案文件中可能使用的旧类名
    "LineDetection": "HoughLineDetection",
    "RectangleDetection": "ContourRectDetection",
    "BlobDetection": "SimpleBlobDetect",
}

_TOOL_CATEGORIES: Dict[str, List[str]] = {
    "预处理": [
        "Grayscale", "GaussianBlur", "HistEqualize", "Morphology",
        "MultiROI", "MedianBlur", "Resize", "AdaptiveThreshold"
    ],
    "特征提取": [
        "CannyEdge", "Threshold", "ContourAnalysis", "BlobDetection",
        "ContourFilter", "LineDetection", "RectangleDetection"
    ],
    "几何检测": [
        "CircleDetection", "HoughLineDetection", "ContourRectDetection", "SimpleBlobDetect"
    ],
    "测量": [
        "AreaMeasure", "DistanceMeasure", "PointMeasure",
        "LineMeasure", "AngleMeasure", "ObjectCount",
        "BrightnessMeasure"
    ],
    "识别": [
        "ColorRecognition", "TemplateMatch", "EdgeMatch", "FastMatch"
    ],
    "工具": [
        "CoordinateTransform", "Calculator", "LogicJudge"
    ],
}


def _register_all_tools():
    module_tools = {
        "preprocess": ["Grayscale", "GaussianBlur", "HistEqualize", "Morphology",
                       "MultiROI", "MedianBlur", "Resize", "AdaptiveThreshold"],
        "feature_extract": ["CannyEdge", "Threshold", "ContourAnalysis", "BlobDetection",
                            "ContourFilter", "LineDetection", "RectangleDetection"],
        "geometry": ["CircleDetection", "HoughLineDetection", "ContourRectDetection", "SimpleBlobDetect"],
        "measure": ["AreaMeasure", "DistanceMeasure", "PointMeasure",
                    "LineMeasure", "AngleMeasure", "ObjectCount",
                    "BrightnessMeasure"],
        "recognize": ["ColorRecognition", "TemplateMatch", "EdgeMatch", "FastMatch"],
        "utility": ["CoordinateTransform", "Calculator", "LogicJudge"],
    }

    for module_name, class_names in module_tools.items():
        try:
            mod = importlib.import_module(f".tools.{module_name}", package=__package__)
            for cls_name in class_names:
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    ALL_TOOLS[cls_name] = cls
        except ImportError as e:
            print(f"[pipeline] 导入模块 {module_name} 失败: {e}")


def get_all_tool_names() -> List[str]:
    return sorted(ALL_TOOLS.keys())


def get_tools_by_category() -> Dict[str, List[str]]:
    return dict(_TOOL_CATEGORIES)


def get_tool_category(tool_class_name: str) -> str:
    """Get the category display name for a tool by its class name."""
    for category, tools in _TOOL_CATEGORIES.items():
        if tool_class_name in tools:
            return category
    return ""


def create_tool(tool_type_name: str, params: Optional[Dict] = None) -> VisionTool:
    # 如果传入的是中文显示名称，自动转换为类名
    resolved_name = CN_TO_EN.get(tool_type_name, tool_type_name)
    if resolved_name not in ALL_TOOLS:
        raise ValueError(f"未知的工具类型: {tool_type_name}")
    tool_class = ALL_TOOLS[resolved_name]
    return tool_class(params)


_register_all_tools()


class PipelineStep:
    def __init__(self, tool: VisionTool, enabled: bool = True,
                 judge_rule: Optional[Dict] = None):
        self.tool = tool
        self.enabled = enabled
        self.judge_rule = judge_rule

    def to_dict(self) -> Dict:
        return {
            "tool_type": type(self.tool).__name__,
            "params": self.tool.to_dict(),
            "enabled": self.enabled,
            "judge_rule": self.judge_rule,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        tool = create_tool(data["tool_type"], data.get("params"))
        return cls(
            tool=tool,
            enabled=data.get("enabled", True),
            judge_rule=data.get("judge_rule"),
        )


class Pipeline:
    def __init__(self, name: str = ""):
        self.name = name
        self.steps: List[PipelineStep] = []

    def add_step(self, tool: VisionTool, enabled: bool = True,
                 judge_rule: Optional[Dict] = None):
        self.steps.append(PipelineStep(tool, enabled, judge_rule))

    def insert_step(self, index: int, tool: VisionTool, enabled: bool = True,
                    judge_rule: Optional[Dict] = None):
        self.steps.insert(index, PipelineStep(tool, enabled, judge_rule))

    def remove_step(self, index: int):
        if 0 <= index < len(self.steps):
            del self.steps[index]

    def move_step(self, from_index: int, to_index: int):
        if 0 <= from_index < len(self.steps) and 0 <= to_index < len(self.steps):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)

    def execute(self, cv_image: np.ndarray) -> Tuple[bool, List[ToolResult], np.ndarray, Dict[int, str]]:
        """执行流水线所有步骤。

        Returns:
            all_passed: 是否全部通过
            results: 各步骤的 ToolResult 列表
            current_image: 流水线处理后的最终图像
            step_roi_map: {step_index: roi_name} 映射，记录每个步骤使用的 ROI 区域名称
        """
        context = PipelineContext(
            original_image=cv_image.copy(),
            current_image=cv_image.copy(),
            regions={},
            results={}
        )

        results: List[ToolResult] = []
        all_passed = True
        step_roi_map: Dict[int, str] = {}  # step_index -> roi_name

        for i, step in enumerate(self.steps):
            if not step.enabled:
                results.append(ToolResult(
                    success=True, passed=True,
                    processed_image=context.current_image.copy(),
                    data={"skipped": True},
                    message=f"步骤{i+1}已跳过",
                    tool_type=type(step.tool).__name__,
                    tool_name=step.tool.display_name,
                    elapsed_ms=0.0,
                ))
                continue

            # 记录该步骤使用的 ROI 区域名称（如果配置了 _input_source 指向 region）
            # 兼容旧方案文件：可能存的是 "input_source"（无下划线前缀）
            input_source = step.tool.params.get("_input_source") or step.tool.params.get("input_source", "current")
            # 兼容中文 "区域:" 前缀（旧方案文件手动编辑可能使用中文）
            if input_source and input_source.startswith("区域:"):
                input_source = "region:" + input_source[3:]
            if input_source and input_source.startswith("region:"):
                step_roi_map[i] = input_source[7:]

            try:
                start = time.time()
                result = step.tool.process(context)
                elapsed = (time.time() - start) * 1000

                result.tool_type = type(step.tool).__name__
                result.tool_name = step.tool.display_name
                result.elapsed_ms = elapsed

                results.append(result)

                if not result.success:
                    all_passed = False
                    # 执行失败但仍继续后续步骤，让所有工具都有机会执行
                    # 但不再更新 context.current_image，保持上一帧图像
                else:
                    if result.processed_image is not None:
                        context.current_image = result.processed_image

                    if result.regions:
                        context.regions.update(result.regions)

                    context.results[type(step.tool).__name__] = result

                if not result.passed:
                    all_passed = False
                    # 不 break，继续执行后续步骤

            except Exception as e:
                results.append(ToolResult(
                    success=False, passed=False,
                    processed_image=context.current_image.copy(),
                    data={},
                    message=f"步骤{i+1}执行异常: {str(e)}",
                    tool_type=type(step.tool).__name__,
                    tool_name=step.tool.display_name,
                    elapsed_ms=0.0,
                ))
                all_passed = False
                # 不 break，继续执行后续步骤

        return all_passed, results, context.current_image, step_roi_map

    def _apply_judge_rule(self, result: ToolResult, rule: Dict) -> bool:
        rule_type = rule.get("type", "threshold")

        if rule_type == "threshold":
            key = rule.get("key", "value")
            min_val = rule.get("min", float("-inf"))
            max_val = rule.get("max", float("inf"))
            value = result.data.get(key, 0)
            return min_val <= value <= max_val

        elif rule_type == "pass_on_fail":
            return True

        return True

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict):
        pipeline = cls(name=data.get("name", ""))
        for step_data in data.get("steps", []):
            try:
                step = PipelineStep.from_dict(step_data)
                pipeline.steps.append(step)
            except Exception as e:
                print(f"[pipeline] 加载步骤失败: {e}")
        return pipeline

    @classmethod
    def from_dict_old(cls, data: Dict):
        pipeline = cls(name=data.get("name", ""))
        tools_data = data.get("tools", data.get("steps", []))
        for tool_data in tools_data:
            try:
                tool_type = tool_data.get("type", tool_data.get("tool_type"))
                tool_type = CN_TO_EN.get(tool_type, tool_type)
                params = tool_data.get("params", {})
                tool = create_tool(tool_type, params)
                pipeline.steps.append(PipelineStep(
                    tool=tool,
                    enabled=tool_data.get("enabled", True),
                    judge_rule=tool_data.get("judge_rule"),
                ))
            except Exception as e:
                print(f"[pipeline] 加载旧格式步骤失败: {e}")
        return pipeline

    def __str__(self):
        return f"Pipeline(name={self.name}, steps={len(self.steps)})"
