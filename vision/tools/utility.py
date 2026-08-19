# -*- coding: utf-8 -*-

import re
import math
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

from .base_tool import VisionTool, ToolResult, PipelineContext


class CoordinateTransform(VisionTool):
    display_name = "坐标转换"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("scale", 0.1)
        self.params.setdefault("unit", "mm")
        self.params.setdefault("mode", "distance")
        self.params.setdefault("input_key", "")

    def process(self, context: PipelineContext) -> ToolResult:
        scale = float(self.params.get("scale", 0.1))
        unit = self.params.get("unit", "mm")
        mode = self.params.get("mode", "distance")
        input_key = self.params.get("input_key", "")

        input_value = None
        if input_key and input_key in context.results:
            result = context.results[input_key]
            if mode == "distance":
                input_value = result.data.get("distance") or result.data.get("avg_length")
            elif mode == "area":
                input_value = result.data.get("total_area") or result.data.get("area")
            elif mode == "point":
                input_value = result.data.get("points")

        if input_value is None:
            return ToolResult(
                success=False, passed=False,
                processed_image=context.current_image.copy(),
                data={},
                message=f"未找到输入数据 (key={input_key})"
            )

        if mode == "point" and isinstance(input_value, list):
            converted = [{"x": p["x"] * scale, "y": p["y"] * scale}
                        for p in input_value]
            result_data = {
                "converted_points": converted,
                "scale": scale,
                "unit": unit,
            }
            message = f"已转换 {len(converted)} 个点坐标 ({unit})"
        else:
            if isinstance(input_value, (list, tuple)):
                input_value = input_value[0] if input_value else 0
            converted = float(input_value) * scale
            result_data = {
                "original_value": float(input_value),
                "converted_value": converted,
                "scale": scale,
                "unit": unit,
            }
            message = f"{converted:.3f} {unit}"

        return ToolResult(
            success=True,
            passed=True,
            processed_image=context.current_image.copy(),
            data=result_data,
            message=message
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout,
                                      QWidget, QLabel)

        widgets = []

        mode_combo = QComboBox(parent)
        mode_combo.addItem("距离转换", "distance")
        mode_combo.addItem("面积转换", "area")
        mode_combo.addItem("点坐标转换", "point")
        current_mode = self.params.get("mode", "distance")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"mode": mode_combo.itemData(i)}))
        widgets.append(("模式:", mode_combo))

        scale_spin = QDoubleSpinBox(parent)
        scale_spin.setRange(0.0001, 1000)
        scale_spin.setDecimals(4)
        scale_spin.setValue(float(self.params.get("scale", 0.1)))
        scale_spin.valueChanged.connect(lambda v: self.params.update({"scale": v}))
        widgets.append(("像素当量:", scale_spin))

        unit_combo = QComboBox(parent)
        unit_combo.addItems(["mm", "cm", "m", "inch", "um"])
        current_unit = self.params.get("unit", "mm")
        idx = unit_combo.findText(current_unit)
        if idx >= 0:
            unit_combo.setCurrentIndex(idx)
        unit_combo.currentTextChanged.connect(
            lambda t: self.params.update({"unit": t}))
        widgets.append(("单位:", unit_combo))

        # 输入源选择 - 从上游步骤的结果中选择
        input_combo = QComboBox(parent)
        input_combo.addItem("-- 请选择 --", "")
        # 从 context_info 中获取上游步骤列表
        context_steps = getattr(self, '_context_step_list', [])
        for step in context_steps:
            step_name = step.get("name", "")
            step_index = step.get("index", -1)
            display = f"[{step_index}] {step_name}"
            input_combo.addItem(display, step_name)
        current_key = self.params.get("input_key", "")
        idx = input_combo.findData(current_key)
        if idx >= 0:
            input_combo.setCurrentIndex(idx)
        input_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"input_key": input_combo.itemData(i)}))
        widgets.append(("输入步骤:", input_combo))

        return widgets


class Calculator(VisionTool):
    display_name = "数值计算"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("expression", "{A} + {B}")
        self.params.setdefault("source_a", "")
        self.params.setdefault("source_b", "")
        self.params.setdefault("source_c", "")
        self.params.setdefault("pass_min", -999999)
        self.params.setdefault("pass_max", 999999)

    def _safe_eval(self, expr: str, a: float, b: float, c: float) -> Tuple[bool, float, str]:
        try:
            expr_safe = expr.replace("{A}", str(a))
            expr_safe = expr_safe.replace("{B}", str(b))
            expr_safe = expr_safe.replace("{C}", str(c))

            # 支持 ** 幂运算符（两个连续 * 号）
            allowed = set("0123456789+-*/()., %")
            if not all(c in allowed or c.isspace() for c in expr_safe):
                return False, 0, "表达式包含非法字符"

            # 使用受限的 eval，仅允许数学运算
            result = eval(expr_safe, {"__builtins__": {}}, {})
            return True, float(result), ""
        except Exception as e:
            return False, 0, f"计算错误: {str(e)}"

    def process(self, context: PipelineContext) -> ToolResult:
        expression = self.params.get("expression", "{A} + {B}")

        def get_value(source_key: str) -> float:
            if not source_key:
                return 0
            parts = source_key.split(".")
            tool_name = parts[0]
            data_key = parts[1] if len(parts) > 1 else None

            if tool_name in context.results:
                result = context.results[tool_name]
                if data_key:
                    return float(result.data.get(data_key, 0))
                else:
                    for key in ["value", "total_area", "distance", "angle",
                                 "count", "avg_length", "converted_value"]:
                        if key in result.data:
                            return float(result.data[key])
                    return 0
            return 0

        a = get_value(self.params.get("source_a", ""))
        b = get_value(self.params.get("source_b", ""))
        c = get_value(self.params.get("source_c", ""))

        success, result, error = self._safe_eval(expression, a, b, c)

        if not success:
            return ToolResult(
                success=False, passed=False,
                processed_image=context.current_image.copy(),
                data={"error": error},
                message=error
            )

        pass_min = float(self.params.get("pass_min", -999999))
        pass_max = float(self.params.get("pass_max", 999999))
        passed = pass_min <= result <= pass_max

        return ToolResult(
            success=True,
            passed=passed,
            processed_image=context.current_image.copy(),
            data={
                "value": result,
                "expression": expression,
                "a": a, "b": b, "c": c,
            },
            message=f"计算结果={result:.3f}"
        )

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QLineEdit, QComboBox, QDoubleSpinBox,
                                      QHBoxLayout, QWidget, QLabel)

        widgets = []

        expr_edit = QLineEdit(parent)
        expr_edit.setText(self.params.get("expression", "{A} + {B}"))
        expr_edit.textChanged.connect(lambda t: self.params.update({"expression": t}))
        widgets.append(("表达式:", expr_edit))

        source_a = QLineEdit(parent)
        source_a.setPlaceholderText("工具名.数据键")
        source_a.setText(self.params.get("source_a", ""))
        source_a.textChanged.connect(lambda t: self.params.update({"source_a": t}))
        widgets.append(("变量A:", source_a))

        source_b = QLineEdit(parent)
        source_b.setPlaceholderText("工具名.数据键")
        source_b.setText(self.params.get("source_b", ""))
        source_b.textChanged.connect(lambda t: self.params.update({"source_b": t}))
        widgets.append(("变量B:", source_b))

        source_c = QLineEdit(parent)
        source_c.setPlaceholderText("工具名.数据键")
        source_c.setText(self.params.get("source_c", ""))
        source_c.textChanged.connect(lambda t: self.params.update({"source_c": t}))
        widgets.append(("变量C:", source_c))

        pass_min = QDoubleSpinBox(parent)
        pass_min.setRange(-999999, 999999)
        pass_min.setValue(float(self.params.get("pass_min", -999999)))
        pass_min.valueChanged.connect(lambda v: self.params.update({"pass_min": v}))
        widgets.append(("合格下限:", pass_min))

        pass_max = QDoubleSpinBox(parent)
        pass_max.setRange(-999999, 999999)
        pass_max.setValue(float(self.params.get("pass_max", 999999)))
        pass_max.valueChanged.connect(lambda v: self.params.update({"pass_max": v}))
        widgets.append(("合格上限:", pass_max))

        return widgets


class LogicJudge(VisionTool):
    display_name = "逻辑判断"

    def __init__(self, params=None):
        super().__init__(params)
        self.params.setdefault("judge_mode", "expression")  # "expression" or "condition"
        self.params.setdefault("expression", "AND(area>100, distance<50)")
        self.params.setdefault("conditions", [])
        self.params.setdefault("logic_type", "and")

    def _resolve_value(self, context: PipelineContext, source_key: str,
                       default: float = 0) -> float:
        """从上下文解析数值，支持 工具名.数据键 格式"""
        if not source_key:
            return default

        parts = source_key.split(".")
        tool_name = parts[0]
        data_key = parts[1] if len(parts) > 1 else None

        # 直接查找：工具名.数据键
        if tool_name in context.results:
            result = context.results[tool_name]
            if data_key:
                return float(result.data.get(data_key, default))
            else:
                for key in ["value", "total_area", "distance", "angle",
                             "count", "avg_length", "converted_value",
                             "score", "match_count", "best_score",
                             "area_ratio", "color_area", "judge_result"]:
                    if key in result.data:
                        return float(result.data[key])
                return default

        # 模糊查找：如果直接没找到，遍历所有结果查找匹配的数据键
        # 支持简写语法如 "area>100" 自动匹配包含 area 键的结果
        if data_key is None:
            for tname, result in context.results.items():
                if hasattr(result, 'data') and result.data:
                    # 尝试精确匹配键名
                    if tool_name in result.data:
                        return float(result.data[tool_name])
                    # 尝试模糊匹配键名（忽略大小写）
                    for rkey, rval in result.data.items():
                        if isinstance(rval, (int, float)) and tool_name.lower() in rkey.lower():
                            return float(rval)
            return default
        else:
            # 有 data_key 但 tool_name 没找到，尝试在所有结果中找 data_key
            for tname, result in context.results.items():
                if hasattr(result, 'data') and data_key in result.data:
                    return float(result.data[data_key])
            return default

    def _parse_expression(self, expr: str) -> Tuple[str, List[Dict]]:
        """
        解析表达式字符串，返回 (logic_type, conditions)
        支持格式：
        - AND(面积>100, 距离<50)
        - OR(面积>100, 距离<50)
        - 面积>100 AND 距离<50
        - 面积>100 OR 距离<50
        - 面积>100（单条件）
        """
        expr = expr.strip()

        # 尝试匹配函数式语法: AND(...) / OR(...)
        func_match = re.match(r'^(AND|OR)\((.+)\)$', expr, re.IGNORECASE)
        if func_match:
            logic_type = func_match.group(1).lower()
            inner = func_match.group(2)
            # 按逗号分割条件（注意括号内的逗号不分割）
            cond_strs = self._split_conditions(inner)
            conditions = []
            for cs in cond_strs:
                cs = cs.strip()
                if cs:
                    cond = self._parse_single_condition(cs)
                    if cond:
                        conditions.append(cond)
            return logic_type, conditions

        # 尝试匹配中缀语法: cond1 AND cond2 / cond1 OR cond2
        and_parts = re.split(r'\s+AND\s+', expr, flags=re.IGNORECASE)
        if len(and_parts) > 1:
            conditions = []
            for part in and_parts:
                cond = self._parse_single_condition(part.strip())
                if cond:
                    conditions.append(cond)
            return "and", conditions

        or_parts = re.split(r'\s+OR\s+', expr, flags=re.IGNORECASE)
        if len(or_parts) > 1:
            conditions = []
            for part in or_parts:
                cond = self._parse_single_condition(part.strip())
                if cond:
                    conditions.append(cond)
            return "or", conditions

        # 单条件
        cond = self._parse_single_condition(expr)
        if cond:
            return "and", [cond]
        return "and", []

    def _split_conditions(self, text: str) -> List[str]:
        """按逗号分割条件，忽略括号内的逗号"""
        parts = []
        depth = 0
        current = ""
        for ch in text:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current)
        return parts

    def _parse_single_condition(self, text: str) -> Optional[Dict]:
        """
        解析单个条件，如:
        - 面积>100
        - 距离<50
        - 面积>=100 AND 距离<=200
        - 面积 == 100
        - 面积 != 100
        - 100 <= 面积 <= 200
        """
        text = text.strip()

        # 尝试匹配范围语法: min <= value <= max
        range_match = re.match(
            r'([\w.]+)\s*<=\s*([\w.]+)\s*<=\s*([\w.]+)', text)
        if range_match:
            return {
                "source": range_match.group(2),
                "min": self._try_parse_number(range_match.group(1)),
                "max": self._try_parse_number(range_match.group(3)),
                "operator": "range",
                "raw": text,
            }

        # 尝试匹配比较运算符: !=, >=, <=, ==, >, <
        op_match = re.match(
            r'([\w.]+)\s*(!=|>=|<=|==|>|<)\s*([\w.\-]+)', text)
        if op_match:
            source = op_match.group(1)
            op = op_match.group(2)
            val = self._try_parse_number(op_match.group(3))
            if op == ">":
                return {"source": source, "min": val, "max": 999999, "operator": ">", "raw": text}
            elif op == "<":
                return {"source": source, "min": -999999, "max": val, "operator": "<", "raw": text}
            elif op == ">=":
                return {"source": source, "min": val, "max": 999999, "operator": ">=", "raw": text}
            elif op == "<=":
                return {"source": source, "min": -999999, "max": val, "operator": "<=", "raw": text}
            elif op == "==":
                return {"source": source, "min": val, "max": val, "operator": "==", "raw": text}
            elif op == "!=":
                return {"source": source, "min": val, "max": val, "operator": "!=", "raw": text}

        return None

    def _try_parse_number(self, text: str) -> float:
        """尝试将字符串解析为数字，如果是变量名则返回0"""
        text = text.strip()
        try:
            return float(text)
        except ValueError:
            return 0

    def _evaluate_condition(self, value: float, cond: Dict) -> bool:
        """评估单个条件"""
        op = cond.get("operator", "range")
        min_val = float(cond.get("min", -999999))
        max_val = float(cond.get("max", 999999))

        if op == "!=":
            return value != min_val
        elif op in (">", ">=", "<", "<=", "==", "range"):
            return min_val <= value <= max_val
        return min_val <= value <= max_val

    def _get_debug_info(self, context: PipelineContext) -> List[Dict]:
        """收集所有上游步骤的调试信息"""
        debug_values = []
        for tool_name, result in context.results.items():
            if hasattr(result, 'data') and result.data:
                for key, val in result.data.items():
                    if isinstance(val, (int, float)):
                        debug_values.append({
                            "source": f"{tool_name}.{key}",
                            "tool": tool_name,
                            "key": key,
                            "value": float(val),
                        })
        return debug_values

    def process(self, context: PipelineContext) -> ToolResult:
        img = context.current_image
        if img is None:
            return ToolResult(
                success=False, passed=False,
                processed_image=np.zeros((100, 100, 3), dtype=np.uint8),
                data={},
                message="无输入图像"
            )

        judge_mode = self.params.get("judge_mode", "expression")
        debug_values = self._get_debug_info(context)

        if judge_mode == "expression":
            # 表达式模式
            expr_str = self.params.get("expression", "")
            logic_type, conditions = self._parse_expression(expr_str)

            if not conditions:
                return ToolResult(
                    success=True, passed=True,
                    processed_image=img.copy(),
                    data={
                        "judge_result": True,
                        "judge_mode": "expression",
                        "expression": expr_str,
                        "condition_count": 0,
                        "debug_values": debug_values,
                    },
                    message="表达式为空，默认通过"
                )

            results = []
            all_passed = True

            for cond in conditions:
                source = cond.get("source", "")
                value = self._resolve_value(context, source)
                passed = self._evaluate_condition(value, cond)

                results.append({
                    "source": source,
                    "value": value,
                    "min": cond.get("min", -999999),
                    "max": cond.get("max", 999999),
                    "operator": cond.get("operator", "range"),
                    "raw": cond.get("raw", ""),
                    "passed": passed,
                })

                if logic_type == "and":
                    all_passed = all_passed and passed
                else:
                    if passed:
                        all_passed = True

            if logic_type == "or":
                all_passed = any(r["passed"] for r in results)

            passed_count = sum(1 for r in results if r["passed"])
            total_count = len(results)

            return ToolResult(
                success=True,
                passed=all_passed,
                processed_image=img.copy(),
                data={
                    "judge_result": all_passed,
                    "judge_mode": "expression",
                    "expression": expr_str,
                    "logic_type": logic_type,
                    "condition_count": total_count,
                    "passed_count": passed_count,
                    "details": results,
                    "debug_values": debug_values,
                },
                message=f"逻辑判断: {passed_count}/{total_count} 通过" +
                        f" ({'通过' if all_passed else '不通过'})"
            )

        else:
            # 条件模式（向后兼容）
            logic_type = self.params.get("logic_type", "and")
            conditions = self.params.get("conditions", [])

            if not conditions:
                return ToolResult(
                    success=True, passed=True,
                    processed_image=img.copy(),
                    data={
                        "judge_result": True,
                        "judge_mode": "condition",
                        "condition_count": 0,
                        "debug_values": debug_values,
                    },
                    message="无条件，默认通过"
                )

            results = []
            all_passed = True

            for cond in conditions:
                source = cond.get("source", "")
                min_val = float(cond.get("min", -999999))
                max_val = float(cond.get("max", 999999))

                value = self._resolve_value(context, source)
                passed = min_val <= value <= max_val

                results.append({
                    "source": source,
                    "value": value,
                    "min": min_val,
                    "max": max_val,
                    "passed": passed,
                })

                if logic_type == "and":
                    all_passed = all_passed and passed
                else:
                    if passed:
                        all_passed = True

            if logic_type == "or":
                all_passed = any(r["passed"] for r in results)

            passed_count = sum(1 for r in results if r["passed"])
            total_count = len(results)

            return ToolResult(
                success=True,
                passed=all_passed,
                processed_image=img.copy(),
                data={
                    "judge_result": all_passed,
                    "judge_mode": "condition",
                    "logic_type": logic_type,
                    "condition_count": total_count,
                    "passed_count": passed_count,
                    "details": results,
                    "debug_values": debug_values,
                },
                message=f"逻辑判断: {passed_count}/{total_count} 通过" +
                        f" ({'通过' if all_passed else '不通过'})"
            )

    def _build_value_row(self, parent, label_text: str,
                         getter, setter):
        from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QDoubleSpinBox)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        spin = QDoubleSpinBox()
        spin.setRange(-999999, 999999)
        spin.setValue(getter())
        spin.valueChanged.connect(setter)

        layout.addWidget(label)
        layout.addWidget(spin)
        layout.addStretch()

        return row

    def get_param_widgets(self, parent):
        from PyQt5.QtWidgets import (QComboBox, QPushButton, QVBoxLayout,
                                      QHBoxLayout, QWidget, QLabel,
                                      QDoubleSpinBox, QLineEdit,
                                      QGroupBox, QScrollArea, QStackedWidget,
                                      QTextEdit, QTableWidget, QTableWidgetItem,
                                      QHeaderView, QAbstractItemView)
        from PyQt5.QtCore import Qt

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 模式选择
        mode_combo = QComboBox()
        mode_combo.addItem("表达式模式（推荐）", "expression")
        mode_combo.addItem("条件模式（兼容）", "condition")
        current_mode = self.params.get("judge_mode", "expression")
        idx = mode_combo.findData(current_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        main_layout.addWidget(QLabel("判断模式:"))
        main_layout.addWidget(mode_combo)

        # 堆叠面板：表达式模式 / 条件模式
        stacked = QStackedWidget()

        # ====== 页面0: 表达式模式 ======
        expr_page = QWidget()
        expr_layout = QVBoxLayout(expr_page)
        expr_layout.setContentsMargins(0, 0, 0, 0)

        # 逻辑类型选择
        expr_logic_combo = QComboBox()
        expr_logic_combo.addItem("全部通过 (AND)", "and")
        expr_logic_combo.addItem("任一通过 (OR)", "or")
        # 从现有表达式解析逻辑类型
        current_expr = self.params.get("expression", "")
        parsed_logic, _ = self._parse_expression(current_expr)
        expr_logic_combo.setCurrentIndex(0 if parsed_logic == "and" else 1)
        expr_layout.addWidget(QLabel("逻辑类型:"))
        expr_layout.addWidget(expr_logic_combo)

        # 条件列表（可视化构建）
        conditions_group = QGroupBox("判断条件（双击修改数值）")
        conditions_layout = QVBoxLayout(conditions_group)

        # 获取上游步骤列表
        context_steps = getattr(self, '_context_step_list', [])

        # 构建可用变量列表（用于下拉选择）
        def _build_var_list():
            var_list = []
            for step in context_steps:
                step_name = step.get("name", "")
                if not step_name:
                    continue
                common_keys = self._get_common_measurement_keys(step_name)
                for key in common_keys:
                    full_name = f"{step_name}.{key}"
                    desc = self._get_key_description(key)
                    var_list.append((full_name, f"{step_name}.{key} ({desc})"))
            return var_list

        # 存储条件列表（每个条件是一个dict）
        expr_conditions = []

        # 从现有表达式解析初始条件
        def _init_from_expr():
            nonlocal expr_conditions
            expr_str = self.params.get("expression", "")
            if expr_str.strip():
                logic_type, conds = self._parse_expression(expr_str)
                # 同步逻辑类型
                idx = expr_logic_combo.findData(logic_type)
                if idx >= 0:
                    expr_logic_combo.setCurrentIndex(idx)
                expr_conditions.clear()
                for c in conds:
                    source = c.get("source", "")
                    op = c.get("operator", ">")
                    val = c.get("min", 0)
                    if op in (">=", ">"):
                        val = c.get("min", 0)
                    elif op in ("<=", "<"):
                        val = c.get("max", 0)
                    elif op == "range":
                        val = c.get("max", 0)
                    expr_conditions.append({
                        "source": source,
                        "operator": op if op != "range" else ">",
                        "value": val if op not in ("<", "<=") else c.get("max", 0),
                    })
            if not expr_conditions:
                expr_conditions.append({"source": "", "operator": ">", "value": 0})

        _init_from_expr()

        # 生成表达式字符串
        def _build_expr_string():
            logic_type = expr_logic_combo.currentData()
            parts = []
            for cond in expr_conditions:
                source = cond.get("source", "")
                op = cond.get("operator", ">")
                val = cond.get("value", 0)
                if source:
                    parts.append(f"{source}{op}{val}")
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            if logic_type == "and":
                return f"AND({', '.join(parts)})"
            else:
                return f"OR({', '.join(parts)})"

        # 更新表达式显示和参数
        def _update_expr():
            expr_str = _build_expr_string()
            self.params.update({"expression": expr_str})
            expr_display.setText(expr_str if expr_str else "（无条件）")

        # 条件滚动区域
        cond_scroll = QScrollArea()
        cond_scroll.setWidgetResizable(True)
        cond_scroll_widget = QWidget()
        cond_scroll_layout = QVBoxLayout(cond_scroll_widget)

        def rebuild_expr_conditions():
            while cond_scroll_layout.count():
                item = cond_scroll_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            var_list = _build_var_list()

            for i, cond in enumerate(expr_conditions):
                row_widget = self._build_expr_condition_row(
                    i, cond, expr_conditions,
                    var_list, rebuild_expr_conditions, _update_expr)
                cond_scroll_layout.addWidget(row_widget)

            cond_scroll_layout.addStretch()

        def add_expr_condition():
            expr_conditions.append({"source": "", "operator": ">", "value": 0})
            rebuild_expr_conditions()
            _update_expr()

        rebuild_expr_conditions()

        cond_scroll.setWidget(cond_scroll_widget)
        conditions_layout.addWidget(cond_scroll)

        add_cond_btn = QPushButton("+ 添加条件")
        add_cond_btn.clicked.connect(add_expr_condition)
        conditions_layout.addWidget(add_cond_btn)

        expr_layout.addWidget(conditions_group)

        # 表达式预览（只读）
        expr_layout.addWidget(QLabel("表达式预览:"))
        expr_display = QLabel("")
        expr_display.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; padding: 6px; "
            "border: 1px solid #444; border-radius: 2px; font-family: Consolas;")
        expr_display.setWordWrap(True)
        expr_layout.addWidget(expr_display)

        # 逻辑类型变更时更新
        expr_logic_combo.currentIndexChanged.connect(lambda i: _update_expr())

        # 初始更新
        _update_expr()

        stacked.addWidget(expr_page)

        # ====== 页面1: 条件模式（向后兼容） ======
        cond_page = QWidget()
        cond_layout = QVBoxLayout(cond_page)
        cond_layout.setContentsMargins(0, 0, 0, 0)

        logic_combo = QComboBox()
        logic_combo.addItem("全部通过 (AND)", "and")
        logic_combo.addItem("任一通过 (OR)", "or")
        current_logic = self.params.get("logic_type", "and")
        idx2 = logic_combo.findData(current_logic)
        if idx2 >= 0:
            logic_combo.setCurrentIndex(idx2)
        logic_combo.currentIndexChanged.connect(
            lambda i: self.params.update({"logic_type": logic_combo.itemData(i)}))
        cond_layout.addWidget(QLabel("逻辑类型:"))
        cond_layout.addWidget(logic_combo)

        context_steps = getattr(self, '_context_step_list', [])

        conditions_group2 = QGroupBox("判断条件")
        conditions_layout2 = QVBoxLayout(conditions_group2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget2 = QWidget()
        scroll_layout2 = QVBoxLayout(scroll_widget2)

        conditions = self.params.get("conditions", [])

        def rebuild_conditions():
            while scroll_layout2.count():
                item = scroll_layout2.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            for i, cond in enumerate(conditions):
                cond_widget = self._build_condition_row(i, cond, conditions,
                                                        rebuild_conditions,
                                                        context_steps)
                scroll_layout2.addWidget(cond_widget)

            scroll_layout2.addStretch()

        def add_condition():
            conditions.append({
                "source": "",
                "min": 0,
                "max": 100,
            })
            rebuild_conditions()

        add_btn = QPushButton("+ 添加条件")
        add_btn.clicked.connect(add_condition)

        scroll.setWidget(scroll_widget2)
        conditions_layout2.addWidget(scroll)
        conditions_layout2.addWidget(add_btn)

        cond_layout.addWidget(conditions_group2)

        stacked.addWidget(cond_page)

        # 模式切换
        def on_mode_changed(idx):
            stacked.setCurrentIndex(idx)

        mode_combo.currentIndexChanged.connect(on_mode_changed)
        stacked.setCurrentIndex(0 if current_mode == "expression" else 1)

        main_layout.addWidget(stacked)

        return [("逻辑判断:", main_widget)]

    def _build_expr_condition_row(self, index: int, condition: Dict,
                                   conditions: List, var_list: List,
                                   refresh_callback, update_callback):
        """构建表达式模式下的单条件行（下拉选择变量 + 运算符 + 数值输入）"""
        from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QDoubleSpinBox,
                                      QComboBox, QPushButton, QWidget)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)

        # 变量选择下拉框
        var_combo = QComboBox()
        var_combo.addItem("-- 选择变量 --", "")
        for full_name, display_text in var_list:
            var_combo.addItem(display_text, full_name)
        var_combo.setFixedWidth(200)
        current_source = condition.get("source", "")
        idx = var_combo.findData(current_source)
        if idx >= 0:
            var_combo.setCurrentIndex(idx)
        var_combo.currentIndexChanged.connect(
            lambda i: _on_var_changed(var_combo.itemData(i)))
        layout.addWidget(QLabel(f"#{index+1}:"))
        layout.addWidget(var_combo)

        # 运算符选择
        op_combo = QComboBox()
        op_combo.addItem(">", ">")
        op_combo.addItem(">=", ">=")
        op_combo.addItem("<", "<")
        op_combo.addItem("<=", "<=")
        op_combo.addItem("==", "==")
        op_combo.addItem("!=", "!=")
        op_combo.setFixedWidth(55)
        current_op = condition.get("operator", ">")
        idx2 = op_combo.findData(current_op)
        if idx2 >= 0:
            op_combo.setCurrentIndex(idx2)
        op_combo.currentIndexChanged.connect(
            lambda i: _on_op_changed(op_combo.itemData(i)))
        layout.addWidget(op_combo)

        # 数值输入
        val_spin = QDoubleSpinBox()
        val_spin.setRange(-999999, 999999)
        val_spin.setDecimals(1)
        val_spin.setValue(float(condition.get("value", 0)))
        val_spin.setFixedWidth(100)
        val_spin.valueChanged.connect(
            lambda v: _on_val_changed(v))
        layout.addWidget(val_spin)

        # 删除按钮
        def remove_cond():
            if 0 <= index < len(conditions):
                conditions.pop(index)
                refresh_callback()
                update_callback()

        del_btn = QPushButton("\u00d7")
        del_btn.setFixedWidth(25)
        del_btn.clicked.connect(remove_cond)
        layout.addWidget(del_btn)

        def _on_var_changed(val):
            conditions[index]["source"] = val if val else ""
            update_callback()

        def _on_op_changed(op):
            conditions[index]["operator"] = op
            update_callback()

        def _on_val_changed(v):
            conditions[index]["value"] = v
            update_callback()

        return row

    def _get_key_description(self, key: str) -> str:
        """获取数据键的中文描述"""
        descriptions = {
            "value": "数值（通用）",
            "total_area": "总面积",
            "max_area": "最大面积",
            "area": "面积",
            "distance": "距离",
            "angle": "角度",
            "count": "数量",
            "avg_length": "平均长度",
            "line_count": "线段数",
            "point_count": "点数",
            "converted_value": "转换后数值",
            "score": "匹配分数",
            "best_score": "最佳匹配分数",
            "match_count": "匹配数量",
            "area_ratio": "颜色占比",
            "color_area": "颜色面积",
            "judge_result": "判断结果",
            "edge_count": "边缘点数",
            "threshold_value": "阈值",
            "filtered_count": "筛选后数量",
            "blob_count": "斑点数量",
            "rect_count": "矩形数量",
            "circle_count": "圆数量",
            "scale_x": "X方向缩放比",
            "scale_y": "Y方向缩放比",
            "condition_count": "条件数量",
            "passed_count": "通过数量",
        }
        return descriptions.get(key, key)

    def _build_condition_row(self, index: int, condition: Dict,
                              conditions: List, refresh_callback,
                              context_steps: list = None):
        from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QDoubleSpinBox,
                                      QComboBox, QPushButton, QWidget)

        if context_steps is None:
            context_steps = []

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)

        # 使用下拉选择替代文本输入
        source_combo = QComboBox()
        source_combo.addItem("-- 请选择 --", "")
        for step in context_steps:
            step_name = step.get("name", "")
            step_index = step.get("index", -1)
            display = f"[{step_index}] {step_name}"
            source_combo.addItem(display, step_name)
        source_combo.setFixedWidth(140)
        current_source = condition.get("source", "")
        idx = source_combo.findData(current_source)
        if idx >= 0:
            source_combo.setCurrentIndex(idx)
        source_combo.currentIndexChanged.connect(
            lambda i: conditions[index].update({"source": source_combo.itemData(i)}))
        layout.addWidget(QLabel(f"#{index+1}:"))
        layout.addWidget(source_combo)

        min_spin = QDoubleSpinBox()
        min_spin.setRange(-999999, 999999)
        min_spin.setValue(float(condition.get("min", 0)))
        min_spin.setFixedWidth(80)
        min_spin.valueChanged.connect(
            lambda v: conditions[index].update({"min": v}))
        layout.addWidget(QLabel("\u2265"))
        layout.addWidget(min_spin)

        max_spin = QDoubleSpinBox()
        max_spin.setRange(-999999, 999999)
        max_spin.setValue(float(condition.get("max", 100)))
        max_spin.setFixedWidth(80)
        max_spin.valueChanged.connect(
            lambda v: conditions[index].update({"max": v}))
        layout.addWidget(QLabel("\u2264"))
        layout.addWidget(max_spin)

        def remove_cond():
            if 0 <= index < len(conditions):
                conditions.pop(index)
                refresh_callback()

        del_btn = QPushButton("\u00d7")
        del_btn.setFixedWidth(25)
        del_btn.clicked.connect(remove_cond)
        layout.addWidget(del_btn)

        return row

    def _get_common_measurement_keys(self, tool_name: str) -> list:
        common_keys = {
            "AreaMeasure": ["total_area", "max_area", "count"],
            "DistanceMeasure": ["distance", "count"],
            "PointMeasure": ["point_count"],
            "LineMeasure": ["avg_length", "line_count"],
            "AngleMeasure": ["angle"],
            "ObjectCount": ["count"],
            "Calculator": ["value"],
            "CoordinateTransform": ["converted_value"],
            "ColorRecognition": ["color_area", "area_ratio"],
            "TemplateMatch": ["match_count", "score", "best_score"],
            "EdgeMatch": ["match_count"],
            "FastMatch": ["score"],
            "CannyEdge": ["edge_count"],
            "Threshold": ["threshold_value"],
            "ContourAnalysis": ["total_area", "max_area", "count"],
            "ContourFilter": ["filtered_count"],
            "BlobDetection": ["blob_count"],
            "LineDetection": ["line_count"],
            "RectangleDetection": ["rect_count"],
            "CircleDetection": ["circle_count"],
            "HoughLineDetection": ["line_count"],
            "ContourRectDetection": ["rect_count"],
            "SimpleBlobDetect": ["blob_count"],
            "Morphology": [],
            "MultiROI": [],
            "Resize": ["scale_x", "scale_y"],
            "Grayscale": [],
            "GaussianBlur": [],
            "MedianBlur": [],
            "HistEqualize": [],
            "AdaptiveThreshold": [],
            "LogicJudge": ["judge_result", "condition_count", "passed_count"],
        }
        return common_keys.get(tool_name, ["value"])
