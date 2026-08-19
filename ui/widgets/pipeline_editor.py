# -*- coding: utf-8 -*-
from typing import Dict, Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from vision.pipeline import Pipeline, create_tool
from vision.tools.base_tool import VisionTool

from .operator_toolbox import OperatorToolbox
from .flow_canvas import FlowCanvas
from .param_config_dialog import ParamConfigDialog, MultiROIEditorDialog


class PipelineEditor(QWidget):
    pipeline_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipeline = Pipeline("未命名")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        toolbar = self._build_toolbar()
        main_layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        self.operator_toolbox = OperatorToolbox()
        self.operator_toolbox.setMinimumWidth(120)
        self.operator_toolbox.setMaximumWidth(160)

        self.flow_canvas = FlowCanvas()
        self.flow_canvas.setMinimumWidth(300)

        splitter.addWidget(self.operator_toolbox)
        splitter.addWidget(self.flow_canvas)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([150, 600])

        main_layout.addWidget(splitter, 1)

        self._connect_signals()

    def _build_toolbar(self) -> QWidget:
        toolbar = QWidget()
        toolbar.setStyleSheet("""
            background-color: #252525; border-bottom: 1px solid #444;
        """)
        toolbar.setFixedHeight(30)
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(6, 1, 6, 1)
        layout.setSpacing(4)

        self.btn_clear = QPushButton("🗑 清空")
        self.btn_clear.setStyleSheet("""
            QPushButton { background-color: #3c3c3c; color: #EF5350; padding: 1px 8px;
                         border: 1px solid #555; border-radius: 3px; font-size: 13px; }
            QPushButton:hover { background-color: #4a2a2a; }
        """)

        layout.addWidget(self.btn_clear)
        layout.addStretch()

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #999; font-size: 13px;")

        self.node_count_label = QLabel("算子: 0")
        self.node_count_label.setStyleSheet("color: #999; font-size: 13px;")

        layout.addWidget(self.node_count_label)
        layout.addWidget(self.status_label)

        return toolbar

    def _connect_signals(self):
        self.btn_clear.clicked.connect(self._clear_pipeline)

        self.flow_canvas.pipeline_changed.connect(self._on_canvas_changed)
        self.flow_canvas.node_config_requested.connect(self._config_step)

    def _on_canvas_changed(self):
        slot_widget = self.flow_canvas.get_slot_widget()
        occupied = slot_widget.get_occupied_slots()
        count = len(occupied)
        self.node_count_label.setText(f"算子: {count}")
        self.status_label.setText(f"已更新 ({count} 个算子)")

        self._sync_to_pipeline()
        self.pipeline_changed.emit()

    def _sync_to_pipeline(self):
        slot_widget = self.flow_canvas.get_slot_widget()
        self._pipeline.steps.clear()

        for slot in slot_widget.get_occupied_slots():
            try:
                tool = create_tool(slot.tool_name)
                if tool:
                    # 合并参数：以保存的参数为主，缺失的键用工具的默认值补充
                    defaults = tool.params.copy()
                    defaults.update(slot.params)
                    tool.params = defaults
                    self._pipeline.add_step(tool, enabled=slot.is_enabled())
            except Exception:
                continue

    def _clear_pipeline(self):
        slot_widget = self.flow_canvas.get_slot_widget()
        if slot_widget.get_occupied_slots():
            reply = QMessageBox.question(self, "确认", "确定清空所有步骤？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.flow_canvas.clear_all()
                self.pipeline_changed.emit()

    def _config_step(self, index: int):
        slot_widget = self.flow_canvas.get_slot_widget()
        # 使用slot_index直接查找，而不是索引过滤后的occupied列表
        slot = slot_widget.get_slot_by_index(index)
        if slot is None or slot.is_empty():
            return

        context_info = self._get_context_info(index)

        tool = create_tool(slot.tool_name)
        if tool is None:
            return
        # 合并参数：以保存的参数为主，缺失的键用工具的默认值补充
        defaults = tool.params.copy()
        defaults.update(slot.params)
        tool.params = defaults

        # 注意: VisionTool.__init__ 会覆盖 display_name 为类名，所以用类名比较
        if tool.name == "MultiROI":
            preview_img = self._get_preview_image()
            if preview_img is not None:
                dialog = MultiROIEditorDialog(tool, preview_img, self)
                if dialog.exec_() == QDialog.Accepted:
                    slot.params = tool.params.copy()
                    self._sync_to_pipeline()
                    self.pipeline_changed.emit()
            else:
                QMessageBox.information(self, "提示",
                    "请先加载一张图片（点击「加载图片」按钮），"
                    "然后才能绘制ROI区域。\n\n"
                    "或者，您也可以在方案JSON文件中手动编辑regions参数。")
        else:
            preview_img = self._get_preview_image()
            dialog = ParamConfigDialog(tool, preview_img, context_info, self)
            if dialog.exec_() == QDialog.Accepted:
                slot.params = tool.params.copy()
                self._sync_to_pipeline()
                self.pipeline_changed.emit()

    def _get_context_info(self, current_step_index: int) -> Dict:
        regions = []
        regions_map = {}  # name -> (x, y, w, h)
        steps = []
        slot_widget = self.flow_canvas.get_slot_widget()
        # 遍历所有slots，使用slot_index进行比较
        for slot in slot_widget._slots:
            if not slot._is_occupied:
                continue
            if slot.slot_index == current_step_index:
                continue
            # slot.tool_name是类名(如"MultiROI")，需要用类名比较
            if slot.slot_index < current_step_index and slot.tool_name == "MultiROI":
                for r in slot.params.get("regions", []):
                    if r.get("enabled", True):
                        name = r.get("name", "")
                        regions.append(name)
                        regions_map[name] = (
                            r.get("x", 0),
                            r.get("y", 0),
                            r.get("width", r.get("w", 100)),
                            r.get("height", r.get("h", 100)),
                        )
            steps.append({
                "index": slot.slot_index,
                "name": slot.tool_name,
            })
        return {"regions": regions, "regions_map": regions_map, "steps": steps}

    def _get_preview_image(self):
        # 优先从MainWindow获取_raw_image
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, '_raw_image') and parent._raw_image is not None:
                return parent._raw_image.copy()
            parent = parent.parent()
        # 如果父链没找到，尝试从顶层窗口获取
        top = self.window()
        if hasattr(top, '_raw_image') and top._raw_image is not None:
            return top._raw_image.copy()
        return None

    def _remove_step(self, index: int):
        slot_widget = self.flow_canvas.get_slot_widget()
        slot = slot_widget.get_slot_by_index(index)
        if slot and slot._is_occupied:
            slot_widget._on_delete_operator(slot.slot_index)
            self.pipeline_changed.emit()

    def set_pipeline(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self.flow_canvas.from_pipeline(pipeline)

        slot_widget = self.flow_canvas.get_slot_widget()
        count = len(slot_widget.get_occupied_slots())
        self.node_count_label.setText(f"算子: {count}")

    def get_pipeline(self) -> Pipeline:
        self._sync_to_pipeline()
        return self._pipeline
