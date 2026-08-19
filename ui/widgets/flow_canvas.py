# -*- coding: utf-8 -*-
from typing import Optional, List, Dict, Any

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from vision.pipeline import Pipeline, PipelineStep, create_tool, get_tools_by_category
from vision.tools.base_tool import VisionTool

from .step_slot_widget import StepSlotWidget, StepSlot, CATEGORY_COLORS
from .param_config_dialog import ParamConfigDialog, MultiROIEditorDialog


class FlowCanvas(QWidget):

    pipeline_changed = pyqtSignal()
    node_config_requested = pyqtSignal(int)
    slot_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slot_widget = StepSlotWidget(self)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._slot_widget, 1)

        self._slot_widget.pipeline_changed.connect(self.pipeline_changed.emit)
        self._slot_widget.node_config_requested.connect(self.node_config_requested.emit)
        self._slot_widget.slot_selected.connect(self.slot_selected.emit)

    def add_node_from_tool(self, tool_name: str, category: str = "",
                           params: Optional[Dict] = None,
                           enabled: bool = True):
        # 先找已有空 slot
        for slot in self._slot_widget._slots:
            if slot.is_empty():
                slot.set_operator(tool_name, category, params or {}, enabled)
                self._slot_widget._update_counts()
                self.pipeline_changed.emit()
                return
        # 所有 slot 都被占用，追加新 slot
        self._slot_widget._add_slot()
        self._slot_widget._slots[-1].set_operator(tool_name, category, params or {}, enabled)
        self._slot_widget._update_counts()
        self.pipeline_changed.emit()

    def clear_all(self):
        self._slot_widget.clear_all()

    def to_pipeline(self) -> Pipeline:
        return self._slot_widget.to_pipeline()

    def from_pipeline(self, pipeline: Pipeline):
        self._slot_widget.from_pipeline(pipeline)

    def get_slot_widget(self) -> StepSlotWidget:
        return self._slot_widget

    def update_node_params(self, step_index: int, params: Dict):
        slots = self._slot_widget.get_occupied_slots()
        if 0 <= step_index < len(slots):
            slots[step_index].params = params
