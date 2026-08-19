# -*- coding: utf-8 -*-
import json
from typing import Optional, Dict, List, Tuple
from PyQt5.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QApplication, QMenu,
                             QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, QPoint, QRect, QByteArray, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import (QDrag, QPixmap, QPainter, QColor, QBrush, QPen,
                         QFont, QMouseEvent, QDragEnterEvent, QDragMoveEvent,
                         QDropEvent, QPainterPath, QCursor)

from ..constants import CATEGORY_COLORS, CATEGORY_ICONS, CATEGORY_LIGHT_COLORS

CATEGORY_COLORS_Q = {k: QColor(v) for k, v in CATEGORY_COLORS.items()}
CATEGORY_LIGHT_Q = {k: QColor(v) for k, v in CATEGORY_LIGHT_COLORS.items()}


class StepSlot(QFrame):
    operator_dropped = pyqtSignal(int, str, str, dict)
    delete_requested = pyqtSignal(int)
    selected = pyqtSignal(int)

    def __init__(self, slot_index: int, parent=None):
        super().__init__(parent)
        self._slot_index = slot_index
        self._tool_name = ""
        self._category = ""
        self._params: Dict = {}
        self._enabled = True
        self._is_occupied = False
        self._drag_start_pos = None
        self._is_hovered = False
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._setup_ui()
        self._update_empty_style()
        self._setup_shadow()

    def _setup_ui(self):
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(6)

        # 序号标签
        self._index_label = QLabel(str(self._slot_index + 1))
        self._index_label.setFixedSize(22, 22)
        self._index_label.setAlignment(Qt.AlignCenter)
        self._index_label.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                color: #999;
                border-radius: 11px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self._layout.addWidget(self._index_label)

        # 图标标签
        self._icon_label = QLabel()
        self._icon_label.setFixedSize(24, 24)
        self._icon_label.setAlignment(Qt.AlignCenter)
        self._icon_label.setStyleSheet("font-size: 16px;")
        self._layout.addWidget(self._icon_label)

        # 名称标签
        self._name_label = QLabel("拖入算子")
        self._name_label.setStyleSheet("color: #888; font-size: 12px;")
        self._layout.addWidget(self._name_label, 1)

        # 启用按钮
        self._enable_btn = QPushButton()
        self._enable_btn.setFixedSize(24, 24)
        self._enable_btn.setCheckable(True)
        self._enable_btn.setChecked(True)
        self._enable_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._enable_btn.clicked.connect(self._on_enable_clicked)
        self._enable_btn.setVisible(False)
        self._layout.addWidget(self._enable_btn)

        # 删除按钮
        self._delete_btn = QPushButton()
        self._delete_btn.setFixedSize(24, 24)
        self._delete_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._delete_btn.setStyleSheet("""
            QPushButton {
                background: #4a2020;
                border: 1px solid #ff5252;
                color: #ff5252;
                font-size: 14px;
                font-weight: bold;
                border-radius: 12px;
            }
            QPushButton:hover {
                background: #5a2a2a;
                color: #ff7070;
                border-color: #ff7070;
            }
            QPushButton:pressed {
                background: #6a3a3a;
            }
        """)
        self._delete_btn.setText("✕")
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._slot_index))
        self._delete_btn.setVisible(False)
        self._layout.addWidget(self._delete_btn)

    def _setup_shadow(self):
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(8)
        self._shadow.setOffset(0, 2)
        self._shadow.setColor(QColor(0, 0, 0, 30))
        self.setGraphicsEffect(self._shadow)

    def _update_empty_style(self):
        self.setStyleSheet("""
            StepSlot {
                background: #2d2d2d;
                border: 1px dashed #555;
                border-radius: 6px;
            }
            StepSlot:hover {
                border-color: #4A90D9;
                background: #1a2a3a;
            }
        """)

    def _update_occupied_style(self):
        color = CATEGORY_COLORS_Q.get(self._category, QColor("#333"))
        border_color = color.name()
        
        self.setStyleSheet(f"""
            StepSlot {{
                background: #2d2d2d;
                border: 1px solid {border_color};
                border-radius: 6px;
            }}
            StepSlot:hover {{
                background: #353535;
                border-color: {QColor(color).lighter(120).name()};
            }}
        """)

    def _get_display_name(self, tool_class_name: str) -> str:
        from vision.pipeline import ALL_TOOLS
        cls = ALL_TOOLS.get(tool_class_name)
        if cls and hasattr(cls, 'display_name'):
            return cls.display_name
        return tool_class_name

    def set_operator(self, tool_name: str, category: str, params: Dict, enabled: bool = True):
        self._tool_name = tool_name
        self._category = category
        self._params = params
        self._enabled = enabled
        self._is_occupied = True

        display = self._get_display_name(tool_name)
        color = CATEGORY_COLORS_Q.get(category, QColor("#333"))
        
        # 更新名称标签
        self._name_label.setText(display)
        self._name_label.setStyleSheet(f"""
            font-weight: 600;
            font-size: 13px;
            color: {color.name()};
        """)

        # 更新图标标签
        self._icon_label.setText(CATEGORY_ICONS.get(category, ""))
        self._icon_label.setStyleSheet(f"font-size: 18px; color: {color.name()};")

        # 更新序号标签颜色
        self._index_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color.name()};
                color: white;
                border-radius: 11px;
                font-weight: bold;
                font-size: 12px;
            }}
        """)

        self._enable_btn.setVisible(True)
        self._enable_btn.setChecked(enabled)
        self._update_enable_btn_style()
        self._delete_btn.setVisible(True)
        self._update_occupied_style()

    def clear_operator(self):
        self._tool_name = ""
        self._category = ""
        self._params = {}
        self._enabled = True
        self._is_occupied = False
        self._name_label.setText("拖入算子")
        self._name_label.setStyleSheet("color: #888; font-size: 12px;")
        self._icon_label.setText("")
        self._icon_label.setStyleSheet("")
        self._index_label.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                color: #999;
                border-radius: 11px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self._enable_btn.setVisible(False)
        self._delete_btn.setVisible(False)
        self._update_empty_style()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self._enable_btn.setChecked(enabled)
        self._update_enable_btn_style()

    def _update_enable_btn_style(self):
        if self._enable_btn.isChecked():
            self._enable_btn.setText("✓")
            self._enable_btn.setStyleSheet("""
                QPushButton {
                    background: #4CAF50;
                    color: white;
                    border: 1px solid #66BB6A;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #43A047;
                }
                QPushButton:pressed {
                    background: #388E3C;
                }
            """)
        else:
            self._enable_btn.setText("○")
            self._enable_btn.setStyleSheet("""
                QPushButton {
                    background: #555;
                    color: #ccc;
                    border: 1px solid #777;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #666;
                    color: #fff;
                }
                QPushButton:pressed {
                    background: #777;
                }
            """)

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def category(self) -> str:
        return self._category

    @property
    def params(self) -> Dict:
        return self._params

    @params.setter
    def params(self, value: Dict):
        self._params = value

    @property
    def slot_index(self) -> int:
        return self._slot_index

    def is_enabled(self) -> bool:
        return self._enabled

    def is_empty(self) -> bool:
        return not self._is_occupied

    def _on_enable_clicked(self, checked: bool):
        self._enabled = checked
        self._update_enable_btn_style()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._is_occupied:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start_pos and (event.pos() - self._drag_start_pos).manhattanLength() > 10:
            self._start_drag()
            self._drag_start_pos = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_start_pos = None
        if event.button() == Qt.LeftButton:
            self.selected.emit(self._slot_index)
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        data = json.dumps({
            "tool_name": self._tool_name,
            "category": self._category,
            "params": self._params,
            "enabled": self._enabled,
            "from_slot": self._slot_index
        })
        mime.setData("application/x-operator", data.encode("utf-8"))
        drag.setMimeData(mime)

        display = self._get_display_name(self._tool_name)
        icon = CATEGORY_ICONS.get(self._category, "")
        
        # 创建更美观的拖拽预览
        pixmap = QPixmap(200, 44)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制圆角矩形背景
        color = CATEGORY_COLORS_Q.get(self._category, QColor("#333"))
        bg_color = QColor(color)
        bg_color.setAlpha(230)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 200, 44, 10, 10)
        
        # 绘制序号
        painter.setPen(Qt.white)
        index_font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(index_font)
        painter.drawText(12, 27, str(self._slot_index + 1))
        
        # 绘制图标
        icon_font = QFont("Segoe UI Emoji", 16)
        painter.setFont(icon_font)
        painter.drawText(38, 28, icon)
        
        # 绘制文本
        text_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        painter.setFont(text_font)
        painter.drawText(68, 28, display)
        
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(100, 22))
        drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasFormat("application/x-operator"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasFormat("application/x-operator"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasFormat("application/x-operator"):
            data = json.loads(bytes(event.mimeData().data("application/x-operator")).decode("utf-8"))
            tool_name = data.get("tool_name", "")
            category = data.get("category", "")
            params = data.get("params", {})
            enabled = data.get("enabled", True)
            from_slot = data.get("from_slot", -1)
            if from_slot >= 0 and from_slot != self._slot_index:
                self.operator_dropped.emit(from_slot, tool_name, category, params)
            elif from_slot < 0:
                self.operator_dropped.emit(self._slot_index, tool_name, category, params)
            event.acceptProposedAction()
        else:
            event.ignore()


class StepSlotWidget(QWidget):
    pipeline_changed = pyqtSignal()
    node_config_requested = pyqtSignal(int)
    slot_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slots: List[StepSlot] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 标题区域
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        header_icon = QLabel("📋")
        header_icon.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(header_icon)
        
        header = QLabel("步骤列表")
        header.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #d4d4d4;
            padding: 0;
        """)
        header_layout.addWidget(header)
        header_layout.addStretch()

        # 添加步骤按钮
        self._btn_add_step = QPushButton("+ 添加步骤")
        self._btn_add_step.setFixedHeight(26)
        self._btn_add_step.setCursor(QCursor(Qt.PointingHandCursor))
        self._btn_add_step.setStyleSheet("""
            QPushButton {
                background: #1a3a5c;
                color: #4A90D9;
                border: 1px solid #2a5a8c;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
                padding: 1px 8px;
            }
            QPushButton:hover {
                background: #2a4a7c;
                color: #5AA0E9;
            }
            QPushButton:pressed {
                background: #0a2a4c;
            }
        """)
        self._btn_add_step.clicked.connect(self._on_add_step)
        header_layout.addWidget(self._btn_add_step)

        layout.addLayout(header_layout)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self._container = QWidget()
        self._container.setStyleSheet("background-color: transparent;")
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(2, 2, 2, 2)
        self._container_layout.setSpacing(4)
        self._container_layout.addStretch()

        scroll.setWidget(self._container)
        layout.addWidget(scroll)

        self._add_slot()

    def _add_slot(self):
        index = len(self._slots)
        slot = StepSlot(index)
        slot.operator_dropped.connect(self._on_operator_dropped)
        slot.delete_requested.connect(self._on_delete_operator)
        slot.selected.connect(self._on_slot_selected)
        self._slots.append(slot)
        self._container_layout.insertWidget(self._container_layout.count() - 1, slot)

    def _remove_last_empty_slots(self):
        while len(self._slots) > 1:
            last = self._slots[-1]
            if not last._is_occupied:
                self._container_layout.removeWidget(last)
                last.deleteLater()
                self._slots.pop()
            else:
                break

    def _on_add_step(self):
        """点击「添加步骤」按钮，在末尾追加一个空 slot"""
        # 先确保末尾已有一个空 slot
        if self._slots and not self._slots[-1]._is_occupied:
            # 末尾已有空 slot，不需要操作
            pass
        else:
            self._add_slot()
        self.pipeline_changed.emit()

    def _on_operator_dropped(self, slot_index: int, tool_name: str, category: str,
                             params: dict):
        if slot_index < len(self._slots):
            self._slots[slot_index].set_operator(tool_name, category, params)
        else:
            self._add_slot()
            self._slots[-1].set_operator(tool_name, category, params)

        # 检查是否所有 slot 都被占用，如果是则追加一个新空 slot
        all_occupied = all(s._is_occupied for s in self._slots)
        if all_occupied:
            self._add_slot()

        self._remove_last_empty_slots()
        self.pipeline_changed.emit()

    def _on_slot_selected(self, slot_index: int):
        self.node_config_requested.emit(slot_index)
        self.slot_selected.emit(slot_index)

    def _on_delete_operator(self, slot_index: int):
        if slot_index < len(self._slots):
            self._slots[slot_index].clear_operator()
            self._remove_last_empty_slots()
            self.pipeline_changed.emit()

    def _update_counts(self):
        pass

    def get_slot_by_index(self, index: int) -> Optional[StepSlot]:
        if 0 <= index < len(self._slots):
            return self._slots[index]
        return None

    def get_occupied_slots(self) -> List[StepSlot]:
        return [s for s in self._slots if s._is_occupied]

    def clear_all(self):
        for slot in self._slots:
            slot.clear_operator()
        self._remove_last_empty_slots()

    def to_pipeline(self):
        from vision.pipeline import Pipeline, create_tool
        pipeline = Pipeline()
        for slot in self._slots:
            if slot._is_occupied:
                tool = create_tool(slot._tool_name, slot._params)
                if tool:
                    pipeline.add_step(tool, slot._enabled)
        return pipeline

    def from_pipeline(self, pipeline):
        from vision.pipeline import get_tool_category
        self.clear_all()
        for i, step in enumerate(pipeline.steps):
            if i >= len(self._slots):
                self._add_slot()
            tool = step.tool
            tool_class_name = tool.__class__.__name__
            self._slots[i].set_operator(
                tool_class_name,
                get_tool_category(tool_class_name),
                tool.to_dict(),
                step.enabled
            )
        if self._slots and self._slots[-1]._is_occupied:
            self._add_slot()
        self._remove_last_empty_slots()
