# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                             QLineEdit, QLabel, QHBoxLayout, QApplication, QFrame)
from PyQt5.QtCore import Qt, QSize, QPoint, QMimeData, QEvent
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QColor, QFont, QIcon, QBrush, QCursor

from ..constants import CATEGORY_ICONS


class DraggableOperatorItem(QListWidgetItem):
    def __init__(self, tool_name: str, category: str):
        super().__init__(tool_name)
        self._tool_name = tool_name
        self._category = category
        self.setForeground(QColor("#d4d4d4"))
        self.setToolTip(f"{category} - {tool_name}")
        self.setSizeHint(QSize(0, 26))

    def tool_name(self) -> str:
        return self._tool_name

    def category(self) -> str:
        return self._category


class OperatorCategoryItem(QListWidgetItem):
    def __init__(self, category: str):
        icon = CATEGORY_ICONS.get(category, "")
        super().__init__(f"{icon}  {category}")
        self._category = category
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        self.setFont(font)
        self.setForeground(QColor("#e0e0e0"))
        self.setFlags(Qt.ItemIsEnabled)
        self.setSizeHint(QSize(0, 24))

    def category(self) -> str:
        return self._category


class ToolboxListWidget(QListWidget):
    def __init__(self, toolbox):
        super().__init__()
        self._toolbox = toolbox
        self._setup_style()

    def _setup_style(self):
        self.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 1px;
            }
            QListWidget::item {
                padding: 2px 6px;
                border-radius: 2px;
                margin: 0;
            }
            QListWidget::item:hover {
                background-color: #2a2d2e;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:selected:!active {
                background-color: #094771;
            }
        """)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if isinstance(item, DraggableOperatorItem):
            self._toolbox._start_drag(item)


class OperatorToolbox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tool_items: List[DraggableOperatorItem] = []
        self._category_rows: Dict[str, int] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title
        title = QLabel("算子工具箱")
        title.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: #cccccc;
            padding: 1px 0;
        """)
        layout.addWidget(title)

        # Search box
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("搜索算子...")
        self._search_box.setClearButtonEnabled(True)
        self._search_box.textChanged.connect(self._filter_tools)
        self._search_box.installEventFilter(self)
        self._search_box.setStyleSheet("""
            QLineEdit {
                padding: 3px 6px;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                background-color: #3c3c3c;
                font-size: 12px;
                color: #cccccc;
            }
            QLineEdit:focus {
                border: 1px solid #007acc;
                background-color: #2d2d2d;
            }
        """)
        layout.addWidget(self._search_box)

        # Tool list
        self._list = ToolboxListWidget(self)
        self._list.setDragEnabled(True)
        self._list.setDefaultDropAction(Qt.CopyAction)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setIconSize(QSize(16, 16))
        self._list.setSpacing(0)
        layout.addWidget(self._list)

        self._populate_tools()

    def _get_display_name(self, tool_class_name: str) -> str:
        from vision.pipeline import ALL_TOOLS
        cls = ALL_TOOLS.get(tool_class_name)
        if cls and hasattr(cls, 'display_name'):
            return cls.display_name
        return tool_class_name

    def _populate_tools(self):
        self._list.clear()
        self._tool_items.clear()
        self._category_rows.clear()

        from vision.pipeline import get_tools_by_category
        categorized = get_tools_by_category()

        row = 0
        for category, tools in categorized.items():
            cat_item = OperatorCategoryItem(category)
            self._list.addItem(cat_item)
            self._category_rows[category] = row
            row += 1

            for tool_name in tools:
                display = self._get_display_name(tool_name)
                item = DraggableOperatorItem(tool_name, category)
                item.setText(display)
                self._list.addItem(item)
                self._tool_items.append(item)
                row += 1

    def _filter_tools(self, text: str):
        self._list.clear()
        if not text.strip():
            self._populate_tools()
            return

        from vision.pipeline import get_tools_by_category
        categorized = get_tools_by_category()

        for category, tools in categorized.items():
            matched = []
            for t in tools:
                display = self._get_display_name(t)
                if text.lower() in display.lower() or text.lower() in t.lower():
                    matched.append(t)
            if matched:
                cat_item = OperatorCategoryItem(category)
                self._list.addItem(cat_item)
                for tool_name in matched:
                    display = self._get_display_name(tool_name)
                    item = DraggableOperatorItem(tool_name, category)
                    item.setText(display)
                    self._list.addItem(item)
                    self._tool_items.append(item)

    def eventFilter(self, obj, event):
        if obj is self._search_box and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Down:
                self._list.setFocus()
                if self._list.count() > 0:
                    self._list.setCurrentRow(0)
                return True
        return super().eventFilter(obj, event)

    def _start_drag(self, item: DraggableOperatorItem):
        drag = QDrag(self._list)
        mime = QMimeData()
        import json
        data = json.dumps({
            "tool_name": item.tool_name(),
            "category": item.category(),
            "params": {},
            "enabled": True,
            "from_slot": -1
        })
        mime.setData("application/x-operator", data.encode("utf-8"))
        drag.setMimeData(mime)

        display = self._get_display_name(item.tool_name())

        pixmap = QPixmap(160, 36)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Simple dark background for drag preview
        painter.setBrush(QBrush(QColor("#3c3c3c")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 160, 36, 4, 4)

        painter.setPen(QColor("#d4d4d4"))
        text_font = QFont("Microsoft YaHei", 10)
        painter.setFont(text_font)
        painter.drawText(12, 23, display)

        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(80, 18))

        result = drag.exec_(Qt.CopyAction)

    def get_all_tool_names(self) -> List[str]:
        return [item.tool_name() for item in self._tool_items]
