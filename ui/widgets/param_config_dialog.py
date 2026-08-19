# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from vision.tools.base_tool import VisionTool


class ParamConfigDialog(QDialog):
    GROUP_STYLE = """
        QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 3px;
                    margin-top: 8px; padding-top: 12px; padding-bottom: 2px; color: #d4d4d4; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #d4d4d4; }
    """
    SCROLL_STYLE = """
        QScrollArea { border: none; background: transparent; }
        QScrollBar:vertical { width: 6px; background: #2d2d2d; }
        QScrollBar::handle:vertical { background: #555; border-radius: 3px; min-height: 20px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """
    BTN_OK_STYLE = """
        QPushButton { background-color: #1976D2; color: #fff; font-size: 13px;
                     font-weight: bold; padding: 4px 14px; border: none; border-radius: 3px; }
        QPushButton:hover { background-color: #1565C0; }
    """
    BTN_CANCEL_STYLE = """
        QPushButton { background-color: #3c3c3c; color: #d4d4d4; padding: 4px 14px;
                     border: 1px solid #555; border-radius: 3px; }
        QPushButton:hover { background-color: #4a4a4a; }
    """

    def __init__(self, tool: VisionTool, preview_image: Optional[np.ndarray] = None,
                 context_info: Optional[Dict] = None, parent=None):
        super().__init__(parent)
        self.tool = tool
        self.preview_image = preview_image
        self.context_info = context_info or {"regions": []}
        self.setWindowTitle(f"参数配置 - {tool.display_name}")
        self.setMinimumWidth(800)
        self.setMinimumHeight(500)
        self.resize(960, 600)
        # 防抖定时器：参数修改后延迟 300ms 自动预览
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._update_preview)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        preview_group = QGroupBox("实时预览")
        preview_group.setStyleSheet(self.GROUP_STYLE)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(6, 10, 6, 6)
        preview_layout.setSpacing(4)

        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #444; border-radius: 2px;"
        )
        preview_layout.addWidget(self.preview_label, 1)

        preview_btn_row = QHBoxLayout()
        preview_btn_row.setContentsMargins(0, 0, 0, 0)
        preview_btn_row.addStretch()
        self.btn_preview = QPushButton("预览")
        self.btn_preview.setStyleSheet("""
            QPushButton { background-color: #1a3a5c; color: #4A90D9; padding: 5px 16px;
                         border: 1px solid #2a5a8c; border-radius: 2px; font-size: 16px; }
            QPushButton:hover { background-color: #2a4a7c; }
        """)
        self.btn_preview.clicked.connect(self._update_preview)
        preview_btn_row.addWidget(self.btn_preview)
        preview_layout.addLayout(preview_btn_row)

        left_layout.addWidget(preview_group, 1)

        right_widget = QWidget()
        right_widget.setMinimumWidth(340)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        if hasattr(self.tool, 'get_input_source_widgets'):
            source_group = QGroupBox("输入源")
            source_group.setStyleSheet(self.GROUP_STYLE)
            source_layout = QVBoxLayout(source_group)
            source_layout.setContentsMargins(8, 10, 8, 4)
            source_layout.setSpacing(4)
            source_widgets = self.tool.get_input_source_widgets(self, self.context_info)
            for label_text, widget in source_widgets:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(QLabel(label_text))
                row.addWidget(widget, 1)
                source_layout.addLayout(row)
                self._connect_auto_preview(widget)
            right_layout.addWidget(source_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet(self.SCROLL_STYLE)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(2)

        self._context_step_list = self.context_info.get("steps", [])
        # 将上游步骤列表设置到工具对象上，供 get_param_widgets 中使用
        self.tool._context_step_list = self._context_step_list

        widgets = self.tool.get_param_widgets(self)
        if widgets:
            current_group = None
            current_group_layout = None

            def finish_group():
                nonlocal current_group, current_group_layout
                if current_group is not None:
                    current_group.setLayout(current_group_layout)
                    scroll_layout.addWidget(current_group)
                    current_group = None
                    current_group_layout = None

            for label_item, widget in widgets:
                if isinstance(label_item, QLabel) and "──" in label_item.text():
                    finish_group()
                    scroll_layout.addWidget(label_item)
                    continue

                if isinstance(label_item, QCheckBox):
                    finish_group()
                    row = QHBoxLayout()
                    row.setContentsMargins(4, 0, 4, 0)
                    row.addWidget(label_item)
                    scroll_layout.addLayout(row)
                    continue

                if isinstance(label_item, QWidget) and not isinstance(label_item, QLabel):
                    finish_group()
                    row = QHBoxLayout()
                    row.setContentsMargins(4, 0, 4, 0)
                    row.addWidget(label_item)
                    scroll_layout.addLayout(row)
                    continue

                label_text = label_item.text() if isinstance(label_item, QLabel) else str(label_item)
                if current_group is None:
                    current_group = QGroupBox("参数设置")
                    current_group.setStyleSheet(self.GROUP_STYLE)
                    current_group_layout = QVBoxLayout(current_group)
                    current_group_layout.setContentsMargins(8, 10, 8, 4)
                    current_group_layout.setSpacing(4)

                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                lbl = QLabel(label_text)
                lbl.setMinimumWidth(80)
                row.addWidget(lbl)
                row.addWidget(widget, 1)
                current_group_layout.addLayout(row)

                # 为参数控件连接信号，实现修改后自动预览
                self._connect_auto_preview(widget)

            finish_group()
        else:
            scroll_layout.addWidget(QLabel("该工具无需额外参数"))

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area, 1)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 4, 0, 0)
        btn_layout.addStretch()

        self.btn_ok = QPushButton("确定")
        self.btn_ok.setStyleSheet(self.BTN_OK_STYLE)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setStyleSheet(self.BTN_CANCEL_STYLE)

        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(btn_layout)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget)

        if self.preview_image is not None:
            self._update_preview()

    def _connect_auto_preview(self, widget):
        """为参数控件连接信号，参数修改后自动触发防抖预览"""
        from PyQt5.QtWidgets import (QSpinBox, QDoubleSpinBox, QComboBox,
                                     QCheckBox, QLineEdit, QSlider)
        # QSpinBox / QDoubleSpinBox
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._on_param_changed)
        # QComboBox
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_param_changed)
        # QCheckBox
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self._on_param_changed)
        # QLineEdit
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._on_param_changed)
        # QSlider
        elif isinstance(widget, QSlider):
            widget.valueChanged.connect(self._on_param_changed)
        # 其他带 valueChanged 的控件
        elif hasattr(widget, 'valueChanged'):
            widget.valueChanged.connect(self._on_param_changed)

    def _on_param_changed(self):
        """参数变更时，重启防抖定时器"""
        self._debounce_timer.start()

    def _update_preview(self):
        if self.preview_image is None:
            return
        try:
            from vision.tools.base_tool import PipelineContext
            context = PipelineContext(
                original_image=self.preview_image,
                current_image=self.preview_image
            )

            source = self.tool.params.get("_input_source", "current")
            if source.startswith("region:"):
                region_name = source[7:]
                regions_map = self.context_info.get("regions_map", {})
                if region_name in regions_map:
                    context.regions[region_name] = regions_map[region_name]

            result = self.tool.process(context)
            if result.processed_image is not None:
                self._show_cv_image(result.processed_image)
            else:
                self._show_cv_image(self.preview_image)
        except Exception as e:
            self.preview_label.setText(f"预览错误: {e}")

    def _show_cv_image(self, cv_img):
        if cv_img is None:
            return
        try:
            if len(cv_img.shape) == 2:
                h, w = cv_img.shape
                q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
            else:
                h, w, ch = cv_img.shape
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
        except Exception as e:
            self.preview_label.setText(f"显示错误: {e}")


class MultiROIEditorDialog(QDialog):
    def __init__(self, tool: VisionTool, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.tool = tool
        self.original_image = image.copy()

        self.regions: List[Dict] = []
        self._load_regions()

        self._selected_idx = 0

        self.setWindowTitle("多区域ROI绘制 - 鼠标拖拽选择区域")
        self.setMinimumSize(900, 650)
        self._setup_ui()

    def _load_regions(self):
        for r in self.tool.params.get("regions", []):
            self.regions.append({
                "name": r.get("name", "未命名"),
                "x": r.get("x", 0),
                "y": r.get("y", 0),
                "w": r.get("width", 200),
                "h": r.get("height", 200),
                "enabled": r.get("enabled", True)
            })
        if not self.regions:
            h, w = self.original_image.shape[:2]
            self.regions.append({
                "name": "区域1", "x": w // 4, "y": h // 4,
                "w": w // 2, "h": h // 2, "enabled": True
            })
        self._selected_idx = 0

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()

        self.image_label = MultiROIEditorLabel(self)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #0d0d0d; border: 1px solid #444; border-radius: 3px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.regions = self.regions
        self.image_label.selected_idx = self._selected_idx
        self.image_label.set_base_image(self.original_image)

        left_layout.addWidget(self.image_label, 1)

        tip = QLabel("提示：在图像上拖拽绘制新区域，拖拽边框/角点调整大小，点击区域选中，右键取消")
        tip.setStyleSheet("color: #999; font-size: 17px; padding: 4px;")
        left_layout.addWidget(tip)

        main_layout.addLayout(left_layout, 2)

        right_widget = QWidget()
        right_widget.setMaximumWidth(280)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        list_group = QGroupBox("区域列表")
        list_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #444;
                        border-radius: 4px; margin-top: 8px; padding-top: 12px; color: #d4d4d4; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #d4d4d4; }
        """)
        list_layout = QVBoxLayout(list_group)

        self.region_list = QListWidget()
        self.region_list.setStyleSheet("""
            QListWidget { background-color: #2d2d2d; border: 1px solid #444;
                         border-radius: 3px; color: #d4d4d4; }
            QListWidget::item:selected { background-color: #1a3a5c; color: #4A90D9; }
        """)
        self._refresh_list()
        self.region_list.currentRowChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.region_list)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ 添加区域")
        self.btn_add.setStyleSheet("""
            QPushButton { background-color: #1a3a5c; color: #4A90D9; padding: 4px 10px;
                         border: 1px solid #2a5a8c; border-radius: 3px; }
            QPushButton:hover { background-color: #2a4a7c; }
        """)
        self.btn_remove = QPushButton("- 删除")
        self.btn_remove.setStyleSheet("""
            QPushButton { background-color: #3c3c3c; color: #EF5350; padding: 4px 10px;
                         border: 1px solid #555; border-radius: 3px; }
            QPushButton:hover { background-color: #4a2a2a; }
        """)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        list_layout.addLayout(btn_row)

        right_layout.addWidget(list_group)

        prop_group = QGroupBox("区域属性")
        prop_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #444;
                        border-radius: 4px; margin-top: 8px; padding-top: 12px; color: #d4d4d4; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #d4d4d4; }
        """)
        prop_layout = QVBoxLayout(prop_group)

        name_row = QHBoxLayout()
        lbl_name = QLabel("名称:")
        lbl_name.setStyleSheet("color: #d4d4d4;")
        name_row.addWidget(lbl_name)
        self.name_edit = QLineEdit()
        self.name_edit.setStyleSheet("background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555; padding: 2px;")
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_row.addWidget(self.name_edit)
        prop_layout.addLayout(name_row)

        self.cb_enable = QCheckBox("启用")
        self.cb_enable.setStyleSheet("color: #d4d4d4;")
        self.cb_enable.toggled.connect(self._on_enable_toggled)
        prop_layout.addWidget(self.cb_enable)

        for name, key, min_v, max_v in [
            ("X", "x", 0, 5000), ("Y", "y", 0, 5000),
            ("宽", "w", 1, 5000), ("高", "h", 1, 5000)
        ]:
            row = QHBoxLayout()
            lbl = QLabel(name + ":")
            lbl.setStyleSheet("color: #d4d4d4;")
            row.addWidget(lbl)
            sp = QSpinBox()
            sp.setStyleSheet("background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555;")
            sp.setRange(min_v, max_v)
            sp.valueChanged.connect(lambda v, k=key: self._on_spinbox_changed(k, v))
            setattr(self, f"sp_{key}", sp)
            row.addWidget(sp)
            prop_layout.addLayout(row)

        prop_layout.addStretch()
        right_layout.addWidget(prop_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_ok = QPushButton("确定")
        self.btn_ok.setStyleSheet("""
            QPushButton { background-color: #1976D2; color: #fff; font-size: 18px;
                         font-weight: bold; padding: 10px 28px; border: none;
                         border-radius: 3px; }
            QPushButton:hover { background-color: #1565C0; }
        """)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setStyleSheet("""
            QPushButton { background-color: #3c3c3c; color: #d4d4d4; padding: 10px 22px;
                         border: 1px solid #555; border-radius: 3px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)

        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(btn_layout)

        main_layout.addWidget(right_widget)

        self.btn_add.clicked.connect(self._add_region)
        self.btn_remove.clicked.connect(self._remove_region)
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)

        self._update_property_panel()
        self._update_display()

    def _refresh_list(self):
        self.region_list.blockSignals(True)
        self.region_list.clear()
        for i, r in enumerate(self.regions):
            status = "✓" if r.get("enabled", True) else "✗"
            self.region_list.addItem(f"{status} {r.get('name', '未命名')} ({r.get('w', 0)}x{r.get('h', 0)})")
        if 0 <= self._selected_idx < len(self.regions):
            self.region_list.setCurrentRow(self._selected_idx)
        self.region_list.blockSignals(False)

    def _on_selection_changed(self, idx):
        self._selected_idx = idx
        self.image_label.selected_idx = idx
        self._update_property_panel()
        self._update_display()

    def _update_property_panel(self):
        idx = self._selected_idx
        if 0 <= idx < len(self.regions):
            r = self.regions[idx]
            self.name_edit.blockSignals(True)
            self.name_edit.setText(r.get("name", ""))
            self.name_edit.blockSignals(False)

            self.cb_enable.blockSignals(True)
            self.cb_enable.setChecked(r.get("enabled", True))
            self.cb_enable.blockSignals(False)

            for key in ["x", "y", "w", "h"]:
                sp = getattr(self, f"sp_{key}", None)
                if sp:
                    sp.blockSignals(True)
                    sp.setValue(r.get(key, 0))
                    sp.blockSignals(False)

            self.name_edit.setEnabled(True)
            self.cb_enable.setEnabled(True)
            for key in ["x", "y", "w", "h"]:
                sp = getattr(self, f"sp_{key}", None)
                if sp:
                    sp.setEnabled(True)
        else:
            self.name_edit.setEnabled(False)
            self.cb_enable.setEnabled(False)
            for key in ["x", "y", "w", "h"]:
                sp = getattr(self, f"sp_{key}", None)
                if sp:
                    sp.setEnabled(False)

    def _on_name_changed(self, text):
        if 0 <= self._selected_idx < len(self.regions):
            self.regions[self._selected_idx]["name"] = text
            self._refresh_list()
            self._update_display()

    def _on_enable_toggled(self, checked):
        if 0 <= self._selected_idx < len(self.regions):
            self.regions[self._selected_idx]["enabled"] = checked
            self._refresh_list()
            self._update_display()

    def _on_spinbox_changed(self, key, value):
        if 0 <= self._selected_idx < len(self.regions):
            self.regions[self._selected_idx][key] = value
            self._refresh_list()
            self._update_display()

    def _add_region(self):
        h, w = self.original_image.shape[:2]
        new_idx = len(self.regions) + 1
        self.regions.append({
            "name": f"区域{new_idx}",
            "x": w // 4,
            "y": h // 4,
            "w": w // 2,
            "h": h // 2,
            "enabled": True
        })
        self._selected_idx = len(self.regions) - 1
        self.image_label.selected_idx = self._selected_idx
        self._refresh_list()
        self._update_property_panel()
        self._update_display()

    def _remove_region(self):
        if 0 <= self._selected_idx < len(self.regions) and len(self.regions) > 1:
            del self.regions[self._selected_idx]
            self._selected_idx = min(self._selected_idx, len(self.regions) - 1)
            self.image_label.selected_idx = self._selected_idx
            self._refresh_list()
            self._update_property_panel()
            self._update_display()

    def _update_display(self):
        self.image_label.update()

    def _on_ok(self):
        regions_data = []
        for r in self.regions:
            regions_data.append({
                "name": r.get("name", "未命名"),
                "x": r.get("x", 0),
                "y": r.get("y", 0),
                "width": r.get("w", 100),
                "height": r.get("h", 100),
                "enabled": r.get("enabled", True)
            })
        self.tool.params["regions"] = regions_data
        self.accept()


class MultiROIEditorLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.regions: List[Dict] = []
        self.selected_idx = -1
        self._drawing = False
        self._draw_start_x = 0
        self._draw_start_y = 0
        self._resizing = False
        self._resize_handle = -1
        self._handle_size = 8
        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self._moving = False
        self._image_w = 0
        self._image_h = 0
        self._temp_region_idx = -1
        self._base_pixmap = None
        self.setMouseTracking(True)

    def set_image_size(self, w, h):
        self._image_w = w
        self._image_h = h

    def set_base_image(self, cv_img):
        if cv_img is None:
            return
        try:
            if len(cv_img.shape) == 2:
                h, w = cv_img.shape
                q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
            else:
                h, w, ch = cv_img.shape
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
            self._base_pixmap = QPixmap.fromImage(q_img)
            self._image_w = w
            self._image_h = h
            self.update()
        except Exception as e:
            self.setText(f"图像加载错误: {e}")

    def _get_scaled_rect(self):
        if self._base_pixmap is None:
            return QRect(0, 0, self.width(), self.height()), 1.0, 1.0
        pix_w = self._base_pixmap.width()
        pix_h = self._base_pixmap.height()
        label_w = self.width()
        label_h = self.height()
        if label_w <= 0 or label_h <= 0:
            return QRect(0, 0, 0, 0), 1.0, 1.0
        scale = min(label_w / pix_w, label_h / pix_h)
        scaled_w = int(pix_w * scale)
        scaled_h = int(pix_h * scale)
        x = (label_w - scaled_w) // 2
        y = (label_h - scaled_h) // 2
        scale_x = self._image_w / scaled_w if scaled_w > 0 else 1
        scale_y = self._image_h / scaled_h if scaled_h > 0 else 1
        return QRect(x, y, scaled_w, scaled_h), scale_x, scale_y

    def _image_to_label(self, img_x, img_y):
        rect, sx, sy = self._get_scaled_rect()
        label_x = rect.x() + img_x / sx if sx > 0 else rect.x()
        label_y = rect.y() + img_y / sy if sy > 0 else rect.y()
        return int(label_x), int(label_y)

    def _label_to_image(self, label_x, label_y):
        rect, sx, sy = self._get_scaled_rect()
        img_x = (label_x - rect.x()) * sx
        img_y = (label_y - rect.y()) * sy
        img_x = max(0, min(img_x, self._image_w - 1)) if self._image_w > 0 else img_x
        img_y = max(0, min(img_y, self._image_h - 1)) if self._image_h > 0 else img_y
        return int(img_x), int(img_y)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self._base_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect, sx, sy = self._get_scaled_rect()
        painter.drawPixmap(rect, self._base_pixmap, self._base_pixmap.rect())

        colors = [
            QColor(0, 255, 0), QColor(255, 255, 0), QColor(0, 255, 255),
            QColor(255, 0, 255), QColor(128, 255, 0), QColor(255, 128, 0),
            QColor(0, 128, 255), QColor(128, 0, 255)
        ]

        for i, r in enumerate(self.regions):
            if not r.get("enabled", True):
                continue
            color = colors[i % len(colors)]
            rx, ry, rw, rh = r.get("x", 0), r.get("y", 0), r.get("w", 100), r.get("h", 100)

            lx, ly = self._image_to_label(rx, ry)
            lx2, ly2 = self._image_to_label(rx + rw, ry + rh)
            lw = lx2 - lx
            lh = ly2 - ly

            pen_width = 3 if i == self.selected_idx else 2
            painter.setPen(QPen(color, pen_width))

            if i == self.selected_idx:
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 50))
            else:
                painter.setBrush(Qt.NoBrush)

            painter.drawRect(lx, ly, lw, lh)

            painter.setPen(QPen(color, 1))
            font = QFont("Arial", 12)
            painter.setFont(font)
            painter.drawText(lx, ly - 3, r.get("name", ""))

            if i == self.selected_idx:
                painter.setBrush(QColor(0, 255, 255))
                painter.setPen(Qt.NoPen)
                hs = self._handle_size
                for pt_x, pt_y in [(lx, ly), (lx + lw, ly), (lx + lw, ly + lh), (lx, ly + lh)]:
                    painter.drawRect(pt_x - hs // 2, pt_y - hs // 2, hs, hs)

        painter.end()

    def _get_handle_at(self, x, y, region):
        rx, ry, rw, rh = region.get("x", 0), region.get("y", 0), region.get("w", 100), region.get("h", 100)
        hs = self._handle_size
        handles = [
            (rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh),
            (rx + rw // 2, ry), (rx + rw, ry + rh // 2),
            (rx + rw // 2, ry + rh), (rx, ry + rh // 2)
        ]
        for idx, (hx, hy) in enumerate(handles):
            hl_x, hl_y = self._image_to_label(hx, hy)
            if abs(x - hl_x) <= hs and abs(y - hl_y) <= hs:
                return idx
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            for i, r in enumerate(reversed(self.regions)):
                idx = len(self.regions) - 1 - i
                handle = self._get_handle_at(x, y, r)
                if handle >= 0:
                    self._resizing = True
                    self._resize_handle = handle
                    self.selected_idx = idx
                    self._drag_offset_x = 0
                    self._drag_offset_y = 0
                    self.update()
                    return

            for i, r in enumerate(reversed(self.regions)):
                idx = len(self.regions) - 1 - i
                rx, ry = self._image_to_label(r.get("x", 0), r.get("y", 0))
                rx2, ry2 = self._image_to_label(r.get("x", 0) + r.get("w", 100), r.get("y", 0) + r.get("h", 100))
                if rx <= x <= rx2 and ry <= y <= ry2:
                    self._moving = True
                    self.selected_idx = idx
                    self._drag_offset_x = x - rx
                    self._drag_offset_y = y - ry
                    self.update()
                    return

            self._drawing = True
            img_x, img_y = self._label_to_image(x, y)
            self._draw_start_x = img_x
            self._draw_start_y = img_y
            new_idx = len(self.regions) + 1
            self.regions.append({
                "name": f"区域{new_idx}",
                "x": img_x, "y": img_y, "w": 1, "h": 1, "enabled": True
            })
            self._temp_region_idx = len(self.regions) - 1
            self.selected_idx = self._temp_region_idx
            self.update()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        x, y = event.x(), event.y()
        if self._drawing and self._temp_region_idx >= 0:
            img_x, img_y = self._label_to_image(x, y)
            r = self.regions[self._temp_region_idx]
            new_x = min(self._draw_start_x, img_x)
            new_y = min(self._draw_start_y, img_y)
            new_w = abs(img_x - self._draw_start_x)
            new_h = abs(img_y - self._draw_start_y)
            r["x"] = max(0, new_x)
            r["y"] = max(0, new_y)
            r["w"] = max(1, min(new_w, self._image_w - r["x"]))
            r["h"] = max(1, min(new_h, self._image_h - r["y"]))
            self.update()
            return
        if self._moving and self.selected_idx >= 0:
            r = self.regions[self.selected_idx]
            img_x, img_y = self._label_to_image(x - self._drag_offset_x, y - self._drag_offset_y)
            r["x"] = max(0, min(img_x, self._image_w - r["w"]))
            r["y"] = max(0, min(img_y, self._image_h - r["h"]))
            self.update()
            return
        if self._resizing and self.selected_idx >= 0:
            r = self.regions[self.selected_idx]
            img_x, img_y = self._label_to_image(x, y)
            h = self._resize_handle
            if h in (0, 4, 7):
                r["w"] = max(1, r["x"] + r["w"] - img_x)
                r["x"] = max(0, min(img_x, r["x"] + r["w"] - 1))
            if h in (1, 4, 5):
                r["w"] = max(1, img_x - r["x"])
            if h in (2, 5, 6):
                r["h"] = max(1, img_y - r["y"])
            if h in (0, 3, 7):
                r["h"] = max(1, r["y"] + r["h"] - img_y)
                r["y"] = max(0, min(img_y, r["y"] + r["h"] - 1))
            if h in (1, 2):
                r["h"] = max(1, img_y - r["y"])
            if h in (3, 6):
                r["w"] = max(1, r["x"] + r["w"] - img_x)
                r["x"] = max(0, min(img_x, r["x"] + r["w"] - 1))
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._drawing and self._temp_region_idx >= 0:
                r = self.regions[self._temp_region_idx]
                if r["w"] < 5 or r["h"] < 5:
                    del self.regions[self._temp_region_idx]
                    self.selected_idx = min(self._temp_region_idx, len(self.regions) - 1)
                else:
                    parent = self.parent()
                    if parent and hasattr(parent, '_refresh_list'):
                        parent._refresh_list()
                    if parent and hasattr(parent, '_update_property_panel'):
                        parent._update_property_panel()
                self._temp_region_idx = -1
            self._drawing = False
            self._moving = False
            self._resizing = False
            self.update()
        super().mouseReleaseEvent(event)
