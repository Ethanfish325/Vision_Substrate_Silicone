# -*- coding: utf-8 -*-
"""
可缩放图片显示控件
==================
支持鼠标滚轮缩放、拖拽平移、双击重置缩放。
"""

from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QPainter, QPen


class ZoomableLabel(QLabel):
    """
    支持鼠标滚轮缩放和拖拽平移的图片显示控件。

    功能:
        - 鼠标滚轮缩放 (Ctrl+滚轮 或 普通滚轮)
        - 鼠标拖拽平移 (缩放后)
        - 双击重置缩放
        - 右键菜单重置
        - 显示当前缩放比例
    """

    zoom_changed = pyqtSignal(float)  # 缩放比例变化信号

    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    ZOOM_STEP = 1.15  # 每次滚轮缩放倍率

    def __init__(self, text="", parent=None):
        super().__init__(parent, Qt.Widget)
        self._pixmap = None          # 原始 QPixmap
        self._zoom = 1.0             # 当前缩放比例
        self._offset = QPoint(0, 0)  # 平移偏移量
        self._drag_start = None      # 拖拽起始点
        self._fit_to_widget = True   # 是否自适应控件大小

        self.setAlignment(Qt.AlignCenter)
        self.setText(text)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)

        # 样式
        self.setStyleSheet("""
            ZoomableLabel {
                background-color: #0d0d0d;
                border: 1px solid #444;
                border-radius: 3px;
            }
        """)

    def set_pixmap(self, pixmap: QPixmap):
        """设置显示的图片（重置缩放状态）"""
        self._pixmap = pixmap
        self._fit_to_widget = True
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._update_display()

    def update_pixmap(self, pixmap: QPixmap):
        """更新图片内容，保持当前缩放和平移状态不变"""
        if pixmap is None:
            return
        self._pixmap = pixmap
        self._update_display()

    def set_image(self, qimage: QImage):
        """通过 QImage 设置图片"""
        if qimage is None:
            return
        self.set_pixmap(QPixmap.fromImage(qimage))

    def clear_pixmap(self):
        """清除图片"""
        self._pixmap = None
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._fit_to_widget = True
        super().clear()
        self.update()

    def get_zoom(self) -> float:
        return self._zoom

    def reset_zoom(self):
        """重置缩放"""
        if self._pixmap is None:
            return
        self._fit_to_widget = True
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._update_display()

    def zoom_in(self):
        """放大"""
        if self._pixmap is None:
            return
        self._fit_to_widget = False
        self._zoom = min(self._zoom * self.ZOOM_STEP, self.MAX_ZOOM)
        self._update_display()

    def zoom_out(self):
        """缩小"""
        if self._pixmap is None:
            return
        self._fit_to_widget = False
        self._zoom = max(self._zoom / self.ZOOM_STEP, self.MIN_ZOOM)
        self._update_display()

    def _update_display(self):
        """更新显示"""
        if self._pixmap is None:
            return

        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return

        if self._fit_to_widget:
            # 自适应：缩放图片以适应控件大小
            scaled = self._pixmap.scaled(
                widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
            self.zoom_changed.emit(1.0)
        else:
            # 自定义缩放
            base_size = self._pixmap.size()
            zoomed_size = base_size * self._zoom
            scaled = self._pixmap.scaled(
                zoomed_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 应用平移偏移
            pixmap = QPixmap(widget_size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            # 计算居中位置 + 偏移
            x = (widget_size.width() - scaled.width()) // 2 + self._offset.x()
            y = (widget_size.height() - scaled.height()) // 2 + self._offset.y()
            painter.drawPixmap(x, y, scaled)
            painter.end()

            self.setPixmap(pixmap)
            self.zoom_changed.emit(self._zoom)

    def resizeEvent(self, event):
        """窗口大小变化时重新计算"""
        super().resizeEvent(event)
        if self._fit_to_widget and self._pixmap is not None:
            self._update_display()

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        if self._pixmap is None:
            return

        # 记录缩放前鼠标位置（相对控件）
        mouse_pos = event.pos()
        widget_center = QPoint(self.width() // 2, self.height() // 2)
        mouse_offset = mouse_pos - widget_center

        self._fit_to_widget = False
        old_zoom = self._zoom

        # 滚轮方向
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * self.ZOOM_STEP, self.MAX_ZOOM)
        elif delta < 0:
            self._zoom = max(self._zoom / self.ZOOM_STEP, self.MIN_ZOOM)

        # 调整偏移量，使鼠标位置对应的图像点保持不变
        zoom_factor = self._zoom / old_zoom
        self._offset = QPoint(
            int(mouse_offset.x() - zoom_factor * (mouse_offset.x() - self._offset.x())),
            int(mouse_offset.y() - zoom_factor * (mouse_offset.y() - self._offset.y()))
        )

        self._update_display()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下：开始拖拽"""
        if event.button() == Qt.LeftButton and not self._fit_to_widget:
            self._drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.RightButton:
            # 右键重置缩放
            self.reset_zoom()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动：拖拽平移"""
        if self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._offset += delta
            self._drag_start = event.pos()
            self._update_display()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放：结束拖拽"""
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            self._drag_start = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """双击重置缩放"""
        if event.button() == Qt.LeftButton:
            self.reset_zoom()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def has_pixmap(self) -> bool:
        return self._pixmap is not None


class ZoomableImageWidget(QWidget):
    """
    带缩放控制按钮的图片显示控件。
    包含 ZoomableLabel + 底部缩放控制栏（放大/缩小/重置/比例显示）。
    """

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._setup_ui(text)

    def _setup_ui(self, text):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = ZoomableLabel(text)
        layout.addWidget(self.label, 1)

        # 底部控制栏
        control_bar = QWidget()
        control_bar.setStyleSheet("background-color: transparent;")
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(4, 0, 4, 0)
        control_layout.setSpacing(4)

        self.btn_zoom_in = QPushButton("＋")
        self.btn_zoom_out = QPushButton("－")
        self.btn_reset = QPushButton("⊙")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #999; font-size: 14px;")

        for btn in [self.btn_zoom_in, self.btn_zoom_out, self.btn_reset]:
            btn.setFixedSize(28, 24)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4;
                    border: 1px solid #555; border-radius: 3px;
                    font-size: 16px; font-weight: bold; padding: 0;
                }
                QPushButton:hover { background-color: #4a4a4a; }
            """)

        control_layout.addWidget(self.btn_zoom_in)
        control_layout.addWidget(self.btn_zoom_out)
        control_layout.addWidget(self.btn_reset)
        control_layout.addWidget(self.zoom_label)
        control_layout.addStretch()

        layout.addWidget(control_bar)

        # 连接信号
        self.btn_zoom_in.clicked.connect(self.label.zoom_in)
        self.btn_zoom_out.clicked.connect(self.label.zoom_out)
        self.btn_reset.clicked.connect(self.label.reset_zoom)
        self.label.zoom_changed.connect(self._on_zoom_changed)

    def _on_zoom_changed(self, zoom: float):
        self.zoom_label.setText(f"{zoom * 100:.0f}%")

    def set_pixmap(self, pixmap: QPixmap):
        self.label.set_pixmap(pixmap)

    def update_pixmap(self, pixmap: QPixmap):
        """更新图片内容，保持当前缩放和平移状态不变"""
        self.label.update_pixmap(pixmap)

    def set_image(self, qimage: QImage):
        self.label.set_image(qimage)

    def clear_pixmap(self):
        self.label.clear_pixmap()

    def has_pixmap(self) -> bool:
        return self.label.has_pixmap()
