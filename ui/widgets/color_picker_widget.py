# -*- coding: utf-8 -*-
"""交互取色控件。

支持在图像上点选或框选像素来获取目标颜色，生成 ColorModel。
复用 MultiROIEditorLabel 的坐标换算思路（图像坐标 <-> 显示坐标）。
"""

from typing import Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QMouseEvent


class ColorPickerWidget(QLabel):
    """可交互取色控件。

    信号:
        color_picked(object): 取色完成，发出 ColorModel
        color_preview(object, int, int): 鼠标移动预览，发出 (ColorModel, x, y)
    """

    color_picked = pyqtSignal(object)
    color_preview = pyqtSignal(object, int, int)

    # 取色模式
    MODE_POINT = "point"
    MODE_RECT = "rect"

    def __init__(self, image_bgr: Optional[np.ndarray] = None, parent=None):
        super().__init__(parent)
        self._image = image_bgr
        self._mode = self.MODE_POINT
        self._rect_start = None
        self._rect_end = None
        self._base_pixmap = None
        self._image_w = 0
        self._image_h = 0
        self._display_scale = 1.0
        self._last_preview = None

        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "background-color: #0d0d0d; border: 1px solid #444; border-radius: 2px;"
        )

        if image_bgr is not None:
            self.set_image(image_bgr)

    # ========== 图像设置 ==========

    def set_image(self, cv_img: np.ndarray):
        """设置显示的图像。"""
        if cv_img is None:
            return
        self._image = cv_img
        try:
            orig_h, orig_w = cv_img.shape[:2]
            self._image_w = orig_w
            self._image_h = orig_h

            # 限制显示尺寸，避免内存不足
            MAX_DISPLAY_SIZE = 2000
            h, w = cv_img.shape[:2]
            scale = 1.0
            if max(h, w) > MAX_DISPLAY_SIZE:
                scale = MAX_DISPLAY_SIZE / max(h, w)
                cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_AREA)
                h, w = cv_img.shape[:2]

            if len(cv_img.shape) == 2:
                q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
            else:
                h, w, ch = cv_img.shape
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)

            if q_img.isNull():
                self.setText("图像过大，内存不足，无法显示")
                self._base_pixmap = None
                return

            self._base_pixmap = QPixmap.fromImage(q_img)
            self._display_scale = scale
            self.update()
        except Exception as e:  # noqa: BLE001
            self.setText(f"图像加载错误: {e}")

    def get_image(self) -> Optional[np.ndarray]:
        """返回原始 BGR 图像。"""
        return self._image

    # ========== 模式设置 ==========

    def set_mode(self, mode: str):
        """设置取色模式：point / rect。"""
        self._mode = mode
        self._rect_start = None
        self._rect_end = None
        self.update()

    def get_mode(self) -> str:
        return self._mode

    # ========== 坐标换算 ==========

    def _get_scaled_rect(self):
        """返回显示区域矩形及缩放比例。"""
        if self._base_pixmap is None:
            return QRect(0, 0, self.width(), self.height()), 1.0, 1.0
        pix_w = self._base_pixmap.width()
        pix_h = self._base_pixmap.height()
        label_w = self.width()
        label_h = self.height()
        if label_w <= 0 or label_h <= 0 or pix_w <= 0 or pix_h <= 0:
            return QRect(0, 0, 0, 0), 1.0, 1.0
        scale = min(label_w / pix_w, label_h / pix_h)
        scaled_w = int(pix_w * scale)
        scaled_h = int(pix_h * scale)
        x = (label_w - scaled_w) // 2
        y = (label_h - scaled_h) // 2
        scale_x = pix_w / scaled_w if scaled_w > 0 else 1
        scale_y = pix_h / scaled_h if scaled_h > 0 else 1
        return QRect(x, y, scaled_w, scaled_h), scale_x, scale_y

    def _label_to_image(self, label_x, label_y):
        """将标签坐标转换为原始图像坐标。"""
        rect, sx, sy = self._get_scaled_rect()
        disp_x = (label_x - rect.x()) * sx
        disp_y = (label_y - rect.y()) * sy
        img_x = disp_x / self._display_scale if self._display_scale > 0 else disp_x
        img_y = disp_y / self._display_scale if self._display_scale > 0 else disp_y
        img_x = max(0, min(img_x, self._image_w / self._display_scale - 1)) if self._image_w > 0 else img_x
        img_y = max(0, min(img_y, self._image_h / self._display_scale - 1)) if self._image_h > 0 else img_y
        return int(img_x), int(img_y)

    def _image_to_label(self, img_x, img_y):
        """将原始图像坐标转换为标签坐标。"""
        rect, sx, sy = self._get_scaled_rect()
        disp_x = img_x * self._display_scale
        disp_y = img_y * self._display_scale
        label_x = rect.x() + disp_x / sx if sx > 0 else rect.x()
        label_y = rect.y() + disp_y / sy if sy > 0 else rect.y()
        return int(label_x), int(label_y)

    # ========== 绘制 ==========

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._base_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect, sx, sy = self._get_scaled_rect()
        painter.drawPixmap(rect, self._base_pixmap, self._base_pixmap.rect())

        # 绘制框选矩形
        if self._mode == self.MODE_RECT and self._rect_start and self._rect_end:
            x0, y0 = self._rect_start
            x1, y1 = self._rect_end
            lx0, ly0 = self._image_to_label(min(x0, x1), min(y0, y1))
            lx1, ly1 = self._image_to_label(max(x0, x1), max(y0, y1))
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QColor(0, 255, 0, 40))
            painter.drawRect(lx0, ly0, lx1 - lx0, ly1 - ly0)

        # 绘制取色点标记
        if self._last_preview is not None:
            model, px, py = self._last_preview
            lx, ly = self._image_to_label(px, py)
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(lx - 6, ly - 6, 12, 12)
            # 显示颜色名称
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            font = QFont("Arial", 10)
            painter.setFont(font)
            painter.drawText(lx + 10, ly, model.name)

        painter.end()

    # ========== 鼠标事件 ==========

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._image is not None:
            img_x, img_y = self._label_to_image(event.x(), event.y())
            if self._mode == self.MODE_POINT:
                self._emit_pick(img_x, img_y)
            else:
                self._rect_start = (img_x, img_y)
                self._rect_end = (img_x, img_y)
                self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._image is not None:
            img_x, img_y = self._label_to_image(event.x(), event.y())
            if self._mode == self.MODE_RECT and self._rect_start:
                self._rect_end = (img_x, img_y)
                self.update()
            # 实时预览取色
            self._emit_preview(img_x, img_y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._mode == self.MODE_RECT \
                and self._rect_start and self._image is not None:
            img_x, img_y = self._label_to_image(event.x(), event.y())
            self._rect_end = (img_x, img_y)
            x0, y0 = self._rect_start
            x1, y1 = self._rect_end
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w >= 3 and h >= 3:
                self._emit_roi_pick(x, y, w, h)
            self._rect_start = None
            self._rect_end = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ========== 取色逻辑 ==========

    def _emit_preview(self, img_x, img_y):
        """鼠标移动时预览取色。"""
        from vision.color.color_sampler import ColorSampler
        try:
            model = ColorSampler.sample_point(
                self._image, img_x, img_y, radius=2, name="预览")
            self._last_preview = (model, img_x, img_y)
            self.color_preview.emit(model, img_x, img_y)
            self.update()
        except Exception:  # noqa: BLE001
            pass

    def _emit_pick(self, img_x, img_y):
        """点选取色。"""
        from vision.color.color_sampler import ColorSampler
        try:
            model = ColorSampler.sample_point(
                self._image, img_x, img_y, radius=3, name="自定义颜色")
            self._last_preview = (model, img_x, img_y)
            self.color_picked.emit(model)
            self.update()
        except Exception as e:  # noqa: BLE001
            print(f"[ColorPicker] 点选取色失败: {e}")

    def _emit_roi_pick(self, x, y, w, h):
        """框选取色。"""
        from vision.color.color_sampler import ColorSampler
        try:
            model = ColorSampler.sample_roi(
                self._image, x, y, w, h, name="自定义颜色")
            self._last_preview = (model, x + w // 2, y + h // 2)
            self.color_picked.emit(model)
            self.update()
        except Exception as e:  # noqa: BLE001
            print(f"[ColorPicker] 框选取色失败: {e}")
