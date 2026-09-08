# -*- coding: utf-8 -*-
"""
位置修正算子配置对话框
========================
在"参考图"(产品标准摆放的照片)上框选一个稳定特征(板角/丝印/mark)作为
定位基准模板,并可设置匹配模式(仅0° / 0°与180°翻转 / 小角度范围)与阈值。

- 框选基准:直接在图像上拖一个矩形,包住特征即可(特征应尽量独特、
  处于参考姿态,后续每次检测都把当前图像校正回该姿态)。
- 高级参数:旋转容差与匹配阈值保留在右侧,便于现场按需微调。
"""
import os
from typing import Optional, List, Dict

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QGroupBox, QGridLayout, QMessageBox,
    QFileDialog, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from vision.tools.position import PositionCorrect
from ui.widgets.param_config_dialog import MultiROIEditorLabel
from core.log_manager import log_info, log_error, log_warning


class PositionCorrectDialog(QDialog):
    """位置修正:参考图上框选基准模板。"""

    def __init__(self, tool: PositionCorrect,
                 preview_image: Optional[np.ndarray] = None,
                 parent=None):
        super().__init__(parent)
        self.tool = tool
        # 允许无预览图启动(仅显示参数),有图时可框选
        self.original_image = None if preview_image is None else preview_image.copy()
        self._candidate = None      # 当前框选的基准 (dict: x/y/w/h)

        self.setWindowTitle("位置修正配置 - 框选定位基准")
        self.setMinimumSize(980, 640)
        self._setup_ui()
        self._load_from_params()
        self._sync_ui_to_canvas()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── 左侧:图像画布 ──
        left = QVBoxLayout()
        self.canvas = MultiROIEditorLabel(self)
        self.canvas.setMinimumSize(620, 460)
        self.canvas.setStyleSheet(
            "background-color:#0d0d0d; border:1px solid #444; border-radius:3px;")
        self.canvas.setText("加载参考图后,在特征上拖框框选基准")
        left.addWidget(self.canvas, 1)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("📷 载入参考图(标准摆放照片)")
        self.btn_load.clicked.connect(self._on_load_ref)
        btn_row.addWidget(self.btn_load)
        btn_row.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#4fc3f7;")
        btn_row.addWidget(self.status_label)
        left.addLayout(btn_row)

        tip = QLabel("提示:框选产品上一个【稳定且独特】的特征(如板角L形、丝印、mark)。"
                     "框要略大于特征本身,特征在参考姿态下应与实际运行时一致。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#999; font-size:13px;")
        left.addWidget(tip)
        main_layout.addLayout(left, 3)

        # ── 右侧:参数 ──
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("基准参数")
        gl = QGridLayout(grp)
        gl.setContentsMargins(8, 12, 8, 8)
        gl.setVerticalSpacing(6)

        gl.addWidget(QLabel("旋转容差:"), 0, 0)
        self.rot_mode = QComboBox()
        self.rot_mode.addItem("仅0°(无翻转)", "0")
        self.rot_mode.addItem("0°/180°翻转", "0and180")
        self.rot_mode.addItem("小角度范围", "range")
        self.rot_mode.currentIndexChanged.connect(self._on_mode_changed)
        gl.addWidget(self.rot_mode, 0, 1)

        self.angle_range_label = QLabel("角度范围(°):")
        self.angle_min = QSpinBox()
        self.angle_min.setRange(-45, 0)
        self.angle_min.setValue(-10)
        self.angle_max = QSpinBox()
        self.angle_max.setRange(0, 45)
        self.angle_max.setValue(10)
        self.angle_step = QDoubleSpinBox()
        self.angle_step.setRange(0.5, 10)
        self.angle_step.setSingleStep(0.5)
        self.angle_step.setValue(2.0)
        self.angle_range_label.setVisible(False)
        self.angle_min.setVisible(False)
        self.angle_max.setVisible(False)
        self.angle_step.setVisible(False)
        gl.addWidget(self.angle_range_label, 1, 0)
        gl.addWidget(self.angle_min, 1, 1)
        gl.addWidget(self.angle_max, 1, 2)
        gl.addWidget(self.angle_step, 1, 3)

        gl.addWidget(QLabel("匹配阈值:"), 2, 0)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.3, 0.99)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.7)
        gl.addWidget(self.threshold, 2, 1)

        self.summary = QLabel("未框选基准")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#aaa;")
        gl.addWidget(self.summary, 3, 0, 1, 4)
        right.addWidget(grp)

        right.addStretch()

        btns = QHBoxLayout()
        self.btn_preview = QPushButton("试运行(校正当前图)")
        self.btn_preview.clicked.connect(self._on_preview)
        btns.addWidget(self.btn_preview)
        btns.addStretch()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        right.addLayout(btns)
        main_layout.addLayout(right, 1)

    # ------------------------------------------------------------------
    # 数据装载 / 同步
    # ------------------------------------------------------------------

    def _load_from_params(self):
        """把 tool.params 读入 UI。"""
        p = self.tool.params
        mode = p.get("rot_mode", "0and180")
        idx = self.rot_mode.findData(mode)
        self.rot_mode.setCurrentIndex(idx if idx >= 0 else 1)
        self.threshold.setValue(float(p.get("threshold", 0.7)))
        self.angle_min.setValue(int(p.get("angle_min", -10)))
        self.angle_max.setValue(int(p.get("angle_max", 10)))
        self.angle_step.setValue(float(p.get("angle_step", 2)))
        # 已有模板时,从 params 取回参考框位置用于展示
        if p.get("ref_w") and p.get("ref_h"):
            self._candidate = {
                "name": "基准", "x": int(p.get("ref_x", 0)),
                "y": int(p.get("ref_y", 0)),
                "w": int(p.get("ref_w", 0)), "h": int(p.get("ref_h", 0)),
                "enabled": True,
            }
            self.summary.setText(
                f"基准已配置: 位置({int(p.get('ref_x',0))}, {int(p.get('ref_y',0))}) "
                f"尺寸 {p.get('ref_w')}×{p.get('ref_h')}px;"
                f" 旋转模式={mode}")
        self._on_mode_changed()

    def _sync_ui_to_canvas(self):
        """把当前基准框同步到画布,供画布钩子回调。"""
        if self.original_image is None:
            self.canvas.regions = []
            return
        self.canvas.set_base_image(self.original_image)
        # 若已有模板但换图了,只提示不清除(工程上参考图应固定)
        if self._candidate is not None:
            self.canvas.regions = [dict(self._candidate)]
            self.canvas.selected_idx = 0
        else:
            self.canvas.regions = []
        self.canvas.update()

    # 画布父类钩子(拖框完成后调用)
    def _refresh_list(self):
        if self.canvas._temp_region_idx is None:
            return
        idx = self.canvas._temp_region_idx
        if idx is not None and idx >= 0 and idx < len(self.canvas.regions):
            r = self.canvas.regions[idx]
            if r["w"] >= 8 and r["h"] >= 8:
                self._candidate = dict(r)
                self.summary.setText(
                    f"基准框: 位置({r['x']}, {r['y']}) 尺寸 {r['w']}×{r['h']}px")
                self.status_label.setText("已框选基准 ✔")
            else:
                del self.canvas.regions[idx]
                self.canvas.update()
        self.canvas._temp_region_idx = -1
        self.canvas.update()

    def _update_property_panel(self):
        pass  # 单框场景无需额外面板

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def _on_mode_changed(self):
        is_range = self.rot_mode.currentData() == "range"
        for w in (self.angle_range_label, self.angle_min,
                  self.angle_max, self.angle_step):
            w.setVisible(is_range)

    def _on_load_ref(self):
        # 优先用宿主相机抓帧;否则文件选择
        img = self._try_capture()
        if img is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择参考图(标准摆放照片)", "",
                "图像 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)")
            if not path:
                return
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                QMessageBox.warning(self, "错误", "无法读取参考图")
                return
        self.original_image = img.copy()
        self._candidate = None
        self.summary.setText("请在特征上拖框框选基准")
        self._sync_ui_to_canvas()

    def _try_capture(self):
        win = self.window()
        cam = getattr(win, "camera_mgr", None)
        if cam is None:
            return None
        try:
            if not (getattr(cam, "is_open", False) or getattr(cam, "is_connected", False)):
                return None
            if hasattr(cam, "capture_once"):
                raw = cam.capture_once()
                if isinstance(raw, tuple) and len(raw) == 4:
                    from camera_manager import raw_to_opencv
                    w, h, pt, data = raw
                    return raw_to_opencv(data, w, h, pt)
                return raw
        except Exception as e:  # noqa: BLE001
            log_warning(f"相机抓帧失败: {e}")
        return None

    def _on_preview(self):
        """用当前框选结果试运行(若已配置模板与图像)。"""
        if self._candidate is None:
            QMessageBox.information(self, "提示", "请先在参考图上框选基准")
            return
        if self.original_image is None:
            return
        # 用当前对话框参数暂存试运行(不落 params)
        self._apply_params()
        # 对参考图本身试运行:应 score≈1 且角度≈0
        from vision.pipeline import PipelineContext
        from vision.tools.position import PositionCorrect
        tmp = PositionCorrect()
        tmp.params = dict(self.tool.params)
        try:
            ctx = PipelineContext(original_image=self.original_image,
                                  current_image=self.original_image)
            res = tmp.process(ctx)
            if bool(res.passed):
                self.status_label.setText(
                    f"试运行: 定位成功 score={res.data.get('score'):.2f} "
                    f"角度={res.data.get('angle_deg'):.0f}°")
                self._show_image(res.processed_image)
            else:
                self.status_label.setText(
                    f"试运行: 定位失败 {res.message}")
        except Exception as e:  # noqa: BLE001
            log_error(f"位置修正试运行失败: {e}")
            self.status_label.setText(f"试运行异常: {e}")

    def _show_image(self, cv_img):
        if cv_img is None:
            return
        try:
            if len(cv_img.shape) == 2:
                h, w = cv_img.shape
                q = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
            else:
                h, w, ch = cv_img.shape
                rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            # 展示于画布(仅预览,不覆盖框选状态)
            self.canvas._base_pixmap = QPixmap.fromImage(q)
            self.canvas.update()
        except Exception as e:  # noqa: BLE001
            self.status_label.setText(f"显示异常: {e}")

    def _apply_params(self):
        """把 UI 状态写入 tool.params。"""
        p = self.tool.params
        p["rot_mode"] = self.rot_mode.currentData()
        p["threshold"] = float(self.threshold.value())
        p["angle_min"] = int(self.angle_min.value())
        p["angle_max"] = int(self.angle_max.value())
        p["angle_step"] = float(self.angle_step.value())
        if self._candidate is not None:
            c = self._candidate
            if self.original_image is not None:
                h, w = self.original_image.shape[:2]
                c["x"] = max(0, min(int(c["x"]), w - 1))
                c["y"] = max(0, min(int(c["y"]), h - 1))
                c["w"] = max(4, min(int(c["w"]), w - c["x"]))
                c["h"] = max(4, min(int(c["h"]), h - c["y"]))
            p["ref_x"], p["ref_y"] = int(c["x"]), int(c["y"])
            p["ref_w"], p["ref_h"] = int(c["w"]), int(c["h"])
            if self.original_image is not None:
                templ = self.original_image[c["y"]:c["y"] + c["h"],
                                            c["x"]:c["x"] + c["w"]].copy()
                # 模板去噪(轻微高斯)提升匹配稳定性
                templ = cv2.GaussianBlur(templ, (3, 3), 0)
                # 复用 set_template(编码 base64 + 写 ref 参数)
                self.tool.set_template(templ, int(c["x"]), int(c["y"]))

    def _on_ok(self):
        if self._candidate is None or self.original_image is None:
            QMessageBox.warning(self, "提示", "请先载入参考图并框选基准")
            return
        try:
            self._apply_params()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"保存基准失败: {e}")
            return
        self.accept()
