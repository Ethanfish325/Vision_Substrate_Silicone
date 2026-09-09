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
        # 优先用模板内保存的参考图(还原上次配置的框选底图);
        # 没有则退回当前预览图(可能是相机画面/测试图)。
        saved_ref = tool.get_reference_image()
        if saved_ref is not None:
            self.original_image = saved_ref
            self._image_source = "已保存的参考图"
        else:
            self.original_image = None if preview_image is None \
                else preview_image.copy()
            self._image_source = "当前画面"
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
        self.btn_load_file = QPushButton("📁 载入参考图")
        self.btn_load_file.setToolTip("从图片文件载入标准摆放照片作为框选底图")
        self.btn_load_file.clicked.connect(self._on_load_file)
        btn_row.addWidget(self.btn_load_file)
        self.btn_use_camera = QPushButton("📷 用相机当前画面作基准图")
        self.btn_use_camera.setToolTip("相机已打开时,把当前采集画面作为框选底图(标准摆放)")
        self.btn_use_camera.clicked.connect(self._on_use_camera)
        btn_row.addWidget(self.btn_use_camera)
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
        self.rot_mode.addItem("任意角度(全周,较慢)", "any")
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
        """把 tool.params 读入 UI。

        注意:此时 _candidate 是从参数还原的"已有基准框"。为了与"只调
        参数不重框基准"解耦,保存时只有【用户新载图后新框选】才重写模板;
        还原的旧候选仅在用户主动拖框覆盖后才更新模板。
        """
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
            # 该候选来自已保存模板:重开对话框不应静默重写模板
            self._candidate["_loaded"] = True
            self.summary.setText(
                f"基准已配置(可直接改参数后确定): "
                f"位置({int(p.get('ref_x',0))}, {int(p.get('ref_y',0))}) "
                f"尺寸 {p.get('ref_w')}×{p.get('ref_h')}px;"
                f" 旋转模式={mode}\n"
                f"如需更换基准,请重新载入/拍摄底图并在特征上拖框。")
        self._on_mode_changed()
        self._set_source_hint()

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
                # 用户主动新框选 → 保存时重写模板
                self._candidate["_loaded"] = False
                self.summary.setText(
                    f"基准框(新): 位置({r['x']}, {r['y']}) 尺寸 {r['w']}×{r['h']}px")
                self.status_label.setText("已框选基准 ✔ (保存时更新)")
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
        mode = self.rot_mode.currentData()
        is_range = mode == "range"
        for w in (self.angle_range_label, self.angle_min,
                  self.angle_max, self.angle_step):
            w.setVisible(is_range)
        # any 模式用固定两级搜索(粗 30° + 细 2°),无需界面参数

    def _set_source_hint(self):
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            self.status_label.setText(f"底图: {self._image_source} ({w}×{h})"
                                      + ("  在特征上拖框框选基准"
                                         if self._candidate is None else
                                         "  ✔ 已框选基准"))
        else:
            self.status_label.setText("尚未载入基准图,请先选择图片来源")

    # 来源:图片文件 / 相机当前画面(统一入口)
    def _set_reference_source(self, img: Optional[np.ndarray],
                              source_label: str) -> bool:
        if img is None:
            return False
        self.original_image = img.copy()
        self._image_source = source_label
        self._candidate = None
        # 记录底图来源与参考图(保存后重开可还原)
        self.tool.set_reference_image(self.original_image)
        self.summary.setText("请在特征上拖框框选基准")
        self._sync_ui_to_canvas()
        self._set_source_hint()
        return True

    def _on_load_file(self):
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
        self._set_reference_source(img, os.path.basename(path))

    def _on_use_camera(self):
        """把相机当前采集画面作为基准图(标准摆放)。"""
        img = self._try_capture()
        if img is None:
            QMessageBox.warning(
                self, "相机不可用",
                "未能从相机获取画面。请先在主界面打开相机,或改用「载入参考图」。")
            return
        self._set_reference_source(img, "相机画面(需产品处于标准摆放位)")

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
        """用当前参数对底图试运行,验证定位(score/角度)。"""
        has_template = bool(self.tool.params.get("template_b64"))
        if self._candidate is None and not has_template:
            QMessageBox.information(
                self, "提示",
                "请先「📷 用相机当前画面作基准图」或「📁 载入参考图」"
                "并框选基准,再试运行。")
            return
        if self.original_image is None:
            return
        # 写入当前参数(新框选才重写模板;还原基准仅更新参数)
        if not self._apply_params():
            QMessageBox.warning(self, "基准无效",
                                "框选区域几乎无纹理(纯色板面/底色),无法定位。\n"
                                "请框选产品上的稳定特征(板角/丝印/mark/定位孔)。")
            return
        # 对当前底图试运行
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

    def _apply_params(self) -> bool:
        """把 UI 参数写入 tool.params;若用户新框选基准则重写模板。

        解耦逻辑(问题6):
            - 仅调整 旋转模式/阈值/角度 → 不动 template_b64(已存模板保留);
            - 新载底图并重新框选(_candidate 非 _loaded) → 才重裁模板;
            - 还原的旧基准(_loaded=True)且未重框 → 只写位置/尺寸等,
              不覆盖模板图。
        """
        p = self.tool.params
        p["rot_mode"] = self.rot_mode.currentData()
        p["threshold"] = float(self.threshold.value())
        p["angle_min"] = int(self.angle_min.value())
        p["angle_max"] = int(self.angle_max.value())
        p["angle_step"] = float(self.angle_step.value())

        if self._candidate is None:
            # 无任何基准(未载图/未框选):仅参数已更新;由确定处提示
            return True
        c = dict(self._candidate)   # copy,不污染画布引用
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            c["x"] = max(0, min(int(c["x"]), w - 1))
            c["y"] = max(0, min(int(c["y"]), h - 1))
            c["w"] = max(4, min(int(c["w"]), w - c["x"]))
            c["h"] = max(4, min(int(c["h"]), h - c["y"]))
        p["ref_x"], p["ref_y"] = int(c["x"]), int(c["y"])
        p["ref_w"], p["ref_h"] = int(c["w"]), int(c["h"])

        # 参考图(底图)始终跟随 current 底图保存(便于还原),不影响模板
        if self.original_image is not None:
            self.tool.set_reference_image(self.original_image)

        # 只有"用户新框选"才重写模板;从已存模板还原且未重框时保留原模板
        if self._candidate.get("_loaded"):
            return True
        if self.original_image is None:
            return True   # 理论不可达(重框需底图),兜底
        templ = self.original_image[c["y"]:c["y"] + c["h"],
                                    c["x"]:c["x"] + c["w"]].copy()
        templ = cv2.GaussianBlur(templ, (3, 3), 0)  # 轻微去噪
        ok = self.tool.set_template(templ, int(c["x"]), int(c["y"]))
        if not ok:
            self.status_label.setText(
                "✗ 基准无特征(纯色板面/底色)——请框选板上稳定特征(板角/丝印/mark)")
            self.summary.setText(
                "基准无效: 框内几乎无纹理,匹配会失败。请改框特征区域。")
        else:
            self.summary.setText(
                f"基准已更新: ({c['x']}, {c['y']}) {c['w']}×{c['h']}px")
        return ok

    def _on_ok(self):
        # 情形A:从未有基准 → 必须载底图并框选(仅参数无意义)
        has_template = bool(self.tool.params.get("template_b64"))
        if self._candidate is None and not has_template:
            QMessageBox.warning(
                self, "提示",
                "尚未配置定位基准。\n请先「📷 用相机当前画面作基准图」或"
                "「📁 载入参考图」,然后在产品稳定特征(板角/丝印/mark)上拖框框选。")
            return
        try:
            ok = self._apply_params()
            if not ok:
                QMessageBox.warning(
                    self, "基准无效",
                    "新框选的区域几乎无纹理(纯色板面/底色),无法用于定位。\n"
                    "请框选产品上的稳定特征(板角/丝印/mark/定位孔),再点确定。")
                return
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"保存基准失败: {e}")
            return
        self.accept()
