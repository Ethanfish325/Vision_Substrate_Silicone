# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from core.paths import ensure_dirs, ICON_FILE
from core.log_manager import init_logger, log_info
from core.config_manager import ConfigManager
from ui.main_window import MainWindow
from ui.widgets.mes_dialog import MESDialog
     

def setup_high_dpi():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def _show_mes_dialog(app: QApplication):
    """启动时弹出 MES 设置窗口（模态）。

    回填上次保存的配置，用户确认后保存选择。
    无论是否使用 MES，都直接进入主界面（不使用 MES 则正常使用，只是不上报）。
    """
    from PyQt5.QtWidgets import QDialog
    dialog = MESDialog()
    dialog.exec_()


def _selftest_barcode(image_path: str) -> int:
    """打包自检:验证"条码识别"在打包环境下是否可用。

    用法(窗口模式无 stdout,结果写入 JSON 文件):
        Vision_Substrate_Silicone.exe --selftest-barcode <图片路径> [输出json]

    输出内容包括:运行环境(frozen/cv2/numpy)、pyzbar 与 zbar DLL 加载情况、
    图片读取结果、识别到的条码与尝试过的解码策略。用于快速定位
    "源码能识别、打包后识别不到"这类缺库/缺 DLL 问题。
    """
    import json
    import traceback

    info = {"image": image_path}
    try:
        info["frozen"] = bool(getattr(sys, "frozen", False))
        info["executable"] = sys.executable
        info["python"] = sys.version.split()[0]
        info["bits"] = 64 if sys.maxsize > 2 ** 32 else 32
        import cv2 as _cv2
        info["cv2"] = getattr(_cv2, "__version__", "?")
        info["cv2_barcode_module"] = hasattr(_cv2, "barcode")
        import numpy as _np
        info["numpy"] = getattr(_np, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        info["env_error"] = f"{type(e).__name__}: {e}"

    # pyzbar / zbar DLL
    try:
        import pyzbar as _pyzbar
        info["pyzbar_file"] = getattr(_pyzbar, "__file__", None)
        from pyzbar import zbar_library as _zl
        libzbar, deps = _zl.load()
        info["zbar_loaded"] = True
        info["zbar_deps"] = len(deps)
    except Exception as e:  # noqa: BLE001
        info["zbar_loaded"] = False
        info["zbar_error"] = f"{type(e).__name__}: {e}"

    # 识别
    try:
        import cv2
        img = cv2.imread(image_path)
        info["image_shape"] = None if img is None else list(img.shape)
        if img is not None:
            from vision.tools.recognize import QRCodeRecognize
            from vision.tools.base_tool import PipelineContext
            tool = QRCodeRecognize()
            ctx = PipelineContext(original_image=img, current_image=img)
            res = tool.process(ctx)
            info["recognized"] = bool(res.data.get("recognized"))
            info["qr_data"] = res.data.get("qr_data")
            info["barcode_count"] = res.data.get("barcode_count")
            info["barcodes"] = res.data.get("barcodes")
            info["strategies_tried"] = list(tool._last_variants)
    except Exception as e:  # noqa: BLE001
        info["recognize_error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()

    out_path = (sys.argv[3] if len(sys.argv) > 3 else
                os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                             "selftest_barcode.json"))
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"[selftest] 结果已写入: {out_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[selftest] 写结果失败: {e}")
    print(f"[selftest] {json.dumps(info, ensure_ascii=False)}")
    return 0


def main():
    setup_high_dpi()

    app = QApplication(sys.argv)
    app.setApplicationName("PCBA导热硅胶检测设备")
    app.setApplicationVersion("1.0.0")

    # 设置应用图标
    if os.path.exists(ICON_FILE):
        app.setWindowIcon(QIcon(ICON_FILE))

    init_logger()
    log_info("=== PCBA导热硅胶检测设备启动 ===")

    ensure_dirs()

    # 启动时先弹出 MES 设置窗口（模态），回填上次配置，保存选择
    _show_mes_dialog(app)

    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #2d2d2d;
            color: #d4d4d4;
            font-size: 21px;
        }
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px 10px;
            min-height: 20px;
            color: #d4d4d4;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
            border-color: #4A90D9;
        }
        QPushButton:pressed {
            background-color: #4A90D9;
            color: #fff;
        }
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 6px;
            min-height: 20px;
            color: #d4d4d4;
        }
        QComboBox:hover {
            border-color: #4A90D9;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #2d2d2d;
            color: #d4d4d4;
            selection-background-color: #1a3a5c;
        }
        QLineEdit {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 6px;
            min-height: 20px;
            color: #d4d4d4;
        }
        QLineEdit:focus {
            border-color: #4A90D9;
        }
        QListWidget {
            background-color: #2d2d2d;
            border: 1px solid #444;
            border-radius: 3px;
            color: #d4d4d4;
        }
        QListWidget::item:hover {
            background-color: #3a3a3a;
        }
        QListWidget::item:selected {
            background-color: #1a3a5c;
            color: #4A90D9;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            background-color: #2d2d2d;
        }
        QTabBar::tab {
            background-color: #3c3c3c;
            padding: 4px 12px;
            border: 1px solid #444;
            border-bottom: none;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
            color: #999;
        }
        QTabBar::tab:selected {
            background-color: #2d2d2d;
            color: #4A90D9;
            font-weight: bold;
        }
        QGroupBox {
            border: 1px solid #444;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 12px;
            color: #d4d4d4;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #d4d4d4;
        }
        QScrollBar:vertical {
            background-color: #1e1e1e;
            width: 8px;
        }
        QScrollBar::handle:vertical {
            background-color: #555;
            border-radius: 4px;
            min-height: 16px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #4A90D9;
        }
        QSplitter::handle {
            background-color: #444;
            width: 2px;
        }
        QMenuBar {
            background-color: #1e1e1e;
            color: #d4d4d4;
            border-bottom: 1px solid #444;
        }
        QMenuBar::item:selected {
            background-color: #3c3c3c;
        }
        QMenu {
            background-color: #2d2d2d;
            color: #d4d4d4;
            border: 1px solid #444;
        }
        QMenu::item:selected {
            background-color: #1a3a5c;
            color: #4A90D9;
        }
        QStatusBar {
            background-color: #1e1e1e;
            color: #999;
            border-top: 1px solid #444;
        }
        QCheckBox {
            color: #d4d4d4;
        }
        QSpinBox {
            background-color: #3c3c3c;
            color: #d4d4d4;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 1px 3px;
        }
        QSpinBox:focus {
            border-color: #4A90D9;
        }
        QTableWidget {
            background-color: #2d2d2d;
            color: #d4d4d4;
            border: 1px solid #444;
            gridline-color: #3a3a3a;
        }
        QHeaderView::section {
            background-color: #3c3c3c;
            color: #999;
            border: none;
            border-bottom: 1px solid #444;
            padding: 2px 6px;
            font-weight: bold;
        }
        QDialog {
            background-color: #2d2d2d;
        }
        QTextEdit {
            background-color: #1e1e1e;
            color: #c8c8c8;
            border: 1px solid #444;
        }
    """)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    # 打包自检入口(不影响正常启动):
    #   Vision_Substrate_Silicone.exe --selftest-barcode <图片路径> [输出json]
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest-barcode":
        sys.exit(_selftest_barcode(sys.argv[2]))
    main()
