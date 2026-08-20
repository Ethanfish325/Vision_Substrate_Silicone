# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from core.paths import ensure_dirs, ICON_FILE
from core.log_manager import init_logger, log_info
from ui.main_window import MainWindow
     

def setup_high_dpi():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


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
 
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #2d2d2d;
            color: #d4d4d4;
            font-size: 12px;
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
    main()
