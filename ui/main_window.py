# -*- coding: utf-8 -*-
import sys
import os
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import json
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QFont, QIcon

from camera_manager import CameraManager, raw_to_opencv
from .widgets.zoomable_label import ZoomableLabel, ZoomableImageWidget
from core.config_manager import ConfigManager
from core.log_manager import log_info, log_error, log_warning, LogManager
from vision.vision_engine import VisionEngine
from vision.pipeline import Pipeline

from .widgets.camera_panel import CameraPanel
from core.paths import SCHEME_DIR
from .widgets.pipeline_editor import PipelineEditor
from .widgets.result_panel import ResultPanel

from core.serial_comm import SerialCommManager
from core.serial_test_workflow import SerialTestWorkflow, WorkflowConfig
from core.nmc_sdk import NMCSDK, Axis_1, Axis_2, Axis_3, Axis_4, Stop_Smooth, Stop_Abrupt
from core.inspection_workflow import InspectionWorkflow
from core.product_manager import list_products, load_product, save_product, create_default_config

import hashlib
from core.paths import USERS_FILE


def _verify_password(input_password: str, stored_hash: str) -> bool:
    """验证密码：对输入密码进行 SHA256 哈希，与存储的哈希值比对"""
    return hashlib.sha256(input_password.encode('utf-8')).hexdigest() == stored_hash


def _load_users() -> dict:
    """加载 users.json 中的用户数据"""
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


class StepLogPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("执行日志")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #d4d4d4; padding: 1px 0;")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a; color: #c8c8c8;
                border: 1px solid #444; border-radius: 3px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 13px; padding: 2px;
            }
        """)
        self.log_text.setMinimumHeight(60)

        layout.addWidget(title)
        layout.addWidget(self.log_text, 1)

    def append_log(self, timestamp: str, step_index: int, tool_name: str,
                   status: str, message: str, elapsed_ms: float):
        color = "#8bc34a" if status == "✓" else "#ff5252"
        line = (
            f'<span style="color:#888;">[{timestamp}]</span> '
            f'<span style="color:#4fc3f7;">步骤{step_index}:</span> '
            f'<span style="color:#e0e0e0;">{tool_name}</span> - '
            f'<span style="color:{color};">{status}</span> '
            f'<span style="color:#b0b0b0;">{message}</span> '
            f'<span style="color:#888;">({elapsed_ms:.1f}ms)</span>'
        )
        self.log_text.append(line)

    def append_info(self, text: str, color: str = "#888"):
        line = f'<span style="color:{color};">{text}</span>'
        self.log_text.append(line)

    def append_separator(self):
        self.log_text.append('<hr style="border: none; border-top: 1px solid #555;">')

    def clear_log(self):
        self.log_text.clear()


class DetectWorker(QThread):
    """后台检测工作线程，避免阻塞UI"""
    finished = pyqtSignal(bool, str, np.ndarray, object)  # passed, message, annotated, results

    def __init__(self, vision_engine, raw_image, scheme_name):
        super().__init__()
        self._vision_engine = vision_engine
        self._raw_image = raw_image
        self._scheme_name = scheme_name

    def run(self):
        try:
            passed, message, annotated = self._vision_engine.execute(
                self._raw_image, scheme_name=self._scheme_name
            )
            results = self._vision_engine.get_last_results()
            self.finished.emit(passed, message, annotated, results)
        except Exception as e:
            self.finished.emit(False, f"检测异常: {str(e)}", self._raw_image, [])


class EngineerTestWorker(QThread):
    """后台工程师测试工作线程，避免阻塞UI"""
    finished = pyqtSignal(bool, str, np.ndarray, object)  # passed, message, annotated, results

    def __init__(self, vision_engine, raw_image, scheme_name):
        super().__init__()
        self._vision_engine = vision_engine
        self._raw_image = raw_image
        self._scheme_name = scheme_name

    def run(self):
        try:
            passed, message, annotated = self._vision_engine.execute(
                self._raw_image, scheme_name=self._scheme_name
            )
            results = self._vision_engine.get_last_results()
            self.finished.emit(passed, message, annotated, results)
        except Exception as e:
            self.finished.emit(False, f"测试异常: {str(e)}", self._raw_image, [])


class WorkflowTestWorker(QThread):
    """后台工作流测试工作线程，避免阻塞UI"""
    finished = pyqtSignal(bool, str, np.ndarray, object)  # passed, message, annotated, results

    def __init__(self, vision_engine, image, scheme_name):
        super().__init__()
        self._vision_engine = vision_engine
        self._image = image
        self._scheme_name = scheme_name

    def run(self):
        try:
            passed, message, annotated = self._vision_engine.execute(
                self._image, scheme_name=self._scheme_name
            )
            results = self._vision_engine.get_last_results()
            self.finished.emit(passed, message, annotated, results)
        except Exception as e:
            self.finished.emit(False, f"自动测试异常: {str(e)}", self._image, [])


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.camera_mgr = CameraManager()
        self.vision_engine = VisionEngine()
        self._detect_worker = None  # 后台检测线程
        self._eng_test_worker = None  # 后台工程师测试线程
        self._workflow_test_worker = None  # 后台工作流测试线程

        self._raw_image = None
        self._raw_width = 0
        self._raw_height = 0

        self._schemes = {}
        self._current_scheme_name = None

        # 步骤导航相关
        self._step_results = []       # List[ToolResult]，流水线各步骤结果
        self._current_step_index = -1  # -1 表示显示最终标注结果
        self._annotated_image = None   # 最终标注结果图（原始图 + 所有 overlay 叠加）

        self._camera_panel = None      # 相机面板，延迟创建
        self._pending_engineer_test = False  # 设计模式测试标记：拍照后自动执行流水线
        # === COMMENTED OUT: 生产模式标记 ===
        # self._pending_detect = False    # 生产模式标记：拍照后自动执行检测
        # === END ===

        # === COMMENTED OUT: 生产模式最近一次检测的标注结果 ===
        # self._last_annotated = None
        # === END ===

        # 用户角色与权限控制
        self._current_user_role = "engineer"   # 当前用户角色: operator / engineer / admin
        self._current_user_name = "工程师"      # 当前用户显示名称
        '''
        # 用户角色与权限控制
        self._current_user_role = "operator"   # 当前用户角色: operator / engineer / admin
        self._current_user_name = "操作员"      # 当前用户显示名称
        '''
        # 串口通信与自动测试
        self._serial_comm: Optional[SerialCommManager] = None
        self._serial_workflow: Optional[SerialTestWorkflow] = None

        # 运动控制卡 - 在初始化时连接
        self._nmc_sdk: Optional[NMCSDK] = None
        self._init_nmc()

        # 自动化检测工作流
        self._inspection_workflow: Optional[InspectionWorkflow] = None
        self._init_inspection_workflow()

        # 自动化检测面板（在 _build_automation_page 中创建）
        self._inspection_panel = None

        self._setup_ui()
        self._load_schemes()
        self._auto_load_default_scheme()
        self._init_sdk()

        # 启动后延迟自动连接相机（等待 UI 完全渲染）
        QTimer.singleShot(500, self._auto_connect_camera)

    def _setup_ui(self):
        self.setWindowTitle("基板硅胶视觉检测系统")
        self.setMinimumSize(1024, 700)

        self._setup_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._setup_mode_toolbar(main_layout)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        # === COMMENTED OUT: 生产模式页面 ===
        # self._build_worker_page()
        # === END ===

        self._build_automation_page()

        self._build_engineer_page()

        self.stack.setCurrentIndex(0)  # 默认显示自动化模式（索引0）

        self.status_label = QLabel("就绪")
        self.scheme_status_label = QLabel("当前方案: 未选择")
        self.scheme_status_label.setStyleSheet("color: #d4d4d4; font-weight: bold;")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.scheme_status_label)

    def _setup_mode_toolbar(self, parent_layout):
        toolbar = QWidget()
        toolbar.setStyleSheet("""
            background-color: #1e1e1e; border-bottom: 1px solid #444;
        """)
        toolbar.setFixedHeight(32)
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(6, 1, 6, 1)
        layout.setSpacing(4)

        # === COMMENTED OUT: 生产模式按钮 ===
        # self.btn_worker_mode = QPushButton("🔧 生产模式")
        # self.btn_worker_mode.setCheckable(True)
        # self.btn_worker_mode.setChecked(True)
        # self.btn_worker_mode.setStyleSheet("""
        #     QPushButton {
        #         background-color: #3c3c3c; color: #d4d4d4; padding: 4px 16px;
        #         border: 1px solid #555; border-radius: 3px; font-size: 18px;
        #         font-weight: bold;
        #     }
        #     QPushButton:checked {
        #         background-color: #1a3a5c; border: 1px solid #4A90D9;
        #         color: #4A90D9;
        #     }
        #     QPushButton:hover { background-color: #4a4a4a; }
        # """)
        # === END ===

        self.btn_automation_mode = QPushButton("🤖 自动化模式")
        self.btn_automation_mode.setCheckable(True)
        self.btn_automation_mode.setChecked(True)  # 默认选中自动化模式
        self.btn_automation_mode.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4; padding: 2px 10px;
                border: 1px solid #555; border-radius: 3px; font-size: 14px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #1a3a2a; border: 1px solid #4CAF50;
                color: #66BB6A;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """)

        self.btn_engineer_mode = QPushButton("⚙ 设计模式")
        self.btn_engineer_mode.setCheckable(True)
        self.btn_engineer_mode.setEnabled(True)  # 默认操作员模式，禁用设计模式
        self.btn_engineer_mode.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4; padding: 2px 10px;
                border: 1px solid #555; border-radius: 3px; font-size: 14px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #3a2a1a; border: 1px solid #E65100;
                color: #E65100;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled {
                background-color: #252525; color: #555555;
                border: 1px solid #3a3a3a;
            }
        """)

        # === COMMENTED OUT: 生产模式按钮 ===
        # layout.addWidget(self.btn_worker_mode)
        # === END ===
        layout.addWidget(self.btn_automation_mode)
        layout.addWidget(self.btn_engineer_mode)
        layout.addStretch()

        self.mode_scheme_label = QLabel("当前方案: 未选择")
        self.mode_scheme_label.setStyleSheet("color: #999; font-size: 13px;")

        layout.addWidget(self.mode_scheme_label)

        # === COMMENTED OUT: 生产模式按钮 ===
        # self.btn_worker_mode.clicked.connect(lambda: self._switch_mode(0))
        # === END ===
        self.btn_automation_mode.clicked.connect(lambda: self._switch_mode(0))  # 索引变化：0=自动化模式
        self.btn_engineer_mode.clicked.connect(lambda: self._switch_mode(1))     # 索引变化：1=设计模式

        parent_layout.addWidget(toolbar)

    def _switch_mode(self, index: int):
        # 如果尝试切换到设计模式但当前用户不是工程师/管理员，阻止切换
        if index == 1 and self._current_user_role not in ("engineer", "admin"):  # 索引变化：1=设计模式
            QMessageBox.warning(self, "权限不足", "请先通过「用户」菜单登录工程师账号")
            # === COMMENTED OUT: 生产模式按钮 ===
            # self.btn_worker_mode.setChecked(True)
            # === END ===
            self.btn_automation_mode.setChecked(True)
            self.btn_engineer_mode.setChecked(False)
            return

        self.stack.setCurrentIndex(index)
        # === COMMENTED OUT: 生产模式按钮 ===
        # self.btn_worker_mode.setChecked(index == 0)
        # === END ===
        self.btn_automation_mode.setChecked(index == 0)  # 索引变化：0=自动化模式
        self.btn_engineer_mode.setChecked(index == 1)     # 索引变化：1=设计模式

        # === COMMENTED OUT: 生产模式 ===
        # mode_names = {0: "生产模式", 1: "自动化模式", 2: "设计模式"}
        # === END ===
        mode_names = {0: "自动化模式", 1: "设计模式"}
        self.status_label.setText(mode_names.get(index, "未知模式"))

    # ──────────────── 用户登录 / 权限控制 ────────────────

    def _show_login_dialog(self):
        """弹出登录对话框，选择角色并输入密码"""
        dialog = QDialog(self)
        dialog.setWindowTitle("登录")
        dialog.setFixedSize(300, 180)
        dialog.setStyleSheet("""
            QDialog { background-color: #2d2d2d; }
            QLabel { color: #d4d4d4; font-size: 13px; }
            QComboBox, QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 3px 8px; font-size: 13px;
            }
            QPushButton {
                background-color: #1a3a5c; color: #4A90D9;
                border: 1px solid #2a5a8c; border-radius: 3px;
                padding: 4px 16px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2a4a7c; }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # 角色选择
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("登录为："))
        role_combo = QComboBox()
        role_combo.addItem("工程师", "engineer")
        role_combo.addItem("管理员", "admin")
        role_layout.addWidget(role_combo, 1)
        layout.addLayout(role_layout)

        # 密码输入
        pwd_layout = QHBoxLayout()
        pwd_layout.addWidget(QLabel("密  码："))
        pwd_input = QLineEdit()
        pwd_input.setEchoMode(QLineEdit.Password)
        pwd_layout.addWidget(pwd_input, 1)
        layout.addLayout(pwd_layout)

        # 错误提示
        error_label = QLabel("")
        error_label.setStyleSheet("color: #ff5252; font-size: 12px;")
        error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(error_label)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_login = QPushButton("登录")
        btn_cancel = QPushButton("取消")
        btn_cancel.setStyleSheet("""
            QPushButton { background-color: #3c3c3c; color: #d4d4d4;
                           border: 1px solid #555; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        btn_layout.addWidget(btn_login)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        def _do_login():
            role_key = role_combo.currentData()
            password = pwd_input.text()
            if not password:
                error_label.setText("请输入密码")
                return

            users = _load_users()
            user_info = users.get(role_key)
            if user_info and _verify_password(password, user_info["password_hash"]):
                self._set_user_role(role_key, user_info["display_name"])
                dialog.accept()
            else:
                error_label.setText("密码错误，请重试")
                pwd_input.clear()
                pwd_input.setFocus()

        pwd_input.returnPressed.connect(_do_login)
        btn_login.clicked.connect(_do_login)
        btn_cancel.clicked.connect(dialog.reject)

        dialog.exec_()

    def _set_user_role(self, role: str, display_name: str):
        """设置当前用户角色，更新 UI 状态"""
        self._current_user_role = role
        self._current_user_name = display_name

        # 更新菜单显示
        self.act_current_user.setText(f"当前用户：{display_name}")

        # 工程师或管理员可以访问设计模式
        is_engineer = role in ("engineer", "admin")
        self.btn_engineer_mode.setEnabled(is_engineer)
        self.act_logout.setEnabled(is_engineer)

        # === COMMENTED OUT: 生产模式 ===
        # 如果当前在设计模式但角色不是工程师，自动切回自动化模式
        if not is_engineer and self.stack.currentIndex() == 1:  # 索引变化：1=设计模式
            self._switch_mode(0)  # 0=自动化模式
        # === END ===

        log_info(f"用户切换: {display_name}({role})")

    def _logout(self):
        """退出登录，回到操作员模式"""
        self._set_user_role("operator", "操作员")
        # === COMMENTED OUT: 生产模式 ===
        # self.status_label.setText("生产模式")
        # === END ===
        self.status_label.setText("自动化模式")

    def _build_worker_page(self):
        """生产模式页面 - 已注释，保留以方便后续恢复"""
        pass
        # === COMMENTED OUT: 生产模式页面 ===
        # page = QWidget()
        # layout = QVBoxLayout(page)
        # layout.setContentsMargins(12, 8, 12, 8)
        # layout.setSpacing(8)
        #
        # top_bar = QWidget()
        # top_bar.setStyleSheet("background-color: #2d2d2d; border: 1px solid #444; border-radius: 4px;")
        # top_bar.setFixedHeight(60)
        # top_layout = QHBoxLayout(top_bar)
        # top_layout.setContentsMargins(12, 4, 12, 4)
        #
        # self.worker_judge = QLabel("就绪")
        # ...（整个方法体已注释）
        # self.stack.addWidget(page)
        # === END ===

    def _refresh_worker_scheme_list(self):
        # === COMMENTED OUT: 生产模式UI引用 ===
        # self.worker_scheme_list.clear()
        # os.makedirs(SCHEME_DIR, exist_ok=True)
        # for filename in sorted(os.listdir(SCHEME_DIR)):
        #     if filename.endswith(".json"):
        #         filepath = os.path.join(SCHEME_DIR, filename)
        #         name = os.path.splitext(filename)[0]
        #         item = QListWidgetItem(name)
        #         item.setData(Qt.UserRole, filepath)
        #         self.worker_scheme_list.addItem(item)
        # === END ===
        pass

    def _import_worker_scheme(self):
        # === COMMENTED OUT: 生产模式UI引用 ===
        # current_item = self.worker_scheme_list.currentItem()
        # if current_item is None:
        #     QMessageBox.warning(self, "提示", "请先在列表中选择一个方案")
        #     return
        #
        # name = current_item.text()
        # filepath = current_item.data(Qt.UserRole)
        #
        # try:
        #     with open(filepath, 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        #     pipeline = Pipeline.from_dict(data)
        # except Exception as e:
        #     QMessageBox.critical(self, "错误", f"加载方案文件失败:\n{e}")
        #     log_error(f"生产模式导入方案失败: {e}")
        #     return
        #
        # self.vision_engine.set_pipeline(pipeline)
        #
        # self.worker_scheme_label.setText(f"当前方案: {name}")
        # self.worker_status_label.setText(f"已导入方案: {name}")
        # self.worker_status_label.setStyleSheet("font-size: 18px; color: #66BB6A;")
        # self.worker_btn_detect.setEnabled(True)
        #
        # if name in self._schemes:
        #     self.eng_scheme_combo.setCurrentText(name)
        # self._current_scheme_name = name
        # self.scheme_status_label.setText(f"当前方案: {name}")
        # self.mode_scheme_label.setText(f"当前方案: {name}")
        #
        # log_info(f"生产模式导入方案: {name}")
        # QMessageBox.information(self, "成功", f"方案「{name}」已导入并应用")
        # self._update_auto_test_btn_state()
        # === END ===
        log_info("_import_worker_scheme 已禁用（生产模式已移除）")

    # ──────────────── 自动化模式页面 ────────────────

    def _build_automation_page(self):
        """构建自动化检测模式页面"""
        from ui.inspection_panel import InspectionPanel

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._inspection_panel = InspectionPanel(
            workflow=self._inspection_workflow,
            parent=page
        )
        layout.addWidget(self._inspection_panel, 1)

        self.stack.addWidget(page)

    # ──────────────── NMC 初始化 ────────────────

    def _init_nmc(self):
        """初始化 NMC 运动控制卡连接"""
        from core.nmc_sdk import Switch_State_Series
        self._nmc_sdk = None
        try:
            sdk = NMCSDK()
            sdk.load_dll()  # 失败会抛出异常
            sdk.set_switch_state(Switch_State_Series)
            sdk.connect()   # 失败会抛出异常
            self._nmc_sdk = sdk
            log_info("NMC 运动控制卡初始化成功")
        except Exception as e:
            log_warning(f"NMC 运动控制卡初始化失败: {e}，运动控制功能不可用")
            self._nmc_sdk = None

    def _init_inspection_workflow(self):
        """初始化自动化检测工作流"""
        self._inspection_workflow = InspectionWorkflow(
            nmc_sdk=self._nmc_sdk,
            camera_mgr=self.camera_mgr,
            vision_engine=self.vision_engine,
            parent=self
        )
        # 传入串口通信管理器（用于扫描头）
        if self._serial_comm is not None:
            self._inspection_workflow.set_serial_comm(self._serial_comm)

    def _build_engineer_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        scheme_bar = QWidget()
        scheme_bar.setStyleSheet("background-color: #252525; border: 1px solid #444; border-radius: 3px;")
        scheme_bar_layout = QHBoxLayout(scheme_bar)
        scheme_bar_layout.setContentsMargins(6, 2, 6, 2)
        scheme_bar_layout.setSpacing(4)

        lbl_scheme = QLabel("方案:")
        lbl_scheme.setStyleSheet("color: #d4d4d4;")
        scheme_bar_layout.addWidget(lbl_scheme)
        self.eng_scheme_combo = QComboBox()
        self.eng_scheme_combo.setMinimumWidth(180)
        self.eng_scheme_combo.setEditable(True)
        self.eng_scheme_combo.setInsertPolicy(QComboBox.NoInsert)
        self.eng_scheme_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4; border: 1px solid #555;
                padding: 4px 8px; border-radius: 3px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4; selection-background-color: #1a3a5c;
            }
        """)
        self.eng_scheme_combo.currentTextChanged.connect(self._on_scheme_combo_changed)
        self.eng_scheme_combo.lineEdit().editingFinished.connect(self._on_scheme_rename)

        self.eng_btn_new = QPushButton("新建")
        self.eng_btn_save = QPushButton("保存")
        self.eng_btn_apply = QPushButton("应用")
        self.eng_btn_rename = QPushButton("重命名")
        self.eng_btn_delete = QPushButton("删除")

        for btn in [self.eng_btn_new, self.eng_btn_save, self.eng_btn_apply, self.eng_btn_rename, self.eng_btn_delete]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4; padding: 2px 8px;
                    border: 1px solid #555; border-radius: 3px; font-size: 13px;
                }
                QPushButton:hover { background-color: #4a4a4a; }
            """)

        self.eng_btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #1a3a5c; color: #4A90D9; padding: 2px 10px;
                border: 1px solid #2a5a8c; border-radius: 3px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2a4a7c; }
        """)

        scheme_bar_layout.addWidget(self.eng_scheme_combo)
        scheme_bar_layout.addWidget(self.eng_btn_new)
        scheme_bar_layout.addWidget(self.eng_btn_save)
        scheme_bar_layout.addWidget(self.eng_btn_apply)
        scheme_bar_layout.addWidget(self.eng_btn_rename)
        scheme_bar_layout.addWidget(self.eng_btn_delete)
        scheme_bar_layout.addStretch()

        eng_splitter = QSplitter(Qt.Horizontal)

        left_eng_panel = QWidget()
        left_eng_layout = QVBoxLayout(left_eng_panel)
        left_eng_layout.setContentsMargins(0, 0, 0, 0)
        left_eng_layout.setSpacing(4)

        test_group = QGroupBox("测试图像")
        test_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 14px; border: 1px solid #444;
                border-radius: 4px; margin-top: 8px; padding-top: 12px; color: #d4d4d4;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #d4d4d4; }
        """)
        test_layout = QVBoxLayout(test_group)
        test_layout.setContentsMargins(2, 6, 2, 2)
        test_layout.setSpacing(2)

        test_toolbar = QHBoxLayout()
        self.eng_btn_run_preview = QPushButton("📷 测试")
        self.eng_btn_run_preview.setStyleSheet("""
            QPushButton {
                background-color: #1a3a5c; color: #4A90D9; padding: 2px 10px;
                border: 1px solid #2a5a8c; border-radius: 3px; font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2a4a7c; }
        """)
        test_toolbar.addWidget(self.eng_btn_run_preview)

        # 设计模式总测试时间显示
        self.eng_time_label = QLabel("")
        self.eng_time_label.setAlignment(Qt.AlignCenter)
        self.eng_time_label.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #4fc3f7;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 4px; padding: 1px 6px;
            min-width: 60px;
        """)
        test_toolbar.addWidget(self.eng_time_label)

        test_toolbar.addStretch()

        # 步骤导航栏
        step_nav_bar = QWidget()
        step_nav_bar.setStyleSheet("background-color: #252525; border: 1px solid #444; border-radius: 3px;")
        step_nav_layout = QHBoxLayout(step_nav_bar)
        step_nav_layout.setContentsMargins(4, 1, 4, 1)
        step_nav_layout.setSpacing(4)

        self.eng_btn_prev_step = QPushButton("◀ 上一步")
        self.eng_btn_prev_step.setEnabled(False)
        self.eng_btn_prev_step.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4; padding: 1px 6px;
                border: 1px solid #555; border-radius: 3px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)

        self.eng_step_label = QLabel("最终结果")
        self.eng_step_label.setAlignment(Qt.AlignCenter)
        self.eng_step_label.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #4A90D9;
            padding: 1px 6px; min-width: 80px;
        """)

        self.eng_btn_next_step = QPushButton("下一步 ▶")
        self.eng_btn_next_step.setEnabled(False)
        self.eng_btn_next_step.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4; padding: 1px 6px;
                border: 1px solid #555; border-radius: 3px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)

        step_nav_layout.addWidget(self.eng_btn_prev_step)
        step_nav_layout.addWidget(self.eng_step_label, 1)
        step_nav_layout.addWidget(self.eng_btn_next_step)

        self.eng_test_display = ZoomableImageWidget("点击「测试」按钮拍照并执行流水线")
        self.eng_test_display.setMinimumSize(240, 180)
        self.eng_test_display.label.setStyleSheet("""
            ZoomableLabel {
                background-color: #0d0d0d; border: 1px solid #444;
                border-radius: 4px;
            }
        """)

        test_layout.addLayout(test_toolbar)
        test_layout.addWidget(step_nav_bar)
        test_layout.addWidget(self.eng_test_display, 1)

        self.eng_result_panel = ResultPanel()
        self.eng_result_panel.setMaximumHeight(140)

        left_eng_layout.addWidget(test_group, 3)
        left_eng_layout.addWidget(self.eng_result_panel, 1)

        right_eng_panel = QWidget()
        right_eng_layout = QVBoxLayout(right_eng_panel)
        right_eng_layout.setContentsMargins(4, 0, 0, 0)
        right_eng_layout.setSpacing(2)

        self.eng_log = StepLogPanel()
        self.eng_log.setMaximumHeight(70)

        # 右侧标签页：流水线编辑 / 产品配置 / 轴控制
        self.eng_right_tabs = QTabWidget()
        self.eng_right_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444; background-color: #2d2d2d;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 3px 10px; border: 1px solid #444;
                border-bottom: none; border-top-left-radius: 3px;
                border-top-right-radius: 3px; font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d; color: #4A90D9;
                border-bottom: 1px solid #2d2d2d;
            }
            QTabBar::tab:hover { background-color: #4a4a4a; }
        """)

        # ── 标签页1: 流水线编辑 ──
        self.pipeline_editor = PipelineEditor()
        self.eng_right_tabs.addTab(self.pipeline_editor, "📋 流水线编辑")

        # ── 标签页2: 产品配置 ──
        self._build_product_config_tab()

        # ── 标签页3: 轴控制 ──
        self._build_axis_control_tab()

        right_eng_layout.addWidget(self.eng_log)
        right_eng_layout.addWidget(self.eng_right_tabs, 1)

        eng_splitter.addWidget(left_eng_panel)
        eng_splitter.addWidget(right_eng_panel)
        eng_splitter.setStretchFactor(0, 2)
        eng_splitter.setStretchFactor(1, 2)

        layout.addWidget(scheme_bar)
        layout.addWidget(eng_splitter, 1)

        self.eng_btn_new.clicked.connect(self._new_scheme)
        self.eng_btn_save.clicked.connect(self._save_current_scheme)
        self.eng_btn_apply.clicked.connect(self._apply_selected_scheme)
        self.eng_btn_rename.clicked.connect(self._rename_scheme)
        self.eng_btn_delete.clicked.connect(self._delete_scheme)
        self.eng_btn_run_preview.clicked.connect(self._run_preview)
        self.eng_btn_prev_step.clicked.connect(self._on_prev_step)
        self.eng_btn_next_step.clicked.connect(self._on_next_step)
        self.pipeline_editor.pipeline_changed.connect(self._on_editor_changed)

        self.stack.addWidget(page)

    # ──────────────── 产品配置标签页 ────────────────

    def _build_product_config_tab(self):
        """构建产品配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 产品列表
        list_label = QLabel("产品型号列表:")
        list_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #d4d4d4;")

        self._eng_product_list = QListWidget()
        self._eng_product_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #444; border-radius: 3px;
                font-size: 12px;
            }
            QListWidget::item { padding: 3px 6px; border-bottom: 1px solid #333; }
            QListWidget::item:selected { background-color: #1a3a5c; color: #4A90D9; }
            QListWidget::item:hover { background-color: #3a3a3a; }
        """)

        # 按钮组
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self._eng_btn_new_product = QPushButton("新建产品")
        self._eng_btn_edit_product = QPushButton("编辑")
        self._eng_btn_delete_product = QPushButton("删除")
        self._eng_btn_refresh_products = QPushButton("刷新")

        for btn in [self._eng_btn_new_product, self._eng_btn_edit_product,
                    self._eng_btn_delete_product, self._eng_btn_refresh_products]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4;
                    padding: 2px 6px; border: 1px solid #555;
                    border-radius: 3px; font-size: 11px;
                }
                QPushButton:hover { background-color: #4a4a4a; }
            """)

        btn_layout.addWidget(self._eng_btn_new_product)
        btn_layout.addWidget(self._eng_btn_edit_product)
        btn_layout.addWidget(self._eng_btn_delete_product)
        btn_layout.addWidget(self._eng_btn_refresh_products)
        btn_layout.addStretch()

        layout.addWidget(list_label)
        layout.addWidget(self._eng_product_list, 1)
        layout.addLayout(btn_layout)

        self.eng_right_tabs.addTab(tab, "📦 产品配置")

        # 连接信号
        self._eng_btn_new_product.clicked.connect(self._on_new_product)
        self._eng_btn_edit_product.clicked.connect(self._on_edit_product)
        self._eng_btn_delete_product.clicked.connect(self._on_delete_product)
        self._eng_btn_refresh_products.clicked.connect(self._refresh_product_list)

        # 初始加载
        self._refresh_product_list()

    def _refresh_product_list(self):
        """刷新产品列表"""
        self._eng_product_list.clear()
        try:
            products = list_products()
            for name in products:
                self._eng_product_list.addItem(name)
        except Exception as e:
            log_error(f"加载产品列表失败: {e}")

    def _on_new_product(self):
        """新建产品"""
        dialog = ProductConfigDialog(self, mode="new")
        if dialog.exec_() == QDialog.Accepted:
            self._refresh_product_list()
            # 通知自动化面板刷新
            if self._inspection_panel is not None:
                self._inspection_panel.refresh_products()

    def _on_edit_product(self):
        """编辑产品"""
        current = self._eng_product_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "提示", "请先选择一个产品")
            return
        name = current.text()
        dialog = ProductConfigDialog(self, mode="edit", product_name=name)
        if dialog.exec_() == QDialog.Accepted:
            self._refresh_product_list()
            if self._inspection_panel is not None:
                self._inspection_panel.refresh_products()

    def _on_delete_product(self):
        """删除产品"""
        current = self._eng_product_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "提示", "请先选择一个产品")
            return
        name = current.text()
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除产品「{name}」吗？\n此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            from core.product_manager import delete_product
            if delete_product(name):
                self._refresh_product_list()
                if self._inspection_panel is not None:
                    self._inspection_panel.refresh_products()
                log_info(f"已删除产品: {name}")
            else:
                QMessageBox.critical(self, "错误", f"删除产品「{name}」失败")

    # ──────────────── 轴控制标签页 ────────────────

    def _build_axis_control_tab(self):
        """构建轴控制标签页（工程师调试用）- 仅控制 Axis_2"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── 轴控制主分组 ──
        axis_group = QGroupBox("轴控制 (Axis_2)")
        axis_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 13px; border: 1px solid #444;
                border-radius: 4px; margin-top: 6px; padding-top: 10px; color: #d4d4d4;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        axis_layout = QVBoxLayout(axis_group)
        axis_layout.setSpacing(4)

        # 固定显示 Axis_2，无选择下拉框
        axis_sel_layout = QHBoxLayout()
        axis_label = QLabel("当前轴: Axis_2")
        axis_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4fc3f7; padding: 3px 6px;")
        axis_sel_layout.addWidget(axis_label)
        axis_sel_layout.addStretch()
        axis_layout.addLayout(axis_sel_layout)

        # 实时状态显示（命令位置、编码器、速度、轴状态）
        status_grid = QGridLayout()
        status_grid.setSpacing(2)

        status_grid.addWidget(QLabel("命令位置:"), 0, 0)
        self._eng_axis_pos_label = QLabel("--")
        self._eng_axis_pos_label.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #4fc3f7;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 1px 6px;
        """)
        status_grid.addWidget(self._eng_axis_pos_label, 0, 1)

        status_grid.addWidget(QLabel("编码器:"), 0, 2)
        self._eng_axis_enc_label = QLabel("--")
        self._eng_axis_enc_label.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #ff9800;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 1px 6px;
        """)
        status_grid.addWidget(self._eng_axis_enc_label, 0, 3)

        status_grid.addWidget(QLabel("速度:"), 1, 0)
        self._eng_axis_vel_label = QLabel("--")
        self._eng_axis_vel_label.setStyleSheet("""
            font-size: 13px; color: #d4d4d4;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 1px 6px;
        """)
        status_grid.addWidget(self._eng_axis_vel_label, 1, 1)

        status_grid.addWidget(QLabel("轴状态:"), 1, 2)
        self._eng_axis_state_label = QLabel("--")
        self._eng_axis_state_label.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #d4d4d4;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 1px 6px;
        """)
        status_grid.addWidget(self._eng_axis_state_label, 1, 3)

        axis_layout.addLayout(status_grid)

        # 速度/目标设置
        param_layout = QHBoxLayout()
        param_layout.setSpacing(4)
        param_layout.addWidget(QLabel("速度:"))
        self._eng_axis_speed = QLineEdit("10000")
        self._eng_axis_speed.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
        """)
        self._eng_axis_speed.setFixedWidth(60)
        param_layout.addWidget(self._eng_axis_speed)
        param_layout.addWidget(QLabel("加速度:"))
        self._eng_axis_acc = QLineEdit("10000")
        self._eng_axis_acc.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
        """)
        self._eng_axis_acc.setFixedWidth(60)
        param_layout.addWidget(self._eng_axis_acc)
        param_layout.addWidget(QLabel("目标位置:"))
        self._eng_axis_target = QLineEdit("0")
        self._eng_axis_target.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
        """)
        self._eng_axis_target.setFixedWidth(60)
        param_layout.addStretch()
        axis_layout.addLayout(param_layout)

        # ── 位置移动行（输入位置并移动到指定位置）──
        move_layout = QHBoxLayout()
        move_layout.setSpacing(4)
        move_layout.addWidget(QLabel("移动到位置:"))
        self._eng_axis_move_pos = QLineEdit("0")
        self._eng_axis_move_pos.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
        """)
        self._eng_axis_move_pos.setFixedWidth(80)
        move_layout.addWidget(self._eng_axis_move_pos)

        self._eng_btn_move_to = QPushButton("移动到")
        self._eng_btn_move_to.setStyleSheet("""
            QPushButton {
                background-color: #1565C0; color: #fff;
                padding: 3px 10px; border: 1px solid #1976D2;
                border-radius: 3px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        move_layout.addWidget(self._eng_btn_move_to)
        move_layout.addStretch()
        axis_layout.addLayout(move_layout)

        # 控制按钮行
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(4)

        self._eng_btn_jog_plus = QPushButton("JOG +")
        self._eng_btn_jog_plus.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; color: #fff;
                padding: 3px 8px; border: 1px solid #4CAF50;
                border-radius: 3px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:pressed { background-color: #1B5E20; }
        """)

        self._eng_btn_jog_minus = QPushButton("JOG -")
        self._eng_btn_jog_minus.setStyleSheet("""
            QPushButton {
                background-color: #C62828; color: #fff;
                padding: 3px 8px; border: 1px solid #EF5350;
                border-radius: 3px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #D32F2F; }
            QPushButton:pressed { background-color: #B71C1C; }
        """)

        self._eng_btn_stop_axis = QPushButton("停止")
        self._eng_btn_stop_axis.setStyleSheet("""
            QPushButton {
                background-color: #E65100; color: #fff;
                padding: 3px 8px; border: 1px solid #FF6D00;
                border-radius: 3px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #BF360C; }
        """)

        self._eng_btn_read_pos = QPushButton("读取位置")
        self._eng_btn_read_pos.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 3px 8px; border: 1px solid #555;
                border-radius: 3px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """)

        ctrl_layout.addWidget(self._eng_btn_jog_plus)
        ctrl_layout.addWidget(self._eng_btn_jog_minus)
        ctrl_layout.addWidget(self._eng_btn_stop_axis)
        ctrl_layout.addWidget(self._eng_btn_read_pos)
        axis_layout.addLayout(ctrl_layout)

        layout.addWidget(axis_group)

        # ── 回零控制分组 ──
        home_group = QGroupBox("回零控制 (Axis_2)")
        home_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 13px; border: 1px solid #444;
                border-radius: 4px; margin-top: 6px; padding-top: 10px; color: #d4d4d4;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        home_layout = QVBoxLayout(home_group)
        home_layout.setSpacing(3)

        # 回零参数行
        home_param_layout = QHBoxLayout()
        home_param_layout.setSpacing(4)
        home_param_layout.addWidget(QLabel("搜索速度:"))
        self._eng_home_speed = QLineEdit("10000")
        self._eng_home_speed.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
        """)
        self._eng_home_speed.setFixedWidth(60)
        home_param_layout.addWidget(self._eng_home_speed)
        home_param_layout.addWidget(QLabel("加速度:"))
        self._eng_home_acc = QLineEdit("10000")
        self._eng_home_acc.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 1px 4px; border-radius: 3px;
                font-size: 12px;
            }
        """)
        self._eng_home_acc.setFixedWidth(60)
        home_param_layout.addWidget(self._eng_home_acc)
        home_param_layout.addStretch()
        home_layout.addLayout(home_param_layout)

        # 回零状态显示
        home_status_layout = QHBoxLayout()
        home_status_layout.addWidget(QLabel("回零状态:"))
        self._eng_home_state_label = QLabel("就绪")
        self._eng_home_state_label.setStyleSheet("""
            font-size: 13px; font-weight: bold; color: #d4d4d4;
            background-color: #1e1e1e; border: 1px solid #444;
            border-radius: 3px; padding: 1px 6px;
        """)
        home_status_layout.addWidget(self._eng_home_state_label, 1)
        home_layout.addLayout(home_status_layout)

        # 回零进度条
        self._eng_home_progress = QProgressBar()
        self._eng_home_progress.setRange(0, 100)
        self._eng_home_progress.setValue(0)
        self._eng_home_progress.setTextVisible(True)
        self._eng_home_progress.setStyleSheet("""
            QProgressBar {
                background-color: #1e1e1e; border: 1px solid #444;
                border-radius: 3px; text-align: center; color: #d4d4d4;
                height: 14px;
            }
            QProgressBar::chunk {
                background-color: #4caf50; border-radius: 2px;
            }
        """)
        home_layout.addWidget(self._eng_home_progress)

        # 回零按钮行
        home_btn_layout = QHBoxLayout()
        home_btn_layout.setSpacing(4)

        self._eng_btn_home_start = QPushButton("开始回零")
        self._eng_btn_home_start.setStyleSheet("""
            QPushButton {
                background-color: #1a3a5c; color: #4A90D9;
                padding: 3px 10px; border: 1px solid #2a5a8c;
                border-radius: 3px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #2a4a7c; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)

        self._eng_btn_home_stop = QPushButton("停止回零")
        self._eng_btn_home_stop.setEnabled(False)
        self._eng_btn_home_stop.setStyleSheet("""
            QPushButton {
                background-color: #E65100; color: #fff;
                padding: 3px 10px; border: 1px solid #FF6D00;
                border-radius: 3px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #BF360C; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; border-color: #3a3a3a; }
        """)

        self._eng_btn_home_set_zero = QPushButton("设为零点")
        self._eng_btn_home_set_zero.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 3px 10px; border: 1px solid #555;
                border-radius: 3px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """)

        home_btn_layout.addWidget(self._eng_btn_home_start)
        home_btn_layout.addWidget(self._eng_btn_home_stop)
        home_btn_layout.addWidget(self._eng_btn_home_set_zero)
        home_btn_layout.addStretch()
        home_layout.addLayout(home_btn_layout)

        layout.addWidget(home_group)

        layout.addStretch()

        # ── 连接信号 ──
        self._eng_btn_move_to.clicked.connect(self._on_eng_axis_move_to)
        # JOG 使用 pressed/released 实现按压-保持行为
        self._eng_btn_jog_plus.pressed.connect(self._on_eng_axis_jog_plus)
        self._eng_btn_jog_plus.released.connect(self._on_eng_axis_jog_stop)
        self._eng_btn_jog_minus.pressed.connect(self._on_eng_axis_jog_minus)
        self._eng_btn_jog_minus.released.connect(self._on_eng_axis_jog_stop)
        self._eng_btn_stop_axis.clicked.connect(self._on_eng_axis_stop)
        self._eng_btn_read_pos.clicked.connect(self._on_eng_axis_read_pos)

        # 回零按钮信号
        self._eng_btn_home_start.clicked.connect(self._on_eng_home_start)
        self._eng_btn_home_stop.clicked.connect(self._on_eng_home_stop)
        self._eng_btn_home_set_zero.clicked.connect(self._on_eng_home_set_zero)

        # ── 实时监控定时器 (200ms) ──
        self._eng_axis_timer = QTimer(self)
        self._eng_axis_timer.timeout.connect(self._on_eng_axis_timer_tick)
        self._eng_axis_timer.start(200)

        # ── 回零状态变量 ──
        self._eng_homing = False
        self._eng_home_phase = 0
        self._eng_home_speed_level = 0
        self._eng_home_axis = 0
        self._eng_home_search_negative = True
        self._eng_home_has_reversed = False

        self.eng_right_tabs.addTab(tab, "🎮 轴控制")

    # ── 轴控制操作 ──

    def _get_axis_index(self) -> int:
        """固定返回 Axis_2 的轴索引（0-based = 1）"""
        return 1  # Axis_2

    def _on_eng_axis_move_to(self):
        """移动到指定位置（使用位置输入框的值）"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            QMessageBox.warning(self, "提示", "NMC 控制卡未连接")
            return
        try:
            target_text = self._eng_axis_move_pos.text().strip()
            if not target_text:
                QMessageBox.warning(self, "提示", "请输入目标位置")
                return
            target = int(target_text)
            speed = float(self._eng_axis_speed.text().strip())
            acc = float(self._eng_axis_acc.text().strip())
            axis = self._get_axis_index()

            from core.nmc_sdk import Profile_S, Position_Absolute

            # 读取当前位置
            try:
                current_pos = self._nmc_sdk.get_position(axis)
            except Exception:
                current_pos = 0
            log_info(f"轴{axis + 1} 当前位置: {current_pos}, 目标: {target}")

            # 1) 确保轴已使能
            try:
                self._nmc_sdk.set_servo_enable(axis, 1)
            except Exception as e:
                log_warning(f"轴{axis + 1} 使能失败(可忽略): {e}")

            # 2) 清除轴状态
            try:
                self._nmc_sdk.clear_axis_state(axis)
            except Exception:
                pass

            # 3) 检查轴当前状态
            try:
                axis_state = self._nmc_sdk.get_axis_state(axis)
                if axis_state != 0:
                    log_warning(f"轴{axis + 1} 当前状态: {axis_state}，尝试清除")
                    self._nmc_sdk.clear_axis_state(axis)
                    import time
                    time.sleep(0.1)
            except Exception:
                pass

            # 4) 设置曲线参数
            ret_profile = self._nmc_sdk.set_axis_profile(axis, 1000, speed, acc, acc, 0, Profile_S)
            log_info(f"轴{axis + 1} set_axis_profile 返回值: {ret_profile}")

            # 5) 使用相对定位模式（从当前位置移动差值）
            diff = target - current_pos
            log_info(f"轴{axis + 1} 尝试相对移动: dist={diff} (目标{target} - 当前位置{current_pos})")
            # 使用 uniaxial_long（独立 c_long 句柄），避免 argtypes(c_double) 冲突
            ret_move = self._nmc_sdk.uniaxial_long(axis, diff, 1)  # 相对模式
            log_info(f"轴{axis + 1} uniaxial_long(相对, dist={diff}) 返回值: {ret_move}")

            if ret_move != 0:
                log_error(f"轴{axis + 1} 相对移动 {diff} 失败，返回值: {ret_move}")
                QMessageBox.warning(self, "移动失败", f"轴移动失败，错误码: {ret_move}")
            else:
                # 等待运动完成（最多等待 5 秒）
                import time
                wait_start = time.time()
                moved = False
                poll_count = 0
                while time.time() - wait_start < 5.0:
                    try:
                        state = self._nmc_sdk.get_axis_state(axis)
                        poll_count += 1
                        if poll_count <= 3 or (poll_count % 20 == 0):
                            log_info(f"轴{axis + 1} 轮询状态: state={state} (第{poll_count}次)")
                        if state == 0:  # 空闲 = 到位
                            moved = True
                            break
                        elif state > 1:  # 非空闲非执行中 = 停止（可能因限位/报警停止）
                            log_warning(f"轴{axis + 1} 运动停止，状态码: {state}")
                            break
                    except Exception as e:
                        log_warning(f"轴{axis + 1} 读取状态异常: {e}")
                    time.sleep(0.05)

                if not moved:
                    log_warning(f"轴{axis + 1} 等待运动完成超时 (共轮询{poll_count}次)")

                # 读取移动后的位置
                try:
                    pos_after = self._nmc_sdk.get_position(axis)
                    log_info(f"轴{axis + 1} 移动后位置: {pos_after}")
                except Exception:
                    pos_after = "?"

                log_info(f"轴{axis + 1} 相对移动 {diff} (速度: {speed}, 加速度: {acc}, 移动后:{pos_after})")
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的数值")
        except Exception as e:
            log_error(f"轴移动失败: {e}")

    def _on_eng_axis_jog_plus(self):
        """JOG 正向移动（按压触发）"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            return
        try:
            speed = float(self._eng_axis_speed.text())
            acc = float(self._eng_axis_acc.text())
            axis = self._get_axis_index()
            self._nmc_sdk.jog(axis, speed, acc)
            log_info(f"轴{axis + 1} JOG+ (速度: {speed})")
        except ValueError:
            pass
        except Exception as e:
            log_error(f"JOG+ 失败: {e}")

    def _on_eng_axis_jog_minus(self):
        """JOG 反向移动（按压触发）"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            return
        try:
            speed = float(self._eng_axis_speed.text())
            acc = float(self._eng_axis_acc.text())
            axis = self._get_axis_index()
            self._nmc_sdk.jog(axis, -speed, acc)
            log_info(f"轴{axis + 1} JOG- (速度: {speed})")
        except ValueError:
            pass
        except Exception as e:
            log_error(f"JOG- 失败: {e}")

    def _on_eng_axis_jog_stop(self):
        """JOG 停止（释放按钮触发）"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            return
        try:
            axis = self._get_axis_index()
            self._nmc_sdk.axis_stop(axis, Stop_Smooth)
            log_info(f"轴{axis + 1} JOG 停止")
        except Exception as e:
            log_error(f"JOG 停止失败: {e}")

    def _on_eng_axis_stop(self):
        """停止轴"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            return
        try:
            axis = self._get_axis_index()
            self._nmc_sdk.axis_stop(axis, Stop_Smooth)
            log_info(f"轴{axis + 1} 已停止")
        except Exception as e:
            log_error(f"停止轴失败: {e}")

    def _on_eng_axis_read_pos(self):
        """读取当前位置"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            QMessageBox.warning(self, "提示", "NMC 控制卡未连接")
            return
        try:
            axis = self._get_axis_index()
            pos = self._nmc_sdk.get_position(axis)
            self._eng_axis_pos_label.setText(str(pos))
            log_info(f"轴{axis + 1} 当前位置: {pos}")
        except Exception as e:
            log_error(f"读取位置失败: {e}")

    # ── 实时监控定时器 ──

    def _on_eng_axis_timer_tick(self):
        """定时刷新轴状态显示（200ms）"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            return
        try:
            axis = self._get_axis_index()

            # 读取命令位置
            pos = self._nmc_sdk.get_position(axis)
            self._eng_axis_pos_label.setText(str(pos))

            # 读取编码器位置
            try:
                enc = self._nmc_sdk.get_encoder(axis)
                self._eng_axis_enc_label.setText(str(enc))
            except Exception:
                self._eng_axis_enc_label.setText("--")

            # 读取速度
            try:
                vel = self._nmc_sdk.get_velocity(axis)
                if isinstance(vel, (tuple, list)):
                    vel_str = f"{vel[0]:.1f}"
                else:
                    vel_str = f"{float(vel):.1f}"
                self._eng_axis_vel_label.setText(vel_str)
            except Exception:
                self._eng_axis_vel_label.setText("--")

            # 读取轴状态
            try:
                state = self._nmc_sdk.get_axis_state(axis)
                state_texts = {0: "停止", 1: "运动中", 2: "暂停"}
                state_str = state_texts.get(state, f"未知({state})")
                self._eng_axis_state_label.setText(state_str)
                # 根据状态设置颜色
                if state == 1:
                    self._eng_axis_state_label.setStyleSheet("""
                        font-size: 16px; font-weight: bold; color: #4caf50;
                        background-color: #1e1e1e; border: 1px solid #444;
                        border-radius: 3px; padding: 2px 8px;
                    """)
                else:
                    self._eng_axis_state_label.setStyleSheet("""
                        font-size: 16px; font-weight: bold; color: #d4d4d4;
                        background-color: #1e1e1e; border: 1px solid #444;
                        border-radius: 3px; padding: 2px 8px;
                    """)
            except Exception:
                self._eng_axis_state_label.setText("--")

        except Exception:
            pass

    # ── 回零控制 ──

    def _on_eng_home_start(self):
        """开始回零 - 双向搜索原点信号（三阶段速度控制）"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            QMessageBox.warning(self, "提示", "NMC 控制卡未连接")
            return

        axis = self._get_axis_index()

        # 如果正在回零，先停止
        if self._eng_homing:
            self._on_eng_home_stop()

        try:
            low_speed = int(self._eng_home_speed.text())
            acc = int(self._eng_home_acc.text())

            # 读取当前位置
            current_pos = self._nmc_sdk.get_position(axis)
            log_info(f"轴{axis + 1} 当前位置: {current_pos}")

            # 读取原点信号状态 (0=OFF=检测到, 1=ON=未检测到)
            home_signal = self._nmc_sdk.get_home(axis)
            home_detected = (home_signal == 0)
            log_info(f"轴{axis + 1} 原点信号: {'检测到(OFF)' if home_detected else '未检测到(ON)'}")

            # 设置软限位（确保限位保护生效）
            if axis in (0, 1, 2):
                pos_limit = 2000
                neg_limit = -60000
            else:
                pos_limit = 1000000
                neg_limit = -1000000

            log_info(f"轴{axis + 1} 设置软限位: +{pos_limit}, {neg_limit}")
            self._nmc_sdk.set_soft_limit(axis, pos_limit, neg_limit)
            self._nmc_sdk.set_soft_limit_enable(axis, 1)

            # 如果已经检测到原点信号，先向负方向离开原点
            if home_detected:
                log_info(f"轴{axis + 1} 已在原点位置，向负方向离开再回零")
                self._nmc_sdk.jog(axis, float(-low_speed), float(acc))
                # 等待原点信号消失（最多等待 3 秒）
                wait_start = time.time()
                while time.time() - wait_start < 3.0:
                    sig = self._nmc_sdk.get_home(axis)
                    if sig != 0:
                        break
                    time.sleep(0.05)
                self._nmc_sdk.axis_stop(axis, Stop_Abrupt)
                time.sleep(0.2)
                self._nmc_sdk.clear_axis_state(axis)
                # 再向负方向多走一点
                self._nmc_sdk.jog(axis, float(-low_speed), float(acc))
                time.sleep(0.2)
                self._nmc_sdk.axis_stop(axis, Stop_Abrupt)
                time.sleep(0.2)
                self._nmc_sdk.clear_axis_state(axis)

            # 根据是否离开过原点决定搜索方向
            if home_detected:
                # 已离开原点，向正方向搜索
                log_info(f"轴{axis + 1} 开始回零: 向正方向搜索原点")
                self._nmc_sdk.jog(axis, float(low_speed), float(acc))
                self._eng_home_search_negative = False
            else:
                # 正常情况，向负方向搜索
                log_info(f"轴{axis + 1} 开始回零: 向负方向搜索原点")
                self._nmc_sdk.jog(axis, float(-low_speed), float(acc))
                self._eng_home_search_negative = True

            # 回零状态机初始化
            self._eng_home_phase = 1
            self._eng_home_speed_level = 0
            self._eng_home_has_reversed = False
            self._eng_homing = True
            self._eng_home_axis = axis

            # 更新UI
            direction = "负方向" if self._eng_home_search_negative else "正方向"
            self._eng_home_state_label.setText(f"正在搜索原点 ({direction})...")
            self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self._eng_btn_home_start.setEnabled(False)
            self._eng_btn_home_stop.setEnabled(True)
            self._eng_home_progress.setValue(0)

            # 启动回零监测定时器 (100ms)
            self._eng_home_timer = QTimer(self)
            self._eng_home_timer.timeout.connect(self._on_eng_home_monitor_tick)
            self._eng_home_timer.start(100)
            log_info(f"轴{axis + 1} 回零开始")

        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的回零参数")
        except Exception as e:
            log_error(f"回零启动失败: {e}")
            QMessageBox.critical(self, "回零错误", f"启动回零失败: {e}")

    def _on_eng_home_stop(self):
        """停止回零运动"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            return
        try:
            axis = self._get_axis_index()
            self._nmc_sdk.axis_stop(axis, Stop_Smooth)
            if hasattr(self, '_eng_home_timer') and self._eng_home_timer is not None:
                self._eng_home_timer.stop()
                self._eng_home_timer = None
            self._eng_btn_home_start.setEnabled(True)
            self._eng_btn_home_stop.setEnabled(False)
            self._eng_home_phase = 0
            self._eng_home_speed_level = 0
            self._eng_homing = False
            pos = self._nmc_sdk.get_position(axis)
            self._eng_home_state_label.setText("已手动停止")
            self._eng_home_state_label.setStyleSheet("color: #f44336;")
            self._eng_home_progress.setValue(0)
            log_info(f"轴{axis + 1} 回零已手动停止，位置: {pos}")
        except Exception as e:
            log_error(f"停止回零失败: {e}")

    def _on_eng_home_set_zero(self):
        """将当前位置设为零点"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            QMessageBox.warning(self, "提示", "NMC 控制卡未连接")
            return
        try:
            axis = self._get_axis_index()
            pos = self._nmc_sdk.get_position(axis)
            self._nmc_sdk.set_position(axis, 0)
            self._nmc_sdk.set_encoder(axis, 0)
            self._eng_home_state_label.setText(f"已设为零点 (原位置: {pos})")
            self._eng_home_state_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            log_info(f"轴{axis + 1} 当前位置 {pos} 已设为零点")
        except Exception as e:
            log_error(f"设为零点失败: {e}")

    def _on_eng_home_monitor_tick(self):
        """回零监测定时器 - 三阶段速度控制"""
        if self._nmc_sdk is None or not self._nmc_sdk.is_open():
            if hasattr(self, '_eng_home_timer') and self._eng_home_timer is not None:
                self._eng_home_timer.stop()
                self._eng_home_timer = None
            self._eng_btn_home_start.setEnabled(True)
            self._eng_btn_home_stop.setEnabled(False)
            return
        try:
            axis = self._eng_home_axis
            low_speed = int(self._eng_home_speed.text())
            acc = int(self._eng_home_acc.text())

            # 读取当前位置
            pos = self._nmc_sdk.get_position(axis)

            # 读取原点信号状态 (0=OFF=检测到, 1=ON=未检测到)
            home_signal = self._nmc_sdk.get_home(axis)
            home_detected = (home_signal == 0)

            # 读取轴状态
            axis_state = self._nmc_sdk.get_axis_state(axis)
            is_moving = (axis_state == 1)

            HIGH_ACC = 100000

            # ============================================================
            # 阶段3：精定位阶段（反向极低速寻找原点边缘）
            # ============================================================
            if self._eng_home_phase == 3:
                if home_detected:
                    log_info(f"轴{axis + 1} 阶段3: 检测到原点信号! 停止轴并设为零点")
                    self._nmc_sdk.jog(axis, 0.0, float(HIGH_ACC))
                    time.sleep(0.3)

                    if hasattr(self, '_eng_home_timer') and self._eng_home_timer is not None:
                        self._eng_home_timer.stop()
                        self._eng_home_timer = None
                    self._eng_btn_home_start.setEnabled(True)
                    self._eng_btn_home_stop.setEnabled(False)

                    stop_pos = self._nmc_sdk.get_position(axis)
                    try:
                        self._nmc_sdk.set_position(axis, 0)
                        self._nmc_sdk.set_encoder(axis, 0)
                        log_info(f"轴{axis + 1} 已将原点边缘位置 {stop_pos} 设为0点")
                    except Exception as e:
                        log_error(f"轴{axis + 1} 设为零点失败: {e}")

                    self._eng_home_state_label.setText(f"轴{axis + 1} 回零完成 ✓")
                    self._eng_home_state_label.setStyleSheet("color: #4caf50; font-weight: bold;")
                    self._eng_home_progress.setValue(100)
                    log_info(f"轴{axis + 1} 回零成功! 原点边缘位置: {stop_pos} → 已设为0点")
                    self._eng_homing = False
                    self._eng_home_phase = 0
                    self._eng_home_speed_level = 0
                    return

                # 轴停止（碰到限位）→ 回零失败
                if not is_moving:
                    if hasattr(self, '_eng_home_timer') and self._eng_home_timer is not None:
                        self._eng_home_timer.stop()
                        self._eng_home_timer = None
                    self._eng_btn_home_start.setEnabled(True)
                    self._eng_btn_home_stop.setEnabled(False)
                    self._eng_home_state_label.setText(f"轴{axis + 1} 回零失败 ✗")
                    self._eng_home_state_label.setStyleSheet("color: #f44336; font-weight: bold;")
                    self._eng_home_progress.setValue(0)
                    log_error(f"轴{axis + 1} 回零失败: 精定位阶段轴停止 (state={axis_state})")
                    self._eng_homing = False
                    self._eng_home_phase = 0
                    self._eng_home_speed_level = 0
                    return

                self._eng_home_state_label.setText("回零状态: 精定位中，等待原点信号...")
                self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")
                return

            # ============================================================
            # 阶段2：减速逼近阶段（保持原方向，逐步降速）
            # ============================================================
            if self._eng_home_phase == 2:
                speed_levels = [
                    low_speed,
                    max(low_speed // 5, 1),
                    max(low_speed // 10, 1),
                    max(low_speed // 20, 1),
                ]

                # 先检查轴是否已停止
                if not is_moving:
                    if hasattr(self, '_eng_home_timer') and self._eng_home_timer is not None:
                        self._eng_home_timer.stop()
                        self._eng_home_timer = None
                    self._eng_btn_home_start.setEnabled(True)
                    self._eng_btn_home_stop.setEnabled(False)
                    self._eng_home_state_label.setText(f"轴{axis + 1} 回零失败 ✗")
                    self._eng_home_state_label.setStyleSheet("color: #f44336; font-weight: bold;")
                    self._eng_home_progress.setValue(0)
                    log_error(f"轴{axis + 1} 回零失败: 减速逼近阶段轴停止 (state={axis_state})")
                    self._eng_homing = False
                    self._eng_home_phase = 0
                    self._eng_home_speed_level = 0
                    return

                # 原点信号已消失 → 进入阶段3
                if not home_detected:
                    log_info(f"轴{axis + 1} 阶段2: 原点信号消失! 进入精定位阶段")
                    self._eng_home_phase = 3
                    self._eng_home_speed_level = 0
                    creep_direction = -1 if self._eng_home_search_negative else 1
                    creep_speed = max(low_speed // 50, 10)
                    self._nmc_sdk.jog(axis, float(creep_direction * creep_speed), float(HIGH_ACC))
                    log_info(f"轴{axis + 1} 阶段3: 反向精定位 speed={creep_speed}")
                    self._eng_home_state_label.setText("回零状态: 精定位中...")
                    self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")
                    return

                # 原点信号仍在 → 逐步降速
                if self._eng_home_speed_level < len(speed_levels) - 1:
                    self._eng_home_speed_level += 1
                    new_speed = speed_levels[self._eng_home_speed_level]
                    direction = 1 if self._eng_home_search_negative else -1
                    log_info(f"轴{axis + 1} 阶段2: 降速到 level={self._eng_home_speed_level}, speed={new_speed}")
                    self._nmc_sdk.jog(axis, float(direction * new_speed), float(HIGH_ACC))
                    self._eng_home_state_label.setText(f"回零状态: 减速逼近 ({self._eng_home_speed_level}/3)...")
                    self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")
                    return
                else:
                    self._eng_home_state_label.setText("回零状态: 最低速逼近中，等待信号消失...")
                    self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")
                    return

            # ============================================================
            # 阶段1：搜索阶段
            # ============================================================

            # 检测到原点信号 → 进入阶段2
            if home_detected:
                log_info(f"轴{axis + 1} 阶段1: 检测到原点信号! 进入减速逼近阶段")
                self._eng_home_phase = 2
                self._eng_home_speed_level = 1
                new_speed = max(low_speed // 5, 1)
                direction = 1 if self._eng_home_search_negative else -1
                self._nmc_sdk.jog(axis, float(direction * new_speed), float(HIGH_ACC))
                log_info(f"轴{axis + 1} 阶段2: 降速到 {new_speed}")
                self._eng_home_state_label.setText("回零状态: 减速逼近 (1/3)...")
                self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")
                return

            # 轴仍在运动中
            if is_moving:
                direction = "负方向" if self._eng_home_search_negative else "正方向"
                self._eng_home_state_label.setText(f"回零状态: 搜索原点 ({direction})...")
                self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")
                return

            # 轴已停止（碰到限位）
            log_info(f"轴{axis + 1} 轴停止: state={axis_state}，位置: {pos}")
            self._nmc_sdk.clear_axis_state(axis)
            time.sleep(0.05)

            # 如果已经反向过一次 → 回零失败
            if self._eng_home_has_reversed:
                if hasattr(self, '_eng_home_timer') and self._eng_home_timer is not None:
                    self._eng_home_timer.stop()
                    self._eng_home_timer = None
                self._eng_btn_home_start.setEnabled(True)
                self._eng_btn_home_stop.setEnabled(False)
                self._eng_home_state_label.setText(f"轴{axis + 1} 回零失败 ✗")
                self._eng_home_state_label.setStyleSheet("color: #f44336; font-weight: bold;")
                self._eng_home_progress.setValue(0)
                log_error(f"轴{axis + 1} 回零失败: 两个方向均未找到原点信号")
                self._eng_homing = False
                self._eng_home_phase = 0
                return

            # 切换方向继续搜索
            self._eng_home_has_reversed = True
            self._eng_home_search_negative = not self._eng_home_search_negative
            self._nmc_sdk.set_soft_limit_enable(axis, 1)
            time.sleep(0.02)

            if self._eng_home_search_negative:
                log_info(f"轴{axis + 1} 碰到正限位，切换向负方向搜索")
                self._nmc_sdk.jog(axis, float(-low_speed), float(acc))
                self._eng_home_state_label.setText("回零状态: 搜索原点 (负方向)...")
            else:
                log_info(f"轴{axis + 1} 碰到负限位，切换向正方向搜索")
                self._nmc_sdk.jog(axis, float(low_speed), float(acc))
                self._eng_home_state_label.setText("回零状态: 搜索原点 (正方向)...")

            self._eng_home_state_label.setStyleSheet("color: #ff9800; font-weight: bold;")

        except Exception as e:
            log_error(f"回零监测异常: {e}")

    def _setup_menu_bar(self):
        menubar = self.menuBar()

        device_menu = menubar.addMenu("设备")
        self.act_open_camera = QAction("相机设置", self)
        self.act_open_camera.triggered.connect(self._open_camera_dialog)
        self.act_close_camera = QAction("关闭相机", self)
        self.act_close_camera.setEnabled(False)
        self.act_close_camera.triggered.connect(self._close_camera)
        self.act_capture = QAction("拍照", self)
        self.act_capture.setShortcut(QKeySequence("F5"))
        self.act_capture.setEnabled(False)
        self.act_capture.triggered.connect(self._capture)
        self.act_load_image = QAction("导入图像", self)
        self.act_load_image.setShortcut(QKeySequence("Ctrl+O"))
        self.act_load_image.triggered.connect(self._load_image)

        device_menu.addAction(self.act_open_camera)
        device_menu.addAction(self.act_close_camera)
        device_menu.addSeparator()
        device_menu.addAction(self.act_capture)
        device_menu.addAction(self.act_load_image)

        scheme_menu = menubar.addMenu("方案")
        self.act_new_scheme = QAction("新建方案", self)
        self.act_new_scheme.triggered.connect(self._new_scheme)
        self.act_save_scheme = QAction("保存方案", self)
        self.act_save_scheme.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_scheme.triggered.connect(self._save_current_scheme)
        self.act_apply_scheme = QAction("应用方案", self)
        self.act_apply_scheme.triggered.connect(self._apply_selected_scheme)
        scheme_menu.addAction(self.act_new_scheme)
        scheme_menu.addAction(self.act_save_scheme)
        scheme_menu.addAction(self.act_apply_scheme)
        scheme_menu.addSeparator()

        self.act_import_scheme = QAction("导入方案", self)
        self.act_import_scheme.triggered.connect(self._import_scheme)
        self.act_export_scheme = QAction("导出方案", self)
        self.act_export_scheme.triggered.connect(self._export_scheme)
        scheme_menu.addAction(self.act_import_scheme)
        scheme_menu.addAction(self.act_export_scheme)

        comm_menu = menubar.addMenu("通信")
        self.act_serial_comm = QAction("串口通信", self)
        self.act_serial_comm.triggered.connect(self._open_serial_dialog)
        comm_menu.addAction(self.act_serial_comm)
        self.act_nmc_control = QAction("运动控制", self)
        self.act_nmc_control.triggered.connect(self._open_nmc_dialog)
        comm_menu.addAction(self.act_nmc_control)

        # ── 用户菜单 ──
        user_menu = menubar.addMenu("用户")
        self.act_current_user = QAction("当前用户：工程师", self)
        self.act_current_user.setEnabled(True)
        self.act_switch_user = QAction("切换用户...", self)
        self.act_switch_user.triggered.connect(self._show_login_dialog)
        self.act_logout = QAction("退出登录", self)
        self.act_logout.setEnabled(True)
        self.act_logout.triggered.connect(self._logout)
        user_menu.addAction(self.act_current_user)
        user_menu.addSeparator()
        user_menu.addAction(self.act_switch_user)
        user_menu.addAction(self.act_logout)

        sys_menu = menubar.addMenu("系统")
        self.act_log_settings = QAction("日志限额设置", self)
        self.act_log_settings.triggered.connect(self._show_log_settings)
        sys_menu.addAction(self.act_log_settings)

        help_menu = menubar.addMenu("帮助")
        self.act_about = QAction("关于", self)
        self.act_about.triggered.connect(self._show_about)
        help_menu.addAction(self.act_about)

    def _load_schemes(self):
        os.makedirs(SCHEME_DIR, exist_ok=True)
        self._schemes = {}
        self.eng_scheme_combo.clear()

        for filename in sorted(os.listdir(SCHEME_DIR)):
            if filename.endswith(".json"):
                filepath = os.path.join(SCHEME_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    pipeline = Pipeline.from_dict(data)
                    name = pipeline.name or os.path.splitext(filename)[0]
                    self._schemes[name] = (pipeline, filepath)
                    self.eng_scheme_combo.addItem(name)
                except Exception as e:
                    log_error(f"加载方案失败 {filename}: {e}")

        self._refresh_worker_scheme_list()

    def _auto_load_default_scheme(self):
        """启动时自动加载名为'默认方案'的方案"""
        default_name = "默认方案"
        if default_name in self._schemes:
            # 直接设置 combo box 选中项，触发 _on_scheme_combo_changed
            idx = self.eng_scheme_combo.findText(default_name)
            if idx >= 0:
                self.eng_scheme_combo.setCurrentIndex(idx)
            # 同时应用到引擎
            pipeline, _ = self._schemes[default_name]
            self.vision_engine.set_pipeline(pipeline)
            self._current_scheme_name = default_name
            self.scheme_status_label.setText(f"当前方案: {default_name}")
            # === COMMENTED OUT: 生产模式UI引用 ===
            # self.worker_scheme_label.setText(f"当前方案: {default_name}")
            # === END ===
            self.mode_scheme_label.setText(f"当前方案: {default_name}")
            self.status_label.setText(f"已自动加载方案: {default_name}")
            log_info(f"自动加载默认方案: {default_name}")

    def _on_scheme_combo_changed(self, name):
        if not name:
            return
        self._current_scheme_name = name
        pipeline, _ = self._schemes.get(name, (None, None))
        if pipeline is not None:
            self.pipeline_editor.set_pipeline(pipeline)
            self.scheme_status_label.setText(f"当前方案: {name}")
            # === COMMENTED OUT: 生产模式UI引用 ===
            # self.worker_scheme_label.setText(f"当前方案: {name}")
            # === END ===
            self.mode_scheme_label.setText(f"当前方案: {name}")

    def _apply_selected_scheme(self):
        if not self._current_scheme_name:
            QMessageBox.warning(self, "提示", "请先选择一个方案")
            return

        pipeline, _ = self._schemes.get(self._current_scheme_name, (None, None))
        if pipeline is None:
            QMessageBox.warning(self, "错误", "方案数据异常")
            return

        self.vision_engine.set_pipeline(pipeline)
        self.scheme_status_label.setText(f"当前方案: {self._current_scheme_name}")
        # === COMMENTED OUT: 生产模式UI引用 ===
        # self.worker_scheme_label.setText(f"当前方案: {self._current_scheme_name}")
        # === END ===
        self.mode_scheme_label.setText(f"当前方案: {self._current_scheme_name}")
        self.status_label.setText(f"已应用方案: {self._current_scheme_name}")
        log_info(f"应用方案: {self._current_scheme_name}")

    def _new_scheme(self):
        name, ok = QInputDialog.getText(self, "新建方案", "请输入方案名称:")
        if ok and name.strip():
            if name in self._schemes:
                QMessageBox.warning(self, "提示", "方案已存在")
                return
            pipeline = Pipeline(name=name.strip())
            self._schemes[name] = (pipeline, None)
            self.eng_scheme_combo.addItem(name)
            self.eng_scheme_combo.setCurrentText(name)
            self._current_scheme_name = name
            self.pipeline_editor.set_pipeline(pipeline)
            log_info(f"新建方案: {name}")
            self._refresh_worker_scheme_list()

    def _delete_scheme(self):
        if not self._current_scheme_name:
            QMessageBox.warning(self, "提示", "请先选择一个方案")
            return

        name = self._current_scheme_name
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定删除方案 '{name}' 吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            pipeline, filepath = self._schemes.get(name, (None, None))
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            del self._schemes[name]
            idx = self.eng_scheme_combo.findText(name)
            if idx >= 0:
                self.eng_scheme_combo.removeItem(idx)
            self._current_scheme_name = None
            log_info(f"删除方案: {name}")
            self._refresh_worker_scheme_list()

    def _rename_scheme(self):
        if not self._current_scheme_name:
            QMessageBox.warning(self, "提示", "请先选择一个方案")
            return

        old_name = self._current_scheme_name
        new_name, ok = QInputDialog.getText(self, "重命名方案", "请输入新名称:",
                                             text=old_name)
        if ok and new_name.strip() and new_name.strip() != old_name:
            self._do_rename_scheme(old_name, new_name.strip())

    def _on_scheme_rename(self):
        if not self._current_scheme_name:
            return
        new_name = self.eng_scheme_combo.currentText().strip()
        if not new_name or new_name == self._current_scheme_name:
            self.eng_scheme_combo.blockSignals(True)
            self.eng_scheme_combo.setCurrentText(self._current_scheme_name)
            self.eng_scheme_combo.blockSignals(False)
            return
        self._do_rename_scheme(self._current_scheme_name, new_name)

    def _do_rename_scheme(self, old_name: str, new_name: str):
        if new_name in self._schemes and new_name != old_name:
            QMessageBox.warning(self, "提示", "方案名已存在")
            self.eng_scheme_combo.blockSignals(True)
            self.eng_scheme_combo.setCurrentText(old_name)
            self.eng_scheme_combo.blockSignals(False)
            return

        pipeline, filepath = self._schemes.pop(old_name)
        pipeline.name = new_name

        if filepath and os.path.exists(filepath):
            try:
                new_filepath = os.path.join(os.path.dirname(filepath), f"{new_name}.json")
                os.rename(filepath, new_filepath)
                filepath = new_filepath
            except Exception:
                pass

        self._schemes[new_name] = (pipeline, filepath)

        idx = self.eng_scheme_combo.findText(old_name)
        if idx >= 0:
            self.eng_scheme_combo.blockSignals(True)
            self.eng_scheme_combo.setItemText(idx, new_name)
            self.eng_scheme_combo.setCurrentText(new_name)
            self.eng_scheme_combo.blockSignals(False)

        self._current_scheme_name = new_name
        self.scheme_status_label.setText(f"当前方案: {new_name}")
        # === COMMENTED OUT: 生产模式UI引用 ===
        # self.worker_scheme_label.setText(f"当前方案: {new_name}")
        # === END ===
        self.mode_scheme_label.setText(f"当前方案: {new_name}")
        log_info(f"重命名方案: {old_name} -> {new_name}")
        self._refresh_worker_scheme_list()

    def _import_scheme(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "导入方案", "", "方案文件 (*.json)")
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pipeline = Pipeline.from_dict(data)
            name = pipeline.name or os.path.splitext(os.path.basename(filepath))[0]

            dest_path = os.path.join(SCHEME_DIR, f"{name}.json")
            with open(dest_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            existing_idx = self.eng_scheme_combo.findText(name)
            if existing_idx >= 0:
                self.eng_scheme_combo.removeItem(existing_idx)

            self._schemes[name] = (pipeline, dest_path)
            self.eng_scheme_combo.addItem(name)
            self.eng_scheme_combo.setCurrentText(name)
            QMessageBox.information(self, "成功", f"方案 '{name}' 导入成功")
            log_info(f"导入方案: {name}")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def _export_scheme(self):
        if not self._current_scheme_name:
            QMessageBox.warning(self, "提示", "请先选择一个方案")
            return
        pipeline, _ = self._schemes.get(self._current_scheme_name, (None, None))
        if pipeline is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出方案", f"{self._current_scheme_name}.json", "方案文件 (*.json)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(pipeline.to_dict(), f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "成功", f"方案已导出到: {filepath}")
            log_info(f"导出方案: {self._current_scheme_name}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _save_current_scheme(self):
        if not self._current_scheme_name:
            QMessageBox.warning(self, "提示", "请先选择一个方案")
            return

        pipeline = self.pipeline_editor.get_pipeline()
        pipeline.name = self._current_scheme_name
        filepath = self._schemes.get(self._current_scheme_name, (None, None))[1]
        self._schemes[self._current_scheme_name] = (pipeline, filepath)

        os.makedirs(SCHEME_DIR, exist_ok=True)
        if filepath is None:
            filepath = os.path.join(SCHEME_DIR, f"{self._current_scheme_name}.json")
            self._schemes[self._current_scheme_name] = (pipeline, filepath)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(pipeline.to_dict(), f, indent=2, ensure_ascii=False)
            self.status_label.setText(f"方案已保存: {self._current_scheme_name}")
            log_info(f"保存方案: {self._current_scheme_name}")
            QMessageBox.information(self, "成功", f"方案 '{self._current_scheme_name}' 保存成功")
        except Exception as e:
            log_error(f"保存方案失败: {e}")
            QMessageBox.critical(self, "保存失败", str(e))

        self._refresh_worker_scheme_list()

    def _on_editor_changed(self):
        if self._current_scheme_name:
            pipeline = self.pipeline_editor.get_pipeline()
            filepath = self._schemes.get(self._current_scheme_name, (None, None))[1]
            self._schemes[self._current_scheme_name] = (pipeline, filepath)
            if self.vision_engine.pipeline is not None:
                self.vision_engine.set_pipeline(pipeline)

    def _init_sdk(self):
        try:
            CameraManager.initialize_sdk()
        except Exception as e:
            log_error(f"SDK初始化失败: {e}")

    def _auto_connect_camera(self):
        """启动时自动搜索并连接相机"""
        self.status_label.setText("正在自动连接相机...")
        log_info("启动自动连接相机...")

        # 创建一个临时的 CameraPanel 用于自动连接，共享 CameraManager 实例
        self._camera_panel = CameraPanel(camera_mgr=self.camera_mgr)
        self._camera_panel.frame_received.connect(self._on_frame_received)
        self._camera_panel.capture_completed.connect(self._on_capture_completed)
        self._camera_panel.status_message.connect(self._on_camera_status_message)

        # 自动枚举并连接
        self._camera_panel.auto_connect_camera()

    def _on_camera_status_message(self, message):
        """相机状态消息回调"""
        self.status_label.setText(message)
        # 如果相机已打开，更新 UI 状态
        if self._camera_panel is not None and self._camera_panel.is_camera_open():
            self.act_open_camera.setEnabled(False)
            self.act_close_camera.setEnabled(True)
            self.act_capture.setEnabled(True)
            self.status_label.setText("相机已自动连接 - " + message)
            log_info("相机自动连接成功")

    def _open_camera_dialog(self):
        # 如果相机已打开，仍然打开设置对话框以允许用户调节参数
        if self._camera_panel is not None and self._camera_panel.is_camera_open():
            dialog = QDialog(self)
            dialog.setWindowTitle("相机设置 - 参数调节")
            dialog.setMinimumWidth(600)
            dialog.setMinimumHeight(450)

            layout = QVBoxLayout(dialog)
            # 创建新的 CameraPanel 共享 camera_mgr，用于参数调节界面
            # 不重新连接 frame_received/capture_completed 信号，避免干扰主界面取流
            settings_panel = CameraPanel(camera_mgr=self.camera_mgr)
            settings_panel.status_message.connect(self._on_camera_status_message)
            layout.addWidget(settings_panel)

            # 更新 settings_panel 的 UI 状态以反映相机已打开
            settings_panel.open_btn.setEnabled(False)
            settings_panel.close_btn.setEnabled(True)
            settings_panel.capture_btn.setEnabled(True)
            settings_panel.trigger_combo.setEnabled(True)
            settings_panel.trigger_btn.setEnabled(self.camera_mgr.is_trigger_mode)
            # 刷新参数显示
            settings_panel._refresh_params()

            btn_close = QPushButton("关闭")
            btn_close.clicked.connect(dialog.accept)
            layout.addWidget(btn_close)

            dialog.exec_()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("相机控制")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(450)

        layout = QVBoxLayout(dialog)
        # 创建一个临时面板用于对话框，不赋值给 self._camera_panel
        # 避免 dialog 关闭后 Qt 自动销毁该面板导致 self._camera_panel 悬空
        panel = CameraPanel(camera_mgr=self.camera_mgr)
        panel.frame_received.connect(self._on_frame_received)
        panel.capture_completed.connect(self._on_capture_completed)
        panel.status_message.connect(self._on_camera_status_message)
        layout.addWidget(panel)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        panel.enumerate_devices()
        dialog.exec_()

    def _close_camera(self):
        try:
            if self._camera_panel is not None and self._camera_panel.is_camera_open():
                self._camera_panel.close_camera()
        except (RuntimeError, AttributeError):
            # Qt 对象已被删除，忽略
            pass
        self.act_open_camera.setEnabled(True)
        self.act_close_camera.setEnabled(False)
        self.act_capture.setEnabled(False)
        # === COMMENTED OUT: 生产模式UI引用 ===
        # self.worker_btn_detect.setEnabled(False)
        # self.worker_display.clear_pixmap()
        # self.worker_display.label.setText("相机已关闭")
        # === END ===
        self._raw_image = None
        self.status_label.setText("相机已关闭")

    def _capture(self):
        if self._camera_panel is not None:
            self._camera_panel.capture_once()

    def _load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "导入图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*.*)")
        if not filepath:
            return
        try:
            img = cv2.imread(filepath)
            if img is None:
                QMessageBox.warning(self, "导入失败", f"无法读取图像: {filepath}")
                return
            self._raw_image = img
            self._raw_height, self._raw_width = img.shape[:2]
            # === COMMENTED OUT: 生产模式UI引用 ===
            # # 导入新图像，清除上一次的检测标注结果
            # self._last_annotated = None
            # display_img = self._overlay_roi_on_image(img)
            # self._show_worker_image(display_img)
            # self.worker_btn_detect.setEnabled(True)
            # === END ===
            self.act_capture.setEnabled(True)
            self.status_label.setText(f"已导入图像: {os.path.basename(filepath)}")
            # === COMMENTED OUT: 生产模式UI引用 ===
            # self.worker_status_label.setText(f"已导入图像: {os.path.basename(filepath)}")
            # === END ===
            log_info(f"导入图像: {filepath} ({self._raw_width}x{self._raw_height})")
        except Exception as e:
            log_error(f"导入图像失败: {e}")
            QMessageBox.critical(self, "导入失败", str(e))

    def _on_frame_received(self, width, height, pixel_type, img_bytes):
        self._raw_width = width
        self._raw_height = height
        self._raw_image = self._convert_to_cv(width, height, pixel_type, img_bytes)
        if self._raw_image is not None:
            # === COMMENTED OUT: 生产模式UI引用 ===
            # # 如果有最近一次检测的标注结果，优先显示它（保持检测结果可见）
            # if self._last_annotated is not None:
            #     self._show_worker_image(self._last_annotated)
            # else:
            #     # 实时预览时，如果已设置流水线，在原始图像上叠加 ROI 框
            #     display_img = self._overlay_roi_on_image(self._raw_image)
            #     self._show_worker_image(display_img)
            # === END ===
            pass

    def _on_capture_completed(self, width, height, pixel_type, img_bytes):
        self._raw_width = width
        self._raw_height = height
        self._raw_image = self._convert_to_cv(width, height, pixel_type, img_bytes)
        # === COMMENTED OUT: 生产模式UI引用 ===
        # # 新拍照，清除上一次的检测标注结果
        # self._last_annotated = None
        # self.worker_btn_detect.setEnabled(True)
        # === END ===
        self.act_capture.setEnabled(True)
        self.act_open_camera.setEnabled(False)
        self.act_close_camera.setEnabled(True)
        if self._raw_image is not None:
            # === COMMENTED OUT: 生产模式UI引用 ===
            # # 拍照完成后，如果已设置流水线，在原始图像上叠加 ROI 框
            # display_img = self._overlay_roi_on_image(self._raw_image)
            # self._show_worker_image(display_img)
            # === END ===
            pass
        self.status_label.setText("拍照完成，可开始检测")
        # === COMMENTED OUT: 生产模式UI引用 ===
        # self.worker_status_label.setText("拍照完成，可开始检测")
        # === END ===

        # 串口自动测试工作流模式：将图像传递给工作流
        if (self._serial_workflow is not None
                and self._serial_workflow.is_running):
            self._serial_workflow.on_capture_completed(self._raw_image)
            return

        # 设计模式测试：拍照后自动执行流水线
        if self._pending_engineer_test:
            self._pending_engineer_test = False
            self._show_engineer_image(self._raw_image)
            self._execute_engineer_test()

        # === COMMENTED OUT: 生产模式自动检测 ===
        # # 生产模式：拍照后自动执行检测
        # if self._pending_detect:
        #     self._pending_detect = False
        #     self._do_detect()
        # === END ===

    def _overlay_roi_on_image(self, cv_img: np.ndarray) -> np.ndarray:
        """在图像上叠加流水线中 MultiROI 工具定义的 ROI 区域框（绿色边框）。

        用于实时预览时，让操作员看到检测区域的位置。
        如果未设置流水线或没有 MultiROI 工具，则返回原始图像的副本。
        """
        if cv_img is None:
            return cv_img
        pipeline = self.vision_engine.pipeline
        if pipeline is None:
            return cv_img.copy()

        result_img = cv_img.copy()
        h_img, w_img = result_img.shape[:2]

        for step in pipeline.steps:
            if not step.enabled:
                continue
            tool_type = type(step.tool).__name__
            if tool_type == "MultiROI":
                raw_regions = step.tool.params.get("regions", [])
                use_pct = step.tool.params.get("use_percentage", False)

                for r in raw_regions:
                    if isinstance(r, dict) and r.get("enabled", True):
                        name = r.get("name", "未命名")
                        if use_pct:
                            x = int(r.get("x", 0) / 100.0 * w_img)
                            y = int(r.get("y", 0) / 100.0 * h_img)
                            w = int(r.get("width", r.get("w", 100)) / 100.0 * w_img)
                            h = int(r.get("height", r.get("h", 100)) / 100.0 * h_img)
                        else:
                            x = r.get("x", 0)
                            y = r.get("y", 0)
                            w = r.get("width", r.get("w", 100))
                            h = r.get("height", r.get("h", 100))

                        # 绘制绿色 ROI 框（预览时统一绿色，无检测结果）
                        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_img, name, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return result_img

    def _cv_to_pixmap(self, cv_img) -> QPixmap:
        """将 OpenCV 图像转换为 QPixmap"""
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = cv_img.shape
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def _show_cv_image(self, cv_img):
        """通用图像显示（仅更新 Engineer 显示区，Worker 显示区已移除）"""
        if cv_img is None:
            return
        try:
            pix = self._cv_to_pixmap(cv_img)
            # === COMMENTED OUT: 生产模式UI引用 ===
            # self.worker_display.update_pixmap(pix)
            # === END ===
            self.eng_test_display.update_pixmap(pix)
        except Exception as e:
            # === COMMENTED OUT: 生产模式UI引用 ===
            # self.worker_display.label.setText(f"图像显示错误: {e}")
            # === END ===
            self.eng_test_display.label.setText(f"图像显示错误: {e}")

    def _show_worker_image(self, cv_img):
        """Worker 模式：显示原始图像 + 标注叠加（已禁用，生产模式已移除）"""
        # === COMMENTED OUT: 生产模式UI引用 ===
        # if cv_img is None:
        #     return
        # try:
        #     pix = self._cv_to_pixmap(cv_img)
        #     self.worker_display.update_pixmap(pix)
        # except Exception as e:
        #     self.worker_display.label.setText(f"图像显示错误: {e}")
        # === END ===
        pass

    def _show_engineer_image(self, cv_img):
        """Engineer 模式：显示原始图像 + 标注叠加（仅更新 eng_test_display）"""
        if cv_img is None:
            return
        try:
            pix = self._cv_to_pixmap(cv_img)
            self.eng_test_display.update_pixmap(pix)
        except Exception as e:
            self.eng_test_display.label.setText(f"图像显示错误: {e}")

    # ────────── 步骤导航 ──────────

    def _update_step_nav_buttons(self):
        """根据当前步骤索引更新导航按钮状态和标签"""
        total = len(self._step_results)
        if total == 0:
            self.eng_btn_prev_step.setEnabled(False)
            self.eng_btn_next_step.setEnabled(False)
            self.eng_step_label.setText("最终结果")
            return

        idx = self._current_step_index
        if idx < 0:
            # 显示最终标注结果
            self.eng_step_label.setText(f"最终结果 ({total} 步)")
            self.eng_btn_prev_step.setEnabled(total > 0)
            self.eng_btn_next_step.setEnabled(total > 0)
        else:
            r = self._step_results[idx]
            name = r.tool_name or r.tool_type or f"步骤{idx+1}"
            status = "✓" if r.passed else "✗"
            self.eng_step_label.setText(f"步骤{idx+1}: {name} {status}")
            self.eng_btn_prev_step.setEnabled(idx > 0)
            self.eng_btn_next_step.setEnabled(idx < total - 1)

    def _show_step_image(self, index: int):
        """显示指定步骤的图像：index=-1 显示最终标注结果，否则显示该步骤的 processed_image"""
        if index < 0:
            # 显示最终标注结果（原始图 + 所有 overlay 叠加）
            if self._annotated_image is not None:
                self._show_engineer_image(self._annotated_image)
            elif self._raw_image is not None:
                self._show_engineer_image(self._raw_image)
        elif 0 <= index < len(self._step_results):
            r = self._step_results[index]
            if r.processed_image is not None:
                self._show_engineer_image(r.processed_image)
            elif self._raw_image is not None:
                self._show_engineer_image(self._raw_image)

    def _on_prev_step(self):
        """上一步"""
        if self._current_step_index < 0:
            # 当前显示最终结果，跳到最后一步
            self._current_step_index = len(self._step_results) - 1
        else:
            self._current_step_index -= 1
        self._show_step_image(self._current_step_index)
        self._update_step_nav_buttons()

    def _on_next_step(self):
        """下一步"""
        total = len(self._step_results)
        if self._current_step_index < 0:
            # 从最终结果跳到第一步
            self._current_step_index = 0
        elif self._current_step_index < total - 1:
            self._current_step_index += 1
        else:
            # 已到最后一步，回到最终结果
            self._current_step_index = -1
        self._show_step_image(self._current_step_index)
        self._update_step_nav_buttons()

    def _convert_to_cv(self, width, height, pixel_type, img_bytes):
        """将相机原始帧数据转换为 OpenCV BGR 图像"""
        try:
            img = raw_to_opencv(img_bytes, width, height, pixel_type)
            if img is None:
                return np.zeros((height, width, 3), dtype=np.uint8)
            # 确保是 3 通道 BGR
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        except Exception as e:
            log_error(f"图像转换失败: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def _run_preview(self):
        if self.vision_engine.pipeline is None:
            QMessageBox.warning(self, "提示", "请先选择并应用一个方案")
            return

        # 每次点击测试按钮都重新拍照获取新图像
        if self._camera_panel is None or not self._camera_panel.is_camera_open():
            QMessageBox.warning(self, "提示", "请先打开相机")
            return

        self._pending_engineer_test = True
        self.eng_btn_run_preview.setEnabled(False)
        self.eng_btn_run_preview.setText("拍照中...")
        self.status_label.setText("正在拍照...")
        QApplication.processEvents()
        self._capture()

    def _execute_engineer_test(self):
        """执行设计模式流水线测试（内部方法，_raw_image 必须非空）"""
        # 如果已有测试线程在运行，不重复启动
        if self._eng_test_worker is not None and self._eng_test_worker.isRunning():
            return

        self.eng_btn_run_preview.setEnabled(False)
        self.eng_btn_run_preview.setText("执行中...")

        self.eng_log.clear_log()
        self.eng_time_label.setText("")
        self.eng_result_panel.clear()
        self.eng_log.append_info(f"══════ 流水线测试开始 ══════", "#4fc3f7")
        self.eng_log.append_info(f"方案: {self._current_scheme_name or '未命名'}", "#888")

        # 在后台线程执行检测，避免阻塞UI
        scheme_name = self._current_scheme_name or "未命名"
        self._eng_test_worker = EngineerTestWorker(
            self.vision_engine, self._raw_image.copy(), scheme_name
        )
        self._eng_test_worker.finished.connect(self._on_engineer_test_finished)
        self._eng_test_worker.start()

    def _on_engineer_test_finished(self, passed, message, annotated, results):
        """工程师测试完成回调（主线程执行，安全更新UI）"""
        try:
            # 存储步骤结果用于导航
            self._step_results = list(results) if results else []
            self._annotated_image = annotated
            self._current_step_index = -1  # 默认显示最终结果

            tool_results = None
            if results:
                total_ms = sum(r.elapsed_ms for r in results)
                tool_results = {
                    "total_elapsed_ms": total_ms,
                    "steps": [
                        {
                            "tool_name": r.tool_name or r.tool_type,
                            "status": "✓" if r.passed else "✗",
                            "elapsed_ms": r.elapsed_ms,
                            "message": r.message,
                        }
                        for r in results
                    ]
                }

            self.eng_result_panel.show_result(passed, message, tool_results=tool_results)

            # 显示最终标注结果并更新导航按钮
            if annotated is not None:
                self._show_engineer_image(annotated)
            self._update_step_nav_buttons()

            for i, r in enumerate(results):
                ts = datetime.now().strftime("%H:%M:%S")
                status = "✓" if r.passed else "✗"
                self.eng_log.append_log(ts, i + 1, r.tool_type, status,
                                        r.message, r.elapsed_ms)

            self.eng_log.append_separator()
            total_ms = sum(r.elapsed_ms for r in results)
            # 更新设计模式总测试时间显示
            self.eng_time_label.setText(f"⏱ {total_ms:.0f}ms")
            if passed:
                self.eng_log.append_info(
                    f"✓ 检测通过 (OK) | 总耗时: {total_ms:.1f}ms", "#8bc34a")
            else:
                self.eng_log.append_info(
                    f"✗ 检测不通过 (NG) | 总耗时: {total_ms:.1f}ms", "#ff5252")

            status = "OK" if passed else "NG"
            self.status_label.setText(f"测试完成: {status}")
            log_info(f"工程师测试完成: {status} | 方案={self._current_scheme_name or '未命名'}")

        except Exception as e:
            log_error(f"测试结果处理异常: {e}")
            self.eng_result_panel.show_result(False, f"测试异常: {str(e)}")
            self.eng_log.append_info(f"✗ 执行异常: {str(e)}", "#ff5252")
            self.status_label.setText("测试异常")
        finally:
            self.eng_btn_run_preview.setEnabled(True)
            self.eng_btn_run_preview.setText("📷 测试")
            self._eng_test_worker = None

    def _do_detect(self):
        # === COMMENTED OUT: 生产模式检测功能（UI元素已移除） ===
        # # 如果没有图像，先自动拍照
        # if self._raw_image is None:
        #     if self._camera_panel is None or not self._camera_panel.is_camera_open():
        #         QMessageBox.warning(self, "提示", "请先打开相机")
        #         return
        #     self.status_label.setText("正在拍照...")
        #     self.worker_status_label.setText("正在拍照...")
        #     self.worker_btn_detect.setEnabled(False)
        #     self.worker_btn_detect.setText("拍照中...")
        #     self._camera_panel.capture_once()
        #     # 拍照完成后 _on_capture_completed 会再次调用 _do_detect
        #     self._pending_detect = True
        #     return
        #
        # if self.vision_engine.pipeline is None:
        #     QMessageBox.warning(self, "提示", "请先选择并应用一个方案")
        #     return
        #
        # # 如果已有检测线程在运行，不重复启动
        # if self._detect_worker is not None and self._detect_worker.isRunning():
        #     return
        #
        # self.worker_btn_detect.setEnabled(False)
        # self.worker_btn_detect.setText("检测中...")
        # self.status_label.setText("检测中...")
        # self.worker_status_label.setText("检测中...")
        #
        # self.worker_log.clear_log()
        # self.worker_time_label.setText("")
        # self.worker_log.append_info(f"══════ 检测开始 ══════", "#4fc3f7")
        #
        # # 在后台线程执行检测，避免阻塞UI
        # scheme_name = self._current_scheme_name or "未命名"
        # self._detect_worker = DetectWorker(
        #     self.vision_engine, self._raw_image.copy(), scheme_name
        # )
        # self._detect_worker.finished.connect(self._on_detect_finished)
        # self._detect_worker.start()
        # === END ===
        log_info("_do_detect 已禁用（生产模式已移除）")

    def _on_detect_finished(self, passed, message, annotated, results):
        """检测完成回调（主线程执行，安全更新UI）- 已禁用（生产模式已移除）"""
        # === COMMENTED OUT: 生产模式检测回调（UI元素已移除） ===
        # try:
        #     if passed:
        #         self.worker_judge.setText("✓ OK")
        #         self.worker_judge.setStyleSheet("""...""")
        #         self.worker_status_label.setText("检测通过 (OK)")
        #     else:
        #         self.worker_judge.setText("✗ NG")
        #         self.worker_judge.setStyleSheet("""...""")
        #         self.worker_status_label.setText("检测不通过 (NG)")
        #
        #     if annotated is not None:
        #         self._last_annotated = annotated
        #         self._show_worker_image(annotated)
        #
        #     for i, r in enumerate(results):
        #         ts = datetime.now().strftime("%H:%M:%S")
        #         status = "✓" if r.passed else "✗"
        #         self.worker_log.append_log(ts, i + 1, r.tool_type, status,
        #                                    r.message, r.elapsed_ms)
        #
        #     self.worker_log.append_separator()
        #     total_ms = sum(r.elapsed_ms for r in results)
        #     self.worker_time_label.setText(f"⏱ {total_ms:.0f}ms")
        #     if passed:
        #         self.worker_log.append_info(
        #             f"✓ 检测通过 (OK) | 总耗时: {total_ms:.1f}ms", "#8bc34a")
        #     else:
        #         self.worker_log.append_info(
        #             f"✗ 检测不通过 (NG) | 总耗时: {total_ms:.1f}ms", "#ff5252")
        #
        #     status = "OK" if passed else "NG"
        #     self.status_label.setText(f"检测完成: {status}")
        #     log_info(f"检测完成: {status} | 方案={self._current_scheme_name or '未命名'}")
        #
        # except Exception as e:
        #     log_error(f"检测结果处理异常: {e}")
        #     self.worker_judge.setText("✗ 异常")
        #     self.worker_judge.setStyleSheet("""...""")
        #     self.worker_log.append_info(f"✗ 执行异常: {str(e)}", "#ff5252")
        #     self.status_label.setText("检测异常")
        #     self.worker_status_label.setText("检测异常")
        # finally:
        #     self.worker_btn_detect.setEnabled(True)
        #     self.worker_btn_detect.setText("📷 开始检测")
        #     # 清除原始图像，确保下次点击"开始检测"时重新拍照
        #     self._raw_image = None
        #     self._raw_width = 0
        #     self._raw_height = 0
        #     self._detect_worker = None
        # === END ===
        log_info("_on_detect_finished 已禁用（生产模式已移除）")

    def _show_log_settings(self):
        """打开日志限额设置对话框"""
        from core.paths import LOGS_DIR, PRODUCTION_DATA_DIR
        from core.log_manager import _get_dir_size, _CLEANUP_DIRS

        dialog = QDialog(self)
        dialog.setWindowTitle("存储空间限额设置")
        dialog.setMinimumWidth(480)
        dialog.setStyleSheet("""
            QDialog { background-color: #2d2d2d; }
            QLabel { color: #d4d4d4; font-size: 16px; }
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 4px 8px; font-size: 16px;
            }
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 6px 20px; border: 1px solid #555;
                border-radius: 3px; font-size: 16px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton#btn_apply {
                background-color: #1a3a5c; color: #4A90D9;
                border: 1px solid #2a5a8c; font-weight: bold;
            }
            QPushButton#btn_apply:hover { background-color: #2a4a7c; }
            QPushButton#btn_cleanup {
                background-color: #E65100; color: #fff;
                border: 1px solid #FF6D00; font-weight: bold;
            }
            QPushButton#btn_cleanup:hover { background-color: #BF360C; }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        # 计算 logs + errors 总大小
        def _calc_total_size():
            total = 0
            for d in _CLEANUP_DIRS:
                total += _get_dir_size(d)
            return total

        try:
            current_size = _calc_total_size()
            size_gb = current_size / (1024 ** 3)
            size_str = f"{size_gb:.2f} GB" if size_gb >= 1 else f"{current_size / (1024 ** 2):.1f} MB"
        except Exception:
            size_str = "未知"

        # 分别显示 logs 和 production data 的大小
        try:
            logs_size = _get_dir_size(LOGS_DIR)
            prod_size = _get_dir_size(PRODUCTION_DATA_DIR)
            logs_str = f"{logs_size / (1024**3):.2f} GB" if logs_size >= 1024**3 else f"{logs_size / (1024**2):.1f} MB"
            prod_str = f"{prod_size / (1024**3):.2f} GB" if prod_size >= 1024**3 else f"{prod_size / (1024**2):.1f} MB"
            detail_str = (f"   ├ 日志(logs): {logs_str}\n"
                          f"   └ 生产数据(production data): {prod_str}")
        except Exception:
            detail_str = ""

        size_label = QLabel(f"📂 当前数据大小: {size_str}")
        size_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4fc3f7;")
        layout.addWidget(size_label)

        detail_label = QLabel(detail_str)
        detail_label.setStyleSheet("color: #999; font-size: 15px; padding-left: 8px;")
        layout.addWidget(detail_label)

        # 日志限额设置
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(Qt.AlignRight)

        max_size_spin = QDoubleSpinBox(dialog)
        max_size_spin.setRange(1, 9999)
        max_size_spin.setDecimals(1)
        max_size_spin.setSuffix(" GB")
        max_size_spin.setValue(self.config.get('system.log_max_size_gb', 50))
        max_size_spin.setToolTip("当 logs + errors 总大小超过此限额时自动清理")
        form_layout.addRow("最大限额:", max_size_spin)

        ratio_spin = QDoubleSpinBox(dialog)
        ratio_spin.setRange(0.1, 0.9)
        ratio_spin.setDecimals(1)
        ratio_spin.setSingleStep(0.1)
        ratio_spin.setSuffix(" (× 最大限额)")
        ratio_spin.setValue(self.config.get('system.log_cleanup_ratio', 0.5))
        ratio_spin.setToolTip("超出限额后清理到 最大限额 × 此比例")
        form_layout.addRow("清理目标比例:", ratio_spin)

        layout.addLayout(form_layout)

        # 说明文字
        hint = QLabel(
            "💡 当 logs + errors 总大小超过「最大限额」时，系统会自动\n"
            "   从最早的文件开始删除，直到总大小降到「最大限额 × 比例」以下。\n"
            "   ⚠ 注意：仅清理 logs 和 errors 目录下的文件，不影响方案配置。\n"
            f"   当前设置: 超过 {max_size_spin.value():.0f}GB 时清理到 {max_size_spin.value() * ratio_spin.value():.0f}GB"
        )
        hint.setStyleSheet("color: #999; font-size: 15px; padding: 8px; "
                           "background-color: #252525; border-radius: 4px;")
        layout.addWidget(hint)

        # 更新提示文字
        def _update_hint():
            max_val = max_size_spin.value()
            ratio_val = ratio_spin.value()
            hint.setText(
                "💡 当 logs + errors 总大小超过「最大限额」时，系统会自动\n"
                "   从最早的文件开始删除，直到总大小降到「最大限额 × 比例」以下。\n"
                "   ⚠ 注意：仅清理 logs 和 errors 目录下的文件，不影响方案配置。\n"
                f"   当前设置: 超过 {max_val:.0f}GB 时清理到 {max_val * ratio_val:.0f}GB"
            )
        max_size_spin.valueChanged.connect(_update_hint)
        ratio_spin.valueChanged.connect(_update_hint)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        btn_cleanup = QPushButton("🗑 立即清理")
        btn_cleanup.setObjectName("btn_cleanup")
        btn_cleanup.setToolTip("立即按当前设置执行一次清理（删除 logs 和 errors 中最旧的文件）")
        btn_cleanup.clicked.connect(lambda: self._do_manual_cleanup(dialog))
        btn_layout.addWidget(btn_cleanup)

        btn_layout.addStretch()

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)

        btn_apply = QPushButton("✓ 应用")
        btn_apply.setObjectName("btn_apply")
        btn_apply.clicked.connect(lambda: self._save_log_settings(
            dialog, max_size_spin.value(), ratio_spin.value()))
        btn_layout.addWidget(btn_apply)

        layout.addLayout(btn_layout)

        dialog.exec_()

    def _save_log_settings(self, dialog: QDialog, max_size_gb: float, cleanup_ratio: float):
        """保存日志限额设置"""
        self.config.set('system.log_max_size_gb', max_size_gb)
        self.config.set('system.log_cleanup_ratio', cleanup_ratio)
        self.config.save()
        log_info(f"存储限额设置已更新: 最大={max_size_gb}GB, 清理比例={cleanup_ratio}")
        QMessageBox.information(dialog, "成功", "存储限额设置已保存")
        dialog.accept()

    def _do_manual_cleanup(self, parent: QWidget):
        """立即执行一次清理（清理 logs + errors 中最旧的文件）"""
        from core.log_manager import _get_dir_size, _CLEANUP_DIRS
        max_size_gb = self.config.get('system.log_max_size_gb', 50)
        cleanup_ratio = self.config.get('system.log_cleanup_ratio', 0.5)
        max_size = int(max_size_gb * 1024 ** 3)
        LogManager.cleanup_now(max_size=max_size, cleanup_ratio=cleanup_ratio)
        log_info(f"手动触发存储清理: 最大={max_size_gb}GB, 比例={cleanup_ratio}")

        # 刷新大小显示
        try:
            total = 0
            for d in _CLEANUP_DIRS:
                total += _get_dir_size(d)
            size_gb = total / (1024 ** 3)
            size_str = f"{size_gb:.2f} GB" if size_gb >= 1 else f"{total / (1024 ** 2):.1f} MB"
        except Exception:
            size_str = "未知"

        QMessageBox.information(parent, "清理完成", f"清理完成\n当前 logs + errors 总大小: {size_str}")

    def _show_about(self):
        QMessageBox.about(self, "关于",
                          "<h3>基板硅胶视觉检测系统</h3>"
                          "<p>版本: 1.0.0</p>"
                          "<p>基于 OpenCV + PyQt5 的视觉识别系统</p>"
                          "<p>支持流水线式视觉工具链设计</p>")

    def _open_serial_dialog(self):
        """打开串口通信窗口（共享 SerialCommManager 实例）。"""
        from .widgets.serial_dialog import SerialDialog
        if self._serial_comm is None:
            self._serial_comm = SerialCommManager()
            # 将串口管理器设置到自动化检测工作流（用于扫描头）
            if self._inspection_workflow is not None:
                self._inspection_workflow.set_serial_comm(self._serial_comm)
        dialog = SerialDialog(self, comm_mgr=self._serial_comm)
        dialog.exec_()
        # 对话框关闭后，根据串口状态更新自动测试按钮
        self._update_auto_test_btn_state()

    def _open_nmc_dialog(self):
        """打开运动控制卡窗口（共享 NMCSDK 实例）。"""
        from .widgets.nmc_control_dialog import NMCControlDialog
        if self._nmc_sdk is None:
            self._nmc_sdk = NMCSDK()
        dialog = NMCControlDialog(self, nmc_sdk=self._nmc_sdk)
        dialog.exec_()

    # ──────────────────────────────────────────────
    # 串口自动测试工作流
    # ──────────────────────────────────────────────

    def _update_auto_test_btn_state(self):
        """根据串口和方案状态更新自动测试按钮。"""
        # === COMMENTED OUT: 生产模式UI引用 ===
        # comm_ok = (self._serial_comm is not None and self._serial_comm.is_open)
        # pipeline_ok = (self.vision_engine.pipeline is not None)
        # workflow_running = (self._serial_workflow is not None
        #                     and self._serial_workflow.is_running)
        #
        # if workflow_running:
        #     self.worker_btn_auto_test.setEnabled(True)
        # else:
        #     self.worker_btn_auto_test.setEnabled(comm_ok and pipeline_ok)
        # === END ===
        pass

    def _toggle_auto_test(self, checked: bool):
        """切换自动测试状态。"""
        if checked:
            self._start_auto_test()
        else:
            self._stop_auto_test()

    def _start_auto_test(self):
        """启动串口自动测试工作流。"""
        # === COMMENTED OUT: 生产模式UI引用 ===
        # # 检查串口
        # if self._serial_comm is None or not self._serial_comm.is_open:
        #     QMessageBox.warning(self, "提示",
        #                         "请先通过「通信 > 串口通信」打开串口连接")
        #     self.worker_btn_auto_test.setChecked(False)
        #     return
        #
        # # 检查方案
        # if self.vision_engine.pipeline is None:
        #     QMessageBox.warning(self, "提示", "请先导入检测方案")
        #     self.worker_btn_auto_test.setChecked(False)
        #     return
        #
        # # 创建并启动工作流
        # self._serial_workflow = SerialTestWorkflow(
        #     comm_mgr=self._serial_comm,
        #     config=WorkflowConfig(),
        #     parent=self,
        # )
        #
        # # 连接信号
        # self._serial_workflow.state_changed.connect(
        #     self._on_workflow_state_changed)
        # self._serial_workflow.capture_requested.connect(
        #     self._on_workflow_capture_requested)
        # self._serial_workflow.test_requested.connect(
        #     self._on_workflow_test_requested)
        # self._serial_workflow.error_occurred.connect(
        #     self._on_workflow_error)
        #
        # # 启动
        # self._serial_workflow.start()
        #
        # # 更新 UI
        # self.worker_btn_auto_test.setText("⏹ 停止自动测试")
        # self.worker_btn_detect.setEnabled(False)
        # self.status_label.setText("自动测试已启动 - 等待触发信号...")
        # self.worker_status_label.setText("自动测试已启动 - 等待触发信号...")
        # log_info("串口自动测试工作流已启动")
        # === END ===
        log_info("_start_auto_test 已禁用（生产模式已移除）")

    def _stop_auto_test(self):
        """停止串口自动测试工作流。"""
        # === COMMENTED OUT: 生产模式UI引用 ===
        # if self._serial_workflow:
        #     self._serial_workflow.stop()
        #     self._serial_workflow.cleanup()
        #     self._serial_workflow = None
        #
        # self.worker_btn_auto_test.setText("🔌 启动自动测试")
        # self.worker_btn_auto_test.setChecked(False)
        # self.worker_btn_detect.setEnabled(
        #     self._raw_image is not None
        #     and self.vision_engine.pipeline is not None
        # )
        # self.status_label.setText("自动测试已停止")
        # self.worker_status_label.setText("自动测试已停止")
        # log_info("串口自动测试工作流已停止")
        # === END ===
        log_info("_stop_auto_test 已禁用（生产模式已移除）")

    def _on_workflow_state_changed(self, state):
        """工作流状态变化时更新 UI。"""
        # === COMMENTED OUT: 生产模式UI引用 ===
        # state_names = {
        #     SerialTestWorkflow.State.IDLE: "空闲",
        #     SerialTestWorkflow.State.WAITING_TRIGGER: "等待触发信号...",
        #     SerialTestWorkflow.State.CAPTURING: "拍照中...",
        #     SerialTestWorkflow.State.TESTING: "检测中...",
        #     SerialTestWorkflow.State.SENDING_RESULT: "发送结果...",
        # }
        # name = state_names.get(state, str(state))
        # self.worker_status_label.setText(f"自动测试: {name}")
        # self.status_label.setText(f"自动测试: {name}")
        # === END ===
        pass

    def _on_workflow_capture_requested(self):
        """工作流请求拍照。"""
        if self._camera_panel is not None and self._camera_panel.is_camera_open():
            self._capture()
        else:
            self._serial_workflow.on_capture_completed(None)

    def _on_workflow_test_requested(self, image):
        """工作流请求执行检测。"""
        if self.vision_engine.pipeline is None:
            self._serial_workflow.on_test_completed(False, "未设置检测方案")
            return

        # 如果已有工作流测试线程在运行，不重复启动
        if self._workflow_test_worker is not None and self._workflow_test_worker.isRunning():
            return

        scheme_name = self._current_scheme_name or "未命名"
        self._workflow_test_worker = WorkflowTestWorker(
            self.vision_engine, image.copy(), scheme_name
        )
        self._workflow_test_worker.finished.connect(self._on_workflow_test_finished)
        self._workflow_test_worker.start()

    def _on_workflow_test_finished(self, passed, message, annotated, results):
        """工作流测试完成回调（主线程执行，安全更新UI）- 已禁用（生产模式已移除）"""
        # === COMMENTED OUT: 生产模式UI引用 ===
        # try:
        #     # 更新显示
        #     if annotated is not None:
        #         self._last_annotated = annotated
        #         self._show_worker_image(annotated)
        #
        #     # 更新 OK/NG 判断
        #     if passed:
        #         self.worker_judge.setText("✓ OK")
        #         self.worker_judge.setStyleSheet("""...""")
        #     else:
        #         self.worker_judge.setText("✗ NG")
        #         self.worker_judge.setStyleSheet("""...""")
        #
        #     # 记录日志
        #     self.worker_log.clear_log()
        #     self.worker_time_label.setText("")
        #     self.worker_log.append_info(...)
        #     for i, r in enumerate(results):
        #         ...
        #     self.worker_log.append_separator()
        #     total_ms = sum(r.elapsed_ms for r in results)
        #     self.worker_time_label.setText(f"⏱ {total_ms:.0f}ms")
        #     ...
        #
        #     # 回调工作流
        #     self._serial_workflow.on_test_completed(passed, message)
        #
        # except Exception as e:
        #     log_error(f"自动测试结果处理异常: {e}")
        #     self._serial_workflow.on_test_completed(False, str(e))
        # finally:
        #     self._workflow_test_worker = None
        # === END ===
        log_info("_on_workflow_test_finished 已禁用（生产模式已移除）")

    def _on_workflow_error(self, error_msg: str):
        """工作流错误处理。"""
        # === COMMENTED OUT: 生产模式UI引用 ===
        # self.worker_status_label.setText(f"自动测试错误: {error_msg}")
        # === END ===
        self.status_label.setText(f"自动测试错误: {error_msg}")
        log_error(f"自动测试错误: {error_msg}")

    def closeEvent(self, event):
        log_info("系统关闭")
        # 停止轴监控定时器
        try:
            if hasattr(self, '_eng_axis_timer') and self._eng_axis_timer is not None:
                self._eng_axis_timer.stop()
                self._eng_axis_timer = None
        except Exception:
            pass
        # 停止回零定时器
        try:
            if hasattr(self, '_eng_home_timer') and self._eng_home_timer is not None:
                self._eng_home_timer.stop()
                self._eng_home_timer = None
        except Exception:
            pass
        # 停止自动化检测工作流
        if self._inspection_workflow is not None:
            self._inspection_workflow.cleanup()
            self._inspection_workflow = None
        # 停止自动测试工作流
        if self._serial_workflow is not None:
            self._serial_workflow.stop()
            self._serial_workflow.cleanup()
            self._serial_workflow = None
        # 关闭串口
        if self._serial_comm is not None:
            self._serial_comm.cleanup()
            self._serial_comm = None
        # 关闭运动控制卡
        if self._nmc_sdk is not None:
            try:
                if self._nmc_sdk._connected:
                    self._nmc_sdk.close_net()
            except Exception:
                pass
            self._nmc_sdk = None
        try:
            if self._camera_panel is not None:
                self._camera_panel.close_camera()
        except (RuntimeError, AttributeError):
            # Qt 对象已被删除，忽略
            pass
        CameraManager.finalize_sdk()
        event.accept()


# ============================================================================
# 产品配置编辑对话框
# ============================================================================

class ProductConfigDialog(QDialog):
    """产品配置编辑对话框，用于新建/编辑产品配置"""

    def __init__(self, parent=None, mode="new", product_name=None):
        """
        Args:
            parent: 父窗口
            mode: "new" 新建 / "edit" 编辑
            product_name: 编辑模式下的产品名称
        """
        super().__init__(parent)
        self._mode = mode
        self._product_name = product_name
        self._config = None

        self.setWindowTitle("新建产品配置" if mode == "new" else f"编辑产品 - {product_name}")
        self.setMinimumSize(640, 640)
        self.resize(780, 720)
        self.setStyleSheet("""
            QDialog { background-color: #2d2d2d; }
            QLabel { color: #d4d4d4; font-size: 15px; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; border-radius: 3px;
                padding: 4px 8px;
            }
            QGroupBox {
                font-weight: bold; font-size: 16px; border: 1px solid #444;
                border-radius: 4px; margin-top: 8px; padding-top: 14px; color: #d4d4d4;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTableWidget {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #444; gridline-color: #333;
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #444; padding: 4px;
            }
        """)

        if mode == "edit" and product_name:
            from core.product_manager import load_product
            self._config = load_product(product_name)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 12, 16, 12)

        # ── 基本信息 ──
        basic_group = QGroupBox("基本信息")
        basic_layout = QFormLayout(basic_group)
        basic_layout.setSpacing(6)

        self._edit_name = QLineEdit(self._config.get("name", "") if self._config else "")
        self._edit_name.setPlaceholderText("输入产品型号名称")
        basic_layout.addRow("产品名称:", self._edit_name)

        self._edit_desc = QLineEdit(self._config.get("description", "") if self._config else "")
        self._edit_desc.setPlaceholderText("产品描述（可选）")
        basic_layout.addRow("描述:", self._edit_desc)

        layout.addWidget(basic_group)

        # ── 相机参数 ──
        camera_group = QGroupBox("相机参数")
        camera_layout = QFormLayout(camera_group)
        camera_layout.setSpacing(6)

        camera = self._config.get("camera", {}) if self._config else {}

        self._edit_exposure = QDoubleSpinBox()
        self._edit_exposure.setRange(1, 1000000)
        self._edit_exposure.setValue(camera.get("exposure_time", 18000))
        self._edit_exposure.setSuffix(" us")
        camera_layout.addRow("曝光时间:", self._edit_exposure)

        self._edit_gain = QDoubleSpinBox()
        self._edit_gain.setRange(0, 48)
        self._edit_gain.setValue(camera.get("gain", 0))
        self._edit_gain.setSuffix(" dB")
        camera_layout.addRow("增益:", self._edit_gain)

        layout.addWidget(camera_group)

        # ── 运动参数 ──
        motion_group = QGroupBox("运动参数")
        motion_layout = QFormLayout(motion_group)
        motion_layout.setSpacing(6)

        motion = self._config.get("motion", {}) if self._config else {}

        self._edit_vmax = QDoubleSpinBox()
        self._edit_vmax.setRange(1, 500000)
        self._edit_vmax.setValue(motion.get("v_max", 50000))
        motion_layout.addRow("最大速度:", self._edit_vmax)

        self._edit_origin = QSpinBox()
        self._edit_origin.setRange(-1000000, 1000000)
        self._edit_origin.setValue(motion.get("origin_position", 0))
        motion_layout.addRow("原点位置:", self._edit_origin)

        self._edit_timeout = QSpinBox()
        self._edit_timeout.setRange(1, 120)
        self._edit_timeout.setValue(motion.get("move_timeout_s", 10))
        self._edit_timeout.setSuffix(" 秒")
        motion_layout.addRow("运动超时:", self._edit_timeout)

        layout.addWidget(motion_group)

        # ── DI 配置 ──
        di_group = QGroupBox("触发配置")
        di_layout = QFormLayout(di_group)
        di_layout.setSpacing(6)

        di_bit = self._config.get("di_bit", 3) if self._config else 3
        self._edit_di_bit = QSpinBox()
        self._edit_di_bit.setRange(0, 31)
        self._edit_di_bit.setValue(di_bit)
        di_layout.addRow("DI 输入位:", self._edit_di_bit)

        layout.addWidget(di_group)

        # ── 扫码配置 ──
        scan_group = QGroupBox("一维码扫码配置")
        scan_layout = QFormLayout(scan_group)
        scan_layout.setSpacing(6)

        barcode_cfg = self._config.get("barcode_scan", {}) if self._config else {}

        self._scan_enabled_cb = QCheckBox("启用扫码")
        self._scan_enabled_cb.setChecked(barcode_cfg.get("enabled", False))
        self._scan_enabled_cb.setStyleSheet("color: #d4d4d4; font-size: 15px; spacing: 8px;")
        scan_layout.addRow("", self._scan_enabled_cb)

        self._scan_position = QSpinBox()
        self._scan_position.setRange(-1000000, 1000000)
        self._scan_position.setValue(barcode_cfg.get("position", 0))
        scan_layout.addRow("扫码位坐标:", self._scan_position)

        self._scan_command = QLineEdit(barcode_cfg.get("command", "01 54 04"))
        self._scan_command.setPlaceholderText("如: 01 54 04")
        scan_layout.addRow("扫描命令(HEX):", self._scan_command)

        self._scan_timeout = QSpinBox()
        self._scan_timeout.setRange(1000, 60000)
        self._scan_timeout.setValue(barcode_cfg.get("timeout_ms", 5000))
        self._scan_timeout.setSuffix(" ms")
        scan_layout.addRow("扫码超时:", self._scan_timeout)

        layout.addWidget(scan_group)

        # ── 位置列表 ──
        pos_group = QGroupBox("检测位置")
        pos_layout = QVBoxLayout(pos_group)
        pos_layout.setSpacing(4)

        self._pos_table = QTableWidget()
        self._pos_table.setColumnCount(4)
        self._pos_table.setHorizontalHeaderLabels(["位置名称", "坐标", "视觉方案", ""])
        self._pos_table.horizontalHeader().setStretchLastSection(False)
        self._pos_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._pos_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._pos_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._pos_table.setMinimumHeight(150)

        pos_layout.addWidget(self._pos_table, 1)

        pos_btn_layout = QHBoxLayout()
        self._btn_add_pos = QPushButton("+ 添加位置")
        self._btn_remove_pos = QPushButton("- 删除选中")
        for btn in [self._btn_add_pos, self._btn_remove_pos]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c; color: #d4d4d4;
                    padding: 4px 12px; border: 1px solid #555;
                    border-radius: 3px;
                }
                QPushButton:hover { background-color: #4a4a4a; }
            """)
        pos_btn_layout.addWidget(self._btn_add_pos)
        pos_btn_layout.addWidget(self._btn_remove_pos)
        pos_btn_layout.addStretch()
        pos_layout.addLayout(pos_btn_layout)

        layout.addWidget(pos_group, 1)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._btn_ok = QPushButton("确定")
        self._btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; color: #fff;
                padding: 6px 24px; border: 1px solid #4CAF50;
                border-radius: 3px; font-weight: bold; font-size: 16px;
            }
            QPushButton:hover { background-color: #388E3C; }
        """)

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c; color: #d4d4d4;
                padding: 6px 24px; border: 1px solid #555;
                border-radius: 3px; font-size: 16px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
        """)

        btn_layout.addWidget(self._btn_ok)
        btn_layout.addWidget(self._btn_cancel)
        layout.addLayout(btn_layout)

        # 连接信号
        self._btn_ok.clicked.connect(self._on_ok)
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_add_pos.clicked.connect(self._add_position_row)
        self._btn_remove_pos.clicked.connect(self._remove_selected_row)

        # 加载已有位置数据
        if self._config:
            positions = self._config.get("positions", [])
            for pos in positions:
                self._add_position_row(
                    name=pos.get("name", ""),
                    position=pos.get("position", 0),
                    scheme=pos.get("scheme", "")
                )

    def _add_position_row(self, name="", position=0, scheme=""):
        """添加一行位置配置"""
        row = self._pos_table.rowCount()
        self._pos_table.insertRow(row)

        # 位置名称
        name_item = QTableWidgetItem(str(name))
        self._pos_table.setItem(row, 0, name_item)

        # 坐标
        pos_item = QTableWidgetItem(str(position))
        self._pos_table.setItem(row, 1, pos_item)

        # 方案选择（下拉框）
        combo = QComboBox()
        combo.setEditable(True)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c; color: #d4d4d4;
                border: 1px solid #555; padding: 2px 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #d4d4d4;
            }
        """)
        # 加载所有可用方案
        from core.paths import SCHEME_DIR
        import os
        combo.addItem("")
        if os.path.exists(SCHEME_DIR):
            for f in sorted(os.listdir(SCHEME_DIR)):
                if f.endswith(".json"):
                    sname = os.path.splitext(f)[0]
                    combo.addItem(sname)
        if scheme:
            combo.setCurrentText(scheme)
        self._pos_table.setCellWidget(row, 2, combo)

        # 删除按钮（占位列）
        self._pos_table.setItem(row, 3, QTableWidgetItem(""))

    def _remove_selected_row(self):
        """删除选中的行"""
        row = self._pos_table.currentRow()
        if row >= 0:
            self._pos_table.removeRow(row)

    def _on_ok(self):
        """确定按钮"""
        name = self._edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入产品名称")
            return

        # 收集位置数据
        positions = []
        for row in range(self._pos_table.rowCount()):
            name_item = self._pos_table.item(row, 0)
            pos_item = self._pos_table.item(row, 1)
            combo = self._pos_table.cellWidget(row, 2)

            pos_name = name_item.text().strip() if name_item else ""
            pos_value = int(pos_item.text().strip()) if pos_item and pos_item.text().strip() else 0
            scheme_name = combo.currentText().strip() if combo else ""

            if pos_name:  # 只保存有名称的位置
                positions.append({
                    "name": pos_name,
                    "position": pos_value,
                    "scheme": scheme_name
                })

        # 构建配置
        config = {
            "name": name,
            "description": self._edit_desc.text().strip(),
            "barcode_scan": {
                "enabled": self._scan_enabled_cb.isChecked(),
                "position": self._scan_position.value(),
                "command": self._scan_command.text().strip() or "01 54 04",
                "timeout_ms": self._scan_timeout.value()
            },
            "camera": {
                "exposure_time": int(self._edit_exposure.value()),
                "gain": self._edit_gain.value()
            },
            "motion": {
                "axis": 1,
                "v_max": int(self._edit_vmax.value()),
                "a_max": 100000,
                "origin_position": self._edit_origin.value(),
                "move_timeout_s": self._edit_timeout.value()
            },
            "di_bit": self._edit_di_bit.value(),
            "poll_interval_ms": 50,
            "positions": positions
        }

        # 保存
        from core.product_manager import save_product
        if save_product(config):
            log_info(f"产品配置已保存: {name}")
            self._config = config
            self.accept()
        else:
            QMessageBox.critical(self, "错误", f"保存产品配置「{name}」失败")
