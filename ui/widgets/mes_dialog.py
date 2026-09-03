"""MES设置窗口
打开软件显示MES设置，可配置是否使用MES、IP地址、端口号、站点编码、员工号。
配置保存到 config.json，下次启动自动回填。
"""
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, \
    QSpinBox, QPushButton, QGroupBox, QLineEdit, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt

from core.config_manager import ConfigManager
from core.log_manager import log_info, log_error


class MESDialog(QDialog):
    """MES设置对话框。

    软件启动时弹出，用于配置 MES 功能：
        - 是否使用 MES 功能（复选框）
        - IP 地址、端口号、站点编码、员工号
    配置通过 ConfigManager 保存到 config.json，下次启动自动回填。
    """

    def __init__(self, manager=None, parent=None):
        super().__init__(parent)
        self._manager = manager or ConfigManager()
        self.setWindowTitle("MES设置")
        self.setMinimumSize(440, 360)
        self._setup_ui()
        self._connect_signals()
        self._load_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 是否使用 MES 功能
        self._chk_enabled = QCheckBox("启用 MES 功能")
        self._chk_enabled.setStyleSheet("color: #d4d4d4; font-size: 14px; font-weight: bold;")
        layout.addWidget(self._chk_enabled)

        # 参数设置分组
        self._param_group = QGroupBox("MES参数")
        param_layout = QVBoxLayout(self._param_group)

        # IP地址
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("IP地址:"))
        self._line_ip = QLineEdit()
        self._line_ip.setPlaceholderText("请输入MES服务器IP地址")
        ip_row.addWidget(self._line_ip, 1)
        param_layout.addLayout(ip_row)

        # 端口号
        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("端口号:"))
        self._spin_port = QSpinBox()
        self._spin_port.setRange(1, 65535)
        self._spin_port.setValue(7010)   # 默认 MES 端口 7010
        port_row.addWidget(self._spin_port, 1)
        param_layout.addLayout(port_row)

        # 站点编码
        site_row = QHBoxLayout()
        site_row.addWidget(QLabel("站点编码:"))
        self._line_site = QLineEdit()
        self._line_site.setPlaceholderText("请输入站点编码")
        site_row.addWidget(self._line_site, 1)
        param_layout.addLayout(site_row)

        # 员工号
        employee_row = QHBoxLayout()
        employee_row.addWidget(QLabel("员工号:"))
        self._line_employee = QLineEdit()
        self._line_employee.setPlaceholderText("请输入员工号/设备工号")
        employee_row.addWidget(self._line_employee, 1)
        param_layout.addLayout(employee_row)

        layout.addWidget(self._param_group)

        # 测试连接按钮
        test_row = QHBoxLayout()
        self.btn_test = QPushButton("测试连接")
        test_row.addStretch(1)
        test_row.addWidget(self.btn_test)
        layout.addLayout(test_row)

        # 底部按钮区域
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_cancel = QPushButton("取消")
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        # 初始状态：默认启用 MES 参数分组
        self._update_param_group_state()

    def _connect_signals(self):
        """绑定信号槽"""
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_test.clicked.connect(self._on_test_connection)
        self._chk_enabled.toggled.connect(self._update_param_group_state)

    def _update_param_group_state(self):
        """根据是否启用 MES 更新参数分组的可用状态"""
        enabled = self._chk_enabled.isChecked()
        self._param_group.setEnabled(enabled)
        self.btn_test.setEnabled(enabled)

    def _load_config(self):
        """从 config.json 回填配置"""
        mes = self._manager.get("mes", {}) or {}
        self._chk_enabled.setChecked(bool(mes.get("enabled", False)))
        self._line_ip.setText(mes.get("ip", "172.16.100.18"))
        self._spin_port.setValue(int(mes.get("port", 7010)))
        self._line_site.setText(mes.get("stationCode", ""))
        self._line_employee.setText(mes.get("operator", ""))
        self._update_param_group_state()

    def _on_ok(self):
        """确定：校验并保存配置"""
        if self._chk_enabled.isChecked():
            ip = self._line_ip.text().strip()
            if not ip:
                QMessageBox.warning(self, "提示", "启用 MES 功能时，请输入 MES 服务器 IP 地址")
                return
            if not self._line_site.text().strip():
                QMessageBox.warning(self, "提示", "启用 MES 功能时，请输入站点编码")
                return
            if not self._line_employee.text().strip():
                QMessageBox.warning(self, "提示", "启用 MES 功能时，请输入员工号")
                return
        self._save_config()
        self.accept()

    def _save_config(self):
        """保存配置到 config.json"""
        try:
            self._manager.set("mes.enabled", self._chk_enabled.isChecked())
            self._manager.set("mes.ip", self._line_ip.text().strip())
            self._manager.set("mes.port", self._spin_port.value())
            self._manager.set("mes.stationCode", self._line_site.text().strip())
            self._manager.set("mes.operator", self._line_employee.text().strip())
            self._manager.save()
            log_info(f"MES 配置已保存: enabled={self._chk_enabled.isChecked()}, "
                     f"ip={self._line_ip.text().strip()}, port={self._spin_port.value()}")
        except Exception as e:  # noqa: BLE001
            log_error(f"保存 MES 配置失败: {e}")

    def _on_test_connection(self):
        """测试与 MES 服务器的连接"""
        from core.mes_client import MESClient, MESError

        ip = self._line_ip.text().strip()
        if not ip:
            QMessageBox.warning(self, "提示", "请先输入 MES 服务器 IP 地址")
            return

        self.btn_test.setEnabled(False)
        self.btn_test.setText("测试中...")
        QApplication.processEvents()

        try:
            client = MESClient(
                ip=ip,
                port=self._spin_port.value(),
                station_code=self._line_site.text().strip(),
                operator=self._line_employee.text().strip(),
            )
            ok = client.test_connection()
            if ok:
                QMessageBox.information(self, "测试连接", "MES 服务器连接成功")
            else:
                QMessageBox.warning(self, "测试连接", "MES 服务器连接失败，请检查 IP/端口/网络")
        except MESError as e:
            QMessageBox.warning(self, "测试连接", f"MES 服务器连接失败:\n{e}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "测试连接", f"测试连接异常:\n{e}")
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("测试连接")

    def set_config(self, cfg: dict):
        """把已有配置回填到界面控件（兼容旧调用）"""
        self._chk_enabled.setChecked(bool(cfg.get("enabled", False)))
        self._line_ip.setText(cfg.get("ip", ""))
        self._spin_port.setValue(cfg.get("port", 7010))
        self._line_site.setText(cfg.get("stationCode", ""))
        self._line_employee.setText(cfg.get("operator", ""))
        self._update_param_group_state()

    def get_config(self) -> dict:
        """读取界面上所有配置，返回字典，供外部程序使用"""
        return {
            "enabled": self._chk_enabled.isChecked(),
            "ip": self._line_ip.text().strip(),
            "port": self._spin_port.value(),
            "stationCode": self._line_site.text().strip(),
            "operator": self._line_employee.text().strip()
        }


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 测试弹出MES对话框
    dlg = MESDialog()
    # 测试默认填入当前MES地址
    default_cfg = {
        "enabled": True,
        "ip": "172.16.100.18",
        "port": 7010,
        "stationCode": "RK01",
        "operator": "15983"
    }
    dlg.set_config(default_cfg)

    # exec_() 弹出模态对话框
    ret_code = dlg.exec_()
    if ret_code == QDialog.Accepted:
        conf = dlg.get_config()
        print("用户确认MES配置：", conf)

    sys.exit(app.exec())
