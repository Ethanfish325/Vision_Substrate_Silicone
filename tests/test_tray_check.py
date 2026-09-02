# -*- coding: utf-8 -*-
"""测试托盘放入判断逻辑（_is_tray_present）。

验证:
    1. 托盘放入时（高电平 active_high=True）返回 True
    2. 托盘未放入时（低电平 active_high=True）返回 False
    3. active_high=False 时电平反转
    4. 未配置传感器时默认返回 True（不阻塞）
"""
import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inspection_workflow import InspectionWorkflow


class FakeController:
    """模拟控制器，可设置输入端口电平"""
    def __init__(self):
        self._levels = {}

    def set_in_level(self, port, level):
        self._levels[port] = level

    def read_in_port(self, port):
        return self._levels.get(port, False)


def test_tray_present_active_high():
    """托盘放入时高电平（active_high=True）"""
    wf = InspectionWorkflow()
    ctrl = FakeController()
    wf._controller = ctrl
    wf._io_ports = {"unload_sensor": 7}
    wf._tray_sensor_active_high = True

    # 托盘放入（高电平）
    ctrl.set_in_level(7, True)
    assert wf._is_tray_present() is True, "托盘放入(高电平)应返回 True"
    print("[PASS] active_high=True，托盘放入(高电平) → True")

    # 托盘未放入（低电平）
    ctrl.set_in_level(7, False)
    assert wf._is_tray_present() is False, "托盘未放入(低电平)应返回 False"
    print("[PASS] active_high=True，托盘未放入(低电平) → False")


def test_tray_present_active_low():
    """托盘放入时低电平（active_high=False）"""
    wf = InspectionWorkflow()
    ctrl = FakeController()
    wf._controller = ctrl
    wf._io_ports = {"unload_sensor": 7}
    wf._tray_sensor_active_high = False

    # 托盘放入（低电平）
    ctrl.set_in_level(7, False)
    assert wf._is_tray_present() is True, "托盘放入(低电平)应返回 True"
    print("[PASS] active_high=False，托盘放入(低电平) → True")

    # 托盘未放入（高电平）
    ctrl.set_in_level(7, True)
    assert wf._is_tray_present() is False, "托盘未放入(高电平)应返回 False"
    print("[PASS] active_high=False，托盘未放入(高电平) → False")


def test_tray_no_sensor():
    """未配置传感器时默认返回 True（不阻塞）"""
    wf = InspectionWorkflow()
    wf._controller = None
    wf._io_ports = {}
    assert wf._is_tray_present() is True, "未配置传感器应返回 True"
    print("[PASS] 未配置传感器 → True（不阻塞）")


if __name__ == "__main__":
    test_tray_present_active_high()
    test_tray_present_active_low()
    test_tray_no_sensor()
    print("\n[PASS] 托盘判断逻辑测试通过")
