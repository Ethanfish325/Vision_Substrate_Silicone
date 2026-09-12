# -*- coding: utf-8 -*-
"""
托盘放入传感器（下料感应 unload_sensor）电平检测 Demo
====================================================
连接 NMC1400 控制卡，读取「下料感应」传感器端口的原始电平，
用于确定托盘放入/取出时传感器的电平是"触点闭合(0)"还是"触点断开(1)"。

用法:
    python tests/test_tray_sensor.py

功能:
    1. 从产品配置读取 unload_sensor 端口号（默认 DX8000_PCBA.json）
    2. 轮询显示该端口电平，放入/取出托盘观察电平变化
    3. 打印托盘放入时的电平状态，供配置判断逻辑使用

操作:
    运行后按提示放入/取出托盘，观察电平变化。按 Ctrl+C 退出。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.controller import Controller, ControllerError  # noqa: E402

PRODUCT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "products", "DX8000_PCBA.json",
)


def get_unload_sensor_port():
    """从产品配置读取 unload_sensor 端口号（1-based → 0-based）。"""
    try:
        with open(PRODUCT_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        io = cfg.get("io", {}) or {}
        in_num = io.get("unload_sensor")
        if in_num:
            return max(0, int(in_num) - 1)
    except Exception as e:
        print(f"[警告] 读取产品配置失败: {e}")
    return None


def main():
    print("=" * 60)
    print("托盘放入传感器（下料感应）电平检测 Demo")
    print("=" * 60)
    print("控制卡: NMC1400（网口自动发现，无需 IP）")

    port = get_unload_sensor_port()
    if port is None:
        print("[警告] 未在产品配置中找到 unload_sensor 端口，默认使用端口 7 (IN8)")
        port = 7
    print(f"下料感应端口: IN{port + 1} (端口 {port})")
    print()

    # 连接控制卡
    print("正在连接控制卡...")
    ctrl = Controller()
    try:
        ctrl.connect()
    except ControllerError as e:
        print(f"连接失败: {e}")
        return
    print(f"连接成功! 控制器状态: {ctrl.get_state_desc()}")
    print()

    # 检查 DLL 是否已加载
    try:
        if not ctrl._dll.is_loaded():
            print("[警告] MCDLL_NET.dll 未加载，无法读取 IO 电平！")
            ctrl.disconnect()
            return
        print("[OK] MCDLL_NET.dll 已加载，按位读取 DI 函数可用")
    except Exception as e:
        print(f"[警告] 检查 IO 函数失败: {e}")

    print()
    print("=" * 60)
    print("开始轮询检测下料感应端口电平...")
    print("请按提示操作，观察电平变化。按 Ctrl+C 退出。")
    print("NMC1400 电平定义: 0=触点闭合(有效)  1=触点断开")
    print("-" * 60)

    # 读取初始电平（原始电平）
    try:
        initial = ctrl.read_in_port_raw(port)
    except Exception as e:
        print(f"读取初始电平失败: {e}")
        ctrl.disconnect()
        return
    print(f"初始电平: {'触点闭合(0)' if initial == 0 else '触点断开(1)'}")
    print()

    prev = initial
    poll_interval = 0.1

    try:
        while True:
            try:
                current = ctrl.read_in_port_raw(port)
            except Exception as e:
                print(f"读取失败: {e}")
                time.sleep(poll_interval)
                continue

            if current != prev:
                state = "触点闭合(0)" if current == 0 else "触点断开(1)"
                action = "放入托盘" if current == 0 else "取出托盘"
                print(f"[变化] IN{port + 1}: {'断开→闭合' if current == 0 else '闭合→断开'} → {state}  ({action})")
                prev = current
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("检测结束。")
        print()
        print("请根据观察结果确定托盘放入时的电平：")
        print("  - 若放入托盘时端口为【触点闭合(0)】，则 read_in_port(port) 返回 True（默认 INPUT_ACTIVE_LOW=True）")
        print("  - 若放入托盘时端口为【触点断开(1)】，则需要把 core/controller.py 的")
        print("    Controller.INPUT_ACTIVE_LOW 改成 False")
        print("=" * 60)
        ctrl.disconnect()


if __name__ == "__main__":
    main()
