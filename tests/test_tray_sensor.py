# -*- coding: utf-8 -*-
"""
托盘放入传感器（下料感应 unload_sensor）电平检测 Demo
====================================================
连接 SMC6480 控制卡，读取「下料感应」传感器端口的电平，
用于确定托盘放入/取出时传感器是高电平还是低电平。

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

DEFAULT_IP = "192.168.1.11"
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
    print(f"目标 IP: {DEFAULT_IP}")

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
        ctrl.connect_eth(DEFAULT_IP)
    except ControllerError as e:
        print(f"连接失败: {e}")
        return
    print(f"连接成功! 控制器状态: {ctrl.get_state_desc()}")
    print()

    # 检查 DLL 是否绑定了读取输入端口函数
    try:
        dll = ctrl._dll
        if getattr(dll, '_in_port_func', None) is None:
            print("[警告] DLL 中未找到读取输入端口(DI)的函数，无法读取 IO 电平！")
            ctrl.disconnect()
            return
        else:
            print("[OK] 已绑定读取输入端口函数")
    except Exception as e:
        print(f"[警告] 检查 IO 函数失败: {e}")

    print()
    print("=" * 60)
    print("开始轮询检测下料感应端口电平...")
    print("请按提示操作，观察电平变化。按 Ctrl+C 退出。")
    print("-" * 60)

    # 读取初始电平
    try:
        initial = ctrl.read_in_port(port)
    except Exception as e:
        print(f"读取初始电平失败: {e}")
        ctrl.disconnect()
        return
    print(f"初始电平: {'高电平(1)' if initial else '低电平(0)'}")
    print()

    prev = initial
    poll_interval = 0.1

    try:
        while True:
            try:
                current = ctrl.read_in_port(port)
            except Exception as e:
                print(f"读取失败: {e}")
                time.sleep(poll_interval)
                continue

            if current != prev:
                state = "高电平(1)" if current else "低电平(0)"
                action = "放入托盘" if current else "取出托盘"
                print(f"[变化] IN{port + 1}: {'低→高' if current else '高→低'} → {state}  ({action})")
                prev = current
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("检测结束。")
        print()
        print("请根据观察结果确定托盘放入时的电平：")
        print("  - 若放入托盘时端口为【高电平(1)】，则判断条件为: read_in_port(port) == True")
        print("  - 若放入托盘时端口为【低电平(0)】，则判断条件为: read_in_port(port) == False")
        print("=" * 60)
        ctrl.disconnect()


if __name__ == "__main__":
    main()
