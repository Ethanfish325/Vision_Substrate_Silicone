# -*- coding: utf-8 -*-
"""
控制卡 IO 电平检测测试 Demo（扫描所有输入端口 + 按键映射扫描）
============================================================
连接 SMC6480 控制卡，扫描所有输入端口（DI）的电平，
用于确定按钮实际对应的端口号，并验证 IO 电平检测是否正常。

用法:
    python test_io_demo.py

功能:
    1. 扫描模式：打印所有端口的初始电平，按下/松开按钮观察电平变化。
    2. 按键映射模式：按顺序提示按下每个按钮，自动检测并记录端口号。

操作:
    运行后选择模式，然后按提示操作。按 Ctrl+C 退出。
"""
import sys
import time

from core.controller import Controller, ControllerError

DEFAULT_IP = "192.168.1.11"
MAX_PORT = 16  # 扫描 0-15 端口

# 需要扫描的按钮名称（按顺序提示用户按下）
BUTTONS = [
    "启动(start)",
    "停止(stop)",
    "复位(reset)",
    "复判OK(rejudge_ok)",
    "复判NG(rejudge_ng)",
    "下料感应(unload_sensor)",
    "下料按钮(unload_btn)",
]


def read_all_ports(ctrl):
    """读取所有输入端口电平，返回 {port: bool}。"""
    result = {}
    for port in range(MAX_PORT):
        try:
            result[port] = ctrl.read_in_port(port)
        except Exception:
            result[port] = None  # 读取失败
    return result


def scan_mode(ctrl):
    """扫描模式：打印初始电平，轮询检测电平变化。"""
    print()
    print("=" * 60)
    print("扫描模式：检测所有端口电平变化")
    print("=" * 60)

    # 打印初始电平
    initial = read_all_ports(ctrl)
    print("初始电平（1=高电平, 0=低电平, X=读取失败）:")
    line = ""
    for port in range(MAX_PORT):
        v = initial.get(port)
        s = "1" if v else ("0" if v is not None else "X")
        line += f"IN{port + 1}:{s}  "
        if (port + 1) % 4 == 0:
            print(line)
            line = ""
    if line:
        print(line)

    print()
    print("开始轮询检测电平变化...")
    print("请按下/松开按钮，观察哪个端口电平变化。按 Ctrl+C 退出。")
    print("-" * 60)

    prev = initial
    poll_interval = 0.05

    try:
        while True:
            current = read_all_ports(ctrl)
            for port in range(MAX_PORT):
                if current.get(port) is not None and prev.get(port) is not None:
                    if current[port] != prev[port]:
                        state = "高电平(1)" if current[port] else "低电平(0)"
                        print(f"[变化] IN{port + 1} (端口{port}): "
                              f"{'低→高' if current[port] else '高→低'} → {state}")
            prev = current
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print()
        print("扫描结束。")


def mapping_mode(ctrl):
    """按键映射模式：按顺序提示按下每个按钮，自动检测并记录端口号。"""
    print()
    print("=" * 60)
    print("按键映射模式：逐个按下按钮，自动检测端口号")
    print("=" * 60)
    print("将按顺序提示你按下每个按钮。")
    print("按下按钮后，程序会检测哪个端口电平变化（上升沿），并记录该端口号。")
    print("按 Ctrl+C 可随时退出。")
    print("-" * 60)

    mapping = {}  # {按钮名: 端口号}
    poll_interval = 0.05

    try:
        for btn in BUTTONS:
            print()
            print(f"▶ 请按下按钮: 【{btn}】 (按下后松开)", flush=True)
            print("   等待检测...", flush=True)

            # 记录初始电平，等待按钮按下（检测到任意端口电平变化）
            base = read_all_ports(ctrl)
            detected = False
            while not detected:
                current = read_all_ports(ctrl)
                for port in range(MAX_PORT):
                    if current.get(port) is not None and base.get(port) is not None:
                        # 检测电平变化（相对初始电平）
                        if current[port] != base[port]:
                            direction = "低→高" if current[port] else "高→低"
                            mapping[btn] = port
                            print(f"   ✅ 检测到 [{btn}] → IN{port + 1} (端口{port}) [{direction}]", flush=True)
                            detected = True
                            break
                time.sleep(poll_interval)

            # 等待按钮松开（该端口回到初始电平），避免误判下一个按钮
            print("   请松开按钮，等待...", flush=True)
            port = mapping[btn]
            while True:
                current = read_all_ports(ctrl)
                if current.get(port) is not None and current[port] == base[port]:
                    break
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print()
        print("手动中断。")

    # 打印结果
    print()
    print("=" * 60)
    print("按键映射扫描结果")
    print("=" * 60)
    if not mapping:
        print("未检测到任何按钮。")
        return

    print(f"{'按钮':<20} {'端口号(0-based)':<18} {'IN编号(1-based)'}")
    print("-" * 50)
    for btn, port in mapping.items():
        print(f"{btn:<20} {port:<18} IN{port + 1}")

    print()
    print("请根据以上结果更新 data/products/DX8000_PCBA.json 的 io 字段。")
    print("注意：io 字段使用 1-based IN 编号（如 IN2 填 2）。")


def main():
    print("=" * 60)
    print("控制卡 IO 电平检测 Demo")
    print("=" * 60)
    print(f"目标 IP: {DEFAULT_IP}")
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

    # 选择模式
    print()
    print("请选择模式:")
    print("  1. 扫描模式（观察所有端口电平变化）")
    print("  2. 按键映射模式（逐个按下按钮，自动记录端口号）")
    try:
        choice = input("请输入 1 或 2: ").strip()
    except EOFError:
        choice = "2"

    if choice == "1":
        scan_mode(ctrl)
    else:
        mapping_mode(ctrl)

    ctrl.disconnect()
    print()
    print("已断开连接。")


if __name__ == "__main__":
    main()
