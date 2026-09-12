# -*- coding: utf-8 -*-
"""
控制卡 IO 电平检测测试 Demo（扫描所有输入端口 + 按键映射扫描）
============================================================
连接 NMC1400 控制卡，扫描所有输入端口（DI00-DI15）的原始电平，
用于确定按钮实际对应的端口号，并验证 IO 电平检测是否正常。

NMC1400 输入电平定义（编程手册 2.5）：
    0 = 触点闭合（硬件灯亮，信号有效）
    1 = 触点断开（硬件灯灭）

用法:
    python tests/test_io_demo.py

功能:
    1. 扫描模式：打印所有端口的初始电平，按下/松开按钮观察电平变化。
    2. 按键映射模式：按顺序提示按下每个按钮，自动检测并记录端口号。

操作:
    运行后选择模式，然后按提示操作。按 Ctrl+C 退出。

注意:
    上层工作流按"上升沿 = 按下"判断，因此 core/controller.py 里的
    Controller.INPUT_ACTIVE_LOW 默认把"0=触点闭合"反相成有效(True)。
    若本 Demo 实测按下的端口读回 1（触点断开为按下），请把该常量改成 False。
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.controller import Controller, ControllerError  # noqa: E402

MAX_PORT = 16  # NMC1400 共 DI00-DI15

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
    """读取所有输入端口的原始电平，返回 {port: int|None}（0=触点闭合, 1=触点断开）。"""
    result = {}
    for port in range(MAX_PORT):
        try:
            result[port] = ctrl.read_in_port_raw(port)
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
    print("初始电平（0=触点闭合/有效, 1=触点断开, X=读取失败）:")
    line = ""
    for port in range(MAX_PORT):
        v = initial.get(port)
        s = str(v) if v is not None else "X"
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
                        state = "触点闭合(0)" if current[port] == 0 else "触点断开(1)"
                        print(f"[变化] IN{port + 1} (端口{port}): "
                              f"{'断开→闭合' if current[port] == 0 else '闭合→断开'} → {state}")
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
                            direction = "断开→闭合" if current[port] == 0 else "闭合→断开"
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
    print("控制卡: NMC1400（网口自动发现，无需 IP）")
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

    # 检查 DLL 是否绑定了读取输入端口函数
    try:
        if not ctrl._dll.is_loaded():
            print("[警告] MCDLL_NET.dll 未加载，无法读取 IO 电平！")
            ctrl.disconnect()
            return
        print("[OK] MCDLL_NET.dll 已加载，按位读取 DI 函数可用")
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
