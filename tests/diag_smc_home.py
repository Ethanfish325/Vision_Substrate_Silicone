# -*- coding: utf-8 -*-
"""
诊断脚本：验证修正后的 Motion_Home_IfHoming / Motion_CheckDown 封装。

背景：这两个函数实为"直接返回值"类型（0=否,1=是），而非"返回错误码+指针输出"。
本脚本通过 SMCSHDLL 封装类验证修正后的调用能正确读取状态。
"""
import ctypes
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.smcsh_dll import SMCSHDLL  # noqa: E402


def main():
    dll = SMCSHDLL("smcsh_mbs.dll")
    err, handle = dll.open_eth("192.168.1.11")
    print(f"SMCOpenEth -> err={err}, handle={handle.value}")
    if err != 0:
        print("连接失败，无法继续诊断")
        return

    axis = 0

    # 通过修正后的封装方法读取状态
    print("\n=== 修正后封装方法验证 ===")
    try:
        down = dll.check_down(handle, axis)
        print(f"  check_down(轴{axis}) -> {down}  (0=未停止, 1=已停止)")
    except Exception as e:
        print(f"  check_down 异常: {e}")

    try:
        homing = dll.home_if_homing(handle, axis)
        print(f"  home_if_homing(轴{axis}) -> {homing}  (0=不在回零, 1=回零中)")
    except Exception as e:
        print(f"  home_if_homing 异常: {e}")

    try:
        reason = dll.get_stop_reason(handle, axis)
        print(f"  get_stop_reason(轴{axis}) -> {reason}")
    except Exception as e:
        print(f"  get_stop_reason 异常: {e}")

    dll.close(handle)
    print("\n诊断完成")


if __name__ == "__main__":
    main()
