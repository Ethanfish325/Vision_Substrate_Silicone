# -*- coding: utf-8 -*-
"""
NMC1400 控制卡连机自检脚本
==========================
在现场一次性确认运动控制卡的底层是否正常（连接、读写、轴状态、IO、运动）。

用法（必须用 32 位 Python 运行）:
    python tests/test_nmc_smoke.py                 # 只读自检（不动机器）
    python tests/test_nmc_smoke.py --jog 0 2000    # 轴0 JOG 2000 脉冲/s 直到回车停止
    python tests/test_nmc_smoke.py --move 0 1000   # 轴0 相对移动 1000 脉冲
    python tests/test_nmc_smoke.py --move-abs 0 0  # 轴0 绝对定位到 0
    python tests/test_nmc_smoke.py --home 0        # 轴0 回零（需确认回零模式！）
    python tests/test_nmc_smoke.py --servo 0 on    # 轴0 伺服使能（on/off）
    python tests/test_nmc_smoke.py --light ok      # 指示灯：ok=绿灯 ng=红灯 busy=黄灯 off=灭灯

安全提示:
    * 不带参数时只做只读自检，不会让任何轴运动。
    * 任何会动轴的操作都需要现场确认机械行程与限位，必要时先按急停。
    * 回零前务必按现场机械/开关情况确认 core/controller.py 里
      DEFAULT_HOME_PARAMS / main_window.HOME_PARAMS 的 home_mode 与触发电平。
"""
import argparse
import os
import sys
import time

# Windows 控制台默认 GBK，_reconfigure_ 避免特殊符号（⚠ ✓ ✗）打印时抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.controller import Controller, ControllerError, DEFAULT_HOME_PARAMS  # noqa: E402
from core.nmc_sdk import AXIS_STATE_DESC, STATION_TYPE_NAMES  # noqa: E402

LINE = "-" * 66


def print_header(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def dump_readonly(ctrl):
    """只读自检：控制卡信息 + 4 个轴状态 + IO 电平。"""
    print_header("只读自检（不会让任何轴运动）")

    try:
        conn, numbers, types = ctrl._dll.get_open_net()
        for num, tp in zip(numbers, types):
            print(f"  站号 {num}: 类型 {tp} -> {STATION_TYPE_NAMES.get(tp, '未知')}")
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] 读取打开参数失败: {e}")

    try:
        print(f"  固件版本: {ctrl.get_version()}   序列号: {ctrl.get_serial_number()}")
        print(f"  运行时间: {ctrl.get_run_time()} s")
        print(f"  链接状态: {'正常' if ctrl.check_link() else '异常'}")
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] 读取系统信息失败: {e}")

    print()
    print(f"  {'轴':<4}{'状态':<6}{'说明':<16}{'规划位置':>12}{'编码器':>12}{'速度':>12}{'伺服':>6}")
    print("  " + LINE)
    for axis in range(ctrl.get_axises()):
        try:
            state = ctrl.get_axis_state(axis)
            pos = ctrl.get_pulse_position(axis)
            enc = ctrl.get_encoder_position(axis)
            vel = ctrl.get_cur_speed(axis)
            servo = ctrl.get_servo_enable(axis)
            desc = AXIS_STATE_DESC.get(state, "未知")
            print(f"  {axis:<4}{state:<6}{desc:<16}{pos:>12}{enc:>12}{vel:>12.0f}{servo:>6}")
        except Exception as e:  # noqa: BLE001
            print(f"  {axis:<4}[读取失败] {e}")

    print()
    try:
        raw = [ctrl.read_in_port_raw(i) for i in range(16)]
        print("  DI00-DI15 原始电平 (0=触点闭合, 1=触点断开):")
        print("    " + " ".join(f"DI{i:02d}={v}" for i, v in enumerate(raw)))
        active = [i for i, v in enumerate(raw) if v == 0]
        print("    当前为触点闭合(有效)的输入: " + (str(["IN%d" % (i + 1) for i in active]) if active else "无"))
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] 读取 DI 失败: {e}")

    try:
        out = ctrl._dll.get_output(ctrl.station)
        print(f"  DO 输出电平字: 0x{out:08X}")
    except Exception as e:  # noqa: BLE001
        print(f"  [警告] 读取 DO 失败: {e}")

    print()
    for axis in range(ctrl.get_axises()):
        try:
            profile = ctrl.get_motion_params(axis)
            soft = ctrl.get_soft_limit(axis)
            print(f"  轴{axis} 曲线: 启动={profile['start_speed']:.0f} 目标={profile['max_speed']:.0f} "
                  f"加速={profile['acc']:.0f} 减速={profile['dec']:.0f} "
                  f"曲线={'S' if profile['profile'] == 1 else 'T'} | "
                  f"软限位: 正={soft[0]} 负={soft[1]} 使能={soft[2]}")
        except Exception as e:  # noqa: BLE001
            print(f"  轴{axis} [读取参数失败] {e}")


def do_jog(ctrl, axis, speed):
    """JOG 运动，回车停止。"""
    print_header(f"JOG 轴{axis} 速度 {speed} 脉冲/s（回车停止）")
    input("确认机械行程安全后按回车开始...")
    ctrl.set_motion_params(axis, min(1000, speed), speed, max(speed * 10, 10000),
                           max(speed * 10, 10000), 0)
    ctrl.vmove(axis, positive=True, speed=speed)
    print("  正在正转，按回车停止...")
    try:
        input()
    finally:
        ctrl.imd_stop(axis)
    time.sleep(0.2)
    print(f"  已停止，当前位置: {ctrl.get_pulse_position(axis)}")


def do_move(ctrl, axis, dist, absolute=False):
    """点位运动。"""
    kind = "绝对定位" if absolute else "相对移动"
    print_header(f"{kind} 轴{axis} -> {dist} 脉冲")
    input("确认机械行程安全后按回车开始...")
    ctrl.set_motion_params(axis, 1000, 20000, 50000, 50000, 0)
    if absolute:
        ctrl.pmove_abs(axis, dist)
    else:
        ctrl.pmove_rel(axis, dist)
    t0 = time.time()
    while not ctrl.check_down(axis):
        if time.time() - t0 > 30:
            print("  [警告] 运动超时（30s），执行立即停止")
            ctrl.imd_stop(axis)
            break
        time.sleep(0.05)
    print(f"  到位: 位置={ctrl.get_pulse_position(axis)} 编码器={ctrl.get_encoder_position(axis)} "
          f"轴状态={ctrl.get_axis_state_desc(axis)}")


def do_home(ctrl, axis):
    """回零。"""
    print_header(f"回零 轴{axis}")
    print("  当前回零参数: " + ", ".join(f"{k}={v}" for k, v in DEFAULT_HOME_PARAMS.items()))
    print("  ⚠️ 回零方向/模式选错会让轴往错误方向找原点！")
    if input("确认回零模式与触发电平正确后输入 yes 继续: ").strip().lower() != "yes":
        print("  已取消。")
        return
    ctrl.set_home_params(axis, **DEFAULT_HOME_PARAMS)
    ctrl.home_move(axis)
    t0 = time.time()
    while True:
        state = ctrl.get_home_state(axis)
        if state == 0:
            print(f"  回零成功，位置={ctrl.get_pulse_position(axis)}")
            break
        if state == 31:
            print("  回零错误（状态码 31），请检查模式/触发电平/传感器接线")
            break
        if time.time() - t0 > 60:
            print("  [警告] 回零超时（60s），停止回零")
            ctrl.home_stop(axis)
            ctrl.imd_stop(axis)
            break
        time.sleep(0.1)


def do_servo(ctrl, axis, on):
    ctrl.set_servo_enable(axis, on)
    print(f"  轴{axis} 伺服{'使能' if on else '关闭'}完成，"
          f"当前 Servo_Logic={ctrl.get_servo_enable(axis)}")


def do_light(ctrl, mode):
    mapping = {
        "ok": (False, True),
        "ng": (True, False),
        "busy": (True, True),
        "off": (False, False),
    }
    red, green = mapping[mode]
    ctrl.set_light_state(red, green)
    print(f"  指示灯已设置: 红={red} 绿={green}（{mode}）")


def check_write_path(ctrl):
    """写入有效性自检。

    ⚠️ 现场常见问题：控制卡同一时刻只接受**一个**控制端。
       如果同时开着本程序 / 官方调试软件 / 另一个测试进程，
       后开的那个会"能读不能写"——所有写函数都返回 0，但实际不生效，
       运动指令也不会执行。这里用"写脉冲模式 -> 读回"来快速判断。
    """
    print_header("写入有效性自检")
    axis = 0
    ok = True

    # 写"当前位置 +1"再读回，最后恢复原值：
    # 位置计数器只是坐标（不会产生任何机械运动），是风险最小、又能真正判定
    # "写操作有没有落到卡上"的检测方式。
    pos0 = None
    try:
        pos0 = ctrl.get_pulse_position(axis)
        if ctrl.get_axis_status(axis)["moving"]:
            print("  轴正在运动，跳过写入自检（避免干扰运动）")
            return True

        ctrl._dll.set_position(axis, pos0 + 1, ctrl.station)
        pos1 = ctrl.get_pulse_position(axis)
        ctrl._dll.set_position(axis, pos0, ctrl.station)   # 立即恢复
        pos2 = ctrl.get_pulse_position(axis)

        ok = (pos1 == pos0 + 1) and (pos2 == pos0)
        print(f"  规划位置 写入 {pos0} -> {pos0 + 1}: 读回={pos1} "
              f"{'OK' if pos1 == pos0 + 1 else 'X 写入未生效！'}")
        print(f"  规划位置 恢复 {pos0}: 读回={pos2} "
              f"{'OK' if pos2 == pos0 else 'X 恢复失败！'}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [错误] 位置读写失败: {e}")
        if pos0 is not None:
            try:
                ctrl._dll.set_position(axis, pos0, ctrl.station)
            except Exception:  # noqa: BLE001
                pass

    if not ok:
        print()
        print("  [!] 写入未生效：请确认没有第二个程序（本程序 / 官方调试软件 / 其它脚本）")
        print("      同时占用控制卡——控制卡同一时刻只接受一个控制端。")
    return ok


def main():
    parser = argparse.ArgumentParser(description="NMC1400 控制卡连机自检")
    parser.add_argument("--jog", nargs=2, metavar=("AXIS", "SPEED"), help="JOG 运动")
    parser.add_argument("--move", nargs=2, metavar=("AXIS", "DIST"), help="相对移动")
    parser.add_argument("--move-abs", nargs=2, metavar=("AXIS", "POS"), help="绝对定位")
    parser.add_argument("--home", metavar="AXIS", help="回零")
    parser.add_argument("--servo", nargs=2, metavar=("AXIS", "ONOFF"), help="伺服使能 on/off")
    parser.add_argument("--light", choices=["ok", "ng", "busy", "off"], help="指示灯")
    parser.add_argument("--timeout", type=int, default=0,
                        help="链接超时(ms)，0=不设置（默认，推荐）；"
                             "⚠️ 设置后断链会让控制卡锁存急停，必须断电重启")
    args = parser.parse_args()

    print_header("NMC1400 控制卡连机自检")
    ctrl = Controller()
    try:
        ctrl.connect(args.timeout)
    except ControllerError as e:
        print(f"连接失败: {e}")
        return 1
    print(f"  连接成功: {ctrl.conn_string}   轴数: {ctrl.get_axises()}   "
          f"控制器状态: {ctrl.get_state_desc()}")

    try:
        dump_readonly(ctrl)
        check_write_path(ctrl)

        if args.servo:
            do_servo(ctrl, int(args.servo[0]), args.servo[1].lower() in ("on", "1", "true", "yes"))
        if args.light:
            do_light(ctrl, args.light)
        if args.jog:
            do_jog(ctrl, int(args.jog[0]), float(args.jog[1]))
            time.sleep(0.3)
            dump_readonly(ctrl)
        if args.move:
            do_move(ctrl, int(args.move[0]), float(args.move[1]), absolute=False)
        if args.move_abs:
            do_move(ctrl, int(args.move_abs[0]), float(args.move_abs[1]), absolute=True)
        if args.home:
            do_home(ctrl, int(args.home))
    except ControllerError as e:
        print(f"[错误] {e}")
        return 2
    except KeyboardInterrupt:
        print("\n手动中断，正在停止所有轴...")
        ctrl.stop_all()
    finally:
        ctrl.disconnect()
    print()
    print("自检结束，控制卡已断开。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
