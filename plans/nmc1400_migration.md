# SMC6480 → NMC1400 运动控制卡迁移说明

> 迁移完成时间：本次提交
> 目标：**界面（UI）保持不变**，只替换底层运动控制逻辑。

## 一、总体结构

```
ui/widgets/nmc_dialog.py  轴控制面板（界面不变，只把底层调用换成真实接口）
ui/main_window.py         回零流程 / 指示灯 / 控制器注入
        │
        ▼
core/controller.py        业务层 Controller（对外接口与旧版保持一致）
        │
        ▼
core/nmc_sdk.py           NMCSDK —— MCDLL_NET.dll 的 ctypes 封装
        │
        ▼
MCDLL_NET.dll             NMC1400（摩升泰）4 轴网络总线运动控制卡

core/smcsh_dll.py         【遗留】SMC6480 旧卡封装，保留但已不再被引用
smcsh_mbs.dll             【遗留】SMC6480 旧卡库
```

## 二、底层坑位（已核对并写进代码注释）

用反汇编 `MCDLL_NET.dll` + 对照《NMCxxxx 编程手册》逐条核对，以下几处**必须**注意：

| 项目 | 结论 | 说明 |
|------|------|------|
| 调用约定 | **`__stdcall`（32 位）** | 必须用 `ctypes.WinDLL` 加载；用 `CDLL`(cdecl) 会导致每次调用后栈指针错位，表现为随机崩溃 |
| `MCF_Uniaxial_Net` 的距离 | **`int32`，不是 `double`** | 手册老版本写的是 `double`，但本 DLL 反汇编为 `fild dword ptr [ebp+0Ch]` + `ret 0x10`，实际是 4 字节整数。传 `double` 会让目标位置变成随机值 |
| `MCF_JOG_Net` / `MCF_Set_Axis_Profile_Net` / `MCF_Search_Home_Set_Net` 的速度、加速度、加加速度 | `double`（8 字节） | `MCF_Set_Axis_Profile_Net` 弹栈 52 字节 = 4 + 5×8 + 4 + 4，已确认 |
| `MCF_Get_Input_Net` | **两个 `uint32` 输出**（DI00-DI31 / DI32-DI47） | 不是 `uint64` |
| `MCF_Set_Output_Net` / `MCF_Get_Output_Net` | **`uint32`** | 不是 `uint64` |
| `MCF_Get_Link_State_Net` | 带 1 个站号参数 | 0=正常，负数=错误码 |
| `MCF_Open_Net` | 站类型 `2` = NMC1400 | 成功返回 0，不需要 IP（DLL 通过网口自动发现） |

## 三、运动函数的返回值语义（实测，非常重要）

`MCF_Uniaxial_Net` / `MCF_JOG_Net` 这类**运动函数**的返回值和其它函数不一样：

| 返回值 | 含义 | 处理 |
|--------|------|------|
| `0` | 命令已下发成功 | 正常 |
| `1` | 轴**正在执行** → **命令被拒绝**（没有生效） | 等轴空闲后重试 |
| `2~28` | 轴因急停/报警/限位等原因停止 → 命令被拒绝 | 报错，提示具体原因 |
| `<0` | DLL 错误码（如 -17 链接中断） | 报错 |

> 一开始把"非 0 即错误"处理，导致点「绝对定位」到当前位置时弹出
> 「绝对定位失败 (轴0 -> 0): 正在执行」——其实 `1` 不是错误码，是"轴忙、命令没生效"。

### ⚠️ "0 距离点位运动"会把轴永久卡死（实测）

给 NMC1400 下发**目标 = 当前位置**的点位运动（0 距离）后：

- 轴会一直停在"正在执行(1)"状态，**位置和速度都不动，但永远不会结束**；
- `MCF_Axis_Stop_Net`（立即/减速停止）**解不开**；
- 之后所有运动命令都被拒绝（返回 1）；
- 只有 `MCF_Clear_Axis_State_Net` 能把它复位。

处理办法（已实现在 `core/controller.py`）：

1. `pmove_abs` / `pmove_rel`：**目标等于当前位置（或相对距离为 0）时直接跳过**，不下发指令；
2. `_run_motion_command`：遇到返回 1 时按 `MOTION_RETRY` 重试，并在最后一次重试前调用
   `_reset_stuck_axis` 复位；
3. `_reset_stuck_axis`：只有确认**位置与速度都没有变化**时才 `Clear_Axis_State`，
   真的在运动的轴不会被误清除。

## 四、旧→新 API 映射（业务层已封装，上层无需改）

| 旧 SMC6480 | 新 NMC1400 |
|------------|------------|
| `SMCOpenEth(ip)` | `MCF_Open_Net(站点数)`（自动发现，无需 IP） |
| `SMCClose` | `MCF_Close_Net` |
| `MSetting_SetStartSpeed/…Speed/Acc/Dec/S曲线` | `MCF_Set_Axis_Profile_Net` + `MCF_Set_Axis_Stop_Profile_Net` |
| `Motion_Pmove_Enter/SetAbsolute/Start` | `MCF_Uniaxial_Net(轴, 位置, 0)` |
| `Motion_Pmove_SetRelative` | `MCF_Uniaxial_Net(轴, 距离, 1)` |
| `Motion_Vmove_*` | `MCF_JOG_Net(轴, ±速度, 加速度)` |
| `Motion_DeclStop` / `Motion_ImdStop` | `MCF_Axis_Stop_Net(轴, 1 / 0)` |
| `Motion_CheckDown` | `MCF_Get_Axis_State_Net(轴) != 1` |
| `Motion_GetPulsePositon` / `GetEncoderPositon` | `MCF_Get_Position_Net` / `MCF_Get_Encoder_Net` |
| `Motion_SetPulsePositon` | `MCF_Set_Position_Net` |
| `Motion_GetCurSpeed` | `MCF_Get_Vel_Net`（取指令速度） |
| `Motion_Home_FindOrigin` | `MCF_Search_Home_Set_Net` + `MCF_Search_Home_Start_Net` |
| `Motion_Home_IfHoming` | `MCF_Search_Home_Get_State_Net == 32`（0=成功, 31=错误） |
| `SMCReadInBit` / `SMCWriteOutBit` | `MCF_Get_Input_Bit_Net` / `MCF_Set_Output_Bit_Net` |

## 五、现场必须核对的三处电平/模式（代码里都是常量，改一处即可）

| 项目 | 位置 | 当前默认 | 现象与改法 |
|------|------|----------|-----------|
| DI 有效电平 | `core/controller.py` → `Controller.INPUT_ACTIVE_LOW` | `True`（读回 0 = 触点闭合 = 按下） | 手册 2.5：0=触点闭合(硬件灯亮)，1=触点断开。若实测按下时读回 **1**，改成 `False` |
| 回零模式 | `ui/main_window.py` → `HOME_PARAMS["home_mode"]`；`ui/widgets/nmc_dialog.py` 用 `core/controller.py` 的 `DEFAULT_HOME_PARAMS` | `3`（原点开关，正方向找原点负边外侧）；沿用旧配置数值 | 必须按《NMCxxxx 编程手册》第 6 章「回原点模式选择参考表」选：3/19/20/21/22=原点开关，17/18=限位开关，33/34=Index。**选错会让轴往错误方向找原点，首次务必低速试** |
| 伺服使能极性 | `core/controller.py` → `Controller.SERVO_ENABLE_LOGIC` | `0`（Servo_Close，手册 3.1："0 表示使能端口输出低电平"） | 若驱动 SON 为高电平有效，改成 `Servo_Open` |
| 指示灯端口 | `core/controller.py` → `LIGHT_RED_PORT` / `LIGHT_GREEN_PORT` | `3` / `4`（DO03 红灯、DO04 绿灯，写 0 点亮） | 接线若不同，只改这两个常量 |

## 六、现场自检步骤

```bat
:: 必须用 32 位 Python（MCDLL_NET.dll 是 32 位）
"C:\Users\fyx\AppData\Local\Programs\Python\Python39-32\python.exe" tests\test_nmc_smoke.py
```

1. **只读自检**（默认，不会让轴运动）：确认连接、站号/类型、4 轴状态/位置/编码器/速度、DI/DO 电平、曲线参数与软限位。
2. **指示灯**：`--light ok|ng|busy|off`
3. **伺服**：`--servo 0 on`（先确认极性！）
4. **点动**：`--jog 0 2000`（回车停止）
5. **点位**：`--move 0 1000` / `--move-abs 0 0`
6. **回零**：`--home 0`（会二次确认；先用低速参数）
7. **按钮映射**：`tests/test_io_demo.py`（打印 DI00-DI15 原始电平，按按钮观察哪个端口变化）
8. **托盘传感器**：`tests/test_tray_sensor.py`

## 七、打包注意

- `MCDLL_NET.dll` 是 32 位库 → 必须用 32 位 Python 打包（`build_32.bat`）。
- `main.spec` 已改为打包 `MCDLL_NET.dll`；`*.dll` 在 `.gitignore` 中，需手动放到项目根目录。
- `runtime_hook.py` 会把 `_internal/`（开发环境为项目根目录）加入 DLL 搜索路径。

## 八、"轴状态不实时更新"的原因（手册 5.10）

`MCF_Get_Axis_State_Net` 读的是**报警/停止原因锁存寄存器**，不是实时的"运动/停止"标志：

> 报警原因寄存器需要通过专用的报警原因清除函数 `MCF_Clear_Axis_State_Net` 来清除寄存器，
> 否则 `MCF_Get_Axis_State_Net` 会一直保存上一次报警原因，除非轴进行新的动作或者新的报警，
> 报警原因寄存器才会刷新。——《NMCxxxx 编程手册》5.10

所以"运动完了状态还停在原来那格"是它本来的特性；旧卡 `Motion_CheckDown` 是实时标志，
两者不能直接划等号。已按此修正：

| 用途 | 现在的做法 |
|------|-----------|
| 判断轴是否在运动 | `Controller.get_axis_status()` / `check_down()`：用**实时速度**（`MCF_Get_Vel_Net` 的命令速度、编码器速度）判断 |
| 显示停止原因 | 锁存码为 2~28 时显示对应中文原因（限位/报警/急停…） |
| 控制器整体状态 | 有轴在动→运动中；否则有轴锁存了异常原因→停止；否则空闲。`22/23/28`（我们主动发的停止命令）视为正常 |

## 九、现场调试注意事项

1. **同一时刻只能有一个程序控制控制卡。**
   实测：当本程序（`main.py`）运行时，另开一个进程打开同一张卡——
   **读得到数据、但所有写操作都被静默忽略**（写函数照样返回 0），运动指令也不会执行。
   跑调试脚本前请先关掉主程序，反之亦然。

2. **`MCF_Set_Switch_State_Net`（网络串联/并联）不再在 `connect()` 里写了。**
   该设置保存在卡里，官方样例程序从不写它；现在只读取并记日志，避免误改现场配置。

3. **官方样例每次运动前的动作**（参考程序 `SLDMotion.cs`）：
   `MCF_Set_Pulse_Mode_Net(axis, Pulse_Dir_H)` + `MCF_Set_Servo_Enable_Net(axis, Servo_Close)`，
   再下发曲线、再定位。若现场出现"能读能写但轴不走"，可先按这个顺序手动执行一次。

4. **尚未完成的验证项**（当时设备还没接驱动器/电机，请接好后现场确认）：
   - `MCF_Set_Position_Net` / `MCF_Set_Encoder_Net` 返回 0，但读回不变（换新进程重开卡再读也一样，排除读缓存）；
   - 运动指令（相对/绝对/JOG）返回 0、轴状态变为"正在执行"，但规划位置、编码器、速度全程无变化；
   - `tests/test_nmc_smoke.py`（默认只读，含"写入有效性自检"）可快速确认现场状态。

5. **已逐项排除的原因**（2026-09-12 实测，供后续省时间）：
   | 怀疑点 | 实测结果 |
   |--------|----------|
   | 伺服使能极性（`MCF_Set_Servo_Enable_Net` 写 0 / 写 1） | 两种情况轴都不动，不是极性 |
   | 链接超时看门狗 | 关闭后仍不动；但**历史链接中断次数在涨**（20→31） |
   | 运动曲线参数（含手册示例值 0/5000/50000/500000） | 曲线写进去能读回，轴仍不动 |
   | 位置模式（绝对/相对）、JOG 速度环 | 都不动 |
   | `MCF_Sorting_Init_Net` 先于 `MCF_Open_Net` | 无改善 |
   | 正负限位 / 原点 / Index / 报警 触发模式 | 全部 `0`（关闭），没在挡 |
   | 通用 IO 触发停止（8 个通道 `MCF_Get_Input_Trigger_Net`） | 全部 `0`（关闭），没在挡 |
   | 硬件急停：`MCF_Set_EMG_Bit_Net(0, 0)` 关闭急停 | 无改善 |
   | 官方样例顺序：`Set_Pulse_Mode` + `Set_Servo_Enable(Servo_Close)` 后再运动 | 无改善 |
   | 电子齿轮 `MCF_Get_Gear_Enable_Net` | 4 轴全部 `0`（关闭） |
   | 缓冲区 `MCF_Buffer_Get_State_Net` | 空闲（状态 0） |
   | 软限位 | 已清零且未使能 |
   | 写操作整体不生效 | 软件限位 / 脉冲模式 / 曲线**都能写能读**；位置/编码器写不进去 —— **后来确认这也是"卡被锁死"的表现**，断电重启后位置写入完全正常 |
   | 网络/卡是否是"假的" | **不是**：4 秒读取期间网卡发出 3614 个单播、收回 3614 个广播应答（1:1 配对），确认有真实设备在应答；卡号=4，运行时间 11.2 小时 |

6. **结论：卡"收下命令但不发脉冲"** —— 运动指令返回 0、轴状态变"正在执行"，但
   规划位置 / 编码器 / 速度全程为 0，且不结束。
   **【2026-09-12 已定案】根因就是第十节的"链接超时看门狗锁存急停"**：
   控制卡断电重启后一切正常（连厂家官方调试软件在锁死状态下也点不动，印证是卡里的状态）。

7. **`connect()` 的链接超时看门狗现在默认为"不设置"**：
   `ctrl.connect()`（默认 `timeout_ms=0`）= 保持控制卡原有配置，和官方样例一致。

## 十、⚠️ 重大坑：链接超时看门狗会把控制卡锁死（已复现并修复）

### 现象
- 运动指令全部返回 0，`MCF_Get_Axis_State_Net` 变为 `1(正在执行)` 并**永久保持**；
- `MCF_Get_Vel_Net` 恒为 `0/0`，`MCF_Get_Position_Net` 恒为 0 且**写不进去**；
- `MCF_Axis_Stop_Net` 解不开，只有 `MCF_Clear_Axis_State_Net` 能复位状态位，但轴仍然不动；
- **连厂家官方调试软件也点不动**，最后**给控制卡断电重启才恢复**。

### 原因
`core/controller.py` 的 `connect()` 里调用了

```python
self._dll.set_link_timeout(1000, 0, self._station)   # 手册 1.3.1
```

手册 1.3.1 的 `MCF_Set_Link_TimeOut_Net` 是「**链接超时紧急停止所有轴**」。
DLL 只在程序运行期间维持链接，因此**程序一退出（或网线抖动、电脑休眠）就会触发一次
"链接超时紧急停止"，控制卡把这个急停状态锁存下来**：

- 日志里的「控制卡历史链接中断次数」就是每次断链的计数（实测 20 → 31 → 39 一路增长）；
- 锁存后所有运动都不执行，且**不报错**，极具迷惑性。

### 修复
1. `connect()` 默认 `timeout_ms=0`，**不再设置**看门狗（与官方样例一致）；
   确需该安全功能时显式传入，例如 `ctrl.connect(5000)`。
2. `disconnect()` 在 `MCF_Close_Net()` 之前**先写 `set_link_timeout(0, 0, 0)` 解除看门狗**，
   保证正常关程序不会把卡锁住。
3. `ui/main_window.py` 的 `closeEvent()` 补上了 `stop_all()` + `disconnect()`
   （之前关程序时**根本没有断开控制卡**）。

### 现场遇到"卡像是死的"时
先给控制卡**断电重启**；以后不要再开这个看门狗即可。

### 断电重启后的实测验证（2026-09-12 20:12，卡恢复正常）
| 验证项 | 结果 |
|--------|------|
| 链接中断次数 | 由 39 归零为 1（重启后只统计了本次会话） |
| 位置寄存器写入 | `写入 -1236 -> -1235: 读回=-1235 OK`，**完全正常**（卡被锁死时这个写是被忽略的） |
| 相对 +50 脉冲 | 规划位置 -1236 → -1186（Δ+50），峰值速度 682，**动了** |
| 相对 -50 脉冲回原位 | 规划位置 -1186 → -1236（Δ-50），峰值速度 690，偏差 0 |
| 业务层 API（`set_motion_params` + `pmove_rel` + `check_down`） | 全程正常，`check_down` 用实时速度判断到位准确 |
| 0 距离绝对定位 | 无异常、无副作用（保护生效） |
| 实时状态显示 | 运动中/空闲切换正确；`MCF_Get_Vel_Net` 确认是**实时**值（连接时还读到 -115330 的余速） |

### 另一个实测结论：正常运动结束后，锁存寄存器也会停在 1(正在执行)
等 2 秒也不回 0，只有 `MCF_Clear_Axis_State_Net` 能清。所以：
- **判断运动/停止必须用实时速度**（已实现），
- 显示上把"锁存码=1 且速度为 0"当作 **空闲**（原始码照常写日志）。

