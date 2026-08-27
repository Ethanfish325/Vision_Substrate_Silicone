# PCBA导热硅胶检测设备 (Vision_Substrate_Silicone)

> **版本**: 1.0.0 | **开发语言**: Python 3.8+ | **GUI框架**: PyQt5 | **图像处理**: OpenCV 4.x | **相机SDK**: 大恒 GalaxySDK (gxipy)

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [模块详解](#模块详解)
- [安装说明](#安装说明)
- [使用说明](#使用说明)
- [打包说明](#打包说明)
- [开发计划](#开发计划)
- [更新日志](#更新日志)

---

## 项目概述

本系统是一个基于机器视觉的工业检测系统，采用**流水线（Pipeline）架构**，支持用户通过图形界面配置视觉检测流程，实现产品的自动检测和判定。系统集成了**工业相机控制**、**串口通信**和**自动化检测工作流**等完整工业自动化功能。

### 操作模式

| 模式 | 说明 |
|------|------|
| **自动化模式 (Automation Mode)** | 手动触发的多位置自动化检测，实时显示位置检测结果和统计信息 |
| **设计模式 (Engineer Mode)** | 完整的方案编辑界面，可配置检测流水线、拖拽式编辑、实时预览 |

---

## 系统架构

```
Vision_Substrate_Silicone/
│
├── main.py                          # 应用程序入口
├── main.spec                        # PyInstaller 打包配置
├── requirements.txt                 # Python 依赖列表
├── README.md                        # 本说明文件
├── runtime_hook.py                  # PyInstaller 运行时钩子（设置 DLL 搜索路径）
├── cleanup_after_build.bat          # 打包后清理脚本
│
├── camera_manager.py                # 相机管理模块（大恒 GalaxySDK 封装）
│
├── core/                            # 核心模块
│   ├── __init__.py
│   ├── paths.py                     # 路径管理（数据目录、方案目录等）
│   ├── config_manager.py            # 配置管理（单例模式）
│   ├── log_manager.py               # 日志管理（按天轮转、自动清理）
│   ├── result_storage.py            # 结果存储（CSV/JSON/图像/XML）
│   ├── serial_comm.py               # 串口通信核心模块
│   ├── serial_test_workflow.py      # 串口自动测试工作流（状态机）
│   ├── product_manager.py           # 产品配置管理器（grid 行列、motion 双轴、io 映射）
│   ├── controller.py                # SMC6480 运动控制卡封装（轴运动、IO 读写）
│   ├── smcsh_dll.py                 # SMC6480 DLL 封装（ctypes 绑定、PE 导出表解析）
│   └── inspection_workflow.py       # 自动化检测工作流（多板卡检测、拼接、QR、运动、DI 触发）
│
├── ui/                              # UI 界面模块
│   ├── __init__.py
│   ├── constants.py                 # 颜色、图标等 UI 常量
│   ├── main_window.py               # 主窗口（工人/工程师双模式）
│   ├── inspection_panel.py          # 检测面板（拼接整图显示）
│   └── widgets/                     # 自定义控件
│       ├── __init__.py
│       ├── camera_panel.py          # 相机控制面板（含白平衡 R/G/B 调节 UI）
│       ├── flow_canvas.py           # 流程画布
│       ├── operator_toolbox.py      # 算子工具箱（支持搜索、拖拽）
│       ├── param_config_dialog.py   # 参数配置对话框（带实时预览）
│       ├── pipeline_editor.py       # 流水线编辑器
│       ├── result_panel.py          # 结果显示面板
│       ├── serial_dialog.py         # 串口通信对话框
│       ├── smc_dialog.py            # SMC6480 轴控制面板
│       ├── step_slot_widget.py      # 步骤插槽控件（支持拖拽排序）
│       └── zoomable_label.py        # 可缩放图片显示控件
│
├── vision/                          # 视觉算法模块
│   ├── __init__.py
│   ├── pipeline.py                  # 流水线定义和管理（工具注册、步骤执行）
│   ├── vision_engine.py             # 视觉引擎（执行入口、结果保存）
│   ├── stitch.py                    # 图像拼接模块（增量累积画布，多板卡拼接）
│   └── tools/                       # 视觉工具集（6 大类，34 种工具）
│       ├── __init__.py
│       ├── base_tool.py             # 工具基类（VisionTool、ToolResult、PipelineContext）
│       ├── preprocess.py            # 预处理工具（8 种）
│       ├── feature_extract.py       # 特征提取工具（7 种）
│       ├── geometry.py              # 几何检测工具（4 种）
│       ├── measure.py               # 测量工具（7 种）
│       ├── recognize.py             # 识别工具（含 QRCodeRecognize 二维码识别、TemplateMatch 多尺度）
│       └── utility.py               # 辅助工具（3 种）
│
├── gxipy/                           # 大恒 GalaxySDK Python 接口
│   ├── __init__.py
│   ├── gxiapi.py                    # 相机 API 封装
│   ├── gxidef.py                    # 常量/枚举定义
│   ├── gxwrapper.py                 # C 接口封装
│   └── dxwrapper.py                 # DxImageProc 图像处理封装
│
├── data/                            # 运行时数据目录（自动创建）
│   ├── icon.png                     # 应用程序图标
│   ├── users.json                   # 用户数据
│   ├── production data/             # 生产检测数据（按日期分目录）
│   │   └── YYYY-MM-DD/
│   │       ├── NG/                  # NG 数据（原始图像 + 结果图像 + CSV 日志）
│   │       └── OK/                  # OK 数据（按 SN 保存 + XML + CSV 日志）
│   ├── schemes/                     # 检测方案文件（JSON 格式）
│   │   └── PCBA.json                # PCBA 视觉方案（MultiROI + TemplateMatch + QRCodeRecognize）
│   ├── products/                    # 产品配置（JSON 格式）
│   │   └── DX8000_PCBA.json         # DX8000 产品配置（grid 行列、motion 双轴、io 映射）
│   └── logs/                        # 系统日志（按天轮转）
│
├── model/                           # 模型与样本文件
│   ├── long_mat.jpg                 # 长条模板图像
│   ├── mat1.png / mat2.png          # 模板匹配图像
│   ├── title.jpg / title.png        # 标题检测样本
│   └── title1.jpg / title1.png      # 标题检测样本
│
├── plans/                           # 开发计划文档
│   ├── stitch_qr_multi_board_plan.md # 多板卡拼接 + QR 识别方案
│   ├── smc6480_axis_control_plan.md  # SMC6480 轴控制方案
│   └── control_mode_plan.md          # 控制模式方案
│
├── test_io_demo.py                  # IO 电平检测测试 Demo（扫描端口 + 按键映射）
├── test_template_match_demo.py      # 模板匹配测试 Demo（多尺度匹配诊断）
│
└── *.dll                            # 大恒相机 SDK DLL + SMC6480 控制卡 DLL
    ├── GxIAPI.dll                   # 相机 API 库
    ├── DxImageProc.dll              # 图像处理库
    └── smcsh_mbs.dll                # SMC6480 运动控制卡库
```

---

## 核心功能

### 1. 流水线式视觉处理

- 支持最多 **20 个步骤**的视觉处理流水线
- 每个步骤可选择不同的视觉工具
- 支持步骤的**启用/禁用**、**拖拽排序**、**参数配置**
- 支持**实时预览**每个步骤的处理效果
- 支持 **ROI 坐标**在缩放/裁剪操作后自动跟踪
- 支持**多区域 ROI**：命名区域、百分比坐标、导出/导入配置

### 2. 六大类视觉工具（34 种）

#### 预处理（8 种）
| 工具 | 功能 |
|------|------|
| `Grayscale` | 灰度化：彩色图像转灰度图 |
| `GaussianBlur` | 高斯滤波：高斯模糊降噪（自动校正核大小为奇数） |
| `HistEqualize` | 直方图均衡化：增强图像对比度 |
| `Morphology` | 形态学操作：腐蚀/膨胀/开运算/闭运算/梯度/顶帽/黑帽，支持结构元素形状选择（矩形/椭圆/十字）和迭代次数 |
| `MultiROI` | 多区域 ROI：命名区域、百分比坐标、启用/禁用、导出/导入 |
| `MedianBlur` | 中值滤波：对椒盐噪声有效（自动校正核大小为奇数） |
| `Resize` | 缩放：按比例或固定尺寸，输出 scale_x/scale_y 用于 ROI 跟踪 |
| `AdaptiveThreshold` | 自适应阈值：均值/高斯局部自适应二值化 |

#### 特征提取（7 种）
| 工具 | 功能 |
|------|------|
| `CannyEdge` | Canny 边缘检测：支持手动阈值和自动 Otsu 阈值 |
| `Threshold` | 阈值分割：集成传统阈值和自适应阈值两种模式 |
| `ContourAnalysis` | 轮廓分析：按面积/周长/x/y/宽/高排序 |
| `BlobDetection` | Blob 检测：基于 SimpleBlobDetector |
| `ContourFilter` | 轮廓筛选：AND/OR 多条件逻辑运算 |
| `LineDetection` | 直线检测：HoughLinesP，支持自动参数估计 |
| `RectangleDetection` | 矩形检测：基于轮廓逼近 |

#### 几何检测（4 种）
| 工具 | 功能 |
|------|------|
| `CircleDetection` | 圆检测：霍夫圆检测，自动参数估计，半径范围限制，圆心距去重 |
| `HoughLineDetection` | 直线检测（霍夫）：自动参数估计 |
| `ContourRectDetection` | 矩形检测（轮廓）：基于轮廓逼近 |
| `SimpleBlobDetect` | Blob 检测（简单）：面积/圆度/凸度/惯性比/颜色过滤，最大数量限制 |

#### 测量（7 种）
| 工具 | 功能 |
|------|------|
| `AreaMeasure` | 面积测量 |
| `DistanceMeasure` | 距离测量 |
| `PointMeasure` | 点测量 |
| `LineMeasure` | 线测量 |
| `AngleMeasure` | 角度测量 |
| `ObjectCount` | 目标计数 |
| `BrightnessMeasure` | 亮度测量 |

#### 识别（5 种）
| 工具 | 功能 |
|------|------|
| `ColorRecognition` | 颜色识别：HSV/Lab 色彩空间切换，区域颜色占比分析，预设颜色库 |
| `TemplateMatch` | 模板匹配：标准模式（支持掩膜）、多角度模式（分数曲线）、特征点模式（SIFT/ORB） |
| `EdgeMatch` | 边缘匹配：基于边缘特征的模板匹配 |
| `FastMatch` | 快速匹配：基于图像金字塔的快速匹配 |
| `FootPadDetect` | 脚垫识别：专用脚垫检测工具 |

#### 辅助工具（3 种）
| 工具 | 功能 |
|------|------|
| `CoordinateTransform` | 坐标转换：像素到物理单位 |
| `Calculator` | 数值计算：表达式 `{A}+{B}`，合格范围判定 |
| `LogicJudge` | 逻辑判断：表达式模式（AND/OR 语法）和条件模式，调试界面显示所有输入值 |

### 3. 方案管理

- 创建、保存、加载、重命名、删除检测方案
- 方案的导入和导出（JSON 格式）
- 方案包含完整的流水线配置和参数
- 方案文件自动保存至 `data/schemes/` 目录
- 支持**产品配置管理**：相机参数、运动参数、检测位置列表、条码扫描配置

### 4. 用户权限管理

| 角色 | 权限 |
|------|------|
| **操作员 (Operator)** | 只能执行检测，不能修改方案 |
| **工程师 (Engineer)** | 可以编辑和配置检测方案 |
| **管理员 (Admin)** | 用户管理和系统配置 |

- 默认账号：`admin` / `admin123`（管理员）、`engineer` / `123456`（工程师）、`operator` / `123456`（操作员）
- 密码使用 **SHA256 哈希**存储

### 5. 相机支持（大恒 Daheng GalaxySDK）

- 支持大恒（Daheng）工业相机（GigE / U3V）
- 设备枚举：同网段 + 跨网段自动搜索
- 打开/关闭、实时取流、单次拍照
- 支持 Bayer / Mono 像素格式
- 支持软触发采集（TriggerMode + TriggerSoftware）
- GigE 网络优化：自动设置包大小、延迟参数
- 参数调节：曝光时间、增益、帧率
- **白平衡调节**：R/G/B 三通道独立系数 + 色温预设（日光/荧光灯/白炽灯）
- 图像后处理：Gamma 校正（γ=2.2）、USM 锐化（强度 0.5）、16bit→8bit 固定映射
- 自动/手动曝光、自动/手动增益切换

#### 相机图像处理流程

```
原始帧数据 (bytes)
    │
    ▼
numpy 数组 (uint8 / uint16)
    │
    ▼
Bayer demosaic (cv2.cvtColor, 固定 Bayer 模式)
    │
    ▼
16bit → 8bit 转换（固定右移 8 位，避免亮度跳动）
    │
    ▼
Gamma 校正（γ=2.2，提亮暗部，增强对比度）
    │
    ▼
USM 锐化（强度 0.5，提升清晰度）
    │
    ▼
BGR 图像输出
```

### 6. 串口通信

- 独立的串口通信对话框，通过菜单栏「通信 > 串口通信」打开
- 端口扫描与选择，支持常用串口参数配置（波特率/数据位/校验位/停止位/流控制）
- 文本/HEX 两种模式发送数据
- 实时接收数据显示（支持 HEX 显示模式），自动滚动
- 收发字节统计，配置持久化
- 串口自动测试工作流：由串口数据触发的自动化测试流程
- **状态机**：`IDLE → WAITING_TRIGGER → CAPTURING → TESTING → SENDING_RESULT`
- **策略模式设计**：`TriggerParser`（触发解析）、`ResultSender`（结果发送）均可扩展

### 7. 自动化检测工作流（多板卡托盘检测）

由 **手动触发**（或 DI 触发）的多板卡托盘自动化检测工作流：

```
IDLE → MONITORING → WAITING → CAPTURING → TESTING
    → (循环: 运动到点位 → 拍照 → 检测 → 拼接 直到所有位置完成)
    → SHOW_RESULT → MONITORING
```

- **多板卡托盘检测**：每个点位对应一张独立板卡，逐点拍照检测
- **图像拼接**：按行列网格紧密排列，实时刷新拼接整图（增量累积画布，内存优化）
- **QR 识别**：每张板卡识别二维码作为 SN，按 SN 保存数据并生成 XML（供 MES 上传）
- **SMC6480 轴运动控制**：起始位 → 各点位（行优先）→ 结束位 → 取出确认
- **DI 触发**：上升沿检测，多按钮 IO 映射（启动/停止/复位/复判OK/复判NG/下料）
- **检测工作线程化**：拍照/检测在工作线程执行，主线程保持空闲，DI 轮询持续运行
- **STOP/复位流程**：STOP 停止所有动作并继续监听 IO，复位回到等待触发状态
- 产品配置管理：grid 行列、motion 双轴、起始/结束位、点位 row/col/x/y/scheme、io 映射
- 每个位置可关联独立的视觉方案
- 统计信息：触发次数、OK 次数、NG 次数

### 8. SMC6480 运动控制卡

- **DLL 封装**（[`core/smcsh_dll.py`](core/smcsh_dll.py:1)）：ctypes 绑定、PE 导出表解析、容错函数绑定
- **控制器封装**（[`core/controller.py`](core/controller.py:1)）：连接（以太网/串口）、轴运动（绝对/相对/JOG/回零）、IO 读写
- **轴控制面板**（[`ui/widgets/smc_dialog.py`](ui/widgets/smc_dialog.py:1)）：手动 JOG、绝对/相对定位、回零、伺服使能
- 双轴运动（X/Y），位置检测到位（get_pulse_position）
- 输出端口控制：红灯/绿灯

### 9. 结果记录

- 自动保存检测结果（OK/NG）
- **NG 数据保存**：原始图像、标注图像、JSON 数据
- **OK 数据保存**：按 QR SN 分目录保存 + 生成 XML（供 MES 上传）+ CSV 日志
- 自动清理过期数据（默认保留 90 天）
- 支持 `overlay_image` 工业叠加图层输出
- 日志系统：按天轮转，自动清理（默认 50GB 限额）

### 10. 图形界面特性

- **深色主题**（VS Code 风格），护眼且专业
- 算子工具箱支持**搜索过滤**
- **拖拽式**流水线编辑
- 参数配置对话框带**实时预览**
- 多区域 ROI **可视化编辑器**（支持命名、百分比坐标）
- 可缩放图片显示控件（支持鼠标滚轮缩放、拖拽平移、双击重置）
- **自动化/设计双模式**切换
- **1024×768 分辨率适配**（工控机屏幕）

---

## 模块详解

### `core/` — 核心模块

| 文件 | 职责 | 设计模式 |
|------|------|----------|
| [`paths.py`](core/paths.py) | 路径管理：数据目录、方案目录、日志目录等 | 函数式 |
| [`config_manager.py`](core/config_manager.py) | 系统配置管理：相机参数、系统参数、显示参数 | **单例模式** |
| [`log_manager.py`](core/log_manager.py) | 日志管理：按天轮转、自动清理（50GB 限额）、后台线程清理 | **单例模式**、自定义 Handler |
| [`result_storage.py`](core/result_storage.py) | 结果存储：NG 数据（图像+JSON）、OK 数据（按 SN + XML）、过期清理 | — |
| [`serial_comm.py`](core/serial_comm.py) | 串口通信：端口扫描、参数配置、异步读取线程、收发统计 | QThread 异步读取 |
| [`serial_test_workflow.py`](core/serial_test_workflow.py) | 串口自动测试工作流：状态机、触发解析、结果发送 | **状态机**、**策略模式** |
| [`product_manager.py`](core/product_manager.py) | 产品配置管理：grid 行列、motion 双轴、io 映射、兼容迁移 | 函数式 |
| [`controller.py`](core/controller.py) | SMC6480 运动控制卡：连接、轴运动、IO 读写、位置检测 | 封装 |
| [`smcsh_dll.py`](core/smcsh_dll.py) | SMC6480 DLL 封装：ctypes 绑定、PE 导出表解析、容错函数绑定 | 封装 |
| [`inspection_workflow.py`](core/inspection_workflow.py) | 自动化检测工作流：多板卡检测、拼接、QR、运动、DI 触发、工作线程 | **状态机**、QTimer、QThread |

### `vision/` — 视觉算法模块

| 文件 | 职责 |
|------|------|
| [`pipeline.py`](vision/pipeline.py) | 流水线定义：工具注册、步骤管理、执行引擎、中文/英文工具名映射 |
| [`vision_engine.py`](vision/vision_engine.py) | 视觉引擎：流水线执行入口、overlay 叠加、ROI 结果绘制、NG 数据保存 |
| [`stitch.py`](vision/stitch.py) | 图像拼接：增量累积画布、行列网格排列、OK/NG 标注、内存优化 |
| [`tools/base_tool.py`](vision/tools/base_tool.py) | 工具基类：`VisionTool`（抽象基类）、`ToolResult`（数据类）、`PipelineContext`（上下文） |

### `ui/` — UI 界面模块

| 文件 | 职责 |
|------|------|
| [`main_window.py`](ui/main_window.py) | 主窗口：自动化/设计双模式、菜单栏、相机控制、串口通信、用户登录 |
| [`constants.py`](ui/constants.py) | UI 常量：颜色、图标、样式表 |
| [`inspection_panel.py`](ui/inspection_panel.py) | 检测面板 |
| [`widgets/camera_panel.py`](ui/widgets/camera_panel.py) | 相机控制面板：曝光/增益/帧率/白平衡 R/G/B 调节 |
| [`widgets/pipeline_editor.py`](ui/widgets/pipeline_editor.py) | 流水线编辑器 |
| [`widgets/operator_toolbox.py`](ui/widgets/operator_toolbox.py) | 算子工具箱：搜索过滤、拖拽 |
| [`widgets/param_config_dialog.py`](ui/widgets/param_config_dialog.py) | 参数配置对话框：实时预览 |
| [`widgets/result_panel.py`](ui/widgets/result_panel.py) | 结果显示面板 |
| [`widgets/serial_dialog.py`](ui/widgets/serial_dialog.py) | 串口通信对话框 |
| [`widgets/zoomable_label.py`](ui/widgets/zoomable_label.py) | 可缩放图片控件：滚轮缩放、拖拽平移、双击重置 |
| [`widgets/step_slot_widget.py`](ui/widgets/step_slot_widget.py) | 步骤插槽控件：拖拽排序 |
| [`widgets/flow_canvas.py`](ui/widgets/flow_canvas.py) | 流程画布 |

### `camera_manager.py` — 相机管理

- 封装大恒 GalaxySDK（gxipy），提供统一接口
- 设备枚举（同网段 + 跨网段）
- 参数读写（曝光/增益/帧率/白平衡）
- 图像转换（Bayer demosaic、Gamma 校正、USM 锐化）
- 实时取流线程（`CameraGrabbingThread`）
- 单例模式

### 设计模式总结

| 模式 | 使用位置 |
|------|----------|
| **单例模式** | `ConfigManager`、`LogManager`、`CameraManager` |
| **状态机** | `InspectionWorkflow`、`SerialTestWorkflow` |
| **策略模式** | `SerialTestWorkflow` 的 `TriggerParser` / `ResultSender` |
| **抽象基类** | `VisionTool`（所有视觉工具的基类） |
| **工厂方法** | `Pipeline.create_tool()` 按名称创建工具实例 |
| **观察者模式** | Qt 信号/槽机制（`pyqtSignal`） |
| **模板方法** | `VisionTool.process()` 定义算法骨架 |

---

## 安装说明

### 环境要求

- Python 3.8 或更高版本
- Windows 7/10/11（64 位）

### 安装步骤

1. **安装 Python 依赖**

   ```bash
   pip install -r requirements.txt
   ```

2. **安装大恒 GalaxySDK**

   - 从大恒官网下载并安装 GalaxySDK（包含 gxipy Python 包）
   - 或将 `gxipy/` 目录及 `GxIAPI.dll`、`DxImageProc.dll` 放置到项目根目录（已预置）
   - 确保 DLL 在系统 PATH 或程序运行目录中可被加载

3. **运行程序**

   ```bash
   python main.py
   ```

### 依赖清单

```
PyQt5>=5.15.0       # GUI 框架
opencv-python>=4.5.0 # 图像处理
numpy>=1.21.0       # 数值计算
pyserial>=3.5       # 串口通信
gxipy               # 大恒 GalaxySDK（内置于项目 gxipy/ 目录）
```

---

## 使用说明

### 1. 登录系统

- 默认管理员账号：`admin` / `admin123`
- 默认工程师账号：`engineer` / `123456`
- 默认操作员账号：`operator` / `123456`

### 2. 自动化模式（多板卡托盘检测）

1. 在工程师模式下创建产品配置（grid 行列、motion 双轴、起始/结束位、点位 row/col/x/y/scheme、io 映射）
2. 为每个位置关联视觉方案（MultiROI + TemplateMatch + QRCodeRecognize）
3. 切换到自动化模式，选择产品
4. 点击「启动监听」后，通过 DI 触发（启动按钮）或手动触发执行检测
5. 系统自动：运动到各点位 → 拍照 → 检测 → 拼接 → 识别 QR（SN）
6. 实时查看拼接整图和各位置检测结果（OK/NG）
7. 检测完成后按 SN 保存数据并生成 XML（供 MES 上传）
8. 按下 STOP 停止所有动作（继续监听 IO），按下复位回到等待触发状态

### 3. 设计模式（工程师模式）

1. 创建或打开检测方案
2. 从算子工具箱拖拽算子到流水线插槽
3. 点击算子配置参数（支持实时预览）
4. 加载测试图像进行预览
5. 保存方案

### 4. 相机操作

1. 打开相机面板（默认在主界面右侧）
2. 点击「刷新」搜索相机设备
3. 从下拉列表选择相机，点击「打开」
4. 调节参数：曝光时间、增益、帧率、白平衡（R/G/B 三通道）
5. 点击「拍照」采集单帧图像
6. 触发模式：切换至「触发模式（软触发）」后，点击「发送软触发」采集

### 5. 串口通信

1. 通过菜单栏「通信 > 串口通信」打开串口通信对话框
2. 点击「扫描端口」检测可用串口
3. 选择端口并配置参数（波特率/数据位/校验位/停止位/流控制）
4. 点击「打开串口」建立连接
5. 在发送区输入数据，选择文本或 HEX 模式，点击「发送」
6. 接收区实时显示接收到的数据

### 6. 串口自动测试工作流

1. 通过菜单栏「通信 > 串口通信」打开串口通信对话框
2. 配置串口参数并打开串口
3. 在串口对话框中启用「自动测试工作流」
4. 系统进入状态机模式：`IDLE → WAITING_TRIGGER → CAPTURING → TESTING → SENDING_RESULT`
5. 串口数据触发后自动执行检测并返回结果

---

## 参数自动校正

系统内置了多项参数自动校正机制，提高易用性和鲁棒性：

| 校正机制 | 说明 |
|----------|------|
| **核大小自动校正** | 高斯滤波、中值滤波、形态学操作的核大小自动调整为奇数 |
| **阈值自动校正** | Canny 边缘检测、直线检测的 low ≤ high 自动校正 |
| **半径自动校正** | 圆检测的 minRadius ≤ maxRadius 自动校正 |
| **自动参数估计** | 直线检测、圆检测、霍夫直线检测支持根据图像尺寸自动估算参数 |
| **自动阈值** | Canny 边缘检测支持 Otsu 算法自动计算最佳阈值 |

---

## 打包说明

使用 PyInstaller 打包为独立可执行文件：

```bash
pyinstaller main.spec
```

打包后的文件位于 `dist/Vision_Substrate_Silicone/` 目录下。

打包完成后可运行 `cleanup_after_build.bat` 清理不需要的大文件（如 Qt5 的 WebEngine、QML 等 DLL 和多语言翻译文件）。

`runtime_hook.py` 会在打包后的程序启动时自动设置 DLL 搜索路径，确保 `GxIAPI.dll` / `DxImageProc.dll` 能被正确加载。

---

## 开发计划

项目包含 19 份详细的开发计划文档，位于 [`plans/`](plans/) 目录：

| 文档 | 说明 |
|------|------|
| [`architecture_optimization_plan.md`](plans/architecture_optimization_plan.md) | 架构优化计划 |
| [`barcode_scan_integration_plan.md`](plans/barcode_scan_integration_plan.md) | 条码扫描集成计划 |
| [`brightness_measure_plan.md`](plans/brightness_measure_plan.md) | 亮度测量计划 |
| [`camera_init_fixed_params_plan.md`](plans/camera_init_fixed_params_plan.md) | 相机初始化固定参数计划 |
| [`default_user_change_record.md`](plans/default_user_change_record.md) | 默认用户变更记录 |
| [`footpad_detect_optimization_plan.md`](plans/footpad_detect_optimization_plan.md) | 脚垫检测优化计划 |
| [`footpad_detect_robustness_plan.md`](plans/footpad_detect_robustness_plan.md) | 脚垫检测鲁棒性计划 |
| [`hikvision_to_daheng_migration_plan.md`](plans/hikvision_to_daheng_migration_plan.md) | 海康威视到大恒迁移计划 |
| [`inspection_workflow_plan.md`](plans/inspection_workflow_plan.md) | 检测工作流计划 |
| [`operator_optimization_plan.md`](plans/operator_optimization_plan.md) | 操作员模式优化计划 |
| [`overlay_image_implementation_plan.md`](plans/overlay_image_implementation_plan.md) | 叠加图层实现计划 |
| [`roi_result_display_plan.md`](plans/roi_result_display_plan.md) | ROI 结果显示计划 |
| [`serial_comm_plan.md`](plans/serial_comm_plan.md) | 串口通信计划 |
| [`serial_test_workflow_plan.md`](plans/serial_test_workflow_plan.md) | 串口测试工作流计划 |
| [`tool_data_passing_architecture.md`](plans/tool_data_passing_architecture.md) | 工具数据传递架构 |
| [`user_settings_menu_plan.md`](plans/user_settings_menu_plan.md) | 用户设置菜单计划 |
| [`visual_pipeline_editor_plan.md`](plans/visual_pipeline_editor_plan.md) | 可视化流水线编辑器计划 |

---

## 注意事项

1. 首次运行时会自动创建 `data/` 目录及其子目录
2. 相机功能需要大恒 GalaxySDK（gxipy）及配套 DLL
3. 如果没有相机，可以使用「加载图像」功能测试流水线
4. 系统日志保存在 `data/logs/` 目录下，自动按天轮转，保留 30 天
5. 方案文件为 JSON 格式，可手动编辑，但建议通过界面操作
6. Canny 边缘检测的「自动阈值」选项使用 Otsu 算法，适用于光照变化大的场景
7. 形态学操作的核大小必须为奇数，系统会自动校正
8. 多区域 ROI 的百分比坐标范围为 0~100，系统自动根据图像分辨率转换为像素坐标
9. 模板匹配的掩膜图像需与模板图像尺寸一致，灰度图中黑色区域将被忽略
10. 逻辑判断的表达式模式中，变量名使用「工具名.数据键」格式，可在调试界面中查看可用变量
11. 串口通信功能依赖 pyserial 库，请确保已安装
12. 白平衡默认值（R=1.5, G=1.0, B=1.8）针对偏绿场景校正，可在相机面板中实时调节
13. Gamma 校正和锐化强度可在 `camera_manager.py` 顶部调整，修改后重启程序生效
14. SMC6480 运动控制卡需要 `smcsh_mbs.dll`（已从 git 排除，需手动放置到项目根目录）
15. 产品配置的 `io` 字段使用 1-based IN 编号（如 IN2 填 2），可用 `test_io_demo.py` 扫描实际端口号
16. 模板匹配支持多尺度搜索（`scale_min`/`scale_max`/`scale_step`），解决模板与目标尺寸不一致问题
17. 相机图像翻转由 `camera_manager.py` 顶部的 `CAMERA_FLIP_180` 控制（True 表示水平+垂直翻转）

---

## 更新日志

### v3.0.0 (2026-08-27) — 多板卡托盘检测 + SMC6480 运动控制

- **多板卡托盘检测**：每个点位对应一张独立板卡，逐点拍照检测
- **图像拼接**（[`vision/stitch.py`](vision/stitch.py:1)）：按行列网格紧密排列，增量累积画布，实时刷新拼接整图
- **QR 识别**（[`vision/tools/recognize.py`](vision/tools/recognize.py:1)）：QRCodeRecognize 算子，多策略识别，识别结果作为板卡 SN
- **XML 导出**（[`core/result_storage.py`](core/result_storage.py:1)）：按 SN 保存数据并生成 XML（供 MES 上传）
- **SMC6480 运动控制卡**（[`core/controller.py`](core/controller.py:1)、[`core/smcsh_dll.py`](core/smcsh_dll.py:1)）：轴运动、IO 读写、位置检测
- **轴控制面板**（[`ui/widgets/smc_dialog.py`](ui/widgets/smc_dialog.py:1)）：手动 JOG、绝对/相对定位、回零、伺服使能
- **DI 触发**：上升沿检测，多按钮 IO 映射（启动/停止/复位/复判OK/复判NG/下料）
- **STOP/复位流程**：STOP 停止所有动作并继续监听 IO，复位回到等待触发状态
- **模板匹配多尺度**：自动搜索最佳缩放比例，解决模板与目标尺寸不一致问题
- **内存优化**：增量拼接 + 检测后释放原始图，避免多张高分辨率图累积溢出
- **检测工作线程化**（`InspectionWorker`）：拍照/检测在工作线程执行，主线程保持空闲，DI 轮询持续运行
- **产品配置新结构**：grid 行列、motion 双轴、起始/结束位、点位 row/col/x/y/scheme、io 映射
- **相机图像翻转**：支持水平+垂直翻转（`CAMERA_FLIP_180`）

### v2.3.1 (2026-07-06) — 1024x768 分辨率适配

- **UI 全面优化适配 1024x768 分辨率**（工控机屏幕）
- 主窗口最小尺寸从 `1400x850` 缩小至 `1024x700`
- 全局字体从 `13px` 缩小至 `12px`，按钮 padding 减小约 40%
- 所有面板边距、间距缩小约 30-50%，充分利用有限屏幕空间
- 模式工具栏高度从 `40px` 降至 `32px`
- 自动化检测面板：位置卡片最小尺寸从 `320x280` 降至 `200x160`
- 设计模式：测试图像区、结果面板、日志面板尺寸全面缩小
- 流水线编辑器：算子工具箱最大宽度从 `220px` 降至 `160px`
- 步骤槽控件：最小高度从 `56px` 降至 `40px`，控件尺寸缩小约 25%
- 相机面板：图像显示最小尺寸从 `640x480` 降至 `400x300`
- 参数配置对话框：最小尺寸从 `960x600` 降至 `800x500`
- 所有对话框（登录、相机设置、产品配置等）尺寸相应缩小

### v2.3.0 (2026-06-26)

- **相机 SDK 迁移**：海康威视 MVS → 大恒 GalaxySDK (gxipy)
- 新增 `gxipy/` 目录及配套 DLL（`GxIAPI.dll`、`DxImageProc.dll`）
- 重写 `camera_manager.py`：大恒 SDK 设备枚举、打开/关闭、取流、参数读写
- 新增跨网段设备搜索（同网段未发现时自动切换）
- 新增 GigE 网络参数优化（包大小、延迟、帧传输）
- 新增白平衡 R/G/B 三通道独立系数设置
- 新增相机面板白平衡 UI（R/G/B 滑块 + 色温预设）
- 新增图像后处理：Gamma 校正（γ=2.2）、USM 锐化（强度 0.5）
- 优化 16bit→8bit 转换：固定右移替代 NORM_MINMAX，避免亮度跳动
- 更新打包配置（`main.spec`）适配大恒 SDK DLL
- 更新 `runtime_hook.py` 适配大恒 DLL 搜索路径

### v2.2.0 (2026-06-10)

- 新增串口通信核心模块（`serial_comm.py`）
- 新增串口通信对话框（`serial_dialog.py`）
- 新增串口自动测试工作流（`serial_test_workflow.py`）
- 新增 PyInstaller 运行时钩子（`runtime_hook.py`）
- 新增 `plans/` 目录下多个开发计划文档
- 新增 `model/` 目录下深度学习样本和标题检测样本数据
- 新增 `data/icon.png` 应用程序图标
- 新增 `data/schemes/默认方案.json` 默认检测方案
- 新增 `.gitignore` 版本控制忽略配置
- 优化项目目录结构，增加 `core/` 核心模块的串口通信相关功能

### v2.1.0 (2026-06-03)

- 新增形态学操作结构元素形状选择（矩形/椭圆/十字）和迭代次数
- 新增多区域 ROI 百分比坐标支持，区域命名，导出/导入功能
- 新增图像缩放 ROI 坐标跟踪（输出 scale_x/scale_y）
- 新增 Canny 边缘检测自动阈值功能（Otsu 算法）
- 新增阈值分割集成自适应阈值模式（均值/高斯）
- 新增轮廓分析多维度排序（x/y/宽/高）
- 新增轮廓筛选 AND/OR 多条件逻辑运算
- 新增直线检测自动参数估计和 HoughLinesP 支持
- 新增矩形检测工具（基于轮廓逼近）
- 新增圆检测自动参数估计、半径范围限制、圆心距去重
- 新增霍夫直线检测自动参数估计
- 新增 Blob 检测增强过滤（圆度/凸度/惯性比/颜色/最大数量）
- 新增颜色识别 HSV/Lab 色彩空间切换和区域颜色占比分析
- 新增模板匹配掩膜支持、多角度分数曲线输出
- 新增逻辑判断表达式解析模式（AND/OR 语法）和调试界面
- 新增 `ToolResult.overlay_image` 字段支持工业叠加图层
- 新增参数自动校正机制（核大小奇数、阈值大小关系、半径大小关系）
- 新增可缩放图片显示控件（`ZoomableLabel`）
- 新增脚垫识别工具（`FootPadDetect`）
- 新增打包后清理脚本（`cleanup_after_build.bat`）
- 优化参数配置对话框的实时预览交互
- 改进算子工具箱的搜索和拖拽体验
- 优化深色主题 UI 样式
- 修复方案加载和保存的兼容性问题

### v1.0.0 (2024-01-01)

- 初始版本发布
- 支持流水线式视觉检测
- 支持工人/工程师双模式
- 支持 30 种视觉工具
- 支持方案管理和用户权限管理
- 支持海康威视相机

---

## 项目分析总结

### 技术栈

| 层面 | 技术 |
|------|------|
| **编程语言** | Python 3.8+ |
| **GUI 框架** | PyQt5（Qt 信号/槽机制） |
| **图像处理** | OpenCV 4.x（numpy 底层） |
| **工业相机** | 大恒 GalaxySDK（gxipy） |
| **运动控制卡** | SMC6480（smcsh_mbs.dll，ctypes 封装） |
| **串口通信** | pyserial（异步 QThread 读取） |
| **打包部署** | PyInstaller |
| **数据存储** | JSON（方案/配置/用户）、CSV（日志）、XML（MES 上传）、图像文件 |

### 代码规模

| 模块 | 文件数 | 代码行数（约） |
|------|--------|---------------|
| `core/` | 9 | 2,500+ |
| `ui/` | 13 | 6,500+ |
| `vision/` | 9 | 4,000+ |
| `camera_manager.py` | 1 | 1,200+ |
| `main.py` | 1 | 220 |
| **总计** | **33+** | **14,000+** |

### 架构特点

1. **模块化分层**：`core/`（业务逻辑）→ `vision/`（视觉算法）→ `ui/`（界面展示）→ `camera_manager.py`（硬件抽象）
2. **可扩展性**：视觉工具通过注册机制动态加载，新增工具只需在对应模块添加类
3. **设计模式应用**：单例模式（配置/日志/相机）、状态机（工作流）、策略模式（串口解析/发送）、抽象基类（视觉工具）
4. **工业级特性**：串口通信、自动化工作流、结果追溯
5. **兼容性**：支持中文/英文工具名映射，兼容旧版方案文件格式
