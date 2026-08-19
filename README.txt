========================================
基板硅胶视觉检测系统 (Vision_Substrate_Silicone)
========================================

版本: 1.0.0
开发语言: Python 3.8+
GUI框架: PyQt5
图像处理: OpenCV 4.x
相机SDK: 大恒 (Daheng) GalaxySDK (gxipy)

--------
项目简介
--------

本系统是一个基于机器视觉的工业检测系统，采用流水线（Pipeline）架构，
支持用户通过图形界面配置视觉检测流程，实现产品的自动检测和判定。

系统支持两种操作模式：
1. 工人模式（Operator Mode）：简化的操作界面，一键执行检测，显示OK/NG结果
2. 设计模式（Engineer Mode）：完整的方案编辑界面，可配置检测流水线

--------
功能特性
--------

1. 流水线式视觉处理
   - 支持最多20个步骤的视觉处理流水线
   - 每个步骤可选择不同的视觉工具
   - 支持步骤的启用/禁用、拖拽排序、参数配置
   - 支持实时预览每个步骤的处理效果
   - 支持 ROI 坐标在缩放/裁剪操作后自动跟踪

2. 六大类视觉工具（共30+种）
   - 预处理（8种）：灰度化、高斯滤波、直方图均衡化、形态学操作（支持结构元素形状选择：矩形/椭圆/十字，迭代次数）、多区域ROI（支持百分比坐标，命名区域，导出/导入）、中值滤波、图像缩放（输出缩放比例用于ROI跟踪）、自适应阈值
   - 特征提取（7种）：Canny边缘检测（支持自动Otsu阈值）、阈值分割（集成传统阈值和自适应阈值两种模式）、轮廓分析（支持按面积/周长/x/y/宽/高排序）、Blob检测、轮廓过滤（支持AND/OR多条件逻辑）、直线检测（支持自动参数估计，HoughLinesP）、矩形检测
   - 几何检测（4种）：圆检测（支持自动参数估计，半径范围限制，圆心距去重）、直线检测(霍夫)（支持自动参数估计）、矩形检测(轮廓)、Blob检测(简单)（支持面积/圆度/凸度/惯性比过滤，颜色过滤，最大数量限制）
   - 测量（6种）：面积测量、距离测量、点测量、线测量、角度测量、目标计数
   - 识别（5种）：颜色识别（支持HSV/Lab色彩空间切换，区域颜色占比分析）、模板匹配（支持掩膜，多角度分数曲线，特征点SIFT/ORB）、边缘匹配、快速匹配（图像金字塔）、脚垫识别
   - 工具（3种）：坐标转换、数值计算、逻辑判断（支持表达式解析：AND(条件1, 条件2)，调试界面显示所有输入值）

3. 方案管理
   - 支持创建、保存、加载、重命名、删除检测方案
   - 支持方案的导入和导出（JSON格式）
   - 方案包含完整的流水线配置和参数
   - 方案文件自动保存至 data/schemes/ 目录

4. 用户权限管理
   - 三种角色：操作员（Operator）、工程师（Engineer）、管理员（Admin）
   - 操作员：只能执行检测，不能修改方案
   - 工程师：可以编辑和配置检测方案
   - 管理员：用户管理和系统配置

5. 相机支持（大恒 Daheng GalaxySDK）
   - 支持大恒（Daheng）工业相机（GigE / U3V）
   - 设备枚举（同网段 + 跨网段自动搜索）
   - 打开/关闭、实时取流、单次拍照
   - 支持 Bayer / Mono 像素格式
   - 支持软触发采集（TriggerMode + TriggerSoftware）
   - GigE 网络优化（自动设置包大小、延迟参数）
   - 参数调节：曝光时间、增益、帧率
   - 白平衡调节：R/G/B 三通道独立系数 + 色温预设（日光/荧光灯/白炽灯）
   - 图像后处理：Gamma 校正、USM 锐化、16bit→8bit 固定映射
   - 自动/手动曝光、自动/手动增益切换

6. 串口通信
   - 独立的串口通信对话框，通过菜单栏「通信 > 串口通信」打开
   - 端口扫描与选择，支持常用串口参数配置（波特率/数据位/校验位/停止位/流控制）
   - 文本/HEX 两种模式发送数据
   - 实时接收数据显示（支持 HEX 显示模式），自动滚动
   - 收发字节统计，配置持久化
   - 串口自动测试工作流：由串口数据触发的自动化测试流程
   - 工作流状态机：IDLE -> WAITING_TRIGGER -> CAPTURING -> TESTING -> SENDING_RESULT
   - 策略模式设计：TriggerParser（触发解析）、ResultSender（结果发送）均可扩展

7. 结果记录
   - 自动保存检测结果（OK/NG）
   - NG数据保存：原始图像、标注图像、JSON数据
   - OK数据保存：CSV日志
   - 自动清理过期数据（默认保留90天）
   - 支持 overlay_image 工业叠加图层输出

8. 图形界面特性
   - 深色主题（VS Code风格），护眼且专业
   - 算子工具箱支持搜索过滤
   - 拖拽式流水线编辑
   - 参数配置对话框带实时预览
   - 多区域ROI可视化编辑器（支持命名、百分比坐标）
   - 可缩放图片显示控件（支持鼠标滚轮缩放、拖拽平移、双击重置）

--------
目录结构
--------

Vision_Substrate_Silicone/
├── main.py                    # 应用程序入口
├── main.spec                  # PyInstaller打包配置
├── requirements.txt           # Python依赖列表
├── ReadME.txt                 # 本说明文件
├── runtime_hook.py            # PyInstaller运行时钩子（设置DLL搜索路径）
├── cleanup_after_build.bat    # 打包后清理脚本
│
├── camera_manager.py          # 相机管理模块（大恒 GalaxySDK 封装）
│                              #   - 设备枚举、打开/关闭、取流
│                              #   - 参数读写（曝光/增益/帧率/白平衡）
│                              #   - 图像转换（Bayer demosaic、Gamma校正、锐化）
│
├── core/                      # 核心模块
│   ├── __init__.py
│   ├── paths.py               # 路径管理（数据目录、方案目录等）
│   ├── config_manager.py      # 配置管理（单例模式）
│   ├── log_manager.py         # 日志管理（按天轮转）
│   ├── result_storage.py      # 结果存储（CSV/JSON/图像）
│   ├── serial_comm.py         # 串口通信核心模块（端口扫描、收发管理、异步读取）
│   └── serial_test_workflow.py # 串口自动测试工作流（状态机、触发解析、结果发送）
│
├── ui/                        # UI界面模块
│   ├── __init__.py
│   ├── constants.py           # 颜色、图标等UI常量
│   ├── main_window.py         # 主窗口（工人/工程师双模式）
│   └── widgets/               # 自定义控件
│       ├── __init__.py
│       ├── camera_panel.py    # 相机控制面板（含白平衡R/G/B调节UI）
│       ├── flow_canvas.py     # 流程画布
│       ├── operator_toolbox.py # 算子工具箱（支持搜索、拖拽）
│       ├── param_config_dialog.py # 参数配置对话框（带实时预览）
│       ├── pipeline_editor.py # 流水线编辑器
│       ├── result_panel.py    # 结果显示面板
│       ├── serial_dialog.py   # 串口通信对话框（端口配置、收发数据、HEX模式）
│       ├── step_slot_widget.py # 步骤插槽控件（支持拖拽排序）
│       └── zoomable_label.py  # 可缩放图片显示控件
│
├── vision/                    # 视觉算法模块
│   ├── __init__.py
│   ├── pipeline.py            # 流水线定义和管理（工具注册、步骤执行）
│   ├── vision_engine.py       # 视觉引擎（执行入口、结果保存）
│   └── tools/                 # 视觉工具集
│       ├── __init__.py
│       ├── base_tool.py       # 工具基类（VisionTool、ToolResult、PipelineContext）
│       ├── preprocess.py      # 预处理工具（8种）
│       ├── feature_extract.py # 特征提取工具（7种）
│       ├── geometry.py        # 几何检测工具（4种）
│       ├── measure.py         # 测量工具（6种）
│       ├── recognize.py       # 识别工具（5种）
│       └── utility.py         # 辅助工具（3种）
│
├── gxipy/                     # 大恒 GalaxySDK Python 接口
│   ├── __init__.py
│   ├── gxiapi.py              # 相机API封装
│   ├── gxidef.py              # 常量/枚举定义
│   ├── gxwrapper.py           # C接口封装
│   └── dxwrapper.py           # DxImageProc 图像处理封装
│
├── DxImageProc.dll            # 大恒图像处理库（DLL）
├── GxIAPI.dll                 # 大恒相机API库（DLL）
├── MCDLL_NET.dll              # 大恒网络通信库（DLL）
│
├── data/                      # 运行时数据目录（自动创建）
│   ├── icon.png               # 应用程序图标
│   ├── users.json             # 用户数据
│   ├── schemes/               # 检测方案文件（JSON格式）
│   │   └── 默认方案.json
│   ├── errors/                # NG数据（按日期分目录）
│   └── logs/                  # 系统日志（按天轮转）
│
├── model/                     # 模型文件
│   ├── deep/                  # 深度学习模型样本
│   │   ├── NG1.jpg ~ NG4.jpg
│   │   └── OK1.jpg ~ OK4.jpg
│   └── titile/                # 标题检测样本
│       ├── NG1.jpg ~ NG3.jpg
│       └── OK1.jpg ~ OK3.jpg
│
└── plans/                     # 开发计划文档
    ├── architecture_optimization_plan.md
    ├── brightness_measure_plan.md
    ├── camera_init_fixed_params_plan.md
    ├── default_user_change_record.md
    ├── footpad_detect_optimization_plan.md
    ├── footpad_detect_robustness_plan.md
    ├── hikvision_to_daheng_migration_plan.md
    ├── nmc_motion_control_integration_plan.md
    ├── operator_optimization_plan.md
    ├── overlay_image_implementation_plan.md
    ├── roi_result_display_plan.md
    ├── serial_comm_plan.md
    ├── serial_test_workflow_plan.md
    ├── tool_data_passing_architecture.md
    ├── user_settings_menu_plan.md
    └── visual_pipeline_editor_plan.md

--------
安装说明
--------

1. 安装 Python 3.8 或更高版本

2. 安装依赖包：
   pip install -r requirements.txt

3. 安装大恒 GalaxySDK：
   - 从大恒官网下载并安装 GalaxySDK（包含 gxipy Python 包）
   - 或将 gxipy/ 目录及 GxIAPI.dll、DxImageProc.dll、MCDLL_NET.dll
     放置到项目根目录（已预置）
   - 确保 DLL 在系统 PATH 或程序运行目录中可被加载

4. 运行程序：
   python main.py

--------
打包说明
--------

使用 PyInstaller 打包为独立可执行文件：
    pyinstaller main.spec

打包后的文件位于 dist/Vision_Substrate_Silicone/ 目录下。

打包完成后可运行 cleanup_after_build.bat 清理不需要的大文件
（如 Qt5 的 WebEngine、QML 等 DLL 和多语言翻译文件）。

runtime_hook.py 会在打包后的程序启动时自动设置 DLL 搜索路径，
确保 GxIAPI.dll / DxImageProc.dll 能被正确加载。

--------
使用说明
--------

1. 登录系统
   - 默认管理员账号：admin / admin123
   - 默认工程师账号：engineer / 123456
   - 默认操作员账号：operator / 123456

2. 工人模式
   - 从方案列表中选择检测方案，点击"导入方案"
   - 点击"导入图像"加载待检测图片，或连接相机后点击"拍照"
   - 点击"开始检测"执行流水线
   - 查看OK/NG结果

3. 设计模式
   - 创建或打开检测方案
   - 从算子工具箱拖拽算子到流水线插槽
   - 点击算子配置参数（支持实时预览）
   - 加载测试图像进行预览
   - 保存方案

4. 相机操作
   a. 打开相机面板（默认在主界面右侧）
   b. 点击"刷新"搜索相机设备
   c. 从下拉列表选择相机，点击"打开"
   d. 调节参数：
      - 曝光时间：滑块或 +/- 按钮，支持自动曝光
      - 增益：滑块或 +/- 按钮，支持自动增益
      - 帧率：滑块或 +/- 按钮
      - 白平衡：R/G/B 三通道独立滑块，或选择色温预设
   e. 点击"拍照"采集单帧图像
   f. 触发模式：切换至"触发模式（软触发）"后，点击"发送软触发"采集

5. 串口通信
   - 通过菜单栏「通信 > 串口通信」打开串口通信对话框
   - 点击"扫描端口"检测可用串口
   - 选择端口并配置参数（波特率/数据位/校验位/停止位/流控制）
   - 点击"打开串口"建立连接
   - 在发送区输入数据，选择文本或HEX模式，点击"发送"
   - 接收区实时显示接收到的数据

--------
视觉工具详解
--------

预处理工具：
  - 灰度化：将彩色图像转为灰度图
  - 高斯滤波：高斯模糊降噪（自动校正核大小为奇数）
  - 直方图均衡化：增强图像对比度
  - 形态学操作：腐蚀、膨胀、开运算、闭运算、梯度、顶帽、黑帽
    支持结构元素形状选择：矩形（MORPH_RECT）、椭圆（MORPH_ELLIPSE）、十字（MORPH_CROSS）
    支持迭代次数设置，自动校正核大小为奇数
  - 多区域ROI：在图像上绘制多个感兴趣区域
    支持区域命名，支持百分比坐标（0~100，自动适配图像分辨率）
    支持区域启用/禁用，支持导出/导入配置
  - 中值滤波：中值滤波降噪（对椒盐噪声有效，自动校正核大小为奇数）
  - 缩放：图像缩放，支持按比例缩放和固定尺寸
    输出 scale_x / scale_y 用于下游步骤的 ROI 坐标跟踪
  - 自适应阈值：根据局部区域自适应计算阈值（均值/高斯），输出二值图

特征提取工具：
  - Canny边缘检测：经典边缘检测算子
    支持手动阈值和自动Otsu阈值（自动模式下使用中值滤波预处理后计算最佳阈值）
    自动校正 low_threshold <= high_threshold
  - 阈值分割：支持传统阈值和自适应阈值两种模式（QStackedWidget切换）
    传统模式：二值化、反二值化、截断、归零、反归零，支持Otsu/Triangle自动阈值
    自适应模式：均值（ADAPTIVE_THRESH_MEAN_C）/高斯（ADAPTIVE_THRESH_GAUSSIAN_C）
  - 轮廓分析：查找并分析图像轮廓
    支持按面积、周长、x坐标、y坐标、宽度、高度排序（升序/降序）
    输出总轮廓数、总面积、最大/最小轮廓信息
  - Blob检测：检测图像中的斑点特征（基于SimpleBlobDetector）
  - 轮廓筛选：按面积、周长、宽高比等筛选轮廓
    支持 AND（全部满足）和 OR（任一满足）两种逻辑运算
  - 直线检测：基于霍夫变换的直线检测
    支持自动参数估计（根据图像对角线长度自动计算阈值）
    支持HoughLinesP（概率霍夫变换）输出线段端点
    自动校正 canny_low <= canny_high
  - 矩形检测：基于轮廓逼近的矩形检测
    检测4顶点凸包轮廓，支持面积范围、宽高比范围过滤
    支持最大检测数量限制

几何检测工具：
  - 圆检测：霍夫圆检测
    支持自动参数估计（根据图像对角线计算dp/minDist）
    支持半径范围限制（minRadius/maxRadius）
    支持圆心距去重（_filter_by_center_distance），避免重复检测
    自动校正 minRadius <= maxRadius
  - 直线检测(霍夫)：基于霍夫变换的直线检测
    支持自动参数估计（根据图像对角线计算阈值）
    自动校正 canny_low <= canny_high
  - 矩形检测(轮廓)：基于轮廓逼近的矩形检测
  - Blob检测(简单)：基于连通域的斑点检测
    支持面积过滤（min_area/max_area）
    支持圆度过滤（min_circularity/max_circularity）
    支持凸度过滤（min_convexity/max_convexity）
    支持惯性比过滤（min_inertia_ratio/max_inertia_ratio）
    支持颜色过滤（filter_by_color + blob_color：0=黑色，255=白色）
    支持最大检测数量限制（max_count）

测量工具：
  - 面积测量：测量指定区域的像素面积
  - 距离测量：测量两点间距离
  - 点测量：获取指定点的像素值
  - 线测量：测量线段长度
  - 角度测量：测量两条线段的夹角
  - 目标计数：统计图像中目标数量

识别工具：
  - 颜色识别：识别图像中的颜色分布
    支持 HSV 和 Lab 两种色彩空间切换
    HSV模式：适合工业零件颜色检测，对光照变化鲁棒
    Lab模式：适合颜色精确测量，色差分析
    支持区域颜色占比分析（analyze_regions），输出每个区域的面积占比
    预设颜色库：HSV_PRESETS（红/绿/蓝/黄/橙/紫/青/白/黑/灰）
    Lab预设：LAB_PRESETS
  - 模板匹配：支持三种匹配模式
    标准模式：支持掩膜（mask），忽略模板中不必要的区域（如背景变化）
    多角度模式：自动旋转模板匹配，输出分数曲线（score_curve）和最佳匹配角度
    特征点模式：SIFT/ORB特征匹配，支持最小匹配数过滤
    支持NMS非极大值抑制去重，分数阈值判定
  - 边缘匹配：基于边缘特征的模板匹配
  - 快速匹配：基于图像金字塔的快速匹配
  - 脚垫识别：专用脚垫检测工具

辅助工具：
  - 坐标转换：坐标系统转换（像素到物理单位）
  - 数值计算：数值运算（加减乘除等），支持表达式 {A}+{B}，合格范围判定
  - 逻辑判断：最终OK/NG判定工具，支持两种模式
    表达式模式（推荐）：支持自然语言表达式解析
      - AND(条件1, 条件2) — 全部条件通过才通过
      - OR(条件1, 条件2)  — 任一条件通过即通过
      - 条件格式：变量名 > 100，变量名 < 50，变量名 >= 100，变量名 <= 50
      - 变量名格式：工具名.数据键（如 AreaMeasure.total_area）
      - 支持范围语法：100 <= 面积 <= 200
      - 支持 != 不等运算符
    条件模式（兼容）：向后兼容的固定条件列表模式
    调试界面：显示所有上游步骤的输出值，便于理解判断逻辑

--------
相机图像处理流程
--------

相机原始数据 → BGR 图像的完整处理链路：

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

配置参数位于 camera_manager.py 顶部：
  - CAMERA_GAMMA            : Gamma 校正值（默认 2.2）
  - CAMERA_SHARPEN_STRENGTH : 锐化强度（默认 0.5）
  - CAMERA_16BIT_CONVERSION : 16bit→8bit 方式（默认 "shift"）
  - CAMERA_BAYER_PATTERN    : Bayer 排列模式（默认 GB2BGR）

--------
参数自动校正
--------

系统内置了多项参数自动校正机制，提高易用性和鲁棒性：

1. 核大小自动校正：高斯滤波、中值滤波、形态学操作的核大小自动调整为奇数
2. 阈值自动校正：Canny边缘检测、直线检测的 low <= high 自动校正
3. 半径自动校正：圆检测的 minRadius <= maxRadius 自动校正
4. 自动参数估计：直线检测、圆检测、霍夫直线检测支持根据图像尺寸自动估算参数
5. 自动阈值：Canny边缘检测支持Otsu算法自动计算最佳阈值

--------
注意事项
--------

1. 首次运行时会自动创建 data/ 目录及其子目录
2. 相机功能需要大恒 GalaxySDK（gxipy）及配套 DLL
3. 如果没有相机，可以使用"加载图像"功能测试流水线
4. 系统日志保存在 data/logs/ 目录下，自动按天轮转，保留30天
5. 方案文件为JSON格式，可手动编辑，但建议通过界面操作
6. Canny边缘检测的"自动阈值"选项使用Otsu算法，适用于光照变化大的场景
7. 形态学操作的核大小必须为奇数，系统会自动校正
8. 多区域ROI的百分比坐标范围为0~100，系统自动根据图像分辨率转换为像素坐标
9. 模板匹配的掩膜图像需与模板图像尺寸一致，灰度图中黑色区域将被忽略
10. 逻辑判断的表达式模式中，变量名使用"工具名.数据键"格式，可在调试界面中查看可用变量
11. 串口通信功能依赖 pyserial 库，请确保已安装（pip install pyserial）
12. 白平衡默认值（R=1.5, G=1.0, B=1.8）针对偏绿场景校正，可在相机面板中实时调节
13. Gamma 校正和锐化强度可在 camera_manager.py 顶部调整，修改后重启程序生效

--------
更新日志
--------

v2.3.0 (2026-06-26)
- 迁移相机SDK：海康威视 MVS → 大恒 GalaxySDK (gxipy)
- 新增 gxipy/ 目录及配套 DLL（GxIAPI.dll、DxImageProc.dll、MCDLL_NET.dll）
- 重写 camera_manager.py：大恒 SDK 设备枚举、打开/关闭、取流、参数读写
- 新增跨网段设备搜索（同网段未发现时自动切换）
- 新增 GigE 网络参数优化（包大小、延迟、帧传输）
- 新增白平衡 R/G/B 三通道独立系数设置
- 新增相机面板白平衡 UI（R/G/B 滑块 + 色温预设）
- 新增图像后处理：Gamma 校正（γ=2.2）
- 新增图像后处理：USM 锐化（强度 0.5）
- 优化 16bit→8bit 转换：固定右移替代 NORM_MINMAX，避免亮度跳动
- 更新打包配置（main.spec）适配大恒 SDK DLL
- 更新 runtime_hook.py 适配大恒 DLL 搜索路径
- 新增 plans/hikvision_to_daheng_migration_plan.md 迁移计划文档
- 新增 plans/camera_init_fixed_params_plan.md 相机初始化参数计划文档
- 新增 plans/default_user_change_record.md 默认用户变更记录

v2.2.0 (2026-06-10)
- 新增串口通信核心模块（serial_comm.py），支持端口扫描、参数配置、收发管理
- 新增串口通信对话框（serial_dialog.py），独立的串口通信窗口界面
- 新增串口自动测试工作流（serial_test_workflow.py），状态机驱动的自动化测试流程
- 新增 PyInstaller 运行时钩子（runtime_hook.py），自动设置 DLL 搜索路径
- 新增 plans/ 目录下多个开发计划文档（亮度测量、脚垫检测优化、ROI结果显示等）
- 新增 model/ 目录下深度学习样本和标题检测样本数据
- 新增 data/icon.png 应用程序图标
- 新增 data/schemes/默认方案.json 默认检测方案
- 新增 .gitignore 版本控制忽略配置
- 优化项目目录结构，增加 core/ 核心模块的串口通信相关功能

v2.1.0 (2026-06-03)
- 新增形态学操作结构元素形状选择（矩形/椭圆/十字）和迭代次数
- 新增多区域ROI百分比坐标支持，区域命名，导出/导入功能
- 新增图像缩放ROI坐标跟踪（输出scale_x/scale_y）
- 新增Canny边缘检测自动阈值功能（Otsu算法）
- 新增阈值分割集成自适应阈值模式（均值/高斯）
- 新增轮廓分析多维度排序（x/y/宽/高）
- 新增轮廓筛选AND/OR多条件逻辑运算
- 新增直线检测自动参数估计和HoughLinesP支持
- 新增矩形检测工具（基于轮廓逼近）
- 新增圆检测自动参数估计、半径范围限制、圆心距去重
- 新增霍夫直线检测自动参数估计
- 新增Blob检测增强过滤（圆度/凸度/惯性比/颜色/最大数量）
- 新增颜色识别HSV/Lab色彩空间切换和区域颜色占比分析
- 新增模板匹配掩膜支持、多角度分数曲线输出
- 新增逻辑判断表达式解析模式（AND/OR语法）和调试界面
- 新增ToolResult overlay_image字段支持工业叠加图层
- 新增参数自动校正机制（核大小奇数、阈值大小关系、半径大小关系）
- 新增可缩放图片显示控件（ZoomableLabel）
- 新增脚垫识别工具（FootPadDetect）
- 新增打包后清理脚本（cleanup_after_build.bat）
- 优化参数配置对话框的实时预览交互
- 改进算子工具箱的搜索和拖拽体验
- 优化深色主题UI样式
- 修复方案加载和保存的兼容性问题

v1.0.0 (2024-01-01)
- 初始版本发布
- 支持流水线式视觉检测
- 支持工人/工程师双模式
- 支持30种视觉工具
- 支持方案管理和用户权限管理
- 支持海康威视相机
