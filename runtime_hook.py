# -*- coding: utf-8 -*-
"""
PyInstaller Runtime Hook
=======================
在打包后的程序启动时执行，用于设置 DLL 搜索路径，
确保 GxIAPI.dll / DxImageProc.dll（大恒相机 SDK）与 MCDLL_NET.dll
（NMC1400 运控卡）能被正确加载。

注意：这三个 DLL 都是 32 位(x86)，因此程序必须用 32 位 Python 打包
（见 main.spec 顶部的位数校验与 build_32.bat）。
"""
import os
import sys

# os.add_dll_directory 返回的句柄必须保持引用，否则目录会被回收、失效
_dll_dir_handles = []


def _setup_daheng_dll_path():
    """
    将本地 DLL 所在目录添加到 DLL 搜索路径中。

    打包后目录结构：
        dist/Vision_Substrate_Silicone/
            Vision_Substrate_Silicone.exe
            _internal/
                GxIAPI.dll          <-- 大恒相机 SDK DLL
                DxImageProc.dll     <-- 大恒图像处理 DLL
                MCDLL_NET.dll       <-- NMC1400 运控卡 DLL
                pyzbar/
                    libzbar-32.dll  <-- 条码识别(pyzbar)依赖
                    libiconv-2.dll
    开发环境：DLL 在项目根目录。

    同时做三件事以兼容各种 DLL 加载方式：
      1) os.add_dll_directory()：Python 3.8+ 推荐方式，对
         ctypes.WinDLL(name, winmode=0) 这类按名加载有效；
      2) 前置到 PATH：兼容不查询 add_dll_directory 目录的老式加载；
      3) 保存句柄引用，避免 add_dll_directory 注册的目录被垃圾回收。
    """
    dirs = []
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        # PyInstaller 6 的 onedir 布局：二进制都在 _internal/ 下
        dirs.append(os.path.join(base_dir, '_internal'))
        # pyzbar 的 zbar DLL 在 _internal/pyzbar/
        dirs.append(os.path.join(base_dir, '_internal', 'pyzbar'))
        # 兼容 DLL 被放在 exe 同级的情况
        dirs.append(base_dir)
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            dirs.append(meipass)
            dirs.append(os.path.join(meipass, 'pyzbar'))
        # exe 同级若有 pyzbar 目录也加上
        dirs.append(os.path.join(base_dir, 'pyzbar'))
    else:
        # 开发环境：DLL 在项目根目录
        dirs.append(os.path.dirname(os.path.abspath(__file__)))

    seen = set()
    for dll_dir in dirs:
        if not dll_dir or dll_dir in seen or not os.path.isdir(dll_dir):
            continue
        seen.add(dll_dir)
        try:
            if hasattr(os, 'add_dll_directory'):
                _dll_dir_handles.append(os.add_dll_directory(dll_dir))
        except OSError as e:
            print(f"[RuntimeHook] add_dll_directory 失败 {dll_dir}: {e}")
        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
        print(f"[RuntimeHook] 已添加 DLL 搜索路径: {dll_dir}")


def _preload_zbar_dlls():
    """以绝对路径预加载 zbar 及其依赖 DLL。

    pyzbar 的 zbar_library.load() 在 Windows 上是"先按当前工作目录、
    再按包目录"用 cdll.LoadLibrary(相对路径) 加载 libzbar-32.dll 与
    libiconv-2.dll。打包后工作目录通常不是程序目录,一旦搜索路径不含
    DLL 目录就会 OSError → pyzbar 不可用(一维码/二维码全部识别不到)。
    这里先用绝对路径把依赖与 zbar 载入进程,后续 pyzbar 按名加载时
    会直接命中已加载模块,从而与工作目录无关。

    注意:必须先加载 libiconv-2.dll,再加载 libzbar-32.dll。
    """
    candidates = []
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        roots = [os.path.join(base_dir, '_internal', 'pyzbar'),
                 os.path.join(base_dir, 'pyzbar'),
                 os.path.join(base_dir, '_internal'),
                 base_dir]
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            roots.append(os.path.join(meipass, 'pyzbar'))
            roots.append(meipass)
    else:
        roots = [os.path.dirname(os.path.abspath(__file__))]
    candidates = roots

    for name in ('libiconv-2.dll', 'libzbar-32.dll'):
        for root in candidates:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            try:
                import ctypes
                ctypes.CDLL(path)
                print(f"[RuntimeHook] 已预加载 {name}: {path}")
            except OSError as e:
                print(f"[RuntimeHook] 预加载 {name} 失败: {e}")
            break


_setup_daheng_dll_path()
_preload_zbar_dlls()
