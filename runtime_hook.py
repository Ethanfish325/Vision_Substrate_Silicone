# -*- coding: utf-8 -*-
"""
PyInstaller Runtime Hook
=======================
在打包后的程序启动时执行，用于设置 DLL 搜索路径，
确保 GxIAPI.dll 和 DxImageProc.dll 能被正确加载。
"""
import os
import sys


def _setup_daheng_dll_path():
    """
    将大恒 SDK DLL 所在目录添加到 DLL 搜索路径中。
    
    打包后目录结构：
        dist/Vision_Substrate_Silicone/
            Vision_Substrate_Silicone.exe
            _internal/
                GxIAPI.dll          <-- 大恒相机 SDK DLL
                DxImageProc.dll     <-- 大恒图像处理 DLL
                ...
    """
    if getattr(sys, 'frozen', False):
        # 打包后的环境：DLL 在 _internal/ 根目录下
        base_dir = os.path.dirname(sys.executable)
        dll_dir = os.path.join(base_dir, '_internal')
    else:
        # 开发环境：DLL 在项目根目录
        dll_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.isdir(dll_dir):
        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
        print(f"[RuntimeHook] 已添加 DLL 搜索路径: {dll_dir}")
    else:
        print(f"[RuntimeHook] 警告: DLL 目录不存在: {dll_dir}")


_setup_daheng_dll_path()
