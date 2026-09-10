@echo off
chcp 65001 >nul
setlocal

rem ============================================================
rem  32 位打包脚本 (必须用 32 位 Python)
rem ============================================================
rem  本项目依赖的 GxIAPI.dll / DxImageProc.dll(大恒相机 SDK)与
rem  smcsh_mbs.dll(SMC6480 运控卡)都是 32 位(x86),因此必须用
rem  32 位 Python 打包;用 64 位打包会在启动时报
rem  "NameError: name 'dll' is not defined"。
rem
rem  如需修改解释器路径,只改下面这一行:
set "PY32=C:\Users\fyx\AppData\Local\Programs\Python\Python39-32\python.exe"

if not exist "%PY32%" (
    echo [错误] 未找到 32 位 Python: %PY32%
    echo        请编辑 build_32.bat 中的 PY32 变量指向已安装的 32 位解释器。
    exit /b 1
)

echo ========================================
echo  打包(32 位) - Vision_Substrate_Silicone
echo ========================================
"%PY32%" -c "import sys,struct;print('解释器:',sys.executable, str(struct.calcsize('P')*8)+'位', sys.version.split()[0])"
if errorlevel 1 (
    echo [错误] 无法运行 32 位 Python。
    exit /b 1
)

echo.
echo [1/2] PyInstaller 打包中(已加 --clean 清理旧构建)...
"%PY32%" -m PyInstaller main.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [错误] 打包失败,请查看上面的日志。
    exit /b 1
)

echo.
echo [2/2] 清理不需要的大文件...
call cleanup_after_build.bat

echo.
echo ========================================
echo  打包完成
echo  输出: dist\Vision_Substrate_Silicone\Vision_Substrate_Silicone.exe
echo ========================================
endlocal
