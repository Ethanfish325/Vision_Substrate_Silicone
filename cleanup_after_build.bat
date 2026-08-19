@echo off
chcp 65001 >nul
echo ========================================
echo  编译后清理脚本 - 删除不需要的大文件
echo ========================================

set "TARGET_DIR=dist\Vision_Substrate_Silicone\_internal\PyQt5\Qt5\bin"
set "TRANS_DIR=dist\Vision_Substrate_Silicone\_internal\PyQt5\Qt5\translations"

echo.
echo [1/2] 删除不需要的 Qt5 DLL 文件...
if exist "%TARGET_DIR%" (
    for %%f in (
        opengl32sw.dll
        libGLESv2.dll
        d3dcompiler_47.dll
        Qt5Quick.dll
        Qt5Qml.dll
        Qt5QmlModels.dll
        Qt5Network.dll
        Qt5Svg.dll
        Qt5DBus.dll
        Qt5WebSockets.dll
    ) do (
        if exist "%TARGET_DIR%\%%f" (
            del /f "%TARGET_DIR%\%%f" >nul 2>&1
            echo   已删除: %%f
        )
    )
) else (
    echo   目录不存在: %TARGET_DIR%
)

echo.
echo [2/2] 删除多余语言的翻译文件（只保留中文和英文）...
if exist "%TRANS_DIR%" (
    for %%f in ("%TRANS_DIR%\*.qm") do (
        echo %%~nxf | findstr /i "zh_CN zh_TW zh _en" >nul
        if errorlevel 1 (
            del /f "%%f" >nul 2>&1
            echo   已删除: %%~nxf
        )
    )
) else (
    echo   目录不存在: %TRANS_DIR%
)

echo.
echo ========================================
echo  清理完成！
echo ========================================
