@echo off
setlocal enabledelayedexpansion
echo Checking Python
call :DOES_PYTHON_EXIST
call :CHECK_DEPENDENCIES
python AutoCardCrop.py
goto :EOF

:DOES_PYTHON_EXIST
echo Checking if Python Exists
py --version >NUL 2>NUL
if errorlevel 1 (
echo Python is not installed on your system.
goto :INSTALL_PYTHON
)
exit /b

:INSTALL_PYTHON
echo Installing Python...
powershell -Command "Start-Process winget -ArgumentList 'install --id Python.Python.3.11 -e --silent --override \"/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 TargetDir=C:\Python\Python311\" --accept-package-agreements --accept-source-agreements' -Verb RunAs -Wait"
set "PATH=C:\Python\Python311\;C:\Python\Python311\Scripts\;%PATH%"
exit /b

:CHECK_DEPENDENCIES
call :check_install "import cv2" OpenCV-Python
call :check_install "from PIL import Image" Pillow
call :check_install "import numpy" NumPy
exit /b

:check_install
echo Checking for %~2...
python -c "%~1" >nul 2>nul
if !errorlevel! neq 0 (
    echo %~2 Not Found. Installing now...
    python -m pip -qqq install %~2
)
exit /b