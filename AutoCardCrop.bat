@echo off && color 02
setlocal enabledelayedexpansion
echo Checking Python
goto :DOES_PYTHON_EXIST

:DOES_PYTHON_EXIST
echo Checking if Python Exists
py --version >NUL 2>NUL
if errorlevel 1 (
echo Python is not installed on your system.
goto :DOWNLOAD_PYTHON
) 
goto :EXECUTE_SCRIPT

:DOWNLOAD_PYTHON
echo Downloading Python 3.11.0...
IF EXIST "%CD%\python-3.11.0-amd64.exe" (
  echo Found Installer at "%CD%\python-3.11.0-amd64.exe"
) ELSE (
  powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe' -OutFile '%CD%\python-3.11.0-amd64.exe'"
  echo Python download completed.
)
goto :INSTALL_PYTHON

:INSTALL_PYTHON
echo Installing Python...
powershell -Command "& {Start-Process '%CD%\python-3.11.0-amd64.exe' -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 TargetDir=C:\Python\Python311' -Wait}"
del python-3.11.0-amd64.exe
timeout /t 30 /nobreak > nul
echo Python installation complete.
goto: :SET_ENV_VAR

:SET_ENV_VAR
for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "UserPath=%%B"

echo !UserPath! | findstr /C:"C:\Python\Python311\" >nul
if !errorlevel! neq 0 (
    setx PATH "C:\Python\Python311\;C:\Python\Python311\Scripts\;!UserPath!"
    set "PATH=C:\Python\Python311\;C:\Python\Python311\Scripts\;%PATH%"
) else (
    echo Python already in PATH, skipping.
)

:EXECUTE_SCRIPT
@color 07
echo Checking Dependencies

python -c "import cv2" >nul 2>nul
if !errorlevel! neq 0 (
  echo opencv-python Not Found. Installing...
  python -m pip -qqq install opencv-python
)

python -c "from PIL import Image" >nul 2>nul
if !errorlevel! neq 0 (
  echo Pillow Not Found. Installing...
  python -m pip -qqq install Pillow
)

python -c "import numpy" >nul 2>nul
if !errorlevel! neq 0 (
  echo NumPy Not Found. Installing...
  python -m pip -qqq install numpy
)

python Rotate.py
goto :EOF
