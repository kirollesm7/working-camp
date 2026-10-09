@echo off
setlocal
title Clothes Counter - server
cd /d "%~dp0app"

call :find_python || goto :no_python

rem First run on a new laptop (or OpenCV 5, which lacks HOG/ML): install the right libraries
%PY% -c "import flask, cv2, numpy, cryptography, openpyxl, PIL, PySide6; assert hasattr(cv2, 'HOGDescriptor') and hasattr(cv2, 'ml')" 1>nul 2>nul
if errorlevel 1 (
  echo Installing the libraries the app needs - first run only, needs internet...
  %PY% -m pip install --upgrade -r "%~dp0requirements.txt"
  if errorlevel 1 goto :pip_failed
)

echo Starting the clothes counter server...
echo The first start on a new laptop takes a few minutes - wait for the links below.
%PY% clothes_server.py
pause
exit /b

:find_python
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3" && exit /b 0
python --version >nul 2>nul && set "PY=python" && exit /b 0
exit /b 1

:no_python
echo.
echo Python is not installed on this laptop.
echo   1. Download it from https://www.python.org/downloads/
echo   2. In the installer tick "Add python.exe to PATH"
echo   3. Run this file again.
start "" https://www.python.org/downloads/
pause
exit /b 1

:pip_failed
echo.
echo Installing the libraries failed. Check the internet connection and run this file again.
pause
exit /b 1
