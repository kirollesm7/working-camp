@echo off
setlocal
title Working Camp - carton counter
cd /d "%~dp0app"

call :find_python || goto :no_python

rem First run on a new laptop: install the libraries the app needs
%PY% -c "import PySide6, serial" 1>nul 2>nul
if errorlevel 1 (
  echo Installing the libraries the app needs - first run only, needs internet...
  %PY% -m pip install --upgrade -r "%~dp0requirements.txt"
  if errorlevel 1 goto :pip_failed
)

rem Start without a console window
start "" %PYW% main.py
exit /b

:find_python
set "PY=" & set "PYW="
py -3 --version >nul 2>nul && set "PY=py -3" && set "PYW=pyw -3" && exit /b 0
python --version >nul 2>nul && set "PY=python" && set "PYW=pythonw" && exit /b 0
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
