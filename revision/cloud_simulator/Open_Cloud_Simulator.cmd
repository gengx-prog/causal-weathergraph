@echo off
"%~dp0..\..\..\.venv\Scripts\python.exe" "%~dp0launch.py"
if errorlevel 1 pause
