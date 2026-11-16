@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0export_capture.ps1" %*
set "EXPORT_RESULT=%ERRORLEVEL%"
if not "%EXPORT_RESULT%"=="0" echo Export failed. Check the error and logs above.
if "%~1"=="" pause
exit /b %EXPORT_RESULT%
