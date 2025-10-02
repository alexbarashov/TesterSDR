@echo off
rem Закрываем процессы, слушающие 8782 (REP) и 8781 (PUB)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8782 " ^| findstr LISTENING') do taskkill /PID %%p /F
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8781 " ^| findstr LISTENING') do taskkill /PID %%p /F
echo.
echo === dsp servers stopped ===
echo Press any key to exit...
pause >nul