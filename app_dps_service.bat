@echo off
REM 1) запустить DSP
start "" "C:\Users\alexb\AppData\Local\Programs\Python\Python39\python.exe" "C:\work\TesterSDR\beacon406\beacon_dsp_service.py
REM 1) запустить сервер
REM start "" "C:\Users\alexb\AppData\Local\Programs\Python\Python39\python.exe" "C:\work\TesterSDR\beacon406\beacon_tester_grok_web.py"
REM 2) дать серверу стартануть и открыть браузер
REM timeout /t 2 >nul
REM start "" http://127.0.0.1:8738/

