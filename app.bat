@echo off
REM 1) запустить сервер
start "" "C:\Users\alexb\AppData\Local\Programs\Python\Python39\python.exe" "C:\work\TesterSDR\beacon406\beacon_tester_web.py"
REM 2) дать серверу стартануть и открыть браузер
timeout /t 2 >nul
start "" http://127.0.0.1:8738/

