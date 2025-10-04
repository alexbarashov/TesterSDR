@echo off
echo ====================================================
echo COSPAS/SARSAT Beacon Tester Web - Startup
echo ====================================================

REM 1) Запуск DSP сервиса
echo Starting DSP service...
start "DSP Service" "C:\Users\alexb\AppData\Local\Programs\Python\Python39\python.exe" "C:\work\TesterSDR\beacon406\beacon_dsp_service.py" --pub tcp://127.0.0.1:8781 --rep tcp://127.0.0.1:8782

REM 2) Пауза для запуска DSP сервиса
timeout /t 3 >nul

REM 3) Запуск веб-интерфейса beacon_tester_web
echo Starting web interface...
start "Web Interface" "C:\Users\alexb\AppData\Local\Programs\Python\Python39\python.exe" "C:\work\TesterSDR\beacon406\beacon_tester_web.py"

REM 4) Пауза для запуска веб-интерфейса
timeout /t 3 >nul

REM 5) Открыть браузер
echo Opening browser...
start "" http://127.0.0.1:8738/

echo ====================================================
echo Services started:
echo DSP Service: ZeroMQ PUB=8781, REP=8782
echo Web Interface: http://127.0.0.1:8738/
echo ====================================================
echo Press any key to exit this window...
pause >nul
