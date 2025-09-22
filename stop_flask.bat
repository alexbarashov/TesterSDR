@echo off
echo === Stopping Flask servers ===

REM Показываем активные Python процессы
echo.
echo Active Python processes:
tasklist /fi "imagename eq python.exe" 2>nul | find "python.exe"

REM Останавливаем все процессы Python (осторожно!)
echo.
echo Killing all Python processes...
taskkill /f /im python.exe 2>nul
if %errorlevel% equ 0 (
    echo Python processes terminated.
) else (
    echo No Python processes found.
)

REM Проверяем и освобождаем порты
echo.
echo Checking ports 8737 and 8738...

REM Порт 8737
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8737.*LISTENING" 2^>nul') do (
    echo Killing process %%a on port 8737
    taskkill /f /pid %%a 2>nul
)

REM Порт 8738
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8738.*LISTENING" 2^>nul') do (
    echo Killing process %%a on port 8738
    taskkill /f /pid %%a 2>nul
)

REM Порт 8739
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8739.*LISTENING" 2^>nul') do (
    echo Killing process %%a on port 8738
    taskkill /f /pid %%a 2>nul
)


REM Порт 8740
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8740.*LISTENING" 2^>nul') do (
    echo Killing process %%a on port 8738
    taskkill /f /pid %%a 2>nul
)

REM Проверяем результат
echo.
echo Checking if ports are free...
netstat -ano | findstr ":8737.*LISTENING" && echo Port 8737 still occupied || echo Port 8737 is free
netstat -ano | findstr ":8738.*LISTENING" && echo Port 8738 still occupied || echo Port 8738 is free
netstat -ano | findstr ":8737.*LISTENING" && echo Port 8737 still occupied || echo Port 8739 is free
netstat -ano | findstr ":8738.*LISTENING" && echo Port 8738 still occupied || echo Port 8740 is free

echo.
echo === Flask servers stopped ===
echo Press any key to exit...
pause >nul