cd C:\Program Files\PothosSDR\bin\

@echo off
:loop
echo --- Передача файла ---
hackrf_transfer.exe -t C:\work\TesterSDR\captures\psk406.iq -f 406037000 -s 1000000 -x 40

:: Ждать 4 секунд
timeout /t 5 >nul

goto loop

