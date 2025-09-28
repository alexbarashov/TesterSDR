@echo off
setlocal ENABLEDELAYEDEXECUTION

REM ===== One-shot HackRF transmission =====
REM Edit the variables below to match your setup.

REM Path to hackrf_transfer.exe
set "HACKRF_EXE=C:\Program Files\PothosSDR\bin\hackrf_transfer.exe"

REM Path to the IQ file to transmit (interleaved complex float32 or s8 depending on your file)
set "IQ_FILE=C:\work\TesterSDR\captures\psk406.iq"

REM RF center frequency in Hz (e.g., 406037000 for 406.037 MHz)
set "FREQ_HZ=406037000"

REM Sample rate in samples per second (must match your IQ file)
set "SAMPLE_RATE=1000000"

REM TX gain (0..47 dB) and RF AMP (0=off, 1=on)
set "TXVGA=40"
set "AMP=1"

echo ===== HackRF one-shot TX =====
echo EXE:  %HACKRF_EXE%
echo IQ :  %IQ_FILE%
echo Freq: %FREQ_HZ% Hz
echo SR  : %SAMPLE_RATE% sps
echo TXG : %TXVGA% dB, AMP=%AMP%
echo =================================

REM --- Sanity checks ---
if not exist "%HACKRF_EXE%" (
  echo [ERROR] hackrf_transfer.exe not found: "%HACKRF_EXE%"
  exit /b 2
)
if not exist "%IQ_FILE%" (
  echo [ERROR] IQ file not found: "%IQ_FILE%"
  exit /b 3
)

REM --- One-shot transmit: play file once and exit (no -R) ---
"%HACKRF_EXE%" -t "%IQ_FILE%" -f %FREQ_HZ% -s %SAMPLE_RATE% -x %TXVGA% -a %AMP%
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo [ERROR] hackrf_transfer exited with code %ERR%
  exit /b %ERR%
) else (
  echo [OK] Transmission finished (single pass).
)

endlocal
