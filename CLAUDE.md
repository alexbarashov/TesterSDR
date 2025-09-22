# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Всегда отвечай на русском языке. Если генерируешь коммиты/описания/тексты — делай это по-русски, если явно не попросили иначе.

## Repository Overview

TesterSDR is a Software-Defined Radio (SDR) application for analyzing and demodulating 406 MHz emergency beacon signals (EPIRB/COSPAS-SARSAT). The system supports multiple SDR hardware backends and can process both real-time and recorded IQ data.

## Architecture

### Core Components

**beacon406/** - Main Python module containing:
- `beacon406-plot.py` - Main visualization application with real-time signal processing, RMS analysis, and PSK demodulation
- `lib/backends.py` - Unified SDR abstraction layer supporting RTL-SDR, HackRF, Airspy, SDRPlay, RSA306, and file playback
- `lib/demod.py` - PSK demodulation algorithms with safe message extraction
- `lib/metrics.py` - Phase metrics and PSK impulse processing
- `lib/config.py` - Backend selection and configuration (defaults to file playback mode)

**beacon406/apps/** - Applications and utilities:
- `tester_sdr_http_ui_stage_1_windows_single_file_app.py` - Web-based UI server (port 8737)
- `beacon_tester_gui.py` - PySide6/Qt GUI application
- `beacon_tester_web.py` - Flask-based COSPAS beacon tester web UI (port 8738)
- `beacon_tester_web_debug.py` - Debug version of beacon tester web UI
- `cospas_beacon_tester_v2.py` - COSPAS-SARSAT beacon tester interface
- Test scripts: `test_cf32_*.py` - Various CF32 file processing and testing utilities
- **gen/** subdirectory - Signal generation tools:
  - `generate_psk406_cf32.py` - Generate test PSK406 signals
  - `psk406_msg_gen.py` - PSK406 message generator
  - `backend_hackrf_tx.py` - HackRF transmit backend
  - `ui_psk406_tx.py` - Transmission UI

**captures/** - Directory for storing CF32 IQ recordings

### Backend System

The application uses a unified backend system (`lib/backends.py`) that abstracts different SDR hardware. Key backends:
- `SoapyBackend` - Universal interface for SoapySDR-supported devices
- `FilePlaybackBackend` - Replay CF32 recordings with optional IF offset compensation
- Hardware-specific implementations for RTL-SDR, HackRF, Airspy, SDRPlay, RSA306

Backend selection is controlled via `lib/config.py`:
- `BACKEND_NAME`: "auto", "soapy_rtl", "soapy_hackrf", "soapy_airspy", "soapy_sdrplay", "rsa306", "file"
- `BACKEND_ARGS`: Hardware-specific parameters or file path for playback
- `"auto"` mode automatically detects available SDR hardware in sequence: RTL-SDR → HackRF → Airspy → SDRPlay → RSA306
- Each SDR has calibration offsets and optimal sample rates defined in `SDR_CALIB_OFFSETS_DB` and `SDR_DEFAULT_HW_SR`

## Development Commands

### Running the Application

```bash
# Main plot application (uses backend from config.py)
cd beacon406
python beacon406-plot.py

# Web UI servers
python apps/tester_sdr_http_ui_stage_1_windows_single_file_app.py  # Port 8737
python beacon_tester_web.py  # Port 8738 (COSPAS beacon tester)
python beacon_tester_web_debug.py  # Debug version of beacon tester

# GUI applications
python beacon_tester_gui.py  # PySide6/Qt GUI
python apps/cospas_beacon_tester_v2.py  # COSPAS tester

# Test scripts (various signal processing tests)
python apps/test_cf32_RMS.py
python apps/test_cf32_to_phase_msg_FFT.py
python apps/test_f32_to_beacon.py

# Signal generation and transmission
python apps/gen/generate_psk406_cf32.py        # Generate test PSK406 signals
python apps/gen/psk406_msg_gen.py             # PSK406 message generator
python apps/gen/ui_psk406_tx.py               # Transmission UI
python apps/gen/backend_hackrf_tx.py          # HackRF transmit backend

# Utilities
python apps/epirb_hex_decoder.py              # EPIRB hex message decoder
```

### Quick Start

Use the provided batch file for easy startup:
```bash
# Windows: Launch HTTP UI server and open browser
app.bat
```

This starts the web-based UI server at http://127.0.0.1:8738/ using `beacon_tester_web.py`.

Note: The batch file uses a hardcoded Python path: `C:\Users\alexb\AppData\Local\Programs\Python\Python39\python.exe`

### Stop Running Services

To stop all Flask servers and free up ports:
```bash
# Windows: Stop all Python processes and Flask servers
stop_flask.bat
```

This will terminate all Python processes and specifically check/free ports 8737, 8738, 8739, and 8740.

### Dependencies

Install required packages:
```bash
pip install numpy matplotlib pyqtgraph PyQt5 PySide6 scipy pyrtlsdr flask
# For SDR hardware support:
pip install SoapySDR
# For RSA306 support (if using Tektronix hardware):
pip install pyVISA
```

**Environment Setup:**
- Python 3.9+ required (hardcoded path in `app.bat` uses Python 3.9)
- Environment variables configured in `.env` file: `PYTHONPATH=${workspaceFolder}`
- VSCode settings support automatic Python path resolution

### Configuration

Edit `beacon406/lib/config.py` to select SDR backend:
```python
# For file playback:
BACKEND_NAME = "file"
BACKEND_ARGS = r"C:/work/TesterSDR/captures/your_recording.cf32"

# For RTL-SDR:
BACKEND_NAME = "soapy_rtl"
BACKEND_ARGS = None

# For HackRF:
BACKEND_NAME = "soapy_hackrf"
BACKEND_ARGS = None

# For auto-detection:
BACKEND_NAME = "auto"
BACKEND_ARGS = None
```

## Signal Processing Pipeline

### PSK Demodulation Flow
1. **Edge Detection** (`demod.py`)
   - `detect_all_steps_by_mean_fast()` - Finds phase transitions using sliding window (40 samples, 0.5 threshold)
   - Starts at sample 25000 to skip initial transients
   - Uses cumulative sum for O(1) complexity per step

2. **Data Extraction** (`demod.py`)
   - `extract_half_bits()` - Samples phase at midpoints between edges
   - `halfbits_to_bytes_fast()` - Converts Manchester-encoded bits (10→1, 01→0)
   - Handles up to 500 half-bits (250 bits, 31.25 bytes)

3. **Phase Processing** (`metrics.py`)
   - LPF at 12 kHz with 129-tap FIR filter (Hamming window)
   - 4x decimation after filtering
   - Linear trend removal for frequency offset compensation
   - Phase zero reference from first 2ms baseline

4. **Pulse Analysis** (`demod.py:calculate_pulse_params()`)
   - PosPhase/NegPhase: Mean of high/low plateaus
   - PhRise/PhFall: 10%-90% transition times with sub-sample interpolation
   - Ass: Asymmetry metric from τ1/τ2 timing differences
   - Tmod: Median period between rising edges

### Key Parameters in `beacon406-plot.py`
- `TARGET_SIGNAL_HZ`: Target signal frequency (406.037 MHz for beacons)
- `IF_OFFSET_HZ`: Intermediate frequency offset for tuning (-37 kHz typical)
- `SAMPLE_RATE_SPS`: Sample rate (typically 1 MHz)
- `RMS_WIN_MS`: RMS window size in milliseconds (1.0 ms)
- `VIS_DECIM`: Visualization decimation factor (2048)
- `PULSE_THRESH_DBM`: Detection threshold (-45 dBm)
- `READ_CHUNK`: SDR read buffer size (65536 samples)

## File Formats

- **CF32 files**: Complex float32 IQ samples (interleaved I/Q pairs)
- Sample rate and center frequency must match recording parameters
- File backend applies IF offset compensation during playback

## Testing

No formal test framework is configured. Test functionality using:
- Scripts in `apps/` directory for specific signal processing tests
- File playback mode with recorded signals in `captures/`
- Real-time testing with connected SDR hardware

### Test Files Available
The `captures/` directory contains multiple test recordings for development and testing:
- **PSK406 test signals**: `psk406msg_f*.cf32` (generated at different frequencies: 50, 75, 100, 150 Hz)
- **Real SDR captures**: `iq_pulse_406-*.cf32`, `iq_pulse_20250916_*.cf32` (actual beacon recordings)
- **AIS signals**: `iq_pulse_AIS_*.cf32` (maritime AIS test signals)
- **DSC signals**: `iq_pulse_DSC_*.cf32` (Digital Selective Calling test data)
- **AM121 data**: `iq_121*.cf32` (121.5 MHz emergency frequency recordings)
- **RSA306 captures**: Various Tektronix RSA306 recordings
- **Floating point data**: `AM121_out.f32` (processed floating point data)

Default test file in config: `psk406msg_f75.cf32` (75 Hz PSK signal)

## Important Notes

- The system uses Windows-style paths with forward slashes (e.g., `C:/work/TesterSDR/`)
- Python 3.13 compatibility (uses proper imports for typing annotations)
- Thread-safe implementation for real-time SDR data processing
- Automatic sample rate decimation for hardware with higher native rates (e.g., 2 MS/s → 1 MS/s)
- Safe demodulation with `phase_demod_psk_msg_safe()` - never raises exceptions, returns empty results on failure
- File backend applies inverse IF offset compensation during playback
- SDR backends handle underflow gracefully without stopping acquisition
- Environment configuration uses `.env` file and VSCode settings for Python path resolution
- Claude Code permissions are configured to allow pip installs and Python execution
- The main application path is hardcoded in `app.bat` to use Python 3.9 from user's AppData
- Web servers run on different ports to avoid conflicts (8737, 8738, 8739, 8740)
- No formal testing framework; use scripts in `apps/` for testing specific functionality

### Port Management
- **8737**: `tester_sdr_http_ui_stage_1_windows_single_file_app.py`
- **8738**: `beacon_tester_web.py` (main COSPAS beacon tester)
- **8739, 8740**: Additional development servers
- Use `stop_flask.bat` to terminate all Python processes and free ports

### Batch Files
- `app.bat`: Start main web UI and open browser automatically
- `stop_flask.bat`: Stop all Flask servers and Python processes
- `beacon406/apps/gen/ui_psk406.bat`: Launch PSK406 transmission UI
- `beacon406/apps/gen/406_msg_send_4sec.bat`: Send 4-second PSK406 message