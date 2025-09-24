"""
test_cf32_to_FM_RMS_FFT.py
--------------------------
Offline FM analysis of one or more .cf32 IQ files:
  • Figure 1: RMS vs Time
  • Figure 2: FM (instantaneous frequency) vs Time
  • Figure 3: FFT magnitude (dB) vs Frequency

COMBAT / STRICT:
  - File names and basic parameters are set **in code** (no CLI).
  - No demodulation (commented placeholders remain).
  - Minimal knobs; safe defaults.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from pathlib import Path

# === path hack (fixed to project root) ===
ROOT = Path(__file__).resolve().parents[1]  # beacon406/
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
# ===================================
# Local module
from lib.processing_fm import fm_discriminator

# === USER CONFIG (set in code, as in original) =======================
IQ_FILES = [
    # Put one or more .cf32 files here (complex64 interleaved I/Q)
    #r"C:/work/TesterSDR/captures/UI_iq_1m.cf32",
    #r"C:/work/TesterSDR/captures/iq_pulse_DSC_dis.cf32",
    r"C:/work/TesterSDR/captures/psk406msg_f100.cf32",
    #r"C:/work/TesterSDR/captures/iq_pulse_AIS_m5.cf32",
]
FS_SPS      = 1_000_000.0   # sample rate (Sa/s)
PRE_LPF_HZ  = 50_000.0      # pre-LPF cutoff before decim (Hz)
DECIM       = 4             # decimation factor (>=1)
SMOOTH_HZ   = 2_000.0       # post-smoothing bandwidth for FM (Hz)
FFT_LEN     = 131072        # FFT length (power of two recommended)
# =====================================================================



# -----------------------------
# Helpers
# -----------------------------
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    w = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, w, mode="same")

def _to_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(eps, x))

def analyze_file(path: str):
    # -------- Load IQ --------
    iq = np.fromfile(path, dtype=np.complex64)
    if iq.size == 0:
        raise RuntimeError(f"Empty IQ or wrong format: {path} (expect .cf32 complex64)")

    # -------- FM discr --------
    out = fm_discriminator(
        iq=iq,
        fs=float(FS_SPS),
        pre_lpf_hz=float(PRE_LPF_HZ),
        decim=int(DECIM) if int(DECIM) >= 1 else 1,
        smooth_hz=float(SMOOTH_HZ),
        detrend=True,
        center=True,
        fir_taps=127,
    )
    freq_hz = out["freq_hz"]
    xs_ms_fm = out["xs_ms"]
    fs_out = float(out["fs_out"])

    # -------- RMS --------
    mag2 = np.abs(iq)**2
    win_rms = max(1, int(DECIM))
    rms_smoothed = np.sqrt(_moving_average(mag2, win=win_rms))
    rms_ds = rms_smoothed[::max(1, int(DECIM))].astype(np.float64)
    xs_ms_rms = (np.arange(rms_ds.size, dtype=np.float64) / fs_out) * 1e3
    rms_rel = rms_ds / (np.max(rms_ds) if np.max(rms_ds) > 0 else 1.0)
    rms_db = _to_db(rms_rel)

    # -------- FFT --------
    z_ds = iq[::max(1, int(DECIM))]
    NFFT = int(FFT_LEN)
    if z_ds.size < NFFT:
        pad = np.zeros(NFFT - z_ds.size, dtype=np.complex64)
        z_fft = np.concatenate([z_ds, pad])
    else:
        z_fft = z_ds[:NFFT]
    spec = np.fft.fftshift(np.fft.fft(z_fft, n=NFFT))
    spec_mag = np.abs(spec) / max(1.0, np.sqrt(NFFT))
    spec_db = _to_db(spec_mag)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0/(FS_SPS/max(1, int(DECIM))))) / 1e3  # kHz

    # -------- PLOTS --------
    base = os.path.basename(path)

    # RMS
    plt.figure()
    plt.plot(xs_ms_rms, rms_db)
    plt.title(f"RMS vs Time — {base}")
    plt.xlabel("Time, ms")
    plt.ylabel("RMS, dB (relative)")
    plt.grid(True)

    # FM
    plt.figure()
    plt.plot(xs_ms_fm, freq_hz)
    plt.title(f"FM Discriminator — {base}")
    plt.xlabel("Time, ms")
    plt.ylabel("Frequency, Hz")
    plt.grid(True)

    # FFT
    plt.figure()
    plt.plot(freqs, spec_db)
    plt.title(f"FFT Magnitude — {base} | NFFT={NFFT}, fs_out={fs_out:.0f} Sa/s (decim={DECIM})")
    plt.xlabel("Frequency, kHz (baseband)")
    plt.ylabel("Magnitude, dB (rel)")
    plt.grid(True)

def main():
    for p in IQ_FILES:
        analyze_file(p)
    plt.show()

    # ---------------- COMMENTED OUT (Demod/UI placeholders) ---------
    # # Future: DSC/AIS demodulation would take 'freq_hz' series and perform
    # # timing recovery + symbol decisions. Intentionally omitted here.
    # # A small parameter panel/GUI could be added if needed — disabled by COMBAT.
    # ---------------------------------------------------------------

if __name__ == "__main__":
    main()
    
