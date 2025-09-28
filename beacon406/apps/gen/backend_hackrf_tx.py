import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.logger import get_logger
log = get_logger(__name__)
import subprocess
import tempfile
from typing import Optional, Tuple, Union

import numpy as np

def _find_hackrf_transfer() -> Optional[str]:
    names = ["hackrf_transfer.exe", "hackrf_transfer"]
    search_paths = os.environ.get("PATH", "").split(os.pathsep)
    # Typical Windows paths
    search_paths += [
        r"C:\Program Files\PothosSDR\bin",
        r"C:\Program Files\HackRF\bin",
    ]
    for p in search_paths:
        for n in names:
            exe = os.path.join(p, n)
            if os.path.isfile(exe):
                return exe
    return None

def _peak_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))

def _iq_cf32_to_sc8(iq: np.ndarray, amp_scale: float = 0.95) -> bytes:
    if iq.dtype != np.complex64:
        iq = iq.astype(np.complex64, copy=False)
    peak = _peak_abs(iq)
    if peak > 1.0:
        iq = iq / peak
    iq = iq * amp_scale
    i8 = np.empty(iq.size * 2, dtype=np.int8)
    i = np.clip(np.round(iq.real * 127.0), -127, 127).astype(np.int8)
    q = np.clip(np.round(iq.imag * 127.0), -127, 127).astype(np.int8)
    i8[0::2] = i
    i8[1::2] = q
    return i8.tobytes()

def _digital_shift(iq: np.ndarray, shift_hz: float, Fs: float) -> np.ndarray:
    if abs(shift_hz) < 1e-6 or iq.size == 0:
        return iq
    n = np.arange(iq.size, dtype=np.float32)
    ph = np.exp(1j * (2.0 * np.pi * (shift_hz / Fs) * n)).astype(np.complex64)
    return (iq * ph).astype(np.complex64, copy=False)

def _resample_linear(iq: np.ndarray, Fs_in: int, Fs_out: int) -> np.ndarray:
    if Fs_in == Fs_out or iq.size == 0:
        return iq
    dur = iq.size / float(Fs_in)
    N_out = int(round(dur * Fs_out))
    if N_out <= 1:
        return iq.copy()
    x = np.arange(iq.size, dtype=np.float64)
    xi = np.linspace(0.0, iq.size - 1.0, N_out, dtype=np.float64)
    i = np.interp(xi, x, iq.real.astype(np.float64))
    q = np.interp(xi, x, iq.imag.astype(np.float64))
    return (i.astype(np.float32) + 1j * q.astype(np.float32)).astype(np.complex64)

def _zero_gap(duration_s: float, Fs: int) -> np.ndarray:
    N = int(round(max(0.0, duration_s) * Fs))
    return np.zeros(N, dtype=np.complex64)

def _calc_center_and_shift(target_signal_hz: float, if_offset_hz: float) -> Tuple[float, float]:
    center = float(target_signal_hz) + float(if_offset_hz)
    digital_shift = -float(if_offset_hz)
    return center, digital_shift

#def _apply_ppm(freq_hz: float, ppm: float) -> float:
#    return float(freq_hz) * (1.0 + float(ppm) / 1e6)

def _apply_freq_corr_hz(freq_hz: float, corr_hz: float) -> float:
    return float(freq_hz) + float(corr_hz)

def _run_hackrf_transfer(
    sc8_bytes: bytes,
    center_freq_hz: float,
    tx_sample_rate_sps: int,
    tx_gain_db: int = 0,
    hw_amp_enabled: bool = False,
    repeat: Union[int, str] = 1,
    gap_s: float = 0.0,
) -> None:
    exe = _find_hackrf_transfer()
    if not exe:
        raise RuntimeError("hackrf_transfer not found in PATH. Install PothosSDR/HackRF tools.")

    # 1) кадр: полезный сигнал + пауза-нули
    gap_n = int(round(max(0.0, gap_s) * tx_sample_rate_sps))
    frame_sc8 = sc8_bytes + (b"\x00" * (gap_n * 2))  # sc8: 2 int8 на сэмпл (I,Q)

    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "frame.sc8")

        def write_bytes(buf: bytes):
            with open(fname, "wb") as f:
                f.write(buf)

        if isinstance(repeat, str) and repeat.lower() == "loop":
            # 2A) бесконечный повтор: один кадр + -R (никаких рестартов процесса)
            write_bytes(frame_sc8)
            cmd = [
                exe, "-t", fname,
                "-f", str(int(round(center_freq_hz))),
                "-s", str(int(tx_sample_rate_sps)),
                "-x", str(int(tx_gain_db)),
            ]
            if hw_amp_enabled:
                cmd += ["-a", "1"]
            cmd += ["-R"]  # repeat indefinitely
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"hackrf_transfer failed: {proc.stderr.strip() or proc.stdout.strip()}")
            log.info(proc.stdout.strip(), proc.stderr.strip())

        else:
            # 2B) конечное число повторов: склеиваем N кадров и один запуск
            N = max(1, int(repeat))
            if N == 1:
                write_bytes(sc8_bytes)  # без паузы в конце — один прогон
            else:
                # [payload + gap] * (N-1) + payload
                big = frame_sc8 * (N - 1) + sc8_bytes
                write_bytes(big)

            cmd = [
                exe, "-t", fname,
                "-f", str(int(round(center_freq_hz))),
                "-s", str(int(tx_sample_rate_sps)),
                "-x", str(int(tx_gain_db)),
            ]
            if hw_amp_enabled:
                cmd += ["-a", "1"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"hackrf_transfer failed: {proc.stderr.strip() or proc.stdout.strip()}")
            log.info(proc.stdout.strip(), proc.stderr.strip())


from typing import Optional, Tuple, Union  # вверху файла должен быть Union

def hackrf_tx_from_array(
    iq_cf32: np.ndarray,
    target_signal_hz: float,
    if_offset_hz: float = -37_000,
    freq_correction_hz: float = 0.0,
    tx_sample_rate_sps: int = 2_000_000,
    repeat: Union[int, str] = 1,
    gap_s: float = 0.0,
    tx_gain_db: int = 0,
    hw_amp_enabled: bool = False,
    amp_scale: float = 0.95,
    transport: str = "hackrf_transfer",
    input_sample_rate_sps: Optional[int] = None,   # <<< NEW
) -> None:
    # --- ключевая правка: не читаем никаких атрибутов у numpy массива ---
    Fs_in = int(input_sample_rate_sps) if input_sample_rate_sps else tx_sample_rate_sps

    center, digital_shift_hz = _calc_center_and_shift(target_signal_hz, if_offset_hz)
    center_set = _apply_freq_corr_hz(center, freq_correction_hz)

    if Fs_in != tx_sample_rate_sps:
        iq_proc = _resample_linear(iq_cf32, Fs_in, tx_sample_rate_sps)
        Fs = tx_sample_rate_sps
    else:
        iq_proc = iq_cf32
        Fs = Fs_in

    iq_proc = _digital_shift(iq_proc, digital_shift_hz, Fs)

    sc8 = _iq_cf32_to_sc8(iq_proc, amp_scale=amp_scale)

    log.info(f"[TX] Target {target_signal_hz:.0f} Hz | IF {if_offset_hz:+.0f} Hz -> LO {center:.0f} Hz | f_set(hz) {center_set:.0f} Hz")
    log.info(f"[TX] Fs_in {Fs_in} -> Fs_tx {tx_sample_rate_sps} | repeat={repeat} gap_s={gap_s} | tx_gain={tx_gain_db} dB | PA={'on' if hw_amp_enabled else 'off'}")
    log.info(f"[TX] digital_shift_hz={digital_shift_hz:+.1f} | amp_scale={amp_scale} | sc8_bytes={len(sc8)}")

    if transport != "hackrf_transfer":
        raise NotImplementedError("Only 'hackrf_transfer' transport implemented in this version.")

    _run_hackrf_transfer(
        sc8, center_set, tx_sample_rate_sps, tx_gain_db=tx_gain_db,
        hw_amp_enabled=hw_amp_enabled, repeat=repeat, gap_s=gap_s
    )


def _read_cf32(path_cf32: str) -> np.ndarray:
    raw = np.fromfile(path_cf32, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError("cf32 file has odd number of floats (not interleaved pairs)")
    i = raw[0::2]
    q = raw[1::2]
    return (i.astype(np.float32) + 1j * q.astype(np.float32)).astype(np.complex64)

def hackrf_tx_from_file(
    path_cf32: str,
    target_signal_hz: float,
    if_offset_hz: float = -37_000,
    freq_correction_hz: float = 0.0,
    tx_sample_rate_sps: int = 2_000_000,
    repeat: Union[int, str] = 1,
    gap_s: float = 0.0,
    tx_gain_db: int = 0,
    hw_amp_enabled: bool = False,
    amp_scale: float = 0.95,
    transport: str = "hackrf_transfer",
    assume_input_fs_sps: Optional[int] = None,   # Fs файла (генератора)
) -> None:
    if not os.path.isfile(path_cf32):
        raise FileNotFoundError(f"cf32 not found: {path_cf32}")
    iq = _read_cf32(path_cf32)

    # НИКАКИХ setattr на ndarray!
    return hackrf_tx_from_array(
        iq,
        target_signal_hz=target_signal_hz,
        if_offset_hz=if_offset_hz,
        freq_correction_hz=freq_correction_hz,
        tx_sample_rate_sps=tx_sample_rate_sps,
        repeat=repeat,
        gap_s=gap_s,
        tx_gain_db=tx_gain_db,
        hw_amp_enabled=hw_amp_enabled,
        amp_scale=amp_scale,
        transport=transport,
        input_sample_rate_sps=assume_input_fs_sps,  # ← передаём Fs явным параметром
    )

