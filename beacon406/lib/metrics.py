# processing_PSK.py — LPF/decim for PSK 406, toggleable
# этот алгоритм не оптимизирован и у него есть выравнивание по фазе 
# не понятно насколько это плохо для анализа CS 406PSK...
# нужно будет оптимизировать 406psk алгоритм 
# например с помощью FM точно вычислить начало передачи 
# несущей и начало модуляции 
# от начало модуляции отступить 50мс и это считать на 0 фазы
# 
#

from typing import Dict, Optional
import numpy as np

# ---------- LPF/Decim settings (toggle here) ----------
LPF_ENABLE: bool      = True        # True -> включить НЧ-фильтр перед детектором фазы
LPF_CUTOFF_HZ: float  = 12_000.0    # частота среза (~12–15 кГц под AIS GMSK 9.6 ksps)
LPF_TAPS: int         = 129         # нечётное число тапов FIR (101..201 ок)
DECIM: int            = 4           # 1 = без прореживания; 2..8 по желанию 

# ------------------------------------------------------

def _design_lowpass(fs: float, fc_hz: float, taps: int) -> np.ndarray:
    """
    Простая оконная FIR-аппроксимация: sinc * Hamming.
    Возвращает вещественные коэффициенты НЧ-фильтра.
    """
    if taps < 5:
        taps = 5
    if taps % 2 == 0:
        taps += 1  # делаем нечётным

    # нормализованная частота (0..0.5)
    fn = float(fc_hz) / (fs * 0.5)
    fn = max(1e-6, min(0.999999, fn))

    n = np.arange(taps, dtype=np.float64)
    m = n - (taps - 1) / 2.0
    # sinc lowpass (идеальный), нормировка на частоту среза
    h = np.sinc(fn * m)
    # Hamming window
    w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (taps - 1))
    h *= w
    # gain до 1.0 (DC-усиление = 1)
    h /= np.sum(h)
    return h.astype(np.float64)

def _maybe_lpf_and_decim(iq: np.ndarray, fs: int) -> tuple[np.ndarray, int]:
    """
    При LPF_ENABLE=True: фильтрует IQ и, если DECIM>1, прореживает.
    Возвращает (iq_out, fs_out).
    """
    if not LPF_ENABLE:
        return iq, fs

    # FIR-LPF
    h = _design_lowpass(fs=float(fs), fc_hz=LPF_CUTOFF_HZ, taps=LPF_TAPS)
    # свёртка по комплексным данным (раздельно по Re/Im)
    re = np.convolve(iq.real.astype(np.float64, copy=False), h, mode="same")
    im = np.convolve(iq.imag.astype(np.float64, copy=False), h, mode="same")
    y = (re + 1j * im).astype(np.complex64, copy=False)

    if DECIM > 1:
      # простое прореживание после антиалиасного LPF
      y = y[::int(DECIM)]
      fs_out = int(round(fs / float(DECIM)))
      fs_out = max(1, fs_out)
    else:
      fs_out = fs

    return y, fs_out

def _remove_linear_trend(y: np.ndarray) -> np.ndarray:
    """
    Убирает линейный тренд (на случай частотного расстроя) из массива y.
    Возвращает detrended сигнал той же длины.
    """
    n = y.size
    if n < 4:
        return y
    x = np.arange(n, dtype=np.float64)
    # МНК по прямой y = a*x + b
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x * x)
    sxy = np.sum(x * y)
    denom = n * sxx - sx * sx
    if denom == 0:
        return y
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return y - (a * x + b)

def process_psk_impulse(
    iq_seg: np.ndarray,
    fs: int,
    baseline_ms: float = 2.0, 
    t0_offset_ms: float = 0.0,
    use_lpf_decim: bool = True,
    remove_slope: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Рабочая схема:
      (A) Гейт по уровню на исходном fs → вырезаем только импульс
      (B) LPF/DECIM (если включены)
      (C) unwrap(angle) — непрерывная фаза
      (D) Ноль фазы = среднее первых baseline_ms на срезе
      (E) (опц.) удаление линейного наклона
    """
    if iq_seg is None or iq_seg.size < 8:
        return {
            "xs_ms":     np.array([], dtype=np.float64),
            "phase_rad": np.array([], dtype=np.float64),
            "title":     "PSK фаза (данных недостаточно)",
        }

    # (A) найдём срез импульса на исходном fs

    iq_slice = iq_seg
    # (B) LPF/DECIM — как у тебя было
    if use_lpf_decim:
        iq_proc, fs_eff = _maybe_lpf_and_decim(iq_slice, fs)
    else:
        iq_proc, fs_eff = iq_slice, fs

    # (C) непрерывная фаза
    phi = np.unwrap(np.angle(iq_proc).astype(np.float64, copy=False))

    # (D) ноль фазы = среднее первых baseline_ms
    base_pts = max(4, int(round(baseline_ms * 1e-3 * fs_eff)))
    base_pts = min(base_pts, phi.size)
    phi0 = float(np.mean(phi[:base_pts])) if base_pts > 0 else float(np.mean(phi))
    phase_rad = phi - phi0

    # (E) (опц.) убрать линейный тренд (частотный оффсет)
    if remove_slope:
        phase_rad = _remove_linear_trend(phase_rad)

    # Ось времени только для среза (как в «правильной» версии)
    xs_ms = 1e3 * (np.arange(phase_rad.size, dtype=np.float64) / float(fs_eff)) + float(t0_offset_ms)

    # Заголовок
    lpf_tag = ""
    if use_lpf_decim and LPF_ENABLE:
        lpf_tag = f" | LPF {LPF_CUTOFF_HZ/1e3:.1f} кГц, taps {LPF_TAPS}, decim×{DECIM}"
    slope_tag = " | detrend" if remove_slope else ""
    title = f"PSK:(ноль=среднее {baseline_ms:.3f} мс){lpf_tag}{slope_tag}"

    return {"xs_ms": xs_ms, "phase_rad": phase_rad, "title": title}
