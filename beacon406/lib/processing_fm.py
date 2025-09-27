from __future__ import annotations
"""
processing_FM.py — универсальный частотный дискриминатор для IQ-последовательностей.

Цели:
- Дать мгновенную частоту (freq_hz[t]) из комплексного сигнала IQ.
- Опционально выполнить предфильтрацию (LPF) и децимацию для снижения шума/нагрузки.
- Предусмотреть постсглаживание, удаление DC/наклона (detrend), центрирование.

Комбат/STRICT_COMPAT принципы:
- Без внешних зависимостей (numpy only).
- Чистые функции; ничего не меняют «снаружи».
- Аддитивно: модуль можно подключить, не меняя существующие файлы.

Рекомендуемое подключение:
    from processing_FM import fm_discriminator
    out = fm_discriminator(iq, fs=backend.actual_sample_rate_sps,
                           pre_lpf_hz=50_000, decim=4, smooth_hz=2_000,
                           detrend=True, center=True)

    freq = out["freq_hz"]; fs_out = out["fs_out"]; t_ms = out["xs_ms"]

Автор: ChatGPT (GPT-5 Thinking)
"""
from lib.logger import get_logger
log = get_logger(__name__)
from dataclasses import dataclass
import numpy as np

# ==========================
# ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ
# ==========================
@dataclass
class FMOptions:
    pre_lpf_hz: float | None = None   # антиалиасный LPF перед децимацией (Гц)
    decim: int = 1                    # коэффициент децимации ≥1
    smooth_hz: float | None = None    # сглаживание частоты простым MA (Гц ширина среза)
    detrend: bool = True              # удалить линейный тренд (наклон) частоты
    center: bool = True               # убрать DC-смещение частоты (центрировать вокруг 0)
    # Параметры FIR (окно)
    fir_taps: int = 127               # длина FIR-фильтра (нечётная)
    fir_beta: float = 0.0             # для окна Кайзера (не используется, но оставлено на будущее)

# ==========================
# НИЗКОУРОВНЕВЫЕ ХЕЛПЕРЫ DSP
# ==========================

def _ensure_1d_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    if not np.iscomplexobj(x):
        raise ValueError("iq must be complex1d array (dtype complex)")
    return x


def dc_block(x: np.ndarray) -> np.ndarray:
    """Удалить DC-компонент (среднее)."""
    return x - np.mean(x)


def _design_fir_lpf(fs: float, cutoff_hz: float, taps: int) -> np.ndarray:
    """Простой FIR LPF (окно Хэмминга), без SciPy.
    cutoff_hz — частота среза (Гц), taps — нечётное число.
    Возвращает коэффициенты фильтра h длины taps.
    """
    if taps % 2 == 0:
        taps += 1
    fc = float(cutoff_hz) / float(fs)  # нормированная (0..0.5)
    if not (0.0 < fc < 0.5):
        raise ValueError("cutoff must be within (0, fs/2)")
    n = np.arange(taps)
    m = n - (taps - 1) / 2.0
    # идеальный sinc LPF
    h = np.sinc(2 * fc * m)
    # окно Хэмминга
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / (taps - 1))
    h *= w
    # нормировка по сумме амплитуд
    h /= np.sum(h)
    return h.astype(np.float64)


def _filt_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Свертка FIR (same-выход по длине)."""
    # mode='same' даёт ту же длину, что у x
    return np.convolve(x, h, mode='same')


def _decimate(x: np.ndarray, q: int) -> np.ndarray:
    if q <= 1:
        return x
    return x[::q]


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    # Сверточное усреднение без фазового сдвига (mode='same')
    w = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, w, mode='same')


def _linear_detrend(x: np.ndarray) -> np.ndarray:
    """Удалить линейный тренд: x <- x - (a*t + b)."""
    n = x.size
    if n < 2:
        return x
    t = np.arange(n, dtype=np.float64)
    # Решим наименьшие квадраты для y = a*t + b
    # [t 1] * [a b]^T ≈ x
    # Используем формулы уменьшенной размерности
    t_mean = t.mean()
    x_mean = x.mean()
    tt = np.sum((t - t_mean) ** 2)
    if tt == 0:
        return x - x_mean
    a = np.sum((t - t_mean) * (x - x_mean)) / tt
    b = x_mean - a * t_mean
    return x - (a * t + b)

# ==========================
# ЧАСТОТНЫЙ ДИСКРИМИНАТОР
# ==========================

def instantaneous_freq_basic(iq: np.ndarray, fs: float) -> np.ndarray:
    """Мгновенная частота по методу dphi = angle(z[n]*conj(z[n-1])).

    Возвращает freq_hz той же длины, что iq (freq[0] = freq[1] для удобства).
    """
    z = _ensure_1d_complex(iq)
    if z.size < 2:
        return np.zeros_like(z, dtype=np.float64)
    # dphi[n] = angle(z[n] * conj(z[n-1]))
    dphi = np.angle(z[1:] * np.conj(z[:-1]))
    # перевод в Гц: f = dphi * fs / (2*pi)
    freq = (dphi * float(fs)) / (2.0 * np.pi)
    # выровняем длину (пусть freq[0] = freq[1])
    freq = np.concatenate(([freq[0]], freq))
    return freq


def fm_discriminator(
    iq: np.ndarray,
    fs: float,
    pre_lpf_hz: float | None = None,
    decim: int = 1,
    smooth_hz: float | None = None,
    detrend: bool = True,
    center: bool = True,
    fir_taps: int = 127,
) -> dict:
    """Главная функция дискриминатора.

    Параметры
    ---------
    iq : complex1d np.ndarray
        Вырезанный по гейту участок IQ.
    fs : float
        Исходная частота дискретизации (Sa/s). Возьми из backend.actual_sample_rate_sps.
    pre_lpf_hz : float | None
        Если задан — выполним FIR-LPF перед децимацией (антиалиасный).
    decim : int
        Коэффициент децимации (>=1). Если >1 — после LPF уменьшаем частоту дискретизации.
    smooth_hz : float | None
        Постсглаживание частоты скользящим средним; задай желаемую «полосу» сглаживания (Гц).
    detrend : bool
        Удалять ли линейный наклон (полезно при CFO/дрифте и для DSC HF).
    center : bool
        Убирать ли DC-смещение частоты (центрировать вокруг 0).
    fir_taps : int
        Длина FIR-фильтра для предфильтрации (нечётная).

    Возвращает
    ----------
    dict с ключами:
      - 'freq_hz': np.ndarray (float64), мгновенная частота, Гц
      - 'fs_out' : float, результирующая частота дискретизации после децимации
      - 'xs_ms'  : np.ndarray, ось времени (мс)
      - 'title'  : str, краткое описание параметров (для UI/графика)
    """
    z = _ensure_1d_complex(iq)
    if z.size == 0:
        return {
            "freq_hz": np.zeros(0, dtype=np.float64),
            "fs_out": float(fs) / max(int(decim), 1),
            "xs_ms": np.zeros(0, dtype=np.float64),
            "title": "FM: empty input",
        }

    # 1) Уберём DC (по желанию; для IQ meestal не критично, но стабилизирует фильтрацию)
    z = dc_block(z)

    # 2) Предфильтрация + децимация (если задано)
    fs_eff = float(fs)
    if pre_lpf_hz is not None and decim >= 1:
        # Спроектируем LPF и применим отдельно к I и Q
        h = _design_fir_lpf(fs_eff, cutoff_hz=float(pre_lpf_hz), taps=int(fir_taps))
        zi = _filt_fir(z.real.astype(np.float64), h)
        zq = _filt_fir(z.imag.astype(np.float64), h)
        z = zi + 1j * zq
    if decim > 1:
        z = _decimate(z, int(decim))
        fs_eff = fs_eff / float(int(decim))

    # 3) Частотная дискриминация
    freq = instantaneous_freq_basic(z, fs=fs_eff)

    # 4) Постобработка: сглаживание, detrend, центрирование
    if smooth_hz is not None and smooth_hz > 0:
        # Оценим окно для MA по желаемой полосе сглаживания
        # Примерная эвристика: окно ≈ fs_out / smooth_hz
        win = int(max(1, round(fs_eff / float(smooth_hz))))
        # Сделаем окном нечётной длины для симметрии
        if win % 2 == 0:
            win += 1
        freq = _moving_average(freq, win)

    if detrend:
        freq = _linear_detrend(freq)

    if center:
        freq = freq - np.median(freq)

    # 5) Ось времени (мс)
    n = freq.size
    xs_ms = (np.arange(n, dtype=np.float64) / fs_eff) * 1e3

    title = (
        f"FM discr | fs_in={fs:.0f} Sa/s | pre_lpf={pre_lpf_hz or 0:.0f} Hz | "
        f"decim={decim} -> fs_out={fs_eff:.0f} | smooth={smooth_hz or 0:.0f} Hz | "
        f"detrend={'y' if detrend else 'n'} | center={'y' if center else 'n'}"
    )

    return {
        "freq_hz": freq.astype(np.float64),
        "fs_out": fs_eff,
        "xs_ms": xs_ms,
        "title": title,
    }

# ==========================
# ТЕСТЫ-«КУРИТЕЛЬНЫЕ» (локально)
# ==========================
if __name__ == "__main__":
    # Простой локальный self-check на синусе с частотой f0 и CFO
    fs = 1_000_000.0
    dur = 0.02  # 20 мс
    t = np.arange(int(fs * dur)) / fs
    f0 = 10_000.0
    cfo = 1_200.0
    phase = 2*np.pi*(f0 + cfo) * t
    iq = np.exp(1j*phase)

    out = fm_discriminator(
        iq, fs,
        pre_lpf_hz=50_000, decim=4,
        smooth_hz=2_000,
        detrend=True, center=True,
    )
    log.info(out["title"])
    log.info("fs_out: %s", out["fs_out"]) 
    log.info("freq stats (Hz): mean=%.1f, std=%.1f", float(np.mean(out["freq_hz"])), float(np.std(out["freq_hz"])))
