
# === locate project root (folder that contains "lib") and add to sys.path ===
import sys
from pathlib import Path

_here = Path(__file__).resolve()
_root = _here
for _ in range(10):  # поднимаемся максимум на 10 уровней
    if (_root / "lib").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
    if _root.parent == _root:
        break
    _root = _root.parent
else:
    raise RuntimeError("Не найден корень проекта с папкой 'lib'. Перемести скрипт в дерево проекта или выставь PYTHONPATH.")

ROOT = _root  # если ниже в коде используется ROOT
# === end path setup ===



from lib.logger import get_logger
log = get_logger(__name__)
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from pathlib import Path
plt.ion()


# === path hack (fixed to project root) ===
ROOT = Path(__file__).resolve().parents[1]  # beacon406/
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
# ===================================

from lib.metrics import process_psk_impulse
from lib.demod import phase_demod_psk_msg_safe

# ==========================
# ПАРАМЕТРЫ
# ==========================
# Эти параметры должны совпадать с теми, что использовались при захвате данных
SAMPLE_RATE_SPS  = 1_000_000 
IF_OFFSET_HZ     = 0 #-25_000
FM_BASELINE_MS   = 2.0        
FM_YLIMIT_KHZ    = 15.0 
PSK_YLIMIT_RAD   = 1.5
PSK_BASELINE_MS  = 10.0

# --- 1) Поиск импульса по RMS(dBm) ---
THRESH_DBM   = -60.0      # подстрой под свой уровень
#THRESH_DBM   = -45.0
WIN_MS       = 1          # окно RMS  ??? для PSK можно 1ms но может быть обрезан последний бит 
GUARD_MS     = 0.0        # добавление поля слева/справа к окну импульса
START_DELAY_MS = 3.0      # обрезание начало сигнала ??? если начальная фаза больше или меньше 1.1 то убивает поиск фронта ???
CALIB_DB     = -30.0      # если хочешь, учти калибровку тракта




# === path hack (fixed to project root) ===
def find_root(project_name: str) -> Path:
    """Ищет корень проекта по имени папки."""
    root = Path(__file__).resolve()
    while root.name != project_name:
        if root.parent == root:  # дошли до корня диска
            raise RuntimeError(f"Папка {project_name} не найдена в пути")
        root = root.parent
    return root

# === настройки ===
ROOT = find_root("TesterSDR")

file_out_iq = r"psk_out_f50.f32"
#TEST 
FILE_IQ_1 = r"psk406msg_f150.cf32"  
FILE_IQ_2 = r"psk406msg_f100.cf32"
FILE_IQ_3 = r"psk406msg_f75.cf32"
FILE_IQ_4 = r"psk406msg_f50.cf32"
FILE_IQ_5 = r"psk406.cf32"
FILE_IQ_6 = r"psk406shot.cf32"

#SDR RTL 
FILE_IQ_7 = r"iq_pulse_406-1m-25k.cf32"
FILE_IQ_8 = r"iq_pulse_406-1m-37k.cf32"
FILE_IQ_9 = r"iq_pulse_406_1024.cf32"

FILE_IQ_10 = r"iq_121.cf32"

FILE_IQ_11 = r"iq_pulse_AIS_m5.cf32"
FILE_IQ_12 = r"iq_pulse_AIS_m1.cf32"
FILE_IQ_13 = r"iq_pulse_AIS_sart.cf32"
FILE_IQ_14 = r"iq_test.cf32"  # -48dbm 
FILE_IQ_15 = r"iq_pulse_DSC_dis.cf32"
FILE_IQ_16 = r"iq_pulse_406-1m-25k.cf32"

#RSA
FILE_IQ_17 = r"rsa406_pulse_20250911_133404_dc.cf32"
FILE_IQ_18 = r"rsa406_pulse_20250911_133526_dc.cf32"

#UI_gen
FILE_IQ_19 = r"UI_iq_1m.cf32"

# формируем шаблон пути (сразу строка!)
FILE_PATH = str(ROOT / "captures" / FILE_IQ_19)
file_out = str(ROOT / "captures" / file_out_iq)


# ==========================
# ОСНОВНАЯ ЛОГИКА
# ==========================

def save_array_bin(arr, filename=file_out, count=None):
    #Сохраняет массив arr в (float), с возможностью ограничения количества значений.
    arr = np.asarray(arr, dtype=np.float32)
    if count is not None:
        arr = arr[:count]
    try:
        arr.tofile(filename)
        print(f"BIN сохранён: {filename} ({arr.size} значений, float32)")
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")

def fileload(file_path):
    #Загружает IQ (CF32)
    if not os.path.exists(file_path):
        print(f"ОШИБКА: Файл не найден: {file_path}")
        return

    print(f"Загрузка IQ-данных из файла: {file_path}")
    try:
        iq_data = np.fromfile(file_path, dtype=np.complex64)
        print(f"Загружено {iq_data.size} сэмплов.")
    except Exception as e:
        print(f"ОШИБКА при загрузке файла: {e}")
        return
    return iq_data

# ---------- helpers: RMS(dBm) и поиск импульса ----------
def _rms_dbm_trace(iq: np.ndarray, fs: float, win_ms: float, calib_db: float = 0.0):
    W = max(1, int(win_ms * 1e-3 * fs))
    # скользящее среднее по |IQ|^2
    p = np.abs(iq)**2
    ma = np.convolve(p, np.ones(W)/W, mode="same")
    rms = np.sqrt(ma + 1e-30)
    dbm = 20*np.log10(rms + 1e-30) + calib_db
    return dbm

def _find_impulse_bounds(rms_dbm: np.ndarray, fs: float, thresh_dbm: float,
                         guard_ms: float = 2.0):
    idx = np.where(rms_dbm > thresh_dbm)[0]
    if idx.size == 0:
        return None, None
    g = int(max(0, round(guard_ms * 1e-3 * fs)))
    i0, i1 = idx[0], idx[-1]
    return max(0, i0 - g), min(rms_dbm.size, i1 + g)

#Если длительность меньше 5 мс, то такой импульс игнорируется, и поиск продолжается дальше до следующего кандидата.
def seg_rms_shot(iq_data):
    rms_dbm = _rms_dbm_trace(iq_data, SAMPLE_RATE_SPS, WIN_MS, calib_db=CALIB_DB)

    # искать импульсы пока не найдём достаточно длинный
    while True:
        i0, i1 = _find_impulse_bounds(rms_dbm, SAMPLE_RATE_SPS, THRESH_DBM, guard_ms=GUARD_MS)
        if i0 is None or i1 is None:
            print("Импульс по порогу RMS не найден.")
            return None

        # переводим длину в мс
        dur_ms = (i1 - i0) / SAMPLE_RATE_SPS * 1e3
        if dur_ms < 5.0:
            # слишком короткий, ищем дальше (обнулим участок и повторим)
            rms_dbm[i0:i1] = -999  # «затираем», чтобы не мешал
            continue
        else:
            break

    # сдвиг границ на размер окна RMS
    #shift = int(WIN_MS * 1e-3 * SAMPLE_RATE_SPS)
    #i0 = min(i1, i0 + shift)
    #i1 = max(i0, i1 - shift)

    i0 = min(i1, i0 + int((START_DELAY_MS + WIN_MS) * 1e-3 * SAMPLE_RATE_SPS))
    i1 = max(i0, i1 - int(WIN_MS * 1e-3 * SAMPLE_RATE_SPS))


    N = i1 - i0
    if N <= 0:
        print("Пустой сегмент.")
        return None

    iq_seg = iq_data[i0:i1]
    print(f"Окно импульса: [{i0}:{i1}] сэмплов (~{dur_ms:.2f} мс)")

    return iq_seg


def seg_rms(iq_data):
    
    rms_dbm = _rms_dbm_trace(iq_data, SAMPLE_RATE_SPS, WIN_MS, calib_db=CALIB_DB)
    i0, i1  = _find_impulse_bounds(rms_dbm, SAMPLE_RATE_SPS, THRESH_DBM, guard_ms=GUARD_MS)
    if i0 is None:
        print("Импульс по порогу RMS не найден.")
        return

    # после i0, i1 = _find_impulse_bounds(...)
    #shift = int(START_DELAY_MS + WIN_MS * 1e-3 * SAMPLE_RATE_SPS)

    i0 = min(i1, i0 + int(START_DELAY_MS + WIN_MS * 1e-3 * SAMPLE_RATE_SPS))
    i1 = max(i0, i1 - int(WIN_MS * 1e-3 * SAMPLE_RATE_SPS))

    N = i1 - i0
    if N <= 0:
        print("Пустой сегмент.")
        return

    iq_seg = iq_data[i0:i1]

    print(f"Окно импульса: [{i0}:{i1}] сэмплов (~{(i1-i0)/SAMPLE_RATE_SPS*1000:.1f} мс)")

    return iq_seg

def plot_rms_trace(iq_data, fs=SAMPLE_RATE_SPS, win_ms=WIN_MS,
                   calib_db=CALIB_DB, thresh_dbm=THRESH_DBM):
    """
    Строит график уровня RMS(dBm) для всего буфера IQ.
    """
    rms_dbm = _rms_dbm_trace(iq_data, fs, win_ms, calib_db)

    xs_ms = np.arange(len(rms_dbm)) / fs * 1000.0  # ось времени в мс

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(xs_ms, rms_dbm, lw=1.0, color="darkred", label="RMS(dBm)")

    # линия порога
    ax.axhline(thresh_dbm, color="royalblue", ls="--", lw=1.2,
               label=f"Порог {thresh_dbm:.1f} dBm")

    ax.set_title("Трасса RMS (скользящее окно)", fontsize=14)
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("Уровень, dBm")
    ax.grid(True, alpha=0.5)
    ax.legend()

    fig.suptitle(f"Fs: {fs/1000:.1f} kSPS, окно RMS={win_ms:.2f} мс",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.94])
    #plt.show()
    plt.show(block=False)
    plt.pause(0.1)  # дать GUI дорисоваться
    

# ==========================
# FFT СЕКТОР ИЗ CF32
# ==========================

def run_self_test_FFT_mask(
    iq_data,
    FFT_N=65536,
    avg=4,
    window="hann",               # "hann" | "rect"
    sector_center_hz=0.0,
    sector_half_width_hz=50_000,
    y_ref_db=0.0,
    *,
    gate_samples=None,           # (i0, i1): взять только этот диапазон сэмплов (до FFT/усреднения)
    remove_dc=True,              # вычесть комплексное среднее перед FFT
    normalize="dBc",             # "dBc" (пик=0) | "dB" (абс. значения)
    y_lim=None,                  # например (-90, 5) для dBc
    show_mask=True               # рисовать лестницу C/S T.001 в dBc
):
    """
    Рисует сектор амплитудного спектра из CF32 «как есть», БЕЗ учёта IF_OFFSET_HZ.
    Ось частот — сырая baseband (-Fs/2..+Fs/2).

    Параметры:
      gate_samples:   кортеж (i0, i1) – «окно анализа» по сэмплам, чтобы не включать чистую несущую до PSK.
      remove_dc:      вычитает среднее (компенсация DC/IQ-несимметрии).
      normalize:      "dBc": график нормируется к максимуму в секторе (пик = 0 dBc).
                      "dB" : абсолютная шкала (учитывает y_ref_db).
      y_lim:          пределы по Y (tuple или None).
      show_mask:      рисовать маску C/S T.001 (в dBc относительно пика).

    Возвращает dict:
      f_peak_Hz, p_peak_dB (абсолютно), p_peak_dBc (=0 при normalize="dBc"),
      n_blocks, FFT_N, df_Hz
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- частота дискретизации
    try:
        SR = float(SAMPLE_RATE_SPS)
    except NameError:
        raise NameError("SAMPLE_RATE_SPS is not defined; set global sample rate (SPS).")

    # --- вход как complex64
    seg = np.asarray(iq_data, dtype=np.complex64)

    # --- гейтинг по времени (до усреднения и FFT)
    if gate_samples is not None:
        i0, i1 = map(int, gate_samples)
        i0 = max(0, i0)
        i1 = min(len(seg), i1)
        seg = seg[i0:i1]

    if seg.size < 16:
        print("Сегмент слишком короткий для FFT.")
        return

    if remove_dc:
        seg = seg - np.mean(seg)

    # --- окно (float32)
    win = (np.hanning(FFT_N) if window == "hann" else np.ones(FFT_N)).astype(np.float32)

    # --- блоки для усреднения
    n_possible = seg.size // FFT_N
    if n_possible == 0:
        seg = np.pad(seg, (0, FFT_N - seg.size))
        n_possible = 1
    n_blocks = int(max(1, min(int(avg), n_possible)))

    # --- FFT и накопление мощности
    psd_acc = None
    w2 = float(np.sum(win ** 2))
    for k in range(n_blocks):
        chunk = seg[k*FFT_N:(k+1)*FFT_N]
        if chunk.size < FFT_N:
            chunk = np.pad(chunk, (0, FFT_N - chunk.size))
        X = np.fft.fftshift(np.fft.fft(chunk * win, n=FFT_N))
        mag2 = (np.abs(X) ** 2) / w2
        psd_acc = mag2 if psd_acc is None else (psd_acc + mag2)

    psd = psd_acc / n_blocks
    psd_db = 10.0 * np.log10(psd + 1e-30) + y_ref_db

    # --- ось частот
    f_axis = np.fft.fftshift(np.fft.fftfreq(FFT_N, d=1.0 / SR))

    # --- сектор
    f0 = float(sector_center_hz)
    hw = abs(float(sector_half_width_hz))
    mask = (f_axis >= f0 - hw) & (f_axis <= f0 + hw)
    if not np.any(mask):
        print("Сектор вне диапазона оси частот. Проверь sector_center_hz/sector_half_width_hz.")
        return

    f_sec = f_axis[mask]
    psd_sec_db = psd_db[mask]

    # --- пик (абсолютно, до нормировки)
    i_pk = int(np.argmax(psd_sec_db))
    f_pk = float(f_sec[i_pk])
    p_pk_abs_db = float(psd_sec_db[i_pk])

    # --- нормировка
    if str(normalize).lower() == "dbc":
        psd_plot = psd_sec_db - p_pk_abs_db
        ylabel = "Амплитуда, dBc"
        p_pk_dBc = 0.0
        mask_offset = 0.0
    else:
        psd_plot = psd_sec_db
        ylabel = "Амплитуда, dB"
        p_pk_dBc = float(p_pk_abs_db - p_pk_abs_db)  # 0.0, для совместимости
        mask_offset = p_pk_abs_db  # маска будет «прилипать» к максимуму

    # --- график
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f_sec / 1e3, psd_plot, lw=1.5, label="Спектр")
    ax.axvline(0.0, linestyle='--', alpha=0.5)

    # --- маска C/S T.001 (в dBc относительно несущей)
    def _plot_cs_t001_mask(ax, p0, half_width_hz):
        # сегменты справа от нуля (влево рисуем зеркально)
        segments = [
            (0.0,   3e3,   0.0),   # 0…3 кГц: 0→-20 dBc (верхняя ступень у несущей)
            (3e3,   7e3, -20.0),   # 3…7 кГц: -20 dBc
            (7e3,  12e3, -30.0),   # 7…12 кГц: -30 dBc
            (12e3, 24e3, -35.0),   # 12…24 кГц: -35 dBc
            (24e3, half_width_hz, -40.0),  # дальше -40 dBc
        ]
        for sgn in (-1.0, +1.0):
            for f1, f2, lvl in segments:
                x1 = sgn * (f1 / 1e3)
                x2 = sgn * (f2 / 1e3)
                y  = p0 + lvl
                ax.plot([x1, x2], [y, y], linestyle='--', linewidth=1.2, alpha=0.9)
        for fb in [3e3, 7e3, 12e3, 24e3]:
            ax.axvline(+fb/1e3, linestyle=':', alpha=0.6)
            ax.axvline(-fb/1e3, linestyle=':', alpha=0.6)

    if show_mask:
        _plot_cs_t001_mask(ax, mask_offset, hw)

    # помечаем найденный пик
    ax.plot([f_pk / 1e3], [0.0 if normalize.lower() == "dbc" else p_pk_abs_db], 'o', label="Пик")

    ax.set_title("FFT сектор (как есть, без IF-сдвигов)")
    ax.set_xlabel("Частота, кГц (baseband)")
    ax.set_ylabel(ylabel)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best")

    fig.suptitle(
        f"Fs: {SR/1000:.1f} kSPS, сектор центр={sector_center_hz/1000:.3f} кГц, ±{sector_half_width_hz/1000:.1f} кГц",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    #plt.show()
    plt.show(block=False)
    plt.pause(0.1)  # дать GUI дорисоваться
    

    return {
        "f_peak_Hz": f_pk,
        "p_peak_dB": p_pk_abs_db,             # абсолютный пик (dB)
        "p_peak_dBc": p_pk_dBc,               # при dBc всегда 0.0
        "n_blocks": int(n_blocks),
        "FFT_N": int(FFT_N),
        "df_Hz": float(SR / FFT_N),
        "normalize": "dBc" if str(normalize).lower() == "dbc" else "dB",
        "gate_samples": tuple(gate_samples) if gate_samples is not None else None,
    }


def run_self_test_psk(iq_data):
    
    t0_offset_ms = 0.0
    t0 = time.perf_counter()
    result = process_psk_impulse(
        iq_seg=iq_data,
        fs=SAMPLE_RATE_SPS,
        baseline_ms=PSK_BASELINE_MS,
        t0_offset_ms=t0_offset_ms, # проверить и включать поиск фронта в 50мс до конца несущей !!! 
        use_lpf_decim=True,
        remove_slope=True,
    )
    dt = time.perf_counter() - t0
    print(f"process_psk_impulse() заняла {dt*1000:.3f} мс")
    
    xs_ms     = result.get("xs_ms")
    phase_rad = result.get("phase_rad")
    title     = result.get("title")

    if xs_ms.size == 0:
        print("Данных для построения графика недостаточно.")
        return

    # запись psk в файл для отладки демодулятора или расчета фазы
    #save_array_bin(phase_rad, file_out)

    #демодуляция и вычисления фазы 
    t0 = time.perf_counter()
    
    msg_hex, stats, edges = phase_demod_psk_msg_safe(data=phase_rad)
    dt = time.perf_counter() - t0
    print(f"phase_demod_psk_msg_safe() заняла {dt*1000:.3f} мс")
    
    FSd = SAMPLE_RATE_SPS/4
    # Guards for STRICT_COMPAT: avoid crashes if edges empty or metrics are NaN
    import numpy as _np
    carrier_ms = ((edges[0]/FSd*1e3) if (edges is not None and len(edges) > 0) else _np.nan)
    _pos = stats.get('PosPhase', _np.nan)
    _neg = stats.get('NegPhase', _np.nan)
    _rise_us = (stats.get('PhRise', _np.nan) / FSd * 1e6) if _np.isfinite(stats.get('PhRise', _np.nan)) else _np.nan
    _fall_us = (stats.get('PhFall', _np.nan) / FSd * 1e6) if _np.isfinite(stats.get('PhFall', _np.nan)) else _np.nan
    _ass = stats.get('Ass', _np.nan)
    _tmod = stats.get('Tmod', _np.nan)
    _fmod_hz = (FSd / _tmod) if _np.isfinite(_tmod) and (_tmod != 0) else _np.nan
    
    # Keep the same variable name 'phase_res' as before (string for title)
    phase_res = (
        f"Carrier={carrier_ms:.3f}ms "
        f"Pos={_pos:.2f}rad "
        f"Neg={_neg:.2f}rad "
        f"Rise={_rise_us:.1f}us "
        f"Fall={_fall_us:.1f}us "
        f"Ass={_ass:.3f}%"
        f"Fmod={_fmod_hz:.3f}Hz"
    )
    title += f'\n {phase_res} \n HEX={msg_hex}'
    
    # --- Построение графика с метками ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(xs_ms, phase_rad, lw=1.5, color='royalblue', label="φ(t)")

    ax.set_title(f"Самотест: {title}", fontsize=14)
    ax.set_xlabel("Время, мс", fontsize=12)
    ax.set_ylabel("Фаза, радиан", fontsize=12)
    ax.set_ylim(-PSK_YLIMIT_RAD, PSK_YLIMIT_RAD)
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    #plt.show()
    plt.show(block=False)
    plt.pause(0.1)  # дать GUI дорисоваться


if __name__ == "__main__":

    iq_data = fileload(file_path=FILE_PATH)

    # 1) RMS-трасса
    plot_rms_trace(iq_data)

    # 2) Поиск импульса и анализ
    iq_seg = seg_rms_shot(iq_data)
    run_self_test_psk(iq_data=iq_seg)
    
    # 3) FFT
    #run_self_test_FFT_mask(iq_data=iq_data, sector_center_hz=0.0, sector_half_width_hz=50_000)
    t0 = time.perf_counter()
    run_self_test_FFT_mask(iq_data=iq_seg, sector_center_hz=0.0, sector_half_width_hz=50_000)
    dt = time.perf_counter() - t0
    print(f"run_self_test_FFT_mask() заняла {dt*1000:.3f} мс")
    
    """ 
    Fs = 1_000_000          # Гц (1 Мc/с)
    cut_us = 100            # сколько обрезать, мкс
    n_cut = int(round(Fs * cut_us / 1e6))   # = 100
    iq_trimmed = iq_seg[n_cut:] if iq_seg.size > n_cut else np.empty(0, dtype=iq_data.dtype)
    run_self_test_FFT_mask(iq_data=iq_trimmed, sector_center_hz=0.0, sector_half_width_hz=50_000)
    """
    # --- финальный «холд», чтобы все окна остались на экране ---
    plt.ioff()   # выключаем интерактив перед финальным блокирующим показом
    plt.show()   # одно блокирующее show удержит все уже открытые окна


    