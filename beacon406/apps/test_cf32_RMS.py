import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
plt.ion()


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
THRESH_DBM   = -45.0      # подстрой под свой уровень
WIN_MS       = 1          # окно RMS  ??? для PSK можно 1ms но может быть обрезан последний бит 
GUARD_MS     = 0.0        # добавление поля слева/справа к окну импульса
START_DELAY_MS = 3.0      # обрезание начало сигнала ??? если начальная фаза больше или меньше 1.1 то убивает поиск фронта ???
CALIB_DB     = -30.0      # если хочешь, учти калибровку тракта

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

file_out_iq = r"psk_out_f50.f32"

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

# формируем шаблон пути (сразу строка!)
FILE_PATH = str(ROOT / "captures" / FILE_IQ_1)
file_out = str(ROOT / "captures" / file_out_iq)


# ==========================
# ОСНОВНАЯ ЛОГИКА
# ==========================

def save_array_bin(arr, filename=FILE_PATH, count=None):
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
    msg_hex, phase_res, edges = phase_demod_psk_msg(data=phase_rad)
    dt = time.perf_counter() - t0
    print(f"phase_demod_psk_msg() заняла {dt*1000:.3f} мс")
    
    FSd = SAMPLE_RATE_SPS/4
    phase_res = (
        f"Carrier={(edges[0]/FSd*1e3)}ms "
        f"Pos={phase_res['PosPhase']:.2f}rad "
        f"Neg={phase_res['NegPhase']:.2f}rad "
        f"Rise={(phase_res['PhRise']/FSd*1e6):.1f}us "
        f"Fall={(phase_res['PhFall']/FSd*1e6):.1f}us "
        f"Ass={phase_res['Ass']:.3f}%"
        f"Fmod={(FSd/phase_res['Tmod']):.3f}Hz"
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
    #iq_seg = seg_rms_shot(iq_data)
    #run_self_test_psk(iq_data=iq_seg)
    
    # --- финальный «холд», чтобы все окна остались на экране ---
    plt.ioff()   # выключаем интерактив перед финальным блокирующим показом
    plt.show()   # одно блокирующее show удержит все уже открытые окна


    