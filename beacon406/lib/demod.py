from lib.logger import get_logger
log = get_logger(__name__)
# demod_psk_message.py
import numpy as np

#window=40           #требует проверки расчитана только на detect_all_steps_by_mean()
#threshold=0.5       #проверял на шумном сообщении может быть нужно подобрать и изменить пропуск окна... 
#start_idx=25000

#max_msg = 31        #Расширенный стандартный формат (long message)
#max_bits_msg = 250  #250 бит (31,25 байт)
max_half_bits =500


def halfbits_to_bytes_fast(half_bits):
    hb = np.asarray(half_bits, dtype=np.uint8)
    if hb.size & 1:               # нечётное — отбрасываем последний (как у тебя)
        hb = hb[:-1]
    even = hb[0::2]; odd = hb[1::2]
    # 10 -> 1, 01 -> 0 ; 00/11 считаем ошибкой (см. ниже "soft resync")
    bad = np.flatnonzero((even ^ odd) == 0)
    if bad.size:
        raise ValueError(f"Недопустимые пары полубитов на позициях: {bad[:8]} ...")
    bits = (even & (~odd))        # [1,0]->1 ; [0,1]->0
    out = np.packbits(bits)       # в байты (MSB-first)
    return out.tolist()

#GPT
def halfbits_to_bytes_(half_bits):
    """
    half_bits : список из 0/1 (длина должна быть чётной).
                Каждая пара = 1 бит:
                [1,0] -> 1
                [0,1] -> 0

    return    : список байтов (int), MSB-first
    """
    # Дополняем к последнему если длина нечетная
    #if len(half_bits) % 2 != 0:
    #    half_bits.append(1 - half_bits[-1])
    
    # удаляем последний если длина нечетная
    if len(half_bits) % 2 != 0:
        half_bits.pop()

    # пары полубитов -> биты
    bits = []
    for i in range(0, len(half_bits), 2):
        pair = half_bits[i:i+2]
        if pair == [1, 0]:
            bits.append("1")
        elif pair == [0, 1]:
            bits.append("0")
        else:
            raise ValueError(f"Недопустимая пара: {pair}")

    # добивка до кратности 8
    pad = (-len(bits)) % 8
    bits = "0"*pad + "".join(bits)

    # биты -> байты
    out = []
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return out

def halfbits_to_bytes(half_bits):
    """
    half_bits : список из 0/1. Каждая пара = 1 бит:
                [1,0] -> 1
                [0,1] -> 0
    return    : список байтов (int), MSB-first
    """
    hb = list(half_bits)  # не мутируем исходные

    # если длина нечётная — отбрасываем последний полубит (как ты и решил)
    if len(hb) % 2 != 0:
        hb.pop()

    # пары полубитов -> биты
    bits_list = []
    for i in range(0, len(hb), 2):
        pair = hb[i:i+2]
        if pair == [1, 0]:
            bits_list.append("1")
        elif pair == [0, 1]:
            bits_list.append("0")
        else:
            raise ValueError(f"Недопустимая пара: {pair}")

    # добивка до кратности 8 НУЛЯМИ В КОНЕЦ (LSB конца байта)
    bits = "".join(bits_list)
    pad = (-len(bits)) % 8
    if pad:
        bits = bits + "0" * pad   # <-- ключевая правка

    # биты -> байты (MSB-first)
    out = []
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return out

def extract_half_bits_vec(data, edges0, mean_period, max_half_bits=None):
    n = len(data)
    if not np.isfinite(mean_period) or mean_period <= 0:
        return []
    # сколько полубитов максимум помещается до конца массива:
    max_by_len = int(np.floor((n - (edges0 + mean_period/2)) / mean_period)) + 1
    k = max_by_len if (max_half_bits is None) else min(max_half_bits, max_by_len)
    idx = edges0 + mean_period/2 + mean_period * np.arange(k, dtype=np.float64)
    idx = np.clip(np.rint(idx).astype(np.int32), 0, n-1)
    return (np.asarray(data)[idx] > 0).astype(np.uint8).tolist()


def extract_half_bits(data, edges0, mean_period, max_half_bits=500):
    #Извлечение битовой последовательности модуляции.
    half_bits = []
    idx = edges0 + mean_period / 2

    while idx < len(data):
        half_bit = 1 if data[int(round(idx))] > 0 else 0
        half_bits.append(half_bit)
        idx += mean_period
        if max_half_bits is not None and len(half_bits) >= max_half_bits:
            break
    return half_bits

def calc_mean_period(edges, n_intervals=28):
    #Вычисляет средний период (в выборках) по первым n_intervals фронтам.
    edges = np.asarray(edges)
    if edges.size < n_intervals + 1:
        raise ValueError(f"Нужно минимум {n_intervals+1} фронтов, а есть {edges.size}")

    # берём первые n_intervals интервалов
    intervals = np.diff(edges[:n_intervals+1])
    return float(np.mean(intervals))

#Gemini хорошо рабоает берем за базовый!!!
def detect_all_steps_by_mean(data, window=40, mean_diff_threshold=0.5, start_idx=25000):
    """
    Детектор фронтов (перепадов), основанный на разнице средних в скользящем окне.
    """
    edges = []
    # Убедимся, что окно четное для простоты деления
    if window % 2 != 0:
        window += 1
    
    half_window = window // 2
    i = start_idx
    
    while i < len(data) - window:
        # 1. Делим окно на две половины
        left_half = data[i : i + half_window]
        right_half = data[i + half_window : i + window]
        
        # 2. Вычисляем среднее для каждой половины
        mean_left = np.mean(left_half)
        mean_right = np.mean(right_half)
        
        # 3. Проверяем, достаточно ли велика разница
        if abs(mean_right - mean_left) > mean_diff_threshold:
            # Фронт находится где-то на границе половин
            edge_idx = i + half_window
            edges.append(edge_idx)
            
            # Пропускаем окно, чтобы не детектировать тот же фронт много раз
            i += 2*window 
        else:
            i += 1
            
    return np.array(edges, dtype=int)

def detect_all_steps_by_mean_fast(data, window=40, mean_diff_threshold=0.5, start_idx=0):
    """
    Быстрый детектор перепада по разнице средних двух половин окна.
    O(1) на шаг за счёт кумсум. Возвращает индексы середины окна (граница половин).
    """
    x = np.asarray(data, dtype=np.float64)
    m = x.size
    if m == 0:
        return np.array([], dtype=int)

    # Окно должно быть чётным и >=2
    window = int(window)
    if window < 2:
        window = 2
    if window % 2:
        window += 1
    hw = window // 2

    if m < window:
        return np.array([], dtype=int)

    i0 = int(max(0, start_idx))
    # допустимые старты окон: j = 0..(m-window)
    n_starts = m - window + 1
    if i0 >= n_starts:
        return np.array([], dtype=int)

    # Кумулятивные суммы: сумма x[a:b] = c[b] - c[a]
    c = np.cumsum(np.concatenate(([0.0], x)))

    # Вектор стартов окон, одинаковая длина для всех расчётов
    idx = np.arange(i0, n_starts, dtype=int)

    # Левые/правые половины: x[j:j+hw] и x[j+hw:j+window]
    sl = c[idx + hw]     - c[idx]
    sr = c[idx + window] - c[idx + hw]

    # Разница средних половин
    delta = np.abs(sr / hw - sl / hw)

    # Порог
    hit_rel = np.flatnonzero(delta > float(mean_diff_threshold))
    if hit_rel.size == 0:
        return np.array([], dtype=int)

    hits = idx[hit_rel]

    # Схлопываем подряд идущие срабатывания в один фронт
    split_pts = np.where(np.diff(hits) > 1)[0] + 1
    groups = np.split(hits, split_pts)
    edges = np.fromiter((g[0] + hw for g in groups if g.size), dtype=int)

    return edges


# 
# расчет параметров фазы
# коде стоит суб-сэмпловая оценка: 
# точки t₁₀ и t₉₀ мы находим функциями _cross_up/_cross_dn 
# с линейной интерполяцией между соседними отсчётами, 
# поэтому получаются дробные индексы 
# (и PhRise/PhFall — тоже дробные, не кратные 1 сэмплу).
# сигнатура: calculate_pulse_params(data, pos, num_pulses=None)
# первый фронт всегда пропускаем
# уровни глобальные: 0.9*PosPhase, 0.9*NegPhase
# PhRise: от пересечения 0.9*NegPhase (вверх) до 0.9*PosPhase (вверх)
# PhFall: от 0.9*PosPhase (вниз) до 0.9*NegPhase (вниз)
# времена в дробных сэмплах (линейная интерполяция)
# Ass оставляем NaN
#
#calculate_pulse_params(data, pos, num_pulses=None) → dict
#Первый фронт всегда пропускаем.
#Глобальные уровни: PosPhase/NegPhase; пороги 0.9*Pos, 0.9*Neg.
#PhRise: t_up(0.9*Neg) → t_up(0.9*Pos); PhFall: t_dn(0.9*Pos) → t_dn(0.9*Neg).
#Ass: через пары rise→fall на L+ = φ0+0.1φ2 и fall→rise на L− = φ0−0.1φ2.
#Tmod: медиана интервалов между t_up(L+) у всех восходящих; значение дробное (в сэмплах).

def calculate_pulse_params(data, pos, num_pulses=None):
    """
    Возвращает dict:
      {
        'N','PosPhase','NegPhase','PhRise','PhFall','Ass','Tmod'
      }
    Правила:
      - Первый фронт всегда пропускается.
      - PhRise: t_up(0.9*NegPhase) → t_up(0.9*PosPhase)
      - PhFall: t_dn(0.9*PosPhase) → t_dn(0.9*NegPhase)
      - Ass: по τ1/τ2 на уровнях L± = φ0 ± 0.1*φ2
      - Tmod: медиана интервалов между соседними t_up(L⁺) на восходящих фронтах
      - Все времена в ДРОБНЫХ сэмплах (линейная интерполяция)
    """
    x = np.asarray(data, dtype=np.float64)
    edges = np.asarray(pos, dtype=int)

    # Пропустить первый фронт и ограничить количеством
    if len(edges) >= 1:
        edges = edges[1:]
    if num_pulses is not None:
        edges = edges[:num_pulses]

    if len(edges) < 2:
        return {
            'N': int(len(edges)),
            'PosPhase': np.nan, 'NegPhase': np.nan,
            'PhRise': np.nan, 'PhFall': np.nan,
            'Ass': np.nan, 'Tmod': np.nan
        }

    # Рабочие окна
    Tb = float(np.median(np.diff(edges)))
    pre_lo_a  = int(max(3, round(0.45*Tb)))
    pre_lo_b  = int(max(3, round(0.25*Tb)))
    post_hi_a = int(max(3, round(0.25*Tb)))
    post_hi_b = int(max(3, round(0.45*Tb)))
    slope_hw  = int(max(10, round(0.60*Tb)))   # окно вокруг фронта
    dir_off   = int(max(2,  round(0.08*Tb)))   # для знака наклона
    n = len(x)

    def _pre_post_mean(e):
        a0 = max(0, e - pre_lo_a); a1 = max(0, e - pre_lo_b)
        b0 = min(n, e + post_hi_a); b1 = min(n, e + post_hi_b)
        pre  = float(np.nanmean(x[a0:a1])) if a1 > a0 else np.nan
        post = float(np.nanmean(x[b0:b1])) if b1 > b0 else np.nan
        return pre, post

    def _cross_up(seg, th):
        for i in range(len(seg)-1):
            y0, y1 = seg[i], seg[i+1]
            if y0 < th <= y1:
                return i + (th - y0)/(y1 - y0)
        return None

    def _cross_dn(seg, th):
        for i in range(len(seg)-1):
            y0, y1 = seg[i], seg[i+1]
            if y0 > th >= y1:
                return i + (y0 - th)/(y0 - y1)
        return None

    # Глобальные плато и пороги
    hi_levels, lo_levels = [], []
    for e in edges:
        pre, post = _pre_post_mean(e)
        if np.isfinite(pre) and np.isfinite(post):
            hi_levels.append(max(pre, post))
            lo_levels.append(min(pre, post))
    PosPhase = float(np.nanmean(hi_levels)) if hi_levels else np.nan
    NegPhase = float(np.nanmean(lo_levels)) if lo_levels else np.nan

    th_pos90 = 0.9 * PosPhase
    th_neg90 = 0.9 * NegPhase  # (<0)

    phi2 = 0.5 * (PosPhase - NegPhase)
    phi0 = 0.5 * (PosPhase + NegPhase)
    L_pos = phi0 + 0.1 * phi2
    L_neg = phi0 - 0.1 * phi2

    # ---- Единый проход ----
    rise_times = []   # ← раздельная инициализация (исправлено)
    fall_times = []   # ← раздельная инициализация (исправлено)
    dirs = []         # +1 rise, -1 fall, 0 flat

    t_Lpos_up_abs = []
    t_Lpos_dn_abs = []
    t_Lneg_up_abs = []
    t_Lneg_dn_abs = []

    for e in edges:
        slope = x[min(n-1, e + dir_off)] - x[max(0, e - dir_off)]
        d = 1 if slope > 0 else (-1 if slope < 0 else 0)
        dirs.append(d)

        s0 = max(0, e - slope_hw); s1 = min(n, e + slope_hw)
        seg = x[s0:s1]
        if len(seg) < 5:
            t_Lpos_up_abs.append(None); t_Lpos_dn_abs.append(None)
            t_Lneg_up_abs.append(None); t_Lneg_dn_abs.append(None)
            continue

        # PhRise / PhFall по 0.9
        if d == 1:
            t_neg90_up = _cross_up(seg, th_neg90)
            t_pos90_up = _cross_up(seg, th_pos90)
            if t_neg90_up is not None and t_pos90_up is not None and t_pos90_up > t_neg90_up:
                rise_times.append((s0 + t_pos90_up) - (s0 + t_neg90_up))
        elif d == -1:
            t_pos90_dn = _cross_dn(seg, th_pos90)
            t_neg90_dn = _cross_dn(seg, th_neg90)
            if t_pos90_dn is not None and t_neg90_dn is not None and t_neg90_dn > t_pos90_dn:
                fall_times.append((s0 + t_neg90_dn) - (s0 + t_pos90_dn))

        # L± для Ass и Tmod
        tu_p = _cross_up(seg, L_pos); td_p = _cross_dn(seg, L_pos)
        tu_n = _cross_up(seg, L_neg); td_n = _cross_dn(seg, L_neg)
        t_Lpos_up_abs.append((s0 + tu_p) if tu_p is not None else None)
        t_Lpos_dn_abs.append((s0 + td_p) if td_p is not None else None)
        t_Lneg_up_abs.append((s0 + tu_n) if tu_n is not None else None)
        t_Lneg_dn_abs.append((s0 + td_n) if td_n is not None else None)

    PhRise = float(np.nanmean(rise_times)) if rise_times else np.nan
    PhFall = float(np.nanmean(fall_times)) if fall_times else np.nan

    # τ1/τ2 и Ass
    tau1_list, tau2_list = [], []
    for i in range(len(edges)-1):
        d1, d2 = dirs[i], dirs[i+1]
        if d1 == 1 and d2 == -1:
            t_up_i = t_Lpos_up_abs[i]
            t_dn_j = t_Lpos_dn_abs[i+1]
            if (t_up_i is not None) and (t_dn_j is not None) and (t_dn_j > t_up_i):
                tau1_list.append(t_dn_j - t_up_i)
        if d1 == -1 and d2 == 1:
            t_dn_i = t_Lneg_dn_abs[i]
            t_up_j = t_Lneg_up_abs[i+1]
            if (t_dn_i is not None) and (t_up_j is not None) and (t_up_j > t_dn_i):
                tau2_list.append(t_up_j - t_dn_i)

    Ass = np.nan
    if tau1_list and tau2_list:
        t1 = float(np.nanmean(tau1_list))
        t2 = float(np.nanmean(tau2_list))
        Ass = abs(t1 - t2) / ((t1 + t2)/2) * 100.0

    # Tmod по восходящим фронтам на L_pos
    rise_marks = [t for d, t in zip(dirs, t_Lpos_up_abs) if d == 1 and t is not None]
    if len(rise_marks) >= 2:
        rise_marks = np.asarray(rise_marks, dtype=float)
        T_intervals = np.diff(rise_marks)
        Tmod = float(np.median(T_intervals)) if len(T_intervals) > 0 else np.nan
    else:
        Tmod = np.nan

    return {
        'N': int(len(edges)),
        'PosPhase': PosPhase,
        'NegPhase': NegPhase,
        'PhRise': PhRise,
        'PhFall': PhFall,
        'Ass': Ass,
        'Tmod': Tmod
    }


def demod_psk_msg(data):

    # 1) Фронты
    window=40           #требует проверки расчитана только на detect_all_steps_by_mean()
    threshold=0.5       #проверял на шумном сообщении может быть нужно подобрать и изменить пропуск окна... 
    start_idx=25000     #требует расчета с учетом прореживания  FS/4 
    
    edges = detect_all_steps_by_mean(data, window=window, mean_diff_threshold=threshold, start_idx=start_idx) 
    if edges is None or len(edges) < 29:   # 28 интервалов = 29 фронтов
        log.warning("Недостаточно фронтов: %d", len(edges) if edges is not None else 0)
        #raise ValueError(f"Недостаточно фронтов: {len(edges) if edges is not None else 0}")
    
    # 2) Период по первым 28 фронтам
    mean_period = calc_mean_period(edges,28)
    
    # 3) Полубиты от первого фронта
    half_bits = extract_half_bits(data, edges0=edges[0], mean_period=mean_period, max_half_bits=max_half_bits)
    #half_bits = extract_half_bits_vec(data, edges0=edges[0], mean_period=mean_period, max_half_bits=max_half_bits)

    # 4) Полубиты -> байты
    #byte_list = halfbits_to_bytes(half_bits)   # список int 0..255
    byte_list = halfbits_to_bytes_fast(half_bits)   # список int 0..255
    msg_bytes = bytes(byte_list)               # именно bytes-объект

    # 5) HEX строка (без пробелов)
    msg_hex = "".join(f"{b:02X}" for b in msg_bytes)
    #msg_hex   = msg_bytes.hex().upper()


    return mean_period, msg_bytes, msg_hex

def phase_demod_psk_msg(data):

    # 1) Фронты
    window=40           #требует проверки расчитана только на detect_all_steps_by_mean()
    threshold=0.5       #проверял на шумном сообщении может быть нужно подобрать и изменить пропуск окна... 
    start_idx=25000     #требует расчета с учетом прореживания  FS/4 
    N=28                #количество фронтов для анализа (первый пропускаем) 
    

    
    edges = detect_all_steps_by_mean_fast(data, window=window, mean_diff_threshold=threshold, start_idx=start_idx) 
    #edges = detect_all_steps_by_mean(data, window=window, mean_diff_threshold=threshold, start_idx=start_idx) 
    if edges is None or len(edges) < 29:   # 28 интервалов = 29 фронтов
        raise ValueError(f"phase_demod_psk_msg - Недостаточно фронтов: {len(edges) if edges is not None else 0}")
    
    phase_res = calculate_pulse_params(data, pos=edges, num_pulses=N)
    
    
    # 2) Период по первым 28 фронтам
    #mean_period = calc_mean_period(edges,28)
    #mean_period = phase_res['Tmod']/2

    mean_period = phase_res.get('Tmod', np.nan) / 2.0
    if not np.isfinite(mean_period) or mean_period < 2:
        # робастная «резервная» оценка по фронтам
        mean_period = float(np.median(np.diff(edges)))

    
    # 3) Полубиты от первого фронта
    half_bits = extract_half_bits(data, edges0=edges[0], mean_period=mean_period, max_half_bits=max_half_bits)

    # 4) Полубиты -> байты
    byte_list = halfbits_to_bytes(half_bits)   # список int 0..255
    msg_bytes = bytes(byte_list)               # именно bytes-объект

    # 5) HEX строка (без пробелов)
    msg_hex = "".join(f"{b:02X}" for b in msg_bytes)
    
    return msg_hex, phase_res, edges  


""" 
def demod_psk_msg(data):

    edges = detect_all_steps_by_mean(data, window=window, mean_diff_threshold=threshold, start_idx=start_idx) 
       
    mean_period = calc_mean_period(edges,28)
    #print("Средний период:", mean_period)
    
    half_bits = extract_half_bits(data, edges0=edges[0], mean_period=mean_period, max_half_bits=500)

    #print("Полученные биты:", "".join(map(str, half_bits)))
    
    # без строк вход и выход 
    msg= halfbits_to_bytes(half_bits)
    #print("MsgB:", "".join(f"{b:02X}" for b in bytes))
    msg_str= "".join(f"{b:02X}" for b in bytes)
    
    #s1="1010101010101010101010101010100110100110010101011001010101010101010101010101100101010101010101010101010101010101010101010101010101101010101010101010011010101010101010101001101001010101010101010101100101010101100110100110101010010101010110101010100101010101101010100110100101101001101001010"
    #s2="101010101010101010101010101010011010011001010101100101010101010101010101010110010101010101010101010101010101010101010101010101010110101010101010101001101010101010101010100110100101010101010101010110010101010110011010011010101001010101011010101010010101010110101010011010010110100110100101"
    # строки вход и выход 
    #half_bits_str="".join(map(str, half_bits))
    #msg_str = halfbits_to_bytes_str(half_bits_str)
    #print("MsgS:",msg_str)  # Выводит шестнадцатеричную строку, например, FFFF...

    return mean_period, msg, msg_str
    """

# GPT алгоритм работает но не всегда 
# Поиск ВСЕХ фронтов PSK (восходящих и нисходящих) методом скользящего окна.
# --- простой детектор фронтов по |dphi| со скользящей суммой ---
def find_psk_edges_sliding_window(
    phi: np.ndarray,
    *,
    N: int,                 # длина окна (≈ 0.5…1 × длительности фронта в сэмплах)
    k_sigma: float = 5.0,   # порог = μ + kσ по «тихому» прологу
    quiet_frac: float = 0.10, # доля начального «тихого» участка
    skip: int = 3,          # после срабатывания перескочить skip*N сэмплов
    do_unwrap: bool = True, # разворачивать фазу
    detrend: bool = 
    True,  # убирать линейный тренд (МНК)
    median_k: int = 0       # 0 – выкл; 3/5 – медианное сглаживание по фазе
):
    """
    Возвращает:
        edges_idx: np.ndarray[int] — индексы фронтов в phi
        Tb_samp: float|nan         — оценка шага между фронтами (≈ полбит), в сэмплах
    """
    assert phi.ndim == 1
    n = phi.size
    if n == 0 or N <= 1:
        return np.array([], dtype=int), float("nan")

    x = phi.astype(np.float64, copy=True)

    # 1) подготовка
    if do_unwrap:
        x = np.unwrap(x)
    if detrend:
        m = np.arange(n, dtype=np.float64)
        sx, sy = m.sum(), x.sum()
        sxx, sxy = np.dot(m, m), np.dot(m, x)
        denom = n * sxx - sx * sx
        if denom != 0:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
            x = x - (a * m + b)
    if median_k and (median_k % 2 == 1) and median_k > 1:
        pad = median_k // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        x = np.median(np.lib.stride_tricks.sliding_window_view(xp, median_k), axis=-1)

    # 2) метрика: m = |dphi|
    dphi = np.diff(x, prepend=x[0])
    m_abs = np.abs(dphi)

    # 3) скользящая сумма по окну N
    S = np.empty_like(m_abs)
    S[:N] = m_abs[:N].sum()
    for i in range(N, n):
        S[i] = S[i-1] + m_abs[i] - m_abs[i-N]

    # автопорог на «тишине»
    q = max(1, int(quiet_frac * n))
    mu, sigma = float(S[:q].mean()), float(S[:q].std())
    thr = mu + k_sigma * sigma
    
    # --- новое: пропускаем левый транзиент и считаем порог по «тихому» до него
    guard_left = 2  # сколько окон N пропустить слева (FIR/decim-транзиент)
    q_end = max(1, min(int(quiet_frac * n), guard_left * N))
    mu, sigma = float(S[:q_end].mean()), float(S[:q_end].std())
    thr = mu + k_sigma * sigma
    
    # 4) проход с уточнением пика и skip
    edges = []
    #i = N
    # старт после guard_left*N сэмплов, чтобы переждать транзиент FIR
    i = max(N, guard_left * N)
    while i < n:
        if S[i] > thr:
            L = max(0, i - N + 1)
            R = i + 1
            loc = L + int(np.argmax(m_abs[L:R]))  # уточнение по argmax |dphi|
            edges.append(loc)
            i = loc + skip * N                    # перепрыгиваем хвост
        else:
            i += 1

    edges = np.array(edges, dtype=int)
    Tb_samp = float(np.median(np.diff(edges))) if edges.size >= 3 else float("nan")
    return edges, Tb_samp

def pick_window_for_phase(xs_ms, front_us, alpha=0.8, min_N=12):
    dt_s   = np.median(np.diff(xs_ms)) / 1000.0
    Fs_phi = 1.0 / dt_s
    N      = max(min_N, int(round(Fs_phi * (front_us*1e-6) * alpha)))
    return N, Fs_phi




# --- SAFE, NON-RAISING DEMODULATOR ---------------------------------------------------
# Возвращает тот же кортеж (msg_hex, phase_res, edges), но НИКОГДА не бросает исключения
# по предсказуемым ситуациям типа "мало фронтов". Вместо этого:
#  - msg_hex = "" (пустая строка)
#  - phase_res = словарь с NaN/дефолтами
#  - edges = массив индексов (в т.ч. пустой)
def phase_demod_psk_msg_safe(
    data,
    *, 
    window: int = 40,
    threshold: float = 0.5,
    start_idx: int = 25000,
    N: int = 28,
    min_edges: int = 29
):
    import numpy as _np
    _EMPTY = {
        "PosPhase": _np.nan,
        "NegPhase": _np.nan,
        "PhRise":   _np.nan,
        "PhFall":   _np.nan,
        "Ass":      _np.nan,
        "Tmod":     _np.nan,
    }
    try:
        if data is None or len(data) == 0:
            return "", dict(_EMPTY), _np.array([], dtype=int)

        edges = detect_all_steps_by_mean_fast(
            data, window=window, mean_diff_threshold=threshold, start_idx=start_idx
        )
        if edges is None or len(edges) < min_edges:
            return "", dict(_EMPTY), (edges if edges is not None else _np.array([], dtype=int))

        # метрики фазы
        phase_res = calculate_pulse_params(data, pos=edges, num_pulses=N)

        # период для извлечения полубитов (если не получится — дадим пустой msg)
        try:
            mean_period = calc_mean_period(edges, 28)
        except Exception:
            return "", phase_res, edges

        # полубиты -> HEX
        half_bits = extract_half_bits(data, edges0=edges[0], mean_period=mean_period, max_half_bits=max_half_bits)
        byte_list = halfbits_to_bytes_fast(half_bits)
        msg_bytes = bytes(byte_list)
        msg_hex   = "".join(f"{b:02X}" for b in msg_bytes)

        return msg_hex, phase_res, edges

    except Exception as _e:
        # Непредвиденная внутренняя ошибка: не роняем приложение
        return "", dict(_EMPTY), _np.array([], dtype=int)
# -------------------------------------------------------------------------------------
