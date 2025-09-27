from lib.logger import get_logger
log = get_logger(__name__)
# plot_bin_f32_front.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional, Dict, Any

# === path hack (fixed to project root) ===
ROOT = Path(__file__).resolve().parents[1]  # beacon406/
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
# ===================================

from lib.demod import phase_demod_psk_msg_safe

# You can override this path via CLI: python plot_bin_f32_front.py "C:\path\to\file.f32"

FILE_IQ_1 = r"C:/work/TesterSDR/captures\psk_shot2fsk_out.f32"
FILE_IQ_2 = r"C:/work/TesterSDR/captures\psk_shot_out.f32"  
FILE_IQ_3 = r"C:/work/TesterSDR/captures\psk_shot_out_trimmed2.f32" #start_idx=0
FILE_IQ_4 = r"C:/work/TesterSDR/captures\psk_msg_out.f32"
FILE_IQ_5 = r"C:/work/TesterSDR/captures\psk_out_f50.f32"
FILE_IQ_6 = r"C:/work/TesterSDR/captures\psk_out_f50.f32"
FILE_IQ_7 = r"C:/work/TesterSDR/captures\psk_out_f100.f32"
FILE_IQ_8 = r"C:/work/TesterSDR/captures/psk_out_f150.f32"

FILE_IQ_9 = r"C:/work/TesterSDR/captures\psk_real_msg_out.f32" 
FILE_IQ_10 = r"C:/work/TesterSDR/captures\psk_real_noise_50k_out.f32" 
FILE_IQ_11 = r"C:/work/TesterSDR/captures\psk_real_noise_msg_out.f32" 

#from pathlib import Path
#FILE_PATH = str(Path(__file__).resolve().parents[2] / "captures" / FILE_IQ_11 )

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
FILE_PATH = str(ROOT / "captures" / FILE_IQ_11)

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

def main():

    p = Path(FILE_PATH)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    y = np.fromfile(p, dtype=np.float32).astype(np.float64)

    plt.figure(figsize=(16, 5))
    plt.plot(y, linewidth=1.1)

    #edges = detect_all_steps_by_mean(data=y) 
    
    msg_hex, phase_res, edges = phase_demod_psk_msg_safe(data=y)
    FS = 1_000_000/4
    phase_res = (
        f"Carrier={(edges[0]/FS*1e3)}ms "
        f"Pos={phase_res['PosPhase']:.2f}rad "
        f"Neg={phase_res['NegPhase']:.2f}rad "
        f"Rise={(phase_res['PhRise']/FS*1e6):.1f}us "
        f"Fall={(phase_res['PhFall']/FS*1e6):.1f}us "
        f"Ass={phase_res['Ass']:.3f}%"
        f"Fmod={(FS/phase_res['Tmod']):.3f}Hz"
        )
    
    if edges.size :
        plt.title(f"{p.name}  |  front @ idx={edges[0]} \n{phase_res}\n HEX={msg_hex}")
        for t in edges:
            plt.axvline(t, color='crimson', lw=1.0, alpha=0.6)
    else:
        plt.title(f"{p.name}  |  front NOT found")
    
    plt.xlabel("Sample index")
    plt.ylabel("Value (float32 units)")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
