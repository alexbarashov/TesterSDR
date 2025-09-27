"""
COSPAS/SARSAT Beacon Tester - Version 2.0
=========================================
Точное воспроизведение интерфейса по изображению.
Одностраничное Flask приложение с аутентичным дизайном.
Порт: 8738 (чтобы не конфликтовать с оригинальным)
"""

from __future__ import annotations
import math
import random
import time
import sys
import os
import threading
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import List
from flask import Flask, jsonify, request, Response

# Добавляем путь к библиотекам
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.hex_decoder import hex_to_bits, build_table_rows
from lib.metrics import process_psk_impulse
from lib.demod import phase_demod_psk_msg_safe
from lib.processing_fm import fm_discriminator
from lib.backends import safe_make_backend  # SDR backend support
from lib.config import BACKEND_NAME, BACKEND_ARGS
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Глобальные переменные для SDR
sdr_backend = None
sdr_device_info = "No SDR detected"

# Глобальные переменные для real-time обработки
sdr_running = False
reader_thread = None
data_lock = threading.Lock()
pulse_queue = queue.Queue()
current_rms_dbm = -100.0  # Текущее значение RMS в dBm
last_pulse_data = None    # Данные о последнем импульсе

# Параметры обработки сигналов (из beacon406-plot.py)
TARGET_SIGNAL_HZ = 406_037_000
IF_OFFSET_HZ = -37_000
CENTER_FREQ_HZ = TARGET_SIGNAL_HZ + IF_OFFSET_HZ
SAMPLE_RATE_SPS = 1_000_000
BB_SHIFT_ENABLE = True
BB_SHIFT_HZ = IF_OFFSET_HZ
RMS_WIN_MS = 1.0
DBM_OFFSET_DB = -30.0
PULSE_THRESH_DBM = -45.0
READ_CHUNK = 65536
EPS = 1e-20

# STRICT_COMPAT: Кольцевой буфер IQ и actual sample rate
iq_ring_buffer = None
actual_sample_rate_sps = SAMPLE_RATE_SPS  # По умолчанию, обновится при init SDR
bp_sample_counter = 0  # Счетчик отсчетов БП-сигнала после NCO

# STRICT_COMPAT: История импульсов для PSK-406 анализа
PULSE_HISTORY_MAX = 50
pulse_history = deque(maxlen=PULSE_HISTORY_MAX)

# Буферы для обработки данных
sample_counter = 0
win_samps = int(RMS_WIN_MS * 1e-3 * SAMPLE_RATE_SPS)
nco_phase = 0.0
nco_k = 2.0 * np.pi * BB_SHIFT_HZ / SAMPLE_RATE_SPS  # Будет пересчитан в init_sdr_backend
tail_p = np.array([], dtype=np.float32)
full_rms = np.array([], dtype=np.float32)
full_idx = np.array([], dtype=np.int64)
rms_history = deque(maxlen=1000)  # Для веб-отображения
time_history = deque(maxlen=1000)

# Состояние детекции импульсов
in_pulse = False
pulse_start_abs = 0
last_pulse_data = None

# STRICT_COMPAT: Кольцевой буфер IQ для хранения 3+ секунд сигнала
class IQRingBuffer:
    def __init__(self, duration_sec: float, sample_rate: float):
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.capacity = int(duration_sec * sample_rate)
        self.buffer = np.zeros(self.capacity, dtype=np.complex64)
        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()
        print(f"[BUFFER] Created: {self.capacity} samples ({duration_sec}s at {sample_rate:.0f} Hz)")

    def write(self, samples: np.ndarray):
        """Записать отсчеты в кольцевой буфер"""
        with self.lock:
            n = len(samples)
            if n >= self.capacity:
                # Если данных больше емкости, берем последние
                self.buffer[:] = samples[-self.capacity:]
                self.write_pos = 0
                self.total_written += n
            else:
                # Запись в две части если переход через границу
                end_pos = self.write_pos + n
                if end_pos <= self.capacity:
                    self.buffer[self.write_pos:end_pos] = samples
                    self.write_pos = end_pos % self.capacity
                else:
                    first_part = self.capacity - self.write_pos
                    self.buffer[self.write_pos:] = samples[:first_part]
                    self.buffer[:n-first_part] = samples[first_part:]
                    self.write_pos = n - first_part
                self.total_written += n

    def get_segment(self, abs_start: int, abs_end: int) -> np.ndarray:
        """Извлечь сегмент по абсолютным индексам"""
        with self.lock:
            # Проверка доступности данных
            oldest_available = max(0, self.total_written - self.capacity)
            if abs_start < oldest_available:
                print(f"[BUFFER] Segment too old: start={abs_start} < oldest={oldest_available}")
                return np.array([], dtype=np.complex64)

            # Вычисляем относительные позиции от начала доступных данных
            available_samples = min(self.total_written, self.capacity)
            buffer_abs_start = self.total_written - available_samples

            start_offset = abs_start - buffer_abs_start
            end_offset = abs_end - buffer_abs_start

            if start_offset < 0 or end_offset > available_samples:
                print(f"[BUFFER] Segment out of bounds: offset={start_offset}..{end_offset}, available={available_samples}")
                return np.array([], dtype=np.complex64)

            # Case 1: Buffer not yet full (linear access)
            if self.total_written <= self.capacity:
                return self.buffer[start_offset:end_offset].copy()

            # Case 2: Buffer is full and wrapped (ring access)
            else:
                # write_pos points to oldest data in ring buffer
                logical_start = (self.write_pos + start_offset) % self.capacity
                logical_end = (self.write_pos + end_offset) % self.capacity

                if logical_end > logical_start:
                    # Segment doesn't wrap around
                    return self.buffer[logical_start:logical_end].copy()
                else:
                    # Segment wraps around buffer boundary
                    return np.concatenate([
                        self.buffer[logical_start:],
                        self.buffer[:logical_end]
                    ]).copy()

def get_actual_fs() -> float:
    """Получить фактический sample rate"""
    global actual_sample_rate_sps
    return actual_sample_rate_sps

# STRICT_COMPAT: PSK-406 анализ с заглушкой демодуляции
def analyze_psk406(iq_seg: np.ndarray, fs: float) -> dict:
    """Анализ PSK-406 сигнала с заглушкой для будущей интеграции"""
    result = {
        'bitrate_bps': None,
        'pos_phase': None,
        'neg_phase': None,
        'ph_rise': None,
        'ph_fall': None,
        'symmetry_pct': None,
        'msg_hex': None,
        'msg_ok': None
    }

    try:
        if iq_seg.size == 0:
            return result

        pulse_len_ms = len(iq_seg) / fs * 1000

        # ИСПРАВЛЕНИЕ: Применяем _find_pulse_segment для правильной обрезки преамбулы
        thresh_dbm = -60.0  # порог как в FILE
        win_ms = 1.0  # окно RMS как в FILE  
        start_delay_ms = 3.0  # обрезание начала как в FILE
        calib_db = -30.0  # калибровка как в FILE
        
        # Применяем _find_pulse_segment для обрезки преамбулы
        iq_seg_trimmed = _find_pulse_segment(iq_seg, fs, thresh_dbm, win_ms, start_delay_ms, calib_db)
        
        if iq_seg_trimmed is not None and len(iq_seg_trimmed) > 0:
            iq_seg = iq_seg_trimmed  
            print(f"[PSK-RUN] _find_pulse_segment success: {len(iq_seg)} samples")
        else:
            print(f"[PSK-RUN] _find_pulse_segment failed, using original: {len(iq_seg)} samples")

        # Используем тот же путь обработки что и рабочая кнопка File
        baseline_ms = 2.0  # PSK_BASELINE_MS - такой же как в FILE режиме
        t0_offset_ms = 0.0

        # ДИАГНОСТИКА: Информация о сегменте перед обработкой
        print(f"[RUN-DIAG] Final segment: {len(iq_seg)} samples, baseline_ms={baseline_ms}")
        print(f"[RUN-DIAG] First 5 IQ values: {iq_seg[:5] if len(iq_seg) >= 5 else iq_seg}")
        print(f"[RUN-DIAG] IQ segment statistics: mean_abs={np.mean(np.abs(iq_seg)):.6f}, max_abs={np.max(np.abs(iq_seg)):.6f}")

        pulse_result = process_psk_impulse(
            iq_seg=iq_seg,  # ПОЛНЫЙ сегмент как в FILE
            fs=fs,
            baseline_ms=baseline_ms,
            t0_offset_ms=t0_offset_ms,
            use_lpf_decim=True,
            remove_slope=True,
        )

        if not pulse_result or "phase_rad" not in pulse_result:
            print(f"[PSK] process_psk_impulse failed")
            return result

        # Настоящий PSK демодулятор на обработанных фазовых данных с правильным start_idx
        phase_data = pulse_result["phase_rad"]
        # Используем тот же start_idx что и в FILE режиме
        safe_start_idx = min(25000, len(phase_data)//4) if len(phase_data) > 100000 else 0
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(
            data=phase_data,
            window=40,
            threshold=0.5,
            start_idx=safe_start_idx,
            N=28,
            min_edges=29
        )

        # Записываем edges[0] в файл для анализа
        edges_info = f"[RUN] edges[0] = {edges[0] if edges is not None and len(edges) > 0 else 'No edges'}"
        edges_info += f", start_idx = {safe_start_idx}, phase_data len = {len(phase_data)}"
        with open("../edges_run_debug.txt", "w") as f:
            f.write(edges_info + "\n")
        print(edges_info)

        # Извлекаем результаты демодуляции
        if phase_res and not np.isnan(phase_res.get("PosPhase", np.nan)):
            # Добавляем фазовые данные для STATE как в FILE
            phase_data = pulse_result["phase_rad"]
            xs_ms = pulse_result.get("xs_ms", [])

            result.update({
                'bitrate_bps': 400.0,  # Стандартный PSK-406
                'pos_phase': float(phase_res.get("PosPhase", 0.78)),
                'neg_phase': float(phase_res.get("NegPhase", -0.78)),
                'ph_rise': float(phase_res.get("PhRise", 25.0)),
                'ph_fall': float(phase_res.get("PhFall", 23.0)),
                'symmetry_pct': float(phase_res.get("Ass", 0.0)),
                'msg_hex': msg_hex if msg_hex else None,
                'msg_ok': bool(msg_hex and len(msg_hex) > 0),
                # Добавляем фазовые данные для отображения
                'phase_data': phase_data.tolist() if hasattr(phase_data, 'tolist') else list(phase_data),
                'xs_ms': xs_ms.tolist() if hasattr(xs_ms, 'tolist') else list(xs_ms)
            })
            print(f"[PSK] Real demod: {msg_hex}, phases: pos={result['pos_phase']:.3f}, neg={result['neg_phase']:.3f}")
            print(f"[PSK] Phase data: {len(result['phase_data'])} points, sample: {result['phase_data'][:3]}")
        else:
            print(f"[PSK] Demodulation failed - no valid phases detected")

        print(f"[PSK] Analyzed segment: {iq_seg.size} samples, {pulse_len_ms:.1f}ms")

        # NEW: FM discriminator с теми же параметрами что в File режиме
        try:
            fm_result = fm_discriminator(
                iq=iq_seg,
                fs=fs,
                pre_lpf_hz=50000,  # антиалиасный фильтр 50 кГц
                decim=4,            # децимация в 4 раза
                smooth_hz=15000,    # сглаживание 15 кГц
                detrend=False,      # отключить удаление линейного тренда
                center=False,       # без центрирования
                fir_taps=127        # длина FIR фильтра
            )

            fm_freq = fm_result.get("freq_hz", [])
            fm_xs = fm_result.get("xs_ms", [])

            # Добавляем FM данные в результат
            result['fm_data'] = fm_freq.tolist() if isinstance(fm_freq, np.ndarray) else list(fm_freq) if fm_freq is not None else []
            result['fm_xs_ms'] = fm_xs.tolist() if isinstance(fm_xs, np.ndarray) else list(fm_xs) if fm_xs is not None else []

            print(f"[FM-RUN] FM processed: {len(result['fm_data'])} freq points, {len(result['fm_xs_ms'])} time points")
        except Exception as e:
            print(f"[FM-RUN] FM processing error: {e}")
            result['fm_data'] = []
            result['fm_xs_ms'] = []

        # Добавляем расчет preamble_ms из edges[0] как в FILE режиме
        FSd = fs / 4.0  # Частота дискретизации после децимации
        if edges is not None and len(edges) > 0:
            preamble_ms = float(edges[0] / FSd * 1e3)
            result['preamble_ms'] = preamble_ms
            # Записываем для отладки
            with open("../preamble_run_debug.txt", "w") as f:
                f.write(f"[RUN] preamble_ms = {preamble_ms:.3f} (edges[0]={edges[0]}, FSd={FSd})\n")
            print(f"[RUN] Calculated preamble_ms = {preamble_ms:.3f}")

    except Exception as e:
        print(f"[PSK] Analysis error: {e}")

    return result

def db10(x: np.ndarray) -> np.ndarray:
    """dB calculation helper"""
    return 10.0 * np.log10(np.maximum(x, EPS))

def init_sdr_backend():
    """Инициализация SDR backend"""
    global sdr_backend, sdr_device_info

    # Если backend не настроен или auto, пробуем инициализацию
    if BACKEND_NAME == "none" or not BACKEND_NAME:
        sdr_device_info = "SDR disabled (config: none)"
        print(f"[SDR] {sdr_device_info}")
        return False

    try:
        print(f"[SDR] Initializing backend: {BACKEND_NAME}")

        extra_kwargs = {"if_offset_hz": IF_OFFSET_HZ} if (BACKEND_NAME == "file") else {}

        sdr_backend = safe_make_backend(
            BACKEND_NAME,
            sample_rate=SAMPLE_RATE_SPS,
            center_freq=CENTER_FREQ_HZ,
            gain_db=30.0,
            agc=False,
            corr_ppm=0,
            device_args=BACKEND_ARGS,
            **extra_kwargs,
        )

        # Получаем информацию об устройстве
        try:
            status = sdr_backend.get_status()
            device_info = status.get('device_info', 'Unknown device')
            driver = status.get('driver', BACKEND_NAME)
            sdr_device_info = f"{driver}: {device_info}"
            print(f"[SDR] Detected: {sdr_device_info}")

            # STRICT_COMPAT: Сохраняем actual sample rate и создаем буфер
            global actual_sample_rate_sps, iq_ring_buffer, win_samps, nco_k
            actual_sample_rate_sps = status.get('actual_sample_rate_sps', SAMPLE_RATE_SPS)
            iq_ring_buffer = IQRingBuffer(duration_sec=3.0, sample_rate=actual_sample_rate_sps)
            # Обновляем win_samps и nco_k с фактическим sample rate
            win_samps = int(RMS_WIN_MS * 1e-3 * actual_sample_rate_sps)
            nco_k = 2.0 * np.pi * BB_SHIFT_HZ / actual_sample_rate_sps
            print(f"[SDR] Actual sample rate: {actual_sample_rate_sps:.3f} Hz")
            print(f"[SDR] IQ buffer created for {iq_ring_buffer.duration_sec} seconds")
        except Exception:
            sdr_device_info = f"{BACKEND_NAME} device"

        return True
    except Exception as e:
        # Обработка ошибки без проблем с кодировкой
        try:
            error_msg = str(e)
        except:
            error_msg = "Unknown error"

        # Определяем тип ошибки и формируем понятное сообщение
        if "not found" in error_msg.lower() or "no device" in error_msg.lower():
            sdr_device_info = f"No SDR devices found ({BACKEND_NAME})"
        elif BACKEND_NAME == "file" and BACKEND_ARGS:
            sdr_device_info = f"File mode: {BACKEND_ARGS[-30:]}"
        else:
            # Убираем русские символы из сообщения об ошибке
            clean_msg = ''.join(c if ord(c) < 128 else '?' for c in error_msg)
            sdr_device_info = f"SDR init failed: {clean_msg[:35]}"

        print(f"[SDR] Initialization failed (non-critical)")
        return False

def start_sdr_capture():
    """Запуск захвата SDR данных в реальном времени"""
    global sdr_running, reader_thread

    if not sdr_backend:
        print("[SDR] Backend not initialized, cannot start capture")
        return False

    if sdr_running:
        print("[SDR] Capture already running")
        return True

    sdr_running = True
    reader_thread = threading.Thread(target=sdr_reader_loop, daemon=True)
    reader_thread.start()
    print("[SDR] Real-time capture started")
    return True

def stop_sdr_capture():
    """Остановка захвата SDR данных"""
    global sdr_running
    sdr_running = False
    print("[SDR] Real-time capture stopped")

def sdr_reader_loop():
    """Основной цикл чтения и обработки SDR данных"""
    global sample_counter, nco_phase, tail_p, full_rms, full_idx, in_pulse, pulse_start_abs, sdr_running

    print("[SDR] Reader loop starting...")
    error_count = 0
    max_errors = 10

    try:
        while sdr_running:
            try:
                # Читаем блок данных
                samples = sdr_backend.read(READ_CHUNK)
                if samples.size == 0:
                    if BACKEND_NAME == "file":
                        print("[SDR] EOF reached in file mode - stopping")
                        break
                    else:
                        print("[SDR] No samples received, retrying...")
                        time.sleep(0.1)
                        continue

                # Обрабатываем блок
                process_samples_realtime(samples)
                error_count = 0  # Сбрасываем счетчик ошибок при успешном чтении

            except Exception as e:
                error_count += 1
                print(f"[SDR] Read error #{error_count}: {e}")

                if error_count >= max_errors:
                    print(f"[SDR] Too many errors ({error_count}), stopping capture")
                    break

                time.sleep(0.1)  # Увеличиваем паузу при ошибках
                continue

    except Exception as e:
        print(f"[SDR] Critical reader loop error: {e}")
    finally:
        print("[SDR] Reader loop exiting...")
        # НЕ вызываем stop_sdr_capture() здесь чтобы избежать рекурсии
        sdr_running = False

def process_samples_realtime(samples: np.ndarray):
    """Обработка блока SDR данных в реальном времени"""
    global sample_counter, nco_phase, tail_p, full_rms, full_idx, in_pulse, pulse_start_abs

    now = time.time()
    base_idx = sample_counter
    x = samples.copy()

    # Baseband shift (если включен)
    if BB_SHIFT_ENABLE and abs(BB_SHIFT_HZ) > 0:
        n = np.arange(samples.size, dtype=np.float64)
        mixer = np.exp(1j * (nco_phase + nco_k * n)).astype(np.complex64)
        x *= mixer
        nco_phase = float((nco_phase + nco_k * samples.size) % (2.0 * np.pi))

    # STRICT_COMPAT: Запись БП-сигнала в кольцевой буфер
    global iq_ring_buffer, bp_sample_counter
    if iq_ring_buffer is not None:
        iq_ring_buffer.write(x)
        bp_sample_counter += len(x)

    # Вычисление мощности
    p_block = (np.abs(x) ** 2)

    # Объединяем с хвостом предыдущего блока
    if tail_p.size:
        p_cont = np.concatenate((tail_p, p_block))
        p_cont_start_idx = base_idx - tail_p.size
    else:
        p_cont = p_block
        p_cont_start_idx = base_idx

    # RMS анализ
    if p_cont.size >= win_samps:
        c = np.cumsum(p_cont, dtype=np.float64)
        S_valid = c[win_samps - 1:] - np.concatenate(([0.0], c[:-win_samps]))
        P_win = S_valid / float(win_samps)

        # Калибровка dBm
        if sdr_backend:
            rms_dbm_vec = db10(P_win) + DBM_OFFSET_DB + sdr_backend.get_calib_offset_db()
        else:
            rms_dbm_vec = db10(P_win) + DBM_OFFSET_DB

        idx_end = p_cont_start_idx + (win_samps - 1) + np.arange(rms_dbm_vec.size, dtype=np.int64)

        # Обновляем буферы RMS
        with data_lock:
            global current_rms_dbm
            # Добавляем данные в истории для веб-отображения
            for i, rms_val in enumerate(rms_dbm_vec):
                time_history.append(now + i * 1e-6)  # Примерное время
                rms_history.append(float(rms_val))
                # Обновляем текущее значение RMS
                current_rms_dbm = float(rms_val)
                
                
        # DEBUG: печать диапазона dBm и доли точек выше порога
        #peak = float(np.max(rms_dbm_vec)) if rms_dbm_vec.size else float('-inf')
        #frac = float(np.mean(rms_dbm_vec >= PULSE_THRESH_DBM)) if rms_dbm_vec.size else 0.0
        #print(f"[RMS] max={peak:.1f} dBm, thr={PULSE_THRESH_DBM:.1f} dBm, over={frac*100:.1f}%")


        # Детекция импульсов
        detect_pulses(rms_dbm_vec, idx_end, x, p_cont_start_idx)

        # Сохраняем хвост для следующего блока
        if p_cont.size > win_samps:
            tail_p = p_cont[-(p_cont.size - win_samps):]
        else:
            tail_p = np.array([], dtype=np.float32)
    else:
        tail_p = p_cont

    sample_counter += samples.size

def detect_pulses(rms_dbm_vec, idx_end, iq_data, start_idx):
    """Детекция импульсов по RMS порогу"""
    global in_pulse, pulse_start_abs, last_pulse_data

    # Поиск превышений порога
    on = rms_dbm_vec >= PULSE_THRESH_DBM
    trans = np.diff(on.astype(np.int8), prepend=on[0])
    start_pos = np.where(trans == 1)[0]
    end_pos = np.where(trans == -1)[0] - 1

    # Обработка начала импульса
    for start_idx_local in start_pos:
        if not in_pulse:
            in_pulse = True
            pulse_start_abs = idx_end[start_idx_local]
            print(f"[PULSE] Started at sample {pulse_start_abs}")

    # Обработка конца импульса
    for end_idx_local in end_pos:
        if in_pulse:
            pulse_end_abs = idx_end[end_idx_local]
            pulse_len_samples = pulse_end_abs - pulse_start_abs + 1
            pulse_len_ms = pulse_len_samples / get_actual_fs() * 1000

            print(f"[PULSE] Ended at sample {pulse_end_abs}, length: {pulse_len_ms:.1f}ms")

            # Если импульс достаточно длинный, пытаемся демодулировать
            if pulse_len_ms >= 400:  # Минимальная длина для PSK406 маяка
                try:
                    # Извлекаем сегмент IQ данных для импульса
                    # (Здесь нужна более сложная логика для извлечения правильного сегмента)
                    process_pulse_segment(pulse_start_abs, pulse_end_abs)
                except Exception as e:
                    print(f"[PULSE] Processing error: {e}")

            in_pulse = False

def process_pulse_segment(start_abs, end_abs, iq_fallback=None):
    """Обработка сегмента импульса для PSK демодуляции"""
    global iq_ring_buffer, last_pulse_data

    # STRICT_COMPAT: Извлекаем сегмент из кольцевого буфера 
    if iq_ring_buffer is not None:
        # Конвертируем индексы RMS в индексы БП-сигнала
        # ИСПРАВЛЕНИЕ: Захватываем больше данных перед импульсом для поиска преамбулы
        preamble_ms = 200.0  # Добавляем 200 мс перед импульсом
        preamble_samples = int(preamble_ms * 1e-3 * get_actual_fs())

        bp_start = max(0, start_abs - win_samps + 1 - preamble_samples)  # Ограничиваем минимумом 0
        bp_end = end_abs + 1

        # Проверяем доступность данных в буфере используя логику класса
        oldest_available = max(0, iq_ring_buffer.total_written - iq_ring_buffer.capacity)
        newest_available = iq_ring_buffer.total_written

        # Корректируем границы под доступные данные
        effective_start = max(bp_start, oldest_available)
        effective_end = min(bp_end, newest_available)

        if effective_start < effective_end:
            iq_segment = iq_ring_buffer.get_segment(effective_start, effective_end)
        else:
            iq_segment = np.array([])  # Пустой массив если данных нет

        print(f"[RUN-EXTRACT] Buffer range: {oldest_available} to {newest_available}")
        print(f"[RUN-EXTRACT] Requested range: {bp_start} to {bp_end}")
        print(f"[RUN-EXTRACT] Effective range: {effective_start} to {effective_end}")
        print(f"[RUN-EXTRACT] Extended segment with preamble: {iq_segment.size if hasattr(iq_segment, 'size') else len(iq_segment)} samples, preamble_samples={preamble_samples}")

        if iq_segment.size > 0:
            print(f"[PULSE] Extracted segment: {iq_segment.size} samples from buffer")
        else:
            print(f"[PULSE] Segment not available in buffer, using fallback")
            iq_segment = iq_fallback if iq_fallback is not None else np.array([])
    else:
        iq_segment = iq_fallback if iq_fallback is not None else np.array([])

    try:
        # STRICT_COMPAT: PSK-406 анализ сегмента импульса
        pulse_info = {
            'start_abs': int(start_abs),
            'end_abs': int(end_abs),
            'length_ms': float((end_abs - start_abs) / get_actual_fs() * 1000),
            'timestamp': time.time(),
            'iq_size': int(iq_segment.size) if hasattr(iq_segment, 'size') else 0
        }

        # Выполняем PSK-анализ если сегмент доступен
        if iq_segment.size > 0:
            psk_result = analyze_psk406(iq_segment, get_actual_fs())
            pulse_info.update(psk_result)
        else:
            # Заполняем PSK-поля значениями по умолчанию если сегмент недоступен
            pulse_info.update({
                'bitrate_bps': None,
                'pos_phase': None,
                'neg_phase': None,
                'ph_rise': None,
                'ph_fall': None,
                'symmetry_pct': None,
                'msg_hex': None,
                'msg_ok': None
            })

        # Добавляем в очередь для обработки веб-интерфейсом
        try:
            pulse_queue.put_nowait(pulse_info)
        except queue.Full:
            # Если очередь переполнена, пропускаем старые данные
            try:
                pulse_queue.get_nowait()
                pulse_queue.put_nowait(pulse_info)
            except queue.Empty:
                pass

        last_pulse_data = pulse_info

        # STRICT_COMPAT: Добавляем в историю импульсов
        global pulse_history
        pulse_history.append(pulse_info.copy())

        # STRICT_COMPAT: Используем единую утилиту для обновления STATE
        update_state_from_results(pulse_info)

        # Дополнительная отладочная информация
        if 'phase_data' in pulse_info and pulse_info['phase_data']:
            print(f"[STATE] Using phase data from analyze_psk406: {len(STATE.phase_data)} points")
        else:
            print(f"[STATE] No phase data from analyze_psk406()")

        print(f"[PULSE] Processed: {pulse_info['length_ms']:.1f}ms, PSK: {pulse_info.get('msg_ok', 'N/A')}")

    except Exception as e:
        print(f"[PULSE] Segment processing error: {e}")

def _find_pulse_segment(iq_data, sample_rate, thresh_dbm, win_ms, start_delay_ms, calib_db):
    """
    Поиск импульса по RMS порогу (из test_cf32_to_phase_msg_FFT.py)
    """
    # Вычисляем RMS в dBm
    W = max(1, int(win_ms * 1e-3 * sample_rate))
    p = np.abs(iq_data)**2
    ma = np.convolve(p, np.ones(W)/W, mode="same")
    rms = np.sqrt(ma + 1e-30)
    rms_dbm = 20*np.log10(rms + 1e-30) + calib_db

    # Ищем импульсы пока не найдём достаточно длинный
    while True:
        # Поиск границ импульса
        idx = np.where(rms_dbm > thresh_dbm)[0]
        if idx.size == 0:
            return None

        i0, i1 = idx[0], idx[-1]

        # Проверяем длительность
        dur_ms = (i1 - i0) / sample_rate * 1e3
        if dur_ms < 5.0:
            # Слишком короткий, ищем дальше
            rms_dbm[i0:i1] = -999  # затираем
            continue
        else:
            break

    # Сдвигаем границы с учетом задержки и окна
    i0 = min(i1, i0 + int((start_delay_ms + win_ms) * 1e-3 * sample_rate))
    i1 = max(i0, i1 - int(win_ms * 1e-3 * sample_rate))

    if i1 <= i0:
        return None

    return iq_data[i0:i1]

def process_cf32_file(file_path):
    """
    Обрабатывает CF32 файл используя библиотеки metrics и demod
    Возвращает словарь с результатами для обновления BeaconState
    """
    try:
        # Читаем CF32 файл
        iq_data = np.fromfile(file_path, dtype=np.complex64)

        if len(iq_data) == 0:
            return {"error": "Empty file"}

        # Параметры обработки из test_cf32_to_phase_msg_FFT.py
        sample_rate = 1000000  # 1 MHz
        baseline_ms = 2.0  # PSK_BASELINE_MS - такой же как в FILE режиме из test_cf32
        t0_offset_ms = 0.0

        # Параметры для поиска импульса
        thresh_dbm = -60.0  # порог для поиска импульса
        win_ms = 1.0  # окно RMS
        start_delay_ms = 3.0  # обрезание начала
        calib_db = -30.0  # калибровка

        # Поиск импульса по RMS как в test_cf32_to_phase_msg_FFT.py
        iq_seg = _find_pulse_segment(iq_data, sample_rate, thresh_dbm, win_ms, start_delay_ms, calib_db)

        if iq_seg is None or len(iq_seg) == 0:
            return {"error": "No pulse found"}

        # ДИАГНОСТИКА: Информация о сегменте перед обработкой
        print(f"[FILE-DIAG] Final segment: {len(iq_seg)} samples, baseline_ms={baseline_ms}")
        print(f"[FILE-DIAG] First 5 IQ values: {iq_seg[:5] if len(iq_seg) >= 5 else iq_seg}")
        print(f"[FILE-DIAG] IQ segment statistics: mean_abs={np.mean(np.abs(iq_seg)):.6f}, max_abs={np.max(np.abs(iq_seg)):.6f}")

        # ДИАГНОСТИКА: Информация о сегменте перед обработкой
        print(f"[FILE-DIAG] Final segment: {len(iq_seg)} samples, baseline_ms={baseline_ms}")
        print(f"[FILE-DIAG] First 5 IQ values: {iq_seg[:5] if len(iq_seg) >= 5 else iq_seg}")
        print(f"[FILE-DIAG] IQ segment statistics: mean_abs={np.mean(np.abs(iq_seg)):.6f}, max_abs={np.max(np.abs(iq_seg)):.6f}")

        # Обрабатываем сигнал с помощью metrics
        pulse_result = process_psk_impulse(
            iq_seg=iq_seg,
            fs=sample_rate,
            baseline_ms=baseline_ms,
            t0_offset_ms=t0_offset_ms,
            use_lpf_decim=True,
            remove_slope=True,  # выравниваем фазу по горизонтали
        )

        # Обработка FM для получения частотной девиации
        fm_result = fm_discriminator(
            iq=iq_seg,
            fs=sample_rate,
            pre_lpf_hz=50000,  # антиалиасный фильтр 50 кГц
            decim=4,            # децимация в 4 раза
            smooth_hz=15000,    # сглаживание 15 кГц
            detrend=False,      # отключить удаление линейного тренда
            center=False,       # без центрирования
            fir_taps=127        # длина FIR фильтра
        )

        fm_freq = fm_result.get("freq_hz", [])
        fm_xs = fm_result.get("xs_ms", [])


        if not pulse_result or "phase_rad" not in pulse_result:
            return {"error": "No pulse detected"}

        # Демодуляция PSK сообщения
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(data=pulse_result["phase_rad"])

        # Записываем edges[0] в файл для анализа
        edges_info = f"[FILE] edges[0] = {edges[0] if edges is not None and len(edges) > 0 else 'No edges'}"
        edges_info += f", Using default start_idx=25000, phase_data len = {len(pulse_result['phase_rad'])}"
        with open("../edges_file_debug.txt", "w") as f:
            f.write(edges_info + "\n")
        print(edges_info)

        # Извлекаем метрики из результата
        phase_data = pulse_result.get("phase_rad", [])
        xs_fm_ms = pulse_result.get("xs_ms", [])

        print(f"DEBUG: phase_data length = {len(phase_data) if hasattr(phase_data, '__len__') else 0}")
        print(f"DEBUG: xs_fm_ms length = {len(xs_fm_ms) if hasattr(xs_fm_ms, '__len__') else 0}")
        if isinstance(phase_data, np.ndarray) and phase_data.size > 0:
            print(f"DEBUG: phase_data sample: min={np.min(phase_data):.3f}, max={np.max(phase_data):.3f}")
        if isinstance(xs_fm_ms, np.ndarray) and xs_fm_ms.size > 0:
            print(f"DEBUG: xs_fm_ms sample: min={np.min(xs_fm_ms):.3f}, max={np.max(xs_fm_ms):.3f}")

        # Безопасное преобразование в список
        if isinstance(phase_data, np.ndarray):
            phase_list = phase_data.tolist()
        else:
            phase_list = list(phase_data) if phase_data is not None else []

        if isinstance(xs_fm_ms, np.ndarray):
            xs_list = xs_fm_ms.tolist()
        else:
            xs_list = list(xs_fm_ms) if xs_fm_ms is not None else []

        # Безопасная обработка edges (может быть numpy массивом)
        if edges is not None and hasattr(edges, '__len__') and len(edges) > 0:
            edges_list = edges.tolist() if hasattr(edges, 'tolist') else list(edges)
        else:
            edges_list = []

        result = {
            "success": True,
            "msg_hex": msg_hex if msg_hex else "",
            "phase_data": phase_list,
            "xs_fm_ms": xs_list,
            "fm_data": fm_freq.tolist() if isinstance(fm_freq, np.ndarray) else list(fm_freq) if fm_freq is not None else [],
            "fm_xs_ms": fm_xs.tolist() if isinstance(fm_xs, np.ndarray) else list(fm_xs) if fm_xs is not None else [],
            "edges": edges_list,
            "file_processed": True
        }

        # Если есть метрики фазы, добавляем их
        print(f"DEBUG: phase_res type: {type(phase_res)}")
        print(f"DEBUG: phase_res content: {phase_res}")
        if phase_res is not None and (isinstance(phase_res, dict) or (hasattr(phase_res, '__len__') and len(phase_res) > 0)):
            # Частота дискретизации после децимации (как в test_cf32_to_phase_msg_FFT.py)
            FSd = sample_rate / 4.0

            ph_rise_val = phase_res.get("PhRise", 0.0)
            ph_fall_val = phase_res.get("PhFall", 0.0)

            # Безопасное преобразование numpy значений
            pos_phase = float(phase_res.get("PosPhase", 0.0)) if phase_res.get("PosPhase") is not None else 0.0
            neg_phase = float(phase_res.get("NegPhase", 0.0)) if phase_res.get("NegPhase") is not None else 0.0

            if ph_rise_val is not None and np.isfinite(ph_rise_val) and float(ph_rise_val) != 0:
                ph_rise = float(ph_rise_val / FSd * 1e6)
            else:
                ph_rise = 0.0

            if ph_fall_val is not None and np.isfinite(ph_fall_val) and float(ph_fall_val) != 0:
                ph_fall = float(ph_fall_val / FSd * 1e6)
            else:
                ph_fall = 0.0

            # Вычисляем t_mod для дальнейших расчетов
            t_mod = float(phase_res.get("Tmod", 0.0)) if phase_res.get("Tmod") is not None else 0.0

            # Вычисляем BitRate: FSd / Tmod как в beacon406-plot.py
            FSd = sample_rate / 4.0  # 250000.0
            bitrate_bps = FSd / t_mod if t_mod > 0 else 0.0

            result.update({
                "pos_phase": pos_phase,
                "neg_phase": neg_phase,
                "ph_rise": ph_rise,
                "ph_fall": ph_fall,
                "asymmetry": float(phase_res.get("Ass", 0.0)) if phase_res.get("Ass") is not None else 0.0,
                "t_mod": t_mod,
                "bitrate_bps": bitrate_bps
            })

        # Дополнительные метрики из pulse_result
        if "rms_dbm" in pulse_result:
            result["rms_dbm"] = float(pulse_result["rms_dbm"]) if pulse_result["rms_dbm"] is not None else 0.0
        if "freq_hz" in pulse_result:
            result["freq_hz"] = float(pulse_result["freq_hz"]) if pulse_result["freq_hz"] is not None else 0.0

        # Дополнительные вычисления для Current таблицы
        # Total,ms из временной оси xs_fm_ms
        if isinstance(xs_list, list) and len(xs_list) > 1:
            total_ms = float(xs_list[-1] - xs_list[0])
        else:
            total_ms = 0.0

        # Prise,ms из ph_rise (мкс -> мс)
        prise_ms = ph_rise / 1000.0 if ph_rise > 0 else 0.0

        # Preamble,ms из carrier_ms = edges[0] / FSd * 1e3 как в beacon406-plot.py
        FSd = sample_rate / 4.0  # 250000.0
        if edges_list and len(edges_list) > 0:
            preamble_ms = float(edges_list[0] / FSd * 1e3)
            # Записываем для отладки
            with open("../preamble_file_debug.txt", "w") as f:
                f.write(f"[FILE] preamble_ms = {preamble_ms:.3f} (edges[0]={edges_list[0]}, FSd={FSd})\n")
        else:
            preamble_ms = float(baseline_ms)  # fallback

        # Symmetry,% из asymmetry (дублируем для symmetry_pct)
        symmetry_pct = float(phase_res.get("Ass", 0.0)) if phase_res.get("Ass") is not None else 0.0

        # Добавляем в результат
        result.update({
            "total_ms": total_ms,
            "prise_ms": prise_ms,
            "preamble_ms": preamble_ms,
            "symmetry_pct": symmetry_pct
        })

        return result

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

@dataclass
class BeaconState:
    """Состояние маяка с реалистичными параметрами 406 МГц"""
    running: bool = False
    protocol: str = "N"
    date: str = "01.08.2025"
    conditions: str = "Normal temperature, Idling"               
    beacon_model: str = "Beacon N"
    beacon_frequency: float = 406025000.0
    message: str = "[no message]"
    hex_message: str = ""  # HEX сообщение из загруженного файла
    current_file: str = ""  # Путь к текущему загруженному файлу
    phase_data: list = field(default_factory=list)  # Данные фазы для графика
    xs_fm_ms: list = field(default_factory=list)  # Временная шкала для графика фазы
    fm_data: list = field(default_factory=list)  # Данные FM частоты для графика
    fm_xs_ms: list = field(default_factory=list)  # Временная шкала для FM графика

    # Текущие измерения (пустые до загрузки файла)
    fs1_hz: float = 0.0
    fs2_hz: float = 0.0
    fs3_hz: float = 0.0

    # Фазовые параметры (пустые до загрузки файла)
    phase_pos_rad: float = 0.0
    phase_neg_rad: float = 0.0
    t_rise_mcs: float = 0.0
    t_fall_mcs: float = 0.0

    # Дополнительные фазовые метрики из demod (пустые до загрузки файла)
    pos_phase: float = 0.0
    neg_phase: float = 0.0
    ph_rise: float = 0.0
    ph_fall: float = 0.0
    asymmetry: float = 0.0
    t_mod: float = 0.0
    rms_dbm: float = 0.0
    freq_hz: float = 0.0

    # Дополнительные параметры (пустые до загрузки файла)
    p_wt: float = 0.0
    prise_ms: float = 0.0
    bitrate_bps: float = 0.0
    symmetry_pct: float = 0.0
    preamble_ms: float = 0.0
    total_ms: float = 0.0
    rep_period_s: float = 0.0


STATE = BeaconState()

def update_state_from_results(res: dict) -> None:
    """Единая утилита для обновления STATE из результатов обработки (File/Run).

    STRICT_COMPAT: Безопасные присваивания только если ключ присутствует.
    Это гарантирует паритет между File и Run режимами.

    Поддерживаемые ключи результата:
    - Тайминги: preamble_ms, total_ms, prise_ms
    - Фаза: pos_phase, neg_phase, ph_rise, ph_fall, symmetry_pct, asymmetry
    - Сигнал: rms_dbm, freq_hz, bitrate_bps, t_mod, p_wt
    - Сообщения: hex_message, msg_hex, message
    - Графики: phase_data, xs_ms, xs_fm_ms, fm_data, fm_xs_ms
    - Файлы: current_file

    Args:
        res: Словарь с результатами обработки сигнала
    """
    # Тайминги/границы
    if "preamble_ms" in res: STATE.preamble_ms = res["preamble_ms"]
    if "total_ms" in res: STATE.total_ms = res["total_ms"]
    if "prise_ms" in res: STATE.prise_ms = res["prise_ms"]

    # Фаза и фронты
    if "pos_phase" in res: STATE.pos_phase = res["pos_phase"]
    if "neg_phase" in res: STATE.neg_phase = res["neg_phase"]
    if "ph_rise" in res: STATE.ph_rise = res["ph_rise"]
    if "ph_fall" in res: STATE.ph_fall = res["ph_fall"]
    if "symmetry_pct" in res: STATE.symmetry_pct = res["symmetry_pct"]
    if "asymmetry" in res: STATE.asymmetry = res["asymmetry"]

    # Сигнальные/общие
    if "rms_dbm" in res: STATE.rms_dbm = res["rms_dbm"]
    if "freq_hz" in res: STATE.freq_hz = res["freq_hz"]
    if "bitrate_bps" in res: STATE.bitrate_bps = res["bitrate_bps"]
    if "t_mod" in res: STATE.t_mod = res["t_mod"]
    if "p_wt" in res: STATE.p_wt = res["p_wt"]

    # Декодированное сообщение
    if "hex_message" in res: STATE.hex_message = res["hex_message"]
    if "msg_hex" in res: STATE.hex_message = res["msg_hex"]  # альтернативный ключ
    if "message" in res: STATE.message = res["message"]

    # Графики (если формируются)
    if "phase_data" in res: STATE.phase_data = res["phase_data"]
    if "xs_ms" in res: STATE.xs_ms = res["xs_ms"]  # ФАЗА - исправлено!
    if "xs_fm_ms" in res: STATE.xs_fm_ms = res["xs_fm_ms"]  # альтернативный ключ (deprecated)
    if "fm_data" in res: STATE.fm_data = res["fm_data"]
    if "fm_xs_ms" in res: STATE.fm_xs_ms = res["fm_xs_ms"]  # FM - основное имя

    # Файловые параметры
    if "current_file" in res: STATE.current_file = res["current_file"]

def init_state_fields() -> None:
    """Инициализация всех полей STATE для гарантии их присутствия в /api/status.

    Вызывается при старте приложения для установки начальных значений.
    """
    # Тайминги/границы
    if not hasattr(STATE, 'preamble_ms'): STATE.preamble_ms = None
    if not hasattr(STATE, 'total_ms'): STATE.total_ms = None
    if not hasattr(STATE, 'prise_ms'): STATE.prise_ms = None

    # Фаза и фронты
    if not hasattr(STATE, 'pos_phase'): STATE.pos_phase = None
    if not hasattr(STATE, 'neg_phase'): STATE.neg_phase = None
    if not hasattr(STATE, 'ph_rise'): STATE.ph_rise = None
    if not hasattr(STATE, 'ph_fall'): STATE.ph_fall = None
    if not hasattr(STATE, 'symmetry_pct'): STATE.symmetry_pct = None
    if not hasattr(STATE, 'asymmetry'): STATE.asymmetry = None

    # Сигнальные/общие
    if not hasattr(STATE, 'rms_dbm'): STATE.rms_dbm = None
    if not hasattr(STATE, 'freq_hz'): STATE.freq_hz = None
    if not hasattr(STATE, 'bitrate_bps'): STATE.bitrate_bps = None
    if not hasattr(STATE, 't_mod'): STATE.t_mod = None
    if not hasattr(STATE, 'p_wt'): STATE.p_wt = None

    # Декодированное сообщение
    if not hasattr(STATE, 'hex_message'): STATE.hex_message = ""
    if not hasattr(STATE, 'message'): STATE.message = "[no message]"

    # Графики
    if not hasattr(STATE, 'phase_data'): STATE.phase_data = []
    if not hasattr(STATE, 'xs_ms'): STATE.xs_ms = []  # Ось времени для фазы
    if not hasattr(STATE, 'xs_fm_ms'): STATE.xs_fm_ms = []  # deprecated
    if not hasattr(STATE, 'fm_data'): STATE.fm_data = []
    if not hasattr(STATE, 'fm_xs_ms'): STATE.fm_xs_ms = []  # Ось времени для FM

    # Файловые параметры
    if not hasattr(STATE, 'current_file'): STATE.current_file = ""

# Инициализируем поля при старте
init_state_fields()

# HTML страница с точным воспроизведением дизайна
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COSPAS/SARSAT Beacon Tester v2.1</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f0f0;
            overflow: hidden;
        }

        /* Верхний заголовок */
        .header {
            background: linear-gradient(180deg, #7bb3d9 0%, #5a9bd4 100%);
            color: white;
            padding: 12px 20px;
            font-size: 20px;  /* увеличен с 16px */
            font-weight: bold;
            text-align: center;
            border-bottom: 1px solid #4a8bc2;
        }

        /* Основной контейнер */
        .container {
            display: flex;
            height: calc(100vh - 52px);  /* обновлено с учетом большего заголовка */
            background: #e8e8e8;
        }

        /* Левая панель */
        .left-panel {
            width: 220px;  /* увеличена ширина с 180px */
            background: #d4e6f1;
            border-right: 1px solid #b3d1ed;
            padding: 10px;
        }

        .panel-section {
            margin-bottom: 12px;
        }

        .section-header {
            background: linear-gradient(180deg, #a8c8e4 0%, #7bb3d9 100%);
            color: #2c3e50;
            font-weight: bold;
            font-size: 13px;  /* увеличен с 11px */
            text-align: center;
            padding: 6px;
            border: 1px solid #6699cc;
            border-radius: 3px;
            margin-bottom: 6px;
        }

        .section-content {
            background: white;
            border: 1px solid #b3d1ed;
            border-radius: 3px;
            padding: 8px;
            font-size: 13px;  /* увеличен с 11px */
        }

        .radio-group label {
            display: block;
            margin: 4px 0;  /* увеличен отступ */
            cursor: pointer;
            font-size: 13px;  /* добавлен размер шрифта */
        }

        .control-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 3px 0;
        }

        .control-input {
            width: 50px;  /* увеличена ширина */
            font-size: 13px;  /* увеличен с 10px */
            padding: 3px 5px;
            border: 1px solid #ccc;
        }

        .radio-inline {
            display: flex;
            gap: 8px;
        }

        .button {
            background: linear-gradient(180deg, #e8f4f8 0%, #d1e7f0 100%);
            border: 1px solid #a8c8e4;
            border-radius: 3px;
            padding: 6px 12px;  /* увеличены отступы */
            font-size: 13px;  /* увеличен с 11px */
            cursor: pointer;
            margin: 3px;
        }

        .button:hover {
            background: linear-gradient(180deg, #f0f8ff 0%, #e0f0f8 100%);
        }

        .button.primary {
            background: linear-gradient(180deg, #7bb3d9 0%, #5a9bd4 100%);
            color: white;
            border-color: #4a8bc2;
        }

        .button.danger {
            background: linear-gradient(180deg, #f8d7da 0%, #f1aeb5 100%);
            color: #721c24;
            border-color: #f1aeb5;
        }

        /* Центральная область */
        .center-panel {
            flex: 1;
            background: #f8f9fa;
            padding: 8px;
            overflow: hidden;
        }

        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            margin-bottom: 8px;
        }

        .info-row-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 8px;
        }

        .info-block {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 6px 10px;  /* увеличены отступы */
            font-size: 14px;  /* увеличен с 11px */
        }

        .info-label {
            font-weight: bold;
            color: #495057;
        }

        .info-value {
            color: #212529;
        }

        .beacon-title {
            font-weight: bold;
            margin: 10px 0 6px 0;
            color: #495057;
            font-size: 16px;  /* добавлен размер шрифта */
        }

        .message-line {
            font-size: 14px;  /* увеличен с 11px */
            color: #6c757d;
            margin-bottom: 10px;
        }

        /* График */
        .chart-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 8px;
            position: relative;
            height: 380px;
            margin-bottom: 8px;
        }

        #phaseChart {
            width: 100%;
            height: 100%;
            background: white;
        }

        .phase-values {
            display: flex;
            justify-content: space-around;
            font-size: 13px;  /* увеличен с 10px */
            color: #495057;
            margin-top: 6px;
        }

        .chart-title {
            text-align: center;
            font-size: 14px;  /* увеличен с 11px */
            color: #6c757d;
            margin-top: 6px;
        }

        /* Правая панель */
        .right-panel {
            width: 320px;  /* увеличена ширина с 280px */
            background: #f8f9fa;
            border-left: 1px solid #dee2e6;
            padding: 10px;
        }

        .stats-header {
            background: linear-gradient(180deg, #a8c8e4 0%, #7bb3d9 100%);
            color: #2c3e50;
            font-weight: bold;
            font-size: 14px;  /* увеличен с 11px */
            text-align: center;
            padding: 6px;
            border: 1px solid #6699cc;
            border-radius: 3px;
            margin-bottom: 10px;
        }

        .stats-content {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 6px;
        }

        .stat-row {
            display: flex;
            justify-content: space-between;
            font-size: 13px;  /* увеличен с 11px */
            padding: 4px 0;  /* увеличен отступ */
            border-bottom: 1px solid #f8f9fa;
        }

        .stat-row:last-child {
            border-bottom: none;
        }

        .stat-label {
            color: #495057;
            font-weight: 500;
        }

        .stat-value {
            color: #212529;
            font-weight: normal;
            font-family: 'Courier New', monospace;
            font-size: 13px;  /* добавлен размер шрифта */
        }

        /* Стили для HTML таблицы Message */
        .message-table-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 0;
            position: relative;
            height: 494px;
            margin-bottom: 8px;
            overflow-y: auto;
        }

        .message-table-header {
            background: #1976D2;
            color: white;
            padding: 12px;
            text-align: center;
        }

        .message-table-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: bold;
        }

        .message-table-header .hex-info {
            margin: 4px 0 0 0;
            font-size: 11px;
        }

        .message-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .message-table th {
            background: #E0E0E0;
            color: #333;
            font-weight: bold;
            font-size: 15px;
            padding: 8px 5px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .message-table td {
            padding: 6px 5px;
            border-bottom: 0.5px solid #DDD;
            color: #333;
            font-size: 14px;
        }

        .message-table tr:nth-child(even) {
            background: #F5F5F5;
        }

        .message-table tr:nth-child(odd) {
            background: white;
        }

        .message-table .binary-content {
            font-family: monospace;
            font-size: 13px;
            color: #0066CC;
        }

        .message-table .field-name {
            font-weight: bold;
        }

        .message-table-footer {
            padding: 8px 12px;
            font-size: 11px;
            color: #666;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }

        /* Стили для HTML таблицы 121 Data */
        .data121-table-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 0;
            position: relative;
            height: 494px;
            margin-bottom: 8px;
            overflow-y: auto;
        }

        .data121-table-header {
            background: #5a9bd4;
            color: white;
            padding: 12px;
            text-align: center;
        }

        .data121-table-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: bold;
        }

        .data121-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .data121-table td {
            padding: 8px 10px;
            border: 1px solid #999;
            color: #333;
            font-size: 14px;
        }

        .data121-table .param-name {
            width: 1%;
            white-space: nowrap;
        }

        .data121-table .param-name {
            font-weight: bold;
        }

        .data121-table .param-value {
            font-weight: normal;
        }

        .data121-table-footer {
            padding: 8px 20px;
            font-size: 11px;
            color: #666;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }

        /* Стили для HTML таблицы Sum Table */
        .sum-table-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 0;
            position: relative;
            height: 494px;
            margin-bottom: 8px;
            overflow-y: auto;
        }

        .sum-params-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-bottom: 10px;
        }

        .sum-params-table .header-406 {
            background: #5a9bd4;
            color: white;
            text-align: center;
            font-weight: bold;
            padding: 8px;
            font-size: 16px;
        }

        .sum-params-table .header-121 {
            background: #5a9bd4;
            color: white;
            text-align: center;
            font-weight: bold;
            padding: 8px;
            font-size: 16px;
        }

        .sum-params-table .subheader {
            background: #87CEEB;
            color: white;
            text-align: center;
            padding: 6px;
            font-size: 14px;
            font-weight: bold;
        }

        .sum-params-table .subheader-empty {
            background: #87CEEB;
            color: white;
            text-align: center;
            padding: 6px;
            font-size: 14px;
            font-weight: bold;
            width: 1%;
        }

        .sum-params-table .param-row {
            background: white;
        }

        .sum-params-table .param-row:nth-child(even) {
            background: #f9f9f9;
        }

        .sum-params-table td {
            padding: 6px 5px;
            border: 1px solid #ccc;
            font-size: 14px;
            text-align: center;
        }

        .sum-params-table .param-name {
            text-align: right;
            font-weight: bold;
            background: #f0f0f0;
            width: 1%;
            white-space: nowrap;
        }

        .sum-message-section {
            margin-top: 10px;
            border-top: 2px solid #5a9bd4;
        }

        .sum-message-header {
            background: #5a9bd4;
            color: white;
            text-align: center;
            padding: 8px;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        COSPAS/SARSAT Beacon Tester
    </div>

    <div class="container">
        <!-- Левая панель -->
        <div class="left-panel">
            <div class="panel-section">
                <div class="section-header">VIEW</div>
                <div class="section-content">
                    <div class="radio-group">
                        <label><input type="radio" name="view" value="phase" checked onchange="changeView('phase')"> 406 Phase</label>
                        <label><input type="radio" name="view" value="fr_stability" onchange="changeView('fr_stability')"> 406 Fr. stability</label>
                        <label><input type="radio" name="view" value="ph_rise_fall" onchange="changeView('ph_rise_fall')"> 406 Ph/Rise/Fall</label>
                        <label><input type="radio" name="view" value="fr_pwr" onchange="changeView('fr_pwr')"> 406 Fr/Pwr</label>
                        <label><input type="radio" name="view" value="inburst_fr" onchange="changeView('inburst_fr')"> 406 Inburst fr</label>
                        <label><input type="radio" name="view" value="sum_table" onchange="changeView('sum_table')"> 406 Sum table</label>
                        <label><input type="radio" name="view" value="message" onchange="changeView('message')"> 406 Message</label>
                        <label><input type="radio" name="view" value="121_data" onchange="changeView('121_data')"> 121 Data</label>
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <div class="section-header">MODE</div>
                <div class="section-content">
                    <div class="control-row">
                        <span>Time scale</span>
                        <select class="control-input" id="timeScale" onchange="onTimeScaleChange()" style="width: 60px;">
                            <option value="1">1%</option>
                            <option value="2">2%</option>
                            <option value="5">5%</option>
                            <option value="10" selected>10%</option>
                            <option value="20">20%</option>
                            <option value="50">50%</option>
                            <option value="100">100%</option>
                        </select>
                    </div>
                    <div class="control-row">
                        <span>Update</span>
                        <div class="radio-inline">
                            <label><input type="radio" name="update" checked> ON</label>
                            <label><input type="radio" name="update"> OFF</label>
                        </div>
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <div class="section-header">FILE</div>
                <div class="section-content">
                    <button class="button" onclick="loadFile()">File</button>
                    <button class="button" onclick="saveFile()">Save</button>
                    <input type="file" id="fileInput" accept=".cf32" style="display: none;" onchange="uploadFile(this)">
                </div>
            </div>

            <div class="panel-section">
                <div class="section-header">TESTER</div>
                <div class="section-content">
                    <button class="button" onclick="measure()">Measure</button>
                    <br>
                    <button class="button primary" onclick="runTest()">Run</button>
                    <button class="button" onclick="contTest()">Cont</button>
                    <button class="button danger" onclick="breakTest()">Break</button>
                    <div style="margin-top: 10px;">
                        <div style="font-size: 12px; color: #6c757d; margin-bottom: 4px;">SDR Status:</div>
                        <div id="sdrStatus" style="font-family: 'Courier New', monospace; font-size: 11px;
                                                   background: #f8f9fa; border: 1px solid #dee2e6;
                                                   border-radius: 3px; padding: 4px 6px; min-height: 20px;
                                                   color: #495057; word-break: break-all;">
                            No SDR detected
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Центральная панель -->
        <div class="center-panel">
            <div class="info-grid">
                <div class="info-block">
                    <div class="info-label">Protocol</div>
                    <div class="info-value" id="protocol">N</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Date</div>
                    <div class="info-value" id="date">01.08.2025</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Conditions</div>
                    <div class="info-value">
                        <a href="#" style="color: #007bff; text-decoration: underline;">Normal temperature, Idling</a>
                    </div>
                </div>
            </div>

            <div class="info-row-2">
                <div class="info-block">
                    <div class="info-label">Beacon Model</div>
                    <div class="info-value" id="beaconModel">Beacon N</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Beacon Frequency</div>
                    <div class="info-value" id="beaconFreq">406025000.0</div>
                </div>
            </div>

            <div class="beacon-title">Beacon 406</div>
            <div class="message-line">Message: <span id="message">[no message]</span></div>

            <div class="chart-container">
                <canvas id="phaseChart"></canvas>
            </div>

            <div class="phase-values">
                <span>Phase+ = <span id="phasePlus">-63.31</span>°</span>
                <span>TRise+ = <span id="tRise">-59.9</span> mcs</span>
                <span>Phase- = <span id="phaseMinus">-64.73</span>°</span>
                <span>TFall- = <span id="tFall">-121.4</span> mcs</span>
            </div>

            <div class="chart-title" id="chartTitle">Fig.8 Phase</div>
        </div>

        <!-- Правая панель -->
        <div class="right-panel">
            <div class="stats-header">Current</div>
            <div class="stats-content" id="statsContent">
                <!-- Статистика будет заполняться JavaScript -->
            </div>
        </div>
    </div>

    <script>
        console.log('=== BEACON TESTER v2.1 LOADED ===');
        let canvas = document.getElementById('phaseChart');
        let ctx = canvas.getContext('2d');
        let currentView = 'phase';
        let currentTimeScale = 10; // Текущий масштаб времени в процентах (по умолчанию 10%)
        const MESSAGE_DURATION_MS = 440; // Длительность сообщения в миллисекундах
        const PHASE_START_OFFSET_MS = 0; // Начинаем отображение графика с 0мс (без смещения)

        function resizeCanvas() {
            // Убеждаемся, что canvas существует
            const currentCanvas = document.getElementById('phaseChart');
            if (currentCanvas) {
                canvas = currentCanvas;  // Обновляем глобальную ссылку
                ctx = canvas.getContext('2d');  // Обновляем контекст
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
            }
        }

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function changeView(viewType) {
            console.log('=== DEBUG: changeView called with:', viewType);
            currentView = viewType;
            console.log('=== DEBUG: currentView set to:', currentView);

            // Восстанавливаем canvas если он был заменен HTML таблицей
            const chartContainer = document.querySelector('.chart-container');
            if (!chartContainer.querySelector('#phaseChart')) {
                chartContainer.innerHTML = '<canvas id="phaseChart"></canvas>';
                // Обновляем глобальные ссылки на canvas
                canvas = document.getElementById('phaseChart');
                ctx = canvas.getContext('2d');
                resizeCanvas(); // Переинициализируем размеры canvas
            }

            // Очищаем canvas при переключении режимов
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const titleEl = document.getElementById('chartTitle');
            const phaseValuesEl = document.querySelector('.phase-values');

            switch(viewType) {
                case 'phase':
                    titleEl.textContent = 'Fig.8 Phase';
                    phaseValuesEl.innerHTML = `
                        <span>Phase+ = <span id="phasePlus">-63.31</span>°</span>
                        <span>TRise+ = <span id="tRise">-59.9</span> mcs</span>
                        <span>Phase- = <span id="phaseMinus">-64.73</span>°</span>
                        <span>TFall- = <span id="tFall">-121.4</span> mcs</span>
                    `;
                    break;
                case 'ph_rise_fall':
                    titleEl.textContent = 'Fig.7 Rise and Fall Times';
                    phaseValuesEl.innerHTML = `
                        <span>TRise = <span id="tRise">59.9</span> mcs</span>
                        <span>TFall = <span id="tFall">121.4</span> mcs</span>
                    `;
                    break;
                case 'fr_stability':
                    titleEl.textContent = 'Fig.6 Frequency Stability';
                    phaseValuesEl.innerHTML = `
                        <span>FS1 = <span id="fs1">406025864.0</span> Hz</span>
                        <span>FS2 = <span id="fs2">406025864.2</span> Hz</span>
                        <span>FS3 = <span id="fs3">406012489.9</span> Hz</span>
                    `;
                    break;
                case 'fr_pwr':
                    titleEl.textContent = 'Fig.9 Frequency/Power';
                    phaseValuesEl.innerHTML = `
                        <span>Freq = <span id="freq">406.025</span> MHz</span>
                        <span>Power = <span id="power">0.572</span> Wt</span>
                    `;
                    break;
                case 'inburst_fr':
                    titleEl.textContent = 'Fig.10 Inburst Frequency';
                    phaseValuesEl.innerHTML = `
                        <span>BitRate = <span id="bitrate">400.318</span> bps</span>
                        <span>Symmetry = <span id="symmetry">4.049</span> %</span>
                    `;
                    break;
                case 'sum_table':
                    titleEl.textContent = 'Summary Table';
                    phaseValuesEl.innerHTML = '<span>See Current panel for all values</span>';
                    break;
                case 'message':
                    titleEl.textContent = 'EPIRB/ELT Beacon Message Decoder';
                    phaseValuesEl.innerHTML = '<span>Decoded COSPAS-SARSAT 406 MHz beacon message</span>';
                    break;
                case '121_data':
                    titleEl.textContent = '121.5 MHz Transmitter Parameters';
                    phaseValuesEl.innerHTML = '<span>121.5 MHz Emergency Locator Transmitter Data</span>';
                    break;
            }
            // Всегда вызываем fetchData() для обновления - специальные режимы обрабатываются внутри fetchData()
            fetchData();
        }

        // Функция расчета смещения в зависимости от масштаба
        function getOffsetForScale(scale) {
            switch(scale) {
                case 1: return 1;  // 1% -> -1ms
                case 2: return 2;  // 2% -> -2ms
                case 5: return 4;  // 5% -> -4ms
                case 10:
                case 20:
                case 50: return 5; // 10%-50% -> -5ms
                default: return 5; // fallback
            }
        }

        function onTimeScaleChange() {
            const timeScaleSelect = document.getElementById('timeScale');
            currentTimeScale = parseInt(timeScaleSelect.value);
            console.log('Time scale changed to:', currentTimeScale + '%');

            // Перерисовываем график с новым масштабом если данные загружены
            if (currentView === 'phase' || currentView === 'inburst_fr') {
                fetchData(); // Обновляем отображение с новым масштабом
            }
        }

        
        function drawChart(data) {
            console.log('DEBUG: drawChart called with currentView:', currentView);

            // Убеждаемся, что работаем с актуальным canvas
            const currentCanvas = document.getElementById('phaseChart');
            if (currentCanvas) {
                canvas = currentCanvas;
                ctx = canvas.getContext('2d');
            }
            console.log('DEBUG: data object:', data);
            console.log('DEBUG: data.hex_message:', data ? data.hex_message : 'no data');
            console.log('DEBUG: data.phase_data type/length:', data ? typeof data.phase_data + '/' + (data.phase_data ? data.phase_data.length : 'null') : 'no data');
            console.log('DEBUG: data.xs_fm_ms type/length:', data ? typeof data.xs_fm_ms + '/' + (data.xs_fm_ms ? data.xs_fm_ms.length : 'null') : 'no data');
            if (currentView === 'ph_rise_fall') {
                console.log('DEBUG: Drawing ph_rise_fall chart');
                drawRiseFallChart(data);
                return;
            } else if (currentView === 'fr_stability') {
                console.log('DEBUG: Drawing frequency chart');
                drawFrequencyChart(data);
                return;
            } else if (currentView === 'fr_pwr') {
                console.log('DEBUG: Drawing frequency/power charts');
                drawFrequencyPowerChart(data);
                return;
            } else if (currentView === '121_data') {
                console.log('DEBUG: Drawing 121_data table');
                draw121DataTable(data);
                return;
            } else if (currentView === 'message') {
                console.log('DEBUG: Drawing message table');
                // Используем HEX сообщение из загруженного файла
                const hexFromFile = data.hex_message || '';
                console.log('DEBUG: Using hex_message from file:', hexFromFile);
                drawMessageTable(hexFromFile);
                return;
            } else if (currentView === 'inburst_fr') {
                console.log('DEBUG: Drawing FM chart for inburst_fr');
                drawFMChart(data);
                return;
            } else if (currentView === 'fr_stability') {

                // Показываем заглушку
                ctx.fillStyle = '#666';
                ctx.font = '16px Arial';
                ctx.fillText('View mode: ' + currentView, width/2 - 80, height/2);
                ctx.font = '12px Arial';
                ctx.fillText('Chart implementation pending', width/2 - 80, height/2 + 25);
                return;
            }

            // Оригинальный график фазы (только для режима 'phase')
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            // Сетка
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии сетки
            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии сетки с временными метками
            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';  // увеличен с 10px
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();

                // Временные метки с учетом масштаба
                const scaledDuration = MESSAGE_DURATION_MS * (currentTimeScale / 100);
                let startOffset;
                if (currentTimeScale === 100) {
                    startOffset = 0; // При 100% начинаем с 0мс
                } else {
                    // Для других масштабов: начало модуляции - смещение в зависимости от масштаба
                    const preambleMs = data?.preamble_ms || 10.0; // fallback к baseline_ms
                    const offsetMs = getOffsetForScale(currentTimeScale);
                    startOffset = Math.max(0, preambleMs - offsetMs);
                }
                const timeMs = (startOffset + i * scaledDuration / 8).toFixed(1);
                ctx.fillText(timeMs, x - 10, height - 5);
            }

            // Нулевая линия
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();

            // Пунктирные линии на уровне ±1.1 радиан
            ctx.strokeStyle = '#FF0000'; // Красные тонкие линии
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]); // Пунктир: 5px линия, 5px пропуск

            // Линия +1.1 рад (фиксированный масштаб 1.25 рад)
            const y_plus_1_1 = height / 2 - (1.1 / 1.25) * (height / 2);
            ctx.beginPath();
            ctx.moveTo(0, y_plus_1_1);
            ctx.lineTo(width, y_plus_1_1);
            ctx.stroke();

            // Линия -1.1 рад
            const y_minus_1_1 = height / 2 + (1.1 / 1.25) * (height / 2);
            ctx.beginPath();
            ctx.moveTo(0, y_minus_1_1);
            ctx.lineTo(width, y_minus_1_1);
            ctx.stroke();

            // Возвращаем сплошную линию для дальнейшего рисования
            ctx.setLineDash([]);

            // Y-axis labels (будут обновлены после определения масштаба)
            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';

            // График данных
            if (data) {
                let phaseData = (data.phase_data || []).map(v => Number(v));
                let xsData = (data.xs_fm_ms || []).map(v => Number(v));

                // Нормализация времени: если максимум < 10, значит это секунды → переводим в мс
                if (xsData.length && xsData.reduce((max, v) => Math.max(max, v), -Infinity) <= 10) {
                    console.warn('DEBUG: xs_fm_ms appears to be in seconds — converting to ms');
                    xsData = xsData.map(v => v * 1000);
                }

                console.log('DEBUG drawChart: phaseData length =', phaseData.length);
                console.log('DEBUG drawChart: phaseData type check:', typeof phaseData[0], phaseData[0]);
                console.log('DEBUG drawChart: xsData type check:', typeof xsData[0], xsData[0]);

                // Проверяем типы данных и конвертируем при необходимости
                for (let i = 0; i < Math.min(5, phaseData.length); i++) {
                    if (typeof phaseData[i] !== 'number') {
                        console.warn(`DEBUG: phaseData[${i}] is not a number:`, typeof phaseData[i], phaseData[i]);
                        phaseData[i] = parseFloat(phaseData[i]) || 0;
                    }
                    if (typeof xsData[i] !== 'number') {
                        console.warn(`DEBUG: xsData[${i}] is not a number:`, typeof xsData[i], xsData[i]);
                        xsData[i] = parseFloat(xsData[i]) || 0;
                    }
                }
                console.log('DEBUG drawChart: xsData length =', xsData.length);
                if (phaseData.length > 0) {
                    console.log('DEBUG drawChart: phaseData sample =', phaseData.slice(0, 5));
                }
                if (xsData.length > 0) {
                    console.log('DEBUG: xsData sample =', xsData.slice(0, 5));
                }

                console.log('DEBUG: Checking if can draw graph:', phaseData ? `phaseData.length=${phaseData.length}` : 'phaseData is null/undefined');

                if (phaseData && phaseData.length > 1) {
                    console.log('DEBUG: Starting to draw REAL phase graph with', phaseData.length, 'points');

                    // Фиксированный масштаб оси Y: ±1.25 радиан
                    const phaseScale = 1.25;

                    console.log(`DEBUG: Using fixed phase scale: ±${phaseScale} rad`);

                    // Обновляем Y-axis метки с фиксированным масштабом
                    ctx.fillStyle = '#6c757d';
                    ctx.font = '12px Arial';
                    ctx.fillText(`+${phaseScale.toFixed(2)} rad`, 5, 15);
                    ctx.fillText('0', 5, height / 2 + 4);
                    ctx.fillText(`-${phaseScale.toFixed(2)} rad`, 5, height - 10);

                    ctx.strokeStyle = '#0066FF'; // Синий для реального графика фазы
                    ctx.lineWidth = 2;
                    ctx.beginPath();

                    // Правильная логика на основе времени
                    console.log('DEBUG: Drawing time-based graph');

                    // Проверяем наличие временных данных
                    if (!xsData || xsData.length === 0 || xsData.length !== phaseData.length) {
                        console.warn('DEBUG: Missing or mismatched time data (xsData), falling back to index-based drawing');
                        const pointsToShow = Math.min(100, phaseData.length);
                        let minY = Infinity, maxY = -Infinity;
                        for (let i = 0; i < pointsToShow; i++) {
                            const x = (i / pointsToShow) * width;
                            const y = height / 2 - (phaseData[i] / phaseScale) * (height / 2);
                            minY = Math.min(minY, y);
                            maxY = Math.max(maxY, y);
                            if (i === 0) {
                                ctx.moveTo(x, y);
                            } else {
                                ctx.lineTo(x, y);
                            }
                        }
                        console.log(`DEBUG fallback: Drew ${pointsToShow} points`);
                        ctx.stroke();
                        return;
                    }

                    // Временное окно для отображения
                    let windowStart;
                    if (currentTimeScale === 100) {
                        windowStart = 0; // При 100% начинаем с 0мс
                    } else {
                        // Для других масштабов: начало модуляции - смещение в зависимости от масштаба
                        const preambleMs = data.preamble_ms || 10.0; // fallback к baseline_ms из кода
                        const offsetMs = getOffsetForScale(currentTimeScale);
                        windowStart = Math.max(0, preambleMs - offsetMs);
                    }
                    const windowDuration = MESSAGE_DURATION_MS * (currentTimeScale / 100.0); // 440 * scale/100
                    const windowEnd = windowStart + windowDuration;

                    console.log(`DEBUG time window: start=${windowStart}ms, duration=${windowDuration}ms, end=${windowEnd}ms, scale=${currentTimeScale}%`);

                    // Фильтруем точки по временному окну
                    const filteredPoints = [];
                    for (let i = 0; i < phaseData.length; i++) {
                        const timeMs = xsData[i];
                        if (Number.isFinite(timeMs) && timeMs >= windowStart && timeMs <= windowEnd) {
                            filteredPoints.push({
                                time: timeMs,
                                phase: phaseData[i],
                                index: i
                            });
                        }
                    }

                    console.log(`DEBUG: Filtered ${filteredPoints.length} points from ${phaseData.length} total points in time range [${windowStart}, ${windowEnd}]ms`);

                    if (filteredPoints.length === 0) {
                        console.warn('DEBUG: No points found in specified time window - falling back to index-based drawing');
                        const pointsToShow = Math.min(100, phaseData.length);
                        let minY = Infinity, maxY = -Infinity;
                        for (let i = 0; i < pointsToShow; i++) {
                            const x = (i / pointsToShow) * width;
                            const y = height / 2 - (phaseData[i] / phaseScale) * (height / 2);
                            minY = Math.min(minY, y);
                            maxY = Math.max(maxY, y);
                            if (i === 0) {
                                ctx.moveTo(x, y);
                            } else {
                                ctx.lineTo(x, y);
                            }
                        }
                        ctx.stroke();
                        console.log(`DEBUG fallback: Drew ${pointsToShow} points`);
                        return;
                    }

                    // Даунсэмплинг до ~1000 точек для производительности (основное исправление)
                    const targetPoints = 1000; // Настраиваемое значение; достаточно для плавного графика при любом размере холста
                    const step = Math.max(1, Math.ceil(filteredPoints.length / targetPoints));
                    console.log(`DEBUG: Downsampling with step=${step} (target ~${targetPoints} points, actual ~${Math.floor(filteredPoints.length / step)})`);

                    let minY = Infinity, maxY = -Infinity;
                    let firstPoint = true;

                    for (let j = 0; j < filteredPoints.length; j += step) {
                        const point = filteredPoints[j];

                        // Преобразуем время в пиксели
                        const normalizedTime = (point.time - windowStart) / windowDuration;
                        const x = normalizedTime * width;
                        const y = height / 2 - (point.phase / phaseScale) * (height / 2);

                        minY = Math.min(minY, y);
                        maxY = Math.max(maxY, y);

                        if (firstPoint) {
                            ctx.moveTo(x, y);
                            console.log(`DEBUG line start: x=${x.toFixed(1)}, y=${y.toFixed(1)}, time=${point.time.toFixed(1)}ms, value=${point.phase.toFixed(8)}`);
                            firstPoint = false;
                        } else {
                            ctx.lineTo(x, y);
                        }

                        // Логируем первые 5 точек
                        if (j / step < 5) {
                            console.log(`DEBUG line point ${j / step}: x=${x.toFixed(1)}, y=${y.toFixed(1)}, time=${point.time.toFixed(1)}ms, value=${point.phase.toFixed(8)}, normalized_time=${normalizedTime.toFixed(4)}`);
                        }
                    }

                    console.log(`DEBUG line coordinates: minY=${minY.toFixed(1)}, maxY=${maxY.toFixed(1)}, range=${(maxY-minY).toFixed(1)}`);
                    console.log(`DEBUG phaseScale used: ${phaseScale.toFixed(6)}`);
                    ctx.stroke();
                    console.log('DEBUG: Time-based phase graph drawing completed');
                } else {
                    console.log('DEBUG: Phase graph NOT drawn - insufficient data or condition not met');
                }
            }
        }       
        

        function showMessageTable(hexMessage) {
            console.log('DEBUG: showMessageTable called with:', hexMessage);

            // Очищаем canvas и показываем HTML таблицу
            const canvas = document.getElementById('phaseChart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Создаем контейнер для HTML таблицы
            const chartContainer = document.querySelector('.chart-container');

            if (!hexMessage || hexMessage.trim() === '') {
                chartContainer.innerHTML = `
                    <div class="message-table-container">
                        <div style="text-align: center; padding: 50px; color: #666;">
                            <h3>No message decoded</h3>
                            <p>Please load a .cf32 file using the File button</p>
                            <p>to decode EPIRB/ELT beacon message</p>
                        </div>
                    </div>
                `;
                return;
            }

            // Данные таблицы (те же что были в canvas версии)
            const tableData = [
                ['1-15', '111111111111111', 'Bit-sync pattern', 'Valid'],
                ['16-24', '000101111', 'Frame-sync pattern', 'Normal Operation'],
                ['25', '1', 'Format Flag', 'Long Format'],
                ['26', '0', 'Protocol Flag', 'Standard/National/RLS'],
                ['27-36', '1000000000', 'Country Code', '512 - Russia'],
                ['37-40', '0000', 'Protocol Code', 'Avionic'],
                ['41-64', '000000100000000000000000', 'Test Data', '0x020000'],
                ['65-74', '0111111111', 'Latitude (PDF-1)', 'Default value'],
                ['75-85', '01111111111', 'Longitude (PDF-1)', 'Default value'],
                ['86-106', '110000100000101101111', 'BCH PDF-1', '0x1820B7'],
                ['107-110', '1000', 'Fixed (1101)', 'Invalid (1000)'],
                ['111', '0', 'Position source', 'External/Unknown'],
                ['112', '0', '121.5 MHz Device', 'Not included'],
                ['113-122', '1111100000', 'Latitude (PDF-2)', 'bin 1111100000'],
                ['123-132', '1111100000', 'Longitude (PDF-2)', 'bin 1111100000'],
                ['133-144', '111001101100', 'BCH PDF-2', '0xE6C']
            ];

            // Создаем HTML таблицу
            let tableHtml = `
                <div class="message-table-container">
                    <div class="message-table-header">
                        <h3>EPIRB/ELT Beacon Message Decoder</h3>
                    </div>
                    <table class="message-table">
                        <thead>
                            <tr>
                                <th style="width: 90px;">Bit Range</th>
                                <th style="width: 200px;">Binary Content</th>
                                <th style="width: 220px;">Field Name</th>
                                <th>Decoded Value</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            for (let i = 0; i < tableData.length; i++) {
                const row = tableData[i];
                tableHtml += `
                    <tr>
                        <td>${row[0]}</td>
                        <td class="binary-content">${row[1]}</td>
                        <td class="field-name">${row[2]}</td>
                        <td>${row[3]}</td>
                    </tr>
                `;
            }

            tableHtml += `
                        </tbody>
                    </table>
                    <div class="message-table-footer">
                        <div>COSPAS-SARSAT 406 MHz Beacon Message (144 bits)</div>
                        <div>Protocol: Long Format, Standard Location</div>
                    </div>
                </div>
            `;

            chartContainer.innerHTML = tableHtml;
        }

        function showSumTable(data) {
            console.log('DEBUG: showSumTable called');

            // Очищаем canvas и показываем HTML таблицу
            const canvas = document.getElementById('phaseChart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Создаем контейнер для HTML таблицы
            const chartContainer = document.querySelector('.chart-container');

// Извлекаем реальные данные из объекта data для колонки Current
            const freq_khz = data.freq_hz ? (data.freq_hz / 1000 + 406000).toFixed(3) : '0.000';
            const pos_phase = data.phase_pos_rad ? data.phase_pos_rad.toFixed(2) : '0.00';
            const neg_phase = data.phase_neg_rad ? data.phase_neg_rad.toFixed(2) : '0.00';
            const t_rise = data.t_rise_mcs ? data.t_rise_mcs.toFixed(2) : '0.00';
            const t_fall = data.t_fall_mcs ? data.t_fall_mcs.toFixed(2) : '0.00';
            const power_wt = data.p_wt ? data.p_wt.toFixed(2) : '0.00';
            const prise_ms = data.prise_ms ? data.prise_ms.toFixed(2) : '0.00';
            const bitrate = data.bitrate_bps ? data.bitrate_bps.toFixed(2) : '0.00';
            const asymmetry = data.symmetry_pct ? data.symmetry_pct.toFixed(2) : '0.00';
            const preamble = data.preamble_ms ? data.preamble_ms.toFixed(2) : '0.00';
            const total_duration = data.total_ms ? data.total_ms.toFixed(2) : '0.00';
            const rep_period = data.rep_period_s ? data.rep_period_s.toFixed(2) : '0.00';

            // Данные для 406 MHz таблицы с реальными значениями в колонке Current
            const params406 = [
                ['Frequency, kHz', '40600.000', '40600.000', '0.000', freq_khz, '0.000'],
                ['+Phase deviation, rad', '1.00', '1.20', '0.00', pos_phase, '0.00'],
                ['-Phase deviation, rad', '-1.00', '-1.20', '0.00', neg_phase, '0.00'],
                ['Phase time rise, mcs', '50.00', '250.00', '0.00', t_rise, '0.00'],
                ['Phase time fall, mcs', '50.00', '250.00', '0.00', t_fall, '0.00'],
                ['Power, Wt', '3.16', '7.94', '0.00', power_wt, '0.00'],
                ['Power rise, ms', '0.00', '0.00', '0.00', prise_ms, '0.00'],
                ['Bit Rate, bps', '396.00', '404.00', '0.00', bitrate, '0.00'],
                ['Asymmetry, %', '0.00', '5.00', '0.00', asymmetry, '0.00'],
                ['CW Preamble, ms', '158.40', '161.60', '0.00', preamble, '0.00'],
                ['Total burst duration, ms', '435.60', '520.00', '0.00', total_duration, '0.00'],
                ['Repetition period, s', '47.50', '52.50', '0.00', rep_period, '0.00'],
                ['Delta Rep. period, s', '4.00', '0.00', '0.00', '0.00', '0.00']
            ];

            // Данные для 121.5 MHz таблицы
            const params121 = [
                ['Carrier Frequency, Hz', '0'],
                ['Power, mW', '0.0'],
                ['Sweep Period, sec', '0.0'],
                ['Modulation Index, %', '0']
            ];

            // Создаем HTML - единый контейнер со всеми таблицами
            let tableHtml = `
                <div class="sum-table-container">
                    <table class="sum-params-table">
                        <tr>
                            <td colspan="6" class="header-406">406 MHz Transmitter Parameters</td>
                        </tr>
                        <tr>
                            <td rowspan="2" class="subheader-empty"></td>
                            <td colspan="2" class="subheader">Limits</td>
                            <td rowspan="2" class="subheader" style="width: 80px;">min</td>
                            <td rowspan="2" class="subheader" style="width: 100px;">Current</td>
                            <td rowspan="2" class="subheader" style="width: 100px;">Measured<br/>Info</td>
                        </tr>
                        <tr>
                            <td class="subheader" style="width: 80px;">min</td>
                            <td class="subheader" style="width: 80px;">max</td>
                        </tr>
            `;

            // Добавляем строки параметров 406 MHz
            for (let i = 0; i < params406.length; i++) {
                const row = params406[i];
                tableHtml += `
                    <tr class="param-row">
                        <td class="param-name">${row[0]}</td>
                        <td>${row[1]}</td>
                        <td>${row[2]}</td>
                        <td>${row[3]}</td>
                        <td>${row[4]}</td>
                        <td>${row[5]}</td>
                    </tr>
                `;
            }

            // Добавляем 121.5 MHz секцию
            tableHtml += `
                        <tr>
                            <td colspan="6" class="header-121">121.5 MHz Transmitter Parameters</td>
                        </tr>
            `;

            for (let i = 0; i < params121.length; i++) {
                const row = params121[i];
                tableHtml += `
                    <tr class="param-row">
                        <td class="param-name">${row[0]}</td>
                        <td colspan="5">${row[1]}</td>
                    </tr>
                `;
            }

            tableHtml += `
                    </table>
            `;

            // Добавляем нашу таблицу Message внутри того же контейнера
            const hexMessage = data.hex_message || 'DEFAULT_HEX';
            if (hexMessage) {
                // Данные таблицы Message (те же что в showMessageTable)
                const messageTableData = [
                    ['1-15', '111111111111111', 'Bit-sync pattern', 'Valid'],
                    ['16-24', '000101111', 'Frame-sync pattern', 'Normal Operation'],
                    ['25', '1', 'Format Flag', 'Long Format'],
                    ['26', '0', 'Protocol Flag', 'Standard/National/RLS'],
                    ['27-36', '1000000000', 'Country Code', '512 - Russia'],
                    ['37-40', '0000', 'Protocol Code', 'Avionic'],
                    ['41-64', '000000100000000000000000', 'Test Data', '0x020000'],
                    ['65-74', '0111111111', 'Latitude (PDF-1)', 'Default value'],
                    ['75-85', '01111111111', 'Longitude (PDF-1)', 'Default value'],
                    ['86-106', '110000100000101101111', 'BCH PDF-1', '0x1820B7'],
                    ['107-110', '1000', 'Fixed (1101)', 'Invalid (1000)'],
                    ['111', '0', 'Position source', 'External/Unknown'],
                    ['112', '0', '121.5 MHz Device', 'Not included'],
                    ['113-122', '1111100000', 'Latitude (PDF-2)', 'bin 1111100000'],
                    ['123-132', '1111100000', 'Longitude (PDF-2)', 'bin 1111100000'],
                    ['133-144', '111001101100', 'BCH PDF-2', '0xE6C']
                ];

                // Добавляем таблицу Message в нашем стиле внутри того же контейнера
                tableHtml += `
                    <div style="margin-top: 10px;">
                        <div class="message-table-header">
                            <h3>EPIRB/ELT Beacon Message Decoder</h3>
                        </div>
                        <table class="message-table">
                            <thead>
                                <tr>
                                    <th style="width: 90px;">Bit Range</th>
                                    <th style="width: 200px;">Binary Content</th>
                                    <th style="width: 220px;">Field Name</th>
                                    <th>Decoded Value</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                for (let i = 0; i < messageTableData.length; i++) {
                    const row = messageTableData[i];
                    tableHtml += `
                        <tr>
                            <td>${row[0]}</td>
                            <td class="binary-content">${row[1]}</td>
                            <td class="field-name">${row[2]}</td>
                            <td>${row[3]}</td>
                        </tr>
                    `;
                }

                tableHtml += `
                            </tbody>
                        </table>
                        <div class="message-table-footer">
                            <div>COSPAS-SARSAT 406 MHz Beacon Message (144 bits)</div>
                            <div>Protocol: Long Format, Standard Location</div>
                        </div>
                    </div>
                `;
            }

            // Закрываем единый контейнер
            tableHtml += `</div>`;

            chartContainer.innerHTML = tableHtml;
        }

        function show121DataTable(data) {
            console.log('DEBUG: show121DataTable called');

            // Очищаем canvas и показываем HTML таблицу
            const canvas = document.getElementById('phaseChart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Создаем контейнер для HTML таблицы
            const chartContainer = document.querySelector('.chart-container');

            // Данные таблицы (те же что были в canvas версии)
            const tableData = [
                ['Carrier Frequency, Hz', '0', 'Low Sweep Frequency, Hz', '0'],
                ['Power, mW', '0.0', 'High Sweep Frequency, Hz', '0'],
                ['Sweep Period, sec', '0.0', 'Sweep Range, Hz', '0'],
                ['Modulation Index, %', '0', '', '']
            ];

            // Создаем HTML таблицу
            let tableHtml = `
                <div class="data121-table-container">
                    <div class="data121-table-header">
                        <h3>121.5 MHz Transmitter Parameters</h3>
                    </div>
                    <table class="data121-table">
                        <tbody>
            `;

            for (let i = 0; i < tableData.length; i++) {
                const row = tableData[i];
                tableHtml += '<tr>';
                for (let j = 0; j < row.length; j++) {
                    if (row[j]) {
                        // Названия параметров (четные колонки) - жирный шрифт
                        const cssClass = (j % 2 === 0) ? 'param-name' : 'param-value';
                        tableHtml += `<td class="${cssClass}">${row[j]}</td>`;
                    } else {
                        tableHtml += '<td></td>';
                    }
                }
                tableHtml += '</tr>';
            }

            tableHtml += `
                        </tbody>
                    </table>
                    <div class="data121-table-footer">
                        <div>Emergency Locator Transmitter (ELT) operating on 121.5 MHz</div>
                        <div>Used for aircraft emergency location and rescue operations</div>
                    </div>
                </div>
            `;

            chartContainer.innerHTML = tableHtml;
        }

        function updateMessageInfo(data) {
            // Обновляем только основную информацию без canvas
            document.getElementById('protocol').textContent = data.protocol;
            document.getElementById('date').textContent = data.date;
            document.getElementById('beaconModel').textContent = data.beacon_model;
            document.getElementById('beaconFreq').textContent = data.beacon_frequency.toFixed(1);

            // Показываем HEX сообщение если есть, иначе обычное сообщение
            if (data.hex_message && data.hex_message !== '') {
                document.getElementById('message').textContent = `HEX: ${data.hex_message}`;
            } else {
                document.getElementById('message').textContent = data.message;
            }

            // Обновление фазовых значений (только если элементы существуют)
            const phasePlusElem = document.getElementById('phasePlus');
            const phaseMinusElem = document.getElementById('phaseMinus');
            const tRiseElem = document.getElementById('tRise');
            const tFallElem = document.getElementById('tFall');

            if (phasePlusElem) phasePlusElem.textContent = (data.phase_pos_rad * 57.2958).toFixed(2);
            if (phaseMinusElem) phaseMinusElem.textContent = (data.phase_neg_rad * 57.2958).toFixed(2);
            if (tRiseElem) tRiseElem.textContent = data.t_rise_mcs.toFixed(1);
            if (tFallElem) tFallElem.textContent = data.t_fall_mcs.toFixed(1);
        }

        function updateStats(data) {
            // Добавляем real-time данные в статистику
            let realtimeSection = '';
            if (data.sdr_capture_active) {
                const pulseInfo = data.latest_pulse ?
                    `Last pulse: ${data.latest_pulse.length_ms.toFixed(1)}ms` :
                    'No pulses detected';

                realtimeSection = `
                    <div class="stat-row" style="background-color: #e8f5e8;"><span class="stat-label">SDR Real-time:</span><span class="stat-value">Active</span></div>
                    <div class="stat-row"><span class="stat-label">RMS Live, dBm</span><span class="stat-value">${data.realtime_rms_dbm.toFixed(1)}</span></div>
                    <div class="stat-row"><span class="stat-label">Pulses Found</span><span class="stat-value">${data.realtime_pulse_count}</span></div>
                    <div class="stat-row"><span class="stat-label">Pulse Info</span><span class="stat-value" title="${pulseInfo}">${pulseInfo.length > 20 ? pulseInfo.substring(0, 17) + '...' : pulseInfo}</span></div>
                `;
            } else if (data.running) {
                realtimeSection = `
                    <div class="stat-row" style="background-color: #ffe8e8;"><span class="stat-label">SDR Real-time:</span><span class="stat-value">Starting...</span></div>
                `;
            }

            const statsHtml = `
                ${realtimeSection}
                <div class="stat-row"><span class="stat-label">FS1,Hz</span><span class="stat-value">${data.fs1_hz.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">FS2,Hz</span><span class="stat-value">${data.fs2_hz.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">FS3,Hz</span><span class="stat-value">${data.fs3_hz.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Phase+,rad</span><span class="stat-value">${data.phase_pos_rad.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Phase-,rad</span><span class="stat-value">${data.phase_neg_rad.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">TRise,mcs</span><span class="stat-value">${data.t_rise_mcs.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">TFall,mcs</span><span class="stat-value">${data.t_fall_mcs.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">P,Wt</span><span class="stat-value">${data.p_wt.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Prise,ms</span><span class="stat-value">${data.prise_ms.toFixed(1)}</span></div>
                <div class="stat-row"><span class="stat-label">BitRate,bps</span><span class="stat-value">${data.bitrate_bps.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Symmetry,%</span><span class="stat-value">${data.symmetry_pct.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Preamble,ms</span><span class="stat-value">${data.preamble_ms.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Total,ms</span><span class="stat-value">${data.total_ms.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">RepPeriod,s</span><span class="stat-value">${data.rep_period_s.toFixed(3)}</span></div>
            `;

            // STRICT_COMPAT: Получаем данные последнего импульса для PSK метрик
            fetch('/api/last_pulse').then(response => response.json()).then(pulseData => {
                if (pulseData.last) {
                    const last = pulseData.last;
                    const pskHtml = `
                        <div class="stat-row" style="background-color: #f8f9fa; margin-top: 5px;"><span class="stat-label">PSK-406 Real-time</span><span class="stat-value">—</span></div>
                        <div class="stat-row"><span class="stat-label">Bitrate,bps</span><span class="stat-value">${last.bitrate_bps !== null ? last.bitrate_bps.toFixed(1) : '—'}</span></div>
                        <div class="stat-row"><span class="stat-label">Pos/Neg phase</span><span class="stat-value">${last.pos_phase !== null && last.neg_phase !== null ? last.pos_phase.toFixed(2) + '/' + last.neg_phase.toFixed(2) : '—'}</span></div>
                        <div class="stat-row"><span class="stat-label">Rise/Fall,μs</span><span class="stat-value">${last.ph_rise !== null && last.ph_fall !== null ? last.ph_rise.toFixed(1) + '/' + last.ph_fall.toFixed(1) : '—'}</span></div>
                        <div class="stat-row"><span class="stat-label">Asymmetry,%</span><span class="stat-value">${last.symmetry_pct !== null ? last.symmetry_pct.toFixed(1) : '—'}</span></div>
                        <div class="stat-row"><span class="stat-label">Message (HEX)</span><span class="stat-value" style="font-family: monospace;">${last.msg_hex_short || '—'}</span></div>
                        <div class="stat-row"><span class="stat-label">CRC/OK</span><span class="stat-value">${last.msg_ok !== null ? (last.msg_ok ? 'OK' : 'FAIL') : '—'}</span></div>
                    `;
                    document.getElementById('statsContent').innerHTML = statsHtml + pskHtml;
                } else {
                    document.getElementById('statsContent').innerHTML = statsHtml;
                }
            }).catch(error => {
                console.error('Error fetching pulse data:', error);
                document.getElementById('statsContent').innerHTML = statsHtml;
            });
        }


        function drawFrequencyPowerChart(data) {
            console.log('DEBUG: drawFrequencyPowerChart called');

            const canvas = document.getElementById('phaseChart');
            if (!canvas) {
                console.error('Canvas not found!');
                return;
            }

            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            const midY = height / 2;

            ctx.clearRect(0, 0, width, height);

            // === ВЕРХНИЙ ГРАФИК: Power (dBm) ===

            // Сетка для верхнего графика
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии (Power: 36-41 dBm)
            for (let i = 0; i <= 5; i++) {
                const y = (midY / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии (время: 0-60 минут)
            for (let i = 0; i <= 10; i++) {
                const x = (width / 10) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, midY);
                ctx.stroke();
            }

            // Y-axis labels для Power
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            ctx.fillText('41.0', 5, 15);
            ctx.fillText('40.0', 5, midY * 0.2);
            ctx.fillText('39.0', 5, midY * 0.4);
            ctx.fillText('38.0', 5, midY * 0.6);
            ctx.fillText('37.0', 5, midY * 0.8);
            ctx.fillText('36.0', 5, midY - 5);

            // X-axis labels для времени (верхний график)
            for (let i = 0; i <= 10; i++) {
                const x = (width / 10) * i;
                const minutes = (i * 6).toString().padStart(2, '0'); // 0, 6, 12, 18... 60 минут
                ctx.fillText(minutes, x - 8, midY - 5);
            }

            // Красные пунктирные референсные линии для Power
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);

            // Верхний предел (41.0 dBm)
            const y_power_max = midY * 0.0;
            ctx.beginPath();
            ctx.moveTo(0, y_power_max);
            ctx.lineTo(width, y_power_max);
            ctx.stroke();

            // Нижний предел (36.0 dBm)
            const y_power_min = midY * 1.0;
            ctx.beginPath();
            ctx.moveTo(0, y_power_min);
            ctx.lineTo(width, y_power_min);
            ctx.stroke();

            ctx.setLineDash([]);

            // График данных Power - синяя линия, 60 точек, центр на 37 dBm
            ctx.strokeStyle = '#0066FF';
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < 60; i++) {
                const x = (i / 59) * width; // 60 точек по ширине
                // Power = 37 ± небольшие колебания
                const powerValue = 37 + Math.sin(i * 0.2) * 0.8 + Math.sin(i * 0.05) * 0.5; // колебания ±1.3 dBm
                const y = midY - ((powerValue - 36) / 5) * midY; // масштаб 36-41 dBm

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // === РАЗДЕЛИТЕЛЬ ===
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(width, midY);
            ctx.stroke();

            // === НИЖНИЙ ГРАФИК: Frequency (Hz) ===

            // Сетка для нижнего графика
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии (Frequency: 406022-406028 Hz)
            for (let i = 0; i <= 6; i++) {
                const y = midY + (midY / 6) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии для нижнего графика
            for (let i = 0; i <= 10; i++) {
                const x = (width / 10) * i;
                ctx.beginPath();
                ctx.moveTo(x, midY);
                ctx.lineTo(x, height);
                ctx.stroke();
            }

            // Y-axis labels для Frequency (406022-406028 Hz, центр 406025)
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            ctx.fillText('406028', 5, midY + 15);
            ctx.fillText('406027', 5, midY + midY * 0.15);
            ctx.fillText('406026', 5, midY + midY * 0.3);
            ctx.fillText('406025', 5, midY + midY * 0.5); // центр
            ctx.fillText('406024', 5, midY + midY * 0.7);
            ctx.fillText('406023', 5, midY + midY * 0.85);
            ctx.fillText('406022', 5, height - 10);

            // X-axis labels для времени (нижний график)
            for (let i = 0; i <= 10; i++) {
                const x = (width / 10) * i;
                const minutes = (i * 6).toString().padStart(2, '0');
                ctx.fillText(minutes, x - 8, height - 5);
            }

            // Красные пунктирные референсные линии для Frequency
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);

            // Верхний предел (406028 Hz)
            const y_freq_max = midY + midY * 0.0;
            ctx.beginPath();
            ctx.moveTo(0, y_freq_max);
            ctx.lineTo(width, y_freq_max);
            ctx.stroke();

            // Нижний предел (406022 Hz)
            const y_freq_min = midY + midY * 1.0;
            ctx.beginPath();
            ctx.moveTo(0, y_freq_min);
            ctx.lineTo(width, y_freq_min);
            ctx.stroke();

            ctx.setLineDash([]);

            // График данных Frequency - синяя линия, 60 точек, центр на 406025 Hz
            ctx.strokeStyle = '#0066FF';
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < 60; i++) {
                const x = (i / 59) * width; // 60 точек по ширине
                // Frequency = 406025 ± небольшие колебания
                const freqValue = 406025 + Math.sin(i * 0.15) * 1.5 + Math.sin(i * 0.08) * 1.0; // колебания ±2.5 Hz
                const y = midY + midY - ((freqValue - 406022) / 6) * midY; // масштаб 406022-406028 Hz

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Заголовки графиков
            ctx.fillStyle = '#495057';
            ctx.font = '12px Arial';
            ctx.fillText('Power vs Time', width/2 - 50, 20);
            ctx.fillText('Frequency vs Time', width/2 - 60, midY + 20);

            // Подписи единиц измерения
            ctx.fillStyle = '#6c757d';
            ctx.font = '10px Arial';
            ctx.fillText('dBm', 5, 45);
            ctx.fillText('Hz', 5, midY + 45);
            ctx.fillText('Time, min', width - 60, midY - 5);
            ctx.fillText('Time, min', width - 60, height - 5);

            console.log('DEBUG: drawFrequencyPowerChart completed');
        }

        function updateDisplay(data) {
            console.log('=== updateDisplay called ===');
            console.log('currentView:', currentView);
            console.log('data.phase_data length:', data.phase_data ? data.phase_data.length : 'null');

            // Восстанавливаем canvas для обычных режимов если он был заменен HTML таблицей
            const chartContainer = document.querySelector('.chart-container');
            if (!chartContainer.querySelector('#phaseChart')) {
                chartContainer.innerHTML = '<canvas id="phaseChart"></canvas>';
                // Обновляем глобальные переменные после восстановления canvas
                canvas = document.getElementById('phaseChart');
                ctx = canvas.getContext('2d');
                resizeCanvas(); // Переинициализируем размеры canvas
            }

            // Обновление основной информации
            document.getElementById('protocol').textContent = data.protocol;
            document.getElementById('date').textContent = data.date;
            document.getElementById('beaconModel').textContent = data.beacon_model;
            document.getElementById('beaconFreq').textContent = data.beacon_frequency.toFixed(1);
            // Показываем HEX сообщение если есть, иначе обычное сообщение
            if (data.hex_message && data.hex_message !== '') {
                document.getElementById('message').textContent = `HEX: ${data.hex_message}`;
            } else {
                document.getElementById('message').textContent = data.message;
            }

            // Обновление статуса SDR
            if (data.sdr_device_info) {
                document.getElementById('sdrStatus').textContent = data.sdr_device_info;
            }

            // Обновление фазовых значений (только если элементы существуют)
            const phasePlusElem = document.getElementById('phasePlus');
            const phaseMinusElem = document.getElementById('phaseMinus');
            const tRiseElem = document.getElementById('tRise');
            const tFallElem = document.getElementById('tFall');

            if (phasePlusElem) phasePlusElem.textContent = (data.phase_pos_rad * 57.2958).toFixed(2);
            if (phaseMinusElem) phaseMinusElem.textContent = (data.phase_neg_rad * 57.2958).toFixed(2);
            if (tRiseElem) tRiseElem.textContent = data.t_rise_mcs.toFixed(1);
            if (tFallElem) tFallElem.textContent = data.t_fall_mcs.toFixed(1);

            // Обновление элементов для режима fr_pwr
            const freqElem = document.getElementById('freq');
            const powerElem = document.getElementById('power');
            if (freqElem) freqElem.textContent = (data.beacon_frequency / 1000000).toFixed(3);  // MHz
            if (powerElem) powerElem.textContent = data.p_wt.toFixed(3);  // Wt

            // Обновление статистики
            updateStats(data);

            // Автообновление отключено - синхронизация с сервером не требуется
            console.log('Display updated, auto-update remains disabled');

            // Обновляем график только для обычных режимов (не специальных)
            if (currentView !== 'message' && currentView !== '121_data') {
                console.log('DEBUG: phase_data:', data.phase_data ? data.phase_data.length : 'null');
                console.log('DEBUG: xs_fm_ms:', data.xs_fm_ms ? data.xs_fm_ms.length : 'null');
                if (data.phase_data && data.phase_data.length > 0) {
                    console.log('DEBUG: phase_data sample:', data.phase_data.slice(0, 5));
                }
                if (data.xs_fm_ms && data.xs_fm_ms.length > 0) {
                    console.log('DEBUG: xs_fm_ms sample:', data.xs_fm_ms.slice(0, 5));
                }

                drawChart(data);
            }
        }


        function drawFMChart(data) {
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            // Сетка (как в Phase графике)
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии сетки
            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии сетки с временными метками
            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();

                // Временные метки с учетом масштаба (используем те же переменные что и Phase)
                const scaledDuration = MESSAGE_DURATION_MS * (currentTimeScale / 100);
                let startOffset;
                if (currentTimeScale === 100) {
                    startOffset = 0; // При 100% начинаем с 0мс
                } else {
                    // Для других масштабов
                    const preambleMs = data?.preamble_ms || 10.0;
                    const offsetMs = getOffsetForScale(currentTimeScale);
                    startOffset = Math.max(0, preambleMs - offsetMs);
                }
                const timeMs = (startOffset + i * scaledDuration / 8).toFixed(1);
                ctx.fillText(timeMs, x - 10, height - 5);
            }

            // Нулевая линия (центральная)
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();

            // Автомасштабирование: определяем масштаб на основе данных
            let FM_SCALE = 3000; // Значение по умолчанию ±3000 Hz

            // Сначала получаем отфильтрованные точки для определения масштаба
            let filteredPointsForScale = [];
            if (data) {
                let fmData = (data.fm_data || []).map(v => Number(v));
                let xsData = (data.fm_xs_ms || []).map(v => Number(v));

                if (fmData.length > 1 && xsData.length === fmData.length) {
                    // Вычисляем временное окно
                    let windowStart;
                    if (currentTimeScale === 100) {
                        windowStart = 0;
                    } else {
                        const preambleMs = data.preamble_ms || 10.0;
                        const offsetMs = getOffsetForScale(currentTimeScale);
                        windowStart = Math.max(0, preambleMs - offsetMs);
                    }
                    const windowDuration = MESSAGE_DURATION_MS * (currentTimeScale / 100.0);
                    const windowEnd = windowStart + windowDuration;

                    // Фильтруем точки по временному окну
                    for (let i = 0; i < fmData.length; i++) {
                        const timeMs = xsData[i];
                        if (Number.isFinite(timeMs) && timeMs >= windowStart && timeMs <= windowEnd) {
                            filteredPointsForScale.push({
                                time: timeMs,
                                fm: fmData[i]
                            });
                        }
                    }

                    // Определяем автоматический масштаб
                    if (filteredPointsForScale.length > 0) {
                        const maxAbsFM = Math.max(...filteredPointsForScale.map(p => Math.abs(p.fm)));
                        // Добавляем небольшой запас (10%) и округляем до красивого числа
                        const scaledMax = maxAbsFM * 1.1;
                        if (scaledMax <= 1500) {
                            FM_SCALE = 1500;
                        } else if (scaledMax <= 3000) {
                            FM_SCALE = 3000;
                        } else if (scaledMax <= 5000) {
                            FM_SCALE = 5000;
                        } else if (scaledMax <= 10000) {
                            FM_SCALE = 10000;
                        } else {
                            FM_SCALE = Math.ceil(scaledMax / 1000) * 1000; // Округляем до ближайшей тысячи
                        }
                    }
                }
            }

            // Динамические пунктирные линии на основе масштаба
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);

            // Определяем шаги сетки на основе масштаба
            let gridSteps = [];
            if (FM_SCALE <= 1500) {
                gridSteps = [250, 500, 750, 1000];
            } else if (FM_SCALE <= 3000) {
                gridSteps = [500, 1000, 1500, 2000, 2500];
            } else if (FM_SCALE <= 5000) {
                gridSteps = [1000, 2000, 3000, 4000];
            } else if (FM_SCALE <= 10000) {
                gridSteps = [2000, 4000, 6000, 8000];
            } else {
                const step = Math.floor(FM_SCALE / 5 / 1000) * 1000;
                gridSteps = [step, step*2, step*3, step*4];
            }

            // Рисуем горизонтальные линии сетки
            for (const step of gridSteps) {
                if (step < FM_SCALE) {
                    const y_plus = height / 2 - (step / FM_SCALE) * (height / 2);
                    const y_minus = height / 2 + (step / FM_SCALE) * (height / 2);

                    // Положительная линия
                    ctx.beginPath();
                    ctx.moveTo(0, y_plus);
                    ctx.lineTo(width, y_plus);
                    ctx.stroke();

                    // Отрицательная линия
                    ctx.beginPath();
                    ctx.moveTo(0, y_minus);
                    ctx.lineTo(width, y_minus);
                    ctx.stroke();
                }
            }

            // Возвращаем сплошную линию
            ctx.setLineDash([]);

            // Динамические подписи оси Y
            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';

            // Максимум
            ctx.fillText(`+${(FM_SCALE/1000).toFixed(1)}kHz`, 5, 15);

            // Промежуточные значения
            for (const step of gridSteps) {
                if (step < FM_SCALE) {
                    const y_plus = height / 2 - (step / FM_SCALE) * (height / 2);
                    const y_minus = height / 2 + (step / FM_SCALE) * (height / 2);

                    ctx.fillText(`+${(step/1000).toFixed(1)}kHz`, 5, y_plus - 2);
                    ctx.fillText(`-${(step/1000).toFixed(1)}kHz`, 5, y_minus - 2);
                }
            }

            // 0
            ctx.fillText('0', 5, height / 2 - 2);

            // Минимум
            ctx.fillText(`-${(FM_SCALE/1000).toFixed(1)}kHz`, 5, height - 5);

            // График FM данных - используем уже отфильтрованные данные
            if (filteredPointsForScale.length > 0) {
                console.log('DEBUG drawFMChart: Using filteredPointsForScale with', filteredPointsForScale.length, 'points');
                console.log(`DEBUG: FM_SCALE = ${FM_SCALE} Hz`);

                // Вычисляем временное окно для отображения
                let windowStart;
                if (currentTimeScale === 100) {
                    windowStart = 0;
                } else {
                    const preambleMs = data.preamble_ms || 10.0;
                    const offsetMs = getOffsetForScale(currentTimeScale);
                    windowStart = Math.max(0, preambleMs - offsetMs);
                }
                const windowDuration = MESSAGE_DURATION_MS * (currentTimeScale / 100.0);

                // Децимация если слишком много точек
                let pointsToDraw = filteredPointsForScale;
                if (filteredPointsForScale.length > 1000) {
                    const step = Math.floor(filteredPointsForScale.length / 1000);
                    pointsToDraw = filteredPointsForScale.filter((_, i) => i % step === 0);
                }

                // Рисуем график FM (зеленая линия как в Phase)
                ctx.strokeStyle = '#28a745';
                ctx.lineWidth = 2;
                ctx.beginPath();

                for (let i = 0; i < pointsToDraw.length; i++) {
                    const point = pointsToDraw[i];
                    const normalizedTime = (point.time - windowStart) / windowDuration;
                    const x = normalizedTime * width;
                    // Используем автоматический масштаб
                    const y = height / 2 - (point.fm / FM_SCALE) * (height / 2);

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();

                // Информация о данных
                const maxFM = Math.max(...filteredPointsForScale.map(p => Math.abs(p.fm)));
                ctx.fillStyle = '#6c757d';
                ctx.font = '10px Arial';
                ctx.fillText(`Max deviation: ±${maxFM.toFixed(0)} Hz (Scale: ±${(FM_SCALE/1000).toFixed(1)}kHz)`, width - 200, 15);
            } else if (data && (data.fm_data || []).length === 0) {
                // Показываем сообщение, если нет данных
                ctx.fillStyle = '#666';
                ctx.font = '16px Arial';
                ctx.fillText('No FM data available', width/2 - 80, height/2);
                ctx.font = '12px Arial';
                ctx.fillText('Please load a CF32 file first', width/2 - 80, height/2 + 25);
            } else if (data) {
                // Есть данные, но ни одна точка не попала в временное окно
                ctx.fillStyle = '#888';
                ctx.font = '14px Arial';
                ctx.fillText('No FM data in current time window', width/2 - 120, height/2);
            }
        }

                function drawRiseFallChart(data) {
            const width = canvas.width;
            const height = canvas.height;
            const midY = height / 2;

            ctx.clearRect(0, 0, width, height);

            // === ВЕРХНИЙ ГРАФИК: Fig.6 Modulation index ===

            // Сетка для верхнего графика
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии верхнего графика
            for (let i = 0; i <= 5; i++) {
                const y = (midY / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии с временными метками
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, midY);
                ctx.stroke();

                // Временные метки для верхнего графика
                const timeMs = (i * 5).toFixed(0);
                ctx.fillText(timeMs, x - 8, midY - 5);
            }

            // Y-axis labels для Modulation index (Phase)
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            ctx.fillText('1.3', 5, 15);
            ctx.fillText('Ph+,rad', 5, 28);
            ctx.fillText('1.1', 5, midY * 0.25);
            ctx.fillText('0.9', 5, midY * 0.4);
            ctx.fillText('1.1', 5, midY * 0.6);
            ctx.fillText('1.3', 5, midY * 0.75);
            ctx.fillText('Ph-,rad', 5, midY - 5);

            // График Ph+ (верхняя синусоида)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 8 * Math.PI; // 4 периода на 40ms
                const y = midY * 0.15 + Math.sin(t) * midY * 0.08; // колебания в районе 1.1
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // График Ph- (нижняя синусоида)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 8 * Math.PI; // 4 периода на 40ms
                const y = midY * 0.85 + Math.sin(t) * midY * 0.08; // колебания в районе -1.1
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Подпись времени
            ctx.fillText('t,m', width - 25, midY - 5);

            // Заголовок верхнего графика
            ctx.fillStyle = '#495057';
            ctx.font = '12px Arial';
            ctx.fillText('Fig.6 Modulation index', width/2 - 60, midY - 15);

            // === РАЗДЕЛИТЕЛЬ ===
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(width, midY);
            ctx.stroke();

            // === НИЖНИЙ ГРАФИК: Fig.7 Rise and Fall Times ===

            // Горизонтальные линии нижнего графика
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const y = midY + (midY / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии нижнего графика
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, midY);
                ctx.lineTo(x, height);
                ctx.stroke();

                // Временные метки для нижнего графика
                const timeMs = (i * 5).toFixed(0);
                ctx.fillText(timeMs, x - 8, height - 5);
            }

            // Центральная линия нижнего графика (ось 0)
            const centerY = midY + midY/2;
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();

            // Y-axis labels для Rise/Fall Times
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            ctx.fillText('300', 5, midY + 15);
            ctx.fillText('Tr,mcs', 5, midY + 28);
            ctx.fillText('150', 5, centerY - midY/5);
            ctx.fillText('0', 5, centerY + 4);
            ctx.fillText('150', 5, centerY + midY/5);
            ctx.fillText('300', 5, height - 15);
            ctx.fillText('Tf,mcs', 5, height - 2);

            // График Tr (верхняя часть - Rise time)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 12 * Math.PI; // больше частота
                const baseY = centerY - midY * 0.25; // в районе 150 mcs
                const y = baseY + Math.sin(t) * 15 + Math.sin(t * 2.3) * 8; // сложный сигнал
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // График Tf (нижняя часть - Fall time)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 12 * Math.PI;
                const baseY = centerY + midY * 0.25; // в районе -150 mcs
                const y = baseY + Math.sin(t + Math.PI/3) * 15 + Math.sin(t * 1.7) * 8; // сложный сигнал со сдвигом
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Подпись времени
            ctx.fillText('t,m', width - 25, height - 5);

            // Заголовок нижнего графика
            ctx.fillStyle = '#495057';
            ctx.font = '12px Arial';
            ctx.fillText('Fig.7 Rise and Fall Times', width/2 - 70, height - 15);
        }

        function drawFrequencyChart(data) {
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            // Сетка
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();

                const timeS = (i * 5).toFixed(0);
                ctx.fillText(timeS + ' s', x - 10, height - 5);
            }

            // Y-axis labels для частоты
            ctx.fillText('406.030 MHz', 5, 20);
            ctx.fillText('406.025 MHz', 5, height/2);
            ctx.fillText('406.020 MHz', 5, height - 10);

            // График частоты
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < width; i += 2) {
                const x = i;
                const y = height/2 + Math.sin(i * 0.005) * 20 + Math.random() * 5;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        function draw121DataTable(data) {
            console.log('DEBUG: draw121DataTable function called!');
            const width = canvas.width;
            const height = canvas.height;
            console.log('DEBUG: Canvas dimensions:', width, 'x', height);

            // Убедимся, что получаем контекст
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);

            // Заголовок таблицы
            ctx.fillStyle = '#5a9bd4';
            ctx.fillRect(0, 0, width, 40);

            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Arial';
            ctx.fillText('121.5 MHz Transmitter Parameters', width/2 - 120, 25);

            // Создание таблицы
            const tableData = [
                ['Carrier Frequency, Hz', '0', 'Low Sweep Frequency, Hz', '0'],
                ['Power, mW', '0.0', 'High Sweep Frequency, Hz', '0'],
                ['Sweep Period, sec', '0.0', 'Sweep Range, Hz', '0'],
                ['Modulation Index, %', '0', '', '']
            ];

            const rowHeight = 35;
            const colWidth = width / 4;
            const startY = 50;

            // Рисуем границы таблицы
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;

            // Горизонтальные линии
            for (let i = 0; i <= tableData.length; i++) {
                const y = startY + i * rowHeight;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии
            for (let i = 0; i <= 4; i++) {
                const x = i * colWidth;
                ctx.beginPath();
                ctx.moveTo(x, startY);
                ctx.lineTo(x, startY + tableData.length * rowHeight);
                ctx.stroke();
            }

            // Заполняем таблицу данными
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';

            for (let row = 0; row < tableData.length; row++) {
                for (let col = 0; col < tableData[row].length; col++) {
                    if (tableData[row][col]) {
                        const x = col * colWidth + 10;
                        const y = startY + row * rowHeight + 22;

                        // Жирный шрифт для названий параметров (левые колонки)
                        if (col % 2 === 0) {
                            ctx.font = 'bold 12px Arial';
                        } else {
                            ctx.font = '12px Arial';
                        }

                        ctx.fillText(tableData[row][col], x, y);
                    }
                }
            }

            // Дополнительная информация внизу
            ctx.fillStyle = '#666';
            ctx.font = '11px Arial';
            ctx.fillText('Emergency Locator Transmitter (ELT) operating on 121.5 MHz', 20, height - 30);
            ctx.fillText('Used for aircraft emergency location and rescue operations', 20, height - 15);
        }

        function drawMessageTable(hexMessage) {
            console.log('DEBUG: drawMessageTable function called with:', hexMessage);
            console.log('DEBUG: currentView at drawMessageTable:', currentView);
            const width = canvas.width;
            const height = canvas.height;
            console.log('DEBUG: Canvas dimensions:', width, 'x', height);

            // Убедимся, что получаем контекст
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);

            // Если нет HEX сообщения из файла, показываем инструкцию
            if (!hexMessage || hexMessage.trim() === '') {
                ctx.fillStyle = '#666';
                ctx.font = '16px Arial';
                ctx.fillText('No message decoded', width/2 - 80, height/2 - 20);
                ctx.font = '12px Arial';
                ctx.fillText('Please load a .cf32 file using the File button', width/2 - 140, height/2 + 10);
                ctx.fillText('to decode EPIRB/ELT beacon message', width/2 - 110, height/2 + 30);
                return;
            }

            // Заголовок таблицы
            ctx.fillStyle = '#1976D2';
            ctx.fillRect(0, 0, width, 45);

            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Arial';
            ctx.fillText('EPIRB/ELT Beacon Message Decoder', width/2 - 130, 18);
            ctx.font = '11px Arial';
            ctx.fillText('HEX: ' + hexMessage, width/2 - 180, 35);

            // Заголовки колонок
            const headers = ['Bit Range', 'Binary Content', 'Field Name', 'Decoded Value'];
            const colWidths = [90, 200, 220, width - 510];
            const rowHeight = 26;
            const startY = 55;
            const headerHeight = 28;

            // Рисуем заголовки
            ctx.fillStyle = '#E0E0E0';
            ctx.fillRect(0, startY, width, headerHeight);

            ctx.fillStyle = '#333';
            ctx.font = 'bold 13px Arial';
            let xPos = 5;
            for (let i = 0; i < headers.length; i++) {
                ctx.fillText(headers[i], xPos, startY + 19);
                xPos += colWidths[i];
            }

            // Получаем данные декодирования через API
            // Пока используем заглушку - в реальности нужно будет передавать через API
            const tableData = [
                ['1-15', '111111111111111', 'Bit-sync pattern', 'Valid'],
                ['16-24', '000101111', 'Frame-sync pattern', 'Normal Operation'],
                ['25', '1', 'Format Flag', 'Long Format'],
                ['26', '0', 'Protocol Flag', 'Standard/National/RLS'],
                ['27-36', '1000000000', 'Country Code', '512 - Russia'],
                ['37-40', '0000', 'Protocol Code', 'Avionic'],
                ['41-64', '000000100000000000000000', 'Test Data', '0x020000'],
                ['65-74', '0111111111', 'Latitude (PDF-1)', 'Default value'],
                ['75-85', '01111111111', 'Longitude (PDF-1)', 'Default value'],
                ['86-106', '110000100000101101111', 'BCH PDF-1', '0x1820B7'],
                ['107-110', '1000', 'Fixed (1101)', 'Invalid (1000)'],
                ['111', '0', 'Position source', 'External/Unknown'],
                ['112', '0', '121.5 MHz Device', 'Not included'],
                ['113-122', '1111100000', 'Latitude (PDF-2)', 'bin 1111100000'],
                ['123-132', '1111100000', 'Longitude (PDF-2)', 'bin 1111100000'],
                ['133-144', '111001101100', 'BCH PDF-2', '0xE6C']
            ];

            // Рисуем строки таблицы
            ctx.font = '12px Arial';
            let currentY = startY + headerHeight;

            for (let row = 0; row < tableData.length; row++) {
                // Чередующиеся цвета фона
                if (row % 2 === 0) {
                    ctx.fillStyle = '#F5F5F5';
                    ctx.fillRect(0, currentY, width, rowHeight);
                }

                // Рисуем данные
                ctx.fillStyle = '#333';
                xPos = 5;
                for (let col = 0; col < tableData[row].length; col++) {
                    // Особый стиль для Binary Content
                    if (col === 1) {
                        ctx.font = '11px monospace';
                        ctx.fillStyle = '#0066CC';
                    } else if (col === 2) {
                        ctx.font = 'bold 12px Arial';
                        ctx.fillStyle = '#333';
                    } else {
                        ctx.font = '12px Arial';
                        ctx.fillStyle = '#333';
                    }

                    // Обрезаем текст если он слишком длинный
                    const text = tableData[row][col];
                    const maxWidth = colWidths[col] - 10;
                    let displayText = text;

                    if (ctx.measureText(text).width > maxWidth) {
                        while (ctx.measureText(displayText + '...').width > maxWidth && displayText.length > 0) {
                            displayText = displayText.slice(0, -1);
                        }
                        displayText += '...';
                    }

                    ctx.fillText(displayText, xPos, currentY + 16);
                    xPos += colWidths[col];
                }

                // Горизонтальная линия
                ctx.strokeStyle = '#DDD';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(0, currentY + rowHeight);
                ctx.lineTo(width, currentY + rowHeight);
                ctx.stroke();

                currentY += rowHeight;
            }

            // Нижняя информация
            ctx.fillStyle = '#666';
            ctx.font = '11px Arial';
            ctx.fillText('COSPAS-SARSAT 406 MHz Beacon Message (144 bits)', 20, height - 30);
            ctx.fillText('Protocol: Long Format, Standard Location', 20, height - 15);
        }

        async function fetchData() {
            console.log('=== fetchData called ===');
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                console.log('fetchData: received data, currentView:', currentView);

                // Обрабатываем специальные режимы
                if (currentView === 'message') {
                    // Для режима message показываем HTML таблицу декодирования и обновляем только Current
                    console.log('DEBUG: Showing HTML message table for hex:', data.hex_message);
                    showMessageTable(data.hex_message || '');
                    // Обновляем только Current таблицу, не трогая canvas
                    updateStats(data);
                    updateMessageInfo(data);
                } else if (currentView === '121_data') {
                    // Для режима 121 показываем HTML таблицу 121 и обновляем только Current
                    console.log('DEBUG: Showing HTML 121 table');
                    show121DataTable(data);
                    // Обновляем только Current таблицу, не трогая canvas
                    updateStats(data);
                    updateMessageInfo(data);
                } else if (currentView === 'sum_table') {
                    // Для режима Sum table показываем HTML сводную таблицу и обновляем только Current
                    console.log('DEBUG: Showing HTML sum table');
                    showSumTable(data);
                    // Обновляем только Current таблицу, не трогая canvas
                    updateStats(data);
                    updateMessageInfo(data);
                } else {
                    // Для остальных режимов рисуем графики и обновляем display
                    updateDisplay(data);
                }
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        // STRICT_COMPAT: Функция для polling системной информации
        async function fetchSystemState() {
            try {
                const response = await fetch('/api/state');
                const data = await response.json();

                // Обновляем системную информацию в интерфейсе
                console.log('System state:', data.sdr_running ? 'SDR Running' : 'SDR Idle',
                           'RMS:', data.current_rms_dbm.toFixed(1), 'dBm',
                           'Buffer:', data.iq_buffer_info.total_written, 'samples');

                // Можно добавить отображение этих данных в интерфейсе
                if (data.sdr_device_info) {
                    document.getElementById('sdrStatus').textContent = data.sdr_device_info;
                }

            } catch (error) {
                console.error('Error fetching system state:', error);
            }
        }

        // Переменная для управления таймером обновления
        let updateTimer = null;
        let systemStateTimer = null;
        let isRunning = false;

        function startUpdating() {
            if (!updateTimer) {
                updateTimer = setInterval(fetchData, 700);
                isRunning = true;
                console.log('Graph updating started');
            }
            // STRICT_COMPAT: Запускаем polling системного состояния каждую секунду
            if (!systemStateTimer) {
                systemStateTimer = setInterval(fetchSystemState, 1000);
                console.log('System state polling started');
            }
        }

        function stopUpdating() {
            if (updateTimer) {
                clearInterval(updateTimer);
                updateTimer = null;
                isRunning = false;
                console.log('Graph updating stopped');
            }
            // STRICT_COMPAT: Останавливаем системный таймер
            if (systemStateTimer) {
                clearInterval(systemStateTimer);
                systemStateTimer = null;
                console.log('System state polling stopped');
            }
        }

        // Функции кнопок
        async function measure() {
            // Показываем статус инициализации
            document.getElementById('sdrStatus').textContent = 'Initializing SDR...';

            const response = await fetch('/api/measure', { method: 'POST' });
            const data = await response.json();

            // Обновляем статус SDR после инициализации
            if (data.sdr_device_info) {
                document.getElementById('sdrStatus').textContent = data.sdr_device_info;
            }

            console.log('Measure triggered, SDR status:', data.sdr_device_info);
        }

        async function runTest() {
            const response = await fetch('/api/run', { method: 'POST' });
            const data = await response.json();
            // Автообновление отключено - обновление только при загрузке файла
            console.log('Test run, auto-update disabled');
        }

        async function contTest() {
            const response = await fetch('/api/cont', { method: 'POST' });
            const data = await response.json();
            // Автообновление отключено - обновление только при загрузке файла
            console.log('Test continue, auto-update disabled');
        }

        async function breakTest() {
            const response = await fetch('/api/break', { method: 'POST' });
            const data = await response.json();
            // Автообновление всегда отключено
            stopUpdating();
            console.log('Test break, auto-update disabled');
        }

        function loadFile() {
            const fileInput = document.getElementById('fileInput');
            fileInput.click();
        }

        async function uploadFile(input) {
            if (input.files && input.files[0]) {
                const file = input.files[0];

                if (!file.name.endsWith('.cf32')) {
                    alert('Выберите файл с расширением .cf32');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (result.status === 'success' && result.processed) {
                        console.log('File uploaded and processed successfully:', result);
                        console.log('=== FORCING DATA UPDATE ===');
                        // Обновляем данные после загрузки
                        fetchData();
                    } else {
                        console.error('Upload failed:', result);
                        if (result.status === 'success' && !result.processed) {
                            alert('Файл загружен, но не удалось обработать: ' + (result.message || 'Unknown error'));
                        } else {
                            alert('Ошибка загрузки файла: ' + (result.error || result.message || 'Unknown error'));
                        }
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                    alert('Ошибка загрузки файла: ' + error.message);
                }

                // Очищаем input для возможности повторной загрузки того же файла
                input.value = '';
            }
        }

        async function saveFile() {
            await fetch('/api/save', { method: 'POST' });
        }

        // Первоначальная загрузка данных (автообновление отключено по умолчанию)
        fetchData();

        // Отключаем автообновление по умолчанию - график обновляется только при загрузке файла
        console.log('Auto-update disabled by default');
    </script>
</body>
</html>
"""

# API routes
@app.route('/')
def index():
    response = Response(HTML_PAGE, mimetype='text/html')
    # Отключаем кэширование для разработки
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
    response.headers['ETag'] = ''
    return response

@app.route('/api/status')
def api_status():
    # Добавляем небольшие вариации для реалистичности
    fs1_var = STATE.fs1_hz + random.uniform(-0.5, 0.5)
    fs2_var = STATE.fs2_hz + random.uniform(-0.5, 0.5)
    fs3_var = STATE.fs3_hz + random.uniform(-0.5, 0.5)


    # Получаем информацию о статусе SDR
    sdr_status = {}
    if sdr_backend:
        try:
            sdr_status = sdr_backend.get_status() or {}
        except Exception:
            sdr_status = {}

    # Получаем real-time данные от RMS анализа
    realtime_rms = 0.0
    realtime_pulse_count = 0
    latest_pulse_info = None

    if sdr_running:
        # Получаем текущее RMS значение (без блокировки для упрощения)
        try:
            realtime_rms = current_rms_dbm
        except NameError:
            realtime_rms = -100.0  # Значение по умолчанию если переменная не определена

        # Считаем количество импульсов в очереди
        realtime_pulse_count = pulse_queue.qsize()

        # Получаем информацию о последнем импульсе
        try:
            # Копируем очередь чтобы получить последний элемент без удаления
            temp_queue = list(pulse_queue.queue)
            if temp_queue:
                latest_pulse_info = temp_queue[-1]
        except:
            pass

    return jsonify({
        'running': STATE.running,
        'protocol': STATE.protocol,
        'date': STATE.date,
        'conditions': STATE.conditions,
        'beacon_model': STATE.beacon_model,
        'beacon_frequency': STATE.beacon_frequency,
        'message': STATE.message,
        'hex_message': STATE.hex_message,
        'fs1_hz': fs1_var,
        'fs2_hz': fs2_var,
        'fs3_hz': fs3_var,
        'phase_pos_rad': STATE.pos_phase,  # Используем новые значения из demod
        'phase_neg_rad': STATE.neg_phase,
        't_rise_mcs': STATE.ph_rise,    # В микросекундах из demod
        't_fall_mcs': STATE.ph_fall,
        'p_wt': STATE.p_wt,
        'prise_ms': STATE.prise_ms,
        'bitrate_bps': STATE.bitrate_bps,
        'symmetry_pct': STATE.symmetry_pct,  # Используем symmetry_pct из demod
        'preamble_ms': STATE.preamble_ms,
        'total_ms': STATE.total_ms,
        'rep_period_s': STATE.rep_period_s,
        'rms_dbm': STATE.rms_dbm,       # Новые метрики
        'freq_hz': STATE.freq_hz,
        't_mod': STATE.t_mod,
        'phase_data': STATE.phase_data,
        'xs_ms': STATE.xs_ms,  # Ось времени для фазы
        'xs_fm_ms': STATE.xs_fm_ms,  # deprecated
        'fm_data': STATE.fm_data,
        'fm_xs_ms': STATE.fm_xs_ms,  # Ось времени для FM
        'sdr_device_info': sdr_device_info,  # Информация об устройстве SDR
        'sdr_status': sdr_status,  # Дополнительный статус SDR
        'realtime_rms_dbm': realtime_rms,  # Real-time RMS данные
        'realtime_pulse_count': realtime_pulse_count,  # Количество обнаруженных импульсов
        'latest_pulse': latest_pulse_info,  # Информация о последнем импульсе
        'sdr_capture_active': sdr_running  # Статус захвата SDR
    })

@app.route('/api/measure', methods=['POST'])
def api_measure():
    global sdr_backend, sdr_device_info

    # Инициализируем SDR если еще не инициализирован
    if not sdr_backend:
        print("[Measure] Initializing SDR backend...")
        success = init_sdr_backend()
        if success:
            print(f"[Measure] SDR initialized: {sdr_device_info}")
        else:
            print(f"[Measure] SDR initialization failed: {sdr_device_info}")
    else:
        print(f"[Measure] SDR already initialized: {sdr_device_info}")

    # Возвращаем статус с информацией о SDR
    return jsonify({
        'status': 'measure triggered',
        'sdr_initialized': sdr_backend is not None,
        'sdr_device_info': sdr_device_info
    })

@app.route('/api/run', methods=['POST'])
def api_run():
    try:
        # Останавливаем предыдущий захват, если был запущен
        stop_sdr_capture()
        time.sleep(0.1)  # Небольшая пауза для корректного завершения

        # Запускаем новый захват
        if start_sdr_capture():
            STATE.running = True
            return jsonify({
                'status': 'running',
                'running': STATE.running,
                'message': 'Real-time SDR capture started'
            })
        else:
            STATE.running = False
            return jsonify({
                'status': 'error',
                'running': STATE.running,
                'message': 'Failed to start SDR capture'
            }), 500
    except Exception as e:
        STATE.running = False
        # Фильтруем русские символы для избежания проблем с кодировкой
        error_msg = ''.join(c if ord(c) < 128 else '?' for c in str(e))
        print(f"[RUN] Error: {error_msg}")
        return jsonify({
            'status': 'error',
            'running': STATE.running,
            'message': f'Error starting capture: {error_msg}'
        }), 500

@app.route('/api/cont', methods=['POST'])
def api_cont():
    STATE.running = True
    return jsonify({'status': 'continue', 'running': STATE.running})

@app.route('/api/break', methods=['POST'])
def api_break():
    # Останавливаем SDR захват
    stop_sdr_capture()
    STATE.running = False
    return jsonify({
        'status': 'stopped',
        'running': STATE.running,
        'message': 'SDR capture stopped'
    })

@app.route('/api/load', methods=['POST'])
def api_load():
    return jsonify({'status': 'load requested'})

@app.route('/api/save', methods=['POST'])
def api_save():
    return jsonify({'status': 'save requested'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Проверяем расширение файла
        if not file.filename.lower().endswith('.cf32'):
            return jsonify({'error': 'Only .cf32 files are allowed'}), 400

        # Безопасное имя файла
        filename = secure_filename(file.filename)

        # Создаем папку uploads если её нет
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'captures', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Путь для сохранения файла
        file_path = os.path.join(upload_dir, filename)

        # Сохраняем файл
        file.save(file_path)

        # Обновляем состояние через единую утилиту
        update_state_from_results({
            "current_file": file_path,
            "message": f"Loaded: {filename}"
        })

        print(f"File uploaded: {filename} -> {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")

        # Обрабатываем загруженный файл
        processing_result = process_cf32_file(file_path)

        if processing_result.get("success"):
            # STRICT_COMPAT: Используем единую утилиту для обновления STATE
            update_state_from_results(processing_result)

            # Дополнительное обновление сообщения через единую утилиту
            update_state_from_results({
                "message": f"Processed: {filename} - Message: {STATE.hex_message[:16]}..."
            })

            print(f"File processed successfully: {len(STATE.phase_data)} phase samples, {len(STATE.xs_fm_ms)} time samples")

        else:
            error_msg = processing_result.get("error", "Unknown error")
            # Используем единую утилиту для обновления сообщения об ошибке
            update_state_from_results({
                "message": f"Error processing {filename}: {error_msg}"
            })
            print(f"Processing error: {error_msg}")

        # Возвращаем правильный статус в зависимости от результата обработки
        if processing_result.get("success"):
            return jsonify({
                'status': 'success',
                'filename': filename,
                'size': os.path.getsize(file_path),
                'path': file_path,
                'processed': True,
                'message': STATE.message
            })
        else:
            return jsonify({
                'status': 'error',
                'error': processing_result.get("error", "Processing failed"),
                'filename': filename,
                'size': os.path.getsize(file_path),
                'path': file_path,
                'processed': False,
                'message': STATE.message
            }), 400

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

# STRICT_COMPAT: Новые API endpoints
@app.route('/api/init', methods=['POST'])
def api_init():
    """Инициализация SDR backend"""
    try:
        success = init_sdr_backend()
        return jsonify({
            'status': 'success' if success else 'failed',
            'sdr_device_info': sdr_device_info,
            'actual_sample_rate_sps': actual_sample_rate_sps,
            'iq_buffer_capacity': iq_ring_buffer.capacity if iq_ring_buffer else 0,
            'iq_buffer_duration_sec': iq_ring_buffer.duration_sec if iq_ring_buffer else 0.0
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/state')
def api_state():
    """Получение текущего состояния системы"""
    with data_lock:
        return jsonify({
            'sdr_running': sdr_running,
            'sdr_device_info': sdr_device_info,
            'current_rms_dbm': current_rms_dbm,
            'sample_counter': int(sample_counter),
            'bp_sample_counter': int(bp_sample_counter),
            'actual_sample_rate_sps': actual_sample_rate_sps,
            'iq_buffer_info': {
                'total_written': int(iq_ring_buffer.total_written) if iq_ring_buffer else 0,
                'capacity': int(iq_ring_buffer.capacity) if iq_ring_buffer else 0,
                'write_pos': int(iq_ring_buffer.write_pos) if iq_ring_buffer else 0,
                'duration_sec': iq_ring_buffer.duration_sec if iq_ring_buffer else 0.0
            },
            'pulse_status': {
                'in_pulse': in_pulse,
                'pulse_start_abs': int(pulse_start_abs)
            },
            'timestamp': time.time()
        })

@app.route('/api/last_pulse')
def api_last_pulse():
    """Получение информации о последнем импульсе и истории"""
    with data_lock:
        # STRICT_COMPAT: Подготовка истории импульсов для передачи
        history_items = []
        for pulse in list(pulse_history)[-20:]:  # Последние ≤20 импульсов
            item = pulse.copy()
            # Усекаем msg_hex для сети (64 символа + ...)
            if item.get('msg_hex') and len(item['msg_hex']) > 64:
                item['msg_hex_short'] = item['msg_hex'][:64] + '...'
            else:
                item['msg_hex_short'] = item.get('msg_hex', '')
            history_items.append(item)

        # Подготовка последнего импульса
        last = None
        if last_pulse_data:
            last = last_pulse_data.copy()
            if last.get('msg_hex') and len(last['msg_hex']) > 64:
                last['msg_hex_short'] = last['msg_hex'][:64] + '...'
            else:
                last['msg_hex_short'] = last.get('msg_hex', '')

        return jsonify({
            'last': last,
            'history': history_items,
            'pulse_queue_size': pulse_queue.qsize(),
            'timestamp': time.time()
        })

if __name__ == '__main__':
    print(">>> Starting COSPAS/SARSAT Beacon Tester v2.1")
    print(">>> Interface available at: http://127.0.0.1:8738/")
    print(">>> SDR will be initialized on Measure button press")
    print(">>> To stop: Ctrl+C")
    app.run(host='127.0.0.1', port=8738, debug=True)