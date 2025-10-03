#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beacon DSP Service (headless)
---------------------------------
Выделенный DSP-движок «как в plot», но без GUI и Flask.
- Инициализация SDR через lib.backends.safe_make_backend
- Кольцевой буфер IQ
- NCO (BB shift)
- Скользящее RMS (1 мс)
- Детектор импульсов с OFF-HANG (анти-распил)
- Вырезка сегмента (pre/post) как в plot
- PSK демодуляция и метрики (через lib.metrics/process_psk_impulse и lib.demod)
- Публикация событий через ZeroMQ PUB (status/pulse/psk)
- Приём команд через ZeroMQ REP (start/stop/set_params/save_sigmf/get_status)
- Запись JSONL-сессии (по желанию)

Зависимости (как в проекте):
  lib.backends.safe_make_backend
  lib.metrics.process_psk_impulse
  lib.demod.phase_demod_psk_msg_safe
  lib.processing_fm.fm_discriminator
  lib.logger (setup_logging, get_logger)

Быстрый старт:
  python beacon_dsp_service.py --pub tcp://127.0.0.1:8781 --rep tcp://127.0.0.1:8782

UI (web) должен подписываться на PUB и слать команды в REP.
"""
from __future__ import annotations
import os
import json
import time
import threading
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import numpy as np

try:
    import zmq  # ZeroMQ для IPC
except Exception:
    zmq = None  # Разрешаем запуск и без IPC (будет только лог)

# Проектные импорты
from lib.backends import safe_make_backend, autodetect_soapy_driver
from lib.metrics import process_psk_impulse
from lib.demod import phase_demod_psk_msg_safe
from lib.processing_fm import fm_discriminator
from lib.logger import get_logger, setup_logging, init_logger

# -----------------------
# Константы (как в plot)
# -----------------------
TARGET_SIGNAL_HZ    = 406_037_000
IF_OFFSET_HZ        = -37_000
CENTER_FREQ_HZ      = TARGET_SIGNAL_HZ + IF_OFFSET_HZ
SAMPLE_RATE_SPS     = 1_000_000
USE_MANUAL_GAIN     = True
TUNER_GAIN_DB       = 30.0
ENABLE_AGC          = False
FREQ_CORR_PPM       = 0

BB_SHIFT_ENABLE     = True
BB_SHIFT_HZ         = IF_OFFSET_HZ
RMS_WIN_MS          = 1.0
VIS_DECIM           = 2048  # влияет только на downsample истории (в сервисе — не критично)
LEVEL_HISTORY_SEC   = 12
DBM_OFFSET_DB       = -30.0
PRINT_EVERY_N_SEC   = 1
READ_CHUNK          = 65_536
PULSE_THRESH_DBM    = -45.0
PULSE_STORE_SEC     = 1.5
PSK_YLIMIT_RAD      = 1.5
PSK_BASELINE_MS     = 2.0
START_DELAY_MS      = 1.0   # обрезка начала Phase и FM (мс)
PHASE_TRIM_END_MS   = 1.0   # обрезка концовки Phase и FM (мс)
EPS                 = 1e-20
DEBUG_IMPULSE_LOG   = True

# Анти-распил
OFF_HANG_MS         = 60.0   # выдержка OFF для склейки импульса
MIN_PULSE_MS_FOR_PSK= 400.0  # отбрасываем коротыши
PRE_FRAC            = 0.25   # как в plot: pre = 0.25*dur, окно = 1.5*dur
WIN_FRAC            = 1.50
MIN_PRE_MS          = 120.0  # минимальный запас до импульса, мс

# Сервис
DEFAULT_PUB_ADDR    = "tcp://127.0.0.1:8781"
DEFAULT_REP_ADDR    = "tcp://127.0.0.1:8782"
JSONL_ENABLED       = True
ROOT                = Path.cwd()
CAPTURE_DIR         = ROOT / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

import logging
import sys

# Исправление кодировки для Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')

setup_logging("DEBUG")
log = init_logger(__name__)
#log = get_logger(__name__)
log.info("DSP service starting (root=%s)",
         logging.getLevelName(logging.getLogger().level))

class AcqState(str, Enum):
    """Состояния акквизиции"""
    STOPPED = "stopped"
    RUNNING = "running"
    RETUNING = "retuning"
    READY = "ready"

@dataclass
class Status:
    sdr: str
    fs: float
    bb_shift_hz: float
    target_signal_hz: float
    thresh_dbm: float
    read_chunk: int
    queue_depth: int
    cpu: float = 0.0

@dataclass
class PulseEvent:
    start_abs: int
    length_ms: float
    peak_dbm: float
    above_thresh_ratio: float
    # Данные для графиков phase и FR (опционально)
    phase_xs_ms: Optional[list] = None
    phase_ys_rad: Optional[list] = None
    fr_xs_ms: Optional[list] = None
    fr_ys_hz: Optional[list] = None
    markers_ms: Optional[list] = None
    preamble_ms: Optional[list] = None
    baud: Optional[float] = None

@dataclass
class PSKEvent:
    start_abs: int
    length_ms: float
    ok: bool
    preamble_ms: Optional[float]
    baud: Optional[float]
    pos_phase: Optional[float]
    neg_phase: Optional[float]
    rise_us: Optional[float]
    fall_us: Optional[float]
    asymmetry_pct: Optional[float]
    hex: Optional[str]


def detect_devices() -> Dict[str, Any]:
    """
    Определяет доступные SDR устройства.

    Returns:
        Dict с информацией о доступных устройствах:
        {
            "rtl": bool,
            "hackrf": bool,
            "airspy": bool,
            "sdrplay": bool,
            "rsa306": bool,
            "detected_backend": str | None,  # первое найденное устройство
            "available_backends": List[str]   # список всех доступных
        }
    """
    result = {
        "rtl": False,
        "hackrf": False,
        "airspy": False,
        "sdrplay": False,
        "rsa306": False,
        "detected_backend": None,
        "available_backends": []
    }

    try:
        # Пробуем использовать autodetect_soapy_driver для поиска устройств
        try:
            detected = autodetect_soapy_driver()
            result["detected_backend"] = detected
        except RuntimeError:
            # Нет устройств - это нормально для detect_devices
            detected = None
            result["detected_backend"] = None

        # Мапинг имен backend на флаги
        backend_mapping = {
            "soapy_rtl": "rtl",
            "soapy_hackrf": "hackrf",
            "soapy_airspy": "airspy",
            "soapy_sdrplay": "sdrplay",
            "rsa": "rsa306",
            "rsa306": "rsa306"
        }

        # Устанавливаем флаг для найденного устройства
        device_key = backend_mapping.get(detected)
        if device_key:
            result[device_key] = True
            result["available_backends"].append(detected)

        # Попробуем дополнительно проверить каждое устройство отдельно
        try:
            import SoapySDR

            # Проверяем доступные драйверы через SoapySDR
            for backend_name, device_key in backend_mapping.items():
                if not backend_name.startswith("soapy_"):
                    continue

                driver = backend_name.replace("soapy_", "")
                try:
                    devices = SoapySDR.Device.enumerate({"driver": driver})
                    if devices:
                        result[device_key] = True
                        if backend_name not in result["available_backends"]:
                            result["available_backends"].append(backend_name)
                except Exception:
                    pass

        except Exception:
            # SoapySDR не доступен - используем только результат autodetect
            pass

        # Проверяем RSA306 отдельно
        try:
            import ctypes as ct
            dll_path = os.getenv("RSA_API_DLL", r"C:/Tektronix/RSA_API/lib/x64/RSA_API.dll")
            if os.path.exists(dll_path):
                L = ct.CDLL(dll_path)
                num = ct.c_int(0)
                ids = ct.POINTER(ct.c_int)()
                sns = ct.POINTER(ct.c_wchar_p)()
                tys = ct.POINTER(ct.c_wchar_p)()
                L.DEVICE_SearchIntW.argtypes = [ct.POINTER(ct.c_int),
                                                ct.POINTER(ct.POINTER(ct.c_int)),
                                                ct.POINTER(ct.POINTER(ct.c_wchar_p)),
                                                ct.POINTER(ct.POINTER(ct.c_wchar_p))]
                L.DEVICE_SearchIntW.restype = ct.c_int
                rc = L.DEVICE_SearchIntW(ct.byref(num), ct.byref(ids), ct.byref(sns), ct.byref(tys))
                if rc == 0 and num.value > 0:
                    result["rsa306"] = True
                    result["available_backends"].append("rsa306")
                    if not result["detected_backend"]:
                        result["detected_backend"] = "rsa306"
        except Exception:
            pass

    except Exception as e:
        # В случае ошибки возвращаем пустой результат
        try:
            print(f"Warning: device detection failed: {str(e)}")
        except:
            print("Warning: device detection failed")

    return result


def select_auto_backend(file_path: Optional[str] = None) -> str:
    """
    Выбирает backend в режиме AUTO по приоритету.

    Args:
        file_path: Путь к файлу (если задан)

    Returns:
        Имя backend для использования

    Raises:
        RuntimeError: Если ни одно устройство не найдено и файл не задан
    """
    devices = detect_devices()

    # Приоритет выбора: RTL -> HackRF -> Airspy -> SDRplay -> RSA306
    priority_order = ["rtl", "hackrf", "airspy", "sdrplay", "rsa306"]
    backend_mapping = {
        "rtl": "soapy_rtl",
        "hackrf": "soapy_hackrf",
        "airspy": "soapy_airspy",
        "sdrplay": "soapy_sdrplay",
        "rsa306": "rsa306"
    }

    # Ищем первое доступное устройство по приоритету
    for device_type in priority_order:
        if devices[device_type]:
            return backend_mapping[device_type]

    # Если SDR не найдено, но задан файл - используем file backend
    if file_path:
        return "file"

    # Ничего не найдено
    raise RuntimeError("No SDR found and no file provided")


# ============================================================================
# Утилиты для slice API (offset/span с поддержкой units)
# ============================================================================

def normalize_slice(
    offset: Any,
    span: Any,
    units: str,
    fs: float,
    n_total: int
) -> Tuple[int, int]:
    """
    Конвертирует offset/span в абсолютные индексы сэмплов.

    Parameters
    ----------
    offset : int | float | str
        Начало среза. Может быть:
        - int: индекс в сэмплах (если units="samples")
        - str: "123", "10ms", "500us", "50%"
    span : int | float | str
        Длина среза (аналогично offset)
    units : str
        "samples" | "time" | "%" | "auto"
    fs : float
        Частота дискретизации (Sa/s)
    n_total : int
        Общая длина данных в сэмплах

    Returns
    -------
    (i0, i1) : Tuple[int, int]
        Индексы начала и конца (i1 не включается)
        Гарантируется: 0 <= i0 < i1 <= n_total
    """

    def parse_value(val: Any, unit_mode: str) -> int:
        """Парсинг одного значения offset или span в сэмплы."""
        # Если уже int
        if isinstance(val, int):
            return val

        # Если float - округляем
        if isinstance(val, float):
            return int(round(val))

        # Если строка
        if isinstance(val, str):
            val = val.strip()

            # Проценты
            if val.endswith('%'):
                percent = float(val[:-1])
                return int(round(percent / 100.0 * n_total))

            # Время с суффиксом
            if val.endswith('us'):
                t_us = float(val[:-2])
                return int(round(t_us * 1e-6 * fs))
            if val.endswith('ms'):
                t_ms = float(val[:-2])
                return int(round(t_ms * 1e-3 * fs))
            if val.endswith('s'):
                t_s = float(val[:-1])
                return int(round(t_s * fs))

            # Без суффикса
            try:
                num = float(val)
                # Если units="auto" и нет суффикса -> сэмплы
                # Если units="samples" -> сэмплы
                # Если units="time" -> секунды
                # Если units="%" -> проценты
                if unit_mode in ("auto", "samples"):
                    return int(round(num))
                elif unit_mode == "time":
                    # По умолчанию секунды
                    return int(round(num * fs))
                elif unit_mode == "%":
                    return int(round(num / 100.0 * n_total))
                else:
                    return int(round(num))
            except ValueError:
                raise ValueError(f"Cannot parse value: {val}")

        raise TypeError(f"Unsupported type for slice value: {type(val)}")

    # Парсим offset и span
    offset_samp = parse_value(offset, units)
    span_samp = parse_value(span, units)

    # Минимальный span = 2
    if span_samp < 2:
        span_samp = 2

    # Вычисляем i0, i1
    i0 = offset_samp
    i1 = i0 + span_samp

    # Обрезка к допустимым границам
    # Разрешаем отрицательный offset (преролл), но обрезаем к [0, n_total]
    if i0 < 0:
        i0 = 0
    if i0 >= n_total:
        i0 = n_total - 1

    if i1 > n_total:
        i1 = n_total
    if i1 <= i0:
        i1 = i0 + 1

    return (i0, i1)


def apply_downsample(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | str,
    method: str = "decimate"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Даунсэмплинг данных до max_samples точек.

    Parameters
    ----------
    x : np.ndarray
        Массив X координат
    y : np.ndarray
        Массив Y значений
    max_samples : int | "all"
        Максимальное количество точек. "all" = без ограничений
    method : str
        "decimate" - равномерная прорежка
        "minmax" - бакетизация с сохранением min/max пиков

    Returns
    -------
    (x_out, y_out) : Tuple[np.ndarray, np.ndarray]
        Даунсэмплированные массивы
    """
    if max_samples == "all" or len(x) <= max_samples:
        return x, y

    n = len(x)
    max_samples = int(max_samples)

    if method == "decimate":
        # Равномерная прорежка
        step = n / max_samples
        indices = np.floor(np.arange(max_samples) * step).astype(int)
        indices = np.clip(indices, 0, n - 1)
        return x[indices], y[indices]

    elif method == "minmax":
        # Бакетизация с сохранением пиков
        # На каждый бакет выдаём 2 точки (min, max)
        n_buckets = max_samples // 2
        if n_buckets < 1:
            n_buckets = 1

        bucket_size = n / n_buckets
        x_out = []
        y_out = []

        for i in range(n_buckets):
            i0 = int(np.floor(i * bucket_size))
            i1 = int(np.floor((i + 1) * bucket_size))
            if i1 > n:
                i1 = n
            if i1 <= i0:
                i1 = i0 + 1

            bucket_y = y[i0:i1]
            if len(bucket_y) == 0:
                continue

            # Находим индексы min и max
            local_min_idx = np.argmin(bucket_y)
            local_max_idx = np.argmax(bucket_y)

            # Глобальные индексы
            idx_min = i0 + local_min_idx
            idx_max = i0 + local_max_idx

            # Добавляем в порядке появления
            if idx_min < idx_max:
                x_out.extend([x[idx_min], x[idx_max]])
                y_out.extend([y[idx_min], y[idx_max]])
            else:
                x_out.extend([x[idx_max], x[idx_min]])
                y_out.extend([y[idx_max], y[idx_min]])

        return np.array(x_out), np.array(y_out)

    else:
        raise ValueError(f"Unknown downsample method: {method}")


class BeaconDSPService:
    def __init__(self,
                 pub_addr: str = DEFAULT_PUB_ADDR,
                 rep_addr: str = DEFAULT_REP_ADDR,
                 backend_name: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 jsonl_path: Optional[Path] = None):
        self.pub_addr = pub_addr
        self.rep_addr = rep_addr
        self.backend_name = backend_name
        self.backend_args = backend_args

        self.backend = None
        self.sample_rate = float(SAMPLE_RATE_SPS)
        self.nco_phase = 0.0
        self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))
        self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))

        # Эффективные значения BB shift (могут отличаться от констант для file-режима)
        self.effective_bb_shift_enable = BB_SHIFT_ENABLE
        self.effective_bb_shift_hz = BB_SHIFT_HZ

        # Кольцевой буфер и история RMS
        self.store_max_samps = int(PULSE_STORE_SEC * self.sample_rate)
        self.samples_start_abs = 0
        self.full_samples = np.empty(0, dtype=np.complex64)
        self.full_idx = np.empty(0, dtype=np.int64)
        self.full_rms = np.empty(0, dtype=np.float32)
        self.tail_p = np.empty(0, dtype=np.float32)
        self.sample_counter = 0

        # Детектор
        self.in_pulse = False
        self.pulse_start_abs: Optional[int] = None
        self.last_rms_dbm = float("-inf")
        self.last_iq_seg: Optional[np.ndarray] = None
        self.last_core_gate: Optional[Tuple[int, int]] = None
        self.last_msg_hex: Optional[str] = None
        self.last_phase_metrics: Optional[Dict[str, Any]] = None
        self.last_impulse_freq_hz: float = 0.0

        # Очередь событий для UI (если нужно)
        self.pulse_queue: deque = deque(maxlen=8)

        # Потоки
        self._reader_stop = False  # Управляет только reader_loop
        self._service_stop = False  # Управляет только REP/PUB сервисом
        self.reader_thread: Optional[threading.Thread] = None
        # RLock на случай вложенных вызовов (start/stop из-под локов и т.п.)
        self._lock = threading.RLock()

        # Воркер для тяжелых операций
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dsp_worker")

        # Менеджмент состояния
        self._acq_state = AcqState.READY
        self._in_flight_operation = False  # Защита от дребезга

        # IPC
        self.ctx = zmq.Context.instance() if zmq else None
        self.pub = None
        self.rep = None
        if self.ctx:
            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.bind(self.pub_addr)
            self.rep = self.ctx.socket(zmq.REP)
            self.rep.bind(self.rep_addr)
            self.rep_thread = threading.Thread(target=self._rep_loop, daemon=True)
            self.rep_thread.start()
        else:
            log.warning("ZeroMQ не установлен — IPC отключён")

        # JSONL
        self.jsonl_path = jsonl_path or (CAPTURE_DIR / time.strftime("dsp_%Y%m%d.jsonl"))
        self.jsonl_fp = open(self.jsonl_path, "a", encoding="utf-8") if JSONL_ENABLED else None

        # Стартуем backend
        self._make_backend()
        # Автостарт только если backend успешно создан
        if self.backend is not None:
            self.start()
        else:
            # В режиме file_wait или если backend не создан - остаёмся в READY
            with self._lock:
                self._acq_state = AcqState.READY

    # ---------------- Backend ----------------
    def _make_backend(self):
        name = self.backend_name or os.environ.get("BACKEND_NAME", None) or "auto"
        args = self.backend_args

        # Обработка AUTO режима
        if name.lower() == "auto":
            try:
                # Извлекаем file_path из args если есть
                file_path = None
                if isinstance(args, dict):
                    file_path = args.get("path") or args.get("file_path")
                elif isinstance(args, str):
                    file_path = args

                # Используем нашу функцию автовыбора
                name = select_auto_backend(file_path)
                log.info(f"AUTO mode selected backend: {name}")
            except RuntimeError as e:
                log.error(f"AUTO backend selection failed: {e}")
                # Fallback на file_wait режим
                name = "file"
                args = {"path": ""}

        try:
            # Используем safe_make_backend с file_wait fallback
            self.backend = safe_make_backend(
                name,
                sample_rate=SAMPLE_RATE_SPS,
                center_freq=float(CENTER_FREQ_HZ),
                gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
                agc=bool(ENABLE_AGC),
                corr_ppm=int(FREQ_CORR_PPM),
                device_args=args,
                if_offset_hz=IF_OFFSET_HZ,
                on_fail="file_wait"  # Мягкий fallback на ожидание файла
            )
            if self.backend is None:
                # Backend в режиме ожидания файла
                log.warning("Backend in file_wait mode - waiting for CF32 file")
                # Отключаем BB shift для file_wait режима
                self.effective_bb_shift_enable = False
                self.effective_bb_shift_hz = 0.0
                self.sample_rate = float(SAMPLE_RATE_SPS)
                self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
                self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))
            else:
                st = self.backend.get_status() or {}
                self.sample_rate = float(st.get("actual_sample_rate_sps",
                                               getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS)))
                self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
                self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))
                log.info("\n=== BACKEND STATUS ===\n" + self.backend.pretty_status() + "\n======================\n")
        except Exception as e:
            log.error(f"Backend init failed: {e}")
            # Fallback на file_wait режим
            log.warning("Fallback to file_wait mode - no backend available")
            try:
                self.backend = safe_make_backend(
                    "file",
                    sample_rate=SAMPLE_RATE_SPS,
                    center_freq=CENTER_FREQ_HZ,
                    if_offset_hz=IF_OFFSET_HZ,
                    on_fail="file_wait"
                )
            except Exception:
                self.backend = None
            self.sample_rate = float(SAMPLE_RATE_SPS)
            self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
            self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))

    # ---------------- Threads ----------------
    def start(self):
        """Idempotent start: запускает акквизицию если она еще не идет"""
        with self._lock:
            if self.reader_thread and self.reader_thread.is_alive():
                # Уже запущено
                if self._acq_state != AcqState.RUNNING:
                    self._acq_state = AcqState.RUNNING
                return
            self._reader_stop = False
            self._acq_state = AcqState.RUNNING
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

    def stop(self):
        """Idempotent stop: останавливает акквизицию если она идет"""
        # Проверяем состояние БЕЗ блокировки join
        thread_to_stop = None
        with self._lock:
            if not self.reader_thread or not self.reader_thread.is_alive():
                # Уже остановлено
                self._acq_state = AcqState.STOPPED
                return
            self._reader_stop = True
            self._acq_state = AcqState.STOPPED
            thread_to_stop = self.reader_thread

        # Join БЕЗ lock чтобы избежать deadlock
        if thread_to_stop:
            thread_to_stop.join(timeout=1.0)
            if thread_to_stop.is_alive():
                log.warning("Reader thread did not stop within timeout")
        if self.jsonl_fp:
            self.jsonl_fp.flush()

    def shutdown(self):
        """Корректное завершение всего сервиса (вызывается при KeyboardInterrupt)"""
        log.info("Shutting down DSP service...")
        with self._lock:
            self._service_stop = True

        # Останавливаем reader
        try:
            self.stop()
        except Exception as e:
            log.error(f"Error stopping reader: {e}")

        # Закрываем ZMQ сокеты
        try:
            if self.rep:
                self.rep.close(linger=0)
            if self.pub:
                self.pub.close(linger=0)
            if self.ctx:
                self.ctx.term()
        except Exception as e:
            log.error(f"Error closing ZMQ: {e}")

        # Завершаем executor
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            log.error(f"Error shutting down executor: {e}")

        # Закрываем JSONL
        try:
            if self.jsonl_fp:
                self.jsonl_fp.close()
        except Exception as e:
            log.error(f"Error closing JSONL: {e}")

        log.info("DSP service shutdown complete")

    # ---------------- Reader loop ----------------
    def _read_block(self, nsamps: int) -> np.ndarray:
        if self.backend is None:
            # В режиме ожидания файла возвращаем пустой массив
            return np.array([], dtype=np.complex64)
        return self.backend.read(nsamps)

    def _append_samples(self, x: np.ndarray):
        if self.full_samples.size == 0:
            self.full_samples = x.copy()
            self.samples_start_abs = self.sample_counter
        else:
            self.full_samples = np.concatenate((self.full_samples, x))
        if self.full_samples.size > self.store_max_samps:
            cut = self.full_samples.size - self.store_max_samps
            self.full_samples = self.full_samples[cut:]
            self.samples_start_abs += cut

    def _reader_loop(self):
        last_stat = 0.0
        while not self._reader_stop:
            try:
                samples = self._read_block(READ_CHUNK)
                if samples.size == 0:
                    time.sleep(0.001)
                    continue
            except Exception as e:
                log.warning(f"read error: {e}")
                time.sleep(0.005)
                continue
            try:
                self._process_samples(samples)
            except Exception as e:
                log.warning(f"process error: {e}")

            # периодический статус
            now = time.time()
            if now - last_stat >= 0.5:
                last_stat = now
                self._emit_status()

    # ---------------- Core processing ----------------
    @staticmethod
    def _db10(x: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(x, EPS))

    def _emit(self, typ: str, payload: Dict[str, Any]):
        obj = {"type": typ, **payload}
        line = json.dumps(obj, ensure_ascii=False)
        if self.jsonl_fp:
            self.jsonl_fp.write(line + "\n")
        if self.pub:
            self.pub.send_string(line)
        else:
            # fallback: логируем
            if typ in ("pulse", "psk"):
                log.info(line)
            else:
                log.debug(line)

    def _emit_status(self):
        # Готовим базовый Status
        st = Status(
            sdr=str(getattr(self.backend, "driver", "?")),
            fs=float(self.sample_rate),
            bb_shift_hz=float(self.effective_bb_shift_hz),
            target_signal_hz=float(TARGET_SIGNAL_HZ),
            thresh_dbm=float(PULSE_THRESH_DBM),
            read_chunk=int(READ_CHUNK),
            queue_depth=len(self.pulse_queue),
        )

        # Добавляем поля t_s и last_rms_dbm для Sliding RMS
        payload = asdict(st)

        # Добавляем новые поля согласно ТЗ
        with self._lock:
            backend_name = self.backend_name or "auto"
            if self.backend is None:
                backend_name = "file_wait"
            elif hasattr(self.backend, 'backend_name'):
                backend_name = self.backend.backend_name

            backend_args_str = ""
            if isinstance(self.backend_args, str):
                backend_args_str = self.backend_args
            elif isinstance(self.backend_args, dict) and "path" in self.backend_args:
                backend_args_str = self.backend_args["path"]

            payload["backend"] = backend_name
            payload["backend_args"] = backend_args_str
            payload["acq_state"] = str(self._acq_state.value)
            payload["ready"] = (self.backend is not None and self._acq_state in (AcqState.READY, AcqState.RUNNING))

        with self._lock:
            if self.full_idx.size > 0 and self.full_rms.size > 0:
                # Даунсэмплинг до ≤1000 точек для веб-интерфейса
                max_points = 1000
                if self.full_idx.size <= max_points:
                    # Используем все точки
                    t_s_array = (self.full_idx - self.full_idx[-1]) / self.sample_rate
                    rms_dbm_array = self.full_rms
                else:
                    # Равномерный даунсэмплинг
                    step = self.full_idx.size // max_points
                    indices = np.arange(0, self.full_idx.size, step)
                    t_s_array = (self.full_idx[indices] - self.full_idx[-1]) / self.sample_rate
                    rms_dbm_array = self.full_rms[indices]

                payload["t_s"] = t_s_array.tolist()
                payload["last_rms_dbm"] = rms_dbm_array.tolist()
            else:
                payload["t_s"] = []
                payload["last_rms_dbm"] = []

        self._emit("status", payload)

    def _process_samples(self, samples: np.ndarray):
        base_idx = self.sample_counter
        x = samples.copy()

        # BB shift (используем эффективные значения)
        if self.effective_bb_shift_enable and abs(self.effective_bb_shift_hz) > 0:
            n = np.arange(x.size, dtype=np.float64)
            mixer = np.exp(1j * (self.nco_phase + self.nco_k * n)).astype(np.complex64)
            x *= mixer
            self.nco_phase = float((self.nco_phase + self.nco_k * x.size) % (2.0 * np.pi))

        with self._lock:
            self._append_samples(x)

        # Мощность кадра
        p_block = (np.abs(x) ** 2)

        # Контекст для скользящего окна
        if self.tail_p.size:
            p_cont = np.concatenate((self.tail_p, p_block))
            p_cont_start_idx = base_idx - self.tail_p.size
        else:
            p_cont = p_block
            p_cont_start_idx = base_idx

        if p_cont.size >= self.win_samps:
            c = np.cumsum(p_cont, dtype=np.float64)
            S_valid = c[self.win_samps - 1:] - np.concatenate(([0.0], c[:-self.win_samps]))
            P_win = S_valid / float(self.win_samps)
            calib_offset_db = self.backend.get_calib_offset_db() if self.backend else 0.0
            rms_dbm_vec = self._db10(P_win) + DBM_OFFSET_DB + calib_offset_db
            idx_end = p_cont_start_idx + (self.win_samps - 1) + np.arange(rms_dbm_vec.size, dtype=np.int64)

            # Храним историю индексов/RMS, срез по хвосту
            with self._lock:
                self.full_idx = np.concatenate((self.full_idx, idx_end))
                self.full_rms = np.concatenate((self.full_rms, rms_dbm_vec.astype(np.float32)))
                if self.full_idx.size > 0:
                    newest_idx = self.full_idx[-1]
                    m_keep = (self.full_idx >= (newest_idx - self.store_max_samps))
                    self.full_idx = self.full_idx[m_keep]
                    self.full_rms = self.full_rms[m_keep]

            # Детектор с OFF-HANG: сначала «сырые» переходы
            on = rms_dbm_vec >= PULSE_THRESH_DBM
            trans = np.diff(on.astype(np.int8), prepend=on[0])
            start_pos = np.where(trans == 1)[0]
            end_pos = np.where(trans == -1)[0] - 1

            # Склейка OFF (anti-chop): если пауза < OFF_HANG_MS → объединяем
            hang_samps = int(round(self.sample_rate * (OFF_HANG_MS * 1e-3)))
            pairs: list[Tuple[int, int]] = []

            # Собираем «сырые» кандидаты (учитывая текущий незакрытый импульс)
            pending_start: Optional[int] = (self.pulse_start_abs if self.in_pulse else None)
            s_idx = 0
            e_idx = 0
            while True:
                if pending_start is None:
                    if s_idx < start_pos.size:
                        pending_start = int(idx_end[start_pos[s_idx]])
                        s_idx += 1
                    else:
                        break
                found_end = None
                while e_idx < end_pos.size:
                    cand_end_abs = int(idx_end[end_pos[e_idx]])
                    e_idx += 1
                    if cand_end_abs >= pending_start:
                        found_end = cand_end_abs
                        break
                if found_end is None:
                    # нет конца — остаёмся «в импульсе»
                    self.in_pulse = True
                    self.pulse_start_abs = pending_start
                    break
                # есть окончание — добавим как сегмент (временно)
                pairs.append((pending_start, found_end))
                pending_start = None
                self.in_pulse = False
                self.pulse_start_abs = None

            # Склейка по OFF_HANG
            if pairs:
                merged: list[Tuple[int, int]] = [pairs[0]]
                for s, e in pairs[1:]:
                    prev_s, prev_e = merged[-1]
                    gap = s - prev_e - 1
                    if gap <= hang_samps:
                        # склеиваем
                        merged[-1] = (prev_s, e)
                    else:
                        merged.append((s, e))
                pairs = merged

            # Обработка пар
            for start_abs, found_end in pairs:
                duration_samps = max(1, found_end - start_abs + 1)
                dur_ms = 1000.0 * duration_samps / float(self.sample_rate)

                # Фильтр длительности
                if dur_ms < MIN_PULSE_MS_FOR_PSK:
                    continue

                # Статистика надпороговых
                try:
                    m = (idx_end >= start_abs) & (idx_end <= found_end)
                    ys = rms_dbm_vec[m]
                    above = np.count_nonzero(ys >= PULSE_THRESH_DBM)
                    ratio = float(above) / max(1, ys.size)
                    peak = float(np.nanmax(ys)) if ys.size else float("nan")
                except Exception:
                    ratio, peak = float("nan"), float("nan")

                # Сохраняем базовое событие pulse для последующего обогащения данными
                pulse_event_data = {
                    "start_abs": int(start_abs),
                    "length_ms": float(dur_ms),
                    "peak_dbm": peak,
                    "above_thresh_ratio": ratio
                }

                # Окно вырезки «как в plot»
                win_len = int(round(duration_samps * WIN_FRAC))
                pre = int(round(duration_samps * PRE_FRAC))
                # Минимальный pre в сэмплах
                min_pre = int(round(self.sample_rate * (MIN_PRE_MS * 1e-3)))
                pre = max(pre, min_pre)

                win_start = max(0, start_abs - pre)
                win_end = win_start + win_len

                with self._lock:
                    seg_start_rel = win_start - self.samples_start_abs
                    seg_end_rel = win_end - self.samples_start_abs
                    seg_start_rel = max(0, seg_start_rel)
                    seg_end_rel = min(self.full_samples.size, seg_end_rel)
                    if seg_end_rel - seg_start_rel <= 8:
                        continue
                    seg = self.full_samples[seg_start_rel:seg_end_rel].astype(np.complex64, copy=False)
                    self.last_iq_seg = seg.copy()

                # Гейт ядра для спектра (по порогу)
                g0 = max(0, int(start_abs - win_start))
                g1 = int(found_end - win_start + 1)
                g1 = min(g1, int(seg.size))
                if g1 - g0 < 8:
                    g1 = min(int(seg.size), g0 + 8)
                self.last_core_gate = (int(g0), int(g1))

                # Trim на края (1 мс, не более 1/4 длительности)
                trim_samps = int(self.sample_rate * 1e-3)
                trim_samps = min(trim_samps, duration_samps // 4)

                freq_start_abs = start_abs + trim_samps
                freq_end_abs   = found_end - trim_samps
                freq_seg_start_rel = max(0, freq_start_abs - self.samples_start_abs)
                freq_seg_end_rel   = min(self.full_samples.size, freq_end_abs - self.samples_start_abs)
                freq_seg = seg
                if freq_seg_end_rel - freq_seg_start_rel >= 8:
                    with self._lock:
                        freq_seg = self.full_samples[freq_seg_start_rel:freq_seg_end_rel].astype(np.complex64, copy=False)

                # Частота по фазе
                if freq_seg.size >= 8:
                    phase_diff = np.angle(freq_seg[1:] * np.conj(freq_seg[:-1]))
                    self.last_impulse_freq_hz = float(phase_diff.mean() * self.sample_rate / (2*np.pi))

                # PSK демодуляция/метрики
                # Используем process_psk_impulse как в plot (без LPF можно оставить use_lpf_decim=True)
                try:
                    res = process_psk_impulse(
                        iq_seg=freq_seg,
                        fs=self.sample_rate,
                        baseline_ms=PSK_BASELINE_MS,
                        t0_offset_ms=0.0,
                        use_lpf_decim=True,
                        remove_slope=True,
                    )
                    xs_ms = res.get("xs_ms")
                    phase_rad = res.get("phase_rad")

                    # FM для справки
                    fm_out = fm_discriminator(
                        iq=freq_seg,
                        fs=self.sample_rate,
                        pre_lpf_hz=50_000,
                        decim=4,
                        smooth_hz=2_000,
                        detrend=True,
                        center=True,
                        fir_taps=127,
                    )

                    # Декод сообщения
                    msg_hex, phase_metrics, edges = phase_demod_psk_msg_safe(data=phase_rad)
                    FSd = self.sample_rate / 4.0
                    baud = 400.75  # если phase_metrics вернёт Tmod, можно пересчитать
                    try:
                        tmod = float(phase_metrics.get("Tmod", float("nan")))
                        if tmod == tmod and tmod > 0:
                            baud = float(FSd / tmod)
                    except Exception:
                        pass

                    # Метрики
                    def _getf(d, k, default=float("nan")):
                        try:
                            return float(d.get(k, default))
                        except Exception:
                            return float("nan")
                    pos = _getf(phase_metrics, "PosPhase")
                    neg = _getf(phase_metrics, "NegPhase")
                    PhRise = _getf(phase_metrics, "PhRise")
                    PhFall = _getf(phase_metrics, "PhFall")
                    ass = _getf(phase_metrics, "Ass")
                    rise_us = (PhRise / FSd * 1e6) if (PhRise == PhRise) else float("nan")
                    fall_us = (PhFall / FSd * 1e6) if (PhFall == PhFall) else float("nan")

                    # Длительности
                    msg_dur_ms = float(xs_ms[-1] - xs_ms[0]) if (xs_ms is not None and len(xs_ms) >= 2) else float("nan")
                    carrier_ms = float(edges[0] / FSd * 1e3) if (edges is not None and len(edges) > 0) else float("nan")

                    # Резюме для Params snapshot
                    power_rms_dbm = float("nan")
                    try:
                        # средняя мощность по окну импульса
                        m = (self.full_idx >= start_abs) & (self.full_idx <= found_end)
                        ys = self.full_rms[m]
                        if ys.size:
                            _mw = 10.0 ** (ys / 10.0)
                            power_rms_dbm = 10.0 * np.log10(float(np.nanmean(_mw)))
                    except Exception:
                        pass

                    self.last_phase_metrics = {
                        "Target Signal (Hz)": float(TARGET_SIGNAL_HZ),
                        "Frequency (Hz)": float(CENTER_FREQ_HZ + IF_OFFSET_HZ),
                        "Frequency Offset (Hz)": float((CENTER_FREQ_HZ + IF_OFFSET_HZ) - TARGET_SIGNAL_HZ),
                        "Message Duration (ms)": float(msg_dur_ms),
                        "Carrier Duration (ms)": float(carrier_ms),
                        "Pos (rad)": pos,
                        "Neg (rad)": neg,
                        "Rise (μs)": rise_us,
                        "Fall (μs)": fall_us,
                        "Asymmetry (%)": ass,
                        "Fmod (Hz)": float(FSd / tmod) if ('tmod' in locals() and tmod and tmod==tmod) else float("nan"),
                        "Power (RMS, dBm)": power_rms_dbm,
                    }
                    self.last_msg_hex = str(msg_hex)

                    # Событие psk
                    # Даунсэмплинг для веб-визуализации (max 1000 точек)
                    def downsample_data(xs, ys, max_points=1000):
                        if xs is None or ys is None or len(xs) <= max_points:
                            return xs, ys
                        # Простой равномерный даунсэмплинг
                        step = len(xs) // max_points
                        if step <= 1:
                            return xs, ys
                        xs_down = xs[::step]
                        ys_down = ys[::step]
                        return xs_down.tolist(), ys_down.tolist()

                    # Подготавливаем данные фазы для графика
                    phase_xs_down, phase_ys_down = downsample_data(xs_ms, phase_rad)

                    # Подготавливаем данные частоты для графика (из FM дискриминатора)
                    fr_xs_ms = None
                    fr_ys_hz = None
                    if fm_out is not None and "xs_ms" in fm_out and "freq_hz" in fm_out:
                        fr_xs_down, fr_ys_down = downsample_data(
                            fm_out["xs_ms"],
                            fm_out["freq_hz"]
                        )
                        fr_xs_ms = fr_xs_down
                        fr_ys_hz = fr_ys_down

                    # Маркеры битов/полубитов из edges
                    markers_ms = None
                    if edges is not None and len(edges) > 0:
                        # Преобразуем индексы фронтов в миллисекунды
                        markers_ms = [float(edge / FSd * 1e3) for edge in edges[:100]]  # Ограничиваем количество

                    # Подготавливаем RMS данные для графика
                    rms_xs_ms = None
                    rms_ys_dbm = None
                    if phase_xs_down is not None:
                        try:
                            # Получаем RMS данные для времени импульса
                            with self._lock:
                                m = (self.full_idx >= start_abs) & (self.full_idx <= found_end)
                                if np.any(m):
                                    pulse_rms_values = self.full_rms[m]
                                    pulse_idx_values = self.full_idx[m]
                                    # Преобразуем индексы в миллисекунды относительно начала импульса
                                    pulse_t_ms = ((pulse_idx_values - start_abs) / self.sample_rate) * 1000.0

                                    # Даунсэмплинг RMS до ≤1000 точек как и для фазы
                                    rms_xs_down, rms_ys_down = downsample_data(pulse_t_ms, pulse_rms_values)
                                    rms_xs_ms = rms_xs_down
                                    rms_ys_dbm = rms_ys_down
                        except Exception as e:
                            log.debug(f"Failed to extract RMS data for pulse: {e}")

                    # Обогащаем pulse событие данными для графиков
                    pulse_event_data["phase_xs_ms"] = phase_xs_down
                    pulse_event_data["phase_ys_rad"] = phase_ys_down
                    pulse_event_data["fr_xs_ms"] = fr_xs_ms
                    pulse_event_data["fr_ys_hz"] = fr_ys_hz
                    pulse_event_data["rms_xs_ms"] = rms_xs_ms
                    pulse_event_data["rms_ys_dbm"] = rms_ys_dbm
                    pulse_event_data["markers_ms"] = markers_ms
                    pulse_event_data["preamble_ms"] = [0, float(carrier_ms)] if carrier_ms == carrier_ms else None
                    pulse_event_data["baud"] = float(baud) if baud == baud else None

                    # Добавляем данные для кнопок UI (Параметры/Сообщение/Спектр)
                    pulse_event_data["phase_metrics"] = self.last_phase_metrics
                    pulse_event_data["msg_hex"] = self.last_msg_hex
                    # IQ сегмент импульса и core gate (уже сохранены в self.last_iq_seg и self.last_core_gate)
                    if self.last_iq_seg is not None and self.last_iq_seg.size > 0:
                        # Конвертируем complex в пары [real, imag] для JSON сериализации
                        iq_real = np.real(self.last_iq_seg).tolist()
                        iq_imag = np.imag(self.last_iq_seg).tolist()
                        # Чередуем real и imag: [r0, i0, r1, i1, ...]
                        iq_interleaved = []
                        for r, i in zip(iq_real, iq_imag):
                            iq_interleaved.extend([r, i])
                        pulse_event_data["iq_seg"] = iq_interleaved
                        pulse_event_data["core_gate"] = list(self.last_core_gate) if self.last_core_gate else None
                    else:
                        pulse_event_data["iq_seg"] = None
                        pulse_event_data["core_gate"] = None

                    # Сохраняем последние данные для REP запросов
                    self.last_pulse_data = pulse_event_data.copy()

                    # Отправляем обогащенное pulse событие
                    self._emit("pulse", pulse_event_data)

                    # Отправляем PSK событие
                    self._emit("psk", asdict(PSKEvent(
                        start_abs=int(start_abs),
                        length_ms=float(dur_ms),
                        ok=bool(msg_hex not in (None, "None")),
                        preamble_ms=float(carrier_ms) if carrier_ms == carrier_ms else None,
                        baud=float(baud) if baud == baud else None,
                        pos_phase=pos if pos == pos else None,
                        neg_phase=neg if neg == neg else None,
                        rise_us=rise_us if rise_us == rise_us else None,
                        fall_us=fall_us if fall_us == fall_us else None,
                        asymmetry_pct=ass if ass == ass else None,
                        hex=str(msg_hex) if msg_hex not in (None, "None") else None,
                    )))

                except Exception as e:
                    log.info(f"PSK демодуляция пропущена: {e}")
                    # Отправляем базовое pulse событие без данных фазы
                    self._emit("pulse", pulse_event_data)

        # Хвост для скользящего окна
        need = max(0, self.win_samps - 1)
        k = min(need, p_cont.size)
        self.tail_p = p_cont[-k:].copy() if k > 0 else np.empty(0, dtype=np.float32)
        self.sample_counter += samples.size

    # ---------------- REP server ----------------
    def _rep_loop(self):
        global PULSE_THRESH_DBM, RMS_WIN_MS, TARGET_SIGNAL_HZ
        while not self._service_stop:
            try:
                msg = self.rep.recv_json(flags=0)
            except Exception:
                time.sleep(0.01)
                continue

            # Трекинг времени выполнения команды
            t_start = time.time()
            cmd = str(msg.get("cmd", "")).lower()

            try:
                # === МГНОВЕННЫЕ КОМАНДЫ (instant ACK) ===

                if cmd == "echo":
                    # Эхо для проверки канала
                    payload = msg.get("payload", {})
                    ack_ms = (time.time() - t_start) * 1000
                    log.debug(f"cmd=echo ack={ack_ms:.1f}ms")
                    self.rep.send_json({"ok": True, "echo": payload})

                elif cmd == "get_status":
                    # Готовим базовый Status
                    st = Status(
                        sdr=str(getattr(self.backend, "driver", "?")),
                        fs=float(self.sample_rate),
                        bb_shift_hz=float(self.effective_bb_shift_hz),
                        target_signal_hz=float(TARGET_SIGNAL_HZ),
                        thresh_dbm=float(PULSE_THRESH_DBM),
                        read_chunk=int(READ_CHUNK),
                        queue_depth=len(self.pulse_queue),
                    )

                    # Добавляем поля t_s и last_rms_dbm для Sliding RMS (как в _emit_status)
                    payload = asdict(st)

                    # Добавляем новые поля согласно ТЗ
                    with self._lock:
                        backend_name = self.backend_name or "auto"
                        if self.backend is None:
                            backend_name = "file_wait"
                        elif hasattr(self.backend, 'backend_name'):
                            backend_name = self.backend.backend_name

                        backend_args_str = ""
                        if isinstance(self.backend_args, str):
                            backend_args_str = self.backend_args
                        elif isinstance(self.backend_args, dict) and "path" in self.backend_args:
                            backend_args_str = self.backend_args["path"]

                        payload["backend"] = backend_name
                        payload["backend_args"] = backend_args_str
                        payload["acq_state"] = str(self._acq_state.value)
                        payload["ready"] = (self.backend is not None and self._acq_state in (AcqState.READY, AcqState.RUNNING))

                        if self.full_idx.size > 0 and self.full_rms.size > 0:
                            # Даунсэмплинг до ≤1000 точек для веб-интерфейса
                            max_points = 1000
                            if self.full_idx.size <= max_points:
                                # Используем все точки
                                t_s_array = (self.full_idx - self.full_idx[-1]) / self.sample_rate
                                rms_dbm_array = self.full_rms
                            else:
                                # Равномерный даунсэмплинг
                                step = self.full_idx.size // max_points
                                indices = np.arange(0, self.full_idx.size, step)
                                t_s_array = (self.full_idx[indices] - self.full_idx[-1]) / self.sample_rate
                                rms_dbm_array = self.full_rms[indices]

                            payload["t_s"] = t_s_array.tolist()
                            payload["last_rms_dbm"] = rms_dbm_array.tolist()
                        else:
                            payload["t_s"] = []
                            payload["last_rms_dbm"] = []

                    ack_ms = (time.time() - t_start) * 1000
                    log.debug(f"cmd=get_status ack={ack_ms:.1f}ms")
                    self.rep.send_json({"ok": True, "status": payload})
                elif cmd == "start_acquire":
                    # Idempotent start
                    self.start()
                    with self._lock:
                        acq_state = str(self._acq_state.value)
                    ack_ms = (time.time() - t_start) * 1000
                    log.debug(f"cmd=start_acquire ack={ack_ms:.1f}ms acq_state={acq_state}")
                    self.rep.send_json({"ok": True, "acq_state": acq_state})

                elif cmd == "stop_acquire":
                    # Idempotent stop
                    self.stop()
                    with self._lock:
                        acq_state = str(self._acq_state.value)
                    ack_ms = (time.time() - t_start) * 1000
                    log.debug(f"cmd=stop_acquire ack={ack_ms:.1f}ms acq_state={acq_state}")
                    self.rep.send_json({"ok": True, "acq_state": acq_state})

                elif cmd == "set_params":
                    # Позволяет менять базовые параметры на лету (минимальный набор)
                    changed = {}
                    if "thresh_dbm" in msg:
                        PULSE_THRESH_DBM = float(msg["thresh_dbm"]); changed["thresh_dbm"] = PULSE_THRESH_DBM
                    if "target_signal_hz" in msg:
                        TARGET_SIGNAL_HZ = float(msg["target_signal_hz"]); changed["target_signal_hz"] = TARGET_SIGNAL_HZ
                    ack_ms = (time.time() - t_start) * 1000
                    log.debug(f"cmd=set_params ack={ack_ms:.1f}ms changed={changed}")
                    self.rep.send_json({"ok": True, "changed": changed})

                elif cmd == "get_last_pulse":
                    # Возвращаем последние данные pulse с фазой и частотой
                    # Поддержка slice API: offset/span с units, max_samples, downsample

                    if not hasattr(self, 'last_iq_seg') or self.last_iq_seg is None or self.last_iq_seg.size == 0:
                        ack_ms = (time.time() - t_start) * 1000
                        log.debug(f"cmd=get_last_pulse ack={ack_ms:.1f}ms error=no_data")
                        self.rep.send_json({"ok": False, "error": "No pulse data available"})
                        continue

                    try:
                        # Извлекаем параметры slice (опциональные)
                        slice_params = msg.get("slice", {})
                        offset = slice_params.get("offset", 0)
                        span = slice_params.get("span", "100%")
                        units = slice_params.get("units", "auto")

                        max_samples = msg.get("max_samples", "all")
                        downsample_method = msg.get("downsample", "decimate")

                        # IQ сегмент последнего импульса
                        iq_seg = self.last_iq_seg.copy()
                        n_total = len(iq_seg)
                        fs = float(self.sample_rate)

                        # Нормализуем slice в индексы
                        i0, i1 = normalize_slice(offset, span, units, fs, n_total)

                        # Вырезаем срез IQ
                        iq_slice = iq_seg[i0:i1]
                        n_slice = len(iq_slice)

                        if n_slice < 2:
                            self.rep.send_json({"ok": False, "error": "Slice too small"})
                            continue

                        # Обработка среза: фаза, FM, RMS
                        # Используем те же функции что и при детекции

                        # 1. Фаза
                        res_phase = process_psk_impulse(
                            iq_seg=iq_slice,
                            fs=fs,
                            baseline_ms=PSK_BASELINE_MS,
                            t0_offset_ms=0.0,
                            use_lpf_decim=True,
                            remove_slope=True,
                        )
                        phase_xs_ms = res_phase.get("xs_ms", np.array([]))
                        phase_ys_rad = res_phase.get("phase_rad", np.array([]))

                        # Обрезаем концовку фазы на PHASE_TRIM_END_MS относительно конца импульса (i1)
                        if len(phase_xs_ms) > 0 and PHASE_TRIM_END_MS > 0 and self.last_core_gate:
                            g0, g1 = self.last_core_gate
                            # Конец импульса в миллисекундах
                            impulse_end_ms = (g1 / fs) * 1000.0
                            # Обрезаем фазу на PHASE_TRIM_END_MS от конца импульса
                            max_phase_time_ms = impulse_end_ms - PHASE_TRIM_END_MS
                            mask = phase_xs_ms <= max_phase_time_ms
                            phase_xs_ms = phase_xs_ms[mask]
                            phase_ys_rad = phase_ys_rad[mask]

                        # 2. FM
                        fm_out = fm_discriminator(
                            iq=iq_slice,
                            fs=fs,
                            pre_lpf_hz=50_000,
                            decim=4,
                            smooth_hz=2_000,
                            detrend=True,
                            center=True,
                            fir_taps=127,
                        )
                        fm_xs_ms = fm_out.get("xs_ms", np.array([]))
                        fm_ys_hz = fm_out.get("freq_hz", np.array([]))

                        # Обрезаем начало и конец FM относительно границ импульса
                        if len(fm_xs_ms) > 0 and self.last_core_gate:
                            g0, g1 = self.last_core_gate
                            impulse_start_ms = (g0 / fs) * 1000.0
                            impulse_end_ms = (g1 / fs) * 1000.0

                            # Обрезаем начало на START_DELAY_MS
                            min_fm_time_ms = impulse_start_ms + START_DELAY_MS
                            # Обрезаем конец на PHASE_TRIM_END_MS
                            max_fm_time_ms = impulse_end_ms - PHASE_TRIM_END_MS

                            mask = (fm_xs_ms >= min_fm_time_ms) & (fm_xs_ms <= max_fm_time_ms)
                            fm_xs_ms = fm_xs_ms[mask]
                            fm_ys_hz = fm_ys_hz[mask]

                        # 3. RMS (на исходном IQ среза без децимации)
                        win_samps = max(1, int(round(fs * (RMS_WIN_MS * 1e-3))))
                        calib_offset_db = getattr(self.backend, 'calib_offset_db', 0.0) if self.backend else 0.0

                        p_slice = np.abs(iq_slice)**2
                        # Скользящее RMS окно (аналогично _process_samples)
                        if p_slice.size >= win_samps:
                            c = np.cumsum(p_slice, dtype=np.float64)
                            S_valid = c[win_samps - 1:] - np.concatenate(([0.0], c[:-win_samps]))
                            P_win = S_valid / float(win_samps)
                            rms_dbm = self._db10(P_win) + DBM_OFFSET_DB + calib_offset_db
                        else:
                            rms_dbm = np.array([], dtype=np.float32)

                        # Создаём временную ось для RMS (в миллисекундах относительно начала среза)
                        rms_indices = np.arange(len(rms_dbm))
                        rms_xs_ms = (rms_indices / fs) * 1000.0

                        # Определяем t_unit для ответа
                        if units == "samples":
                            t_unit = "samples"
                            # Конвертируем мс в индексы
                            phase_xs_out = (phase_xs_ms / 1000.0 * fs).astype(int) + i0
                            fm_xs_out = (fm_xs_ms / 1000.0 * fs).astype(int) + i0 if len(fm_xs_ms) > 0 else fm_xs_ms
                            rms_xs_out = rms_indices + i0
                        else:
                            t_unit = "ms"
                            phase_xs_out = phase_xs_ms
                            fm_xs_out = fm_xs_ms
                            rms_xs_out = rms_xs_ms

                        # Применяем downsample если нужно
                        if max_samples != "all":
                            phase_xs_out, phase_ys_rad = apply_downsample(
                                phase_xs_out, phase_ys_rad, max_samples, downsample_method
                            )
                            if len(fm_xs_out) > 0:
                                fm_xs_out, fm_ys_hz = apply_downsample(
                                    fm_xs_out, fm_ys_hz, max_samples, downsample_method
                                )
                            if len(rms_xs_out) > 0:
                                rms_xs_out, rms_dbm = apply_downsample(
                                    rms_xs_out, rms_dbm, max_samples, downsample_method
                                )

                        # Формируем ответ
                        result = {
                            "t_unit": t_unit,
                            "phase": {
                                "x": phase_xs_out.tolist() if isinstance(phase_xs_out, np.ndarray) else phase_xs_out,
                                "y": phase_ys_rad.tolist() if isinstance(phase_ys_rad, np.ndarray) else phase_ys_rad
                            },
                            "fm": {
                                "x": fm_xs_out.tolist() if isinstance(fm_xs_out, np.ndarray) else fm_xs_out,
                                "y": fm_ys_hz.tolist() if isinstance(fm_ys_hz, np.ndarray) else fm_ys_hz
                            },
                            "rms": {
                                "x": rms_xs_out.tolist() if isinstance(rms_xs_out, np.ndarray) else rms_xs_out,
                                "y": rms_dbm.tolist() if isinstance(rms_dbm, np.ndarray) else rms_dbm
                            },
                            "meta": {
                                "fs": fs,
                                "i0": int(i0),
                                "i1": int(i1),
                                "src_len": int(n_total),
                                "out_len": int(len(phase_xs_out)),
                                "units_req": units,
                                "downsample": downsample_method if max_samples != "all" else "none",
                                "core_gate": list(self.last_core_gate) if self.last_core_gate else None
                            }
                        }

                        ack_ms = (time.time() - t_start) * 1000
                        log.debug(f"cmd=get_last_pulse ack={ack_ms:.1f}ms slice=[{i0}:{i1}] units={units}")
                        self.rep.send_json({"ok": True, "pulse": result})

                    except Exception as e:
                        ack_ms = (time.time() - t_start) * 1000
                        log.error(f"cmd=get_last_pulse ack={ack_ms:.1f}ms error: {e}")
                        self.rep.send_json({"ok": False, "error": str(e)})

                elif cmd == "save_sigmf":
                    # Сохранение последнего сегмента в SigMF
                    fn = CAPTURE_DIR / time.strftime("pulse_%Y%m%d_%H%M%S.cf32")
                    seg = (self.last_iq_seg.copy() if isinstance(self.last_iq_seg, np.ndarray) else np.empty(0, np.complex64))
                    if seg.size:
                        # Мгновенный ACK, сохранение в воркере
                        ack_ms = (time.time() - t_start) * 1000
                        log.debug(f"cmd=save_sigmf ack={ack_ms:.1f}ms path={fn}")
                        self.rep.send_json({"ok": True, "path": str(fn)})
                        # Асинхронное сохранение
                        def save_task():
                            seg.astype(np.complex64).tofile(str(fn))
                            self._emit("file_saved", {"path": str(fn)})
                        self._executor.submit(save_task)
                    else:
                        ack_ms = (time.time() - t_start) * 1000
                        log.debug(f"cmd=save_sigmf ack={ack_ms:.1f}ms error=no_segment")
                        self.rep.send_json({"ok": False, "error": "no_segment"})
                elif cmd == "get_sdr_config":
                    # Возвращаем текущую конфигурацию SDR
                    with self._lock:
                        config = {
                            "backend_name": self.backend_name if self.backend_name else "file",
                            "center_freq_hz": self.backend.center_freq if (self.backend and hasattr(self.backend, 'center_freq')) else 406000000,
                            "sample_rate_sps": self.sample_rate,
                            "bb_shift_enable": self.effective_bb_shift_enable,
                            "bb_shift_hz": self.effective_bb_shift_hz,
                            "freq_corr_hz": 0,  # TODO: получить из backend
                            "agc": False,  # TODO: получить из backend
                            "gain_db": self.backend.gain if (self.backend and hasattr(self.backend, 'gain')) else 30.0,
                            "bias_t": False,  # TODO: получить из backend
                            "antenna": "RX",  # TODO: получить из backend
                            "device": getattr(self.backend, 'device_info', 'Unknown device') if self.backend else "File wait mode"
                        }
                    ack_ms = (time.time() - t_start) * 1000
                    log.debug(f"cmd=get_sdr_config ack={ack_ms:.1f}ms")
                    self.rep.send_json({"ok": True, "config": config})
                elif cmd == "set_sdr_config":
                    # === МГНОВЕННЫЙ ACK, ТЯЖЕЛАЯ РАБОТА В ВОРКЕРЕ ===
                    # Определяем формат и извлекаем config
                    if "config" in msg:
                        config = msg["config"]
                    else:
                        # Плоский формат - берем все ключи кроме служебных
                        config = {k: v for k, v in msg.items() if k not in ("cmd",)}

                    # Проверка на дребезг (anti-spam)
                    with self._lock:
                        if self._in_flight_operation:
                            ack_ms = (time.time() - t_start) * 1000
                            log.debug(f"cmd=set_sdr_config ack={ack_ms:.1f}ms info=busy")
                            self.rep.send_json({"ok": True, "info": "busy", "backend_changed": False, "retuned": False})
                            continue

                    # Быстрая проверка: нужно ли менять backend?
                    old_backend_name = self.backend_name or "auto"
                    old_backend_args = self.backend_args

                    new_backend_name = config.get("backend_name", old_backend_name)
                    new_backend_args = config.get("backend_args", old_backend_args)

                    # Поддержка backend_args как строки (путь) или словаря {"path": "..."}
                    if isinstance(new_backend_args, dict) and "path" in new_backend_args:
                        new_backend_args = new_backend_args["path"]

                    backend_changed = (new_backend_name != old_backend_name or new_backend_args != old_backend_args)

                    # === МГНОВЕННЫЙ ACK ===
                    ack_ms = (time.time() - t_start) * 1000
                    log.info(f"cmd=set_sdr_config ack={ack_ms:.1f}ms backend_changed={backend_changed}")
                    self.rep.send_json({
                        "ok": True,
                        "backend_changed": backend_changed,
                        "retuned": backend_changed
                    })

                    # Если backend меняется - запускаем тяжелую работу в воркере
                    if backend_changed:
                        with self._lock:
                            self._in_flight_operation = True
                            self._acq_state = AcqState.RETUNING

                        def retune_task():
                            try:
                                log.info(f"Retuning to backend={new_backend_name} args={new_backend_args}")

                                # Останавливаем текущий backend
                                if self.backend:
                                    self.stop()
                                    if hasattr(self.backend, 'close'):
                                        self.backend.close()
                                    self.backend = None

                                # Создаем новый backend
                                if new_backend_name == "auto":
                                    actual_backend = select_auto_backend(new_backend_args)
                                else:
                                    actual_backend = new_backend_name

                                self.backend = safe_make_backend(
                                    actual_backend,
                                    sample_rate=config.get("sample_rate_sps", SAMPLE_RATE_SPS),
                                    center_freq=config.get("center_freq_hz", CENTER_FREQ_HZ),
                                    gain_db=config.get("gain_db", TUNER_GAIN_DB),
                                    agc=ENABLE_AGC,
                                    corr_ppm=FREQ_CORR_PPM,
                                    device_args=new_backend_args,
                                    if_offset_hz=IF_OFFSET_HZ,
                                    on_fail="file_wait"
                                )

                                # Обновить параметры сервиса от backend
                                with self._lock:
                                    st = self.backend.get_status() or {} if self.backend else {}
                                    self.sample_rate = float(st.get("actual_sample_rate_sps",
                                                    getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS)))
                                    self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
                                    self.nco_k = 2.0*np.pi*(BB_SHIFT_HZ/float(self.sample_rate))
                                    self.backend_name = new_backend_name
                                    self.backend_args = new_backend_args

                                    # Принудительно отключаем BB_SHIFT в файловом режиме
                                    if actual_backend == "file":
                                        self.effective_bb_shift_enable = False
                                        self.effective_bb_shift_hz = 0.0
                                        log.info("BB_SHIFT принудительно отключен в файловом режиме")
                                    else:
                                        self.effective_bb_shift_enable = config.get("bb_shift_enable", BB_SHIFT_ENABLE)
                                        self.effective_bb_shift_hz = config.get("bb_shift_hz", BB_SHIFT_HZ)

                                    # Если backend успешно создан - переходим в ready
                                    if self.backend:
                                        self._acq_state = AcqState.READY
                                        self._in_flight_operation = False

                                # ВАЖНО: вызывать start() вне лока, чтобы исключить самозахват
                                if self.backend:
                                    self.start()
                                else:
                                    # Backend в режиме ожидания файла
                                    with self._lock:
                                        self._acq_state = AcqState.STOPPED
                                        self._in_flight_operation = False

                                # Уведомление о завершении ретюна
                                self._emit("retune_done", {
                                    "backend": actual_backend,
                                    "backend_args": new_backend_args,
                                    "success": (self.backend is not None)
                                })
                                # Отправить обновлённый статус
                                self._emit_status()

                            except Exception as e:
                                log.error(f"Retune failed: {e}")
                                with self._lock:
                                    self._acq_state = AcqState.STOPPED
                                    self._in_flight_operation = False
                                self._emit("retune_done", {"backend": new_backend_name, "success": False, "error": str(e)})
                                # Отправить статус даже при ошибке
                                self._emit_status()

                        # Запускаем в воркере
                        self._executor.submit(retune_task)

                    else:
                        # Backend не меняется - применяем легкие параметры
                        try:
                            if "gain_db" in config and hasattr(self.backend, 'set_gain'):
                                self.backend.set_gain(float(config["gain_db"]))

                            if "freq_corr_hz" in config and hasattr(self.backend, 'set_freq_correction'):
                                self.backend.set_freq_correction(float(config["freq_corr_hz"]))

                            # BB_SHIFT параметры (если backend не менялся)
                            if "bb_shift_enable" in config:
                                # Проверяем файловый режим
                                if hasattr(self.backend, 'backend_name') and self.backend.backend_name == "file":
                                    self.effective_bb_shift_enable = False
                                    self.effective_bb_shift_hz = 0.0
                                    log.warning("BB_SHIFT принудительно отключен в файловом режиме")
                                else:
                                    self.effective_bb_shift_enable = bool(config["bb_shift_enable"])

                            if "bb_shift_hz" in config:
                                if hasattr(self.backend, 'backend_name') and self.backend.backend_name == "file":
                                    self.effective_bb_shift_hz = 0.0
                                else:
                                    self.effective_bb_shift_hz = float(config["bb_shift_hz"])

                        except Exception as e:
                            log.warning(f"Failed to apply light config params: {e}")

                else:
                    # Неизвестная команда
                    ack_ms = (time.time() - t_start) * 1000
                    log.warning(f"cmd={cmd} ack={ack_ms:.1f}ms error=unknown_cmd")
                    self.rep.send_json({"ok": False, "error": "unknown_cmd"})

            except Exception as e:
                # Любая ошибка в обработке - всегда отвечаем
                ack_ms = (time.time() - t_start) * 1000
                log.error(f"cmd={cmd} ack={ack_ms:.1f}ms error={e}")
                self.rep.send_json({"ok": False, "error": str(e)})

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Headless Beacon DSP Service")
    ap.add_argument("--pub", default=DEFAULT_PUB_ADDR, help="ZeroMQ PUB address (bind)")
    ap.add_argument("--rep", default=DEFAULT_REP_ADDR, help="ZeroMQ REP address (bind)")
    ap.add_argument("--backend", default=os.environ.get("BACKEND_NAME", "auto"),
                    help="Backend name (auto/soapy_rtl/soapy_hackrf/soapy_airspy/soapy_sdrplay/rsa306/file)")
    ap.add_argument("--backend-args", default=None, help="Backend args JSON (for soapy: device args, for file: path/params)")
    ap.add_argument("--jsonl", default=None, help="Path to JSONL session file")
    args = ap.parse_args()

    backend_args = None
    if args.backend_args:
        try:
            backend_args = json.loads(args.backend_args)
        except Exception as e:
            log.warning(f"backend-args JSON parse error: {e}")

    jsonl_path = Path(args.jsonl) if args.jsonl else None

    svc = BeaconDSPService(pub_addr=args.pub, rep_addr=args.rep,
                           backend_name=args.backend, backend_args=backend_args,
                           jsonl_path=jsonl_path)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")
        svc.shutdown()

