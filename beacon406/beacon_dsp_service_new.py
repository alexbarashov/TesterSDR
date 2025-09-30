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
from lib.logger import get_logger, setup_logging

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

setup_logging()
log = get_logger(__name__)

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