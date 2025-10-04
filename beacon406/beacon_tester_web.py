"""
COSPAS/SARSAT Beacon Tester - Version 2.0 ZeroMQ Edition
=========================================================
Веб-интерфейс для beacon_dsp_service.py через ZeroMQ.
Не выполняет прямую работу с SDR - только подключается к DSP сервису.
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
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from flask import Flask, jsonify, request, Response

# Добавляем путь к библиотекам
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.hex_decoder import hex_to_bits, build_table_rows
from lib.logger import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
import numpy as np
from werkzeug.utils import secure_filename

# ZeroMQ для подключения к DSP сервису
try:
    import zmq
except Exception:
    zmq = None
    log.error("pyzmq not installed - ZeroMQ functionality disabled")

app = Flask(__name__)

# ============================================================================
# ZeroMQ Configuration
# ============================================================================
DEFAULT_PUB_ADDR = os.environ.get("DSP_PUB_ADDR", "tcp://127.0.0.1:8781")
DEFAULT_REP_ADDR = os.environ.get("DSP_REP_ADDR", "tcp://127.0.0.1:8782")

# ============================================================================
# Global State - теперь заполняется из DSP сервиса
# ============================================================================
data_lock = threading.Lock()

# Кэш последних событий от DSP сервиса
last_status_event: Dict[str, Any] = {}
last_pulse_event: Dict[str, Any] = {}
last_psk_event: Dict[str, Any] = {}

# История импульсов (заполняется из pulse событий)
PULSE_HISTORY_MAX = 50
pulse_history = deque(maxlen=PULSE_HISTORY_MAX)

# Буферы для веб-отображения (RMS timeline)
rms_history = deque(maxlen=1000)
time_history = deque(maxlen=1000)

# Состояние подключения к DSP сервису
dsp_connected = False
dsp_connection_error = ""

# ============================================================================
# ZeroMQ Client для подключения к beacon_dsp_service.py
# ============================================================================
class DspServiceClient:
    """ZeroMQ клиент для beacon_dsp_service.py"""

    def __init__(self, pub_addr: str = DEFAULT_PUB_ADDR, rep_addr: str = DEFAULT_REP_ADDR):
        if zmq is None:
            raise RuntimeError("pyzmq not installed")

        self.pub_addr = pub_addr
        self.rep_addr = rep_addr
        self.ctx = zmq.Context.instance()
        self._req_lock = threading.Lock()
        self._req_socket_broken = False
        self._stop = False

        # SUB socket для событий (status/pulse/psk)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(self.pub_addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub_thread: Optional[threading.Thread] = None
        self._event_callback = None

        # REQ socket для команд (будет создан по требованию)
        self.req = None
        self._ensure_req_socket()

        log.info(f"DSP client initialized: PUB={pub_addr}, REP={rep_addr}")

    def _ensure_req_socket(self):
        """Создание/пересоздание REQ сокета при необходимости"""
        if self.req is not None and not self._req_socket_broken:
            return

        try:
            if self.req is not None:
                self.req.close(0)
        except Exception:
            pass

        self.req = self.ctx.socket(zmq.REQ)
        try:
            self.req.setsockopt(zmq.RCVTIMEO, 3000)  # 3s timeout
            self.req.setsockopt(zmq.SNDTIMEO, 1000)  # 1s send
            self.req.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        self.req.connect(self.rep_addr)
        self._req_socket_broken = False

    def send_cmd(self, cmd: str, **kwargs) -> Dict[str, Any]:
        """Отправка команды через REQ/REP с обработкой ошибок"""
        msg = {"cmd": cmd, **kwargs}
        t0 = time.perf_counter()

        with self._req_lock:
            try:
                self._ensure_req_socket()
                self.req.send_json(msg)
                rep = self.req.recv_json()
                self._req_socket_broken = False

                dt_ms = (time.perf_counter() - t0) * 1000
                ok = rep.get("ok", False)
                log.debug(f"[ZMQ cmd={cmd}] ack={dt_ms:.1f}ms ok={ok}")

                return rep if isinstance(rep, dict) else {"ok": False, "error": "bad reply"}

            except zmq.error.Again:
                self._req_socket_broken = True
                dt_ms = (time.perf_counter() - t0) * 1000
                log.warning(f"[ZMQ cmd={cmd}] TIMEOUT after {dt_ms:.1f}ms")
                return {"ok": False, "error": "DSP service timeout"}

            except Exception as e:
                self._req_socket_broken = True
                dt_ms = (time.perf_counter() - t0) * 1000
                log.error(f"[ZMQ cmd={cmd}] ERROR after {dt_ms:.1f}ms: {e}")
                return {"ok": False, "error": f"Communication error: {e}"}

    # Команды DSP сервиса
    def get_status(self) -> Dict[str, Any]:
        return self.send_cmd("get_status")

    def start_acquire(self) -> Dict[str, Any]:
        return self.send_cmd("start_acquire")

    def stop_acquire(self) -> Dict[str, Any]:
        return self.send_cmd("stop_acquire")

    def set_params(self, **params) -> Dict[str, Any]:
        return self.send_cmd("set_params", **params)

    def get_last_pulse(self, slice_params=None, max_samples="all", downsample="decimate") -> Dict[str, Any]:
        kwargs = {}
        if slice_params is not None:
            kwargs["slice"] = slice_params
        if max_samples != "all":
            kwargs["max_samples"] = max_samples
        if downsample != "decimate":
            kwargs["downsample"] = downsample
        return self.send_cmd("get_last_pulse", **kwargs)

    def save_sigmf(self) -> Dict[str, Any]:
        return self.send_cmd("save_sigmf")

    def set_sdr_config(self, **cfg) -> Dict[str, Any]:
        return self.send_cmd("set_sdr_config", **cfg)

    def get_sdr_config(self) -> Dict[str, Any]:
        return self.send_cmd("get_sdr_config")

    def echo(self, **payload) -> Dict[str, Any]:
        return self.send_cmd("echo", payload=payload)

    # SUB поток для событий
    def subscribe_events(self, callback):
        """Запуск фонового потока для приёма событий"""
        self._event_callback = callback
        if self._sub_thread and self._sub_thread.is_alive():
            return
        self._stop = False
        self._sub_thread = threading.Thread(target=self._sub_loop, daemon=True)
        self._sub_thread.start()
        log.info("SUB event listener started")

    def _sub_loop(self):
        """Основной цикл приёма событий из PUB канала"""
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        while not self._stop:
            try:
                socks = dict(poller.poll(200))
                if self.sub in socks and socks[self.sub] == zmq.POLLIN:
                    line = self.sub.recv_string(flags=zmq.NOBLOCK)
                    obj = json.loads(line)
                    if self._event_callback:
                        self._event_callback(obj)
            except Exception as e:
                log.debug(f"[SUB loop] error: {e}")
                time.sleep(0.1)

    def close(self):
        """Закрытие соединений"""
        self._stop = True
        try:
            if self._sub_thread:
                self._sub_thread.join(timeout=0.5)
        except Exception:
            pass

        with self._req_lock:
            try:
                if self.req:
                    self.req.close(0)
            except Exception:
                pass

        try:
            self.sub.close(0)
        except Exception:
            pass

        log.info("DSP client closed")

# ============================================================================
# Глобальный клиент DSP сервиса
# ============================================================================
dsp_client: Optional[DspServiceClient] = None

def init_dsp_client():
    """Инициализация подключения к DSP сервису"""
    global dsp_client, dsp_connected, dsp_connection_error

    if dsp_client is not None:
        return True

    try:
        dsp_client = DspServiceClient(pub_addr=DEFAULT_PUB_ADDR, rep_addr=DEFAULT_REP_ADDR)
        dsp_client.subscribe_events(on_dsp_event)

        # Проверка связи через echo
        rep = dsp_client.echo(test="hello")
        if rep.get("ok", False):
            dsp_connected = True
            dsp_connection_error = ""
            log.info("DSP service connected successfully")
            return True
        else:
            dsp_connected = False
            dsp_connection_error = rep.get("error", "Connection failed")
            log.warning(f"DSP service connection check failed: {dsp_connection_error}")
            return False

    except Exception as e:
        dsp_connected = False
        dsp_connection_error = str(e)
        log.error(f"Failed to initialize DSP client: {e}")
        return False

def on_dsp_event(event: Dict[str, Any]):
    """Обработка событий от DSP сервиса (вызывается из SUB потока)"""
    global last_status_event, last_pulse_event, last_psk_event, dsp_connected

    event_type = event.get("type")

    with data_lock:
        if event_type == "status":
            # Обновляем кэш статуса
            last_status_event = event
            dsp_connected = True

            # Обновляем RMS историю для веб-отображения
            t_s = event.get("t_s", [])
            rms_dbm = event.get("last_rms_dbm", [])
            if t_s and rms_dbm and len(t_s) == len(rms_dbm):
                # Добавляем в историю (ограничиваем до 1000 точек)
                for i in range(min(len(t_s), 100)):  # Берём последние 100 точек чтобы не перегружать
                    time_history.append(t_s[i])
                    rms_history.append(rms_dbm[i])

        elif event_type == "pulse":
            # Сохраняем последнее pulse событие
            last_pulse_event = event

            # Добавляем в историю импульсов
            pulse_info = {
                'start_abs': event.get('start_abs', 0),
                'length_ms': event.get('length_ms', 0),
                'peak_dbm': event.get('peak_dbm', 0),
                'above_thresh_ratio': event.get('above_thresh_ratio', 0),
                'timestamp': time.time(),
                # Метаданные для кнопок UI
                'phase_metrics': event.get('phase_metrics'),
                'msg_hex': event.get('msg_hex'),
                # Данные графиков (если есть)
                'phase_xs_ms': event.get('phase_xs_ms'),
                'phase_ys_rad': event.get('phase_ys_rad'),
                'fr_xs_ms': event.get('fr_xs_ms'),
                'fr_ys_hz': event.get('fr_ys_hz'),
                'rms_xs_ms': event.get('rms_xs_ms'),
                'rms_ys_dbm': event.get('rms_ys_dbm'),
                'preamble_ms': event.get('preamble_ms'),
                'baud': event.get('baud'),
            }
            pulse_history.append(pulse_info)

            log.debug(f"[EVENT] pulse: {pulse_info['length_ms']:.1f}ms, peak={pulse_info['peak_dbm']:.1f}dBm")

        elif event_type == "psk":
            # Сохраняем PSK событие
            last_psk_event = event
            log.debug(f"[EVENT] psk: hex={event.get('hex', 'N/A')}")

# ============================================================================
# BeaconState - состояние маяка (теперь заполняется из DSP событий)
# ============================================================================
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
    hex_message: str = ""
    current_file: str = ""

    # Данные графиков
    phase_data: list = field(default_factory=list)
    xs_ms: list = field(default_factory=list)
    fm_data: list = field(default_factory=list)
    fm_xs_ms: list = field(default_factory=list)

    # Метрики (заполняются из pulse/psk событий)
    fs1_hz: float = 0.0
    fs2_hz: float = 0.0
    fs3_hz: float = 0.0
    phase_pos_rad: float = 0.0
    phase_neg_rad: float = 0.0
    t_rise_mcs: float = 0.0
    t_fall_mcs: float = 0.0
    pos_phase: float = 0.0
    neg_phase: float = 0.0
    ph_rise: float = 0.0
    ph_fall: float = 0.0
    asymmetry: float = 0.0
    t_mod: float = 0.0
    rms_dbm: float = 0.0
    freq_hz: float = 0.0
    p_wt: float = 0.0
    prise_ms: float = 0.0
    bitrate_bps: float = 0.0
    symmetry_pct: float = 0.0
    preamble_ms: float = 0.0
    total_ms: float = 0.0
    rep_period_s: float = 0.0

STATE = BeaconState()

def update_state_from_pulse_event(event: Dict[str, Any]):
    """Обновление STATE из pulse события DSP сервиса"""
    # Фазовые данные для графика
    phase_xs = event.get('phase_xs_ms', [])
    phase_ys = event.get('phase_ys_rad', [])

    if phase_xs and phase_ys:
        STATE.xs_ms = list(phase_xs) if not isinstance(phase_xs, list) else phase_xs
        STATE.phase_data = list(phase_ys) if not isinstance(phase_ys, list) else phase_ys

    # FM данные для графика
    fm_xs = event.get('fr_xs_ms', [])
    fm_ys = event.get('fr_ys_hz', [])

    if fm_xs and fm_ys:
        STATE.fm_xs_ms = list(fm_xs) if not isinstance(fm_xs, list) else fm_xs
        STATE.fm_data = list(fm_ys) if not isinstance(fm_ys, list) else fm_ys

    # Метрики из phase_metrics
    metrics = event.get('phase_metrics', {})
    if metrics:
        STATE.pos_phase = float(metrics.get('Pos (rad)', 0.0))
        STATE.neg_phase = float(metrics.get('Neg (rad)', 0.0))
        STATE.ph_rise = float(metrics.get('Rise (μs)', 0.0))
        STATE.ph_fall = float(metrics.get('Fall (μs)', 0.0))
        STATE.asymmetry = float(metrics.get('Asymmetry (%)', 0.0))
        STATE.symmetry_pct = STATE.asymmetry
        STATE.rms_dbm = float(metrics.get('Power (RMS, dBm)', 0.0))
        STATE.freq_hz = float(metrics.get('Frequency (Hz)', 0.0))

        # Вычисляем битрейт из Fmod
        fmod = metrics.get('Fmod (Hz)', 0.0)
        if fmod and fmod > 0:
            STATE.bitrate_bps = float(fmod)

        # Timing из метрик
        msg_dur = metrics.get('Message Duration (ms)', 0.0)
        carrier_dur = metrics.get('Carrier Duration (ms)', 0.0)
        if msg_dur:
            STATE.total_ms = float(msg_dur)
        if carrier_dur:
            STATE.preamble_ms = float(carrier_dur)

    # HEX сообщение
    msg_hex = event.get('msg_hex')
    if msg_hex:
        STATE.hex_message = str(msg_hex)
        STATE.message = f"Message: {msg_hex[:16]}..."

def update_state_from_psk_event(event: Dict[str, Any]):
    """Обновление STATE из psk события DSP сервиса"""
    msg_hex = event.get('hex')
    if msg_hex:
        STATE.hex_message = str(msg_hex)
        STATE.message = f"Message: {msg_hex[:16]}..."

    # Дополнительные метрики из PSK события
    preamble = event.get('preamble_ms')
    if preamble is not None:
        STATE.preamble_ms = float(preamble)

    baud = event.get('baud')
    if baud is not None:
        STATE.bitrate_bps = float(baud)

    pos = event.get('pos_phase')
    if pos is not None:
        STATE.pos_phase = float(pos)

    neg = event.get('neg_phase')
    if neg is not None:
        STATE.neg_phase = float(neg)

    rise = event.get('rise_us')
    if rise is not None:
        STATE.ph_rise = float(rise)

    fall = event.get('fall_us')
    if fall is not None:
        STATE.ph_fall = float(fall)

    asym = event.get('asymmetry_pct')
    if asym is not None:
        STATE.asymmetry = float(asym)
        STATE.symmetry_pct = float(asym)

# ============================================================================
# CF32 File Processing (локальная обработка файлов)
# ============================================================================
def process_cf32_file(file_path):
    """
    Локальная обработка CF32 файла через DSP сервис.
    Переключает backend в file режим и запрашивает данные.
    """
    try:
        if not dsp_client:
            return {"error": "DSP service not connected"}

        # Переключаем DSP сервис в файловый режим
        log.info(f"Switching DSP service to file mode: {file_path}")
        rep = dsp_client.set_sdr_config(backend_name="file", backend_args=file_path)

        if not rep.get("ok", False):
            error = rep.get("error", "Failed to switch to file mode")
            log.error(f"File mode switch failed: {error}")
            return {"error": error}

        # Даём время на переключение backend
        time.sleep(0.5)

        # Запускаем обработку
        rep = dsp_client.start_acquire()
        if not rep.get("ok", False):
            error = rep.get("error", "Failed to start file processing")
            log.error(f"File processing start failed: {error}")
            return {"error": error}

        # Ждём появления pulse события (максимум 5 секунд)
        max_wait = 5.0
        wait_step = 0.1
        waited = 0.0

        while waited < max_wait:
            time.sleep(wait_step)
            waited += wait_step

            with data_lock:
                if last_pulse_event:
                    # Получили pulse - обновляем STATE
                    update_state_from_pulse_event(last_pulse_event)
                    if last_psk_event:
                        update_state_from_psk_event(last_psk_event)

                    # Формируем результат
                    result = {
                        "success": True,
                        "msg_hex": STATE.hex_message,
                        "phase_data": STATE.phase_data,
                        "xs_ms": STATE.xs_ms,
                        "fm_data": STATE.fm_data,
                        "fm_xs_ms": STATE.fm_xs_ms,
                        "pos_phase": STATE.pos_phase,
                        "neg_phase": STATE.neg_phase,
                        "ph_rise": STATE.ph_rise,
                        "ph_fall": STATE.ph_fall,
                        "asymmetry": STATE.asymmetry,
                        "symmetry_pct": STATE.symmetry_pct,
                        "bitrate_bps": STATE.bitrate_bps,
                        "preamble_ms": STATE.preamble_ms,
                        "total_ms": STATE.total_ms,
                        "rms_dbm": STATE.rms_dbm,
                        "freq_hz": STATE.freq_hz,
                        "file_processed": True
                    }

                    log.info(f"File processed successfully: {file_path}")
                    return result

        # Таймаут - нет pulse событий
        log.warning(f"File processing timeout: no pulse events received")
        return {"error": "No pulse detected in file (timeout)"}

    except Exception as e:
        log.error(f"File processing error: {e}")
        return {"error": f"Processing failed: {str(e)}"}

# ============================================================================
# HTML страница (без изменений - сохраняем оригинальный UI)
# ============================================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COSPAS/SARSAT Beacon Tester v2.1 (ZMQ)</title>
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
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            border-bottom: 1px solid #4a8bc2;
        }

        /* Основной контейнер */
        .container {
            display: flex;
            height: calc(100vh - 52px);
            background: #e8e8e8;
        }

        /* Левая панель */
        .left-panel {
            width: 220px;
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
            font-size: 13px;
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
            font-size: 13px;
        }

        .radio-group label {
            display: block;
            margin: 4px 0;
            cursor: pointer;
            font-size: 13px;
        }

        .control-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 3px 0;
        }

        .control-input {
            width: 50px;
            font-size: 13px;
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
            padding: 6px 12px;
            font-size: 13px;
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
            padding: 6px 10px;
            font-size: 14px;
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
            font-size: 16px;
        }

        .message-line {
            font-size: 14px;
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
            font-size: 13px;
            color: #495057;
            margin-top: 6px;
        }

        .chart-title {
            text-align: center;
            font-size: 14px;
            color: #6c757d;
            margin-top: 6px;
        }

        /* Правая панель */
        .right-panel {
            width: 320px;
            background: #f8f9fa;
            border-left: 1px solid #dee2e6;
            padding: 10px;
        }

        .stats-header {
            background: linear-gradient(180deg, #a8c8e4 0%, #7bb3d9 100%);
            color: #2c3e50;
            font-weight: bold;
            font-size: 14px;
            text-align: center;
            padding: 8px;
            border: 1px solid #6699cc;
            border-radius: 3px;
            margin-bottom: 8px;
        }

        .stats-table {
            width: 100%;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 13px;
        }

        .stats-table td {
            padding: 6px 10px;
            border-bottom: 1px solid #f0f0f0;
        }

        .stats-table td:first-child {
            font-weight: bold;
            color: #495057;
            width: 50%;
        }

        .stats-table td:last-child {
            color: #212529;
            text-align: right;
        }

        .stats-table tr:last-child td {
            border-bottom: none;
        }

        /* DSP Connection Status */
        .dsp-status {
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 8px;
            margin-bottom: 8px;
            font-size: 12px;
        }

        .dsp-status.connected {
            border-color: #28a745;
            background: #d4edda;
        }

        .dsp-status.disconnected {
            border-color: #dc3545;
            background: #f8d7da;
        }
    </style>
</head>
<body>
    <div class="header">
        COSPAS/SARSAT Beacon Tester v2.1 (ZeroMQ Edition)
    </div>

    <div class="container">
        <!-- Левая панель управления -->
        <div class="left-panel">
            <!-- DSP Connection Status -->
            <div id="dspStatus" class="dsp-status disconnected">
                DSP: Connecting...
            </div>

            <!-- Protocol Selection -->
            <div class="panel-section">
                <div class="section-header">Protocol</div>
                <div class="section-content">
                    <div class="radio-group">
                        <label><input type="radio" name="protocol" value="N" checked> ELT</label>
                        <label><input type="radio" name="protocol" value="E"> EPIRB</label>
                        <label><input type="radio" name="protocol" value="P"> PLB</label>
                    </div>
                </div>
            </div>

            <!-- Frequency Control -->
            <div class="panel-section">
                <div class="section-header">Frequency MHz</div>
                <div class="section-content">
                    <div class="control-row">
                        <span>F1:</span>
                        <input type="text" class="control-input" id="freq1" value="406.025">
                    </div>
                    <div class="control-row">
                        <span>F2:</span>
                        <input type="text" class="control-input" id="freq2" value="406.028">
                    </div>
                    <div class="control-row">
                        <span>F3:</span>
                        <input type="text" class="control-input" id="freq3" value="406.040">
                    </div>
                </div>
            </div>

            <!-- Phase Settings -->
            <div class="panel-section">
                <div class="section-header">Phase</div>
                <div class="section-content">
                    <div class="control-row">
                        <span>Pos:</span>
                        <input type="text" class="control-input" id="phasePos" value="0.78">
                    </div>
                    <div class="control-row">
                        <span>Neg:</span>
                        <input type="text" class="control-input" id="phaseNeg" value="-0.78">
                    </div>
                    <div class="control-row">
                        <span>t rise:</span>
                        <input type="text" class="control-input" id="tRise" value="25">
                    </div>
                    <div class="control-row">
                        <span>t fall:</span>
                        <input type="text" class="control-input" id="tFall" value="23">
                    </div>
                </div>
            </div>

            <!-- Control Buttons -->
            <div class="panel-section">
                <button class="button primary" style="width: 100%;" onclick="measure()">Measure</button>
                <button class="button primary" style="width: 100%;" onclick="run()">Run</button>
                <button class="button" style="width: 100%;" onclick="cont()">Cont</button>
                <button class="button danger" style="width: 100%;" onclick="stop()">Break</button>
            </div>

            <!-- File Upload -->
            <div class="panel-section">
                <div class="section-header">CF32 File</div>
                <div class="section-content">
                    <input type="file" id="fileInput" accept=".cf32" style="font-size: 11px; width: 100%;">
                    <button class="button" style="width: 100%; margin-top: 5px;" onclick="uploadFile()">Load</button>
                </div>
            </div>

            <!-- Additional Controls -->
            <div class="panel-section">
                <button class="button" style="width: 100%;" onclick="load()">Load</button>
                <button class="button" style="width: 100%;" onclick="save()">Save</button>
            </div>
        </div>

        <!-- Центральная область с графиком и данными -->
        <div class="center-panel">
            <!-- Info Grid (3 columns) -->
            <div class="info-grid">
                <div class="info-block">
                    <div class="info-label">Date</div>
                    <div class="info-value" id="date">01.08.2025</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Conditions</div>
                    <div class="info-value" id="conditions">Normal</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Beacon</div>
                    <div class="info-value" id="beaconModel">Beacon N</div>
                </div>
            </div>

            <!-- Frequency Measurements (2 columns) -->
            <div class="info-row-2">
                <div class="info-block">
                    <div class="info-label">Freq. F1, Hz</div>
                    <div class="info-value" id="freqF1">0.0</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Freq. F2, Hz</div>
                    <div class="info-value" id="freqF2">0.0</div>
                </div>
            </div>

            <div class="info-row-2">
                <div class="info-block">
                    <div class="info-label">Freq. F3, Hz</div>
                    <div class="info-value" id="freqF3">0.0</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Beacon Freq., Hz</div>
                    <div class="info-value" id="beaconFreq">406025000.0</div>
                </div>
            </div>

            <!-- Beacon Title and Message -->
            <div class="beacon-title">Beacon "Название буя"</div>
            <div class="message-line" id="messageText">[no message]</div>

            <!-- Phase Chart -->
            <div class="chart-container">
                <canvas id="phaseChart"></canvas>
            </div>

            <div class="phase-values">
                <span>Phase+: <span id="phasePosValue">0.00</span></span>
                <span>Phase-: <span id="phaseNegValue">0.00</span></span>
                <span>t rise: <span id="tRiseValue">0.0</span> µs</span>
                <span>t fall: <span id="tFallValue">0.0</span> µs</span>
            </div>

            <div class="chart-title">Hexadecimal: <span id="hexMessage" style="font-family: monospace;">-</span></div>
        </div>

        <!-- Правая панель с таблицей текущих значений -->
        <div class="right-panel">
            <div class="stats-header">Current</div>
            <table class="stats-table">
                <tr><td>P,Wt</td><td id="statPWt">0.0</td></tr>
                <tr><td>Prise,ms</td><td id="statPrise">0.0</td></tr>
                <tr><td>BitRate,bps</td><td id="statBitRate">0.0</td></tr>
                <tr><td>Symmetry,%</td><td id="statSymmetry">0.0</td></tr>
                <tr><td>Preamble,ms</td><td id="statPreamble">0.0</td></tr>
                <tr><td>Total,ms</td><td id="statTotal">0.0</td></tr>
                <tr><td>Rep.period,s</td><td id="statRepPeriod">0.0</td></tr>
            </table>

            <!-- SDR Device Info -->
            <div class="stats-header" style="margin-top: 20px;">SDR Device</div>
            <div class="info-block" id="sdrDeviceInfo" style="font-size: 12px; min-height: 60px;">
                Waiting for DSP service...
            </div>

            <!-- Real-time Stats -->
            <div class="stats-header" style="margin-top: 20px;">Real-time</div>
            <table class="stats-table">
                <tr><td>RMS, dBm</td><td id="realtimeRMS">-100.0</td></tr>
                <tr><td>Pulses</td><td id="realtimePulses">0</td></tr>
            </table>
        </div>
    </div>

    <script>
        // Global state
        let chart = null;

        // Initialize chart
        function initChart() {
            const canvas = document.getElementById('phaseChart');
            const ctx = canvas.getContext('2d');

            // Set canvas resolution
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            chart = { canvas, ctx, width: rect.width, height: rect.height };
        }

        // Draw phase chart
        function drawPhaseChart(phaseData, timeData) {
            if (!chart) return;

            const { ctx, width, height } = chart;

            // Clear canvas
            ctx.clearRect(0, 0, width, height);

            if (!phaseData || !timeData || phaseData.length === 0 || timeData.length === 0) {
                // Draw empty grid
                drawGrid(ctx, width, height);
                return;
            }

            // Ensure equal lengths
            const minLen = Math.min(phaseData.length, timeData.length);
            const phase = phaseData.slice(0, minLen);
            const time = timeData.slice(0, minLen);

            // Find data bounds
            const minTime = Math.min(...time);
            const maxTime = Math.max(...time);
            const minPhase = Math.min(...phase);
            const maxPhase = Math.max(...phase);

            // Add margins
            const marginX = 40;
            const marginY = 30;
            const chartWidth = width - 2 * marginX;
            const chartHeight = height - 2 * marginY;

            // Draw grid
            drawGrid(ctx, width, height, marginX, marginY);

            // Draw data
            ctx.strokeStyle = '#2c3e50';
            ctx.lineWidth = 1.5;
            ctx.beginPath();

            for (let i = 0; i < phase.length; i++) {
                const x = marginX + (time[i] - minTime) / (maxTime - minTime + 1e-9) * chartWidth;
                const y = height - marginY - (phase[i] - minPhase) / (maxPhase - minPhase + 1e-9) * chartHeight;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Draw axes labels
            ctx.fillStyle = '#495057';
            ctx.font = '11px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Time, ms', width / 2, height - 5);

            ctx.save();
            ctx.translate(15, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Phase, rad', 0, 0);
            ctx.restore();
        }

        function drawGrid(ctx, width, height, marginX = 40, marginY = 30) {
            const chartWidth = width - 2 * marginX;
            const chartHeight = height - 2 * marginY;

            // Grid lines
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;

            // Vertical grid
            for (let i = 0; i <= 10; i++) {
                const x = marginX + (i / 10) * chartWidth;
                ctx.beginPath();
                ctx.moveTo(x, marginY);
                ctx.lineTo(x, height - marginY);
                ctx.stroke();
            }

            // Horizontal grid
            for (let i = 0; i <= 10; i++) {
                const y = marginY + (i / 10) * chartHeight;
                ctx.beginPath();
                ctx.moveTo(marginX, y);
                ctx.lineTo(width - marginX, y);
                ctx.stroke();
            }

            // Axes
            ctx.strokeStyle = '#495057';
            ctx.lineWidth = 2;
            ctx.strokeRect(marginX, marginY, chartWidth, chartHeight);
        }

        // Update UI from status
        function updateUI(data) {
            // DSP Connection Status
            const dspStatus = document.getElementById('dspStatus');
            const sdrDeviceInfo = document.getElementById('sdrDeviceInfo');

            if (data.sdr_status && data.sdr_status.backend) {
                dspStatus.className = 'dsp-status connected';
                dspStatus.textContent = `DSP: Connected (${data.sdr_status.backend})`;

                // Update SDR device info
                const backend = data.sdr_status.backend || 'unknown';
                const fs = data.sdr_status.fs || 0;
                const ready = data.sdr_status.ready || false;
                const acq_state = data.sdr_status.acq_state || 'stopped';

                sdrDeviceInfo.innerHTML = `
                    Backend: ${backend}<br>
                    Fs: ${(fs / 1e6).toFixed(3)} MHz<br>
                    State: ${acq_state}<br>
                    Ready: ${ready ? 'Yes' : 'No'}
                `;
            } else {
                dspStatus.className = 'dsp-status disconnected';
                dspStatus.textContent = 'DSP: Disconnected';
                sdrDeviceInfo.textContent = data.sdr_device_info || 'Waiting for DSP service...';
            }

            // Basic info
            document.getElementById('date').textContent = data.date || '01.08.2025';
            document.getElementById('conditions').textContent = data.conditions || 'Normal';
            document.getElementById('beaconModel').textContent = data.beacon_model || 'Beacon N';

            // Frequencies
            document.getElementById('freqF1').textContent = (data.fs1_hz || 0).toFixed(1);
            document.getElementById('freqF2').textContent = (data.fs2_hz || 0).toFixed(1);
            document.getElementById('freqF3').textContent = (data.fs3_hz || 0).toFixed(1);
            document.getElementById('beaconFreq').textContent = (data.beacon_frequency || 406025000).toFixed(1);

            // Message
            document.getElementById('messageText').textContent = data.message || '[no message]';
            document.getElementById('hexMessage').textContent = data.hex_message || '-';

            // Phase values
            document.getElementById('phasePosValue').textContent = (data.phase_pos_rad || 0).toFixed(3);
            document.getElementById('phaseNegValue').textContent = (data.phase_neg_rad || 0).toFixed(3);
            document.getElementById('tRiseValue').textContent = (data.t_rise_mcs || 0).toFixed(1);
            document.getElementById('tFallValue').textContent = (data.t_fall_mcs || 0).toFixed(1);

            // Stats table
            document.getElementById('statPWt').textContent = (data.p_wt || 0).toFixed(1);
            document.getElementById('statPrise').textContent = (data.prise_ms || 0).toFixed(1);
            document.getElementById('statBitRate').textContent = (data.bitrate_bps || 0).toFixed(1);
            document.getElementById('statSymmetry').textContent = (data.symmetry_pct || 0).toFixed(1);
            document.getElementById('statPreamble').textContent = (data.preamble_ms || 0).toFixed(1);
            document.getElementById('statTotal').textContent = (data.total_ms || 0).toFixed(1);
            document.getElementById('statRepPeriod').textContent = (data.rep_period_s || 0).toFixed(1);

            // Real-time stats
            document.getElementById('realtimeRMS').textContent = (data.realtime_rms_dbm || -100).toFixed(1);
            document.getElementById('realtimePulses').textContent = data.realtime_pulse_count || 0;

            // Update chart
            if (data.phase_data && data.xs_ms) {
                drawPhaseChart(data.phase_data, data.xs_ms);
            }
        }

        // Fetch status periodically
        function fetchStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => updateUI(data))
                .catch(err => console.error('Status fetch error:', err));
        }

        // Control functions
        function measure() {
            fetch('/api/measure', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Measure:', data);
                    fetchStatus();
                })
                .catch(err => console.error('Measure error:', err));
        }

        function run() {
            fetch('/api/run', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Run:', data);
                    fetchStatus();
                })
                .catch(err => console.error('Run error:', err));
        }

        function cont() {
            fetch('/api/cont', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Cont:', data);
                    fetchStatus();
                })
                .catch(err => console.error('Cont error:', err));
        }

        function stop() {
            fetch('/api/break', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Break:', data);
                    fetchStatus();
                })
                .catch(err => console.error('Break error:', err));
        }

        function load() {
            fetch('/api/load', { method: 'POST' })
                .then(r => r.json())
                .then(data => console.log('Load:', data))
                .catch(err => console.error('Load error:', err));
        }

        function save() {
            fetch('/api/save', { method: 'POST' })
                .then(r => r.json())
                .then(data => console.log('Save:', data))
                .catch(err => console.error('Save error:', err));
        }

        function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];

            if (!file) {
                alert('Please select a CF32 file first');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                console.log('Upload:', data);
                if (data.status === 'success') {
                    alert('File processed successfully!');
                    fetchStatus();
                } else {
                    alert('File processing failed: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error('Upload error:', err);
                alert('Upload failed: ' + err);
            });
        }

        // Initialize
        window.addEventListener('load', () => {
            initChart();
            fetchStatus();
            setInterval(fetchStatus, 500);  // Poll every 500ms
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (chart) {
                const canvas = chart.canvas;
                const rect = canvas.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                canvas.width = rect.width * dpr;
                canvas.height = rect.height * dpr;
                chart.ctx.scale(dpr, dpr);
                chart.width = rect.width;
                chart.height = rect.height;
                fetchStatus();  // Redraw
            }
        });
    </script>
</body>
</html>
"""

# ============================================================================
# Flask Routes - адаптированы для работы через ZeroMQ
# ============================================================================
@app.route('/')
def index():
    response = Response(HTML_PAGE, mimetype='text/html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
    response.headers['ETag'] = ''
    return response

@app.route('/api/status')
def api_status():
    """
    Главный endpoint для UI - комбинирует данные из:
    1. Кэшированных событий от DSP сервиса (SUB)
    2. Fallback на get_status если нужно (REP)
    """
    # Небольшие вариации для реалистичности
    fs1_var = STATE.fs1_hz + random.uniform(-0.5, 0.5)
    fs2_var = STATE.fs2_hz + random.uniform(-0.5, 0.5)
    fs3_var = STATE.fs3_hz + random.uniform(-0.5, 0.5)

    # Получаем статус от DSP сервиса
    sdr_status = {}
    realtime_rms = -100.0
    realtime_pulse_count = 0
    latest_pulse_info = None

    if dsp_client and dsp_connected:
        with data_lock:
            # Используем кэшированный статус из SUB событий
            if last_status_event:
                sdr_status = {
                    'backend': last_status_event.get('backend', '?'),
                    'backend_args': last_status_event.get('backend_args', ''),
                    'acq_state': last_status_event.get('acq_state', 'stopped'),
                    'ready': last_status_event.get('ready', False),
                    'fs': last_status_event.get('fs', 0),
                    'bb_shift_hz': last_status_event.get('bb_shift_hz', 0),
                    'thresh_dbm': last_status_event.get('thresh_dbm', -60),
                }

                # RMS из последнего статуса
                rms_vals = last_status_event.get('last_rms_dbm', [])
                if rms_vals and len(rms_vals) > 0:
                    realtime_rms = float(rms_vals[-1])

            # Количество импульсов из истории
            realtime_pulse_count = len(pulse_history)

            # Последний импульс из истории
            if pulse_history:
                latest_pulse_info = {
                    'length_ms': pulse_history[-1].get('length_ms', 0),
                    'peak_dbm': pulse_history[-1].get('peak_dbm', 0),
                    'msg_hex': pulse_history[-1].get('msg_hex', ''),
                    'timestamp': pulse_history[-1].get('timestamp', 0),
                }

    # Проверка длин массивов
    phase_len = len(STATE.phase_data) if STATE.phase_data else 0
    xs_ms_len = len(STATE.xs_ms) if STATE.xs_ms else 0
    fm_len = len(STATE.fm_data) if STATE.fm_data else 0
    fm_xs_len = len(STATE.fm_xs_ms) if STATE.fm_xs_ms else 0

    if phase_len > 0 and xs_ms_len > 0 and phase_len != xs_ms_len:
        log.warning(f"Phase data length mismatch! phase_data={phase_len} vs xs_ms={xs_ms_len}")
    if fm_len > 0 and fm_xs_len > 0 and fm_len != fm_xs_len:
        log.warning(f"FM data length mismatch! fm_data={fm_len} vs fm_xs_ms={fm_xs_len}")

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
        'phase_pos_rad': STATE.pos_phase,
        'phase_neg_rad': STATE.neg_phase,
        't_rise_mcs': STATE.ph_rise,
        't_fall_mcs': STATE.ph_fall,
        'p_wt': STATE.p_wt,
        'prise_ms': STATE.prise_ms,
        'bitrate_bps': STATE.bitrate_bps,
        'symmetry_pct': STATE.symmetry_pct,
        'preamble_ms': STATE.preamble_ms,
        'total_ms': STATE.total_ms,
        'rep_period_s': STATE.rep_period_s,
        'rms_dbm': STATE.rms_dbm,
        'freq_hz': STATE.freq_hz,
        't_mod': STATE.t_mod,
        'phase_data': STATE.phase_data,
        'xs_ms': STATE.xs_ms,
        'fm_data': STATE.fm_data,
        'fm_xs_ms': STATE.fm_xs_ms,
        'sdr_device_info': sdr_status.get('backend', 'No DSP connection'),
        'sdr_status': sdr_status,
        'realtime_rms_dbm': realtime_rms,
        'realtime_pulse_count': realtime_pulse_count,
        'latest_pulse': latest_pulse_info,
        'sdr_capture_active': STATE.running,
        'dsp_connected': dsp_connected,
        'dsp_error': dsp_connection_error
    })

@app.route('/api/measure', methods=['POST'])
def api_measure():
    """
    Measure - инициализация/проверка DSP сервиса.
    Было: init_sdr_backend()
    Стало: REQ get_status к DSP сервису
    """
    global dsp_connected, dsp_connection_error

    if not dsp_client:
        success = init_dsp_client()
        if not success:
            return jsonify({
                'status': 'error',
                'sdr_initialized': False,
                'sdr_device_info': dsp_connection_error or 'DSP service not available'
            })

    # Получаем статус от DSP
    rep = dsp_client.get_status()

    if rep.get("ok", False):
        status = rep.get("status", {})
        backend = status.get("backend", "?")
        ready = status.get("ready", False)
        fs = status.get("fs", 0)

        dsp_connected = True
        dsp_connection_error = ""

        device_info = f"{backend} @ {fs/1e6:.3f} MHz, ready={ready}"

        log.info(f"DSP service status: {device_info}")

        return jsonify({
            'status': 'measure triggered',
            'sdr_initialized': ready,
            'sdr_device_info': device_info
        })
    else:
        error = rep.get("error", "Unknown error")
        dsp_connected = False
        dsp_connection_error = error

        log.error(f"DSP service error: {error}")

        return jsonify({
            'status': 'error',
            'sdr_initialized': False,
            'sdr_device_info': f"DSP error: {error}"
        })

@app.route('/api/run', methods=['POST'])
def api_run():
    """
    Run - запуск захвата.
    Было: start_sdr_capture()
    Стало: REQ start_acquire к DSP сервису
    """
    try:
        if not dsp_client:
            return jsonify({
                'status': 'error',
                'running': False,
                'message': 'DSP service not connected'
            }), 500

        # Отправляем команду start_acquire
        rep = dsp_client.start_acquire()

        if rep.get("ok", False):
            STATE.running = True
            acq_state = rep.get("acq_state", "running")

            log.info(f"DSP acquisition started: {acq_state}")

            return jsonify({
                'status': 'running',
                'running': True,
                'message': f'Real-time capture started (state: {acq_state})'
            })
        else:
            error = rep.get("error", "Unknown error")
            STATE.running = False

            log.error(f"DSP start_acquire failed: {error}")

            return jsonify({
                'status': 'error',
                'running': False,
                'message': f'Failed to start capture: {error}'
            }), 500

    except Exception as e:
        STATE.running = False
        log.error(f"RUN error: {e}")
        return jsonify({
            'status': 'error',
            'running': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/cont', methods=['POST'])
def api_cont():
    """Continue - возобновление (пока не используется с DSP)"""
    STATE.running = True
    return jsonify({'status': 'continue', 'running': STATE.running})

@app.route('/api/break', methods=['POST'])
def api_break():
    """
    Break - остановка захвата.
    Было: stop_sdr_capture()
    Стало: REQ stop_acquire к DSP сервису
    """
    try:
        if not dsp_client:
            STATE.running = False
            return jsonify({
                'status': 'stopped',
                'running': False,
                'message': 'DSP service not connected'
            })

        # Отправляем команду stop_acquire
        rep = dsp_client.stop_acquire()

        STATE.running = False

        if rep.get("ok", False):
            acq_state = rep.get("acq_state", "stopped")
            log.info(f"DSP acquisition stopped: {acq_state}")
        else:
            log.warning(f"DSP stop_acquire returned: {rep}")

        return jsonify({
            'status': 'stopped',
            'running': False,
            'message': 'Capture stopped'
        })

    except Exception as e:
        STATE.running = False
        log.error(f"BREAK error: {e}")
        return jsonify({
            'status': 'stopped',
            'running': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/load', methods=['POST'])
def api_load():
    """Load - загрузка конфигурации (не реализовано)"""
    return jsonify({'status': 'load requested'})

@app.route('/api/save', methods=['POST'])
def api_save():
    """Save - сохранение конфигурации (не реализовано)"""
    return jsonify({'status': 'save requested'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """
    Upload CF32 file.
    Теперь использует DSP сервис в file режиме для обработки.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.cf32'):
            return jsonify({'error': 'Only .cf32 files are allowed'}), 400

        filename = secure_filename(file.filename)
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'captures', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)

        file.save(file_path)

        STATE.current_file = file_path
        STATE.message = f"Loaded: {filename}"

        log.info(f"File uploaded: {filename} -> {file_path}, size: {os.path.getsize(file_path)} bytes")

        # Обработка через DSP сервис
        processing_result = process_cf32_file(file_path)

        if processing_result.get("success"):
            STATE.message = f"Processed: {filename} - Message: {STATE.hex_message[:16]}..."
            log.info(f"File processed: {len(STATE.phase_data)} phase samples, {len(STATE.xs_ms)} time samples")

            return jsonify({
                'status': 'success',
                'filename': filename,
                'size': os.path.getsize(file_path),
                'path': file_path,
                'processed': True,
                'message': STATE.message
            })
        else:
            error_msg = processing_result.get("error", "Unknown error")
            STATE.message = f"Error processing {filename}: {error_msg}"
            log.error(f"File processing error: {error_msg}")

            return jsonify({
                'status': 'error',
                'error': error_msg,
                'filename': filename,
                'size': os.path.getsize(file_path),
                'path': file_path,
                'processed': False,
                'message': STATE.message
            }), 400

    except Exception as e:
        log.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/state')
def api_state():
    """
    Получение текущего состояния системы.
    Теперь возвращает данные от DSP сервиса вместо локальных.
    """
    with data_lock:
        # Формируем статус на основе кэша от DSP
        sdr_info = {}
        if last_status_event:
            sdr_info = {
                'backend': last_status_event.get('backend', '?'),
                'acq_state': last_status_event.get('acq_state', 'stopped'),
                'fs': last_status_event.get('fs', 0),
                'bb_shift_hz': last_status_event.get('bb_shift_hz', 0),
                'thresh_dbm': last_status_event.get('thresh_dbm', -60),
                'ready': last_status_event.get('ready', False),
            }

        return jsonify({
            'sdr_running': STATE.running,
            'sdr_device_info': sdr_info.get('backend', 'No connection'),
            'current_rms_dbm': STATE.rms_dbm,
            'actual_sample_rate_sps': sdr_info.get('fs', 0),
            'dsp_connected': dsp_connected,
            'dsp_info': sdr_info,
            'pulse_count': len(pulse_history),
            'timestamp': time.time()
        })

@app.route('/api/last_pulse')
def api_last_pulse():
    """
    Получение информации о последнем импульсе.
    Теперь из кэша pulse событий от DSP сервиса.
    """
    with data_lock:
        # Последние 20 импульсов из истории
        history_items = []
        for pulse in list(pulse_history)[-20:]:
            item = {
                'start_abs': pulse.get('start_abs', 0),
                'length_ms': pulse.get('length_ms', 0),
                'peak_dbm': pulse.get('peak_dbm', 0),
                'timestamp': pulse.get('timestamp', 0),
                'msg_hex_short': (pulse.get('msg_hex', '') or '')[:64]
            }
            history_items.append(item)

        # Последний импульс
        last = None
        if pulse_history:
            last_pulse = pulse_history[-1]
            last = {
                'start_abs': last_pulse.get('start_abs', 0),
                'length_ms': last_pulse.get('length_ms', 0),
                'peak_dbm': last_pulse.get('peak_dbm', 0),
                'timestamp': last_pulse.get('timestamp', 0),
                'msg_hex': last_pulse.get('msg_hex', ''),
                'msg_hex_short': (last_pulse.get('msg_hex', '') or '')[:64],
                'phase_metrics': last_pulse.get('phase_metrics'),
            }

        return jsonify({
            'last': last,
            'history': history_items,
            'pulse_queue_size': len(pulse_history),
            'timestamp': time.time()
        })

# ============================================================================
# Application startup
# ============================================================================
if __name__ == '__main__':
    log.info("=" * 60)
    log.info("COSPAS/SARSAT Beacon Tester v2.1 (ZeroMQ Edition)")
    log.info("=" * 60)
    log.info("Web interface: http://127.0.0.1:8738/")
    log.info(f"DSP service PUB: {DEFAULT_PUB_ADDR}")
    log.info(f"DSP service REP: {DEFAULT_REP_ADDR}")
    log.info("-" * 60)

    # Инициализация DSP клиента при старте
    init_success = init_dsp_client()

    if init_success:
        log.info("DSP client initialized successfully")
    else:
        log.warning("DSP client initialization failed - will retry on Measure")
        log.warning("Make sure beacon_dsp_service.py is running:")
        log.warning("  python beacon406/beacon_dsp_service.py")

    log.info("=" * 60)
    log.info("To stop: Ctrl+C")
    log.info("=" * 60)

    # Запуск Flask приложения
    app.run(host='127.0.0.1', port=8738, debug=False, threaded=True)
