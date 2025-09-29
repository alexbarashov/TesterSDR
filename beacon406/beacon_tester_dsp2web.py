"""
COSPAS/SARSAT Beacon Tester DSP2Web - Full UI Client v2.1
=========================================================
Тонкий веб-клиент с полным UI для подключения к beacon_dsp_service.py

Архитектура:
- ZeroMQ SUB клиент для получения событий (status/pulse/psk)
- ZeroMQ REQ клиент для отправки команд
- Flask веб-сервер с полным UI и REST API
- Без локальной DSP/SDR логики

Запуск:
python beacon_tester_dsp2web.py [--pub tcp://127.0.0.1:8781] [--rep tcp://127.0.0.1:8782] [--host 127.0.0.1] [--port 8738]
"""

import os
import sys
import time
import json
import queue
import threading
import argparse
from collections import deque
from typing import Dict, Any, Optional

import zmq
from flask import Flask, render_template_string, jsonify, request, Response

# Настройка логирования
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.logger import get_logger, setup_logging
from lib.hex_decoder import hex_to_bits, build_table_rows

setup_logging()
log = get_logger(__name__)

app = Flask(__name__)

# === Конфигурация адресов ===
DEFAULT_PUB_ADDR = "tcp://127.0.0.1:8781"
DEFAULT_REP_ADDR = "tcp://127.0.0.1:8782"
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8738

# === Глобальное состояние ===
class DSPState:
    def __init__(self):
        self.status = {}
        self.pulse = None
        self.psk = None
        self.log = deque(maxlen=64)
        self.last_status_time = 0
        self.lock = threading.Lock()

dsp_state = DSPState()

# === ZeroMQ клиенты ===
zmq_context = None
sub_socket = None
req_socket = None
req_lock = threading.Lock()

# === SSE очередь ===
events_queue = queue.Queue(maxsize=100)

# === Конфигурация подключения ===
pub_addr = DEFAULT_PUB_ADDR
rep_addr = DEFAULT_REP_ADDR

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='COSPAS/SARSAT Beacon Tester DSP2Web Full UI Client')
    parser.add_argument('--pub', default=None, help=f'PUB address for SUB connection (default: {DEFAULT_PUB_ADDR})')
    parser.add_argument('--rep', default=None, help=f'REP address for REQ connection (default: {DEFAULT_REP_ADDR})')
    parser.add_argument('--host', default=DEFAULT_WEB_HOST, help=f'Web server host (default: {DEFAULT_WEB_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_WEB_PORT, help=f'Web server port (default: {DEFAULT_WEB_PORT})')
    return parser.parse_args()

def get_zmq_addresses():
    """Получение адресов ZeroMQ с учетом приоритета CLI -> ENV -> default"""
    global pub_addr, rep_addr

    args = parse_args()

    # PUB адрес: CLI -> ENV -> default
    if args.pub:
        pub_addr = args.pub
    elif os.getenv('DSP_PUB'):
        pub_addr = os.getenv('DSP_PUB')
    else:
        pub_addr = DEFAULT_PUB_ADDR

    # REP адрес: CLI -> ENV -> default
    if args.rep:
        rep_addr = args.rep
    elif os.getenv('DSP_REP'):
        rep_addr = os.getenv('DSP_REP')
    else:
        rep_addr = DEFAULT_REP_ADDR

    return args.host, args.port

def init_zmq():
    """Инициализация ZeroMQ контекста и сокетов"""
    global zmq_context, sub_socket, req_socket

    zmq_context = zmq.Context()

    # SUB сокет для получения событий
    sub_socket = zmq_context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Подписка на все сообщения
    sub_socket.setsockopt(zmq.RCVTIMEO, 1000)  # Таймаут 1 сек для переподключения

    # REQ сокет для команд
    req_socket = zmq_context.socket(zmq.REQ)
    req_socket.setsockopt(zmq.RCVTIMEO, 2000)  # Таймаут 2 сек
    req_socket.setsockopt(zmq.SNDTIMEO, 2000)

def connect_sub():
    """Подключение SUB сокета с переподключением"""
    backoff = 0.5
    max_backoff = 5.0

    while True:
        try:
            log.info(f"Connecting SUB to {pub_addr}")
            sub_socket.connect(pub_addr)
            return True
        except Exception as e:
            log.error(f"SUB connection failed: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

def send_command(cmd_data: Dict[str, Any]) -> Dict[str, Any]:
    """Отправка команды через REQ сокет с повторными попытками"""
    with req_lock:
        # Подключение REQ сокета при первом использовании
        if not hasattr(send_command, '_connected'):
            try:
                log.info(f"Connecting REQ to {rep_addr}")
                req_socket.connect(rep_addr)
                send_command._connected = True
            except Exception as e:
                log.error(f"REQ connection failed: {e}")
                return {"ok": False, "err": f"Connection failed: {e}"}

        # Повторные попытки отправки
        for attempt in range(3):
            try:
                req_socket.send_json(cmd_data)
                response = req_socket.recv_json()
                return response
            except zmq.Again:
                log.warning(f"REQ timeout, attempt {attempt + 1}/3")
                if attempt == 2:
                    return {"ok": False, "err": "Request timeout"}
            except Exception as e:
                log.error(f"REQ error: {e}")
                return {"ok": False, "err": str(e)}

def sub_thread():
    """Поток для получения событий через SUB"""
    log.info("Starting SUB thread")

    # Подключение с переподключением
    connect_sub()

    backoff = 0.5
    max_backoff = 5.0

    while True:
        try:
            # Получение сообщения
            message = sub_socket.recv_string(zmq.DONTWAIT)
            backoff = 0.5  # Сброс бэкоффа при успешном получении

            try:
                event = json.loads(message)
                process_event(event)
            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON from SUB: {e}")

        except zmq.Again:
            # Таймаут - нормально, продолжаем
            time.sleep(0.01)
        except Exception as e:
            log.error(f"SUB error: {e}, reconnecting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

            # Переподключение
            try:
                sub_socket.disconnect(pub_addr)
            except:
                pass
            connect_sub()

def process_event(event: Dict[str, Any]):
    """Обработка полученного события"""
    event_type = event.get('type')

    with dsp_state.lock:
        if event_type == 'status':
            dsp_state.status = event
            dsp_state.last_status_time = time.time()
            add_log_entry('status', f"Status update: {event.get('sdr', 'unknown')}")

        elif event_type == 'pulse':
            dsp_state.pulse = event
            add_log_entry('pulse', f"Pulse: {event.get('length_ms', 0):.1f}ms, {event.get('peak_dbm', -999):.1f}dBm")

        elif event_type == 'psk':
            dsp_state.psk = event
            hex_data = event.get('hex', '')
            status = 'OK' if event.get('ok') else 'FAIL'
            add_log_entry('psk', f"PSK {status}: {len(hex_data)} hex chars")

    # Отправка в SSE очередь (без блокировки)
    try:
        # Создаем компактную версию события для SSE
        compact_event = {
            'type': event_type,
            'timestamp': time.time()
        }

        if event_type == 'status':
            compact_event.update({
                'sdr': event.get('sdr'),
                'cpu': event.get('cpu', 0)
            })
        elif event_type == 'pulse':
            compact_event.update({
                'length_ms': event.get('length_ms'),
                'peak_dbm': event.get('peak_dbm')
            })
        elif event_type == 'psk':
            compact_event.update({
                'ok': event.get('ok'),
                'hex_length': len(event.get('hex', ''))
            })

        # Добавляем в очередь, удаляя старые при переполнении
        if events_queue.full():
            try:
                events_queue.get_nowait()
            except queue.Empty:
                pass
        events_queue.put(compact_event)

    except Exception as e:
        log.warning(f"Failed to queue SSE event: {e}")

def add_log_entry(event_type: str, message: str):
    """Добавление записи в лог"""
    entry = {
        'timestamp': time.time(),
        'type': event_type,
        'message': message
    }
    dsp_state.log.append(entry)

# === Flask маршруты ===

@app.route('/')
def index():
    """Главная страница"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/state')
def api_state():
    """Получение текущего состояния"""
    with dsp_state.lock:
        return jsonify({
            'status': dsp_state.status,
            'last_pulse': dsp_state.pulse,
            'last_psk': dsp_state.psk,
            'log': list(dsp_state.log)
        })

@app.route('/api/control/start', methods=['POST'])
def api_start():
    """Команда start_acquire"""
    response = send_command({"cmd": "start_acquire"})
    return jsonify(response)

@app.route('/api/control/stop', methods=['POST'])
def api_stop():
    """Команда stop_acquire"""
    response = send_command({"cmd": "stop_acquire"})
    return jsonify(response)

@app.route('/api/control/params', methods=['POST'])
def api_params():
    """Команда set_params"""
    params = request.get_json() or {}

    # Создаем команду с плоской структурой
    cmd_data = {"cmd": "set_params"}
    cmd_data.update(params)

    response = send_command(cmd_data)
    return jsonify(response)

@app.route('/api/control/save_sigmf', methods=['POST'])
def api_save_sigmf():
    """Команда save_sigmf"""
    response = send_command({"cmd": "save_sigmf"})
    return jsonify(response)

@app.route('/api/control/get_status', methods=['POST'])
def api_get_status():
    """Команда get_status"""
    response = send_command({"cmd": "get_status"})
    return jsonify(response)

@app.route('/api/health')
def api_health():
    """Проверка состояния подключения"""
    with dsp_state.lock:
        # Эвристика: есть ли status за последние 10 секунд
        zmq_connected = (time.time() - dsp_state.last_status_time) < 10.0

        return jsonify({
            'ok': True,
            'zmq_connected': zmq_connected,
            'pub_addr': pub_addr,
            'rep_addr': rep_addr
        })

@app.route('/api/events')
def api_events():
    """Server-Sent Events для real-time обновлений"""
    def event_stream():
        while True:
            try:
                # Получаем событие с таймаутом
                event = events_queue.get(timeout=30)

                event_type = event.get('type', 'unknown')
                event_data = json.dumps(event)

                yield f"event: {event_type}\ndata: {event_data}\n\n"

            except queue.Empty:
                # Heartbeat каждые 30 секунд
                yield f"event: heartbeat\ndata: {{}}\n\n"
            except Exception as e:
                log.error(f"SSE error: {e}")
                break

    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/last_pulse')
def api_last_pulse():
    """Подробная информация о последнем импульсе"""
    with dsp_state.lock:
        pulse_data = dsp_state.pulse
        psk_data = dsp_state.psk

        # Объединяем данные импульса и PSK если они относятся к одному событию
        result = {}
        if pulse_data:
            result.update(pulse_data)
        if psk_data:
            result.update(psk_data)

        return jsonify(result if result else None)

@app.route('/api/sdr/get_config')
def api_sdr_get_config():
    """Получение текущей конфигурации SDR"""
    try:
        # Отправить запрос на получение конфигурации DSP сервису
        response = send_dsp_command({'cmd': 'get_sdr_config'})
        if response and response.get('ok'):
            return jsonify(response.get('config', {}))
        else:
            # Возвратить конфигурацию по умолчанию если DSP сервис недоступен
            return jsonify({
                'center_freq_hz': 406000000,
                'sample_rate_sps': 1000000,
                'bb_shift_enable': True,
                'bb_shift_hz': -37000,
                'freq_corr_hz': 0,
                'agc': False,
                'gain_db': 30.0,
                'bias_t': False,
                'antenna': 'RX',
                'device': 'Not connected'
            })
    except Exception as e:
        log.error(f"Failed to get SDR config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sdr/set_config', methods=['POST'])
def api_sdr_set_config():
    """Применение конфигурации SDR"""
    try:
        config_data = request.get_json()
        if not config_data:
            return jsonify({'ok': False, 'error': 'No config data provided'}), 400

        # Отправить команду на изменение конфигурации
        command = {
            'cmd': 'set_sdr_config',
            'config': config_data
        }

        response = send_dsp_command(command)
        if response and response.get('ok'):
            return jsonify({
                'ok': True,
                'applied': response.get('applied', {}),
                'retuned': response.get('retuned', False)
            })
        else:
            return jsonify({
                'ok': False,
                'error': response.get('error', 'Failed to apply SDR config')
            }), 500

    except Exception as e:
        log.error(f"Failed to set SDR config: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500

# === Полный HTML шаблон с адаптацией под DSP2Web ===
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COSPAS/SARSAT Beacon Tester — DSP Client v2.1</title>
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
            position: relative;
        }

        /* Индикатор подключения */
        .connection-status {
            position: absolute;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 14px;
            font-weight: normal;
        }
        .connected { color: #90ee90; }
        .disconnected { color: #ff6b6b; }

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

        .button {
            background: linear-gradient(180deg, #e8f4f8 0%, #d1e7f0 100%);
            border: 1px solid #a8c8e4;
            border-radius: 3px;
            padding: 6px 12px;
            margin: 2px;
            cursor: pointer;
            font-size: 13px;
            color: #2c3e50;
            width: 100%;
            text-align: center;
        }

        .button:hover {
            background: linear-gradient(180deg, #d1e7f0 0%, #b8dce8 100%);
        }

        .button.primary {
            background: linear-gradient(180deg, #5a9bd4 0%, #4a8bc2 100%);
            color: white;
            font-weight: bold;
        }

        .button.primary:hover {
            background: linear-gradient(180deg, #4a8bc2 0%, #3a7bb0 100%);
        }

        /* Центральная область */
        .main-content {
            flex: 1;
            background: #f8f9fa;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }

        .info-line {
            background: white;
            padding: 8px 12px;
            margin-bottom: 8px;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 13px;
            color: #495057;
        }

        .message-line {
            background: #e8f5e8;
            padding: 8px 12px;
            margin-bottom: 8px;
            border: 1px solid #c3e6cb;
            border-radius: 3px;
            font-size: 13px;
            color: #155724;
        }

        /* График */
        .chart-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 8px;
            position: relative;
            flex: 1;
            margin-bottom: 8px;
        }

        #phaseChart {
            width: 100%;
            height: 100%;
            display: block;
        }

        .phase-values {
            display: flex;
            justify-content: space-around;
            background: #f8f9fa;
            padding: 6px;
            border-top: 1px solid #dee2e6;
            font-size: 13px;
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
            font-size: 13px;
            text-align: center;
            padding: 6px;
            border: 1px solid #6699cc;
            border-radius: 3px;
            margin-bottom: 6px;
        }

        .stats-content {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 8px;
            font-size: 12px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
        }

        .stats-table th,
        .stats-table td {
            padding: 4px 6px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
            font-size: 12px;
        }

        .stats-table th {
            background: #e9ecef;
            font-weight: bold;
        }

        .stats-table tr:hover {
            background: #f5f5f5;
        }

        /* История событий */
        .history-section {
            margin-top: 15px;
        }

        .history-entry {
            padding: 4px 6px;
            border-bottom: 1px solid #e9ecef;
            font-size: 11px;
            color: #6c757d;
        }

        .history-entry.pulse {
            border-left: 3px solid #28a745;
        }

        .history-entry.psk {
            border-left: 3px solid #fd7e14;
        }

        .history-entry.status {
            border-left: 3px solid #007bff;
        }

        /* МОДАЛЬНОЕ ОКНО SDR CONFIG */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }

        .modal-content {
            background-color: #fefefe;
            margin: 5% auto;
            padding: 0;
            border: 1px solid #888;
            width: 80%;
            max-width: 800px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .modal-header {
            background: linear-gradient(180deg, #7bb3d9 0%, #5a9bd4 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h2 {
            margin: 0;
            font-size: 18px;
        }

        .close {
            color: white;
            font-size: 24px;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
        }

        .close:hover {
            color: #ddd;
        }

        .modal-tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            background: #f8f9fa;
        }

        .tab-button {
            flex: 1;
            padding: 12px 16px;
            border: none;
            background: transparent;
            cursor: pointer;
            font-size: 14px;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }

        .tab-button:hover {
            background: #e9ecef;
        }

        .tab-button.active {
            background: white;
            border-bottom-color: #5a9bd4;
            color: #5a9bd4;
            font-weight: 600;
        }

        .tab-content {
            display: none;
            padding: 20px;
            min-height: 400px;
        }

        .tab-content.active {
            display: block;
        }

        .config-grid {
            display: grid;
            gap: 16px;
        }

        .config-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .config-row label {
            min-width: 180px;
            font-size: 14px;
            font-weight: 500;
        }

        .config-row input[type="text"],
        .config-row input[type="number"],
        .config-row select {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .config-row input[type="checkbox"] {
            transform: scale(1.2);
        }

        .modal-footer {
            padding: 16px 20px;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            background: #f8f9fa;
            border-radius: 0 0 8px 8px;
        }

        .preset-btn {
            margin-right: auto;
        }
    </style>
</head>
<body>
    <div class="header">
        COSPAS/SARSAT Beacon Tester — DSP Client v2.1
        <div class="connection-status" id="connectionStatus">
            <span id="connection-indicator">● Connecting...</span>
        </div>
    </div>

    <div class="container">
        <!-- Левая панель -->
        <div class="left-panel">
            <div class="panel-section">
                <div class="section-header">VIEW</div>
                <div class="section-content">
                    <div class="radio-group">
                        <label><input type="radio" name="view" value="phase" checked onchange="changeView('phase')"> 406 Phase</label>
                        <label><input type="radio" name="view" value="inburst_fr" onchange="changeView('inburst_fr')"> 406 in-burst FR</label>
                        <label><input type="radio" name="view" value="fr_stability" onchange="changeView('fr_stability')"> 406 Fr. stability</label>
                        <label><input type="radio" name="view" value="ph_rise_fall" onchange="changeView('ph_rise_fall')"> 406 Ph/Rise/Fall</label>
                        <label><input type="radio" name="view" value="message" onchange="changeView('message')"> Message</label>
                        <label><input type="radio" name="view" value="sum_table" onchange="changeView('sum_table')"> Sum table</label>
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
                            <option value="5">5%</option>
                            <option value="10" selected>10%</option>
                            <option value="25">25%</option>
                            <option value="50">50%</option>
                            <option value="100">100%</option>
                        </select>
                    </div>
                    <div class="control-row">
                        <label style="font-size: 12px;">
                            <input type="checkbox" id="showLastDataOnSwitch" checked onchange="onShowLastDataToggle()">
                            Show last data on view switch
                        </label>
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <div class="section-header">DSP CONTROL</div>
                <div class="section-content">
                    <button class="button primary" onclick="startAcquire()">Start</button>
                    <button class="button" onclick="stopAcquire()">Stop</button>
                    <button class="button" onclick="getStatus()">Status</button>
                    <button class="button" onclick="saveSigMF()">Save SigMF</button>
                </div>
            </div>

            <div class="panel-section">
                <div class="section-header">PARAMS</div>
                <div class="section-content">
                    <div class="control-row">
                        <span>Target signal (Hz)</span>
                        <input type="text" class="control-input" id="target_signal_hz" value="406037000" placeholder="406.037M">
                    </div>
                    <div class="control-row">
                        <span>Threshold (dBm)</span>
                        <input type="number" class="control-input" id="thresh_dbm" value="-45" step="0.5" min="-120" max="0">
                    </div>
                    <div class="control-row">
                        <button class="button" onclick="applyParams()">Apply</button>
                        <button class="button" onclick="openSDRConfig()">SDR…</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Центральная область -->
        <div class="main-content">
            <div class="info-line">Protocol: <span id="protocol">C/S-406</span> | Date: <span id="date">2025-09-29</span> | Model: <span id="beaconModel">EPIRB</span> | Freq: <span id="beaconFreq">406.037</span> MHz</div>
            <div class="message-line">Message: <span id="message">[no message]</span></div>

            <div class="chart-container">
                <canvas id="phaseChart"></canvas>
            </div>

            <div class="phase-values">
                <span>Phase+ = <span id="phasePlus">—</span>°</span>
                <span>TRise+ = <span id="tRise">—</span> μs</span>
                <span>Phase- = <span id="phaseMinus">—</span>°</span>
                <span>TFall- = <span id="tFall">—</span> μs</span>
            </div>

            <div class="chart-title" id="chartTitle">Fig.8 Phase</div>
        </div>

        <!-- Правая панель -->
        <div class="right-panel">
            <div class="stats-header">Signal Parameters</div>
            <div class="stats-content" id="statsContent">
                <table class="stats-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Target freq, kHz</td><td id="target-frequency">—</td></tr>
                    <tr><td>Frequency, kHz</td><td id="signal-frequency">—</td></tr>
                    <tr><td>+Phase deviation, rad</td><td id="signal-pos-phase">—</td></tr>
                    <tr><td>−Phase deviation, rad</td><td id="signal-neg-phase">—</td></tr>
                    <tr><td>Phase time rise, µs</td><td id="signal-rise-us">—</td></tr>
                    <tr><td>Phase time fall, µs</td><td id="signal-fall-us">—</td></tr>
                    <tr><td>Power, Wt</td><td id="signal-power">—</td></tr>
                    <tr><td>Power rise, ms</td><td id="signal-power-rise">—</td></tr>
                    <tr><td>Bit Rate, bps</td><td id="signal-baud">—</td></tr>
                    <tr><td>Asymmetry, %</td><td id="signal-asymmetry">—</td></tr>
                    <tr><td>CW Preamble, ms</td><td id="signal-preamble">—</td></tr>
                    <tr><td>Total burst duration, ms</td><td id="signal-duration">—</td></tr>
                    <tr><td>Repetition period, s</td><td id="signal-period">—</td></tr>
                </table>

                <div class="history-section">
                    <div class="stats-header" style="margin-top: 10px;">Recent Events</div>
                    <div id="eventHistory" style="max-height: 200px; overflow-y: auto;">
                        <!-- События будут добавляться JavaScript -->
                    </div>
                </div>
            </div>        </div>
    </div>

    <script>
        console.log('=== DSP2WEB FULL UI CLIENT v2.1 LOADED ===');

        // Глобальные переменные
        let canvas = document.getElementById('phaseChart');
        let ctx = canvas.getContext('2d');
        let currentView = 'phase';
        let currentTimeScale = 10;
        let eventSource = null;
        let showLastDataOnSwitch = true; // Состояние чекбокса

        // === УПРАВЛЕНИЕ СОЕДИНЕНИЕМ ===
        function updateConnectionStatus(connected, zmqConnected = false) {
            const indicator = document.getElementById('connection-indicator');
            if (connected && zmqConnected) {
                indicator.innerHTML = '● Connected to DSP';
                indicator.className = 'connected';
            } else if (connected) {
                indicator.innerHTML = '● Connected (DSP offline)';
                indicator.className = 'disconnected';
            } else {
                indicator.innerHTML = '● Disconnected';
                indicator.className = 'disconnected';
            }
        }

        function checkHealth() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    updateConnectionStatus(data.ok, data.zmq_connected);
                })
                .catch(error => {
                    updateConnectionStatus(false);
                });
        }

        // === SSE EVENT STREAM ===
        function startEventStream() {
            if (eventSource) {
                eventSource.close();
            }

            eventSource = new EventSource('/api/events');

            eventSource.addEventListener('status', function(e) {
                const data = JSON.parse(e.data);
                updateLiveStatus(data);
                logEvent('status', 'Status update received');
            });

            eventSource.addEventListener('pulse', function(e) {
                const data = JSON.parse(e.data);
                logEvent('pulse', `Pulse: ${data.length_ms?.toFixed(1) || '?'}ms, ${data.peak_dbm?.toFixed(1) || '?'}dBm`);

                // Умное обновление в зависимости от текущего режима просмотра
                if (currentView === 'phase' || currentView === 'inburst_fr') {
                    loadDataForView(currentView, true); // forceLoad для SSE событий
                } else if (currentView === 'message' || currentView === 'sum_table') {
                    loadLastPulse();
                }
                // Для остальных режимов не обновляем автоматически
            });

            eventSource.addEventListener('psk', function(e) {
                const data = JSON.parse(e.data);
                const status = data.ok ? 'OK' : 'FAIL';
                logEvent('psk', `PSK ${status}: ${data.hex_length || 0} hex chars`);

                // Умное обновление в зависимости от текущего режима просмотра
                if (currentView === 'message' || currentView === 'sum_table') {
                    loadLastPulse(); // PSK события особенно важны для сообщений
                } else if (currentView === 'phase' || currentView === 'inburst_fr') {
                    loadDataForView(currentView, true); // forceLoad для SSE событий
                }
                // Для остальных режимов не обновляем автоматически
            });

            eventSource.addEventListener('heartbeat', function(e) {
                // Heartbeat received - connection is alive
            });

            eventSource.onopen = function() {
                updateConnectionStatus(true, true);
            };

            eventSource.onerror = function() {
                updateConnectionStatus(false);
            };
        }

        // === API ФУНКЦИИ ===
        function sendCommand(url, data = {}) {
            return fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if (result.ok === false) {
                    logEvent('status', `Command failed: ${result.err || 'Unknown error'}`);
                    throw new Error(result.err || 'Command failed');
                }
                return result;
            })
            .catch(error => {
                logEvent('status', `Request failed: ${error.message}`);
                throw error;
            });
        }

        function startAcquire() {
            sendCommand('/api/control/start')
                .then(result => {
                    logEvent('status', 'Acquisition started');
                    refreshState();
                });
        }

        function stopAcquire() {
            sendCommand('/api/control/stop')
                .then(result => {
                    logEvent('status', 'Acquisition stopped');
                    refreshState();
                });
        }

        function getStatus() {
            sendCommand('/api/control/get_status')
                .then(result => {
                    logEvent('status', 'Status requested');
                    refreshState();
                });
        }

        function saveSigMF() {
            sendCommand('/api/control/save_sigmf')
                .then(result => {
                    logEvent('status', 'SigMF save requested');
                });
        }

        function applyParams() {
            const targetSignalInput = document.getElementById('target_signal_hz').value;
            const threshInput = document.getElementById('thresh_dbm').value;

            // Парсинг target signal с поддержкой суффиксов
            const targetSignalHz = parseFrequencyInput(targetSignalInput);
            const threshDbm = parseFloat(threshInput);

            if (isNaN(targetSignalHz) || isNaN(threshDbm)) {
                logEvent('error', 'Invalid parameter values');
                return;
            }

            const params = {
                target_signal_hz: targetSignalHz,
                thresh_dbm: threshDbm
            };

            sendCommand('/api/control/params', params)
                .then(result => {
                    logEvent('status', 'Parameters updated');
                    // Обновить отображаемое значение Target в статусе
                    updateTargetDisplay(targetSignalHz);
                })
                .catch(error => {
                    logEvent('error', `Failed to update parameters: ${error.message}`);
                });
        }

        // === ПАРСИНГ ЧАСТОТЫ С СУФФИКСАМИ ===
        function parseFrequencyInput(input) {
            if (!input) return NaN;

            const trimmed = input.trim().toLowerCase();
            let value = parseFloat(trimmed);

            if (isNaN(value)) return NaN;

            // Обработка суффиксов
            if (trimmed.includes('ghz') || trimmed.includes('g')) {
                value *= 1e9;
            } else if (trimmed.includes('mhz') || trimmed.includes('m')) {
                value *= 1e6;
            } else if (trimmed.includes('khz') || trimmed.includes('k')) {
                value *= 1e3;
            }

            return Math.round(value);
        }

        // === ОБНОВЛЕНИЕ ОТОБРАЖЕНИЯ TARGET ===
        function updateTargetDisplay(targetHz) {
            if (targetHz) {
                // Обновить в строке статуса (MHz)
                const beaconFreqElement = document.getElementById('beaconFreq');
                if (beaconFreqElement) {
                    const targetMHz = (targetHz / 1e6).toFixed(3);
                    beaconFreqElement.textContent = targetMHz;
                }

                // Обновить в Signal Parameters (kHz)
                const targetFreqElement = document.getElementById('target-frequency');
                if (targetFreqElement) {
                    const targetKHz = (targetHz / 1e3).toFixed(1);
                    targetFreqElement.textContent = targetKHz;
                }
            }
        }

        // === ОТКРЫТИЕ SDR CONFIG ===
        function openSDRConfig() {
            loadSDRConfig();
            document.getElementById('sdrModal').style.display = 'block';
            logEvent('status', 'SDR Config modal opened');
        }

        function closeSDRConfig() {
            document.getElementById('sdrModal').style.display = 'none';
        }

        function switchTab(tabName) {
            // Скрыть все вкладки
            document.getElementById('radioTab').classList.remove('active');
            document.getElementById('recordingTab').classList.remove('active');

            // Убрать активный класс со всех кнопок
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));

            // Показать нужную вкладку
            document.getElementById(tabName + 'Tab').classList.add('active');

            // Активировать нужную кнопку
            event.target.classList.add('active');
        }

        function loadSDRConfig() {
            // Загрузить текущую конфигурацию SDR
            fetch('/api/sdr/get_config')
                .then(response => response.json())
                .then(config => {
                    document.getElementById('center_freq_hz').value = config.center_freq_hz || '';
                    document.getElementById('sample_rate_sps').value = config.sample_rate_sps || '';
                    document.getElementById('bb_shift_hz').value = config.bb_shift_hz || '';
                    document.getElementById('bb_shift_enable').checked = config.bb_shift_enable || false;
                    document.getElementById('freq_corr_hz').value = config.freq_corr_hz || '';
                    document.getElementById('agc').checked = config.agc || false;
                    document.getElementById('gain_db').value = config.gain_db || '';
                    document.getElementById('bias_t').checked = config.bias_t || false;
                    document.getElementById('antenna').value = config.antenna || 'RX';
                    document.getElementById('device').value = config.device || 'Not detected';
                })
                .catch(error => {
                    logEvent('error', `Failed to load SDR config: ${error.message}`);
                });
        }

        function applySDRConfig() {
            // Собрать данные из формы
            const config = {
                center_freq_hz: parseFrequencyInput(document.getElementById('center_freq_hz').value),
                sample_rate_sps: parseFrequencyInput(document.getElementById('sample_rate_sps').value),
                bb_shift_hz: parseFrequencyInput(document.getElementById('bb_shift_hz').value),
                bb_shift_enable: document.getElementById('bb_shift_enable').checked,
                freq_corr_hz: parseFrequencyInput(document.getElementById('freq_corr_hz').value),
                agc: document.getElementById('agc').checked,
                gain_db: parseFloat(document.getElementById('gain_db').value),
                bias_t: document.getElementById('bias_t').checked,
                antenna: document.getElementById('antenna').value
            };

            // Отправить конфигурацию
            fetch('/api/sdr/set_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(result => {
                if (result.ok) {
                    logEvent('status', 'SDR configuration applied' + (result.retuned ? ' (retuned)' : ''));
                    closeSDRConfig();
                } else {
                    logEvent('error', `Failed to apply SDR config: ${result.error || 'Unknown error'}`);
                }
            })
            .catch(error => {
                logEvent('error', `Failed to apply SDR config: ${error.message}`);
            });
        }

        function resetDefaults() {
            // Загрузить значения по умолчанию
            document.getElementById('center_freq_hz').value = '406000000';
            document.getElementById('sample_rate_sps').value = '1000000';
            document.getElementById('bb_shift_hz').value = '-37000';
            document.getElementById('bb_shift_enable').checked = true;
            document.getElementById('freq_corr_hz').value = '0';
            document.getElementById('agc').checked = false;
            document.getElementById('gain_db').value = '30';
            document.getElementById('bias_t').checked = false;
            document.getElementById('antenna').value = 'RX';
        }

        function loadPreset() {
            const presetName = prompt('Enter preset name to load:');
            if (!presetName) return;

            try {
                const presets = JSON.parse(localStorage.getItem('dsp2web.sdr.presets') || '{}');
                const preset = presets[presetName];

                if (!preset) {
                    logEvent('error', `Preset '${presetName}' not found`);
                    alert(`Preset '${presetName}' not found`);
                    return;
                }

                // Загрузить значения из пресета
                document.getElementById('center_freq_hz').value = preset.center_freq_hz || '';
                document.getElementById('sample_rate_sps').value = preset.sample_rate_sps || '';
                document.getElementById('bb_shift_hz').value = preset.bb_shift_hz || '';
                document.getElementById('bb_shift_enable').checked = preset.bb_shift_enable || false;
                document.getElementById('freq_corr_hz').value = preset.freq_corr_hz || '';
                document.getElementById('agc').checked = preset.agc || false;
                document.getElementById('gain_db').value = preset.gain_db || '';
                document.getElementById('bias_t').checked = preset.bias_t || false;
                document.getElementById('antenna').value = preset.antenna || 'RX';

                logEvent('status', `Preset '${presetName}' loaded`);

            } catch (error) {
                logEvent('error', `Failed to load preset: ${error.message}`);
            }
        }

        function savePreset() {
            const presetName = prompt('Enter preset name to save:');
            if (!presetName) return;

            try {
                // Собрать текущую конфигурацию
                const preset = {
                    center_freq_hz: document.getElementById('center_freq_hz').value,
                    sample_rate_sps: document.getElementById('sample_rate_sps').value,
                    bb_shift_hz: document.getElementById('bb_shift_hz').value,
                    bb_shift_enable: document.getElementById('bb_shift_enable').checked,
                    freq_corr_hz: document.getElementById('freq_corr_hz').value,
                    agc: document.getElementById('agc').checked,
                    gain_db: document.getElementById('gain_db').value,
                    bias_t: document.getElementById('bias_t').checked,
                    antenna: document.getElementById('antenna').value,
                    saved_at: new Date().toISOString()
                };

                // Загрузить существующие пресеты
                const presets = JSON.parse(localStorage.getItem('dsp2web.sdr.presets') || '{}');
                presets[presetName] = preset;

                // Сохранить обновленные пресеты
                localStorage.setItem('dsp2web.sdr.presets', JSON.stringify(presets));

                logEvent('status', `Preset '${presetName}' saved`);

            } catch (error) {
                logEvent('error', `Failed to save preset: ${error.message}`);
            }
        }

        // === ОБНОВЛЕНИЕ СОСТОЯНИЯ ===
        function refreshState() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    updateCurrentStats(data);
                    if (data.log && data.log.length > 0) {
                        updateEventHistory(data.log);
                    }
                })
                .catch(error => {
                    console.error('Error refreshing state:', error);
                });
        }

        function loadLastPulse() {
            fetch('/api/last_pulse')
                .then(response => response.json())
                .then(data => {
                    if (data) {
                        updateDisplay(data);
                        renderSignalParams(data);
                    }
                })
                .catch(error => {
                    console.error('Error loading last pulse:', error);
                });
        }

        // === ОБНОВЛЕНИЕ UI ===
        function updateLiveStatus(data) {
            // Обновляем live индикаторы из SSE событий
            if (data.sdr) {
                document.getElementById('sdr-device').textContent = data.sdr;
            }
            if (data.cpu !== undefined) {
                document.getElementById('cpu-usage').textContent = data.cpu.toFixed(1) + '%';
            }
        }

        function updateCurrentStats(data) {
            // Обновляем параметры управления если есть
            if (data.status && data.status.thresh_dbm !== undefined) {
                document.getElementById('thresh_dbm').value = data.status.thresh_dbm;
            }
            if (data.status && data.status.target_signal_hz !== undefined) {
                document.getElementById('target_signal_hz').value = data.status.target_signal_hz;
                updateTargetDisplay(data.status.target_signal_hz);
            }
        }

        function renderSignalParams(data) {
            // Функция для отображения параметров сигнала в правой панели
            if (!data) {
                // Если данных нет - показываем прочерки
                document.getElementById('signal-frequency').textContent = '—';
                document.getElementById('signal-pos-phase').textContent = '—';
                document.getElementById('signal-neg-phase').textContent = '—';
                document.getElementById('signal-rise-us').textContent = '—';
                document.getElementById('signal-fall-us').textContent = '—';
                document.getElementById('signal-power').textContent = '—';
                document.getElementById('signal-power-rise').textContent = '—';
                document.getElementById('signal-baud').textContent = '—';
                document.getElementById('signal-asymmetry').textContent = '—';
                document.getElementById('signal-preamble').textContent = '—';
                document.getElementById('signal-duration').textContent = '—';
                document.getElementById('signal-period').textContent = '—';
                return;
            }

            // 1. Frequency, kHz (406000 + offset/1000)
            if (data.freq_hz !== undefined) {
                const freqKhz = (406000 + data.freq_hz / 1000);
                document.getElementById('signal-frequency').textContent = freqKhz.toFixed(3);
            } else if (data.frequency_offset_hz !== undefined) {
                const freqKhz = (406000 + data.frequency_offset_hz / 1000);
                document.getElementById('signal-frequency').textContent = freqKhz.toFixed(3);
            } else {
                document.getElementById('signal-frequency').textContent = '406.037';
            }

            // 2. +Phase deviation, rad
            if (data.pos_phase !== undefined) {
                document.getElementById('signal-pos-phase').textContent = data.pos_phase.toFixed(2);
            } else {
                document.getElementById('signal-pos-phase').textContent = '—';
            }

            // 3. −Phase deviation, rad
            if (data.neg_phase !== undefined) {
                document.getElementById('signal-neg-phase').textContent = data.neg_phase.toFixed(2);
            } else {
                document.getElementById('signal-neg-phase').textContent = '—';
            }

            // 4. Phase time rise, µs
            if (data.rise_us !== undefined) {
                document.getElementById('signal-rise-us').textContent = data.rise_us.toFixed(1);
            } else {
                document.getElementById('signal-rise-us').textContent = '—';
            }

            // 5. Phase time fall, µs
            if (data.fall_us !== undefined) {
                document.getElementById('signal-fall-us').textContent = data.fall_us.toFixed(1);
            } else {
                document.getElementById('signal-fall-us').textContent = '—';
            }

            // 6. Power, Wt (если есть данные о мощности)
            if (data.power_wt !== undefined) {
                document.getElementById('signal-power').textContent = data.power_wt.toFixed(2);
            } else if (data.p_wt !== undefined) {
                document.getElementById('signal-power').textContent = data.p_wt.toFixed(2);
            } else {
                document.getElementById('signal-power').textContent = '—';
            }

            // 7. Power rise, ms (из rise_us -> ms)
            if (data.rise_us !== undefined) {
                const powerRiseMs = data.rise_us / 1000;
                document.getElementById('signal-power-rise').textContent = powerRiseMs.toFixed(2);
            } else if (data.prise_ms !== undefined) {
                document.getElementById('signal-power-rise').textContent = data.prise_ms.toFixed(2);
            } else {
                document.getElementById('signal-power-rise').textContent = '—';
            }

            // 8. Bit Rate, bps
            if (data.baud !== undefined) {
                document.getElementById('signal-baud').textContent = data.baud.toFixed(0);
            } else if (data.t_mod !== undefined && data.t_mod > 0) {
                const baud = 1000000 / data.t_mod; // из микросекунд в биты/сек
                document.getElementById('signal-baud').textContent = baud.toFixed(0);
            } else {
                document.getElementById('signal-baud').textContent = '—';
            }

            // 9. Asymmetry, %
            if (data.asymmetry_pct !== undefined) {
                document.getElementById('signal-asymmetry').textContent = data.asymmetry_pct.toFixed(1);
            } else {
                document.getElementById('signal-asymmetry').textContent = '—';
            }

            // 10. CW Preamble, ms
            if (data.preamble_ms !== undefined) {
                document.getElementById('signal-preamble').textContent = data.preamble_ms.toFixed(1);
            } else {
                document.getElementById('signal-preamble').textContent = '—';
            }

            // 11. Total burst duration, ms
            if (data.length_ms !== undefined) {
                document.getElementById('signal-duration').textContent = data.length_ms.toFixed(1);
            } else {
                document.getElementById('signal-duration').textContent = '—';
            }

            // 12. Repetition period, s (если доступно)
            if (data.period_s !== undefined) {
                document.getElementById('signal-period').textContent = data.period_s.toFixed(1);
            } else {
                document.getElementById('signal-period').textContent = '—';
            }
        }

        function updateDisplay(data) {
            // Обновляем основную информацию
            if (data.protocol) {
                document.getElementById('protocol').textContent = data.protocol || 'C/S-406';
            }
            if (data.date) {
                document.getElementById('date').textContent = data.date || '2025-09-29';
            }
            if (data.beacon_model) {
                document.getElementById('beaconModel').textContent = data.beacon_model || 'EPIRB';
            }
            if (data.beacon_frequency) {
                document.getElementById('beaconFreq').textContent = data.beacon_frequency.toFixed(3);
            }
            if (data.hex) {
                const shortHex = data.hex.length > 32 ? data.hex.substring(0, 32) + '...' : data.hex;
                document.getElementById('message').textContent = shortHex;
            }

            // Обновляем phase values
            if (data.pos_phase !== undefined) {
                document.getElementById('phasePlus').textContent = (data.pos_phase * 180 / Math.PI).toFixed(1);
            }
            if (data.neg_phase !== undefined) {
                document.getElementById('phaseMinus').textContent = (data.neg_phase * 180 / Math.PI).toFixed(1);
            }
            if (data.rise_us !== undefined) {
                document.getElementById('tRise').textContent = data.rise_us.toFixed(1);
            }
            if (data.fall_us !== undefined) {
                document.getElementById('tFall').textContent = data.fall_us.toFixed(1);
            }

            // Рисуем график в зависимости от текущего режима
            drawChart(data);
        }

        function logEvent(type, message) {
            const timestamp = new Date().toLocaleTimeString();
            const eventHistory = document.getElementById('eventHistory');

            const entry = document.createElement('div');
            entry.className = `history-entry ${type}`;
            entry.innerHTML = `<small>[${timestamp}]</small> ${message}`;

            eventHistory.appendChild(entry);
            eventHistory.scrollTop = eventHistory.scrollHeight;

            // Ограничиваем количество записей
            while (eventHistory.children.length > 50) {
                eventHistory.removeChild(eventHistory.firstChild);
            }
        }

        function updateEventHistory(logEntries) {
            const eventHistory = document.getElementById('eventHistory');
            eventHistory.innerHTML = '';

            logEntries.slice(-20).forEach(entry => {
                const div = document.createElement('div');
                div.className = `history-entry ${entry.type}`;
                const timestamp = new Date(entry.timestamp * 1000).toLocaleTimeString();
                div.innerHTML = `<small>[${timestamp}]</small> ${entry.message}`;
                eventHistory.appendChild(div);
            });

            eventHistory.scrollTop = eventHistory.scrollHeight;
        }

        // === УПРАВЛЕНИЕ ВИДАМИ ===
        function changeView(viewType) {
            currentView = viewType;
            console.log('View changed to:', viewType);

            const titleEl = document.getElementById('chartTitle');

            // Обновляем заголовок
            switch(viewType) {
                case 'phase':
                    titleEl.textContent = 'Fig.8 Phase';
                    break;
                case 'inburst_fr':
                    titleEl.textContent = 'Fig.10 Inburst Frequency';
                    break;
                case 'fr_stability':
                    titleEl.textContent = 'Frequency Stability';
                    break;
                case 'ph_rise_fall':
                    titleEl.textContent = 'Phase Rise/Fall';
                    break;
                case 'message':
                    titleEl.textContent = 'Message Decode';
                    break;
                case 'sum_table':
                    titleEl.textContent = 'Summary Table';
                    break;
                case '121_data':
                    titleEl.textContent = '121.5 MHz Data';
                    break;
            }

            // МГНОВЕННО рендерим пустой шаблон
            renderEmptyTemplate(viewType);

            // Затем асинхронно загружаем данные
            loadDataForView(viewType);
        }

        // === СПЕЦИАЛИЗИРОВАННЫЕ ЗАГРУЗЧИКИ ===
        function loadDataForView(viewType, forceLoad = false) {
            console.log('Loading data for view:', viewType, 'forceLoad:', forceLoad);

            // Проверяем настройку "Show last data on view switch"
            if (!forceLoad && !showLastDataOnSwitch) {
                console.log('Skipping data load - showLastDataOnSwitch is disabled');
                return; // Оставляем пустой шаблон
            }

            if (viewType === 'phase' || viewType === 'inburst_fr') {
                // Графики фазы и частоты - загружаем из pulse данных
                loadGraphData(viewType);
            } else if (viewType === 'message' || viewType === 'sum_table') {
                // Сообщения и таблицы - загружаем общие pulse данные
                loadLastPulse();
            } else if (viewType === '121_data') {
                // Данные 121 МГц - специальный эндпоинт (пока заглушка)
                load121Data();
            } else {
                // Остальные режимы - общая загрузка
                loadLastPulse();
            }
        }

        function loadGraphData(viewType) {
            // Для графиков используем текущий time scale
            const span_ms = currentTimeScale * 10; // currentTimeScale это ms/div

            fetch('/api/last_pulse')
                .then(response => response.json())
                .then(data => {
                    if (data) {
                        // Фильтруем данные по типу графика
                        if (viewType === 'phase' && data.phase_xs_ms && data.phase_ys_rad) {
                            drawPhaseView(data);
                        } else if (viewType === 'inburst_fr' && data.fr_xs_ms && data.fr_ys_hz) {
                            drawInburstFRView(data);
                        } else {
                            // Если данных нет, оставляем пустой шаблон
                            console.log('No data available for', viewType);
                        }
                    }
                })
                .catch(error => {
                    console.error('Error loading graph data:', error);
                });
        }

        function load121Data() {
            // Заглушка для данных 121 МГц
            console.log('121 MHz data not implemented yet');
            // Оставляем пустой шаблон
        }

        function onShowLastDataToggle() {
            const checkbox = document.getElementById('showLastDataOnSwitch');
            showLastDataOnSwitch = checkbox.checked;
            console.log('Show last data on switch:', showLastDataOnSwitch);
        }

        function onTimeScaleChange() {
            const select = document.getElementById('timeScale');
            currentTimeScale = parseInt(select.value);
            console.log('Time scale changed to:', currentTimeScale + '%');

            // Для графиков фазы и частоты перезагружаем с новым масштабом
            if (currentView === 'phase' || currentView === 'inburst_fr') {
                loadDataForView(currentView);
            } else {
                // Для остальных режимов используем общую загрузку
                loadLastPulse();
            }
        }

        // === РИСОВАНИЕ ГРАФИКОВ ===
        function resizeCanvas() {
            if (canvas) {
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
            }
        }

        function drawChart(data) {
            if (!canvas || !ctx) return;

            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            if (currentView === 'message') {
                drawMessageView(data);
            } else if (currentView === 'sum_table') {
                drawSummaryView(data);
            } else if (currentView === '121_data') {
                draw121View(data);
            } else if (currentView === 'phase') {
                drawPhaseView(data);
            } else if (currentView === 'inburst_fr') {
                drawInburstFRView(data);
            } else {
                // Для остальных режимов используем базовый вид фазы
                drawPhaseView(data);
            }
        }

        function drawPhaseView(data) {
            const width = canvas.width;
            const height = canvas.height;

            // Сетка
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии (10 делений)
            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии (8 делений для времени)
            ctx.fillStyle = '#6c757d';
            ctx.font = '10px Arial';
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();

                // Временные метки
                if (data && data.phase_xs_ms && data.phase_xs_ms.length > 0) {
                    const timeRange = data.phase_xs_ms[data.phase_xs_ms.length - 1] - data.phase_xs_ms[0];
                    const timeMs = (data.phase_xs_ms[0] + i * timeRange / 8).toFixed(1);
                    ctx.fillText(timeMs + ' ms', x - 15, height - 5);
                }
            }

            // Нулевая линия
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();

            // Пунктирные линии на ±1.1 радиан
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);

            const phaseScale = 1.25; // радиан
            const y_plus_1_1 = height / 2 - (1.1 / phaseScale) * (height / 2);
            ctx.beginPath();
            ctx.moveTo(0, y_plus_1_1);
            ctx.lineTo(width, y_plus_1_1);
            ctx.stroke();

            const y_minus_1_1 = height / 2 + (1.1 / phaseScale) * (height / 2);
            ctx.beginPath();
            ctx.moveTo(0, y_minus_1_1);
            ctx.lineTo(width, y_minus_1_1);
            ctx.stroke();

            ctx.setLineDash([]);

            // Y-axis метки
            ctx.fillStyle = '#6c757d';
            ctx.font = '10px Arial';
            ctx.fillText('+' + phaseScale.toFixed(2) + ' rad', 5, 15);
            ctx.fillText('0', 5, height / 2 + 4);
            ctx.fillText('-' + phaseScale.toFixed(2) + ' rad', 5, height - 10);

            // Рисуем данные фазы если есть
            if (data && data.phase_xs_ms && data.phase_ys_rad &&
                data.phase_xs_ms.length > 0 && data.phase_ys_rad.length > 0) {

                ctx.strokeStyle = '#0066FF';
                ctx.lineWidth = 2;
                ctx.beginPath();

                const xsMs = data.phase_xs_ms;
                const ysRad = data.phase_ys_rad;
                const xMin = xsMs[0];
                const xMax = xsMs[xsMs.length - 1];
                const xRange = xMax - xMin || 1;

                for (let i = 0; i < xsMs.length; i++) {
                    const x = ((xsMs[i] - xMin) / xRange) * width;
                    const y = height / 2 - (ysRad[i] / phaseScale) * (height / 2);

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();

                // Рисуем маркеры битов если есть
                if (data.markers_ms && data.markers_ms.length > 0) {
                    ctx.strokeStyle = '#00AA00';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([2, 2]);

                    for (const markerMs of data.markers_ms) {
                        if (markerMs >= xMin && markerMs <= xMax) {
                            const x = ((markerMs - xMin) / xRange) * width;
                            ctx.beginPath();
                            ctx.moveTo(x, 0);
                            ctx.lineTo(x, height);
                            ctx.stroke();
                        }
                    }
                    ctx.setLineDash([]);
                }

                // Подсветка преамбулы если есть
                if (data.preamble_ms && data.preamble_ms.length === 2) {
                    const [t0, t1] = data.preamble_ms;
                    if (t0 >= xMin && t1 <= xMax) {
                        const x0 = ((t0 - xMin) / xRange) * width;
                        const x1 = ((t1 - xMin) / xRange) * width;

                        ctx.fillStyle = 'rgba(255, 255, 0, 0.1)';
                        ctx.fillRect(x0, 0, x1 - x0, height);
                    }
                }

            } else {
                // Заглушка
                ctx.fillStyle = '#6c757d';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('No phase data available', width / 2, height / 2);
                ctx.fillText('Waiting for pulse detection...', width / 2, height / 2 + 25);
                ctx.textAlign = 'left';
            }
        }

        function drawInburstFRView(data) {
            const width = canvas.width;
            const height = canvas.height;

            // Сетка
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии (10 делений)
            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии (8 делений для времени)
            ctx.fillStyle = '#6c757d';
            ctx.font = '10px Arial';
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();

                // Временные метки
                if (data && data.fr_xs_ms && data.fr_xs_ms.length > 0) {
                    const timeRange = data.fr_xs_ms[data.fr_xs_ms.length - 1] - data.fr_xs_ms[0];
                    const timeMs = (data.fr_xs_ms[0] + i * timeRange / 8).toFixed(1);
                    ctx.fillText(timeMs + ' ms', x - 15, height - 5);
                }
            }

            // Центральная линия
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();

            // Y-axis метки для частоты
            ctx.fillStyle = '#6c757d';
            ctx.font = '10px Arial';

            // Рисуем данные частоты если есть
            if (data && data.fr_xs_ms && data.fr_ys_hz &&
                data.fr_xs_ms.length > 0 && data.fr_ys_hz.length > 0) {

                // Находим диапазон частот
                const ysHz = data.fr_ys_hz;
                const minFreq = Math.min(...ysHz);
                const maxFreq = Math.max(...ysHz);
                const freqRange = maxFreq - minFreq || 1000;
                const centerFreq = (minFreq + maxFreq) / 2;

                // Y-axis метки
                ctx.fillText('+' + (freqRange / 2).toFixed(0) + ' Hz', 5, 15);
                ctx.fillText(centerFreq.toFixed(0) + ' Hz', 5, height / 2 + 4);
                ctx.fillText('-' + (freqRange / 2).toFixed(0) + ' Hz', 5, height - 10);

                ctx.strokeStyle = '#FF6600';
                ctx.lineWidth = 2;
                ctx.beginPath();

                const xsMs = data.fr_xs_ms;
                const xMin = xsMs[0];
                const xMax = xsMs[xsMs.length - 1];
                const xRange = xMax - xMin || 1;

                for (let i = 0; i < xsMs.length; i++) {
                    const x = ((xsMs[i] - xMin) / xRange) * width;
                    const y = height / 2 - ((ysHz[i] - centerFreq) / (freqRange / 2)) * (height / 2);

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();

                // Рисуем маркеры битов если есть
                if (data.markers_ms && data.markers_ms.length > 0) {
                    ctx.strokeStyle = '#00AA00';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([2, 2]);

                    for (const markerMs of data.markers_ms) {
                        if (markerMs >= xMin && markerMs <= xMax) {
                            const x = ((markerMs - xMin) / xRange) * width;
                            ctx.beginPath();
                            ctx.moveTo(x, 0);
                            ctx.lineTo(x, height);
                            ctx.stroke();
                        }
                    }
                    ctx.setLineDash([]);
                }

                // Подсветка преамбулы если есть
                if (data.preamble_ms && data.preamble_ms.length === 2) {
                    const [t0, t1] = data.preamble_ms;
                    if (t0 >= xMin && t1 <= xMax) {
                        const x0 = ((t0 - xMin) / xRange) * width;
                        const x1 = ((t1 - xMin) / xRange) * width;

                        ctx.fillStyle = 'rgba(255, 255, 0, 0.1)';
                        ctx.fillRect(x0, 0, x1 - x0, height);
                    }
                }

                // Показываем битрейт если есть
                if (data.baud) {
                    ctx.fillStyle = '#333';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'right';
                    ctx.fillText('BitRate: ' + data.baud.toFixed(2) + ' bps', width - 10, 20);
                    ctx.textAlign = 'left';
                }

            } else {
                // Заглушка
                ctx.fillStyle = '#6c757d';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('No frequency data available', width / 2, height / 2);
                ctx.fillText('Waiting for pulse detection...', width / 2, height / 2 + 25);
                ctx.textAlign = 'left';
            }
        }

        function drawMessageView(data) {
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#333';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Message Decode', canvas.width / 2, 30);

            if (data && data.hex) {
                ctx.font = '12px monospace';
                ctx.textAlign = 'left';
                ctx.fillText(`HEX: ${data.hex}`, 20, 60);

                ctx.fillText('Protocol: COSPAS-SARSAT', 20, 80);
                ctx.fillText('Type: Emergency Beacon', 20, 100);
                if (data.ok) {
                    ctx.fillText('Status: Decoded successfully', 20, 120);
                } else {
                    ctx.fillText('Status: Decode failed', 20, 120);
                }
            } else {
                ctx.textAlign = 'center';
                ctx.fillText('No message data available', canvas.width / 2, canvas.height / 2);
            }
        }

        function drawSummaryView(data) {
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#333';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Summary Table', canvas.width / 2, 30);

            ctx.font = '12px Arial';
            ctx.textAlign = 'left';

            let y = 60;
            const lineHeight = 20;

            if (data) {
                if (data.baud) {
                    ctx.fillText(`Baud Rate: ${data.baud.toFixed(1)} bps`, 20, y);
                    y += lineHeight;
                }
                if (data.pos_phase !== undefined) {
                    ctx.fillText(`Phase+: ${(data.pos_phase * 180 / Math.PI).toFixed(2)}°`, 20, y);
                    y += lineHeight;
                }
                if (data.neg_phase !== undefined) {
                    ctx.fillText(`Phase-: ${(data.neg_phase * 180 / Math.PI).toFixed(2)}°`, 20, y);
                    y += lineHeight;
                }
                if (data.rise_us !== undefined) {
                    ctx.fillText(`Rise time: ${data.rise_us.toFixed(1)} μs`, 20, y);
                    y += lineHeight;
                }
                if (data.fall_us !== undefined) {
                    ctx.fillText(`Fall time: ${data.fall_us.toFixed(1)} μs`, 20, y);
                    y += lineHeight;
                }
                if (data.asymmetry_pct !== undefined) {
                    ctx.fillText(`Asymmetry: ${data.asymmetry_pct.toFixed(1)}%`, 20, y);
                    y += lineHeight;
                }
            } else {
                ctx.textAlign = 'center';
                ctx.fillText('No summary data available', canvas.width / 2, canvas.height / 2);
            }
        }

        function draw121View(data) {
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#333';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('121.5 MHz Data', canvas.width / 2, 30);

            ctx.font = '12px Arial';
            ctx.fillText('121.5 MHz band analysis not available in DSP mode', canvas.width / 2, canvas.height / 2);
            ctx.fillText('This view requires dedicated 121.5 MHz hardware', canvas.width / 2, canvas.height / 2 + 20);
        }

        // === ПУСТЫЕ ШАБЛОНЫ ===
        function renderEmptyTemplate(viewType) {
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            if (viewType === 'phase' || viewType === 'inburst_fr' ||
                viewType === 'fr_stability' || viewType === 'ph_rise_fall') {
                renderEmptyGraphTemplate(viewType);
            } else if (viewType === 'message' || viewType === 'sum_table' || viewType === '121_data') {
                renderEmptyTableTemplate(viewType);
            }
        }

        function renderEmptyGraphTemplate(viewType) {
            const width = canvas.width;
            const height = canvas.height;

            // Сетка
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии
            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }

            // Центральная линия для графиков
            if (viewType === 'phase' || viewType === 'inburst_fr') {
                ctx.strokeStyle = '#adb5bd';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                ctx.lineTo(width, height / 2);
                ctx.stroke();
            }

            // Специальные линии для фазы
            if (viewType === 'phase') {
                ctx.strokeStyle = '#FF0000';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);

                const phaseScale = 1.25;
                const y_plus_1_1 = height / 2 - (1.1 / phaseScale) * (height / 2);
                ctx.beginPath();
                ctx.moveTo(0, y_plus_1_1);
                ctx.lineTo(width, y_plus_1_1);
                ctx.stroke();

                const y_minus_1_1 = height / 2 + (1.1 / phaseScale) * (height / 2);
                ctx.beginPath();
                ctx.moveTo(0, y_minus_1_1);
                ctx.lineTo(width, y_minus_1_1);
                ctx.stroke();

                ctx.setLineDash([]);

                // Y-axis метки для фазы
                ctx.fillStyle = '#6c757d';
                ctx.font = '10px Arial';
                ctx.fillText('+' + phaseScale.toFixed(2) + ' rad', 5, 15);
                ctx.fillText('0', 5, height / 2 + 4);
                ctx.fillText('-' + phaseScale.toFixed(2) + ' rad', 5, height - 10);
            }

            // Сообщение ожидания
            ctx.fillStyle = '#6c757d';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data...', width / 2, height / 2 - 30);
            ctx.font = '12px Arial';
            ctx.fillText('Switch to acquisition mode to see live data', width / 2, height / 2 - 10);
            ctx.textAlign = 'left';
        }

        function renderEmptyTableTemplate(viewType) {
            const width = canvas.width;
            const height = canvas.height;

            // Фон
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, width, height);

            // Заголовок
            ctx.fillStyle = '#333';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';

            let title = '';
            switch(viewType) {
                case 'message': title = 'Message Decode'; break;
                case 'sum_table': title = 'Summary Table'; break;
                case '121_data': title = '121.5 MHz Data'; break;
            }
            ctx.fillText(title, width / 2, 30);

            // Рамка для таблицы
            ctx.strokeStyle = '#dee2e6';
            ctx.lineWidth = 1;
            ctx.strokeRect(20, 50, width - 40, height - 100);

            // Заголовок таблицы
            ctx.fillStyle = '#e9ecef';
            ctx.fillRect(20, 50, width - 40, 30);

            ctx.fillStyle = '#495057';
            ctx.font = '12px Arial';
            ctx.textAlign = 'left';

            if (viewType === 'message') {
                ctx.fillText('Field', 30, 70);
                ctx.fillText('Value', width / 2, 70);
            } else if (viewType === 'sum_table') {
                ctx.fillText('Parameter', 30, 70);
                ctx.fillText('Value', width / 2, 70);
                ctx.fillText('Units', width - 100, 70);
            } else if (viewType === '121_data') {
                ctx.fillText('121.5 MHz Parameter', 30, 70);
                ctx.fillText('Status', width - 150, 70);
            }

            // Сообщение ожидания
            ctx.fillStyle = '#6c757d';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data...', width / 2, height / 2 + 20);
            ctx.font = '12px Arial';
            ctx.fillText('Data will appear here when available', width / 2, height / 2 + 40);
            ctx.textAlign = 'left';
        }

        // === ИНИЦИАЛИЗАЦИЯ ===
        function initialize() {
            console.log('Initializing DSP2Web Full UI client...');

            resizeCanvas();
            startEventStream();
            refreshState();
            loadLastPulse();
            checkHealth();

            // Периодические обновления
            setInterval(checkHealth, 5000);
            setInterval(refreshState, 10000);

            console.log('DSP2Web Full UI client initialized');
        }

        // Обработчики событий
        window.addEventListener('resize', resizeCanvas);
        document.addEventListener('DOMContentLoaded', initialize);

        // Закрытие модального окна при клике вне его
        window.addEventListener('click', function(event) {
            const modal = document.getElementById('sdrModal');
            if (event.target === modal) {
                closeSDRConfig();
            }
        });
    </script>

    <!-- SDR Config Modal -->
    <div id="sdrModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>SDR Configuration</h2>
                <span class="close" onclick="closeSDRConfig()">&times;</span>
            </div>

            <div class="modal-tabs">
                <button class="tab-button active" onclick="switchTab('radio')">Radio</button>
                <button class="tab-button" onclick="switchTab('recording')">Recording</button>
            </div>

            <!-- Radio Tab -->
            <div id="radioTab" class="tab-content active">
                <div class="config-grid">
                    <div class="config-row">
                        <label>Center frequency (Hz):</label>
                        <input type="text" id="center_freq_hz" placeholder="406.000M">
                    </div>
                    <div class="config-row">
                        <label>Sample rate (S/s):</label>
                        <input type="text" id="sample_rate_sps" placeholder="1M">
                    </div>
                    <div class="config-row">
                        <label>Baseband shift (Hz):</label>
                        <input type="text" id="bb_shift_hz" placeholder="-37k">
                        <label style="margin-left: 10px;">
                            <input type="checkbox" id="bb_shift_enable"> Enable
                        </label>
                    </div>
                    <div class="config-row">
                        <label>Frequency correction (Hz):</label>
                        <input type="text" id="freq_corr_hz" placeholder="0">
                    </div>
                    <div class="config-row">
                        <label>
                            <input type="checkbox" id="agc"> AGC
                        </label>
                    </div>
                    <div class="config-row">
                        <label>Manual gain (dB):</label>
                        <input type="number" id="gain_db" step="0.5" min="0" max="60">
                    </div>
                    <div class="config-row">
                        <label>
                            <input type="checkbox" id="bias_t"> Bias-T
                        </label>
                    </div>
                    <div class="config-row">
                        <label>Antenna/Port:</label>
                        <select id="antenna">
                            <option value="RX">RX</option>
                            <option value="TX">TX</option>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>Device/Driver:</label>
                        <input type="text" id="device" readonly>
                    </div>
                </div>
            </div>

            <!-- Recording Tab -->
            <div id="recordingTab" class="tab-content">
                <div class="config-grid">
                    <div class="config-row">
                        <label>
                            <input type="checkbox" id="sigmf_save"> SigMF save
                        </label>
                    </div>
                    <div class="config-row">
                        <label>Path / Filename pattern:</label>
                        <input type="text" id="sigmf_path" placeholder="./captures/recording_YYYYMMDD_HHMMSS.sigmf-data">
                    </div>
                    <div class="config-row">
                        <label>Datatype:</label>
                        <select id="sigmf_datatype">
                            <option value="cf32">cf32 (Complex Float32)</option>
                            <option value="ci16">ci16 (Complex Int16)</option>
                        </select>
                    </div>
                    <div class="config-row">
                        <label>Chunk size (MB):</label>
                        <input type="number" id="sigmf_chunk_mb" value="100" min="1" max="1000">
                    </div>
                </div>
            </div>

            <div class="modal-footer">
                <button class="button preset-btn" onclick="loadPreset()">Load Preset</button>
                <button class="button preset-btn" onclick="savePreset()">Save Preset</button>
                <button class="button" onclick="resetDefaults()">Defaults</button>
                <button class="button primary" onclick="applySDRConfig()">Apply</button>
                <button class="button" onclick="closeSDRConfig()">Cancel</button>
            </div>
        </div>
    </div>
</body>
</html>
'''

def main():
    """Основная функция"""
    global pub_addr, rep_addr

    # Парсинг аргументов и получение адресов
    web_host, web_port = get_zmq_addresses()

    log.info("=" * 60)
    log.info("COSPAS/SARSAT Beacon Tester DSP2Web Full UI Client v2.1")
    log.info("=" * 60)
    log.info(f"ZeroMQ PUB address: {pub_addr}")
    log.info(f"ZeroMQ REP address: {rep_addr}")
    log.info(f"Web interface: http://{web_host}:{web_port}/")
    log.info("=" * 60)

    # Инициализация ZeroMQ
    init_zmq()

    # Запуск SUB потока
    sub_thread_obj = threading.Thread(target=sub_thread, daemon=True)
    sub_thread_obj.start()

    # Запуск Flask приложения
    try:
        # Используем waitress для production
        try:
            from waitress import serve
            log.info("Using waitress WSGI server")
            serve(app, host=web_host, port=web_port, threads=4)
        except ImportError:
            log.info("waitress not available, using Flask dev server")
            app.run(host=web_host, port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        # Очистка ресурсов
        if zmq_context:
            zmq_context.term()

if __name__ == '__main__':
    main()