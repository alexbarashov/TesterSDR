"""
COSPAS/SARSAT Beacon Tester - Version 2.0
Веб-интерфейс, читает данные из pulses.json от beacon_dsp_service.py
"""

import os
import json
import time
from flask import Flask, jsonify
from lib.logger import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)
app = Flask(__name__)

# Путь к JSON от DSP-сервиса
PULSES_JSON = "pulses.json"

# Состояние для веб
class State:
    def __init__(self):
        self.running = False
        self.last_pulse = None
        self.pulse_history = []

STATE = State()

@app.route('/')
def index():
    return jsonify({'status': 'Beacon Tester Web Interface', 'version': '2.0'})

@app.route('/api/status')
def api_status():
    # Читаем pulses.json
    pulses = []
    try:
        if os.path.exists(PULSES_JSON):
            with open(PULSES_JSON, 'r') as f:
                for line in f:
                    try:
                        pulse = json.loads(line.strip())
                        pulses.append(pulse)
                    except json.JSONDecodeError:
                        continue
            STATE.pulse_history = pulses[-20:]  # Последние 20 импульсов
            STATE.last_pulse = pulses[-1] if pulses else None
    except Exception as e:
        log.error(f"Error reading pulses.json: {e}")

    return jsonify({
        'running': STATE.running,
        'last_pulse': STATE.last_pulse,
        'pulse_history': STATE.pulse_history,
        'sdr_device_info': "Managed by DSP service",
        'realtime_rms_dbm': STATE.last_pulse.get('rms_dbm', -100.0) if STATE.last_pulse else -100.0
    })

@app.route('/api/run', methods=['POST'])
def api_run():
    STATE.running = True
    return jsonify({'status': 'running', 'message': 'Reading from DSP service'})

@app.route('/api/break', methods=['POST'])
def api_break():
    STATE.running = False
    return jsonify({'status': 'stopped', 'message': 'Stopped reading'})

if __name__ == "__main__":
    log.info("Starting COSPAS/SARSAT Beacon Tester Web v2.0")
    log.info("Interface available at: http://127.0.0.1:8738/")
    log.info("Ensure beacon_dsp_service.py is running")
    app.run(host='127.0.0.1', port=8738, debug=True)
