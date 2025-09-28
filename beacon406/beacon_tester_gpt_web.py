#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin Web for Beacon Tester — ZMQ glue to DSP
--------------------------------------------
Эта версия веб-сервера НЕ трогает SDR и не делает DSP.
Она только:
  • подписывается на события от headless DSP-сервиса (ZeroMQ SUB)
  • отдаёт состояние через REST (/api/state)
  • пересылает команды в DSP (ZeroMQ REQ) через /api/control/*

Запуск:
  1) Установить зависимости:    pip install flask pyzmq waitress
  2) Запустить DSP сервис:      python beacon_dsp_service.py --pub tcp://127.0.0.1:8781 --rep tcp://127.0.0.1:8782
  3) Запустить веб:             waitress-serve --listen=127.0.0.1:8738 beacon_tester_web:app

Переменные окружения (опционально):
  DSP_PUB = tcp://127.0.0.1:8781
  DSP_REP = tcp://127.0.0.1:8782
"""
from __future__ import annotations
import os
import time
import json
import threading
from typing import Any, Dict

from flask import Flask, jsonify, request

try:
    import zmq
except Exception:  # pragma: no cover
    zmq = None

# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------
ZMQ_PUB_URL = os.environ.get("DSP_PUB", "tcp://127.0.0.1:8781")
ZMQ_REP_URL = os.environ.get("DSP_REP", "tcp://127.0.0.1:8782")

# Хранилище состояния, обновляется слушателем DSP
DSP_STATE: Dict[str, Any] = {
    "status": {},   # последний статус
    "pulse": None,  # последний импульс
    "psk":   None,  # последний PSK-результат
    "log":   [],    # несколько последних событий
}
DSP_LOG_MAX = 128
_STATE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# ZMQ listener (SUB): принимает строки JSON {type: status|pulse|psk, ...}
# ---------------------------------------------------------------------------
_listener_started = False
_req_sock = None


def _start_dsp_listener_once():
    global _listener_started
    if _listener_started:
        return
    _listener_started = True

    if zmq is None:
        print("[WEB] pyzmq не установлен — IPC отключён")
        return

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(ZMQ_PUB_URL)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    def _loop():
        print(f"[WEB] SUB listening at {ZMQ_PUB_URL}")
        while True:
            try:
                line = sub.recv_string()
                evt = json.loads(line)
                typ = evt.get("type")
                if typ in ("status", "pulse", "psk"):
                    with _STATE_LOCK:
                        DSP_STATE[typ] = evt
                        DSP_STATE["log"].append(evt)
                        if len(DSP_STATE["log"]) > DSP_LOG_MAX:
                            DSP_STATE["log"] = DSP_STATE["log"][-DSP_LOG_MAX:]
            except Exception as e:  # pragma: no cover
                print("[WEB] DSP listener error:", e)
                time.sleep(0.1)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()


# ---------------------------------------------------------------------------
# ZMQ request helper (REQ): отправляет команды в DSP REP сокет
# ---------------------------------------------------------------------------

def dsp_send(cmd: Dict[str, Any], timeout_ms: int = 2000) -> Dict[str, Any]:
    global _req_sock
    if zmq is None:
        return {"ok": False, "err": "pyzmq-missing"}
    if _req_sock is None:
        ctx = zmq.Context.instance()
        _req_sock = ctx.socket(zmq.REQ)
        _req_sock.connect(ZMQ_REP_URL)
        _req_sock.RCVTIMEO = timeout_ms
        _req_sock.SNDTIMEO = timeout_ms
        print(f"[WEB] REQ connected to {ZMQ_REP_URL}")
    try:
        _req_sock.send_json(cmd)
        return _req_sock.recv_json()
    except Exception as e:  # pragma: no cover
        return {"ok": False, "err": str(e)}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/api/state", methods=["GET"])
def api_state():
    with _STATE_LOCK:
        st  = DSP_STATE.get("status") or {}
        pls = DSP_STATE.get("pulse")
        psk = DSP_STATE.get("psk")
        log = DSP_STATE.get("log", [])
        # не раздуваем ответ
        tail = log[-64:]
    return jsonify({
        "status": st,
        "last_pulse": pls,
        "last_psk": psk,
        "log": tail,
        "pub": ZMQ_PUB_URL,
        "rep": ZMQ_REP_URL,
    })


@app.route('/api/control/start', methods=['POST'])
def api_start():
    return jsonify(dsp_send({"cmd": "start_acquire"}))


@app.route('/api/control/stop', methods=['POST'])
def api_stop():
    return jsonify(dsp_send({"cmd": "stop_acquire"}))


@app.route('/api/control/params', methods=['POST'])
def api_params():
    data = request.get_json(force=True) or {}
    payload = {"cmd": "set_params"}
    payload.update(data)
    return jsonify(dsp_send(payload))


@app.route('/api/control/save_sigmf', methods=['POST'])
def api_save_sigmf():
    return jsonify(dsp_send({"cmd": "save_sigmf"}))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
_start_dsp_listener_once()


if __name__ == '__main__':
    # в проде — waitress; если её нет, используем встроенный сервер без дебага
    try:
        from waitress import serve
        print("[WEB] starting waitress on 127.0.0.1:8738")
        serve(app, listen='127.0.0.1:8738')
    except Exception:
        print("[WEB] starting Flask dev server (no debug, no reloader)")
        app.run(host='127.0.0.1', port=8738, debug=False, use_reloader=False, threaded=False)
