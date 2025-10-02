#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beacon_dsp2plot.py — UI-клиент к beacon_dsp_service.py (ZeroMQ)
Обновлённая версия согласно ТЗ 251001-3:
- Мгновенный ACK для всех команд
- Single poll loop (без SUB событий)
- Status bar с backend/state/ready/fs/bb_shift/thresh
- Анти-дребезг кнопок
- Стабильный рендер в file режиме
"""
import os
import sys
import json
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')

# UI
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import filedialog
import tkinter as tk
import signal
import queue

# ZMQ
try:
    import zmq
except Exception:
    zmq = None

# -------------------- Параметры подключения к сервису --------------------
PUB_ADDR_DEFAULT = "tcp://127.0.0.1:8781"
REP_ADDR_DEFAULT = "tcp://127.0.0.1:8782"

# -------------------- Вспомогательные --------------------
EPS = 1e-20

def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, EPS))

# -------------------- DspServiceClient --------------------
class DspServiceClient:
    """Тонкая обёртка над ZeroMQ PUB/REP для beacon_dsp_service.py."""
    def __init__(self, pub_addr: str = PUB_ADDR_DEFAULT, rep_addr: str = REP_ADDR_DEFAULT):
        if zmq is None:
            raise RuntimeError("pyzmq не установлен. Установите пакет 'pyzmq'")
        self.ctx = zmq.Context.instance()
        self.pub_addr = pub_addr
        self.rep_addr = rep_addr
        self._req_lock = threading.Lock()
        self._req_socket_broken = False
        self._stop = False

        # SUB (для pulse/psk событий)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(self.pub_addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub_thread = None
        self._cb = None

        # REQ (will be created/recreated by _ensure_req_socket)
        self.req = None
        self._ensure_req_socket()

    def _ensure_req_socket(self):
        """Ensure REQ socket is connected and ready. Recreate if needed."""
        if self.req is not None and not self._req_socket_broken:
            return

        try:
            if self.req is not None:
                self.req.close(0)
        except Exception:
            pass

        self.req = self.ctx.socket(zmq.REQ)
        # Короткий таймаут для мгновенного ACK
        try:
            self.req.setsockopt(zmq.RCVTIMEO, 2000)  # 2s для ACK
            self.req.setsockopt(zmq.SNDTIMEO, 1000)  # 1s send
            self.req.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        self.req.connect(self.rep_addr)
        self._req_socket_broken = False

    # ---- commands (REQ/REP) с мгновенным ACK и логированием ----
    def send_cmd(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Отправляет команду и ждёт быстрый ACK.
        Возвращает {"ok": bool, "error"?: str, ...}
        Печатает: [cmd=<name>] ack=<dt> ms → {ok: ...}
        """
        msg = {"cmd": name, **kwargs}
        t0 = time.perf_counter()

        with self._req_lock:
            try:
                self._ensure_req_socket()
                self.req.send_json(msg)
                rep = self.req.recv_json()
                self._req_socket_broken = False

                dt_ms = (time.perf_counter() - t0) * 1000
                ok_str = "OK" if rep.get("ok", False) else "ERR"
                print(f"[cmd={name}] ack={dt_ms:.1f} ms → {ok_str}")

                return rep if isinstance(rep, dict) else {"ok": False, "error": "bad reply"}

            except zmq.error.Again:
                self._req_socket_broken = True
                dt_ms = (time.perf_counter() - t0) * 1000
                print(f"[cmd={name}] TIMEOUT after {dt_ms:.1f} ms")
                return {"ok": False, "error": "Timeout waiting for DSP service"}

            except Exception as e:
                self._req_socket_broken = True
                dt_ms = (time.perf_counter() - t0) * 1000
                print(f"[cmd={name}] ERROR after {dt_ms:.1f} ms: {e}")
                return {"ok": False, "error": f"Communication error: {e}"}

    # Команды согласно ТЗ
    def echo(self, **payload) -> Dict[str, Any]:
        return self.send_cmd("echo", payload=payload)

    def get_status(self) -> Dict[str, Any]:
        return self.send_cmd("get_status")

    def start_acquire(self) -> Dict[str, Any]:
        return self.send_cmd("start_acquire")

    def stop_acquire(self) -> Dict[str, Any]:
        return self.send_cmd("stop_acquire")

    def set_sdr_config(self, **cfg) -> Dict[str, Any]:
        return self.send_cmd("set_sdr_config", **cfg)

    def save_sigmf(self) -> Dict[str, Any]:
        return self.send_cmd("save_sigmf")

    def set_params(self, **params) -> Dict[str, Any]:
        return self.send_cmd("set_params", **params)

    # ---- SUB stream ----
    def subscribe_events(self, callback):
        """Подписка на pulse/psk события из PUB канала."""
        self._cb = callback
        if self._sub_thread and self._sub_thread.is_alive():
            return
        self._stop = False
        self._sub_thread = threading.Thread(target=self._sub_loop, daemon=True)
        self._sub_thread.start()

    def _sub_loop(self):
        """SUB loop для приёма pulse/psk событий."""
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        while not self._stop:
            socks = dict(poller.poll(200))
            if self.sub in socks and socks[self.sub] == zmq.POLLIN:
                try:
                    line = self.sub.recv_string(flags=zmq.NOBLOCK)
                    obj = json.loads(line)
                    if self._cb:
                        self._cb(obj)
                except Exception:
                    pass

    def close(self):
        self._stop = True
        try:
            if self._sub_thread:
                self._sub_thread.join(timeout=0.3)
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

# -------------------- UI (single poll loop) --------------------
class Dsp2PlotUI:
    def __init__(self, pub_addr: str, rep_addr: str, autostart: bool = False):
        # Состояние
        self.last_status: Dict[str, Any] = {}
        self.sample_rate: float = 1_000_000.0
        self.autostart = autostart

        # Анти-дребезг: время последней команды
        self._last_cmd_time: float = 0.0
        self._cmd_debounce_ms: float = 150.0

        # Данные для графиков
        self._lvl_x: np.ndarray = np.array([], dtype=np.float64)
        self._lvl_y: np.ndarray = np.array([], dtype=np.float64)
        self._pulse_rms_x: np.ndarray = np.array([], dtype=np.float64)
        self._pulse_rms_y: np.ndarray = np.array([], dtype=np.float64)
        self._phase_x: np.ndarray = np.array([], dtype=np.float64)
        self._phase_y: np.ndarray = np.array([], dtype=np.float64)
        self._fm_x: np.ndarray = np.array([], dtype=np.float64)
        self._fm_y: np.ndarray = np.array([], dtype=np.float64)

        # Данные для второго окна (кнопки Параметры/Сообщение/Спектр)
        self.pulse_updates_enabled: bool = True
        self.last_phase_metrics: Optional[Dict[str, Any]] = None
        self.last_msg_hex: Optional[str] = None
        self.last_iq_seg: Optional[np.ndarray] = None
        self.last_core_gate: Optional[tuple] = None

        # Очередь для потокобезопасной передачи pulse событий из SUB в главный поток
        self._pulse_q: queue.Queue = queue.Queue(maxsize=2)
        self._pulse_event_counter: int = 0  # Счётчик для диагностики

        # Клиент сервиса
        self.client = DspServiceClient(pub_addr=pub_addr, rep_addr=rep_addr)
        self._svc_proc = None

        # Подписка на pulse/psk события
        self.client.subscribe_events(self._on_pulse_event)

        # Логирование старта
        print(f"[startup] Connecting to DSP service")
        print(f"[startup] REP: {rep_addr}")
        print(f"[startup] PUB: {pub_addr}")

        # Инициализация: echo → get_status
        print("[startup] Testing connection with echo...")
        echo_rep = self.client.echo(test="hello")
        if not echo_rep.get("ok", False):
            print(f"[startup] WARNING: echo failed: {echo_rep}")

        print("[startup] Getting initial status...")
        st_rep = self.client.get_status()
        if st_rep.get("ok", False):
            self.last_status = st_rep.get("status", {})
            self.sample_rate = float(self.last_status.get("fs", self.sample_rate))
            print(f"[startup] Initial status: backend={self.last_status.get('backend', '?')} ready={self.last_status.get('ready', False)}")
        else:
            print(f"[startup] WARNING: get_status failed: {st_rep}")

        # ----- Figure 1: Sliding RMS -----
        self.fig1, self.ax_lvl = plt.subplots(num="Sliding RMS — realtime", figsize=(9, 2.5))
        self.fig1.subplots_adjust(bottom=0.35, top=0.85)

        (self.ln_lvl,) = self.ax_lvl.plot([], [], lw=1.2)
        self.ax_lvl.set_xlabel("Время, с")
        self.ax_lvl.set_ylabel("RMS, dBm")
        self.ax_lvl.grid(True, alpha=0.3)

        # Линия порога (пунктирная)
        (self.ln_thresh,) = self.ax_lvl.plot([], [], linestyle='--', color='red', alpha=0.7, lw=1.5)

        # Status bar (title)
        self._update_status_bar()

        # Кнопки backend
        buttons = [
            ("Auto", "auto", 0.02),
            ("RTL", "soapy_rtl", 0.11),
            ("HackRF", "soapy_hackrf", 0.20),
            ("Airspy", "soapy_airspy", 0.29),
            ("SDRPlay", "soapy_sdrplay", 0.38),
            ("RSA306", "rsa306", 0.47),
            ("File", "file", 0.56),
        ]
        self.backend_buttons = {}
        for label, backend, left in buttons:
            ax = self.fig1.add_axes([left, 0.01, 0.08, 0.06])
            btn = Button(ax, label)
            btn.on_clicked(lambda _evt, b=backend: self._on_backend_select(b))
            self.backend_buttons[backend] = btn

        ax_button_start = self.fig1.add_axes([0.66, 0.01, 0.08, 0.06])
        self.btn_start = Button(ax_button_start, "Start")
        self.btn_start.on_clicked(self._on_start)

        ax_button_stop = self.fig1.add_axes([0.75, 0.01, 0.08, 0.06])
        self.btn_stop = Button(ax_button_stop, "Stop")
        self.btn_stop.on_clicked(self._on_stop)

        ax_button_exit = self.fig1.add_axes([0.86, 0.01, 0.12, 0.06])
        self.btn_exit = Button(ax_button_exit, "Выход")
        self.btn_exit.on_clicked(self._on_exit)

        # ----- Figure 2: Pulse + Phase + FM -----
        self.fig2, (self.ax_pulse, self.ax_phase, self.ax_fm) = plt.subplots(
            3, 1, figsize=(14, 8.6), height_ratios=[1, 1, 1], num="Pulse Analysis"
        )
        (self.ln_pulse,) = self.ax_pulse.plot([], [], lw=1.4)
        self.ax_pulse.set_ylabel("RMS, dBm")
        self.ax_pulse.grid(True, alpha=0.3)
        self.ax_pulse.set_xlabel("Время, мс")

        (self.ln_phase,) = self.ax_phase.plot([], [], lw=1.4)
        self.ax_phase.set_ylabel("Фаза, rad")
        self.ax_phase.grid(True, alpha=0.3)
        self.ax_phase.set_ylim(-1.5, +1.5)

        (self.ln_fm,) = self.ax_fm.plot([], [], lw=1.4)
        self.ax_fm.set_xlabel("Время, мс")
        self.ax_fm.set_ylabel("FM, Hz")
        self.ax_fm.grid(True, alpha=0.3)

        self.fig2.subplots_adjust(hspace=0.5, top=0.92, bottom=0.15)
        self.fig2.suptitle("Импульс: RMS + Фаза + FM")

        # Нижние кнопки (по образцу beacon406_PSK_FM-plot.py)
        ax_button_msg = self.fig2.add_axes([0.06, 0.01, 0.15, 0.06])
        self.btn_msg = Button(ax_button_msg, "Сообщение")
        self.btn_msg.on_clicked(self._on_show_message)

        ax_button_stat = self.fig2.add_axes([0.22, 0.01, 0.15, 0.06])
        self.btn_stat = Button(ax_button_stat, "Статус SDR")
        self.btn_stat.on_clicked(self._on_show_sdr_status)

        ax_button_params = self.fig2.add_axes([0.38, 0.01, 0.15, 0.06])
        self.btn_params = Button(ax_button_params, "Параметры")
        self.btn_params.on_clicked(self._on_show_params)

        ax_button_spec = self.fig2.add_axes([0.54, 0.01, 0.15, 0.06])
        self.btn_spec = Button(ax_button_spec, "Спектр")
        self.btn_spec.on_clicked(self._on_show_spectrum)

        ax_button_save = self.fig2.add_axes([0.70, 0.01, 0.15, 0.06])
        self.btn_save = Button(ax_button_save, "Save IQ")
        self.btn_save.on_clicked(self._on_save_iq)

        ax_button_stop = self.fig2.add_axes([0.86, 0.01, 0.12, 0.06])
        self.btn_stop = Button(ax_button_stop, "Стоп")
        self.btn_stop.on_clicked(self._on_toggle_pulse_updates)

        # Single poll таймер (500ms для снижения нагрузки)
        self._timer = self.fig1.canvas.new_timer(interval=500)  # 500ms poll
        self._timer.add_callback(self._poll_status)
        self._timer.start()

        # Обработчики закрытия окон
        try:
            self.fig1.canvas.mpl_connect('close_event', self._on_close)
            self.fig2.canvas.mpl_connect('close_event', self._on_close)
            self.fig1.canvas.mpl_connect('key_press_event', self._on_key)
            self.fig2.canvas.mpl_connect('key_press_event', self._on_key)
        except Exception:
            pass

        # Autostart после инициализации
        if self.autostart:
            print("[startup] --autostart: sending start_acquire")
            self.client.start_acquire()

    # ---------- Pulse events from SUB ----------
    def _on_pulse_event(self, obj: Dict[str, Any]):
        """
        Обработка pulse/psk событий из PUB канала (в SUB потоке).
        НЕ обновляет графики напрямую - складывает снимок в очередь для главного потока.
        """
        typ = obj.get("type")
        if typ == "pulse":
            self._pulse_event_counter += 1

            # Диагностика: печатаем ключи первых 3 событий
            if self._pulse_event_counter <= 3:
                keys = sorted(obj.keys())
                print(f"[pulse_event #{self._pulse_event_counter}] received keys: {keys}")
                # Печатаем размеры массивов
                px = obj.get("phase_xs_ms", [])
                py = obj.get("phase_ys_rad", [])
                fx = obj.get("fr_xs_ms", [])
                fy = obj.get("fr_ys_hz", [])
                rms = obj.get("rms_ms_dbm", [])
                print(f"[pulse_event #{self._pulse_event_counter}] array sizes: phase_xs={len(px) if px else 0}, phase_ys={len(py) if py else 0}, fr_xs={len(fx) if fx else 0}, fr_ys={len(fy) if fy else 0}, rms={len(rms) if rms else 0}")

            # Собираем снимок данных (НЕ обновляем графики здесь!)
            snapshot = {
                "phase_xs_ms": obj.get("phase_xs_ms", []),
                "phase_ys_rad": obj.get("phase_ys_rad", []),
                "fr_xs_ms": obj.get("fr_xs_ms", []),
                "fr_ys_hz": obj.get("fr_ys_hz", []),
                "rms_ms_dbm": obj.get("rms_ms_dbm", []),
                "iq_seg": obj.get("iq_seg"),
                "core_gate": obj.get("core_gate"),
                "phase_metrics": obj.get("phase_metrics"),
                "msg_hex": obj.get("msg_hex"),
            }

            # Валидация схемы
            px = snapshot["phase_xs_ms"]
            py = snapshot["phase_ys_rad"]
            fx = snapshot["fr_xs_ms"]
            fy = snapshot["fr_ys_hz"]
            rms = snapshot["rms_ms_dbm"]

            valid = True
            if px and py and len(px) != len(py):
                print(f"[pulse_event] schema mismatch: len(phase_xs_ms)={len(px)} != len(phase_ys_rad)={len(py)}")
                valid = False
            if fx and fy and len(fx) != len(fy):
                print(f"[pulse_event] schema mismatch: len(fr_xs_ms)={len(fx)} != len(fr_ys_hz)={len(fy)}")
                valid = False
            if px and rms and len(px) != len(rms):
                print(f"[pulse_event] schema mismatch: len(phase_xs_ms)={len(px)} != len(rms_ms_dbm)={len(rms)}")
                valid = False

            if not valid:
                return

            # Кладём снимок в очередь (очистив если полна)
            try:
                while self._pulse_q.full():
                    try:
                        self._pulse_q.get_nowait()
                    except queue.Empty:
                        break
                self._pulse_q.put_nowait(snapshot)
            except queue.Full:
                pass  # Очередь заполнена, пропускаем

        elif typ == "psk":
            # PSK decoded message
            try:
                msg_hex = obj.get("msg_hex")
                if msg_hex is not None:
                    self.last_msg_hex = str(msg_hex)
            except Exception:
                pass

    # ---------- Single poll loop ----------
    def _poll_status(self):
        """
        Основной цикл опроса (вызывается таймером каждые 500ms):
        1. get_status
        2. Обновить status bar
        3. Обновить кнопки
        4. Обновить RMS график (Sliding RMS)
        5. Извлечь последние pulse события из очереди и обновить графики второго окна
        """
        try:
            st_rep = self.client.get_status()
            if not st_rep.get("ok", False):
                # Если ошибка - показать в status bar
                error = st_rep.get("error", "Connection error")
                self.fig1.suptitle(f"ERROR: {error}", fontsize=10, color='red')
                return

            st = st_rep.get("status", {})
            self.last_status = st

            # Обновить sample_rate
            fs = float(st.get("fs", self.sample_rate))
            if fs > 0:
                self.sample_rate = fs

            # Обновить status bar
            self._update_status_bar()

            # Обновить состояние кнопок
            self._update_buttons()

            # Обновить RMS график (Sliding RMS)
            self._update_rms_plot(st)

            # Обновить графики второго окна из очереди pulse событий
            self._update_pulse_plots()

        except Exception as e:
            # Не падать при ошибках poll
            print(f"[poll_status] error: {e}")
            self.fig1.suptitle(f"ERROR: {e}", fontsize=10, color='red')

    def _update_status_bar(self):
        """Обновить строку статуса (suptitle) на Fig1."""
        s = self.last_status
        backend = s.get("backend", "?")
        acq_state = s.get("acq_state", "?")
        ready = s.get("ready", False)
        fs = s.get("fs", 0) or 0
        bb_shift = s.get("bb_shift_hz", 0) or 0
        thresh = s.get("thresh_dbm", -60.0)

        # Подсветка retuning
        state_str = acq_state
        if acq_state == "retuning":
            state_str = f"⏳ {acq_state}"

        txt = f"backend={backend} | state={state_str} | ready={ready} | fs={fs:.0f} | bb_shift={bb_shift:.0f} | thresh={thresh:.1f}"
        self.fig1.suptitle(txt, fontsize=10)

    def _update_buttons(self):
        """Обновить состояние кнопок Start/Stop по acq_state."""
        acq_state = self.last_status.get("acq_state", "stopped")

        # Start активна если stopped, неактивна если running/retuning
        if acq_state == "stopped":
            self.btn_start.ax.set_facecolor('lightgreen')
            self.btn_stop.ax.set_facecolor('lightgray')
        elif acq_state == "running":
            self.btn_start.ax.set_facecolor('lightgray')
            self.btn_stop.ax.set_facecolor('lightcoral')
        else:  # retuning, ready
            self.btn_start.ax.set_facecolor('lightyellow')
            self.btn_stop.ax.set_facecolor('lightyellow')

    def _update_rms_plot(self, st: Dict[str, Any]):
        """Обновить RMS график (Sliding RMS) из get_status."""
        t_s = st.get("t_s", [])
        last_rms_dbm = st.get("last_rms_dbm", [])
        thresh_dbm = st.get("thresh_dbm")

        if t_s and last_rms_dbm:
            self._lvl_x = np.array(t_s, dtype=np.float64)
            self._lvl_y = np.array(last_rms_dbm, dtype=np.float64)
            self.ln_lvl.set_data(self._lvl_x, self._lvl_y)
            if self._lvl_x.size > 1:
                x_min, x_max = self._lvl_x[0], self._lvl_x[-1]
                self.ax_lvl.set_xlim(x_min, x_max)
                ymin, ymax = np.nanmin(self._lvl_y), np.nanmax(self._lvl_y)
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                    self.ax_lvl.set_ylim(ymin - 2, ymax + 2)

                # Обновить линию порога
                if thresh_dbm is not None and np.isfinite(thresh_dbm):
                    self.ln_thresh.set_data([x_min, x_max], [thresh_dbm, thresh_dbm])
                else:
                    self.ln_thresh.set_data([], [])

            self.fig1.canvas.draw_idle()

    def _update_pulse_plots(self):
        """
        Извлекает последнее pulse событие из очереди и обновляет графики второго окна.
        Вызывается на главном потоке из _poll_status().
        """
        # Извлекаем последний снимок из очереди (пропуская старые)
        snapshot = None
        while True:
            try:
                snapshot = self._pulse_q.get_nowait()
            except queue.Empty:
                break

        if snapshot is None:
            return  # Нет новых данных

        # Диагностика: печатаем что пришло
        px = snapshot.get("phase_xs_ms", [])
        py = snapshot.get("phase_ys_rad", [])
        print(f"[update_pulse_plots] Received snapshot: phase_xs={len(px) if px else 0}, phase_ys={len(py) if py else 0}, pulse_updates_enabled={self.pulse_updates_enabled}")

        # Сохраняем метрики для кнопок (всегда, даже если pulse_updates_enabled=False)
        try:
            iq_seg = snapshot.get("iq_seg")
            if iq_seg is not None and len(iq_seg) > 0:
                self.last_iq_seg = np.array(iq_seg, dtype=np.complex64)

            core_gate = snapshot.get("core_gate")
            if core_gate is not None:
                self.last_core_gate = tuple(core_gate)

            metrics = snapshot.get("phase_metrics")
            if metrics is not None:
                self.last_phase_metrics = metrics

            msg_hex = snapshot.get("msg_hex")
            if msg_hex is not None:
                self.last_msg_hex = str(msg_hex)
        except Exception as e:
            print(f"[update_pulse_plots] Ошибка сохранения метрик: {e}")

        # Обновляем графики только если разрешено
        if not self.pulse_updates_enabled:
            return

        # Извлекаем данные для графиков
        px = np.array(snapshot.get("phase_xs_ms", []), dtype=np.float64)
        py = np.array(snapshot.get("phase_ys_rad", []), dtype=np.float64)
        fx = np.array(snapshot.get("fr_xs_ms", []), dtype=np.float64)
        fy = np.array(snapshot.get("fr_ys_hz", []), dtype=np.float64)
        rms = np.array(snapshot.get("rms_ms_dbm", []), dtype=np.float64)

        # Проверка валидности данных
        need_draw = False

        # RMS график
        if px.size > 1 and rms.size == px.size:
            self._pulse_rms_x = px
            self._pulse_rms_y = rms
            self.ln_pulse.set_data(px, rms)
            self.ax_pulse.set_xlim(px.min(), px.max())
            ymin, ymax = np.nanmin(rms), np.nanmax(rms)
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                self.ax_pulse.set_ylim(ymin - 2, ymax + 2)
            need_draw = True
        elif px.size > 0 or rms.size > 0:
            print(f"[update_pulse_plots] skip RMS: len mismatch px={px.size} rms={rms.size}")

        # Фаза график
        if px.size > 1 and py.size == px.size:
            self._phase_x = px
            self._phase_y = py
            self.ln_phase.set_data(px, py)
            self.ax_phase.set_xlim(px.min(), px.max())
            self.ax_phase.set_ylim(-1.5, +1.5)
            need_draw = True
        elif px.size > 0 or py.size > 0:
            print(f"[update_pulse_plots] skip Phase: len mismatch px={px.size} py={py.size}")

        # FM график
        if fx.size > 1 and fy.size == fx.size:
            self._fm_x = fx
            self._fm_y = fy
            self.ln_fm.set_data(fx, fy)
            self.ax_fm.set_xlim(fx.min(), fx.max())
            ymin, ymax = np.nanmin(fy), np.nanmax(fy)
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                self.ax_fm.set_ylim(ymin - 50, ymax + 50)
            need_draw = True
        elif fx.size > 0 or fy.size > 0:
            print(f"[update_pulse_plots] skip FM: len mismatch fx={fx.size} fy={fy.size}")

        if need_draw:
            self.fig2.canvas.draw_idle()

    def _can_send_command(self) -> bool:
        """Проверка анти-дребезга: разрешить команду если прошло >= debounce_ms."""
        now = time.perf_counter()
        dt_ms = (now - self._last_cmd_time) * 1000
        if dt_ms < self._cmd_debounce_ms:
            return False
        self._last_cmd_time = now
        return True

    # ---------- UI callbacks ----------
    def _on_start(self, _event):
        if not self._can_send_command():
            return
        self.client.start_acquire()

    def _on_stop(self, _event):
        if not self._can_send_command():
            return
        self.client.stop_acquire()

    def _on_backend_select(self, backend_name: str):
        if not self._can_send_command():
            return

        # For file backend, ask for file first
        if backend_name == "file":
            try:
                root = tk.Tk()
                root.withdraw()
                path = filedialog.askopenfilename(
                    title="Выберите CF32 или SigMF",
                    filetypes=[("CF32", "*.cf32"), ("F32", "*.f32"), ("SigMF", "*.sigmf-meta"), ("All", "*.*")]
                )
                root.destroy()
                if not path:
                    return  # User cancelled
            except Exception as e:
                print(f"[file_dialog] error: {e}")
                return
        else:
            path = None

        # Отправить set_sdr_config с мгновенным ACK
        cfg = {"backend_name": backend_name}
        if backend_name == "file" and path:
            cfg["backend_args"] = path
            cfg["bb_shift_enable"] = False
            cfg["bb_shift_hz"] = 0

        print(f"[backend_select] switching to {backend_name}")
        self.client.set_sdr_config(**cfg)
        # Не ждать долго — poll_status покажет retuning → ready

    def _on_toggle_pulse_updates(self, _event):
        """Переключение обновлений второго окна (Стоп/Старт)."""
        self.pulse_updates_enabled = not self.pulse_updates_enabled
        try:
            self.btn_stop.label.set_text("Старт" if not self.pulse_updates_enabled else "Стоп")
        except Exception:
            pass
        state = "включено" if self.pulse_updates_enabled else "выключено"
        print(f"[pulse_window] Обновления второго окна: {state}")

    def _on_show_message(self, _event):
        """Показывает декодированное сообщение EPIRB/ELT в отдельном окне."""
        hex_msg = self.last_msg_hex
        if not hex_msg or hex_msg == "None":
            fig, ax = plt.subplots(figsize=(8, 3))
            try:
                fig.canvas.manager.set_window_title("Декодированное сообщение")
            except Exception:
                pass
            ax.axis('off')
            ax.text(0.5, 0.5, "Нет сообщения для декодирования.\nСначала должен быть обнаружен PSK импульс.",
                    ha='center', va='center', fontsize=12)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            return

        try:
            # Импортируем модули декодирования
            import sys
            from pathlib import Path
            beacon_lib = Path(__file__).parent / "lib"
            if str(beacon_lib) not in sys.path:
                sys.path.insert(0, str(beacon_lib))
            from hex_decoder import hex_to_bits, build_table_rows

            bits = hex_to_bits(hex_msg)
            if len(bits) != 144:
                bits = (bits + [0]*144)[:144]

            headers = ["Binary Range", "Binary Content", "Field Name", "Decoded Value"]
            rows = build_table_rows(bits)

            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('off')
            fig.suptitle(f"EPIRB/ELT Beacon Parameters Decoder\nHEX: {hex_msg}",
                        fontsize=11, fontweight='bold')

            tbl = ax.table(cellText=[headers] + rows, loc='center', cellLoc='left',
                          colWidths=[0.12, 0.25, 0.28, 0.35])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)

            for i in range(len(headers)):
                cell = tbl[(0, i)]
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.05)

            for i in range(1, len(rows) + 1):
                for j in range(len(headers)):
                    cell = tbl[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f0f0')
                    cell.set_height(0.04)
                    if j == 2:
                        field_name = rows[i-1][2].lower()
                        if any(key in field_name for key in ['country', 'mmsi', 'lat', 'lon', 'id']):
                            cell.set_facecolor('#e8f4ff')

            try:
                fig.canvas.manager.set_window_title("EPIRB/ELT Beacon Decoder")
            except Exception:
                pass
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 3))
            try:
                fig.canvas.manager.set_window_title("Ошибка декодирования")
            except Exception:
                pass
            ax.axis('off')
            ax.text(0.5, 0.5, f"Ошибка декодирования сообщения:\n{str(e)}\n\nHEX: {hex_msg}",
                    ha='center', va='center', fontsize=10)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

    def _calculate_fallback_metrics(self):
        """
        Fallback-расчёт метрик из сохранённых данных графиков,
        если phase_metrics недоступен от сервиса.
        """
        m = {}
        try:
            px = self._phase_x
            py = self._phase_y
            fx = self._fm_x
            fy = self._fm_y
            rms_y = self._pulse_rms_y

            # Target Signal / Frequency Offset
            m["Target Signal (Hz)"] = float('nan')
            m["Frequency Offset (Hz)"] = float('nan')

            # Message Duration
            if px.size > 1:
                m["Message Duration (ms)"] = float(px[-1] - px[0])
            else:
                m["Message Duration (ms)"] = float('nan')

            # Carrier Duration (временно NaN - требуется edge detection)
            m["Carrier Duration (ms)"] = float('nan')

            # Pos/Neg: медианы по плато
            if py.size > 0:
                pos_mask = py > 0.6
                neg_mask = py < -0.6
                m["Pos (rad)"] = float(np.median(py[pos_mask])) if np.any(pos_mask) else float('nan')
                m["Neg (rad)"] = float(np.median(py[neg_mask])) if np.any(neg_mask) else float('nan')
            else:
                m["Pos (rad)"] = float('nan')
                m["Neg (rad)"] = float('nan')

            # Rise/Fall: переходы -0.88 → +0.88 (упрощённо)
            rise_us = float('nan')
            fall_us = float('nan')
            if py.size > 1 and px.size == py.size:
                # Ищем первый переход снизу вверх через 0
                for i in range(1, len(py)):
                    if py[i-1] < -0.88 and py[i] > 0.88:
                        rise_us = (px[i] - px[i-1]) * 1000  # ms to μs
                        break
                # Ищем первый переход сверху вниз
                for i in range(1, len(py)):
                    if py[i-1] > 0.88 and py[i] < -0.88:
                        fall_us = (px[i] - px[i-1]) * 1000  # ms to μs
                        break
            m["Rise (μs)"] = rise_us
            m["Fall (μs)"] = fall_us

            # Asymmetry: упрощённая оценка
            m["Asymmetry (%)"] = float('nan')

            # Fmod: оценка по средней длительности полубита
            fmod_hz = float('nan')
            if py.size > 2 and px.size == py.size:
                # Ищем переходы через 0
                zero_crossings = []
                for i in range(1, len(py)):
                    if (py[i-1] < 0 and py[i] >= 0) or (py[i-1] > 0 and py[i] <= 0):
                        # Интерполяция для точного времени
                        t_cross = px[i-1] + (0 - py[i-1]) / (py[i] - py[i-1]) * (px[i] - px[i-1])
                        zero_crossings.append(t_cross)
                if len(zero_crossings) >= 2:
                    # Средняя длительность полубита в мс
                    intervals = np.diff(zero_crossings)
                    mean_halfbit_ms = np.mean(intervals)
                    if mean_halfbit_ms > 0:
                        fmod_hz = 1000.0 / (2 * mean_halfbit_ms)  # Полный бит = 2 полубита
            m["Fmod (Hz)"] = fmod_hz

            # Power (RMS): медиана по массиву RMS
            if rms_y.size > 0:
                m["Power (RMS, dBm)"] = float(np.median(rms_y))
            else:
                m["Power (RMS, dBm)"] = float('nan')

        except Exception as e:
            print(f"[fallback_metrics] error: {e}")

        return m

    def _on_show_params(self, _event):
        """Показывает таблицу параметров фазы."""
        m = self.last_phase_metrics
        hex_msg = self.last_msg_hex
        is_approx = False

        # Если метрики недоступны - пробуем fallback
        if not m:
            # Проверяем, есть ли данные для fallback
            if self._phase_x.size > 0 and self._phase_y.size > 0:
                m = self._calculate_fallback_metrics()
                is_approx = True
            else:
                fig, ax = plt.subplots(figsize=(8, 3))
                try:
                    fig.canvas.manager.set_window_title("Phase Parameters")
                except Exception:
                    pass
                ax.axis('off')
                ax.text(0.5, 0.5, "Нет параметров.\nСначала должен быть обнаружен PSK импульс.",
                        ha='center', va='center', fontsize=12)
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)
                return

        headers = ["Parameter", "Value"]
        rows = [
            ["Target Signal (Hz)", f"{m.get('Target Signal (Hz)', float('nan')):.3f}"],
            ["Frequency Offset (Hz)", f"{m.get('Frequency Offset (Hz)', float('nan')):.3f}"],
            ["Message Duration (ms)", f"{m.get('Message Duration (ms)', float('nan')):.3f}"],
            ["Carrier Duration (ms)", f"{m.get('Carrier Duration (ms)', float('nan')):.3f}"],
            ["Pos (rad)", f"{m.get('Pos (rad)', float('nan')):.3f}"],
            ["Neg (rad)", f"{m.get('Neg (rad)', float('nan')):.3f}"],
            ["Rise (μs)", f"{m.get('Rise (μs)', float('nan')):.1f}"],
            ["Fall (μs)", f"{m.get('Fall (μs)', float('nan')):.1f}"],
            ["Asymmetry (%)", f"{m.get('Asymmetry (%)', float('nan')):.3f}"],
            ["Fmod (Hz)", f"{m.get('Fmod (Hz)', float('nan')):.3f}"],
            ["Power (RMS, dBm)", f"{m.get('Power (RMS, dBm)', float('nan')):.2f}"],
            ["HEX", str(hex_msg) if hex_msg is not None else ""]
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        tbl = ax.table(cellText=[headers] + rows, loc='center', cellLoc='left',
                      colWidths=[0.5, 0.5])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        for i in range(len(headers)):
            cell = tbl[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.06)

        for i in range(1, len(rows) + 1):
            for j in range(len(headers)):
                cell = tbl[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                cell.set_height(0.05)

        title = "Phase Parameters"
        if is_approx:
            title += " (approx - calculated from graphs)"
        else:
            title += " (snapshot)"
        fig.suptitle(title, fontsize=12, fontweight='bold')
        try:
            fig.canvas.manager.set_window_title("Phase Parameters")
        except Exception:
            pass
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def _on_show_sdr_status(self, _event):
        """Показывает таблицу статуса SDR."""
        st = self.last_status
        if not st:
            fig, ax = plt.subplots(figsize=(6, 2))
            try:
                fig.canvas.manager.set_window_title("SDR Status")
            except Exception:
                pass
            ax.axis('off')
            ax.text(0.5, 0.5, "Нет данных статуса.\nПодключитесь к DSP сервису.",
                    ha='center', va='center')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            return

        preferred = [
            "backend", "driver", "requested_sample_rate_sps", "actual_sample_rate_sps",
            "requested_center_freq_hz", "actual_center_freq_hz", "bandwidth_hz",
            "hw_sample_rate_sps", "decim", "agc_on", "overall_gain_db", "stage_gains_db",
            "corr_ppm", "calib_offset_db", "file_path", "if_offset_hz", "mix_shift_hz", "eof",
            "device_info", "ref_level_dbm", "acq_state", "ready", "fs", "bb_shift_hz", "thresh_dbm"
        ]
        keys, seen = [], set()
        for k in preferred:
            if k in st and k not in seen:
                keys.append(k)
                seen.add(k)
        for k in sorted(st.keys()):
            if k not in seen:
                keys.append(k)
                seen.add(k)

        headers = ["Key", "Value"]
        rows = []
        for k in keys:
            v = st.get(k, "")
            if k == "stage_gains_db" and isinstance(v, dict):
                try:
                    v = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                except Exception:
                    v = str(v)
            rows.append([str(k), str(v)])

        if not rows:
            rows = [["status", "нет данных"]]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        tbl = ax.table(cellText=[headers] + rows, loc='center', cellLoc='left',
                      colWidths=[0.4, 0.6])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        for i in range(len(headers)):
            cell = tbl[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.05)

        for i in range(1, len(rows) + 1):
            for j in range(len(headers)):
                cell = tbl[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                cell.set_height(0.04)
                if j == 0:
                    key_name = rows[i-1][0].lower()
                    if any(word in key_name for word in ['actual', 'file_path', 'backend', 'driver']):
                        cell.set_facecolor('#e8f4ff')

        fig.suptitle("SDR Status (snapshot)", fontsize=12, fontweight='bold')
        try:
            fig.canvas.manager.set_window_title("SDR Status")
        except Exception:
            pass
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def _on_show_spectrum(self, _event):
        """Показывает спектр ядра импульса."""
        seg = self.last_iq_seg
        gate = self.last_core_gate
        if seg is None or seg.size < 8:
            print("[spectrum] Нет сегмента для построения спектра. Сначала должен быть обнаружен импульс.")
            return
        if not gate or gate[1] - gate[0] < 8:
            print("[spectrum] Нет валидного ядра импульса (gate). Спектр не построен.")
            return

        self._plot_fft_sector(
            seg, FFT_N=65536, avg=4, window="hann",
            sector_center_hz=0.0, sector_half_width_hz=50_000,
            y_ref_db=0.0, gate_samples=gate, remove_dc=True,
            normalize="dBc", y_lim=None, show_mask=True
        )

    def _plot_fft_sector(self, iq_data, FFT_N=65536, avg=4, window="hann",
                         sector_center_hz=0.0, sector_half_width_hz=50_000,
                         y_ref_db=0.0, *, gate_samples=None, remove_dc=True,
                         normalize="dBc", y_lim=None, show_mask=True):
        """Рисует сектор спектра с маской C/S T.001."""
        SR = float(self.sample_rate)
        seg = np.asarray(iq_data, dtype=np.complex64)

        if gate_samples is not None:
            i0, i1 = map(int, gate_samples)
            i0 = max(0, i0)
            i1 = min(len(seg), i1)
            seg = seg[i0:i1]

        if seg.size < 16:
            print("[spectrum] Сегмент слишком короткий для FFT")
            return

        if remove_dc:
            seg = seg - np.mean(seg)

        win = (np.hanning(FFT_N) if window == "hann" else np.ones(FFT_N)).astype(np.float32)

        n_possible = seg.size // FFT_N
        if n_possible == 0:
            seg = np.pad(seg, (0, FFT_N - seg.size))
            n_possible = 1
        n_blocks = int(max(1, min(int(avg), n_possible)))

        psd_acc = None
        w2 = float(np.sum(win ** 2))
        for k in range(n_blocks):
            chunk = seg[k*FFT_N:(k+1)*FFT_N]
            if chunk.size < FFT_N:
                chunk = np.pad(chunk, (0, FFT_N - chunk.size))
            X = np.fft.fftshift(np.fft.fft(chunk * win, n=FFT_N))
            mag2 = (np.abs(X) ** 2) / w2
            psd_acc = mag2 if psd_acc is None else (psd_acc + mag2)

        psd = psd_acc / n_blocks
        psd_db = 10.0 * np.log10(psd + 1e-30) + y_ref_db

        f_axis = np.fft.fftshift(np.fft.fftfreq(FFT_N, d=1.0 / SR))

        f0 = float(sector_center_hz)
        hw = abs(float(sector_half_width_hz))
        mask = (f_axis >= f0 - hw) & (f_axis <= f0 + hw)
        if not np.any(mask):
            print("[spectrum] Сектор вне диапазона частотной оси.")
            return

        f_sec = f_axis[mask]
        psd_sec_db = psd_db[mask]

        i_pk = int(np.argmax(psd_sec_db))
        f_pk = float(f_sec[i_pk])
        p_pk_abs_db = float(psd_sec_db[i_pk])

        if str(normalize).lower() == "dbc":
            psd_plot = psd_sec_db - p_pk_abs_db
            ylabel = "Амплитуда, dBc"
            p_pk_dBc = 0.0
            mask_offset = 0.0
        else:
            psd_plot = psd_sec_db
            ylabel = "Амплитуда, dB"
            p_pk_dBc = float(p_pk_abs_db - p_pk_abs_db)
            mask_offset = p_pk_abs_db

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(f_sec / 1e3, psd_plot, lw=1.5, label="Спектр")
        ax.axvline(0.0, linestyle='--', alpha=0.5)

        def _plot_cs_t001_mask(ax, p0, half_width_hz):
            segments = [
                (0.0, 3e3, 0.0), (3e3, 7e3, -20.0), (7e3, 12e3, -30.0),
                (12e3, 24e3, -35.0), (24e3, half_width_hz, -40.0),
            ]
            for sgn in (-1.0, +1.0):
                for f1, f2, lvl in segments:
                    x1 = sgn * (f1 / 1e3)
                    x2 = sgn * (f2 / 1e3)
                    y = p0 + lvl
                    ax.plot([x1, x2], [y, y], linestyle='--', linewidth=1.2, alpha=0.9)
            for fb in [3e3, 7e3, 12e3, 24e3]:
                ax.axvline(+fb/1e3, linestyle=':', alpha=0.6)
                ax.axvline(-fb/1e3, linestyle=':', alpha=0.6)

        if show_mask:
            _plot_cs_t001_mask(ax, mask_offset, hw)

        ax.plot([f_pk / 1e3], [0.0 if normalize.lower() == "dbc" else p_pk_abs_db], 'o', label="Пик")
        ax.set_title("FFT сектор (статичный, без IF-сдвигов)")
        ax.set_xlabel("Частота, кГц (baseband)")
        ax.set_ylabel(ylabel)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.5)
        ax.legend(loc="best")

        fig.suptitle(
            f"Fs: {SR/1000:.1f} kSPS, сектор центр={sector_center_hz/1000:.3f} кГц, ±{sector_half_width_hz/1000:.1f} кГц",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show(block=False)
        plt.pause(0.1)

    def _on_save_iq(self, _event):
        """Сохраняет последний IQ сегмент импульса в CF32."""
        if not self._can_send_command():
            return
        data = self.last_iq_seg
        if data is None or data.size == 0:
            print("[save_iq] Нет данных для сохранения. Сначала должен быть обнаружен импульс.")
            return

        import time
        from pathlib import Path
        root = Path(__file__).parent.parent / "captures"
        root.mkdir(exist_ok=True)
        fname = root / time.strftime("iq_pulse_%Y%m%d_%H%M%S.cf32")
        data.astype(np.complex64).tofile(str(fname))
        print(f"[save_iq] Pulse IQ сохранён: {fname} ({data.size} samples)")

    def _on_save_sigmf(self, _event):
        if not self._can_send_command():
            return
        rep = self.client.save_sigmf()
        if rep.get("ok", False):
            path = rep.get("path", "?")
            print(f"[save] Saved to {path}")

    def _on_exit(self, _event):
        self.clean_exit(0)

    def _on_close(self, _evt):
        self.clean_exit(0)

    def _on_key(self, evt):
        if evt.key in ('q', 'escape'):
            self.clean_exit(0)

    def clean_exit(self, code: int = 0):
        try:
            self._timer.stop()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass
        try:
            plt.close('all')
        except Exception:
            pass
        try:
            if getattr(self, '_svc_proc', None):
                self._svc_proc.terminate()
        except Exception:
            pass
        import sys
        sys.exit(code)

# -------------------- Auto-start dsp_service --------------------
import subprocess
from shutil import which

def _candidate_service_paths() -> list:
    here = Path(__file__).resolve().parent
    names = [
        "beacon_dsp_service.py",
        str(here / "beacon_dsp_service.py"),
        str(here.parent / "beacon_dsp_service.py"),
        str(Path.cwd() / "beacon_dsp_service.py"),
    ]
    out = []
    for n in names:
        try:
            pn = Path(n)
            if pn.exists():
                out.append(str(pn))
        except Exception:
            pass
    # dedup
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def _spawn_service_if_needed(client):
    """Try get_status(); if it fails, spawn the service and wait until ready."""
    try:
        rep = client.get_status()
        if isinstance(rep, dict) and rep.get("ok", True):
            return None  # already running
    except Exception:
        pass
    # locate file and spawn
    for cand in _candidate_service_paths():
        try:
            py = sys.executable or which("python") or which("python3")
            if not py:
                break
            print(f"[auto] launching dsp_service: {py} {cand}")
            proc = subprocess.Popen([py, cand], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # wait up to ~6s for readiness
            t0 = time.time()
            while time.time() - t0 < 6.0:
                try:
                    rep = client.get_status()
                    if isinstance(rep, dict) and rep.get("ok", False):
                        print("[auto] dsp_service is ready")
                        return proc
                except Exception:
                    pass
                time.sleep(0.4)
            try:
                proc.terminate()
            except Exception:
                pass
        except Exception as e:
            print("[auto] launch error:", e)
    print("[auto] dsp_service not found or failed to start")
    return None

# -------------------- main --------------------
def parse_args(argv):
    import argparse
    ap = argparse.ArgumentParser(description="beacon_dsp2plot — UI для beacon_dsp_service")
    ap.add_argument("--pub", default=PUB_ADDR_DEFAULT, help="ZeroMQ PUB адрес сервиса")
    ap.add_argument("--rep", default=REP_ADDR_DEFAULT, help="ZeroMQ REP адрес сервиса")
    ap.add_argument("--autostart", action="store_true", help="Отправить start_acquire при запуске")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Create client first for auto-start check
    temp_client = DspServiceClient(pub_addr=args.pub, rep_addr=args.rep)

    # Auto-start service if not running
    proc = None
    try:
        proc = _spawn_service_if_needed(temp_client)
    except Exception as e:
        print('[auto] spawn skipped:', e)
    finally:
        temp_client.close()

    # Create UI (will reconnect to service)
    ui = Dsp2PlotUI(pub_addr=args.pub, rep_addr=args.rep, autostart=args.autostart)
    if proc is not None:
        ui._svc_proc = proc

    def _sig(_signum, _frame):
        try:
            ui.clean_exit(0)
        except SystemExit:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig)
        except Exception:
            pass

    plt.show()

if __name__ == "__main__":
    main()
