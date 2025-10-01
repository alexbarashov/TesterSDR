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

        # Нижние кнопки
        ax_button_save = self.fig2.add_axes([0.70, 0.01, 0.15, 0.06])
        self.btn_save = Button(ax_button_save, "Save SigMF")
        self.btn_save.on_clicked(self._on_save_sigmf)

        ax_button_exit2 = self.fig2.add_axes([0.86, 0.01, 0.12, 0.06])
        self.btn_exit2 = Button(ax_button_exit2, "Выход")
        self.btn_exit2.on_clicked(self._on_exit)

        # Single poll таймер
        self._timer = self.fig1.canvas.new_timer(interval=300)  # 300ms poll
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
        """Обработка pulse/psk событий из PUB канала (в SUB потоке)."""
        typ = obj.get("type")
        if typ == "pulse":
            # Pulse event - обновить графики Phase/FM/RMS
            px = np.array(obj.get("phase_xs_ms", []), dtype=np.float64)
            py = np.array(obj.get("phase_ys_rad", []), dtype=np.float64)
            fx = np.array(obj.get("fr_xs_ms", []), dtype=np.float64)
            fy = np.array(obj.get("fr_ys_hz", []), dtype=np.float64)
            rms = np.array(obj.get("rms_ms_dbm", []), dtype=np.float64)

            if px.size > 1 and rms.size == px.size:
                self._pulse_rms_x = px
                self._pulse_rms_y = rms
                self.ln_pulse.set_data(px, rms)
                self.ax_pulse.set_xlim(px.min(), px.max())
                ymin, ymax = np.nanmin(rms), np.nanmax(rms)
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                    self.ax_pulse.set_ylim(ymin - 2, ymax + 2)

            if px.size > 1 and py.size == px.size:
                self._phase_x = px
                self._phase_y = py
                self.ln_phase.set_data(px, py)
                self.ax_phase.set_xlim(px.min(), px.max())

            if fx.size > 1 and fy.size == fx.size:
                self._fm_x = fx
                self._fm_y = fy
                self.ln_fm.set_data(fx, fy)
                self.ax_fm.set_xlim(fx.min(), fx.max())
                ymin, ymax = np.nanmin(fy), np.nanmax(fy)
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                    self.ax_fm.set_ylim(ymin - 50, ymax + 50)

            self.fig2.canvas.draw_idle()

        elif typ == "psk":
            # PSK decoded message
            pass  # Можно добавить обработку декодированных сообщений

    # ---------- Single poll loop ----------
    def _poll_status(self):
        """
        Основной цикл опроса (вызывается таймером каждые 300ms):
        1. get_status
        2. Обновить status bar
        3. Обновить кнопки
        4. Обновить RMS график (Sliding RMS)
        """
        st_rep = self.client.get_status()
        if not st_rep.get("ok", False):
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
        if t_s and last_rms_dbm:
            self._lvl_x = np.array(t_s, dtype=np.float64)
            self._lvl_y = np.array(last_rms_dbm, dtype=np.float64)
            self.ln_lvl.set_data(self._lvl_x, self._lvl_y)
            if self._lvl_x.size > 1:
                self.ax_lvl.set_xlim(self._lvl_x[0], self._lvl_x[-1])
                ymin, ymax = np.nanmin(self._lvl_y), np.nanmax(self._lvl_y)
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
                    self.ax_lvl.set_ylim(ymin - 2, ymax + 2)
            self.fig1.canvas.draw_idle()

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
