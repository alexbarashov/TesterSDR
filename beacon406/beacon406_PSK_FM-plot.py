#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtl_soapy_RMS_FSK_SaveIQ_p_f.py
Realtime Sliding RMS + Pulse/ view for RTL-SDR via SoapySDR
Adds a "Save IQ" button to dump current RAW IQ buffer in CF32 (complex64), bit-for-bit.
"""

import time
from pathlib import Path
import threading
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import queue
from tkinter import filedialog
import tkinter as tk


from lib.backends import safe_make_backend
from lib.metrics import process_psk_impulse
from lib.demod import phase_demod_psk_msg_safe
from lib.config import BACKEND_NAME, BACKEND_ARGS
from lib.hex_decoder import hex_to_bits, build_table_rows
# FM discriminator
from lib.processing_fm import fm_discriminator
from lib.logger import get_logger, setup_logging

setup_logging()                 # единая точка инициализации логов
log = get_logger(__name__)      # единственный логгер в модуле


# SoapySDR import handled by sdr_backends
# ==========================
# ПАРАМЕТРЫ
# ==========================
TARGET_SIGNAL_HZ    = 406_037_000     # Целевая частота (пример: 406 PSK)
IF_OFFSET_HZ        = -37_000         # ПЧ-смещение (настраиваемся ниже цели)
#TARGET_SIGNAL_HZ    = 161_975_000     # Целевая частота (пример: AIS GMSK)
#IF_OFFSET_HZ        = +25_000         # ПЧ-смещение (настраиваемся ниже цели)
#TARGET_SIGNAL_HZ    = 156_525_000     # Целевая частота (пример: DSC FSK)
#IF_OFFSET_HZ        = -25_000         # ПЧ-смещение (настраиваемся ниже цели)
CENTER_FREQ_HZ      = TARGET_SIGNAL_HZ + IF_OFFSET_HZ
#SAMPLE_RATE_SPS     = 1_024_000
SAMPLE_RATE_SPS     = 1_000_000
USE_MANUAL_GAIN     = True
TUNER_GAIN_DB       = 30.0             # не используется 
ENABLE_AGC          = False
FREQ_CORR_PPM       = 0

# ---- SDR backend selection (switch here only) ----
#BACKEND_NAME = "soapy_rtl"   # "soapy_hackrf" | "soapy_airspy" | "soapy_sdrplay" | "file"
#BACKEND_ARGS = None            # e.g., {"driver":"rtlsdr"} or for file: {"path": r"C:/path/to/trace.cf32"}

BB_SHIFT_ENABLE     = True            # Если true, хранит уже смещённый БП
BB_SHIFT_HZ         = (IF_OFFSET_HZ)
RMS_WIN_MS          = 1.0
VIS_DECIM           = 2048
LEVEL_HISTORY_SEC   = 12
DBM_OFFSET_DB       = -30.0           # Внешний аттенюатор dBm 
PRINT_EVERY_N_SEC   = 1
READ_CHUNK          = 65_536
#READ_CHUNK          = 16_384
PULSE_THRESH_DBM    = -45.0
PULSE_STORE_SEC     = 1.5
PSK_YLIMIT_RAD       = 1.5
PSK_BASELINE_MS      = 2.0
EPS = 1e-20
DEBUG_IMPULSE_LOG   = True


def _is_file_wait_mode():
    try:
        from lib.config import BACKEND_NAME, BACKEND_ARGS
    except Exception:
        return False
    if BACKEND_NAME != "file":
        return False
    if not BACKEND_ARGS:
        return True
    # если словарь есть, но пути нет/пустой
    path = BACKEND_ARGS.get("path") if isinstance(BACKEND_ARGS, dict) else None
    return (path is None) or (str(path).strip() == "")


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
FILE_IQ = "iq_pulse_%Y%m%d_%H%M%S.cf32"

# формируем шаблон пути (сразу строка!)
file_path = str(ROOT / "captures" / FILE_IQ)


def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, EPS))

class SoapySlidingRMS:
    def __init__(self):
        # Params table snapshot
        self.last_phase_metrics = None
        self.last_msg_hex = None
 
        extra_kwargs = {"if_offset_hz": IF_OFFSET_HZ} if (BACKEND_NAME == "file") else {}


        # --- перед блоком создания backend:
        self.backend = None
        self.sample_rate = float(SAMPLE_RATE_SPS)  # временно для подписей осей

        # режим "ждём файл", если выбран backend=file, но путь к файлу не задан
        def _is_file_wait_mode():
            try:
                from lib.config import BACKEND_NAME, BACKEND_ARGS
            except Exception:
                return False
            if BACKEND_NAME != "file":
                return False
            if not BACKEND_ARGS:
                return True
            path = BACKEND_ARGS.get("path") if isinstance(BACKEND_ARGS, dict) else None
            return (path is None) or (str(path).strip() == "")

        file_wait = _is_file_wait_mode()
        extra_kwargs = {"if_offset_hz": IF_OFFSET_HZ} if not file_wait else {}

        if file_wait:
            log.info("File-wait mode: SDR not needed. Press 'Open' and select .cf32")
        else:
            try:
                # попытка создать SDR-backend (auto/rtlsdr/hackrf/airspy/sdrplay/rsa и т.п.)
                
                self.backend = safe_make_backend(
                    BACKEND_NAME,
                    sample_rate=SAMPLE_RATE_SPS,
                    center_freq=float(CENTER_FREQ_HZ),
                    gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
                    agc=bool(ENABLE_AGC),
                    corr_ppm=int(FREQ_CORR_PPM),
                    device_args=BACKEND_ARGS,
                    # если нужен IF-сдвиг для file: пробросим так же; для SDR он игнорируется где не нужно
                    if_offset_hz=IF_OFFSET_HZ,
                    on_fail="file_wait",          # <--- ключевая строчка
                    # fallback_args={"path": ""}  # опционально переопределить поведение file-backend
                )

                # обновим Fs, если получилось
                try:
                    st = self.backend.get_status() or {}
                    self.sample_rate = float(
                        st.get("actual_sample_rate_sps",
                            getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
                    )
                except Exception:
                    pass

            except RuntimeError as e:
                # нет устройств — не падаем, уходим в режим ожидания файла
                if "устройства не найдены" in str(e):
                    log.warning("SDR not found - switching to file wait mode. Press 'Open'")
                    self.backend = None
                else:
                    # другие ошибки всё же пробрасываем
                    raise

        log.info(f"Backend sample rate (init): {self.sample_rate:.2f} Sa/s")

        # Реальный Fs: если backend ещё нет — берём SAMPLE_RATE_SPS для первичной инициализации осей
        _st = {}
        if self.backend is not None:
            try:
                _st = self.backend.get_status() or {}
            except Exception:
                _st = {}
        self.sample_rate = float(
            _st.get("actual_sample_rate_sps",
                    getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
        )
        log.info(f"Backend sample rate: {self.sample_rate:.2f} Sa/s")

        # --- State ---

        #self.sample_rate = float(getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
        log.debug(f"Backend sample rate: {self.sample_rate:.2f} Sa/s")
        
        # Печать полного статуса бэкенда в консоль
        try:
            log.info("\n=== BACKEND STATUS ===\n" + self.backend.pretty_status() + "\n======================\n")
        except Exception:
            try:
                log.info("\n=== BACKEND STATUS ===\n", self.backend.get_status(), "\n======================\n")
            except Exception:
                pass

        # Реальный Fs берём из статуса/атрибутов бэкенда
        _st = {}
        try:
            _st = self.backend.get_status() or {}
        except Exception:
            _st = {}
        self.sample_rate = float(
            _st.get("actual_sample_rate_sps",
                    getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
        )
        log.info(f"Backend sample rate: {self.sample_rate:.2f} Sa/s")
        
        self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
        self.tail_p = np.empty(0, dtype=np.float32)
        self.nco_phase = 0.0
        self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))
        max_points = int(LEVEL_HISTORY_SEC * self.sample_rate / max(VIS_DECIM, 1))
        
        
        self.rms_history = deque(maxlen=max_points)
        self.time_history = deque(maxlen=max_points)
        self.last_rms_dbm = float("-inf")
        self.sample_counter = 0
        self._stop = False
        self.last_print = 0.0
        self.full_idx = np.empty(0, dtype=np.int64)
        self.full_rms = np.empty(0, dtype=np.float32)
        self.store_max_samps = int(PULSE_STORE_SEC * self.sample_rate)
        self.samples_start_abs = 0
        self.full_samples = np.empty(0, dtype=np.complex64)
        self.in_pulse = False
        self.pulse_start_abs = None
        self.pulse_records = []
        self.pulse_records_limit = 5
        self.last_impulse_freq_hz = 0.0

        # --- NEW: flag to enable/disable updates of the second (pulse+PSK) window ---
        self.pulse_updates_enabled = True
        
        # --- NEW: Store the last processed IQ segment ---
        self.last_iq_seg = None

        
        # --- Core gate of last pulse (relative to last_iq_seg) ---
        self.last_core_gate = None
# --- Multithreading ---
        self.data_lock = threading.Lock()
        self.pulse_data_queue = queue.Queue()

        # --- UI: Figure 1 (RMS timeline) ---
        self.fig1, self.ax_lvl = plt.subplots(num="Sliding RMS — realtime", figsize=(9, 2.5))
        self.fig1.subplots_adjust(bottom=0.35, top=0.85)  # Освобождаем место для кнопок

        # Позиционируем первое окно слева
        try:
            mngr = self.fig1.canvas.manager
            mngr.window.wm_geometry("+50+50")  # x=50, y=50 от левого верхнего угла экрана
        except Exception:
            pass
        try:
            self.fig1.canvas.manager.set_window_title("Sliding RMS — realtime")
        except Exception:
            pass
        (self.ln_lvl,) = self.ax_lvl.plot([], [], lw=1.2)
        self.ax_lvl.set_xlabel("Время, с")
        self.ax_lvl.set_ylabel("RMS, dBm")
        self.ax_lvl.grid(True, alpha=0.3)
        # Кнопки выбора backend
        ax_btn_auto = self.fig1.add_axes([0.02, 0.01, 0.08, 0.06])
        self.btn_auto = Button(ax_btn_auto, "Auto")
        self.btn_auto.on_clicked(lambda event: self._on_backend_select(event, "auto"))

        ax_btn_rtl = self.fig1.add_axes([0.11, 0.01, 0.08, 0.06])
        self.btn_rtl = Button(ax_btn_rtl, "RTL")
        self.btn_rtl.on_clicked(lambda event: self._on_backend_select(event, "soapy_rtl"))

        ax_btn_hackrf = self.fig1.add_axes([0.20, 0.01, 0.08, 0.06])
        self.btn_hackrf = Button(ax_btn_hackrf, "HackRF")
        self.btn_hackrf.on_clicked(lambda event: self._on_backend_select(event, "soapy_hackrf"))

        ax_btn_airspy = self.fig1.add_axes([0.29, 0.01, 0.08, 0.06])
        self.btn_airspy = Button(ax_btn_airspy, "Airspy")
        self.btn_airspy.on_clicked(lambda event: self._on_backend_select(event, "soapy_airspy"))

        ax_btn_sdrplay = self.fig1.add_axes([0.38, 0.01, 0.08, 0.06])
        self.btn_sdrplay = Button(ax_btn_sdrplay, "SDRPlay")
        self.btn_sdrplay.on_clicked(lambda event: self._on_backend_select(event, "soapy_sdrplay"))

        ax_btn_rsa306 = self.fig1.add_axes([0.47, 0.01, 0.08, 0.06])
        self.btn_rsa306 = Button(ax_btn_rsa306, "RSA306")
        self.btn_rsa306.on_clicked(lambda event: self._on_backend_select(event, "rsa306"))

        ax_btn_file = self.fig1.add_axes([0.56, 0.01, 0.08, 0.06])
        self.btn_file_mode = Button(ax_btn_file, "File")
        self.btn_file_mode.on_clicked(lambda event: self._on_backend_select(event, "file"))

        # Кнопка выбора файла
        ax_button_file = self.fig1.add_axes([0.73, 0.01, 0.12, 0.06])
        self.btn_file = Button(ax_button_file, "Открыть")
        self.btn_file.on_clicked(self._on_file_select)

        ax_button_exit = self.fig1.add_axes([0.86, 0.01, 0.12, 0.06])
        self.btn_exit = Button(ax_button_exit, "Выход")
        self.btn_exit.on_clicked(self._on_exit)

        # Сохраняем все кнопки backend для обновления цвета
        self.backend_buttons = {
            "auto": self.btn_auto,
            "soapy_rtl": self.btn_rtl,
            "soapy_hackrf": self.btn_hackrf,
            "soapy_airspy": self.btn_airspy,
            "soapy_sdrplay": self.btn_sdrplay,
            "rsa306": self.btn_rsa306,
            "file": self.btn_file_mode
        }

        # Подсвечиваем текущий backend
        self._update_backend_buttons(BACKEND_NAME)

        # --- UI: Figure 2 (Pulse + PSK + FM) ---
        self.fig2, (self.ax_pulse, self.ax_phase, self.ax_fm) = plt.subplots(
            3, 1, figsize=(14, 8.6), height_ratios=[1, 1, 1]  # sharex убран
        )
        # после создания:

        self.ax_fm.sharex(self.ax_phase)

        # Позиционируем второе окно справа от первого
        try:
            mngr2 = self.fig2.canvas.manager
            mngr2.window.wm_geometry("+750+50")  # x=750 (справа от первого), y=50
        except Exception:
            pass

        try:
            self.fig2.canvas.manager.set_window_title("Pulse window — RMS (top) + Δf (bottom)")
        except Exception:
            pass
        
        (self.ln_pulse,) = self.ax_pulse.plot([], [], lw=1.4)
        self.ax_pulse.set_ylabel("RMS, dBm")
        self.ax_pulse.grid(True, alpha=0.3)
        self.ax_pulse.set_xlabel("Время, мс (RMS окно)")

        # Фаза (середина)
        (self.ln_phase,) = self.ax_phase.plot([], [], lw=1.4)
        self.ax_phase.set_ylabel("Фаза, rad")
        self.ax_phase.grid(True, alpha=0.3)
        self.ax_phase.set_ylim(-PSK_YLIMIT_RAD, PSK_YLIMIT_RAD)

        # FM (низ)
        (self.ln_fm,) = self.ax_fm.plot([], [], lw=1.4)
        self.ax_fm.set_xlabel("Время, мс (0 = старт импульса)")
        self.ax_fm.set_ylabel("FM, Hz")
        self.ax_fm.grid(True, alpha=0.3)

        self.fig2.subplots_adjust(hspace=0.5, top=0.92, bottom=0.15)
        self.fig2.suptitle(f"Импульс: RMS + Фаза (±{PSK_YLIMIT_RAD:.0f} rad) + FM(Hz)")

        # --- Button: Save IQ (writes CF32, bit-for-bit) ---
        ax_button_save = self.fig2.add_axes([0.70, 0.01, 0.15, 0.06])
        # --- Button: Spectrum (static) ---
        ax_button_spec = self.fig2.add_axes([0.54, 0.01, 0.15, 0.06])
        self.btn_spec = Button(ax_button_spec, "Спектр")
        self.btn_spec.on_clicked(self._on_show_spectrum)

        # Кнопка: Статус SDR (таблица)
        ax_button_stat = self.fig2.add_axes([0.22, 0.01, 0.15, 0.06])
        self.btn_stat = Button(ax_button_stat, "Статус SDR")
        self.btn_stat.on_clicked(self._on_show_sdr_status)

        # --- Button: Params (snapshot) ---
        ax_button_params = self.fig2.add_axes([0.38, 0.01, 0.15, 0.06])
        self.btn_params = Button(ax_button_params, "Параметры")
        self.btn_params.on_clicked(self._on_show_params)
        self.btn_save = Button(ax_button_save, "Save IQ")
        self.btn_save.on_clicked(self._on_save_iq)

        # --- NEW: Stop/Start toggle button for pulse+PSK window updates ---
        ax_button_stop = self.fig2.add_axes([0.86, 0.01, 0.12, 0.06])
        self.btn_stop = Button(ax_button_stop, "Стоп")
        self.btn_stop.on_clicked(self._on_toggle_pulse_updates)

        # --- Button: Message decoder ---
        ax_button_msg = self.fig2.add_axes([0.06, 0.01, 0.15, 0.06])
        self.btn_msg = Button(ax_button_msg, "Сообщение")
        self.btn_msg.on_clicked(self._on_show_message)

    # ---- Message decoder window ----
    def _on_show_message(self, _event):
        """Показывает декодированное сообщение EPIRB/ELT в отдельном окне."""
        
        hex_msg = getattr(self, "last_msg_hex", None)

        if not hex_msg or hex_msg == "None":
            # Показываем сообщение об отсутствии данных
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
            # Преобразуем HEX в биты
            bits = hex_to_bits(hex_msg)
            if len(bits) != 144:
                # Дополняем или обрезаем до 144 бит
                bits = (bits + [0]*144)[:144]

            # Заголовки таблицы
            headers = ["Binary Range", "Binary Content", "Field Name", "Decoded Value"]

            # Получаем данные для таблицы
            rows = build_table_rows(bits)

            # Создаем окно Matplotlib с таблицей
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('off')

            # Заголовок окна
            fig.suptitle(f"EPIRB/ELT Beacon Parameters Decoder\nHEX: {hex_msg}",
                        fontsize=11, fontweight='bold')

            # Создаем таблицу
            tbl = ax.table(cellText=[headers] + rows,
                          loc='center',
                          cellLoc='left',
                          colWidths=[0.12, 0.25, 0.28, 0.35])

            # Настройка стилей таблицы
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)

            # Стиль заголовка
            for i in range(len(headers)):
                cell = tbl[(0, i)]
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.05)

            # Чередующиеся цвета строк
            for i in range(1, len(rows) + 1):
                for j in range(len(headers)):
                    cell = tbl[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#f0f0f0')
                    cell.set_height(0.04)

                    # Выделяем важные поля дополнительным цветом
                    if j == 2:  # колонка Field Name
                        field_name = rows[i-1][2].lower()
                        if any(key in field_name for key in ['country', 'mmsi', 'lat', 'lon', 'id']):
                            cell.set_facecolor('#e8f4ff')

            # Настройка окна
            try:
                fig.canvas.manager.set_window_title("EPIRB/ELT Beacon Decoder")
            except Exception:
                pass

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

        except Exception as e:
            # Показываем ошибку
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

    # ---- Params window (snapshot, table) ----
    def _on_show_params(self, _event):
        
        m = getattr(self, "last_phase_metrics", None)
        hex_msg = getattr(self, "last_msg_hex", None)

        if not m:
            fig, ax = plt.subplots(figsize=(8, 3))
            try:
                fig.canvas.manager.set_window_title("Phase Parameters")
            except Exception:
                pass
            ax.axis('off')
            ax.text(0.5, 0.5, "No parameters yet — trigger a PSK pulse first.",
                    ha='center', va='center', fontsize=12)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            return

        # Заголовки таблицы
        headers = ["Parameter", "Value"]

        # Строки данных
        rows = [
            ["Target Signal (Hz)",     f"{m.get('Target Signal (Hz)', float('nan')):.3f}"],
            ["Frequency Offset (Hz)",  f"{m.get('Frequency Offset (Hz)', float('nan')):.3f}"],
            ["Message Duration (ms)",  f"{m.get('Message Duration (ms)', float('nan')):.3f}"],
            ["Carrier Duration (ms)",  f"{m.get('Carrier Duration (ms)', float('nan')):.3f}"],
            ["Pos (rad)",              f"{m.get('Pos (rad)', float('nan')):.3f}"],
            ["Neg (rad)",              f"{m.get('Neg (rad)', float('nan')):.3f}"],
            ["Rise (μs)",              f"{m.get('Rise (μs)', float('nan')):.1f}"],
            ["Fall (μs)",              f"{m.get('Fall (μs)', float('nan')):.1f}"],
            ["Asymmetry (%)",          f"{m.get('Asymmetry (%)', float('nan')):.3f}"],
            ["Fmod (Hz)",              f"{m.get('Fmod (Hz)', float('nan')):.3f}"],
            ["Power (RMS, dBm)",       f"{m.get('Power (RMS, dBm)', float('nan')):.2f}"],
            ["HEX",                    str(hex_msg) if hex_msg is not None else ""]
        ]

        # Создаем окно
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        # Создаем таблицу
        tbl = ax.table(cellText=[headers] + rows,
                      loc='center',
                      cellLoc='left',
                      colWidths=[0.5, 0.5])

        # Настройка стилей таблицы
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        # Стиль заголовка
        for i in range(len(headers)):
            cell = tbl[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.06)

        # Чередующиеся цвета строк
        for i in range(1, len(rows) + 1):
            for j in range(len(headers)):
                cell = tbl[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                cell.set_height(0.05)

        # Заголовок окна
        fig.suptitle("Phase Parameters (snapshot)", fontsize=12, fontweight='bold')

        # Настройка окна
        try:
            fig.canvas.manager.set_window_title("Phase Parameters")
        except Exception:
            pass

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        
    def _on_show_sdr_status(self, _event):

        if self.backend is None:
            
            fig, ax = plt.subplots(figsize=(6, 2))
            try: fig.canvas.manager.set_window_title("SDR Status")
            except Exception: pass
            ax.axis('off')
            ax.text(0.5, 0.5, "Backend ещё не выбран.\nНажмите «Открыть» и выберите файл.",
                    ha='center', va='center')
            plt.tight_layout(); plt.show(block=False); plt.pause(0.1)
            return


        try:
            st = self.backend.get_status()
        except Exception:
            st = {}
        try:
            pretty = self.backend.pretty_status()
        except Exception:
            pretty = None

        preferred = [
            "backend", "driver",
            "requested_sample_rate_sps", "actual_sample_rate_sps",
            "requested_center_freq_hz", "actual_center_freq_hz",
            "bandwidth_hz",
            "hw_sample_rate_sps", "decim",
            "agc_on", "overall_gain_db", "stage_gains_db",
            "corr_ppm", "calib_offset_db",
            "file_path", "if_offset_hz", "mix_shift_hz", "eof",
            "device_info", "ref_level_dbm",
        ]
        keys, seen = [], set()
        for k in preferred:
            if k in st and k not in seen:
                keys.append(k); seen.add(k)
        for k in sorted(st.keys()):
            if k not in seen:
                keys.append(k); seen.add(k)

        # Заголовки таблицы
        headers = ["Key", "Value"]

        # Строки данных
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

        # Создаем окно
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # Создаем таблицу
        tbl = ax.table(cellText=[headers] + rows,
                      loc='center',
                      cellLoc='left',
                      colWidths=[0.4, 0.6])

        # Настройка стилей таблицы
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        # Стиль заголовка
        for i in range(len(headers)):
            cell = tbl[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.05)

        # Чередующиеся цвета строк
        for i in range(1, len(rows) + 1):
            for j in range(len(headers)):
                cell = tbl[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                cell.set_height(0.04)

                # Выделяем важные поля
                if j == 0:  # колонка Key
                    key_name = rows[i-1][0].lower()
                    if any(word in key_name for word in ['actual', 'file_path', 'backend', 'driver']):
                        cell.set_facecolor('#e8f4ff')

        # Заголовок окна
        title = "SDR Status (snapshot)"
        if pretty:
            title += " — см. также консольный вывод"
        fig.suptitle(title, fontsize=12, fontweight='bold')

        # Настройка окна
        try:
            fig.canvas.manager.set_window_title("SDR Status")
        except Exception:
            pass

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def _measure_frequency_phase(self, iq_seg: np.ndarray, fs: float) -> float:
        """
        Измеряет частоту по среднему приращению фазы.
        Идея: f = (1 / 2*pi) * d(phase)/dt
        """
        if iq_seg.size < 2:
            return 0.0
        # A. “Lag-1 mean”
        # Вычисляем разность фаз между соседними сэмплами
        # np.angle(x_2 / x_1) = angle(x_2) - angle(x_1)
        phase_diff = np.angle(iq_seg[1:] * np.conj(iq_seg[:-1]))
        freq_hz = phase_diff.mean() * fs / (2*np.pi)
        
        # B. “Vector-sum” (рекомендуется, без деления на N) 
        # дает большую погрешность частоты чем A.
        #z = iq_seg[1:] * np.conj(iq_seg[:-1])
        #freq_hz = np.angle(z.sum()) * fs / (2*np.pi)
        
        return freq_hz

    """
    def _measure_frequency_fft(self, iq_seg: np.ndarray, fs: float) -> float:
        #Измеряет доминирующую частоту в сегменте IQ-данных с помощью БПФ.
        if iq_seg.size == 0:
            return 0.0

        n = iq_seg.size
        # Выполняем БПФ
        fft_result = np.fft.fft(iq_seg)
        # Находим частотную ось
        freqs = np.fft.fftfreq(n, d=1/fs)
        # Находим индекс максимальной амплитуды
        peak_idx = np.argmax(np.abs(fft_result))
        # Возвращаем частоту в Гц
        return freqs[peak_idx]
    """

    # ---------- UI callbacks ----------
    def _on_backend_select(self, _event, backend_name):
        """Переключает backend SDR."""
        log.info(f"\n[INFO] Переключение на backend: {backend_name}")

        # Если выбран file, сначала нужно выбрать файл
        if backend_name == "file":
            # Открываем диалог выбора файла
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Выберите CF32 файл",
                filetypes=[(
                    "CF32 files", "*.cf32"),
                    ("All files", "*.*")
                ],
                initialdir=str(ROOT / "captures")
            )
            root.destroy()

            if not file_path:
                log.info("File selection cancelled")
                return

            device_args = {"path": file_path}
        else:
            device_args = None

        # Останавливаем текущий backend
        try:
            self.stop()
            time.sleep(0.1)
        except Exception:
            pass

        # Создаём новый backend
        try:
            extra_kwargs = {"if_offset_hz": IF_OFFSET_HZ} if (backend_name == "file") else {}

            self.backend = safe_make_backend(
                backend_name,
                sample_rate=SAMPLE_RATE_SPS,
                center_freq=float(CENTER_FREQ_HZ),
                gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
                agc=bool(ENABLE_AGC),
                corr_ppm=int(FREQ_CORR_PPM),
                device_args=device_args,
                **extra_kwargs,
            )

            # Обновляем параметры
            try:
                _st = self.backend.get_status() or {}
                self.sample_rate = float(
                    _st.get("actual_sample_rate_sps",
                            getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
                )
                log.info(f"Backend {backend_name} activated, sample rate: {self.sample_rate:.2f} Sa/s")

                # Печать статуса
                log.info("\n=== BACKEND STATUS ===\n" + self.backend.pretty_status() + "\n======================\n")
            except Exception:
                pass

            # Обновляем параметры окна RMS
            self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
            self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))

            # Сбрасываем состояние
            self._stop = False
            self.sample_counter = 0
            self.samples_start_abs = 0
            self.full_samples = np.empty(0, dtype=np.complex64)
            self.full_idx = np.empty(0, dtype=np.int64)
            self.full_rms = np.empty(0, dtype=np.float32)
            self.tail_p = np.empty(0, dtype=np.float32)
            self.last_iq_seg = None
            self.last_core_gate = None
            self.in_pulse = False
            self.pulse_start_abs = None
            self.nco_phase = 0.0
            self.last_impulse_freq_hz = 0.0

            # Очищаем историю
            self.rms_history.clear()
            self.time_history.clear()

            # Очищаем очередь импульсов
            while not self.pulse_data_queue.empty():
                try:
                    self.pulse_data_queue.get_nowait()
                except Exception:
                    pass

            # Перезапускаем поток чтения
            if hasattr(self, 'reader_thread'):
                try:
                    self.reader_thread.join(timeout=0.5)
                except Exception:
                    pass

            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            # Обновляем цвета кнопок
            self._update_backend_buttons(backend_name)

            log.info(f"Backend {backend_name} successfully activated")

        except Exception as e:
            log.error(f"Backend {backend_name} activation error: {e}")
            # Попытаемся вернуться к исходному backend
            self._restore_original_backend()

    def _update_backend_buttons(self, active_backend):
        """Обновляет цвета кнопок backend."""
        for backend_name, button in self.backend_buttons.items():
            if backend_name == active_backend:
                button.color = 'lightgreen'
                button.hovercolor = 'lightgreen'
            else:
                button.color = '0.85'
                button.hovercolor = '0.95'

    def _restore_original_backend(self):
        """Восстанавливает исходный backend при ошибке."""
        try:
            extra_kwargs = {"if_offset_hz": IF_OFFSET_HZ} if (BACKEND_NAME == "file") else {}
            self.backend = safe_make_backend(
                BACKEND_NAME,
                sample_rate=SAMPLE_RATE_SPS,
                center_freq=float(CENTER_FREQ_HZ),
                gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
                agc=bool(ENABLE_AGC),
                corr_ppm=int(FREQ_CORR_PPM),
                device_args=BACKEND_ARGS,
                **extra_kwargs,
            )
            _st = self.backend.get_status() or {}
            self.sample_rate = float(
                _st.get("actual_sample_rate_sps",
                        getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
            )
            self._stop = False
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()
        except Exception:
            pass

    def _on_file_select(self, _event):
        """Открывает диалог выбора CF32 файла и перезагружает backend."""
        log.info("\n[INFO] Нажата кнопка Файл")

        # Создаём скрытое tk окно для диалога
        root = tk.Tk()
        root.withdraw()

        # Открываем диалог выбора файла
        file_path = filedialog.askopenfilename(
            title="Выберите CF32 файл",
            filetypes=[
                ("CF32 files", "*.cf32"),
                ("All files", "*.*")
            ],
            initialdir=str(ROOT / "captures")
        )

        root.destroy()

        if not file_path:
            log.info("File selection cancelled")
            return

        log.info(f"Selected file: {file_path}")

        # Останавливаем текущий backend
        try:
            self.stop()
            time.sleep(0.1)  # Даём время на остановку
        except Exception:
            pass

        # Пересоздаём backend с новым файлом
        try:
       
            extra_kwargs = {"if_offset_hz": IF_OFFSET_HZ}

            self.backend = safe_make_backend(
                "file",
                sample_rate=SAMPLE_RATE_SPS,
                center_freq=float(CENTER_FREQ_HZ),
                gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
                agc=bool(ENABLE_AGC),
                corr_ppm=int(FREQ_CORR_PPM),
                device_args={"path": file_path},
                **extra_kwargs,
            )

            # Обновляем параметры
            try:
                _st = self.backend.get_status() or {}
                self.sample_rate = float(
                    _st.get("actual_sample_rate_sps",
                            getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
                )
                log.info(f"New file loaded, sample rate: {self.sample_rate:.2f} Sa/s")

                # Печать статуса
                log.info("\n=== BACKEND STATUS ===\n" + self.backend.pretty_status() + "\n======================\n")
            except Exception:
                pass

            # Обновляем параметры окна RMS после смены sample_rate
            self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
            self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))

            # Сбрасываем состояние
            self._stop = False
            self.sample_counter = 0
            self.samples_start_abs = 0
            self.full_samples = np.empty(0, dtype=np.complex64)
            self.full_idx = np.empty(0, dtype=np.int64)
            self.full_rms = np.empty(0, dtype=np.float32)
            self.tail_p = np.empty(0, dtype=np.float32)
            self.last_iq_seg = None
            self.last_core_gate = None
            self.in_pulse = False
            self.pulse_start_abs = None
            self.nco_phase = 0.0
            self.last_impulse_freq_hz = 0.0

            # Очищаем историю
            self.rms_history.clear()
            self.time_history.clear()

            # Очищаем очередь импульсов
            while not self.pulse_data_queue.empty():
                try:
                    self.pulse_data_queue.get_nowait()
                except Exception:
                    pass

            # Перезапускаем поток чтения
            if hasattr(self, 'reader_thread'):
                try:
                    self.reader_thread.join(timeout=0.5)
                except Exception:
                    pass

            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            log.info(f"File {file_path} successfully loaded and analysis started")

        except Exception as e:
            log.error(f"File loading error: {e}")
            # Попытаемся вернуться к исходному backend
            try:
                extra_kwargs = {"if_offset_hz": IF_OFFSET_HZ} if (BACKEND_NAME == "file") else {}
                self.backend = safe_make_backend(
                    BACKEND_NAME,
                    sample_rate=SAMPLE_RATE_SPS,
                    center_freq=float(CENTER_FREQ_HZ),
                    gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
                    agc=bool(ENABLE_AGC),
                    corr_ppm=int(FREQ_CORR_PPM),
                    device_args=BACKEND_ARGS,
                    **extra_kwargs,
                )
                _st = self.backend.get_status() or {}
                self.sample_rate = float(
                    _st.get("actual_sample_rate_sps",
                            getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS))
                )
                self._stop = False
                self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self.reader_thread.start()
            except Exception:
                pass

    def _on_exit(self, _event):
        log.info("\n[INFO] Нажата кнопка Выход")
        self.stop()
        plt.close(self.fig1)
        plt.close(self.fig2)



    def _on_show_spectrum(self, _event):
        """
        Рисует статичный спектр «ядра импульса», определённого по RMS-порогам:
        i_start_core — первое превышение порога; i_end_core — возврат ниже порога.
        Используем gate_samples=(g0,g1), где g0,g1 заданы относительно last_iq_seg.
        """
        with self.data_lock:
            seg = None if self.last_iq_seg is None else self.last_iq_seg.copy()
            gate = None if self.last_core_gate is None else tuple(self.last_core_gate)
        if seg is None or seg.size < 8:
            log.warning("No segment for spectrum. Pulse must be detected first")
            return
        if not gate or gate[1] - gate[0] < 8:
            log.warning("No valid pulse core (gate). Spectrum not built")
            return

        self._plot_fft_sector(
            seg,
            FFT_N=65536,
            avg=4,
            window="hann",
            sector_center_hz=0.0,
            sector_half_width_hz=50_000,
            y_ref_db=0.0,
            gate_samples=gate,
            remove_dc=True,
            normalize="dBc",
            y_lim=None,
            show_mask=True
        )

    def _on_toggle_pulse_updates(self, _event):
        """Toggle updates for the second (pulse+PSK) window without stopping SDR.
        Keeps the last plotted curves; when stopped, incoming pulse data is discarded.
        """
        self.pulse_updates_enabled = not self.pulse_updates_enabled
        try:
            self.btn_stop.label.set_text("Старт" if not self.pulse_updates_enabled else "Стоп")
        except Exception:
            pass
        state = "выключено" if not self.pulse_updates_enabled else "включено"
        log.info(f"Updating pulse/PSK window: {state}")

    def _plot_fft_sector(self, 
        iq_data,
        FFT_N=65536,
        avg=4,
        window="hann",               # "hann" | "rect"
        sector_center_hz=0.0,
        sector_half_width_hz=50_000,
        y_ref_db=0.0,
        *,
        gate_samples=None,           # (i0, i1): взять только этот диапазон сэмплов (до FFT/усреднения)
        remove_dc=True,              # вычесть комплексное среднее перед FFT
        normalize="dBc",             # "dBc" (пик=0) | "dB" (абс. значения)
        y_lim=None,                  # например (-90, 5) для dBc
        show_mask=True               # рисовать лестницу C/S T.001 в dBc
    ):
            """
            Рисует сектор амплитудного спектра из CF32 «как есть», БЕЗ учёта IF_OFFSET_HZ.
            Ось частот — сырая baseband (-Fs/2..+Fs/2).

            Параметры:
              gate_samples:   кортеж (i0, i1) – «окно анализа» по сэмплам, чтобы не включать чистую несущую до PSK.
              remove_dc:      вычитает среднее (компенсация DC/IQ-несимметрии).
              normalize:      "dBc": график нормируется к максимуму в секторе (пик = 0 dBc).
                              "dB" : абсолютная шкала (учитывает y_ref_db).
              y_lim:          пределы по Y (tuple или None).
              show_mask:      рисовать маску C/S T.001 (в dBc относительно пика).

            Возвращает dict:
              f_peak_Hz, p_peak_dB (абсолютно), p_peak_dBc (=0 при normalize="dBc"),
              n_blocks, FFT_N, df_Hz
            """

            # --- частота дискретизации
            SR = float(self.sample_rate)

            # --- вход как complex64
            seg = np.asarray(iq_data, dtype=np.complex64)

            # --- гейтинг по времени (до усреднения и FFT)
            if gate_samples is not None:
                i0, i1 = map(int, gate_samples)
                i0 = max(0, i0)
                i1 = min(len(seg), i1)
                seg = seg[i0:i1]

            if seg.size < 16:
                log.debug("Segment too short for FFT")
                return

            if remove_dc:
                seg = seg - np.mean(seg)

            # --- окно (float32)
            win = (np.hanning(FFT_N) if window == "hann" else np.ones(FFT_N)).astype(np.float32)

            # --- блоки для усреднения
            n_possible = seg.size // FFT_N
            if n_possible == 0:
                seg = np.pad(seg, (0, FFT_N - seg.size))
                n_possible = 1
            n_blocks = int(max(1, min(int(avg), n_possible)))

            # --- FFT и накопление мощности
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

            # --- ось частот
            f_axis = np.fft.fftshift(np.fft.fftfreq(FFT_N, d=1.0 / SR))

            # --- сектор
            f0 = float(sector_center_hz)
            hw = abs(float(sector_half_width_hz))
            mask = (f_axis >= f0 - hw) & (f_axis <= f0 + hw)
            if not np.any(mask):
                log.debug("Sector out of frequency axis range. Check sector_center_hz/sector_half_width_hz")
                return

            f_sec = f_axis[mask]
            psd_sec_db = psd_db[mask]

            # --- пик (абсолютно, до нормировки)
            i_pk = int(np.argmax(psd_sec_db))
            f_pk = float(f_sec[i_pk])
            p_pk_abs_db = float(psd_sec_db[i_pk])

            # --- нормировка
            if str(normalize).lower() == "dbc":
                psd_plot = psd_sec_db - p_pk_abs_db
                ylabel = "Амплитуда, dBc"
                p_pk_dBc = 0.0
                mask_offset = 0.0
            else:
                psd_plot = psd_sec_db
                ylabel = "Амплитуда, dB"
                p_pk_dBc = float(p_pk_abs_db - p_pk_abs_db)  # 0.0, для совместимости
                mask_offset = p_pk_abs_db  # маска будет «прилипать» к максимуму

            # --- график
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(f_sec / 1e3, psd_plot, lw=1.5, label="Спектр")
            ax.axvline(0.0, linestyle='--', alpha=0.5)

            # --- маска C/S T.001 (в dBc относительно несущей)
            def _plot_cs_t001_mask(ax, p0, half_width_hz):
                # сегменты справа от нуля (влево рисуем зеркально)
                segments = [
                    (0.0,   3e3,   0.0),   # 0…3 кГц: 0→-20 dBc (верхняя ступень у несущей)
                    (3e3,   7e3, -20.0),   # 3…7 кГц: -20 dBc
                    (7e3,  12e3, -30.0),   # 7…12 кГц: -30 dBc
                    (12e3, 24e3, -35.0),   # 12…24 кГц: -35 dBc
                    (24e3, half_width_hz, -40.0),  # дальше -40 dBc
                ]
                for sgn in (-1.0, +1.0):
                    for f1, f2, lvl in segments:
                        x1 = sgn * (f1 / 1e3)
                        x2 = sgn * (f2 / 1e3)
                        y  = p0 + lvl
                        ax.plot([x1, x2], [y, y], linestyle='--', linewidth=1.2, alpha=0.9)
                for fb in [3e3, 7e3, 12e3, 24e3]:
                    ax.axvline(+fb/1e3, linestyle=':', alpha=0.6)
                    ax.axvline(-fb/1e3, linestyle=':', alpha=0.6)

            if show_mask:
                _plot_cs_t001_mask(ax, mask_offset, hw)

            # помечаем найденный пик
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
            #plt.show()
            plt.show(block=False)
            plt.pause(0.1)  # дать GUI дорисоваться
    

            return {
                "f_peak_Hz": f_pk,
                "p_peak_dB": p_pk_abs_db,             # абсолютный пик (dB)
                "p_peak_dBc": p_pk_dBc,               # при dBc всегда 0.0
                "n_blocks": int(n_blocks),
                "FFT_N": int(FFT_N),
                "df_Hz": float(SR / FFT_N),
                "normalize": "dBc" if str(normalize).lower() == "dbc" else "dB",
                "gate_samples": tuple(gate_samples) if gate_samples is not None else None,
            }
    def _on_save_iq(self, _event):
        """
        [MODIFIED] Now saves the last IQ segment that was processed for the pulse/PSK view.
        """
        with self.data_lock:
            # Save the last processed IQ segment instead of the full buffer.
            data = self.last_iq_seg.copy() if self.last_iq_seg is not None else np.empty(0)
        
        if data.size == 0:
            log.warning("No data to save. Pulse must be detected first")
            return
        
        fname = time.strftime(file_path)
        data.astype(np.complex64).tofile(fname)  # CF32 bit-for-bit
        log.info(f"Pulse IQ buffer saved: {fname} ({data.size} samples)")

    # ---------- SDR read & process ----------
    def _read_block(self, nsamps: int) -> np.ndarray:
        return self.backend.read(nsamps)
    def _reader_loop(self):
        try:
            while not self._stop:
                try:
                    samples = self._read_block(READ_CHUNK)
                    if samples.size == 0:
                        if BACKEND_NAME == "file":
                            log.info("EOF (file): stopping")
                            self.stop()
                            return  # или break
                        else:
                            # SDR (RTL/HackRF/Airspy/…): пустой блок = андерфлоу → просто пропустим
                            time.sleep(0.001)
                            continue

                except RuntimeError as e:
                    log.info(f"\n[WARN] readStream: {e}")
                    time.sleep(0.001)
                    continue
                try:

                    self._process_samples(samples)

                except Exception as e:

                    log.info(f"\n[WARN] Обработка блока пропущена: {e}")

                    # продолжаем цикл, не останавливаем поток
        except Exception as e:
            log.info(f"\nОШИБКА чтения: {e}")
            self._stop = True

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

    def _process_samples(self, samples: np.ndarray):
        now = time.time()
        base_idx = self.sample_counter
        x = samples.copy()

        # Optional baseband shift
        if BB_SHIFT_ENABLE and abs(BB_SHIFT_HZ) > 0:
            n = np.arange(samples.size, dtype=np.float64)
            mixer = np.exp(1j * (self.nco_phase + self.nco_k * n)).astype(np.complex64)
            x *= mixer
            self.nco_phase = float((self.nco_phase + self.nco_k * samples.size) % (2.0 * np.pi))

        with self.data_lock:
            self._append_samples(x)

        p_block = (np.abs(x) ** 2)

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
            #rms_dbm_vec = db10(P_win) + DBM_OFFSET_DB
            rms_dbm_vec = db10(P_win) + DBM_OFFSET_DB + self.backend.get_calib_offset_db()

            idx_end = p_cont_start_idx + (self.win_samps - 1) + np.arange(rms_dbm_vec.size, dtype=np.int64)

            with self.data_lock:
                self.full_idx = np.concatenate((self.full_idx, idx_end))
                self.full_rms = np.concatenate((self.full_rms, rms_dbm_vec.astype(np.float32)))
                max_idx_span = self.store_max_samps
                if self.full_idx.size > 0:
                    newest_idx = self.full_idx[-1]
                    m_keep = (self.full_idx >= (newest_idx - max_idx_span))
                    self.full_idx = self.full_idx[m_keep]
                    self.full_rms = self.full_rms[m_keep]

            on = rms_dbm_vec >= PULSE_THRESH_DBM
            trans = np.diff(on.astype(np.int8), prepend=on[0])
            start_pos = np.where(trans == 1)[0]
            end_pos = np.where(trans == -1)[0] - 1

            if self.in_pulse:
                start_abs = self.pulse_start_abs
            else:
                start_abs = None

            s_idx = 0
            e_idx = 0
            while True:
                if start_abs is None:
                    if s_idx < start_pos.size:
                        start_abs = int(idx_end[start_pos[s_idx]])
                        s_idx += 1
                    else:
                        break
                found_end = None
                while e_idx < end_pos.size:
                    cand_end_abs = int(idx_end[end_pos[e_idx]])
                    e_idx += 1
                    if cand_end_abs >= start_abs:
                        found_end = cand_end_abs
                        break

                if found_end is None:
                    self.in_pulse = True
                    self.pulse_start_abs = start_abs
                    break

                self.in_pulse = False
                self.pulse_start_abs = None

                duration_samps = max(1, found_end - start_abs + 1)
                win_len = int(round(duration_samps * 1.5))
                pre = int(round(duration_samps * 0.25))
                win_start = max(0, start_abs - pre)
                win_end = win_start + win_len

                # --- NEW: логируем длительность импульса при обнаружении ---
                try:
                    dur_ms = 1000.0 * duration_samps / float(self.sample_rate)
                    if DEBUG_IMPULSE_LOG:
                        log.info(f"Pulse detected: start={start_abs}, length: {dur_ms:.1f}ms")
                except Exception:
                    pass

                with self.data_lock:
                    m = (self.full_idx >= win_start) & (self.full_idx <= win_end)
                    seg_start_rel = win_start - self.samples_start_abs
                    seg_end_rel = win_end - self.samples_start_abs
                    if np.any(m) or (seg_end_rel > 0 and seg_start_rel < self.full_samples.size):
                        pulse_data = {
                            "xs_ms": None,
                            "ys": None,
                            "xs_fm_ms": None,
                            "phase_rad": None,
                            "pulse_title": None,
                            "title": None
                        }
                        if np.any(m):
                            t0_ms = 1000.0 * (start_abs / float(self.sample_rate))
                            xs_ms = 1000.0 * (self.full_idx[m] / float(self.sample_rate)) - t0_ms
                            ys = self.full_rms[m]
                            pulse_data["xs_ms"] = xs_ms
                            pulse_data["ys"] = ys

                            dur_ms = 1000.0 * duration_samps / float(self.sample_rate)
                            pulse_data["pulse_title"] = f"Импульс: длит.≈ {dur_ms:.1f} мс, порог {PULSE_THRESH_DBM:.1f} dBm, окно 1.5×"

                            # --- NEW: финальный лог для контроля (совпадает с web-стилем) ---
                            try:
                                if DEBUG_IMPULSE_LOG:
                                    log.info(f"Pulse processed: {dur_ms:.1f}ms")
                            except Exception:
                                pass

                        if seg_end_rel > 0 and seg_start_rel < self.full_samples.size:
                            seg_start_rel = max(0, seg_start_rel)
                            seg_end_rel = min(self.full_samples.size, seg_end_rel)
                            
                            seg = self.full_samples[seg_start_rel:seg_end_rel].astype(np.complex64, copy=False)

                            # --- NEW: Store the segment for the Save IQ button ---
                            self.last_iq_seg = seg.copy()



                            # --- Save core gate indices for spectrum (based on RMS threshold crossings) ---
                            try:
                                g0 = max(0, int(start_abs - win_start))
                                g1 = int(found_end - win_start + 1)  # end-exclusive
                                g1 = min(g1, int(seg.size))
                                if g1 - g0 < 8:
                                    g1 = min(int(seg.size), g0 + 8)
                                self.last_core_gate = (int(g0), int(g1))
                            except Exception:
                                self.last_core_gate = None
                            # --- Начало изменений для измерения частоты ---
                            
                            # По умолчанию работаем с целым сегментом
                            freq_seg = seg  # <— ДОБАВЬ ЭТУ СТРОКУ

                            # 1 мс на края (но не больше 1/4 длительности импульса — чтобы не «съесть» короткие)
                            trim_samps = int(self.sample_rate * 1e-3)
                            duration_samps = max(1, found_end - start_abs + 1)
                            trim_samps = min(trim_samps, duration_samps // 4)

                            freq_start_abs = start_abs + trim_samps
                            freq_end_abs   = found_end - trim_samps

                            freq_seg_start_rel = max(0, freq_start_abs - self.samples_start_abs)
                            freq_seg_end_rel   = min(self.full_samples.size, freq_end_abs - self.samples_start_abs)

                            # Если обрезанный отрезок валиден — возьмем его; иначе останется весь seg
                            if freq_seg_start_rel < freq_seg_end_rel and (freq_seg_end_rel - freq_seg_start_rel) >= 8:
                                freq_seg = self.full_samples[freq_seg_start_rel:freq_seg_end_rel].astype(np.complex64, copy=False)

                            # Измеряем частоту (если совсем коротко — просто не обновим значение)
                            if freq_seg.size >= 8:
                                self.last_impulse_freq_hz = self._measure_frequency_phase(freq_seg, self.sample_rate)

                            # --- Конец изменений ---

                            # Оригинальный сегмент для  PSK-демодуляции
                            if seg.size >= 8:
                                t0_offset_ms = 0.0
                                _res = process_psk_impulse(
                                    iq_seg=freq_seg,
                                    fs=self.sample_rate,
                                    baseline_ms=PSK_BASELINE_MS,
                                    t0_offset_ms=t0_offset_ms,
                                    use_lpf_decim=True,
                                    remove_slope=True,  # включи True, если хочешь выровнять фазу по горизонтали
                                    
                                )

                                pulse_data["xs_fm_ms"] = _res.get("xs_ms")
                                pulse_data["phase_rad"]  = _res.get("phase_rad")
                                pulse_data["title"] = _res.get("title")
                                
                                # --- FM discriminator на том же freq_seg ---
                                fm_out = fm_discriminator(
                                    iq=freq_seg,
                                    fs=self.sample_rate,
                                    pre_lpf_hz=50_000,   # антиалиас перед децимацией (под PSK-полосу)
                                    decim=4,             # до ~250 кС/с при Fs=1 МС/с; ось времени в мс совпадает по нулю
                                    smooth_hz=2_000,     # лёгкое сглаживание
                                    detrend=True,        # убрать наклон (CFO/дрейф)
                                    center=True,         # центрировать вокруг 0
                                    fir_taps=127,
                                )
                                pulse_data["fm_xs_ms"] = fm_out["xs_ms"]       # время (мс, от 0)
                                pulse_data["fm_hz"]    = fm_out["freq_hz"]     # мгновенная частота, Гц

                                
                                
                                
                                try:
                                    msg_hex, phase_res, edges= phase_demod_psk_msg_safe(data=pulse_data["phase_rad"])
                                    
                                    # --- snapshot metrics for Params table (no locks) ---
                                    # Phase-domain nominal sample rate
                                    try:
                                        FSd = self.sample_rate / 4.0
                                    except Exception:
                                        FSd = 250000.0

                                    # Message duration
                                    msg_dur_ms = float("nan")
                                    try:
                                        if isinstance(pulse_data, dict):
                                            xs_fm_ms = pulse_data.get("xs_fm_ms", None)
                                            xs_ms = pulse_data.get("xs_ms", None)
                                            if xs_fm_ms is not None and len(xs_fm_ms) >= 2:
                                                msg_dur_ms = float(xs_fm_ms[-1] - xs_fm_ms[0])
                                            elif xs_ms is not None and len(xs_ms) >= 2:
                                                msg_dur_ms = float(xs_ms[-1] - xs_ms[0])
                                    except Exception:
                                        pass

                                    # Carrier duration: time to first edge
                                    try:
                                        carrier_ms = float(edges[0] / FSd * 1e3) if (edges is not None and len(edges) > 0) else float("nan")
                                    except Exception:
                                        carrier_ms = float("nan")

                                    # Extract PSK metrics
                                    def _getf(d, k, default=float("nan")):
                                        try:
                                            return float(d.get(k, default))
                                        except Exception:
                                            return float("nan")
                                    pos = _getf(phase_res, "PosPhase")
                                    neg = _getf(phase_res, "NegPhase")
                                    PhRise = _getf(phase_res, "PhRise")
                                    PhFall = _getf(phase_res, "PhFall")
                                    ass = _getf(phase_res, "Ass")
                                    tmod = _getf(phase_res, "Tmod")
                                    fmod_hz = (FSd / tmod) if (tmod > 0 and tmod != float('inf')) else float("nan")
                                    rise_us = (PhRise / FSd * 1e6) if (PhRise == PhRise) else float("nan")
                                    fall_us = (PhFall / FSd * 1e6) if (PhFall == PhFall) else float("nan")

                                    # Tuned frequency and offset
                                    try:
                                        tuned_freq_hz = float(CENTER_FREQ_HZ) + float(IF_OFFSET_HZ)
                                    except Exception:
                                        tuned_freq_hz = float("nan")
                                    try:
                                        _target_hz = float(TARGET_SIGNAL_HZ)
                                    except Exception:
                                        _target_hz = float("nan")
                                    try:
                                        _if_off = float(IF_OFFSET_HZ)
                                    except Exception:
                                        _if_off = float("nan")
                                    if _target_hz == _target_hz and tuned_freq_hz == tuned_freq_hz:
                                        freq_offset_hz = tuned_freq_hz - _target_hz
                                    else:
                                        freq_offset_hz = _if_off

                                    # Power (RMS, dBm) over the detected pulse window
                                    power_rms_dbm = float("nan")
                                    try:
                                        ys_seg = pulse_data.get("ys", None) if isinstance(pulse_data, dict) else None
                                        if ys_seg is not None and len(ys_seg) > 0:
                                            _mw = 10.0 ** (ys_seg / 10.0)
                                            _mw_mean = float(np.nanmean(_mw))
                                            if _mw_mean > 0.0:
                                                power_rms_dbm = 10.0 * np.log10(_mw_mean)
                                    except Exception:
                                        pass

                                    _snapshot = {
                                        "Target Signal (Hz)": float(TARGET_SIGNAL_HZ) if 'TARGET_SIGNAL_HZ' in globals() else float("nan"),
                                        "Frequency (Hz)": tuned_freq_hz,
                                        "Frequency Offset (Hz)": freq_offset_hz,
                                        "Message Duration (ms)": msg_dur_ms,
                                        "Carrier Duration (ms)": carrier_ms,
                                        "Pos (rad)": pos,
                                        "Neg (rad)": neg,
                                        "Rise (μs)": rise_us,
                                        "Fall (μs)": fall_us,
                                        "Asymmetry (%)": ass,
                                        "Fmod (Hz)": fmod_hz,
                                        "Power (RMS, dBm)": power_rms_dbm,
                                    }

                                    self.last_phase_metrics = _snapshot
                                    self.last_msg_hex = str(msg_hex)
                                    #log.info("MsgS:",msg_hex)
                                    FSd = self.sample_rate/4
                                    phase_res = (
                                    f"Carrier={(edges[0]/FSd*1e3)}ms "
                                    f"Pos={phase_res['PosPhase']:.2f}rad "
                                    f"Neg={phase_res['NegPhase']:.2f}rad "
                                    f"Rise={(phase_res['PhRise']/FSd*1e6):.1f}us "
                                    f"Fall={(phase_res['PhFall']/FSd*1e6):.1f}us "
                                    f"Ass={phase_res['Ass']:.3f}%"
                                    f"Fmod={(FSd/phase_res['Tmod']):.3f}Hz"
                                    )
                                    
                                    pulse_data["title"] += f'\n {phase_res} \n HEX={msg_hex}'
                                except Exception as e:
                                    pulse_data["title"] += f"\n[PSK демодуляция пропущена: {e}]"
                        
                        if pulse_data["xs_ms"] is not None or pulse_data["xs_fm_ms"] is not None:
                            self.pulse_data_queue.put(pulse_data)

                start_abs = None

            with self.data_lock:
                self.last_rms_dbm = float(rms_dbm_vec[-1])
                step = max(1, VIS_DECIM)
                xs_top = (idx_end[::step] / float(self.sample_rate)).astype(np.float64)
                ys_top = rms_dbm_vec[::step].astype(np.float32)
                self.time_history.extend(map(float, xs_top))
                self.rms_history.extend(map(float, ys_top))

        # keep tail for sliding RMS
        need = max(0, self.win_samps - 1)
        k = min(need, p_cont.size)
        if k > 0:
            self.tail_p = p_cont[-k:].copy()
        else:
            self.tail_p = np.empty(0, dtype=np.float32)

        self.sample_counter += samples.size

        if PRINT_EVERY_N_SEC and (now - self.last_print) >= PRINT_EVERY_N_SEC:
            self.last_print = now
            log.debug(
                f"RMS(last): {self.last_rms_dbm:7.2f} dBm | "
                f"Freq(last impulse): {self.last_impulse_freq_hz/1000:.3f} kHz | "
                f"win={self.win_samps} samp ({RMS_WIN_MS:.2f} ms) | "
                f"CF={CENTER_FREQ_HZ/1e6:.6f} MHz | IF={IF_OFFSET_HZ:+} Hz"
            )

    # ---------- Plot updates ----------
    def _update_all_plots_combined(self, _frame):
        # 1) Main RMS plot
        with self.data_lock:
            if len(self.rms_history) >= 2:
                xs = np.array(self.time_history)
                ys = np.array(self.rms_history)
                tmax = xs[-1]
                tmin = max(0.0, tmax - LEVEL_HISTORY_SEC)
                mask = (xs >= tmin)
                xs = xs[mask]
                ys = ys[mask]
                self.ln_lvl.set_data(xs, ys)
                self.ax_lvl.set_xlim(max(0.0, tmin), max(5.0, tmax))
                if np.isfinite(np.nanmin(ys)) and np.isfinite(np.nanmax(ys)):
                    pad = 3.0
                    self.ax_lvl.set_ylim(np.nanmin(ys) - pad, np.nanmax(ys) + pad)
                self.ax_lvl.set_title(
                    f"Target: {TARGET_SIGNAL_HZ/1e6:.6f} MHz | "
                    f"IF: {IF_OFFSET_HZ:+} Hz | Fs: {self.sample_rate/1e6:.3f} MS/s | "
                    f"RMS: {RMS_WIN_MS:.2f} ms (N={self.win_samps})"
                )
               

        # 2) Pulse/PSK plot (only when new data arrives)
        #    If updates are disabled, drop queued data and keep the last plot.
        if not self.pulse_updates_enabled:
            try:
                while not self.pulse_data_queue.empty():
                    self.pulse_data_queue.get_nowait()
            except Exception:
                pass
            return self.ln_lvl,

        while not self.pulse_data_queue.empty():
            try:
                pulse_data = self.pulse_data_queue.get_nowait()
                if pulse_data:
                    if DEBUG_IMPULSE_LOG:
                        ys_len = 0
                        if pulse_data.get("ys") is not None:
                            ys_len = len(pulse_data["ys"])
                        log.debug(f"[IMPULSE] points={ys_len}")

                    if pulse_data["xs_ms"] is not None:
                        self.ln_pulse.set_data(pulse_data["xs_ms"], pulse_data["ys"])
                        self.ax_pulse.set_xlim(float(pulse_data["xs_ms"].min()), float(pulse_data["xs_ms"].max()))
                        ypad = 3.0
                        self.ax_pulse.set_ylim(float(np.nanmin(pulse_data["ys"])) - ypad, float(np.nanmax(pulse_data["ys"])) + ypad)
                        self.ax_pulse.set_title(pulse_data["pulse_title"])

                    # Фаза (середина)
                    if pulse_data.get("xs_fm_ms") is not None and pulse_data.get("phase_rad") is not None:
                        self.ln_phase.set_data(pulse_data["xs_fm_ms"], pulse_data["phase_rad"])
                        t_start = 0.0
                        t_end = float(pulse_data["xs_fm_ms"].max())
                        self.ax_phase.set_xlim(t_start, t_end)
                        self.ax_phase.set_ylim(-PSK_YLIMIT_RAD, PSK_YLIMIT_RAD)
                        self.ax_phase.set_title(pulse_data["title"])

                    # FM (низ)
                    if pulse_data.get("fm_xs_ms") is not None and pulse_data.get("fm_hz") is not None:
                        self.ln_fm.set_data(pulse_data["fm_xs_ms"], pulse_data["fm_hz"])
                        # sharex=True → ось X уже синхронна, но на всякий случай:
                        self.ax_fm.set_xlim(0.0, float(pulse_data["fm_xs_ms"].max()))
                        # ось Y по FM оставим авто (можно прижать вручную при желании)
                        #self.ax_fm.set_ylim(-3000, 3000)
                        # Y — вот так:
                        fm = pulse_data.get("fm_hz")
                        if fm is not None and len(fm):
                            ymin, ymax = float(np.nanmin(fm)), float(np.nanmax(fm))
                            if ymin == ymax:
                                # защита на плоскую линию
                                pad = max(1.0, abs(ymin) * 0.1)
                                self.ax_fm.set_ylim(ymin - pad, ymax + pad)
                            else:
                                pad = 0.05 * (ymax - ymin)
                                self.ax_fm.set_ylim(ymin - pad, ymax + pad)
                                                
                        
                        
                        
                    self.fig2.canvas.draw_idle()
            except queue.Empty:
                pass

        return self.ln_lvl,

    # ---------- Run/Stop ----------
    def run(self):
        self.ani = FuncAnimation(
            self.fig1,
            self._update_all_plots_combined,
            interval=120,
            blit=False,
            cache_frame_data=False
        )

        self.ani = FuncAnimation(self.fig1, self._update_all_plots_combined,
                            interval=120, blit=False, cache_frame_data=False)

        # --- NEW: поток читаем только если backend уже создан ---
        if self.backend is not None:
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

        try:
            plt.show()
        finally:
            self.stop()
            if hasattr(self, "reader_thread"):
                try:
                    self.reader_thread.join()
                except Exception:
                    pass

    def stop(self):
        if not self._stop:
            self._stop = True
            try:
                self.backend.stop()
            except Exception:
                pass

if __name__ == "__main__":
    log.info("Starting Sliding RMS + Pulse + PSK. Ctrl+C to exit. 'Save IQ' button available.")
    analyzer = SoapySlidingRMS()
    try:
        analyzer.run()
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop()
        log.info("\nОстановлено.")