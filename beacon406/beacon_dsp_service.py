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
from lib.backends import safe_make_backend
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
    rms_win_ms: float
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
        self._stop = False
        self.reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

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
        self.start()

    # ---------------- Backend ----------------
    def _make_backend(self):
        name = self.backend_name or os.environ.get("BACKEND_NAME", None) or "soapy_rtl"
        args = self.backend_args
        try:
            self.backend = safe_make_backend(
                name,
                sample_rate=SAMPLE_RATE_SPS,
                center_freq=float(CENTER_FREQ_HZ),
                gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
                agc=bool(ENABLE_AGC),
                corr_ppm=int(FREQ_CORR_PPM),
                device_args=args,
                if_offset_hz=IF_OFFSET_HZ,
            )
            st = self.backend.get_status() or {}
            self.sample_rate = float(st.get("actual_sample_rate_sps",
                                           getattr(self.backend, "actual_sample_rate_sps", SAMPLE_RATE_SPS)))
            self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
            self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))
            log.info("\n=== BACKEND STATUS ===\n" + self.backend.pretty_status() + "\n======================\n")
        except Exception as e:
            log.error(f"Backend init failed: {e}")
            raise

    # ---------------- Threads ----------------
    def start(self):
        if self.reader_thread and self.reader_thread.is_alive():
            return
        self._stop = False
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def stop(self):
        self._stop = True
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
        if self.jsonl_fp:
            self.jsonl_fp.flush()

    # ---------------- Reader loop ----------------
    def _read_block(self, nsamps: int) -> np.ndarray:
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
        while not self._stop:
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
        st = Status(
            sdr=str(getattr(self.backend, "driver", "?")),
            fs=float(self.sample_rate),
            bb_shift_hz=float(BB_SHIFT_HZ if BB_SHIFT_ENABLE else 0.0),
            rms_win_ms=float(RMS_WIN_MS),
            thresh_dbm=float(PULSE_THRESH_DBM),
            read_chunk=int(READ_CHUNK),
            queue_depth=len(self.pulse_queue),
        )
        self._emit("status", asdict(st))

    def _process_samples(self, samples: np.ndarray):
        base_idx = self.sample_counter
        x = samples.copy()

        # BB shift
        if BB_SHIFT_ENABLE and abs(BB_SHIFT_HZ) > 0:
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
            rms_dbm_vec = self._db10(P_win) + DBM_OFFSET_DB + self.backend.get_calib_offset_db()
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
                    if fm_out is not None and "xs_ms" in fm_out and "inst_freq_hz" in fm_out:
                        fr_xs_down, fr_ys_down = downsample_data(
                            fm_out["xs_ms"],
                            fm_out["inst_freq_hz"]
                        )
                        fr_xs_ms = fr_xs_down
                        fr_ys_hz = fr_ys_down

                    # Маркеры битов/полубитов из edges
                    markers_ms = None
                    if edges is not None and len(edges) > 0:
                        # Преобразуем индексы фронтов в миллисекунды
                        markers_ms = [float(edge / FSd * 1e3) for edge in edges[:100]]  # Ограничиваем количество

                    # Обогащаем pulse событие данными для графиков
                    pulse_event_data["phase_xs_ms"] = phase_xs_down
                    pulse_event_data["phase_ys_rad"] = phase_ys_down
                    pulse_event_data["fr_xs_ms"] = fr_xs_ms
                    pulse_event_data["fr_ys_hz"] = fr_ys_hz
                    pulse_event_data["markers_ms"] = markers_ms
                    pulse_event_data["preamble_ms"] = [0, float(carrier_ms)] if carrier_ms == carrier_ms else None
                    pulse_event_data["baud"] = float(baud) if baud == baud else None

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
        global PULSE_THRESH_DBM, RMS_WIN_MS
        while not self._stop:
            try:
                msg = self.rep.recv_json(flags=0)
            except Exception:
                time.sleep(0.01)
                continue
            try:
                cmd = str(msg.get("cmd", "")).lower()
                if cmd == "get_status":
                    self.rep.send_json({"ok": True, "status": asdict(Status(
                        sdr=str(getattr(self.backend, "driver", "?")),
                        fs=float(self.sample_rate),
                        bb_shift_hz=float(BB_SHIFT_HZ if BB_SHIFT_ENABLE else 0.0),
                        rms_win_ms=float(RMS_WIN_MS),
                        thresh_dbm=float(PULSE_THRESH_DBM),
                        read_chunk=int(READ_CHUNK),
                        queue_depth=len(self.pulse_queue),
                    ))})
                elif cmd == "start_acquire":
                    self.start(); self.rep.send_json({"ok": True})
                elif cmd == "stop_acquire":
                    self.stop(); self.rep.send_json({"ok": True})
                elif cmd == "set_params":
                    # Позволяет менять базовые параметры на лету (минимальный набор)
                    changed = {}
                    if "thresh_dbm" in msg:
                        PULSE_THRESH_DBM = float(msg["thresh_dbm"]); changed["thresh_dbm"] = PULSE_THRESH_DBM
                    if "rms_win_ms" in msg:
                        RMS_WIN_MS = float(msg["rms_win_ms"]); self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
                        changed["rms_win_ms"] = RMS_WIN_MS
                    self.rep.send_json({"ok": True, "changed": changed})
                elif cmd == "get_last_pulse":
                    # Возвращаем последние данные pulse с фазой и частотой
                    if hasattr(self, 'last_pulse_data'):
                        self.rep.send_json({"ok": True, "pulse": self.last_pulse_data})
                    else:
                        self.rep.send_json({"ok": False, "error": "No pulse data available"})
                elif cmd == "save_sigmf":
                    # Сохранение последнего сегмента в SigMF — заглушка (впиши свою функцию сохранения)
                    # Здесь можно дернуть твой writer из lib.sigio, если он есть
                    fn = CAPTURE_DIR / time.strftime("pulse_%Y%m%d_%H%M%S.cf32")
                    seg = (self.last_iq_seg.copy() if isinstance(self.last_iq_seg, np.ndarray) else np.empty(0, np.complex64))
                    if seg.size:
                        seg.astype(np.complex64).tofile(str(fn))
                        self.rep.send_json({"ok": True, "path": str(fn)})
                    else:
                        self.rep.send_json({"ok": False, "err": "no_segment"})
                else:
                    self.rep.send_json({"ok": False, "err": "unknown_cmd"})
            except Exception as e:
                self.rep.send_json({"ok": False, "err": str(e)})

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Headless Beacon DSP Service")
    ap.add_argument("--pub", default=DEFAULT_PUB_ADDR, help="ZeroMQ PUB address (bind)")
    ap.add_argument("--rep", default=DEFAULT_REP_ADDR, help="ZeroMQ REP address (bind)")
    ap.add_argument("--backend", default=os.environ.get("BACKEND_NAME", "soapy_rtl"), help="Backend name (soapy_rtl/soapy_hackrf/soapy_airspy/soapy_sdrplay/file)")
    ap.add_argument("--backend-args", default=None, help="Backend args JSON")
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
        log.info("Stopping DSP service...")
        svc.stop()
        log.info("Done")

