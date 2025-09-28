#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beacon_dsp_service.py - DSP-сервис на базе beacon406_PSK_FM-plot.py
Читает SDR, детектирует импульсы, анализирует PSK/FM, логирует результаты.
Без визуализации. Для интеграции с веб.
"""

import time
import threading
import queue
import numpy as np
from lib.backends import safe_make_backend
from lib.metrics import process_psk_impulse
from lib.demod import phase_demod_psk_msg_safe
from lib.config import BACKEND_NAME, BACKEND_ARGS
from lib.processing_fm import fm_discriminator
from lib.logger import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# Параметры (из plot-скрипта, можно корректировать)
TARGET_SIGNAL_HZ = 406_037_000
IF_OFFSET_HZ = -37_000
CENTER_FREQ_HZ = TARGET_SIGNAL_HZ + IF_OFFSET_HZ
SAMPLE_RATE_SPS = 1_000_000
USE_MANUAL_GAIN = True
TUNER_GAIN_DB = 30.0
ENABLE_AGC = False
FREQ_CORR_PPM = 0
BB_SHIFT_ENABLE = True
BB_SHIFT_HZ = IF_OFFSET_HZ
RMS_WIN_MS = 1.0
DBM_OFFSET_DB = -30.0
PULSE_THRESH_DBM = -45.0
READ_CHUNK = 65_536
PSK_BASELINE_MS = 2.0
EPS = 1e-20
DEBUG_IMPULSE_LOG = True

class BeaconDSPService:
    def __init__(self):
        # Инициализация backend (как в plot)
        self.backend = safe_make_backend(
            BACKEND_NAME,
            sample_rate=SAMPLE_RATE_SPS,
            center_freq=float(CENTER_FREQ_HZ),
            gain_db=float(TUNER_GAIN_DB) if USE_MANUAL_GAIN else None,
            agc=bool(ENABLE_AGC),
            corr_ppm=int(FREQ_CORR_PPM),
            device_args=BACKEND_ARGS,
            if_offset_hz=IF_OFFSET_HZ,
            on_fail="file_wait"
        )
        self.sample_rate = self.backend.actual_sample_rate_sps  # Реальный Fs
        log.info(f"Backend sample rate: {self.sample_rate:.2f} Sa/s")

        # Параметры RMS и буферы (как в plot)
        self.win_samps = max(1, int(round(self.sample_rate * (RMS_WIN_MS * 1e-3))))
        self.tail_p = np.empty(0, dtype=np.float32)
        self.nco_phase = 0.0
        self.nco_k = 2.0 * np.pi * (BB_SHIFT_HZ / float(self.sample_rate))
        self.sample_counter = 0
        self._stop = False
        self.last_rms_dbm = float("-inf")
        self.in_pulse = False
        self.pulse_start_abs = None
        self.full_samples = np.empty(0, dtype=np.complex64)  # Буфер для импульсов

        # Очередь для результатов импульсов (для интеграции)
        self.pulse_queue = queue.Queue()

        # Поток чтения
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)

    def _reader_loop(self):
        while not self._stop:
            samples = self.backend.read(READ_CHUNK)
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue

            # Сдвиг BB если нужно (как в plot)
            if BB_SHIFT_ENABLE:
                nco_vec = np.exp(1j * (self.nco_phase + self.nco_k * np.arange(len(samples))))
                samples *= nco_vec.astype(np.complex64)
                self.nco_phase = (self.nco_phase + self.nco_k * len(samples)) % (2.0 * np.pi)

            # RMS расчёт (как в plot)
            p_cont = (samples.real**2 + samples.imag**2).astype(np.float32)
            p = np.concatenate([self.tail_p, p_cont]) if self.tail_p.size else p_cont
            cs = np.cumsum(p, dtype=np.float64)
            if len(p) >= self.win_samps:
                P_win = (cs[self.win_samps-1:] - np.concatenate(([0.0], cs[:-self.win_samps]))) / self.win_samps
            else:
                P_win = np.empty(0, dtype=np.float32)

            rms_dbm_vec = 10.0 * np.log10(np.maximum(P_win, EPS)) + DBM_OFFSET_DB
            on = (rms_dbm_vec >= PULSE_THRESH_DBM).astype(np.int8)
            trans = np.diff(on, prepend=on[0])

            # Детекция импульсов (как в plot)
            pairs = []
            for i, t in enumerate(trans):
                if t == 1:
                    if not self.in_pulse:
                        self.in_pulse = True
                        self.pulse_start_abs = self.sample_counter + (self.win_samps - 1) + i
                elif t == -1:
                    if self.in_pulse:
                        pulse_end_abs = self.sample_counter + (self.win_samps - 1) + (i - 1)
                        pairs.append((self.pulse_start_abs, pulse_end_abs))
                        self.in_pulse = False

            # Обработка пар (импульсов)
            for start_abs, end_abs in pairs:
                dur_ms = (end_abs - start_abs + 1) / self.sample_rate * 1000
                log.info(f"Pulse detected: start={start_abs}, length={dur_ms:.1f}ms")

                # Извлечение сегмента и анализ (PSK/FM как в plot)
                seg = self.full_samples[start_abs:end_abs + 1]  # Адаптировать буфер если нужно
                pulse_data = self._process_pulse(seg, dur_ms)
                self.pulse_queue.put(pulse_data)
                log.info(f"Pulse processed: {dur_ms:.1f}ms, PSK ok={pulse_data['msg_ok']}, HEX={pulse_data['msg_hex']}")

            # Обновление буфера (как в plot)
            self.full_samples = np.append(self.full_samples, samples)  # Урезать по лимиту если нужно
            self.sample_counter += len(samples)
            self.tail_p = p_cont[-(self.win_samps - 1):] if len(p_cont) >= (self.win_samps - 1) else p_cont.copy()

    def _process_pulse(self, iq_seg, dur_ms):
        # PSK/FM анализ (как в plot)
        pulse_result = process_psk_impulse(iq_seg, fs=self.sample_rate, baseline_ms=PSK_BASELINE_MS)
        phase_data = pulse_result.get("phase_rad", [])
        xs_ms = pulse_result.get("xs_ms", [])

        # Демодуляция PSK
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(phase_data, window=40, threshold=0.5, start_idx=0, N=28, min_edges=29)

        # FM
        fm_result = fm_discriminator(iq_seg, fs=self.sample_rate, pre_lpf_hz=50000, decim=4, smooth_hz=15000)
        fm_hz = fm_result.get("freq_hz", [])
        fm_xs_ms = fm_result.get("xs_ms", [])

        return {
            'dur_ms': dur_ms,
            'msg_hex': msg_hex,
            'msg_ok': bool(msg_hex),
            'phase_data': phase_data.tolist(),
            'xs_ms': xs_ms.tolist(),
            'fm_hz': fm_hz.tolist(),
            'fm_xs_ms': fm_xs_ms.tolist()
        }

    def run(self):
        self.reader_thread.start()
        log.info("DSP service started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)  # Мониторинг
                # Здесь можно добавить вывод в файл или сокет
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._stop = True
        self.backend.stop()
        self.reader_thread.join()
        log.info("DSP service stopped.")

if __name__ == "__main__":
    service = BeaconDSPService()
    service.run()