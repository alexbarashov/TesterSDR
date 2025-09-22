"""
COSPAS/SARSAT Beacon Tester - Version 2.0
=========================================
Точное воспроизведение интерфейса по изображению.
Одностраничное Flask приложение с аутентичным дизайном.
Порт: 8738 (чтобы не конфликтовать с оригинальным)
"""

from __future__ import annotations
import math
import random
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List
from flask import Flask, jsonify, request, Response

# Добавляем путь к библиотекам
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.hex_decoder import hex_to_bits, build_table_rows
from lib.metrics import process_psk_impulse
from lib.demod import phase_demod_psk_msg_safe
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

def _find_pulse_segment(iq_data, sample_rate, thresh_dbm, win_ms, start_delay_ms, calib_db):
    """
    Поиск импульса по RMS порогу (из test_cf32_to_phase_msg_FFT.py)
    """
    # Вычисляем RMS в dBm
    W = max(1, int(win_ms * 1e-3 * sample_rate))
    p = np.abs(iq_data)**2
    ma = np.convolve(p, np.ones(W)/W, mode="same")
    rms = np.sqrt(ma + 1e-30)
    rms_dbm = 20*np.log10(rms + 1e-30) + calib_db

    # Ищем импульсы пока не найдём достаточно длинный
    while True:
        # Поиск границ импульса
        idx = np.where(rms_dbm > thresh_dbm)[0]
        if idx.size == 0:
            return None

        i0, i1 = idx[0], idx[-1]

        # Проверяем длительность
        dur_ms = (i1 - i0) / sample_rate * 1e3
        if dur_ms < 5.0:
            # Слишком короткий, ищем дальше
            rms_dbm[i0:i1] = -999  # затираем
            continue
        else:
            break

    # Сдвигаем границы с учетом задержки и окна
    i0 = min(i1, i0 + int((start_delay_ms + win_ms) * 1e-3 * sample_rate))
    i1 = max(i0, i1 - int(win_ms * 1e-3 * sample_rate))

    if i1 <= i0:
        return None

    return iq_data[i0:i1]

def process_cf32_file(file_path):
    """
    Обрабатывает CF32 файл используя библиотеки metrics и demod
    Возвращает словарь с результатами для обновления BeaconState
    """
    try:
        # Читаем CF32 файл
        iq_data = np.fromfile(file_path, dtype=np.complex64)

        if len(iq_data) == 0:
            return {"error": "Empty file"}

        # Параметры обработки из test_cf32_to_phase_msg_FFT.py
        sample_rate = 1000000  # 1 MHz
        baseline_ms = 10.0  # PSK_BASELINE_MS из test_cf32
        t0_offset_ms = 0.0

        # Параметры для поиска импульса
        thresh_dbm = -60.0  # порог для поиска импульса
        win_ms = 1.0  # окно RMS
        start_delay_ms = 3.0  # обрезание начала
        calib_db = -30.0  # калибровка

        # Поиск импульса по RMS как в test_cf32_to_phase_msg_FFT.py
        iq_seg = _find_pulse_segment(iq_data, sample_rate, thresh_dbm, win_ms, start_delay_ms, calib_db)

        if iq_seg is None or len(iq_seg) == 0:
            return {"error": "No pulse found"}

        # Обрабатываем сигнал с помощью metrics
        pulse_result = process_psk_impulse(
            iq_seg=iq_seg,
            fs=sample_rate,
            baseline_ms=baseline_ms,
            t0_offset_ms=t0_offset_ms,
            use_lpf_decim=True,
            remove_slope=True,  # выравниваем фазу по горизонтали
        )

        if not pulse_result or "phase_rad" not in pulse_result:
            return {"error": "No pulse detected"}

        # Демодуляция PSK сообщения
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(data=pulse_result["phase_rad"])

        # Извлекаем метрики из результата
        phase_data = pulse_result.get("phase_rad", [])
        xs_fm_ms = pulse_result.get("xs_ms", [])

        print(f"DEBUG: phase_data length = {len(phase_data) if hasattr(phase_data, '__len__') else 0}")
        print(f"DEBUG: xs_fm_ms length = {len(xs_fm_ms) if hasattr(xs_fm_ms, '__len__') else 0}")
        if isinstance(phase_data, np.ndarray) and phase_data.size > 0:
            print(f"DEBUG: phase_data sample: min={np.min(phase_data):.3f}, max={np.max(phase_data):.3f}")
        if isinstance(xs_fm_ms, np.ndarray) and xs_fm_ms.size > 0:
            print(f"DEBUG: xs_fm_ms sample: min={np.min(xs_fm_ms):.3f}, max={np.max(xs_fm_ms):.3f}")

        # Безопасное преобразование в список
        if isinstance(phase_data, np.ndarray):
            phase_list = phase_data.tolist()
        else:
            phase_list = list(phase_data) if phase_data is not None else []

        if isinstance(xs_fm_ms, np.ndarray):
            xs_list = xs_fm_ms.tolist()
        else:
            xs_list = list(xs_fm_ms) if xs_fm_ms is not None else []

        # Безопасная обработка edges (может быть numpy массивом)
        if edges is not None and hasattr(edges, '__len__') and len(edges) > 0:
            edges_list = edges.tolist() if hasattr(edges, 'tolist') else list(edges)
        else:
            edges_list = []

        result = {
            "success": True,
            "msg_hex": msg_hex if msg_hex else "",
            "phase_data": phase_list,
            "xs_fm_ms": xs_list,
            "edges": edges_list,
            "file_processed": True
        }

        # Если есть метрики фазы, добавляем их
        print(f"DEBUG: phase_res type: {type(phase_res)}")
        print(f"DEBUG: phase_res content: {phase_res}")
        if phase_res is not None and (isinstance(phase_res, dict) or (hasattr(phase_res, '__len__') and len(phase_res) > 0)):
            # Частота дискретизации после децимации (как в test_cf32_to_phase_msg_FFT.py)
            FSd = sample_rate / 4.0

            ph_rise_val = phase_res.get("PhRise", 0.0)
            ph_fall_val = phase_res.get("PhFall", 0.0)

            # Безопасное преобразование numpy значений
            pos_phase = float(phase_res.get("PosPhase", 0.0)) if phase_res.get("PosPhase") is not None else 0.0
            neg_phase = float(phase_res.get("NegPhase", 0.0)) if phase_res.get("NegPhase") is not None else 0.0

            if ph_rise_val is not None and np.isfinite(ph_rise_val) and float(ph_rise_val) != 0:
                ph_rise = float(ph_rise_val / FSd * 1e6)
            else:
                ph_rise = 0.0

            if ph_fall_val is not None and np.isfinite(ph_fall_val) and float(ph_fall_val) != 0:
                ph_fall = float(ph_fall_val / FSd * 1e6)
            else:
                ph_fall = 0.0

            # Вычисляем t_mod для дальнейших расчетов
            t_mod = float(phase_res.get("Tmod", 0.0)) if phase_res.get("Tmod") is not None else 0.0

            # Вычисляем BitRate: FSd / Tmod как в beacon406-plot.py
            FSd = sample_rate / 4.0  # 250000.0
            bitrate_bps = FSd / t_mod if t_mod > 0 else 0.0

            result.update({
                "pos_phase": pos_phase,
                "neg_phase": neg_phase,
                "ph_rise": ph_rise,
                "ph_fall": ph_fall,
                "asymmetry": float(phase_res.get("Ass", 0.0)) if phase_res.get("Ass") is not None else 0.0,
                "t_mod": t_mod,
                "bitrate_bps": bitrate_bps
            })

        # Дополнительные метрики из pulse_result
        if "rms_dbm" in pulse_result:
            result["rms_dbm"] = float(pulse_result["rms_dbm"]) if pulse_result["rms_dbm"] is not None else 0.0
        if "freq_hz" in pulse_result:
            result["freq_hz"] = float(pulse_result["freq_hz"]) if pulse_result["freq_hz"] is not None else 0.0

        # Дополнительные вычисления для Current таблицы
        # Total,ms из временной оси xs_fm_ms
        if isinstance(xs_list, list) and len(xs_list) > 1:
            total_ms = float(xs_list[-1] - xs_list[0])
        else:
            total_ms = 0.0

        # Prise,ms из ph_rise (мкс -> мс)
        prise_ms = ph_rise / 1000.0 if ph_rise > 0 else 0.0

        # Preamble,ms из carrier_ms = edges[0] / FSd * 1e3 как в beacon406-plot.py
        FSd = sample_rate / 4.0  # 250000.0
        if edges_list and len(edges_list) > 0:
            preamble_ms = float(edges_list[0] / FSd * 1e3)
        else:
            preamble_ms = float(baseline_ms)  # fallback

        # Symmetry,% из asymmetry (дублируем для symmetry_pct)
        symmetry_pct = float(phase_res.get("Ass", 0.0)) if phase_res.get("Ass") is not None else 0.0

        # Добавляем в результат
        result.update({
            "total_ms": total_ms,
            "prise_ms": prise_ms,
            "preamble_ms": preamble_ms,
            "symmetry_pct": symmetry_pct
        })

        return result

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

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
    hex_message: str = ""  # HEX сообщение из загруженного файла
    current_file: str = ""  # Путь к текущему загруженному файлу
    phase_data: list = field(default_factory=list)  # Данные фазы для графика
    xs_fm_ms: list = field(default_factory=list)  # Временная шкала для графика фазы

    # Текущие измерения (пустые до загрузки файла)
    fs1_hz: float = 0.0
    fs2_hz: float = 0.0
    fs3_hz: float = 0.0

    # Фазовые параметры (пустые до загрузки файла)
    phase_pos_rad: float = 0.0
    phase_neg_rad: float = 0.0
    t_rise_mcs: float = 0.0
    t_fall_mcs: float = 0.0

    # Дополнительные фазовые метрики из demod (пустые до загрузки файла)
    pos_phase: float = 0.0
    neg_phase: float = 0.0
    ph_rise: float = 0.0
    ph_fall: float = 0.0
    asymmetry: float = 0.0
    t_mod: float = 0.0
    rms_dbm: float = 0.0
    freq_hz: float = 0.0

    # Дополнительные параметры (пустые до загрузки файла)
    p_wt: float = 0.0
    prise_ms: float = 0.0
    bitrate_bps: float = 0.0
    symmetry_pct: float = 0.0
    preamble_ms: float = 0.0
    total_ms: float = 0.0
    rep_period_s: float = 0.0


STATE = BeaconState()

# HTML страница с точным воспроизведением дизайна
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COSPAS/SARSAT Beacon Tester v2.1</title>
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
            font-size: 20px;  /* увеличен с 16px */
            font-weight: bold;
            text-align: center;
            border-bottom: 1px solid #4a8bc2;
        }

        /* Основной контейнер */
        .container {
            display: flex;
            height: calc(100vh - 52px);  /* обновлено с учетом большего заголовка */
            background: #e8e8e8;
        }

        /* Левая панель */
        .left-panel {
            width: 220px;  /* увеличена ширина с 180px */
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
            font-size: 13px;  /* увеличен с 11px */
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
            font-size: 13px;  /* увеличен с 11px */
        }

        .radio-group label {
            display: block;
            margin: 4px 0;  /* увеличен отступ */
            cursor: pointer;
            font-size: 13px;  /* добавлен размер шрифта */
        }

        .control-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 3px 0;
        }

        .control-input {
            width: 50px;  /* увеличена ширина */
            font-size: 13px;  /* увеличен с 10px */
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
            padding: 6px 12px;  /* увеличены отступы */
            font-size: 13px;  /* увеличен с 11px */
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
            padding: 6px 10px;  /* увеличены отступы */
            font-size: 14px;  /* увеличен с 11px */
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
            font-size: 16px;  /* добавлен размер шрифта */
        }

        .message-line {
            font-size: 14px;  /* увеличен с 11px */
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
            font-size: 13px;  /* увеличен с 10px */
            color: #495057;
            margin-top: 6px;
        }

        .chart-title {
            text-align: center;
            font-size: 14px;  /* увеличен с 11px */
            color: #6c757d;
            margin-top: 6px;
        }

        /* Правая панель */
        .right-panel {
            width: 320px;  /* увеличена ширина с 280px */
            background: #f8f9fa;
            border-left: 1px solid #dee2e6;
            padding: 10px;
        }

        .stats-header {
            background: linear-gradient(180deg, #a8c8e4 0%, #7bb3d9 100%);
            color: #2c3e50;
            font-weight: bold;
            font-size: 14px;  /* увеличен с 11px */
            text-align: center;
            padding: 6px;
            border: 1px solid #6699cc;
            border-radius: 3px;
            margin-bottom: 10px;
        }

        .stats-content {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 6px;
        }

        .stat-row {
            display: flex;
            justify-content: space-between;
            font-size: 13px;  /* увеличен с 11px */
            padding: 4px 0;  /* увеличен отступ */
            border-bottom: 1px solid #f8f9fa;
        }

        .stat-row:last-child {
            border-bottom: none;
        }

        .stat-label {
            color: #495057;
            font-weight: 500;
        }

        .stat-value {
            color: #212529;
            font-weight: normal;
            font-family: 'Courier New', monospace;
            font-size: 13px;  /* добавлен размер шрифта */
        }

        /* Стили для HTML таблицы Message */
        .message-table-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 0;
            position: relative;
            height: 494px;
            margin-bottom: 8px;
            overflow-y: auto;
        }

        .message-table-header {
            background: #1976D2;
            color: white;
            padding: 12px;
            text-align: center;
        }

        .message-table-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: bold;
        }

        .message-table-header .hex-info {
            margin: 4px 0 0 0;
            font-size: 11px;
        }

        .message-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .message-table th {
            background: #E0E0E0;
            color: #333;
            font-weight: bold;
            font-size: 15px;
            padding: 8px 5px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .message-table td {
            padding: 6px 5px;
            border-bottom: 0.5px solid #DDD;
            color: #333;
            font-size: 14px;
        }

        .message-table tr:nth-child(even) {
            background: #F5F5F5;
        }

        .message-table tr:nth-child(odd) {
            background: white;
        }

        .message-table .binary-content {
            font-family: monospace;
            font-size: 13px;
            color: #0066CC;
        }

        .message-table .field-name {
            font-weight: bold;
        }

        .message-table-footer {
            padding: 8px 12px;
            font-size: 11px;
            color: #666;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }

        /* Стили для HTML таблицы 121 Data */
        .data121-table-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 0;
            position: relative;
            height: 494px;
            margin-bottom: 8px;
            overflow-y: auto;
        }

        .data121-table-header {
            background: #5a9bd4;
            color: white;
            padding: 12px;
            text-align: center;
        }

        .data121-table-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: bold;
        }

        .data121-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }

        .data121-table td {
            padding: 8px 10px;
            border: 1px solid #999;
            color: #333;
            font-size: 14px;
        }

        .data121-table .param-name {
            width: 1%;
            white-space: nowrap;
        }

        .data121-table .param-name {
            font-weight: bold;
        }

        .data121-table .param-value {
            font-weight: normal;
        }

        .data121-table-footer {
            padding: 8px 20px;
            font-size: 11px;
            color: #666;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }

        /* Стили для HTML таблицы Sum Table */
        .sum-table-container {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 0;
            position: relative;
            height: 494px;
            margin-bottom: 8px;
            overflow-y: auto;
        }

        .sum-params-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-bottom: 10px;
        }

        .sum-params-table .header-406 {
            background: #5a9bd4;
            color: white;
            text-align: center;
            font-weight: bold;
            padding: 8px;
            font-size: 16px;
        }

        .sum-params-table .header-121 {
            background: #5a9bd4;
            color: white;
            text-align: center;
            font-weight: bold;
            padding: 8px;
            font-size: 16px;
        }

        .sum-params-table .subheader {
            background: #87CEEB;
            color: white;
            text-align: center;
            padding: 6px;
            font-size: 14px;
            font-weight: bold;
        }

        .sum-params-table .subheader-empty {
            background: #87CEEB;
            color: white;
            text-align: center;
            padding: 6px;
            font-size: 14px;
            font-weight: bold;
            width: 1%;
        }

        .sum-params-table .param-row {
            background: white;
        }

        .sum-params-table .param-row:nth-child(even) {
            background: #f9f9f9;
        }

        .sum-params-table td {
            padding: 6px 5px;
            border: 1px solid #ccc;
            font-size: 14px;
            text-align: center;
        }

        .sum-params-table .param-name {
            text-align: right;
            font-weight: bold;
            background: #f0f0f0;
            width: 1%;
            white-space: nowrap;
        }

        .sum-message-section {
            margin-top: 10px;
            border-top: 2px solid #5a9bd4;
        }

        .sum-message-header {
            background: #5a9bd4;
            color: white;
            text-align: center;
            padding: 8px;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        COSPAS/SARSAT Beacon Tester
    </div>

    <div class="container">
        <!-- Левая панель -->
        <div class="left-panel">
            <div class="panel-section">
                <div class="section-header">VIEW</div>
                <div class="section-content">
                    <div class="radio-group">
                        <label><input type="radio" name="view" value="phase" checked onchange="changeView('phase')"> 406 Phase</label>
                        <label><input type="radio" name="view" value="fr_stability" onchange="changeView('fr_stability')"> 406 Fr. stability</label>
                        <label><input type="radio" name="view" value="ph_rise_fall" onchange="changeView('ph_rise_fall')"> 406 Ph/Rise/Fall</label>
                        <label><input type="radio" name="view" value="fr_pwr" onchange="changeView('fr_pwr')"> 406 Fr/Pwr</label>
                        <label><input type="radio" name="view" value="inburst_fr" onchange="changeView('inburst_fr')"> 406 Inburst fr</label>
                        <label><input type="radio" name="view" value="sum_table" onchange="changeView('sum_table')"> 406 Sum table</label>
                        <label><input type="radio" name="view" value="message" onchange="changeView('message')"> 406 Message</label>
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
                            <option value="2">2%</option>
                            <option value="5">5%</option>
                            <option value="10" selected>10%</option>
                            <option value="20">20%</option>
                            <option value="50">50%</option>
                            <option value="100">100%</option>
                        </select>
                    </div>
                    <div class="control-row">
                        <span>Update</span>
                        <div class="radio-inline">
                            <label><input type="radio" name="update" checked> ON</label>
                            <label><input type="radio" name="update"> OFF</label>
                        </div>
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <div class="section-header">FILE</div>
                <div class="section-content">
                    <button class="button" onclick="loadFile()">File</button>
                    <button class="button" onclick="saveFile()">Save</button>
                    <input type="file" id="fileInput" accept=".cf32" style="display: none;" onchange="uploadFile(this)">
                </div>
            </div>

            <div class="panel-section">
                <div class="section-header">TESTER</div>
                <div class="section-content">
                    <button class="button" onclick="measure()">Measure</button>
                    <br>
                    <button class="button primary" onclick="runTest()">Run</button>
                    <button class="button" onclick="contTest()">Cont</button>
                    <button class="button danger" onclick="breakTest()">Break</button>
                </div>
            </div>
        </div>

        <!-- Центральная панель -->
        <div class="center-panel">
            <div class="info-grid">
                <div class="info-block">
                    <div class="info-label">Protocol</div>
                    <div class="info-value" id="protocol">N</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Date</div>
                    <div class="info-value" id="date">01.08.2025</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Conditions</div>
                    <div class="info-value">
                        <a href="#" style="color: #007bff; text-decoration: underline;">Normal temperature, Idling</a>
                    </div>
                </div>
            </div>

            <div class="info-row-2">
                <div class="info-block">
                    <div class="info-label">Beacon Model</div>
                    <div class="info-value" id="beaconModel">Beacon N</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Beacon Frequency</div>
                    <div class="info-value" id="beaconFreq">406025000.0</div>
                </div>
            </div>

            <div class="beacon-title">Beacon 406</div>
            <div class="message-line">Message: <span id="message">[no message]</span></div>

            <div class="chart-container">
                <canvas id="phaseChart"></canvas>
            </div>

            <div class="phase-values">
                <span>Phase+ = <span id="phasePlus">-63.31</span>°</span>
                <span>TRise+ = <span id="tRise">-59.9</span> mcs</span>
                <span>Phase- = <span id="phaseMinus">-64.73</span>°</span>
                <span>TFall- = <span id="tFall">-121.4</span> mcs</span>
            </div>

            <div class="chart-title" id="chartTitle">Fig.8 Phase</div>
        </div>

        <!-- Правая панель -->
        <div class="right-panel">
            <div class="stats-header">Current</div>
            <div class="stats-content" id="statsContent">
                <!-- Статистика будет заполняться JavaScript -->
            </div>
        </div>
    </div>

    <script>
        console.log('=== BEACON TESTER v2.1 LOADED ===');
        let canvas = document.getElementById('phaseChart');
        let ctx = canvas.getContext('2d');
        let currentView = 'phase';
        let currentTimeScale = 10; // Текущий масштаб времени в процентах (по умолчанию 10%)
        const MESSAGE_DURATION_MS = 440; // Длительность сообщения в миллисекундах
        const PHASE_START_OFFSET_MS = 0; // Начинаем отображение графика с 0мс (без смещения)

        function resizeCanvas() {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
        }

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function changeView(viewType) {
            console.log('=== DEBUG: changeView called with:', viewType);
            currentView = viewType;
            console.log('=== DEBUG: currentView set to:', currentView);

            // Восстанавливаем canvas если он был заменен HTML таблицей
            const chartContainer = document.querySelector('.chart-container');
            if (!chartContainer.querySelector('#phaseChart')) {
                chartContainer.innerHTML = '<canvas id="phaseChart"></canvas>';
                // Обновляем глобальные ссылки на canvas
                canvas = document.getElementById('phaseChart');
                ctx = canvas.getContext('2d');
                resizeCanvas(); // Переинициализируем размеры canvas
            }

            // Очищаем canvas при переключении режимов
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const titleEl = document.getElementById('chartTitle');
            const phaseValuesEl = document.querySelector('.phase-values');

            switch(viewType) {
                case 'phase':
                    titleEl.textContent = 'Fig.8 Phase';
                    phaseValuesEl.innerHTML = `
                        <span>Phase+ = <span id="phasePlus">-63.31</span>°</span>
                        <span>TRise+ = <span id="tRise">-59.9</span> mcs</span>
                        <span>Phase- = <span id="phaseMinus">-64.73</span>°</span>
                        <span>TFall- = <span id="tFall">-121.4</span> mcs</span>
                    `;
                    break;
                case 'ph_rise_fall':
                    titleEl.textContent = 'Fig.7 Rise and Fall Times';
                    phaseValuesEl.innerHTML = `
                        <span>TRise = <span id="tRise">59.9</span> mcs</span>
                        <span>TFall = <span id="tFall">121.4</span> mcs</span>
                    `;
                    break;
                case 'fr_stability':
                    titleEl.textContent = 'Fig.6 Frequency Stability';
                    phaseValuesEl.innerHTML = `
                        <span>FS1 = <span id="fs1">406025864.0</span> Hz</span>
                        <span>FS2 = <span id="fs2">406025864.2</span> Hz</span>
                        <span>FS3 = <span id="fs3">406012489.9</span> Hz</span>
                    `;
                    break;
                case 'fr_pwr':
                    titleEl.textContent = 'Fig.9 Frequency/Power';
                    phaseValuesEl.innerHTML = `
                        <span>Freq = <span id="freq">406.025</span> MHz</span>
                        <span>Power = <span id="power">0.572</span> Wt</span>
                    `;
                    break;
                case 'inburst_fr':
                    titleEl.textContent = 'Fig.10 Inburst Frequency';
                    phaseValuesEl.innerHTML = `
                        <span>BitRate = <span id="bitrate">400.318</span> bps</span>
                        <span>Symmetry = <span id="symmetry">4.049</span> %</span>
                    `;
                    break;
                case 'sum_table':
                    titleEl.textContent = 'Summary Table';
                    phaseValuesEl.innerHTML = '<span>See Current panel for all values</span>';
                    break;
                case 'message':
                    titleEl.textContent = 'EPIRB/ELT Beacon Message Decoder';
                    phaseValuesEl.innerHTML = '<span>Decoded COSPAS-SARSAT 406 MHz beacon message</span>';
                    break;
                case '121_data':
                    titleEl.textContent = '121.5 MHz Transmitter Parameters';
                    phaseValuesEl.innerHTML = '<span>121.5 MHz Emergency Locator Transmitter Data</span>';
                    break;
            }
            // Всегда вызываем fetchData() для обновления - специальные режимы обрабатываются внутри fetchData()
            fetchData();
        }

        // Функция расчета смещения в зависимости от масштаба
        function getOffsetForScale(scale) {
            switch(scale) {
                case 1: return 1;  // 1% -> -1ms
                case 2: return 2;  // 2% -> -2ms
                case 5: return 4;  // 5% -> -4ms
                case 10:
                case 20:
                case 50: return 5; // 10%-50% -> -5ms
                default: return 5; // fallback
            }
        }

        function onTimeScaleChange() {
            const timeScaleSelect = document.getElementById('timeScale');
            currentTimeScale = parseInt(timeScaleSelect.value);
            console.log('Time scale changed to:', currentTimeScale + '%');

            // Перерисовываем график с новым масштабом если данные загружены
            if (currentView === 'phase') {
                fetchData(); // Обновляем отображение с новым масштабом
            }
        }

        
        function drawChart(data) {
            console.log('DEBUG: drawChart called with currentView:', currentView);
            console.log('DEBUG: data object:', data);
            console.log('DEBUG: data.hex_message:', data ? data.hex_message : 'no data');
            console.log('DEBUG: data.phase_data type/length:', data ? typeof data.phase_data + '/' + (data.phase_data ? data.phase_data.length : 'null') : 'no data');
            console.log('DEBUG: data.xs_fm_ms type/length:', data ? typeof data.xs_fm_ms + '/' + (data.xs_fm_ms ? data.xs_fm_ms.length : 'null') : 'no data');
            if (currentView === 'ph_rise_fall') {
                console.log('DEBUG: Drawing ph_rise_fall chart');
                drawRiseFallChart(data);
                return;
            } else if (currentView === 'fr_stability') {
                console.log('DEBUG: Drawing frequency chart');
                drawFrequencyChart(data);
                return;
            } else if (currentView === '121_data') {
                console.log('DEBUG: Drawing 121_data table');
                draw121DataTable(data);
                return;
            } else if (currentView === 'message') {
                console.log('DEBUG: Drawing message table');
                // Используем HEX сообщение из загруженного файла
                const hexFromFile = data.hex_message || '';
                console.log('DEBUG: Using hex_message from file:', hexFromFile);
                drawMessageTable(hexFromFile);
                return;
            } else if (currentView !== 'phase') {
                // Для других режимов просто очищаем canvas
                const width = canvas.width;
                const height = canvas.height;
                ctx.clearRect(0, 0, width, height);

                // Показываем заглушку
                ctx.fillStyle = '#666';
                ctx.font = '16px Arial';
                ctx.fillText('View mode: ' + currentView, width/2 - 80, height/2);
                ctx.font = '12px Arial';
                ctx.fillText('Chart implementation pending', width/2 - 80, height/2 + 25);
                return;
            }

            // Оригинальный график фазы (только для режима 'phase')
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            // Сетка
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии сетки
            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии сетки с временными метками
            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';  // увеличен с 10px
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();

                // Временные метки с учетом масштаба
                const scaledDuration = MESSAGE_DURATION_MS * (currentTimeScale / 100);
                let startOffset;
                if (currentTimeScale === 100) {
                    startOffset = 0; // При 100% начинаем с 0мс
                } else {
                    // Для других масштабов: начало модуляции - смещение в зависимости от масштаба
                    const preambleMs = data?.preamble_ms || 10.0; // fallback к baseline_ms
                    const offsetMs = getOffsetForScale(currentTimeScale);
                    startOffset = Math.max(0, preambleMs - offsetMs);
                }
                const timeMs = (startOffset + i * scaledDuration / 8).toFixed(1);
                ctx.fillText(timeMs, x - 10, height - 5);
            }

            // Нулевая линия
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();

            // Пунктирные линии на уровне ±1.1 радиан
            ctx.strokeStyle = '#999999';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]); // Пунктир: 5px линия, 5px пропуск

            // Линия +1.1 рад (фиксированный масштаб 1.25 рад)
            const y_plus_1_1 = height / 2 - (1.1 / 1.25) * (height / 2);
            ctx.beginPath();
            ctx.moveTo(0, y_plus_1_1);
            ctx.lineTo(width, y_plus_1_1);
            ctx.stroke();

            // Линия -1.1 рад
            const y_minus_1_1 = height / 2 + (1.1 / 1.25) * (height / 2);
            ctx.beginPath();
            ctx.moveTo(0, y_minus_1_1);
            ctx.lineTo(width, y_minus_1_1);
            ctx.stroke();

            // Возвращаем сплошную линию для дальнейшего рисования
            ctx.setLineDash([]);

            // Y-axis labels (будут обновлены после определения масштаба)
            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';

            // График данных
            if (data) {
                let phaseData = (data.phase_data || []).map(v => Number(v));
                let xsData = (data.xs_fm_ms || []).map(v => Number(v));

                // Нормализация времени: если максимум < 10, значит это секунды → переводим в мс
                if (xsData.length && xsData.reduce((max, v) => Math.max(max, v), -Infinity) <= 10) {
                    console.warn('DEBUG: xs_fm_ms appears to be in seconds — converting to ms');
                    xsData = xsData.map(v => v * 1000);
                }

                console.log('DEBUG drawChart: phaseData length =', phaseData.length);
                console.log('DEBUG drawChart: phaseData type check:', typeof phaseData[0], phaseData[0]);
                console.log('DEBUG drawChart: xsData type check:', typeof xsData[0], xsData[0]);

                // Проверяем типы данных и конвертируем при необходимости
                for (let i = 0; i < Math.min(5, phaseData.length); i++) {
                    if (typeof phaseData[i] !== 'number') {
                        console.warn(`DEBUG: phaseData[${i}] is not a number:`, typeof phaseData[i], phaseData[i]);
                        phaseData[i] = parseFloat(phaseData[i]) || 0;
                    }
                    if (typeof xsData[i] !== 'number') {
                        console.warn(`DEBUG: xsData[${i}] is not a number:`, typeof xsData[i], xsData[i]);
                        xsData[i] = parseFloat(xsData[i]) || 0;
                    }
                }
                console.log('DEBUG drawChart: xsData length =', xsData.length);
                if (phaseData.length > 0) {
                    console.log('DEBUG drawChart: phaseData sample =', phaseData.slice(0, 5));
                }
                if (xsData.length > 0) {
                    console.log('DEBUG: xsData sample =', xsData.slice(0, 5));
                }

                console.log('DEBUG: Checking if can draw graph:', phaseData ? `phaseData.length=${phaseData.length}` : 'phaseData is null/undefined');

                if (phaseData && phaseData.length > 1) {
                    console.log('DEBUG: Starting to draw REAL phase graph with', phaseData.length, 'points');

                    // Фиксированный масштаб оси Y: ±1.25 радиан
                    const phaseScale = 1.25;

                    console.log(`DEBUG: Using fixed phase scale: ±${phaseScale} rad`);

                    // Обновляем Y-axis метки с фиксированным масштабом
                    ctx.fillStyle = '#6c757d';
                    ctx.font = '12px Arial';
                    ctx.fillText(`+${phaseScale.toFixed(2)} rad`, 5, 15);
                    ctx.fillText('0', 5, height / 2 + 4);
                    ctx.fillText(`-${phaseScale.toFixed(2)} rad`, 5, height - 10);

                    ctx.strokeStyle = '#FF0000'; // Красный для реального графика фазы
                    ctx.lineWidth = 2;
                    ctx.beginPath();

                    // Правильная логика на основе времени
                    console.log('DEBUG: Drawing time-based graph');

                    // Проверяем наличие временных данных
                    if (!xsData || xsData.length === 0 || xsData.length !== phaseData.length) {
                        console.warn('DEBUG: Missing or mismatched time data (xsData), falling back to index-based drawing');
                        const pointsToShow = Math.min(100, phaseData.length);
                        let minY = Infinity, maxY = -Infinity;
                        for (let i = 0; i < pointsToShow; i++) {
                            const x = (i / pointsToShow) * width;
                            const y = height / 2 - (phaseData[i] / phaseScale) * (height / 2);
                            minY = Math.min(minY, y);
                            maxY = Math.max(maxY, y);
                            if (i === 0) {
                                ctx.moveTo(x, y);
                            } else {
                                ctx.lineTo(x, y);
                            }
                        }
                        console.log(`DEBUG fallback: Drew ${pointsToShow} points`);
                        ctx.stroke();
                        return;
                    }

                    // Временное окно для отображения
                    let windowStart;
                    if (currentTimeScale === 100) {
                        windowStart = 0; // При 100% начинаем с 0мс
                    } else {
                        // Для других масштабов: начало модуляции - смещение в зависимости от масштаба
                        const preambleMs = data.preamble_ms || 10.0; // fallback к baseline_ms из кода
                        const offsetMs = getOffsetForScale(currentTimeScale);
                        windowStart = Math.max(0, preambleMs - offsetMs);
                    }
                    const windowDuration = MESSAGE_DURATION_MS * (currentTimeScale / 100.0); // 440 * scale/100
                    const windowEnd = windowStart + windowDuration;

                    console.log(`DEBUG time window: start=${windowStart}ms, duration=${windowDuration}ms, end=${windowEnd}ms, scale=${currentTimeScale}%`);

                    // Фильтруем точки по временному окну
                    const filteredPoints = [];
                    for (let i = 0; i < phaseData.length; i++) {
                        const timeMs = xsData[i];
                        if (Number.isFinite(timeMs) && timeMs >= windowStart && timeMs <= windowEnd) {
                            filteredPoints.push({
                                time: timeMs,
                                phase: phaseData[i],
                                index: i
                            });
                        }
                    }

                    console.log(`DEBUG: Filtered ${filteredPoints.length} points from ${phaseData.length} total points in time range [${windowStart}, ${windowEnd}]ms`);

                    if (filteredPoints.length === 0) {
                        console.warn('DEBUG: No points found in specified time window - falling back to index-based drawing');
                        const pointsToShow = Math.min(100, phaseData.length);
                        let minY = Infinity, maxY = -Infinity;
                        for (let i = 0; i < pointsToShow; i++) {
                            const x = (i / pointsToShow) * width;
                            const y = height / 2 - (phaseData[i] / phaseScale) * (height / 2);
                            minY = Math.min(minY, y);
                            maxY = Math.max(maxY, y);
                            if (i === 0) {
                                ctx.moveTo(x, y);
                            } else {
                                ctx.lineTo(x, y);
                            }
                        }
                        ctx.stroke();
                        console.log(`DEBUG fallback: Drew ${pointsToShow} points`);
                        return;
                    }

                    // Даунсэмплинг до ~1000 точек для производительности (основное исправление)
                    const targetPoints = 1000; // Настраиваемое значение; достаточно для плавного графика при любом размере холста
                    const step = Math.max(1, Math.ceil(filteredPoints.length / targetPoints));
                    console.log(`DEBUG: Downsampling with step=${step} (target ~${targetPoints} points, actual ~${Math.floor(filteredPoints.length / step)})`);

                    let minY = Infinity, maxY = -Infinity;
                    let firstPoint = true;

                    for (let j = 0; j < filteredPoints.length; j += step) {
                        const point = filteredPoints[j];

                        // Преобразуем время в пиксели
                        const normalizedTime = (point.time - windowStart) / windowDuration;
                        const x = normalizedTime * width;
                        const y = height / 2 - (point.phase / phaseScale) * (height / 2);

                        minY = Math.min(minY, y);
                        maxY = Math.max(maxY, y);

                        if (firstPoint) {
                            ctx.moveTo(x, y);
                            console.log(`DEBUG line start: x=${x.toFixed(1)}, y=${y.toFixed(1)}, time=${point.time.toFixed(1)}ms, value=${point.phase.toFixed(8)}`);
                            firstPoint = false;
                        } else {
                            ctx.lineTo(x, y);
                        }

                        // Логируем первые 5 точек
                        if (j / step < 5) {
                            console.log(`DEBUG line point ${j / step}: x=${x.toFixed(1)}, y=${y.toFixed(1)}, time=${point.time.toFixed(1)}ms, value=${point.phase.toFixed(8)}, normalized_time=${normalizedTime.toFixed(4)}`);
                        }
                    }

                    console.log(`DEBUG line coordinates: minY=${minY.toFixed(1)}, maxY=${maxY.toFixed(1)}, range=${(maxY-minY).toFixed(1)}`);
                    console.log(`DEBUG phaseScale used: ${phaseScale.toFixed(6)}`);
                    ctx.stroke();
                    console.log('DEBUG: Time-based phase graph drawing completed');
                } else {
                    console.log('DEBUG: Phase graph NOT drawn - insufficient data or condition not met');
                }
            }
        }       
        

        function showMessageTable(hexMessage) {
            console.log('DEBUG: showMessageTable called with:', hexMessage);

            // Очищаем canvas и показываем HTML таблицу
            const canvas = document.getElementById('phaseChart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Создаем контейнер для HTML таблицы
            const chartContainer = document.querySelector('.chart-container');

            if (!hexMessage || hexMessage.trim() === '') {
                chartContainer.innerHTML = `
                    <div class="message-table-container">
                        <div style="text-align: center; padding: 50px; color: #666;">
                            <h3>No message decoded</h3>
                            <p>Please load a .cf32 file using the File button</p>
                            <p>to decode EPIRB/ELT beacon message</p>
                        </div>
                    </div>
                `;
                return;
            }

            // Данные таблицы (те же что были в canvas версии)
            const tableData = [
                ['1-15', '111111111111111', 'Bit-sync pattern', 'Valid'],
                ['16-24', '000101111', 'Frame-sync pattern', 'Normal Operation'],
                ['25', '1', 'Format Flag', 'Long Format'],
                ['26', '0', 'Protocol Flag', 'Standard/National/RLS'],
                ['27-36', '1000000000', 'Country Code', '512 - Russia'],
                ['37-40', '0000', 'Protocol Code', 'Avionic'],
                ['41-64', '000000100000000000000000', 'Test Data', '0x020000'],
                ['65-74', '0111111111', 'Latitude (PDF-1)', 'Default value'],
                ['75-85', '01111111111', 'Longitude (PDF-1)', 'Default value'],
                ['86-106', '110000100000101101111', 'BCH PDF-1', '0x1820B7'],
                ['107-110', '1000', 'Fixed (1101)', 'Invalid (1000)'],
                ['111', '0', 'Position source', 'External/Unknown'],
                ['112', '0', '121.5 MHz Device', 'Not included'],
                ['113-122', '1111100000', 'Latitude (PDF-2)', 'bin 1111100000'],
                ['123-132', '1111100000', 'Longitude (PDF-2)', 'bin 1111100000'],
                ['133-144', '111001101100', 'BCH PDF-2', '0xE6C']
            ];

            // Создаем HTML таблицу
            let tableHtml = `
                <div class="message-table-container">
                    <div class="message-table-header">
                        <h3>EPIRB/ELT Beacon Message Decoder</h3>
                    </div>
                    <table class="message-table">
                        <thead>
                            <tr>
                                <th style="width: 90px;">Bit Range</th>
                                <th style="width: 200px;">Binary Content</th>
                                <th style="width: 220px;">Field Name</th>
                                <th>Decoded Value</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            for (let i = 0; i < tableData.length; i++) {
                const row = tableData[i];
                tableHtml += `
                    <tr>
                        <td>${row[0]}</td>
                        <td class="binary-content">${row[1]}</td>
                        <td class="field-name">${row[2]}</td>
                        <td>${row[3]}</td>
                    </tr>
                `;
            }

            tableHtml += `
                        </tbody>
                    </table>
                    <div class="message-table-footer">
                        <div>COSPAS-SARSAT 406 MHz Beacon Message (144 bits)</div>
                        <div>Protocol: Long Format, Standard Location</div>
                    </div>
                </div>
            `;

            chartContainer.innerHTML = tableHtml;
        }

        function showSumTable(data) {
            console.log('DEBUG: showSumTable called');

            // Очищаем canvas и показываем HTML таблицу
            const canvas = document.getElementById('phaseChart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Создаем контейнер для HTML таблицы
            const chartContainer = document.querySelector('.chart-container');

            // Данные для 406 MHz таблицы из скриншота
            const params406 = [
                ['Frequency, kHz', '40600.000', '40600.000', '0.000', '406025.954', '0.000'],
                ['+Phase deviation, rad', '1.00', '1.20', '0.00', '1.09', '0.00'],
                ['-Phase deviation, rad', '-1.00', '-1.20', '0.00', '-1.11', '0.00'],
                ['Phase time rise, mcs', '50.00', '250.00', '0.00', '111.07', '0.00'],
                ['Phase time fall, mcs', '50.00', '250.00', '0.00', '70.36', '0.00'],
                ['Power, Wt', '3.16', '7.94', '0.00', '0.00', '0.00'],
                ['Power rise, ms', '0.00', '0.00', '0.00', '0.00', '0.00'],
                ['Bit Rate, bps', '396.00', '404.00', '0.00', '400.01', '0.00'],
                ['Asymmetry, %', '0.00', '5.00', '0.00', '0.00', '0.00'],
                ['CW Preamble, ms', '158.40', '161.60', '0.00', '157.53', '0.00'],
                ['Total burst duration, ms', '435.60', '520.00', '0.00', '518.76', '0.00'],
                ['Repetition period, s', '47.50', '52.50', '0.00', '0.00', '0.00'],
                ['Delta Rep. period, s', '4.00', '0.00', '0.00', '0.00', '0.00']
            ];

            // Данные для 121.5 MHz таблицы
            const params121 = [
                ['Carrier Frequency, Hz', '0'],
                ['Power, mW', '0.0'],
                ['Sweep Period, sec', '0.0'],
                ['Modulation Index, %', '0']
            ];

            // Создаем HTML - единый контейнер со всеми таблицами
            let tableHtml = `
                <div class="sum-table-container">
                    <table class="sum-params-table">
                        <tr>
                            <td colspan="6" class="header-406">406 MHz Transmitter Parameters</td>
                        </tr>
                        <tr>
                            <td rowspan="2" class="subheader-empty"></td>
                            <td colspan="2" class="subheader">Limits</td>
                            <td rowspan="2" class="subheader" style="width: 80px;">min</td>
                            <td rowspan="2" class="subheader" style="width: 100px;">Current</td>
                            <td rowspan="2" class="subheader" style="width: 100px;">Measured<br/>Info</td>
                        </tr>
                        <tr>
                            <td class="subheader" style="width: 80px;">min</td>
                            <td class="subheader" style="width: 80px;">max</td>
                        </tr>
            `;

            // Добавляем строки параметров 406 MHz
            for (let i = 0; i < params406.length; i++) {
                const row = params406[i];
                tableHtml += `
                    <tr class="param-row">
                        <td class="param-name">${row[0]}</td>
                        <td>${row[1]}</td>
                        <td>${row[2]}</td>
                        <td>${row[3]}</td>
                        <td>${row[4]}</td>
                        <td>${row[5]}</td>
                    </tr>
                `;
            }

            // Добавляем 121.5 MHz секцию
            tableHtml += `
                        <tr>
                            <td colspan="6" class="header-121">121.5 MHz Transmitter Parameters</td>
                        </tr>
            `;

            for (let i = 0; i < params121.length; i++) {
                const row = params121[i];
                tableHtml += `
                    <tr class="param-row">
                        <td class="param-name">${row[0]}</td>
                        <td colspan="5">${row[1]}</td>
                    </tr>
                `;
            }

            tableHtml += `
                    </table>
            `;

            // Добавляем нашу таблицу Message внутри того же контейнера
            const hexMessage = data.hex_message || 'DEFAULT_HEX';
            if (hexMessage) {
                // Данные таблицы Message (те же что в showMessageTable)
                const messageTableData = [
                    ['1-15', '111111111111111', 'Bit-sync pattern', 'Valid'],
                    ['16-24', '000101111', 'Frame-sync pattern', 'Normal Operation'],
                    ['25', '1', 'Format Flag', 'Long Format'],
                    ['26', '0', 'Protocol Flag', 'Standard/National/RLS'],
                    ['27-36', '1000000000', 'Country Code', '512 - Russia'],
                    ['37-40', '0000', 'Protocol Code', 'Avionic'],
                    ['41-64', '000000100000000000000000', 'Test Data', '0x020000'],
                    ['65-74', '0111111111', 'Latitude (PDF-1)', 'Default value'],
                    ['75-85', '01111111111', 'Longitude (PDF-1)', 'Default value'],
                    ['86-106', '110000100000101101111', 'BCH PDF-1', '0x1820B7'],
                    ['107-110', '1000', 'Fixed (1101)', 'Invalid (1000)'],
                    ['111', '0', 'Position source', 'External/Unknown'],
                    ['112', '0', '121.5 MHz Device', 'Not included'],
                    ['113-122', '1111100000', 'Latitude (PDF-2)', 'bin 1111100000'],
                    ['123-132', '1111100000', 'Longitude (PDF-2)', 'bin 1111100000'],
                    ['133-144', '111001101100', 'BCH PDF-2', '0xE6C']
                ];

                // Добавляем таблицу Message в нашем стиле внутри того же контейнера
                tableHtml += `
                    <div style="margin-top: 10px;">
                        <div class="message-table-header">
                            <h3>EPIRB/ELT Beacon Message Decoder</h3>
                        </div>
                        <table class="message-table">
                            <thead>
                                <tr>
                                    <th style="width: 90px;">Bit Range</th>
                                    <th style="width: 200px;">Binary Content</th>
                                    <th style="width: 220px;">Field Name</th>
                                    <th>Decoded Value</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                for (let i = 0; i < messageTableData.length; i++) {
                    const row = messageTableData[i];
                    tableHtml += `
                        <tr>
                            <td>${row[0]}</td>
                            <td class="binary-content">${row[1]}</td>
                            <td class="field-name">${row[2]}</td>
                            <td>${row[3]}</td>
                        </tr>
                    `;
                }

                tableHtml += `
                            </tbody>
                        </table>
                        <div class="message-table-footer">
                            <div>COSPAS-SARSAT 406 MHz Beacon Message (144 bits)</div>
                            <div>Protocol: Long Format, Standard Location</div>
                        </div>
                    </div>
                `;
            }

            // Закрываем единый контейнер
            tableHtml += `</div>`;

            chartContainer.innerHTML = tableHtml;
        }

        function show121DataTable(data) {
            console.log('DEBUG: show121DataTable called');

            // Очищаем canvas и показываем HTML таблицу
            const canvas = document.getElementById('phaseChart');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Создаем контейнер для HTML таблицы
            const chartContainer = document.querySelector('.chart-container');

            // Данные таблицы (те же что были в canvas версии)
            const tableData = [
                ['Carrier Frequency, Hz', '0', 'Low Sweep Frequency, Hz', '0'],
                ['Power, mW', '0.0', 'High Sweep Frequency, Hz', '0'],
                ['Sweep Period, sec', '0.0', 'Sweep Range, Hz', '0'],
                ['Modulation Index, %', '0', '', '']
            ];

            // Создаем HTML таблицу
            let tableHtml = `
                <div class="data121-table-container">
                    <div class="data121-table-header">
                        <h3>121.5 MHz Transmitter Parameters</h3>
                    </div>
                    <table class="data121-table">
                        <tbody>
            `;

            for (let i = 0; i < tableData.length; i++) {
                const row = tableData[i];
                tableHtml += '<tr>';
                for (let j = 0; j < row.length; j++) {
                    if (row[j]) {
                        // Названия параметров (четные колонки) - жирный шрифт
                        const cssClass = (j % 2 === 0) ? 'param-name' : 'param-value';
                        tableHtml += `<td class="${cssClass}">${row[j]}</td>`;
                    } else {
                        tableHtml += '<td></td>';
                    }
                }
                tableHtml += '</tr>';
            }

            tableHtml += `
                        </tbody>
                    </table>
                    <div class="data121-table-footer">
                        <div>Emergency Locator Transmitter (ELT) operating on 121.5 MHz</div>
                        <div>Used for aircraft emergency location and rescue operations</div>
                    </div>
                </div>
            `;

            chartContainer.innerHTML = tableHtml;
        }

        function updateMessageInfo(data) {
            // Обновляем только основную информацию без canvas
            document.getElementById('protocol').textContent = data.protocol;
            document.getElementById('date').textContent = data.date;
            document.getElementById('beaconModel').textContent = data.beacon_model;
            document.getElementById('beaconFreq').textContent = data.beacon_frequency.toFixed(1);

            // Показываем HEX сообщение если есть, иначе обычное сообщение
            if (data.hex_message && data.hex_message !== '') {
                document.getElementById('message').textContent = `HEX: ${data.hex_message}`;
            } else {
                document.getElementById('message').textContent = data.message;
            }

            // Обновление фазовых значений
            document.getElementById('phasePlus').textContent = (data.phase_pos_rad * 57.2958).toFixed(2);
            document.getElementById('phaseMinus').textContent = (data.phase_neg_rad * 57.2958).toFixed(2);
            document.getElementById('tRise').textContent = data.t_rise_mcs.toFixed(1);
            document.getElementById('tFall').textContent = data.t_fall_mcs.toFixed(1);
        }

        function updateStats(data) {
            const statsHtml = `
                <div class="stat-row"><span class="stat-label">FS1,Hz</span><span class="stat-value">${data.fs1_hz.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">FS2,Hz</span><span class="stat-value">${data.fs2_hz.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">FS3,Hz</span><span class="stat-value">${data.fs3_hz.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Phase+,rad</span><span class="stat-value">${data.phase_pos_rad.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Phase-,rad</span><span class="stat-value">${data.phase_neg_rad.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">TRise,mcs</span><span class="stat-value">${data.t_rise_mcs.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">TFall,mcs</span><span class="stat-value">${data.t_fall_mcs.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">P,Wt</span><span class="stat-value">${data.p_wt.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Prise,ms</span><span class="stat-value">${data.prise_ms.toFixed(1)}</span></div>
                <div class="stat-row"><span class="stat-label">BitRate,bps</span><span class="stat-value">${data.bitrate_bps.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Symmetry,%</span><span class="stat-value">${data.symmetry_pct.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Preamble,ms</span><span class="stat-value">${data.preamble_ms.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">Total,ms</span><span class="stat-value">${data.total_ms.toFixed(3)}</span></div>
                <div class="stat-row"><span class="stat-label">RepPeriod,s</span><span class="stat-value">${data.rep_period_s.toFixed(3)}</span></div>
            `;
            document.getElementById('statsContent').innerHTML = statsHtml;
        }


        function updateDisplay(data) {
            console.log('=== updateDisplay called ===');
            console.log('currentView:', currentView);
            console.log('data.phase_data length:', data.phase_data ? data.phase_data.length : 'null');

            // Восстанавливаем canvas для обычных режимов если он был заменен HTML таблицей
            const chartContainer = document.querySelector('.chart-container');
            if (!chartContainer.querySelector('#phaseChart')) {
                chartContainer.innerHTML = '<canvas id="phaseChart"></canvas>';
                resizeCanvas(); // Переинициализируем размеры canvas
            }

            // Обновление основной информации
            document.getElementById('protocol').textContent = data.protocol;
            document.getElementById('date').textContent = data.date;
            document.getElementById('beaconModel').textContent = data.beacon_model;
            document.getElementById('beaconFreq').textContent = data.beacon_frequency.toFixed(1);
            // Показываем HEX сообщение если есть, иначе обычное сообщение
            if (data.hex_message && data.hex_message !== '') {
                document.getElementById('message').textContent = `HEX: ${data.hex_message}`;
            } else {
                document.getElementById('message').textContent = data.message;
            }

            // Обновление фазовых значений
            document.getElementById('phasePlus').textContent = (data.phase_pos_rad * 57.2958).toFixed(2);
            document.getElementById('phaseMinus').textContent = (data.phase_neg_rad * 57.2958).toFixed(2);
            document.getElementById('tRise').textContent = data.t_rise_mcs.toFixed(1);
            document.getElementById('tFall').textContent = data.t_fall_mcs.toFixed(1);

            // Обновление статистики
            updateStats(data);

            // Автообновление отключено - синхронизация с сервером не требуется
            console.log('Display updated, auto-update remains disabled');

            // Обновляем график только для обычных режимов (не специальных)
            if (currentView !== 'message' && currentView !== '121_data') {
                console.log('DEBUG: phase_data:', data.phase_data ? data.phase_data.length : 'null');
                console.log('DEBUG: xs_fm_ms:', data.xs_fm_ms ? data.xs_fm_ms.length : 'null');
                if (data.phase_data && data.phase_data.length > 0) {
                    console.log('DEBUG: phase_data sample:', data.phase_data.slice(0, 5));
                }
                if (data.xs_fm_ms && data.xs_fm_ms.length > 0) {
                    console.log('DEBUG: xs_fm_ms sample:', data.xs_fm_ms.slice(0, 5));
                }

                drawChart(data);
            }
        }

        function drawRiseFallChart(data) {
            const width = canvas.width;
            const height = canvas.height;
            const midY = height / 2;

            ctx.clearRect(0, 0, width, height);

            // === ВЕРХНИЙ ГРАФИК: Fig.6 Modulation index ===

            // Сетка для верхнего графика
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            // Горизонтальные линии верхнего графика
            for (let i = 0; i <= 5; i++) {
                const y = (midY / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии с временными метками
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, midY);
                ctx.stroke();

                // Временные метки для верхнего графика
                const timeMs = (i * 5).toFixed(0);
                ctx.fillText(timeMs, x - 8, midY - 5);
            }

            // Y-axis labels для Modulation index (Phase)
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            ctx.fillText('1.3', 5, 15);
            ctx.fillText('Ph+,rad', 5, 28);
            ctx.fillText('1.1', 5, midY * 0.25);
            ctx.fillText('0.9', 5, midY * 0.4);
            ctx.fillText('1.1', 5, midY * 0.6);
            ctx.fillText('1.3', 5, midY * 0.75);
            ctx.fillText('Ph-,rad', 5, midY - 5);

            // График Ph+ (верхняя синусоида)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 8 * Math.PI; // 4 периода на 40ms
                const y = midY * 0.15 + Math.sin(t) * midY * 0.08; // колебания в районе 1.1
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // График Ph- (нижняя синусоида)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 8 * Math.PI; // 4 периода на 40ms
                const y = midY * 0.85 + Math.sin(t) * midY * 0.08; // колебания в районе -1.1
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Подпись времени
            ctx.fillText('t,m', width - 25, midY - 5);

            // Заголовок верхнего графика
            ctx.fillStyle = '#495057';
            ctx.font = '12px Arial';
            ctx.fillText('Fig.6 Modulation index', width/2 - 60, midY - 15);

            // === РАЗДЕЛИТЕЛЬ ===
            ctx.strokeStyle = '#adb5bd';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(width, midY);
            ctx.stroke();

            // === НИЖНИЙ ГРАФИК: Fig.7 Rise and Fall Times ===

            // Горизонтальные линии нижнего графика
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const y = midY + (midY / 5) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии нижнего графика
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, midY);
                ctx.lineTo(x, height);
                ctx.stroke();

                // Временные метки для нижнего графика
                const timeMs = (i * 5).toFixed(0);
                ctx.fillText(timeMs, x - 8, height - 5);
            }

            // Центральная линия нижнего графика (ось 0)
            const centerY = midY + midY/2;
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();

            // Y-axis labels для Rise/Fall Times
            ctx.fillStyle = '#6c757d';
            ctx.font = '11px Arial';
            ctx.fillText('300', 5, midY + 15);
            ctx.fillText('Tr,mcs', 5, midY + 28);
            ctx.fillText('150', 5, centerY - midY/5);
            ctx.fillText('0', 5, centerY + 4);
            ctx.fillText('150', 5, centerY + midY/5);
            ctx.fillText('300', 5, height - 15);
            ctx.fillText('Tf,mcs', 5, height - 2);

            // График Tr (верхняя часть - Rise time)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 12 * Math.PI; // больше частота
                const baseY = centerY - midY * 0.25; // в районе 150 mcs
                const y = baseY + Math.sin(t) * 15 + Math.sin(t * 2.3) * 8; // сложный сигнал
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // График Tf (нижняя часть - Fall time)
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < width; i++) {
                const x = i;
                const t = (i / width) * 12 * Math.PI;
                const baseY = centerY + midY * 0.25; // в районе -150 mcs
                const y = baseY + Math.sin(t + Math.PI/3) * 15 + Math.sin(t * 1.7) * 8; // сложный сигнал со сдвигом
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Подпись времени
            ctx.fillText('t,m', width - 25, height - 5);

            // Заголовок нижнего графика
            ctx.fillStyle = '#495057';
            ctx.font = '12px Arial';
            ctx.fillText('Fig.7 Rise and Fall Times', width/2 - 70, height - 15);
        }

        function drawFrequencyChart(data) {
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            // Сетка
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;

            for (let i = 0; i <= 10; i++) {
                const y = (height / 10) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            ctx.fillStyle = '#6c757d';
            ctx.font = '12px Arial';
            for (let i = 0; i <= 8; i++) {
                const x = (width / 8) * i;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();

                const timeS = (i * 5).toFixed(0);
                ctx.fillText(timeS + ' s', x - 10, height - 5);
            }

            // Y-axis labels для частоты
            ctx.fillText('406.030 MHz', 5, 20);
            ctx.fillText('406.025 MHz', 5, height/2);
            ctx.fillText('406.020 MHz', 5, height - 10);

            // График частоты
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < width; i += 2) {
                const x = i;
                const y = height/2 + Math.sin(i * 0.005) * 20 + Math.random() * 5;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        function draw121DataTable(data) {
            console.log('DEBUG: draw121DataTable function called!');
            const width = canvas.width;
            const height = canvas.height;
            console.log('DEBUG: Canvas dimensions:', width, 'x', height);

            // Убедимся, что получаем контекст
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);

            // Заголовок таблицы
            ctx.fillStyle = '#5a9bd4';
            ctx.fillRect(0, 0, width, 40);

            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Arial';
            ctx.fillText('121.5 MHz Transmitter Parameters', width/2 - 120, 25);

            // Создание таблицы
            const tableData = [
                ['Carrier Frequency, Hz', '0', 'Low Sweep Frequency, Hz', '0'],
                ['Power, mW', '0.0', 'High Sweep Frequency, Hz', '0'],
                ['Sweep Period, sec', '0.0', 'Sweep Range, Hz', '0'],
                ['Modulation Index, %', '0', '', '']
            ];

            const rowHeight = 35;
            const colWidth = width / 4;
            const startY = 50;

            // Рисуем границы таблицы
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;

            // Горизонтальные линии
            for (let i = 0; i <= tableData.length; i++) {
                const y = startY + i * rowHeight;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Вертикальные линии
            for (let i = 0; i <= 4; i++) {
                const x = i * colWidth;
                ctx.beginPath();
                ctx.moveTo(x, startY);
                ctx.lineTo(x, startY + tableData.length * rowHeight);
                ctx.stroke();
            }

            // Заполняем таблицу данными
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';

            for (let row = 0; row < tableData.length; row++) {
                for (let col = 0; col < tableData[row].length; col++) {
                    if (tableData[row][col]) {
                        const x = col * colWidth + 10;
                        const y = startY + row * rowHeight + 22;

                        // Жирный шрифт для названий параметров (левые колонки)
                        if (col % 2 === 0) {
                            ctx.font = 'bold 12px Arial';
                        } else {
                            ctx.font = '12px Arial';
                        }

                        ctx.fillText(tableData[row][col], x, y);
                    }
                }
            }

            // Дополнительная информация внизу
            ctx.fillStyle = '#666';
            ctx.font = '11px Arial';
            ctx.fillText('Emergency Locator Transmitter (ELT) operating on 121.5 MHz', 20, height - 30);
            ctx.fillText('Used for aircraft emergency location and rescue operations', 20, height - 15);
        }

        function drawMessageTable(hexMessage) {
            console.log('DEBUG: drawMessageTable function called with:', hexMessage);
            console.log('DEBUG: currentView at drawMessageTable:', currentView);
            const width = canvas.width;
            const height = canvas.height;
            console.log('DEBUG: Canvas dimensions:', width, 'x', height);

            // Убедимся, что получаем контекст
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);

            // Если нет HEX сообщения из файла, показываем инструкцию
            if (!hexMessage || hexMessage.trim() === '') {
                ctx.fillStyle = '#666';
                ctx.font = '16px Arial';
                ctx.fillText('No message decoded', width/2 - 80, height/2 - 20);
                ctx.font = '12px Arial';
                ctx.fillText('Please load a .cf32 file using the File button', width/2 - 140, height/2 + 10);
                ctx.fillText('to decode EPIRB/ELT beacon message', width/2 - 110, height/2 + 30);
                return;
            }

            // Заголовок таблицы
            ctx.fillStyle = '#1976D2';
            ctx.fillRect(0, 0, width, 45);

            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Arial';
            ctx.fillText('EPIRB/ELT Beacon Message Decoder', width/2 - 130, 18);
            ctx.font = '11px Arial';
            ctx.fillText('HEX: ' + hexMessage, width/2 - 180, 35);

            // Заголовки колонок
            const headers = ['Bit Range', 'Binary Content', 'Field Name', 'Decoded Value'];
            const colWidths = [90, 200, 220, width - 510];
            const rowHeight = 26;
            const startY = 55;
            const headerHeight = 28;

            // Рисуем заголовки
            ctx.fillStyle = '#E0E0E0';
            ctx.fillRect(0, startY, width, headerHeight);

            ctx.fillStyle = '#333';
            ctx.font = 'bold 13px Arial';
            let xPos = 5;
            for (let i = 0; i < headers.length; i++) {
                ctx.fillText(headers[i], xPos, startY + 19);
                xPos += colWidths[i];
            }

            // Получаем данные декодирования через API
            // Пока используем заглушку - в реальности нужно будет передавать через API
            const tableData = [
                ['1-15', '111111111111111', 'Bit-sync pattern', 'Valid'],
                ['16-24', '000101111', 'Frame-sync pattern', 'Normal Operation'],
                ['25', '1', 'Format Flag', 'Long Format'],
                ['26', '0', 'Protocol Flag', 'Standard/National/RLS'],
                ['27-36', '1000000000', 'Country Code', '512 - Russia'],
                ['37-40', '0000', 'Protocol Code', 'Avionic'],
                ['41-64', '000000100000000000000000', 'Test Data', '0x020000'],
                ['65-74', '0111111111', 'Latitude (PDF-1)', 'Default value'],
                ['75-85', '01111111111', 'Longitude (PDF-1)', 'Default value'],
                ['86-106', '110000100000101101111', 'BCH PDF-1', '0x1820B7'],
                ['107-110', '1000', 'Fixed (1101)', 'Invalid (1000)'],
                ['111', '0', 'Position source', 'External/Unknown'],
                ['112', '0', '121.5 MHz Device', 'Not included'],
                ['113-122', '1111100000', 'Latitude (PDF-2)', 'bin 1111100000'],
                ['123-132', '1111100000', 'Longitude (PDF-2)', 'bin 1111100000'],
                ['133-144', '111001101100', 'BCH PDF-2', '0xE6C']
            ];

            // Рисуем строки таблицы
            ctx.font = '12px Arial';
            let currentY = startY + headerHeight;

            for (let row = 0; row < tableData.length; row++) {
                // Чередующиеся цвета фона
                if (row % 2 === 0) {
                    ctx.fillStyle = '#F5F5F5';
                    ctx.fillRect(0, currentY, width, rowHeight);
                }

                // Рисуем данные
                ctx.fillStyle = '#333';
                xPos = 5;
                for (let col = 0; col < tableData[row].length; col++) {
                    // Особый стиль для Binary Content
                    if (col === 1) {
                        ctx.font = '11px monospace';
                        ctx.fillStyle = '#0066CC';
                    } else if (col === 2) {
                        ctx.font = 'bold 12px Arial';
                        ctx.fillStyle = '#333';
                    } else {
                        ctx.font = '12px Arial';
                        ctx.fillStyle = '#333';
                    }

                    // Обрезаем текст если он слишком длинный
                    const text = tableData[row][col];
                    const maxWidth = colWidths[col] - 10;
                    let displayText = text;

                    if (ctx.measureText(text).width > maxWidth) {
                        while (ctx.measureText(displayText + '...').width > maxWidth && displayText.length > 0) {
                            displayText = displayText.slice(0, -1);
                        }
                        displayText += '...';
                    }

                    ctx.fillText(displayText, xPos, currentY + 16);
                    xPos += colWidths[col];
                }

                // Горизонтальная линия
                ctx.strokeStyle = '#DDD';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(0, currentY + rowHeight);
                ctx.lineTo(width, currentY + rowHeight);
                ctx.stroke();

                currentY += rowHeight;
            }

            // Нижняя информация
            ctx.fillStyle = '#666';
            ctx.font = '11px Arial';
            ctx.fillText('COSPAS-SARSAT 406 MHz Beacon Message (144 bits)', 20, height - 30);
            ctx.fillText('Protocol: Long Format, Standard Location', 20, height - 15);
        }

        async function fetchData() {
            console.log('=== fetchData called ===');
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                console.log('fetchData: received data, currentView:', currentView);

                // Обрабатываем специальные режимы
                if (currentView === 'message') {
                    // Для режима message показываем HTML таблицу декодирования и обновляем только Current
                    console.log('DEBUG: Showing HTML message table for hex:', data.hex_message);
                    showMessageTable(data.hex_message || '');
                    // Обновляем только Current таблицу, не трогая canvas
                    updateStats(data);
                    updateMessageInfo(data);
                } else if (currentView === '121_data') {
                    // Для режима 121 показываем HTML таблицу 121 и обновляем только Current
                    console.log('DEBUG: Showing HTML 121 table');
                    show121DataTable(data);
                    // Обновляем только Current таблицу, не трогая canvas
                    updateStats(data);
                    updateMessageInfo(data);
                } else if (currentView === 'sum_table') {
                    // Для режима Sum table показываем HTML сводную таблицу и обновляем только Current
                    console.log('DEBUG: Showing HTML sum table');
                    showSumTable(data);
                    // Обновляем только Current таблицу, не трогая canvas
                    updateStats(data);
                    updateMessageInfo(data);
                } else {
                    // Для остальных режимов рисуем графики и обновляем display
                    updateDisplay(data);
                }
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        // Переменная для управления таймером обновления
        let updateTimer = null;
        let isRunning = false;

        function startUpdating() {
            if (!updateTimer) {
                updateTimer = setInterval(fetchData, 700);
                isRunning = true;
                console.log('Graph updating started');
            }
        }

        function stopUpdating() {
            if (updateTimer) {
                clearInterval(updateTimer);
                updateTimer = null;
                isRunning = false;
                console.log('Graph updating stopped');
            }
        }

        // Функции кнопок
        async function measure() {
            await fetch('/api/measure', { method: 'POST' });
        }

        async function runTest() {
            const response = await fetch('/api/run', { method: 'POST' });
            const data = await response.json();
            // Автообновление отключено - обновление только при загрузке файла
            console.log('Test run, auto-update disabled');
        }

        async function contTest() {
            const response = await fetch('/api/cont', { method: 'POST' });
            const data = await response.json();
            // Автообновление отключено - обновление только при загрузке файла
            console.log('Test continue, auto-update disabled');
        }

        async function breakTest() {
            const response = await fetch('/api/break', { method: 'POST' });
            const data = await response.json();
            // Автообновление всегда отключено
            stopUpdating();
            console.log('Test break, auto-update disabled');
        }

        function loadFile() {
            const fileInput = document.getElementById('fileInput');
            fileInput.click();
        }

        async function uploadFile(input) {
            if (input.files && input.files[0]) {
                const file = input.files[0];

                if (!file.name.endsWith('.cf32')) {
                    alert('Выберите файл с расширением .cf32');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (result.status === 'success' && result.processed) {
                        console.log('File uploaded and processed successfully:', result);
                        console.log('=== FORCING DATA UPDATE ===');
                        // Обновляем данные после загрузки
                        fetchData();
                    } else {
                        console.error('Upload failed:', result);
                        if (result.status === 'success' && !result.processed) {
                            alert('Файл загружен, но не удалось обработать: ' + (result.message || 'Unknown error'));
                        } else {
                            alert('Ошибка загрузки файла: ' + (result.error || result.message || 'Unknown error'));
                        }
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                    alert('Ошибка загрузки файла: ' + error.message);
                }

                // Очищаем input для возможности повторной загрузки того же файла
                input.value = '';
            }
        }

        async function saveFile() {
            await fetch('/api/save', { method: 'POST' });
        }

        // Первоначальная загрузка данных (автообновление отключено по умолчанию)
        fetchData();

        // Отключаем автообновление по умолчанию - график обновляется только при загрузке файла
        console.log('Auto-update disabled by default');
    </script>
</body>
</html>
"""

# API routes
@app.route('/')
def index():
    response = Response(HTML_PAGE, mimetype='text/html')
    # Отключаем кэширование для разработки
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
    response.headers['ETag'] = ''
    return response

@app.route('/api/status')
def api_status():
    # Добавляем небольшие вариации для реалистичности
    fs1_var = STATE.fs1_hz + random.uniform(-0.5, 0.5)
    fs2_var = STATE.fs2_hz + random.uniform(-0.5, 0.5)
    fs3_var = STATE.fs3_hz + random.uniform(-0.5, 0.5)

    print(f"API STATUS DEBUG: phase_data length = {len(STATE.phase_data)}")
    print(f"API STATUS DEBUG: xs_fm_ms length = {len(STATE.xs_fm_ms)}")
    print(f"API STATUS DEBUG: pos_phase = {STATE.pos_phase}")
    print(f"API STATUS DEBUG: neg_phase = {STATE.neg_phase}")
    if len(STATE.phase_data) > 0:
        print(f"API STATUS DEBUG: phase_data sample = {STATE.phase_data[:5]}")
    if len(STATE.xs_fm_ms) > 0:
        print(f"API STATUS DEBUG: xs_fm_ms sample = {STATE.xs_fm_ms[:5]}")

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
        'phase_pos_rad': STATE.pos_phase,  # Используем новые значения из demod
        'phase_neg_rad': STATE.neg_phase,
        't_rise_mcs': STATE.ph_rise,       # В микросекундах из demod
        't_fall_mcs': STATE.ph_fall,
        'p_wt': STATE.p_wt,
        'prise_ms': STATE.prise_ms,
        'bitrate_bps': STATE.bitrate_bps,
        'symmetry_pct': STATE.asymmetry,  # Используем asymmetry из demod
        'preamble_ms': STATE.preamble_ms,
        'total_ms': STATE.total_ms,
        'rep_period_s': STATE.rep_period_s,
        'rms_dbm': STATE.rms_dbm,       # Новые метрики
        'freq_hz': STATE.freq_hz,
        't_mod': STATE.t_mod,
        'phase_data': STATE.phase_data,
        'xs_fm_ms': STATE.xs_fm_ms
    })

@app.route('/api/measure', methods=['POST'])
def api_measure():
    return jsonify({'status': 'measure triggered'})

@app.route('/api/run', methods=['POST'])
def api_run():
    STATE.running = True
    return jsonify({'status': 'running', 'running': STATE.running})

@app.route('/api/cont', methods=['POST'])
def api_cont():
    STATE.running = True
    return jsonify({'status': 'continue', 'running': STATE.running})

@app.route('/api/break', methods=['POST'])
def api_break():
    STATE.running = False
    return jsonify({'status': 'stopped', 'running': STATE.running})

@app.route('/api/load', methods=['POST'])
def api_load():
    return jsonify({'status': 'load requested'})

@app.route('/api/save', methods=['POST'])
def api_save():
    return jsonify({'status': 'save requested'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Проверяем расширение файла
        if not file.filename.lower().endswith('.cf32'):
            return jsonify({'error': 'Only .cf32 files are allowed'}), 400

        # Безопасное имя файла
        filename = secure_filename(file.filename)

        # Создаем папку uploads если её нет
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'captures', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Путь для сохранения файла
        file_path = os.path.join(upload_dir, filename)

        # Сохраняем файл
        file.save(file_path)

        # Обновляем состояние
        STATE.current_file = file_path
        STATE.message = f"Loaded: {filename}"

        print(f"File uploaded: {filename} -> {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")

        # Обрабатываем загруженный файл
        processing_result = process_cf32_file(file_path)

        if processing_result.get("success"):
            # Обновляем STATE результатами обработки
            STATE.hex_message = processing_result.get("msg_hex", "")

            # Обновляем метрики в Current таблице
            if "pos_phase" in processing_result:
                STATE.pos_phase = processing_result["pos_phase"]
            if "neg_phase" in processing_result:
                STATE.neg_phase = processing_result["neg_phase"]
            if "ph_rise" in processing_result:
                STATE.ph_rise = processing_result["ph_rise"]
            if "ph_fall" in processing_result:
                STATE.ph_fall = processing_result["ph_fall"]
            if "asymmetry" in processing_result:
                STATE.asymmetry = processing_result["asymmetry"]
            if "t_mod" in processing_result:
                STATE.t_mod = processing_result["t_mod"]
            if "rms_dbm" in processing_result:
                STATE.rms_dbm = processing_result["rms_dbm"]
            if "freq_hz" in processing_result:
                STATE.freq_hz = processing_result["freq_hz"]

            # Добавляем новые метрики для Current таблицы
            if "bitrate_bps" in processing_result:
                STATE.bitrate_bps = processing_result["bitrate_bps"]
            if "symmetry_pct" in processing_result:
                STATE.symmetry_pct = processing_result["symmetry_pct"]
            if "total_ms" in processing_result:
                STATE.total_ms = processing_result["total_ms"]
            if "prise_ms" in processing_result:
                STATE.prise_ms = processing_result["prise_ms"]
            if "preamble_ms" in processing_result:
                STATE.preamble_ms = processing_result["preamble_ms"]

            # Сохраняем данные фазы для графика
            STATE.phase_data = processing_result.get("phase_data", [])
            STATE.xs_fm_ms = processing_result.get("xs_fm_ms", [])
            STATE.message = f"Processed: {filename} - Message: {STATE.hex_message[:16]}..."

            print(f"File processed successfully: {len(STATE.phase_data)} phase samples, {len(STATE.xs_fm_ms)} time samples")

        else:
            error_msg = processing_result.get("error", "Unknown error")
            STATE.message = f"Error processing {filename}: {error_msg}"
            print(f"Processing error: {error_msg}")

        # Возвращаем правильный статус в зависимости от результата обработки
        if processing_result.get("success"):
            return jsonify({
                'status': 'success',
                'filename': filename,
                'size': os.path.getsize(file_path),
                'path': file_path,
                'processed': True,
                'message': STATE.message
            })
        else:
            return jsonify({
                'status': 'error',
                'error': processing_result.get("error", "Processing failed"),
                'filename': filename,
                'size': os.path.getsize(file_path),
                'path': file_path,
                'processed': False,
                'message': STATE.message
            }), 400

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(">>> Starting COSPAS/SARSAT Beacon Tester v2.1")
    print(">>> Interface available at: http://127.0.0.1:8738/")
    print(">>> To stop: Ctrl+C")
    app.run(host='127.0.0.1', port=8738, debug=True)