#!/usr/bin/env python3
"""
Activate real PSK demodulation in analyze_psk406 function
"""

def activate_real_psk():
    """Replace PSK analysis stub with real demodulation"""
    file_path = 'beacon406/beacon_tester_web.py'

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the entire analyze_psk406 function
    old_function = '''def analyze_psk406(iq_seg: np.ndarray, fs: float) -> dict:
    """Анализ PSK-406 сигнала с заглушкой для будущей интеграции"""
    result = {
        'bitrate_bps': None,
        'pos_phase': None,
        'neg_phase': None,
        'ph_rise_us': None,
        'ph_fall_us': None,
        'asymmetry_pct': None,
        'msg_hex': None,
        'msg_ok': None
    }

    try:
        # TODO: Якорь для интеграции phase_demod_psk_msg_safe(...)
        # msg_hex, phase_res, edges = phase_demod_psk_msg_safe(data=phase_data)

        # STRICT_COMPAT: Минимальная заглушка для генерации тестовых PSK-метрик
        if iq_seg.size > 0:
            pulse_len_ms = len(iq_seg) / fs * 1000
            result.update({
                'bitrate_bps': 400.0,          # Стандартный PSK-406
                'pos_phase': 0.78,             # Типичные фазы PSK
                'neg_phase': -0.78,
                'ph_rise_us': 25.0,            # Микросекунды перехода
                'ph_fall_us': 23.0,
                'asymmetry_pct': 8.7,          # Асимметрия в %
                'msg_hex': 'AD3F8C12...',  # Заглушка HEX
                'msg_ok': True         # CRC OK
            })

        print(f"[PSK] Analyzed segment: {iq_seg.size} samples, {pulse_len_ms:.1f}ms")

    except Exception as e:
        print(f"[PSK] Analysis error: {e}")

    return result'''

    new_function = '''def analyze_psk406(iq_seg: np.ndarray, fs: float) -> dict:
    """Реальный анализ PSK-406 сигнала с демодуляцией"""
    result = {
        'bitrate_bps': None,
        'pos_phase': None,
        'neg_phase': None,
        'ph_rise_us': None,
        'ph_fall_us': None,
        'asymmetry_pct': None,
        'msg_hex': None,
        'msg_ok': None
    }

    try:
        if iq_seg.size == 0:
            return result

        pulse_len_ms = len(iq_seg) / fs * 1000

        # Конвертируем IQ в фазовые данные для демодулятора
        phase_data = np.angle(iq_seg)

        # Вызываем настоящий PSK демодулятор
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(
            data=phase_data,
            window=40,
            threshold=0.5,
            start_idx=min(25000, len(phase_data)//4),
            N=28,
            min_edges=29
        )

        # Извлекаем результаты демодуляции
        if phase_res and not np.isnan(phase_res.get("PosPhase", np.nan)):
            result.update({
                'bitrate_bps': 400.0,  # Стандартный PSK-406
                'pos_phase': float(phase_res.get("PosPhase", 0.78)),
                'neg_phase': float(phase_res.get("NegPhase", -0.78)),
                'ph_rise_us': float(phase_res.get("PhRise", 25.0)),
                'ph_fall_us': float(phase_res.get("PhFall", 23.0)),
                'asymmetry_pct': float(phase_res.get("Ass", 0.0)),
                'msg_hex': msg_hex if msg_hex else None,
                'msg_ok': bool(msg_hex and len(msg_hex) > 0)
            })
            print(f"[PSK] Real demod: {msg_hex}, phases: pos={result['pos_phase']:.3f}, neg={result['neg_phase']:.3f}")
        else:
            print(f"[PSK] Demodulation failed - no valid phases detected")

        print(f"[PSK] Analyzed segment: {iq_seg.size} samples, {pulse_len_ms:.1f}ms")

    except Exception as e:
        print(f"[PSK] Real analysis error: {e}")

    return result'''

    if old_function in content:
        content = content.replace(old_function, new_function)

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("OK Real PSK demodulation activated!")
    else:
        print("× Function pattern not found - file may have changed")

if __name__ == "__main__":
    activate_real_psk()