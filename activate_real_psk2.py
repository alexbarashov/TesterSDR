#!/usr/bin/env python3
"""
Activate real PSK demodulation by replacing specific TODO lines
"""

def activate_real_psk():
    """Replace PSK analysis stub with real demodulation"""
    file_path = 'beacon406/beacon_tester_web.py'

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace just the commented line with real call
    old_line = "        # msg_hex, phase_res, edges = phase_demod_psk_msg_safe(data=phase_data)"
    new_lines = '''        if iq_seg.size == 0:
            return result

        pulse_len_ms = len(iq_seg) / fs * 1000
        phase_data = np.angle(iq_seg)
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(
            data=phase_data,
            window=40,
            threshold=0.5,
            start_idx=min(25000, len(phase_data)//4),
            N=28,
            min_edges=29
        )'''

    # Replace the TODO block with real implementation
    todo_block = '''        # TODO: Якорь для интеграции phase_demod_psk_msg_safe(...)
        # msg_hex, phase_res, edges = phase_demod_psk_msg_safe(data=phase_data)

        # TODO: Якорь для интеграции calc_phase3_2(...)
        # Pos/NegPhase, PhRise/Fall, Asymmetry вычисления

        # Заглушка: возвращаем тестовые значения если сегмент достаточно длинный
        if iq_seg.size > 0:
            pulse_len_ms = len(iq_seg) / fs * 1000
            if pulse_len_ms >= 400:  # Минимальная длина PSK-406
                result.update({
                    'bitrate_bps': 400.0,  # Заглушка: ~400 bps для PSK-406
                    'pos_phase': 0.78,     # ~π/4
                    'neg_phase': -0.78,    # ~-π/4
                    'ph_rise_us': 25.0,    # 25 мкс
                    'ph_fall_us': 23.0,    # 23 мкс
                    'asymmetry_pct': 8.7,  # 8.7%
                    'msg_hex': 'AD3F8C12...',  # Заглушка HEX
                    'msg_ok': True         # CRC OK
                })'''

    real_block = '''        if iq_seg.size == 0:
            return result

        pulse_len_ms = len(iq_seg) / fs * 1000
        phase_data = np.angle(iq_seg)

        # Настоящий PSK демодулятор
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
            print(f"[PSK] Demodulation failed - no valid phases detected")'''

    if todo_block in content:
        content = content.replace(todo_block, real_block)

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("OK Real PSK demodulation activated!")
    else:
        print("× TODO block not found - file may have changed")

if __name__ == "__main__":
    activate_real_psk()