#!/usr/bin/env python3
"""
Fix PSK analysis to properly segment pulse from noisy IQ buffer
"""

def fix_pulse_segmentation():
    """Add proper pulse segmentation to analyze_psk406"""
    file_path = 'beacon406/beacon_tester_web.py'

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the analyze_psk406 function and fix the segmentation
    old_block = '''        pulse_len_ms = len(iq_seg) / fs * 1000
        phase_data = np.angle(iq_seg)

        # Настоящий PSK демодулятор
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(
            data=phase_data,
            window=40,
            threshold=0.5,
            start_idx=min(25000, len(phase_data)//4),
            N=28,
            min_edges=29
        )'''

    new_block = '''        pulse_len_ms = len(iq_seg) / fs * 1000

        # КРИТИЧНО: Вырезаем только импульс из IQ сегмента, убирая шум по краям
        clean_iq = _find_pulse_segment(
            iq_data=iq_seg,
            sample_rate=fs,
            thresh_dbm=-60.0,  # Порог детекции
            win_ms=1.0,        # RMS окно
            start_delay_ms=10.0,  # Задержка начала
            calib_db=-30.0     # Калибровка
        )

        if clean_iq is None or len(clean_iq) < 1000:
            print(f"[PSK] No clean pulse found in {len(iq_seg)} samples")
            return result

        print(f"[PSK] Clean pulse: {len(clean_iq)} samples from {len(iq_seg)} total")
        phase_data = np.angle(clean_iq)

        # Настоящий PSK демодулятор на очищенных данных
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(
            data=phase_data,
            window=40,
            threshold=0.5,
            start_idx=min(25000, len(phase_data)//4),
            N=28,
            min_edges=29
        )'''

    if old_block in content:
        content = content.replace(old_block, new_block)

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("OK Pulse segmentation fixed!")
    else:
        print("× PSK block not found - file may have changed")

if __name__ == "__main__":
    fix_pulse_segmentation()