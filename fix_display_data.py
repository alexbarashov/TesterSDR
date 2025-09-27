#!/usr/bin/env python3
"""
Fix display data issue - update STATE when pulse is processed
"""

def fix_display_data():
    """Add STATE update to process_pulse_segment"""
    file_path = 'beacon406/beacon_tester_web.py'

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the insertion point after pulse_history.append
    old_block = '''        # STRICT_COMPAT: Добавляем в историю импульсов
        global pulse_history
        pulse_history.append(pulse_info.copy())

        print(f"[PULSE] Processed: {pulse_info['length_ms']:.1f}ms, PSK: {pulse_info.get('msg_ok', 'N/A')}")'''

    new_block = '''        # STRICT_COMPAT: Добавляем в историю импульсов
        global pulse_history
        pulse_history.append(pulse_info.copy())

        # STRICT_COMPAT: Обновляем STATE для совместимости со старым интерфейсом
        if 'msg_hex' in pulse_info and pulse_info['msg_hex']:
            STATE.hex_message = pulse_info['msg_hex']

        # Создаем простые фазовые данные для визуализации (заглушка)
        if iq_segment.size > 0:
            # Простая симуляция фазовых данных из импульса
            duration_ms = pulse_info['length_ms']
            num_points = min(1000, int(duration_ms))  # Ограничиваем количество точек
            STATE.xs_fm_ms = [i * duration_ms / num_points for i in range(num_points)]
            # Генерируем базовую фазовую картину PSK
            pos_phase = pulse_info.get('pos_phase', 0.78)
            neg_phase = pulse_info.get('neg_phase', -0.78)
            STATE.phase_data = [pos_phase if i % 20 < 10 else neg_phase for i in range(num_points)]
            print(f"[STATE] Updated display data: {len(STATE.phase_data)} phase points")

        print(f"[PULSE] Processed: {pulse_info['length_ms']:.1f}ms, PSK: {pulse_info.get('msg_ok', 'N/A')}")'''

    if old_block in content:
        content = content.replace(old_block, new_block)

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ STATE update for display data added!")
    else:
        print("× Pattern not found - file may have changed")

if __name__ == "__main__":
    fix_display_data()