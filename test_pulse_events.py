#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки pulse событий от DSP сервиса
"""
import zmq
import json
import sys

def main():
    pub_addr = "tcp://127.0.0.1:8781"

    print(f"Подключаюсь к PUB: {pub_addr}")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(pub_addr)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    print("Ожидаю события...")
    pulse_count = 0

    try:
        while True:
            raw = sub.recv_string()
            obj = json.loads(raw)
            typ = obj.get("type")

            if typ == "pulse":
                pulse_count += 1
                print(f"\n=== PULSE EVENT #{pulse_count} ===")
                print(f"Keys: {sorted(obj.keys())}")

                # Размеры массивов
                px = obj.get("phase_xs_ms", [])
                py = obj.get("phase_ys_rad", [])
                fx = obj.get("fr_xs_ms", [])
                fy = obj.get("fr_ys_hz", [])
                rms = obj.get("rms_ms_dbm", [])

                print(f"Array sizes:")
                print(f"  phase_xs_ms: {len(px) if px else 0}")
                print(f"  phase_ys_rad: {len(py) if py else 0}")
                print(f"  fr_xs_ms: {len(fx) if fx else 0}")
                print(f"  fr_ys_hz: {len(fy) if fy else 0}")
                print(f"  rms_ms_dbm: {len(rms) if rms else 0}")

                # Метаданные
                print(f"Metadata:")
                print(f"  msg_hex: {obj.get('msg_hex')}")
                print(f"  phase_metrics: {obj.get('phase_metrics') is not None}")
                print(f"  iq_seg: {len(obj.get('iq_seg', [])) if obj.get('iq_seg') else 0} samples")
                print(f"  core_gate: {obj.get('core_gate')}")

                if pulse_count >= 3:
                    print("\nПолучено 3 события, завершаю...")
                    break

            elif typ == "status":
                acq = obj.get("status", {}).get("acq_state", "?")
                print(f"[status] acq_state={acq}", end="\r")

    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    finally:
        sub.close()
        ctx.term()

if __name__ == "__main__":
    main()
