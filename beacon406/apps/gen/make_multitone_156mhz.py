# save as make_multitone_156mhz.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.logger import get_logger
log = get_logger(__name__)
import numpy as np

# Параметры
fs = 1_000_000            # sample rate для HackRF
dur_s = 10                # длительность, сек
fc = 156_050_000          # центровая частота передатчика (Гц)
# 9 тонов: от -25 кГц до +25 кГц шагом 6.25 кГц
offsets_hz = np.arange(-25_000, 25_000 + 1, 6_250, dtype=float)

t = np.arange(int(fs*dur_s)) / fs
# Суммируем комплексные экспоненты (мульти-тон)
x = np.zeros_like(t, dtype=np.complex64)
for f in offsets_hz:
    x += np.exp(1j * 2*np.pi * f * t)

# Нормализация уровня (запас по пику, чтобы не клиповать 8-бит)
x /= np.max(np.abs(x)) * 1.25  # ~ -2 дБFS на сумме; при необходимости ещё уменьшите

# Небольшой плавный fade-in/out, чтобы убрать щелчки на старте/конце
ramp = int(0.005 * fs)  # 5 мс
win = np.ones_like(t, dtype=float)
win[:ramp] = np.linspace(0, 1, ramp)
win[-ramp:] = np.linspace(1, 0, ramp)
x *= win

# В int8: HackRF ожидает I,Q попеременно, signed int8
iq_i = np.clip(np.real(x) * 127, -128, 127).astype(np.int8)
iq_q = np.clip(np.imag(x) * 127, -128, 127).astype(np.int8)
iq_interleaved = np.empty(iq_i.size*2, dtype=np.int8)
iq_interleaved[0::2] = iq_i
iq_interleaved[1::2] = iq_q

out_path = r"C:\work\TesterSDR\captures\multitone_156_025_075_9tones_10s_1Msps.iq"
iq_interleaved.tofile(out_path)
log.info("Saved:", out_path)
log.info("Center for TX:", fc, "Hz")
