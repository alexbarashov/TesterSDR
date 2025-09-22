import numpy as np
import time
from pathlib import Path

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
FILE_IQ = "psk406msg_f75.cf32"

# формируем шаблон пути (сразу строка!)
FILE_PATH = str(ROOT / "captures" / FILE_IQ)


def save_psk406_iq(filename=FILE_PATH, sample_rate=1_000_000, bit_rate=400, pre_ms=25.0, post_ms=25.0,
                   noise_dbfs=-60.0, carrier_sec=0.16):
    bit_samples = int(sample_rate / bit_rate)
    hex_message = "FFFED080020000007FDFFB0020B783E0F66C"
    #hex_message = "FFFED08"
    
    def hex_to_bits(h):
        return [int(b) for c in h for b in f"{int(c, 16):04b}"]
    bits = hex_to_bits(hex_message)

    phase_low, phase_high = -1.1, 1.1
    front = 75
    iq = []
    current_phase = 0.0

    # --- helper: quiet complex noise around 0 at target level (dBFS) ---
    def quiet_with_noise(n_samples, dbfs):
        amp = 10.0 ** (dbfs / 20.0)                  # linear target RMS amplitude
        sigma = amp / np.sqrt(2.0)                   # per I/Q for complex Gaussian
        nI = np.random.normal(0.0, sigma, n_samples)
        nQ = np.random.normal(0.0, sigma, n_samples)
        return (nI + 1j * nQ).astype(np.complex64)

    # --- 25 ms before: near-zero with ~-50 dBFS complex noise ---
    pre_len = int((pre_ms / 1000.0) * sample_rate)
    if pre_len > 0:
        iq.extend(list(quiet_with_noise(pre_len, noise_dbfs)))

    # ---- Carrier I=1, Q=0 (phase=0) ----
    carrier_len = int(carrier_sec * sample_rate)
    carrier_samples = [np.exp(1j * 0.0)] * carrier_len
    iq += carrier_samples
    
    """
    with open("carrier_check.txt", "w") as f:
        for i, sample in enumerate(carrier_samples[:10]):
            f.write(f"{i}: I = {sample.real:.6f}, Q = {sample.imag:.6f}\n")
    print("✅ Первые 10 значений несущей записаны в carrier_check.txt")
    """
    
    # ---- PSK406 message ----
    for bit in bits:
        ph1, ph2 = (phase_high, phase_low) if bit else (phase_low, phase_high)
        for target in (ph1, ph2):
            # плавный фронт
            for j in range(front):
                interp = current_phase + (target - current_phase) * (j + 1) / front
                iq.append(np.exp(1j * interp))
            # полупериод на целевой фазе
            iq += [np.exp(1j * target)] * (bit_samples // 2 - front)
            current_phase = target

    # --- 25 ms after: near-zero with ~-50 dBFS complex noise ---
    post_len = int((post_ms / 1000.0) * sample_rate)
    if post_len > 0:
        iq.extend(list(quiet_with_noise(post_len, noise_dbfs)))

    # ---- Normalize (keeps noise at same relative level) & save CF32 ----
    iq = np.array(iq, dtype=np.complex64)
    peak = np.max(np.abs(iq))
    if peak > 0:
        iq /= peak
    iq.tofile(filename)
    print(f"✅ Сигнал PSK406 сохранён в формате .cf32: {filename}")

save_psk406_iq()
