# psk406_msg_gen.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.logger import get_logger
log = get_logger(__name__)
import numpy as np

def _dbfs_to_linear(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)

def _gen_noise(n: int, amp_lin: float, rng: np.random.Generator) -> np.ndarray:
    # Complex white Gaussian noise with target RMS ~ amp_lin/sqrt(2) per I/Q
    noise_i = rng.normal(0.0, amp_lin / np.sqrt(2.0), size=n).astype(np.float32)
    noise_q = rng.normal(0.0, amp_lin / np.sqrt(2.0), size=n).astype(np.float32)
    return (noise_i + 1j * noise_q).astype(np.complex64)

def _phase_ramp(start: float, stop: float, n: int) -> np.ndarray:
    # Linear ramp for smooth transition (in radians)
    return np.linspace(start, stop, n, endpoint=False, dtype=np.float32)

def _clip_norm(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    m = np.max(np.abs(x)) if x.size else 1.0
    if m == 0.0:
        return x
    if m > peak:
        x = (x / m) * peak
    return x

def _hex_to_bits(hex_message: str) -> np.ndarray:
    # Remove whitespace
    s = "".join(hex_message.split()).lower()
    if len(s) % 2 != 0:
        raise ValueError("HEX length must be even")
    try:
        data = bytes.fromhex(s)
    except ValueError as e:
        raise ValueError(f"Invalid HEX: {e}")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)

def generate_psk406_cf32(
    sample_rate_sps: int = 1_000_000,
    bit_rate_bps: float = 400.0,
    hex_message: str = "FFFED080020000007FDFFB0020B783E0F66C",
    phase_low_high: tuple[float, float] = (-1.1, 1.1),
    front_samples: int = 75,
    pre_silence_ms: float = 25.0,
    post_silence_ms: float = 25.0,
    carrier_sec: float = 0.16,
    noise_dbfs: float = -60.0,
    normalize: bool = True,
    return_array: bool = True,
    save_path: str  = None,
    rng_seed: int  = 12345,
) -> np.ndarray  :
    """Generate PSK-406 IQ buffer (complex64) with carrier, message, and silences.

    - sample_rate_sps: Fs for generated buffer
    - bit_rate_bps:   fixed 400.0 for PSK-406
    - hex_message:    payload bytes encoded as hex
    - phase_low_high: tuple (phi0, phi1) in radians (e.g., -1.1, +1.1)
    - front_samples:  smooth edge length (samples) for each bit
    - pre/post silence and carrier with optional noise (noise_dbfs)
    - normalize:      peak normalize to 1.0 before returning
    - save_path:      if set, write interleaved float32 (I,Q,...) to file
    - rng_seed:       deterministic noise generation
    """
    Fs = float(sample_rate_sps)
    bit_T = 1.0 / float(bit_rate_bps)
    bit_samples = int(round(Fs * bit_T))
    if bit_samples < 10:
        raise ValueError("bit_samples too small; increase sample_rate_sps")

    if front_samples >= bit_samples // 2:
        raise ValueError("front_samples must be < bit_samples/2")

    # Prepare sections
    rng = np.random.default_rng(rng_seed)
    pre_N  = int(round(pre_silence_ms * 1e-3 * Fs))
    post_N = int(round(post_silence_ms * 1e-3 * Fs))
    carr_N = int(round(carrier_sec * Fs))

    noise_amp = _dbfs_to_linear(noise_dbfs)

    # Pre-silence (noise)
    pre = _gen_noise(pre_N, noise_amp, rng) if pre_N > 0 else np.zeros(0, np.complex64)
    # Carrier (phase=0)
    carrier = (np.ones(carr_N, dtype=np.complex64) if carr_N > 0
               else np.zeros(0, np.complex64))


    # --- Message PSK: полубитовая структура с плавными переходами ---
    bits = _hex_to_bits(hex_message)
    phi0, phi1 = float(phase_low_high[0]), float(phase_low_high[1])

    bit_samples = int(round(sample_rate_sps / bit_rate_bps))
    half = bit_samples // 2
    if front_samples >= half // 2:
        raise ValueError("front_samples must be < half_bit_samples/2")

    msg = np.zeros(bits.size * bit_samples, np.complex64)

    # Вспомогалка: сформировать сегмент "от prev -> target" длиной segN с плавным фронтом
    def _seg(prev_phase: float, target_phase: float, segN: int, frontN: int) -> np.ndarray:
        if segN <= 0:
            return np.zeros(0, np.complex64)
        rampN = min(frontN, segN)
        if rampN > 0:
            ramp = _phase_ramp(prev_phase, target_phase, rampN)
            steadyN = segN - rampN
            if steadyN > 0:
                phase_vec = np.concatenate([ramp, np.full(steadyN, target_phase, np.float32)])
            else:
                phase_vec = ramp
        else:
            phase_vec = np.full(segN, target_phase, np.float32)
        return np.exp(1j * phase_vec.astype(np.float32)).astype(np.complex64)

    idx = 0
    prev_phase = 0.0  # начинаем с нулевой фазы (переход от несущей 0 рад)

    for b in bits:
        # Бит разбиваем на две половины с противоположными фазами
        first = phi0 if b == 0 else phi1
        second = phi1 if b == 0 else phi0

        seg1 = _seg(prev_phase, first, half, front_samples)
        prev_phase = float(first)
        seg2 = _seg(prev_phase, second, bit_samples - half, front_samples)
        prev_phase = float(second)

        msg[idx:idx+half] = seg1
        msg[idx+half:idx+bit_samples] = seg2
        idx += bit_samples
    # --- конец патча ---

    # Post-silence (noise)
    post = _gen_noise(post_N, noise_amp, rng) if post_N > 0 else np.zeros(0, np.complex64)

    sig = np.concatenate([pre, carrier, msg, post]).astype(np.complex64)
    if normalize and sig.size:
        sig = _clip_norm(sig, peak=0.999)

    if save_path is not None:
        # interleaved float32 I,Q
        inter = np.empty(sig.size * 2, dtype=np.float32)
        inter[0::2] = sig.real.astype(np.float32, copy=False)
        inter[1::2] = sig.imag.astype(np.float32, copy=False)
        with open(save_path, "wb") as f:
            inter.tofile(f)

    return sig if return_array else None
