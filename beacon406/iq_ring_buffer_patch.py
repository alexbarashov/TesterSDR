"""
STRICT_COMPAT патч для добавления кольцевого буфера IQ в beacon_tester_web.py
Шаг 1: Добавляет кольцевой буфер для хранения 3+ секунд БП-сигнала
"""

# STRICT_COMPAT: Кольцевой буфер IQ для хранения 3+ секунд сигнала
class IQRingBuffer:
    def __init__(self, duration_sec: float, sample_rate: float):
        import numpy as np
        import threading
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.capacity = int(duration_sec * sample_rate)
        self.buffer = np.zeros(self.capacity, dtype=np.complex64)
        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, samples):
        """Записать отсчеты в кольцевой буфер"""
        import numpy as np
        with self.lock:
            n = len(samples)
            if n >= self.capacity:
                # Если данных больше емкости, берем последние
                self.buffer[:] = samples[-self.capacity:]
                self.write_pos = 0
                self.total_written += n
            else:
                # Запись в две части если переход через границу
                end_pos = self.write_pos + n
                if end_pos <= self.capacity:
                    self.buffer[self.write_pos:end_pos] = samples
                    self.write_pos = end_pos % self.capacity
                else:
                    first_part = self.capacity - self.write_pos
                    self.buffer[self.write_pos:] = samples[:first_part]
                    self.buffer[:n-first_part] = samples[first_part:]
                    self.write_pos = n - first_part
                self.total_written += n

    def get_segment(self, abs_start: int, abs_end: int):
        """Извлечь сегмент по абсолютным индексам"""
        import numpy as np
        with self.lock:
            # Проверка доступности данных
            oldest_available = max(0, self.total_written - self.capacity)
            if abs_start < oldest_available:
                return np.array([], dtype=np.complex64)

            # Вычисляем позиции в буфере
            start_offset = abs_start - (self.total_written - min(self.total_written, self.capacity))
            end_offset = abs_end - (self.total_written - min(self.total_written, self.capacity))

            if start_offset < 0 or end_offset > self.capacity:
                return np.array([], dtype=np.complex64)

            # Извлекаем данные с учетом кольцевой структуры
            if end_offset > start_offset:
                return self.buffer[start_offset:end_offset].copy()
            else:
                return np.concatenate([
                    self.buffer[start_offset:],
                    self.buffer[:end_offset]
                ]).copy()


# STRICT_COMPAT: Добавляем глобальные переменные для буфера и actual sample rate
# Вставить после строки 45 (после last_pulse_data = None)
"""
# STRICT_COMPAT: Кольцевой буфер IQ и actual sample rate
iq_ring_buffer = None
actual_sample_rate_sps = SAMPLE_RATE_SPS  # По умолчанию, обновится при init SDR
bp_sample_counter = 0  # Счетчик отсчетов БП-сигнала после NCO

def get_actual_fs() -> float:
    \"\"\"Получить фактический sample rate\"\"\"
    global actual_sample_rate_sps
    return actual_sample_rate_sps
"""

# STRICT_COMPAT: Обновление init_sdr_backend для сохранения actual_sample_rate
# Найти строку "# Получаем информацию об устройстве" (около 106)
# После status = sdr_backend.get_status() добавить:
"""
        # STRICT_COMPAT: Сохраняем actual sample rate и создаем буфер
        global actual_sample_rate_sps, iq_ring_buffer, win_samps
        actual_sample_rate_sps = status.get('actual_sample_rate_sps', SAMPLE_RATE_SPS)
        iq_ring_buffer = IQRingBuffer(duration_sec=3.0, sample_rate=actual_sample_rate_sps)
        # Обновляем win_samps с фактическим sample rate
        win_samps = int(RMS_WIN_MS * 1e-3 * actual_sample_rate_sps)
        print(f"[SDR] Actual sample rate: {actual_sample_rate_sps:.3f} Hz")
        print(f"[SDR] IQ buffer created for {iq_ring_buffer.duration_sec} seconds")
"""

# STRICT_COMPAT: Запись БП-сигнала в кольцевой буфер
# В функции process_samples_realtime, после применения NCO (строка ~218)
# После строки "nco_phase = float((nco_phase + nco_k * samples.size) % (2.0 * np.pi))"
# Добавить:
"""
    # STRICT_COMPAT: Запись БП-сигнала в кольцевой буфер
    global iq_ring_buffer, bp_sample_counter
    if iq_ring_buffer is not None:
        iq_ring_buffer.write(x)
        bp_sample_counter += len(x)
"""

# STRICT_COMPAT: Обновление detect_pulses для использования буфера
# В функции detect_pulses добавить возможность извлечения сегмента из буфера
# После строки "pulse_len_ms = pulse_len_samples / SAMPLE_RATE_SPS * 1000" (~290)
# Заменить SAMPLE_RATE_SPS на get_actual_fs():
"""
            pulse_len_samples = pulse_end_abs - pulse_start_abs + 1
            pulse_len_ms = pulse_len_samples / get_actual_fs() * 1000
"""

# STRICT_COMPAT: Извлечение сегмента из буфера при обнаружении импульса
# В process_pulse_segment добавить извлечение из буфера
# В начале функции process_pulse_segment (строка ~305) добавить:
"""
def process_pulse_segment(pulse_start_abs, pulse_end_abs, iq_fallback=None):
    \"\"\"Обработка сегмента импульса\"\"\"
    global iq_ring_buffer, last_pulse_data

    # STRICT_COMPAT: Извлекаем сегмент из кольцевого буфера
    if iq_ring_buffer is not None:
        # Конвертируем индексы RMS в индексы БП-сигнала
        bp_start = pulse_start_abs - win_samps + 1
        bp_end = pulse_end_abs + 1
        iq_segment = iq_ring_buffer.get_segment(bp_start, bp_end)

        if iq_segment.size > 0:
            print(f"[PULSE] Extracted segment: {iq_segment.size} samples from buffer")
        else:
            print(f"[PULSE] Segment not available in buffer, using fallback")
            iq_segment = iq_fallback if iq_fallback is not None else np.array([])
    else:
        iq_segment = iq_fallback if iq_fallback is not None else np.array([])
"""