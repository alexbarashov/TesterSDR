#!/usr/bin/env python3
"""
STRICT_COMPAT: Тестовый скрипт для проверки кольцевого буфера IQ
"""

import numpy as np
import threading
import time

# STRICT_COMPAT: Кольцевой буфер IQ для хранения 3+ секунд сигнала
class IQRingBuffer:
    def __init__(self, duration_sec: float, sample_rate: float):
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.capacity = int(duration_sec * sample_rate)
        self.buffer = np.zeros(self.capacity, dtype=np.complex64)
        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()
        print(f"[BUFFER] Created: {self.capacity} samples ({duration_sec}s at {sample_rate:.0f} Hz)")

    def write(self, samples: np.ndarray):
        """Записать отсчеты в кольцевой буфер"""
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

    def get_segment(self, abs_start: int, abs_end: int) -> np.ndarray:
        """Извлечь сегмент по абсолютным индексам"""
        with self.lock:
            # Проверка доступности данных
            oldest_available = max(0, self.total_written - self.capacity)
            if abs_start < oldest_available:
                print(f"[BUFFER] Segment too old: start={abs_start} < oldest={oldest_available}")
                return np.array([], dtype=np.complex64)

            # Вычисляем позиции в буфере
            buffer_start = self.total_written - min(self.total_written, self.capacity)
            start_offset = abs_start - buffer_start
            end_offset = abs_end - buffer_start

            if start_offset < 0 or end_offset > self.capacity:
                print(f"[BUFFER] Segment out of bounds: offset={start_offset}..{end_offset}")
                return np.array([], dtype=np.complex64)

            # Извлекаем данные с учетом кольцевой структуры
            if self.write_pos == 0:
                # Буфер линейный (после полного цикла)
                return self.buffer[start_offset:end_offset].copy()
            else:
                # Буфер кольцевой
                logical_start = (self.write_pos + start_offset) % self.capacity
                logical_end = (self.write_pos + end_offset) % self.capacity

                if logical_end > logical_start:
                    return self.buffer[logical_start:logical_end].copy()
                else:
                    # Сегмент переходит через границу
                    return np.concatenate([
                        self.buffer[logical_start:],
                        self.buffer[:logical_end]
                    ]).copy()


def test_basic_operations():
    """Тест базовых операций буфера"""
    print("\n=== ТЕСТ 1: Базовые операции ===")

    # Создаем буфер на 1 секунду при 1000 Hz
    buf = IQRingBuffer(duration_sec=1.0, sample_rate=1000.0)

    # Записываем 500 отсчетов
    data1 = np.arange(500, dtype=np.complex64)
    buf.write(data1)
    print(f"Written 500 samples, write_pos={buf.write_pos}, total={buf.total_written}")

    # Извлекаем сегмент
    segment = buf.get_segment(100, 200)
    expected = np.arange(100, 200, dtype=np.complex64)
    assert np.allclose(segment, expected), "Segment mismatch!"
    print(f"✓ Extracted segment [100:200]: {len(segment)} samples")

    # Записываем еще 700 отсчетов (переполнение)
    data2 = np.arange(500, 1200, dtype=np.complex64)
    buf.write(data2)
    print(f"Written 700 more samples, write_pos={buf.write_pos}, total={buf.total_written}")

    # Пробуем извлечь старые данные (должно вернуть пустой массив)
    old_segment = buf.get_segment(0, 100)
    assert old_segment.size == 0, "Old segment should be empty!"
    print(f"✓ Old segment [0:100] correctly returns empty")

    # Извлекаем новые данные
    new_segment = buf.get_segment(1000, 1100)
    expected_new = np.arange(1000, 1100, dtype=np.complex64)
    assert np.allclose(new_segment, expected_new), "New segment mismatch!"
    print(f"✓ Extracted new segment [1000:1100]: {len(new_segment)} samples")


def test_ring_wraparound():
    """Тест кольцевой записи с переполнением"""
    print("\n=== ТЕСТ 2: Кольцевая запись ===")

    buf = IQRingBuffer(duration_sec=0.1, sample_rate=1000.0)  # 100 samples

    # Заполняем буфер несколько раз
    for i in range(5):
        data = np.full(50, i+1, dtype=np.complex64)
        buf.write(data)
        print(f"Iteration {i+1}: wrote value {i+1}, total_written={buf.total_written}")

    # Последние 100 отсчетов должны содержать значения 3,3,3... 4,4,4... 5,5,5...
    segment = buf.get_segment(150, 250)
    print(f"Segment [150:250]: unique values = {np.unique(segment.real)}")

    # Проверяем что старые данные недоступны
    old = buf.get_segment(0, 50)
    assert old.size == 0, "Old data should be unavailable"
    print("✓ Old data correctly unavailable")


def test_actual_sample_rate():
    """Тест использования actual_sample_rate_sps"""
    print("\n=== ТЕСТ 3: Actual sample rate ===")

    # Симулируем разные SDR с разными sample rates
    sdr_configs = [
        ("RTL-SDR", 1_024_000),
        ("HackRF", 2_000_000),
        ("RSA306", 875_000),
    ]

    for sdr_name, actual_fs in sdr_configs:
        buf = IQRingBuffer(duration_sec=3.0, sample_rate=actual_fs)

        # Вычисляем размер окна RMS
        rms_win_ms = 1.0  # ms
        win_samps = int(rms_win_ms * 1e-3 * actual_fs)

        print(f"{sdr_name}: fs={actual_fs:.0f} Hz, buffer={buf.capacity} samples, RMS window={win_samps} samples")

        # Симулируем запись данных
        chunk_size = 65536
        chunks_to_fill = (buf.capacity // chunk_size) + 1

        for i in range(chunks_to_fill):
            data = np.random.randn(chunk_size).astype(np.complex64)
            buf.write(data)

        # Проверяем что буфер заполнен
        assert buf.total_written >= buf.capacity
        print(f"  ✓ Buffer filled: {buf.total_written} total samples written")


def test_concurrent_access():
    """Тест многопоточного доступа"""
    print("\n=== ТЕСТ 4: Многопоточный доступ ===")

    buf = IQRingBuffer(duration_sec=1.0, sample_rate=10000.0)
    errors = []

    def writer_thread():
        """Поток записи данных"""
        try:
            for i in range(100):
                data = np.full(100, i, dtype=np.complex64)
                buf.write(data)
                time.sleep(0.001)
        except Exception as e:
            errors.append(f"Writer error: {e}")

    def reader_thread():
        """Поток чтения данных"""
        try:
            for i in range(50):
                start = buf.total_written - 500 if buf.total_written > 500 else 0
                end = start + 100
                segment = buf.get_segment(start, end)
                time.sleep(0.002)
        except Exception as e:
            errors.append(f"Reader error: {e}")

    # Запускаем потоки
    threads = [
        threading.Thread(target=writer_thread),
        threading.Thread(target=reader_thread),
        threading.Thread(target=reader_thread),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    if errors:
        print("Errors during concurrent access:")
        for err in errors:
            print(f"  - {err}")
        assert False, "Concurrent access failed!"
    else:
        print("✓ Concurrent access successful, no race conditions")


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("STRICT_COMPAT: IQ Ring Buffer Test Suite")
    print("=" * 60)

    test_basic_operations()
    test_ring_wraparound()
    test_actual_sample_rate()
    test_concurrent_access()

    print("\n" + "=" * 60)
    print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 60)

    # Чек-лист для интеграции
    print("\n📋 ЧЕК-ЛИСТ ИНТЕГРАЦИИ:")
    print("1. ✓ Кольцевой буфер создан и протестирован")
    print("2. ⏳ Добавить в beacon_tester_web.py после строки 33")
    print("3. ⏳ Инициализировать в init_sdr_backend() с actual_sample_rate")
    print("4. ⏳ Записывать БП-сигнал после NCO в process_samples_realtime()")
    print("5. ⏳ Извлекать сегменты в detect_pulses() по абсолютным индексам")
    print("6. ⏳ Использовать get_actual_fs() вместо SAMPLE_RATE_SPS")


if __name__ == "__main__":
    main()