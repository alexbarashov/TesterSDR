#!/usr/bin/env python3
"""
STRICT_COMPAT: Simple IQ Ring Buffer Test
Step 1: Test ring buffer for 3+ seconds baseband signal storage
"""

import numpy as np
import threading
import time

# STRICT_COMPAT: IQ Ring buffer class
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
        """Write samples to ring buffer"""
        with self.lock:
            n = len(samples)
            if n >= self.capacity:
                # If data exceeds capacity, take the last samples
                self.buffer[:] = samples[-self.capacity:]
                self.write_pos = 0
                self.total_written += n
            else:
                # Write in two parts if wrapping around boundary
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
        """Extract segment by absolute indices"""
        with self.lock:
            # Check data availability
            oldest_available = max(0, self.total_written - self.capacity)
            if abs_start < oldest_available:
                print(f"[BUFFER] Segment too old: start={abs_start} < oldest={oldest_available}")
                return np.array([], dtype=np.complex64)

            # Calculate relative positions from start of available data
            available_samples = min(self.total_written, self.capacity)
            buffer_abs_start = self.total_written - available_samples

            start_offset = abs_start - buffer_abs_start
            end_offset = abs_end - buffer_abs_start

            if start_offset < 0 or end_offset > available_samples:
                print(f"[BUFFER] Segment out of bounds: offset={start_offset}..{end_offset}, available={available_samples}")
                return np.array([], dtype=np.complex64)

            # Case 1: Buffer not yet full (linear access)
            if self.total_written <= self.capacity:
                return self.buffer[start_offset:end_offset].copy()

            # Case 2: Buffer is full and wrapped (ring access)
            else:
                # write_pos points to oldest data in ring buffer
                logical_start = (self.write_pos + start_offset) % self.capacity
                logical_end = (self.write_pos + end_offset) % self.capacity

                if logical_end > logical_start:
                    # Segment doesn't wrap around
                    return self.buffer[logical_start:logical_end].copy()
                else:
                    # Segment wraps around buffer boundary
                    return np.concatenate([
                        self.buffer[logical_start:],
                        self.buffer[:logical_end]
                    ]).copy()


def main():
    """Run basic tests"""
    print("=" * 60)
    print("STRICT_COMPAT: IQ Ring Buffer Test")
    print("=" * 60)

    # Test 1: Basic operations
    print("\n=== TEST 1: Basic Operations ===")
    buf = IQRingBuffer(duration_sec=1.0, sample_rate=1000.0)

    # Write 500 samples
    data1 = np.arange(500, dtype=np.complex64)
    buf.write(data1)
    print(f"Written 500 samples, write_pos={buf.write_pos}, total={buf.total_written}")

    # Extract segment
    segment = buf.get_segment(100, 200)
    expected = np.arange(100, 200, dtype=np.complex64)
    assert np.allclose(segment, expected), "Segment mismatch!"
    print(f"OK Extracted segment [100:200]: {len(segment)} samples")

    # Test 2: Ring wraparound
    print("\n=== TEST 2: Ring Wraparound ===")
    buf2 = IQRingBuffer(duration_sec=0.1, sample_rate=1000.0)  # 100 samples

    # Fill buffer multiple times
    for i in range(5):
        data = np.full(50, i+1, dtype=np.complex64)
        buf2.write(data)
        print(f"Iteration {i+1}: wrote value {i+1}, total_written={buf2.total_written}")

    # Check that old data is unavailable
    old = buf2.get_segment(0, 50)
    assert old.size == 0, "Old data should be unavailable"
    print("OK Old data correctly unavailable")

    # Test 3: SDR sample rates
    print("\n=== TEST 3: Different Sample Rates ===")
    sdr_configs = [
        ("RTL-SDR", 1_024_000),
        ("HackRF", 2_000_000),
        ("RSA306", 875_000),
    ]

    for sdr_name, actual_fs in sdr_configs:
        buf3 = IQRingBuffer(duration_sec=3.0, sample_rate=actual_fs)
        rms_win_ms = 1.0  # ms
        win_samps = int(rms_win_ms * 1e-3 * actual_fs)
        print(f"{sdr_name}: fs={actual_fs:.0f} Hz, buffer={buf3.capacity} samples, RMS window={win_samps} samples")

    print("\n" + "=" * 60)
    print("OK ALL TESTS PASSED!")
    print("=" * 60)

    # Integration checklist
    print("\nINTEGRATION CHECKLIST:")
    print("1. OK Ring buffer created and tested")
    print("2. ⏳ Add to beacon_tester_web.py after line 33")
    print("3. ⏳ Initialize in init_sdr_backend() with actual_sample_rate")
    print("4. ⏳ Write BP signal after NCO in process_samples_realtime()")
    print("5. ⏳ Extract segments in detect_pulses() by absolute indices")
    print("6. ⏳ Use get_actual_fs() instead of SAMPLE_RATE_SPS")

if __name__ == "__main__":
    main()