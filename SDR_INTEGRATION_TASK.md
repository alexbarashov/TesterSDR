# Задание: Интеграция SDR в веб-интерфейс COSPAS Beacon Tester

## Цель
Интегрировать модуль SDR backend.py в веб-интерфейс beacon_tester_web.py для работы с реальными SDR устройствами в режиме реального времени, аналогично реализации в beacon406-plot.py.

## Технические требования

### 1. Увеличение версии
- Изменить версию с 2.0 на 2.1 в заголовке файла

### 2. Импорты и SDR параметры
```python
# Добавить импорты
from lib.backends import make_backend  # SDR backend support
from lib.config import BACKEND_NAME, BACKEND_ARGS

# SDR параметры (аналогично beacon406-plot.py)
TARGET_SIGNAL_HZ    = 406_037_000     # Целевая частота (406 МГц маяки)
IF_OFFSET_HZ        = -37_000         # ПЧ-смещение
CENTER_FREQ_HZ      = TARGET_SIGNAL_HZ + IF_OFFSET_HZ
SAMPLE_RATE_SPS     = 1_000_000       # 1 МГц частота дискретизации
USE_MANUAL_GAIN     = True
TUNER_GAIN_DB       = 30.0
ENABLE_AGC          = False
FREQ_CORR_PPM       = 0

# Параметры обработки сигналов
BB_SHIFT_ENABLE     = True
BB_SHIFT_HZ         = IF_OFFSET_HZ
RMS_WIN_MS          = 1.0
DBM_OFFSET_DB       = -30.0
PULSE_THRESH_DBM    = -45.0
READ_CHUNK          = 65536
```

### 3. Глобальные переменные SDR
```python
# SDR состояние
sdr_backend = None
sdr_running = False
sdr_device_info = "No SDR detected"
sdr_capture_mode = "idle"  # "idle", "single", "continuous"

# Threading для фонового чтения SDR
reader_thread = None
data_queue = queue.Queue(maxsize=10)
```

### 4. Функции инициализации SDR
```python
def init_sdr_backend():
    """Инициализация SDR backend"""
    global sdr_backend, sdr_device_info
    try:
        print(f"[SDR] Trying to initialize backend: {BACKEND_NAME}")
        sdr_backend = make_backend(BACKEND_NAME, BACKEND_ARGS)

        # Настройка параметров
        sdr_backend.set_center_freq(CENTER_FREQ_HZ)
        sdr_backend.set_sample_rate(SAMPLE_RATE_SPS)

        if USE_MANUAL_GAIN:
            sdr_backend.set_gain_mode(False)
            sdr_backend.set_gain(TUNER_GAIN_DB)
        else:
            sdr_backend.set_gain_mode(True)

        sdr_backend.set_freq_correction(FREQ_CORR_PPM)

        # Получаем информацию об устройстве
        try:
            sdr_device_info = sdr_backend.get_device_info()
        except Exception:
            sdr_device_info = f"{BACKEND_NAME} device detected"

        print(f"[SDR] Backend {BACKEND_NAME} initialized")
        return True

    except Exception as e:
        print(f"[SDR] Backend initialization error: {e}")
        sdr_device_info = "No SDR detected"
        return False
```

### 5. Функции управления захватом
```python
def start_sdr_capture():
    """Запуск захвата SDR"""
    global sdr_running
    try:
        if sdr_backend:
            sdr_backend.start()
            sdr_running = True
            print("[SDR] Capture started")
            return True
    except Exception as e:
        print(f"[SDR] Capture start error: {e}")
        sdr_running = False
    return False

def stop_sdr_capture():
    """Остановка захвата SDR"""
    global sdr_running
    try:
        if sdr_backend:
            sdr_backend.stop()
            sdr_running = False
            print("[SDR] Capture stopped")
    except Exception as e:
        print(f"[SDR] Capture stop error: {e}")
```

### 6. Фоновый поток для чтения данных
```python
def sdr_reader_thread():
    """Фоновый поток для чтения данных SDR"""
    global sdr_running

    print("[READER] SDR reader thread started")

    while sdr_running:
        try:
            if sdr_backend and sdr_running:
                # Чтение данных
                iq_data = read_sdr_data()
                if iq_data is not None and len(iq_data) > 0:
                    # Обработка импульса
                    result = process_captured_pulse(iq_data)
                    if result:
                        # Помещаем в очередь для веб-интерфейса
                        try:
                            data_queue.put_nowait(result)
                        except queue.Full:
                            # Очередь полная, удаляем старые данные
                            try:
                                data_queue.get_nowait()
                                data_queue.put_nowait(result)
                            except queue.Empty:
                                pass
                else:
                    time.sleep(0.01)  # Короткая пауза если нет данных
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"[READER] Processing error: {e}")
            time.sleep(0.001)

    print("[READER] SDR reader thread stopped")

def read_sdr_data():
    """Чтение данных с SDR"""
    try:
        if sdr_backend and sdr_running:
            return sdr_backend.read_samples(READ_CHUNK)
    except Exception as e:
        print(f"[SDR] Data read error: {e}")
        return None

def process_captured_pulse(iq_data):
    """Обработка захваченного импульса"""
    try:
        # Детекция импульса по RMS
        iq_seg = _find_pulse_segment_realtime(iq_data)
        if iq_seg is None:
            return None

        # Обработка сигнала
        pulse_result = process_psk_impulse(
            iq_seg=iq_seg,
            fs=SAMPLE_RATE_SPS,
            baseline_ms=10.0,
            t0_offset_ms=0.0,
            use_lpf_decim=True,
            remove_slope=True
        )

        if not pulse_result or "phase_rad" not in pulse_result:
            return None

        # Демодуляция PSK
        msg_hex, phase_res, edges = phase_demod_psk_msg_safe(
            data=pulse_result["phase_rad"]
        )

        return {
            "success": True,
            "msg_hex": msg_hex if msg_hex else "",
            "phase_data": pulse_result.get("phase_rad", []),
            "xs_fm_ms": pulse_result.get("xs_ms", []),
            "pulse_detected": True
        }

    except Exception as e:
        print(f"[PULSE PROCESS] Processing error: {e}")
        return None
```

### 7. Обновление API маршрутов

#### Measure - инициализация SDR
```python
@app.route('/api/measure', methods=['POST'])
def api_measure():
    """Инициализация SDR при первом запуске и проверка статуса"""
    global sdr_capture_mode

    print(f"[MEASURE] Called api_measure(), sdr_backend is None: {sdr_backend is None}")

    # Инициализируем SDR если он не был инициализирован
    if sdr_backend is None:
        if init_sdr_backend():
            return jsonify({
                'status': 'SDR initialized',
                'sdr_device_info': sdr_device_info
            })
        else:
            return jsonify({
                'status': 'SDR initialization failed',
                'sdr_device_info': sdr_device_info
            })

    # SDR уже инициализирован, проверяем статус
    try:
        sdr_status = {}
        if hasattr(sdr_backend, 'get_center_freq'):
            sdr_status['center_freq'] = sdr_backend.get_center_freq()
        if hasattr(sdr_backend, 'get_sample_rate'):
            sdr_status['sample_rate'] = sdr_backend.get_sample_rate()
        if hasattr(sdr_backend, 'get_gain'):
            sdr_status['gain'] = sdr_backend.get_gain()

        return jsonify({
            'status': 'SDR ready',
            'sdr_device_info': sdr_device_info,
            'sdr_status': sdr_status
        })

    except Exception as e:
        print(f"[MEASURE] Status check failed: {e}")
        return jsonify({
            'status': 'SDR connection issue',
            'sdr_device_info': sdr_device_info
        })
```

#### RUN - одиночный захват
```python
@app.route('/api/run', methods=['POST'])
def api_run():
    """Запуск одиночного захвата импульса"""
    global sdr_capture_mode, reader_thread

    if sdr_backend is None:
        return jsonify({'status': 'error', 'message': 'SDR not initialized'})

    sdr_capture_mode = "single"

    # Запускаем захват
    if start_sdr_capture():
        # Запускаем фоновый поток если его нет
        if reader_thread is None or not reader_thread.is_alive():
            reader_thread = threading.Thread(target=sdr_reader_thread, daemon=True)
            reader_thread.start()

        return jsonify({'status': 'single capture started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start capture'})
```

#### CONT - непрерывный захват
```python
@app.route('/api/cont', methods=['POST'])
def api_cont():
    """Запуск непрерывного захвата импульсов"""
    global sdr_capture_mode, reader_thread

    if sdr_backend is None:
        return jsonify({'status': 'error', 'message': 'SDR not initialized'})

    sdr_capture_mode = "continuous"

    # Запускаем захват
    if start_sdr_capture():
        # Запускаем фоновый поток если его нет
        if reader_thread is None or not reader_thread.is_alive():
            reader_thread = threading.Thread(target=sdr_reader_thread, daemon=True)
            reader_thread.start()

        return jsonify({'status': 'continuous capture started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start capture'})
```

#### BREAK - остановка захвата
```python
@app.route('/api/break', methods=['POST'])
def api_break():
    """Остановка захвата импульсов"""
    global sdr_capture_mode, reader_thread

    sdr_capture_mode = "idle"
    stop_sdr_capture()

    # Ждем остановки потока
    if reader_thread and reader_thread.is_alive():
        # Не джойним текущий поток сам к себе
        if threading.current_thread() != reader_thread:
            reader_thread.join(timeout=2.0)

    return jsonify({'status': 'capture stopped'})
```

### 8. Обновление API статуса
```python
# В функции api_status() добавить:
# Обработка данных из очереди SDR
try:
    while not data_queue.empty():
        sdr_data = data_queue.get_nowait()
        if sdr_data and sdr_data.get("success"):
            # Обновляем STATE данными из SDR
            STATE.hex_message = sdr_data.get("msg_hex", "")
            STATE.phase_data = sdr_data.get("phase_data", [])
            STATE.xs_fm_ms = sdr_data.get("xs_ms", [])
            STATE.message = f"SDR captured: {STATE.hex_message[:16]}..."
except queue.Empty:
    pass

# Добавить в JSON ответ:
'sdr_active': sdr_running,
'sdr_backend': BACKEND_NAME,
'sdr_center_freq': CENTER_FREQ_HZ,
'sdr_sample_rate': SAMPLE_RATE_SPS,
'sdr_status': sdr_status,
'sdr_device_info': sdr_device_info,
'sdr_capture_mode': sdr_capture_mode
```

### 9. Обновление HTML интерфейса
```html
<!-- Добавить поле для отображения SDR устройства -->
<div style="margin-bottom: 10px;">
    <span style="color: #666; font-size: 12px;">SDR Device: </span>
    <span id="sdrStatus" style="color: #333; font-size: 12px;">Not initialized</span>
</div>
```

### 10. Обновление JavaScript
```javascript
// В функции measure()
async function measure() {
    console.log('Measure button clicked');
    try {
        const response = await fetch('/api/measure', { method: 'POST' });
        const data = await response.json();
        console.log('Measure response:', data);

        // Обновляем информацию об SDR устройстве
        const sdrStatus = document.getElementById('sdrStatus');
        if (sdrStatus && data.sdr_device_info) {
            sdrStatus.textContent = data.sdr_device_info;
            console.log('Updated SDR status to:', data.sdr_device_info);
        }
    } catch (error) {
        console.error('Measure error:', error);
    }
}
```

### 11. Оптимизация производительности
```python
# Функция прореживания данных для веб-интерфейса
def _decimate_for_web(data, max_points=2000):
    """Прореживает данные для веб-интерфейса до максимального количества точек"""
    if not data or len(data) <= max_points:
        return data

    # Равномерная выборка
    step = len(data) // max_points
    if step < 1:
        step = 1

    return data[::step][:max_points]

# Использование в API статуса:
'phase_data': _decimate_for_web(STATE.phase_data),
'xs_fm_ms': _decimate_for_web(STATE.xs_fm_ms),
```

### 12. Исправление JavaScript для больших массивов
```javascript
// Заменить Math.min(...phaseData) на безопасные функции
const minPhase = phaseData.reduce((min, val) => Math.min(min, val), phaseData[0]);
const maxPhase = phaseData.reduce((max, val) => Math.max(max, val), phaseData[0]);
const minXs = xsData.reduce((min, val) => Math.min(min, val), xsData[0]);
const maxXs = xsData.reduce((max, val) => Math.max(max, val), xsData[0]);
```

## Ожидаемый результат

После реализации веб-интерфейс должен:

1. **Measure** - инициализировать SDR и показывать информацию об устройстве
2. **RUN** - выполнять одиночный захват импульса и останавливаться
3. **Cont** - выполнять непрерывный захват импульсов
4. **Break** - останавливать любой активный захват
5. Отображать данные фазы в реальном времени
6. Декодировать EPIRB сообщения
7. Поддерживать загрузку CF32 файлов параллельно с SDR

## Тестирование

1. Проверить инициализацию SDR (RSA306B, HackRF, RTL-SDR)
2. Проверить режимы single/continuous захвата
3. Проверить отображение данных в веб-интерфейсе
4. Проверить совместимость с загрузкой файлов
5. Проверить производительность с большими массивами данных