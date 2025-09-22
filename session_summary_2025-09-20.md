# Работа с beacon_tester_web.py - Сессия 20.09.2025

## Обзор выполненных работ

Сегодня была проведена полная реконструкция функционала beacon_tester_web.py с интеграцией обработки CF32 файлов и исправлением проблем с переключением режимов.

## Основные изменения

### 1. Добавление библиотек обработки сигналов

**Добавленные импорты:**
```python
from lib.metrics import process_psk_impulse
from lib.demod import phase_demod_psk_msg_safe
import numpy as np
from werkzeug.utils import secure_filename
```

### 2. Функция поиска импульса

**Создана функция `_find_pulse_segment()`:**
- Поиск импульса по RMS порогу
- Основана на коде из `test_cf32_to_phase_msg_FFT.py`
- Фильтрация коротких импульсов (менее 5ms)
- Компенсация задержки и окна

```python
def _find_pulse_segment(iq_data, sample_rate, thresh_dbm, win_ms, start_delay_ms, calib_db):
    # Вычисляем RMS в dBm
    W = max(1, int(win_ms * 1e-3 * sample_rate))
    p = np.abs(iq_data)**2
    ma = np.convolve(p, np.ones(W)/W, mode="same")
    rms = np.sqrt(ma + 1e-30)
    rms_dbm = 20*np.log10(rms + 1e-30) + calib_db
    # ... логика поиска импульса
```

### 3. Функция обработки CF32 файлов

**Создана функция `process_cf32_file()`:**
- Чтение CF32 файлов (complex float32)
- Поиск импульса через RMS анализ
- Обработка через `process_psk_impulse()` из библиотеки metrics
- PSK демодуляция через `phase_demod_psk_msg_safe()`
- Безопасное преобразование numpy массивов в списки
- Извлечение метрик фазы (PosPhase, NegPhase, PhRise, PhFall, Asymmetry, Tmod)

### 4. Обновление структуры данных BeaconState

**Добавлены новые поля:**
```python
@dataclass
class BeaconState:
    # Новые поля для CF32 обработки
    hex_message: str = ""  # HEX сообщение из файла
    current_file: str = ""  # Путь к файлу
    phase_data: list = field(default_factory=list)  # Данные фазы для графика
    xs_fm_ms: list = field(default_factory=list)  # Временная шкала

    # Метрики из demod (все изначально 0.0)
    pos_phase: float = 0.0
    neg_phase: float = 0.0
    ph_rise: float = 0.0
    ph_fall: float = 0.0
    asymmetry: float = 0.0
    t_mod: float = 0.0
    rms_dbm: float = 0.0
    freq_hz: float = 0.0
```

### 5. Файловая загрузка и обработка

**Endpoint `/api/upload`:**
- Проверка расширения файла (.cf32)
- Безопасное сохранение в `captures/uploads/`
- Автоматическая обработка через `process_cf32_file()`
- Обновление STATE с результатами обработки
- Сохранение фазовых данных для графика

**HTML интерфейс:**
```html
<button class="button" onclick="loadFile()">File</button>
<input type="file" id="fileInput" accept=".cf32" style="display: none;" onchange="uploadFile(this)">
```

### 6. Исправление обработки numpy массивов

**Проблема:** "The truth value of an array with more than one element is ambiguous"

**Решение:**
- Добавлены проверки `isinstance(phase_data, np.ndarray) and phase_data.size > 0`
- Использование `np.isfinite()` для скалярных значений
- Обертка `float()` для всех numpy значений перед передачей в JSON

### 7. Визуализация фазовых данных

**Обновлена функция `drawChart()`:**
- Поддержка реальных данных из `phase_data` и `xs_fm_ms`
- Временное масштабирование по оси X
- Масштабирование фазы ±1.5 радиан
- Автоматическое определение диапазона времени

### 8. Исправление переключения режимов

**Главная проблема:** При переключении с "121" на "Message" оставалась таблица 121

**Решение:**
1. **Централизованная обработка в `fetchData()`:**
```javascript
async function fetchData() {
    const response = await fetch('/api/status');
    const data = await response.json();

    // Обрабатываем специальные режимы
    if (currentView === 'message') {
        drawMessageTable(data.hex_message || '');
    } else if (currentView === '121_data') {
        draw121DataTable(data);
    } else {
        updateDisplay(data);
    }
}
```

2. **Упрощение `changeView()`:**
```javascript
function changeView(viewType) {
    currentView = viewType;

    // Очищаем canvas при переключении
    const canvas = document.getElementById('phaseChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Обновляем заголовки...

    // Всегда вызываем fetchData() - специальные режимы обрабатываются внутри
    fetchData();
}
```

3. **Удаление дублирующей логики из `updateDisplay()`**

### 9. Декодирование EPIRB сообщений

**Функция `drawMessageTable()`:**
- Отображение таблицы декодирования 144-битных сообщений COSPAS-SARSAT
- Поля: Bit Range, Binary Content, Field Name, Decoded Value
- Инструкция при отсутствии HEX сообщения
- Использование HEX данных из загруженного CF32 файла

### 10. Таблица параметров 121.5 МГц

**Функция `draw121DataTable()`:**
- Параметры передатчика ELT на 121.5 МГц
- Carrier Frequency, Power, Sweep Period, Modulation Index
- Пустые значения до загрузки данных

## Исправленные ошибки

### 1. Numpy array truth value errors
- **Проблема:** Прямое использование numpy массивов в условиях
- **Решение:** Добавлены проверки типов и размеров массивов

### 2. Переключение режимов
- **Проблема:** Контент предыдущего режима оставался при переключении
- **Решение:** Централизованная обработка в fetchData() с очисткой canvas

### 3. Пустые графики
- **Проблема:** Данные фазы не передавались на frontend
- **Решение:** Правильное сохранение phase_data и xs_fm_ms в STATE

### 4. Отсутствие HEX сообщений
- **Проблема:** msg_hex не сохранялся в STATE
- **Решение:** Обновление STATE.hex_message при обработке файла

## Параметры обработки

**Используемые константы:**
- Sample Rate: 1 MHz
- RMS Threshold: -60.0 dBm
- RMS Window: 1.0 ms
- Start Delay: 3.0 ms
- Calibration: -30.0 dB
- PSK Baseline: 10.0 ms

## Структура файлов

```
beacon406/
├── beacon_tester_web.py           # Основной веб-сервер
├── lib/
│   ├── metrics.py                 # Обработка PSK импульсов
│   ├── demod.py                   # PSK демодуляция
│   └── hex_decoder.py             # Декодирование EPIRB сообщений
└── ../captures/uploads/           # Загруженные CF32 файлы
```

## Результат

✅ **Полностью функциональная система:**
- Загрузка CF32 файлов через веб-интерфейс
- Автоматическая обработка сигналов маяков 406 МГц
- Визуализация фазовых данных в реальном времени
- Корректное переключение между режимами просмотра
- Декодирование COSPAS-SARSAT сообщений
- Отображение метрик в таблице Current

**Веб-интерфейс доступен:** http://127.0.0.1:8738/

## Тестирование

Проверено на файле `psk406msg_f100.cf32`:
- ✅ Загрузка файла успешна
- ✅ Обработка сигнала выполнена
- ✅ HEX сообщение декодировано
- ✅ Фазовые данные отображены на графике
- ✅ Метрики обновлены в Current таблице
- ✅ Переключение режимов работает корректно

## Технические детали

**Цепочка обработки CF32:**
1. Файл → numpy.fromfile() → complex64 массив
2. RMS анализ → поиск импульса по порогу
3. process_psk_impulse() → фазовые данные и метрики
4. phase_demod_psk_msg_safe() → HEX сообщение
5. Обновление STATE → передача на frontend
6. Визуализация в canvas → таблицы декодирования

**Безопасность:**
- Проверка расширений файлов (.cf32 только)
- secure_filename() для безопасных имен
- Обработка ошибок во всех критических функциях
- Изоляция uploads в отдельной папке