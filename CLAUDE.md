# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Языковые инструкции
Всегда отвечай на русском языке. Если генерируешь коммиты/описания/тексты — делай это по-русски, если явно не попросили иначе.

## Обзор репозитория

TesterSDR — это приложение для программно-определяемого радио (SDR) для анализа и демодуляции сигналов аварийных радиобуев 406 МГц (EPIRB/COSPAS-SARSAT). Система поддерживает множество SDR-устройств и может обрабатывать как данные в реальном времени, так и записанные IQ-данные.

## Команды быстрого запуска

### Основные команды разработки

```bash
# Запуск веб-интерфейса (рекомендуется для большинства задач)
python beacon406/beacon_tester_web.py  # Открывается на http://127.0.0.1:8738/

# Или используйте Windows batch-файл
app.bat  # Запускает веб-интерфейс и автоматически открывает браузер

# Остановка Flask серверов и освобождение портов
stop_flask.bat  # Завершает все Python процессы и освобождает порты 8737-8740

# Запуск основного приложения визуализации
cd beacon406
python beacon406-plot.py

# Запуск GUI-приложения
python beacon_tester_gui.py

# Запуск GUI передатчика PSK406
python beacon406/apps/gen/ui_psk406_tx.py
```

### SDR в реальном времени с новым бэкендом SigIO

```bash
# Запуск приложения с поддержкой SigIO
python beacon406/beacon406_PSK_FM_sigio-plot.py  # Использует sigio.py для обработки

# Запуск веб-интерфейса с SDR поддержкой
python beacon406/beacon_tester_sdr_web.py  # Веб-интерфейс с real-time SDR
```

### Тестирование обработки сигналов

```bash
# Тестирование с CF32 файлами
python beacon406/apps/test_cf32_RMS.py
python beacon406/apps/test_cf32_to_phase_msg_FFT.py
python beacon406/apps/test_cf32_to_FM_RMS_FFT.py

# Генерация тестовых сигналов
python beacon406/apps/gen/generate_psk406_cf32.py

# Циклическая передача сигнала через HackRF (каждые 4 секунды)
beacon406/apps/gen/406_msg_send_4sec.bat

# Декодирование EPIRB сообщений из hex
python beacon406/apps/epirb_hex_decoder.py
```

## Архитектура

### Основные компоненты

**beacon406/** - Основной Python модуль:
- `beacon406-plot.py` - Основное приложение визуализации с real-time обработкой, RMS-анализом и PSK-демодуляцией
- `beacon406_PSK_FM-plot.py` - Расширенная версия с поддержкой FM демодуляции
- `beacon406_PSK_FM_sigio-plot.py` - Версия с новым SigIO бэкендом
- `beacon_tester_gui.py` - PySide6/Qt GUI со спектром, водопадом, PSK-демодуляцией
- `beacon_tester_web.py` - Веб-интерфейс Flask (порт 8738) с загрузкой CF32 файлов
- `beacon_tester_sdr_web.py` - Веб-интерфейс с real-time SDR поддержкой

**beacon406/lib/** - Библиотеки обработки:
- `backends.py` - Унифицированный слой SDR абстракции (RTL-SDR, HackRF, Airspy, SDRPlay, RSA306)
- `backends_sigio.py` - Альтернативный бэкенд с SigIO интеграцией
- `sigio.py` - Новая библиотека ввода-вывода сигналов
- `demod.py` - PSK-демодуляция с безопасным извлечением сообщений
- `metrics.py` - Фазовые метрики и обработка PSK импульсов
- `processing_fm.py` - FM демодуляция и обработка
- `config.py` - Настройка SDR бэкенда
- `hex_decoder.py` - Декодер EPIRB/COSPAS-SARSAT сообщений

**beacon406/apps/** - Утилиты и тесты:
- `epirb_hex_decoder.py` - Автономный декодер EPIRB HEX
- `test_cf32_*.py` - Тестовые утилиты CF32 обработки
- **gen/** - Генерация и передача сигналов:
  - `generate_psk406_cf32.py` - Генерация PSK406 сигналов
  - `psk406_msg_gen.py` - Генератор PSK406 сообщений
  - `backend_hackrf_tx.py` - HackRF передача
  - `ui_psk406_tx.py` - GUI передатчика
  - `make_multitone_156mhz.py` - Многотональные тестовые сигналы
  - `406_msg_send_4sec.bat` - Циклическая передача HackRF

**captures/** - CF32 IQ записи (исключен из Git)
**captures/uploads/** - Файлы веб-загрузок

### Система бэкендов

#### Стандартный бэкенд (backends.py)
- Поддержка: RTL-SDR, HackRF, Airspy, SDRPlay, RSA306, воспроизведение файлов
- Автоопределение устройств в режиме `"auto"`
- Калибровочные смещения в `SDR_CALIB_OFFSETS_DB`
- Оптимальные частоты дискретизации в `SDR_DEFAULT_HW_SR`

#### SigIO бэкенд (backends_sigio.py + sigio.py)
- Новая архитектура с улучшенной буферизацией
- Поддержка тех же SDR через sigio.open_iq()
- Оптимизированная обработка потоков данных

### Конфигурация бэкенда

Редактируйте `beacon406/lib/config.py`:
```python
# Автоопределение SDR
BACKEND_NAME = "auto"
BACKEND_ARGS = None

# Конкретное устройство
BACKEND_NAME = "soapy_hackrf"  # или "soapy_rtl", "soapy_airspy", "soapy_sdrplay", "rsa306"

# Воспроизведение файла
BACKEND_NAME = "file"
BACKEND_ARGS = r"C:/work/TesterSDR/captures/your_recording.cf32"
```

## Конвейер обработки сигналов

### PSK-демодуляция
1. **Детекция импульсов**: RMS расчет → пороговая детекция (-60 dBm)
2. **Извлечение фазы**: Комплексный аргумент IQ данных
3. **Фильтрация**: ФНЧ 12 кГц, 129-отводный FIR (окно Хэмминга)
4. **Децимация**: 4x после фильтрации
5. **Детекция фронтов**: Скользящее окно 40 отсчетов, порог 0.5
6. **Манчестерское декодирование**: 10→1, 01→0
7. **Декодирование COSPAS-SARSAT**: 144-битные сообщения маяков

### FM-демодуляция (processing_fm.py)
1. **Детекция огибающей**: Амплитуда комплексного сигнала
2. **Нормализация**: Автомасштабирование для визуализации
3. **Спектральный анализ**: FFT с окном Хэмминга

## Веб-интерфейс

### Маршруты API (beacon_tester_web.py)
- `GET /` - Главная страница
- `POST /api/upload` - Загрузка CF32 файлов
- `GET /api/status` - Текущие метрики
- `POST /api/run` - Запуск SDR обработки (beacon_tester_sdr_web.py)
- `POST /api/stop` - Остановка SDR (beacon_tester_sdr_web.py)

### JavaScript фронтенд
- Canvas график фазы с прореживанием до ~1000 точек
- Таблица декодированных EPIRB сообщений
- Асинхронная загрузка файлов
- Real-time обновления через polling

## Зависимости

```bash
pip install numpy matplotlib pyqtgraph PySide6 scipy flask werkzeug
pip install SoapySDR  # Для SDR устройств
pip install pyrtlsdr  # Для прямой поддержки RTL-SDR
pip install pyVISA    # Для Tektronix RSA306
```

## Важные параметры

### Основные константы обработки
- `SAMPLE_RATE`: 1 МГц (стандартная частота дискретизации)
- `RMS_THRESH_DBM`: -60 dBm (порог детекции импульсов)
- `RMS_WIN_MS`: 1.0 мс (окно RMS)
- `PSK_BASELINE_MS`: 10.0 мс (фазовая референция)
- `TARGET_SIGNAL_HZ`: 406.037 МГц (частота маяков)
- `IF_OFFSET_HZ`: -37 кГц (промежуточная частота)

### Калибровочные смещения SDR (dB)
- RTL-SDR: 0.0
- HackRF: -10.0
- Airspy: 20.0
- SDRPlay: 0.0
- RSA306: 12.0

## Специфика Windows

- Пути используют прямые слеши: `C:/work/TesterSDR/`
- Python путь в batch файлах: `C:\Users\alexb\AppData\Local\Programs\Python\Python39\python.exe`
- Порты Flask: 8737-8740 (освобождаются через stop_flask.bat)
- Все сервисы привязаны к 127.0.0.1 (localhost)

## Форматы файлов

- **CF32**: Комплексные float32 IQ (чередующиеся I/Q пары)
- **Поддерживаемые расширения**: .cf32, .iq, .f32, .bin, .wav, .raw, .dat
- **Тестовые файлы в captures/**: PSK406, AIS, DSC, AM121 сигналы

## Отладка и тестирование

- Используйте режим воспроизведения файлов для разработки
- Тестовые скрипты в `beacon406/apps/test_*.py`
- Веб-интерфейс для быстрой проверки CF32 файлов
- Real-time тестирование с подключенным SDR