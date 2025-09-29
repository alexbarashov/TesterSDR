# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Языковые инструкции
Всегда отвечай на русском языке. Если генерируешь коммиты/описания/тексты — делай это по-русски, если явно не попросили иначе.

## Версия Python
Проект использует Python 3.9. Весь код должен быть совместим с Python 3.9.

## Обзор репозитория

TesterSDR — это приложение для программно-определяемого радио (SDR) для анализа и демодуляции сигналов аварийных радиобуев 406 МГц (EPIRB/COSPAS-SARSAT). Система поддерживает множество SDR-устройств и может обрабатывать как данные в реальном времени, так и записанные IQ-данные.

## Команды быстрого запуска

### Основные команды разработки

```bash
# Запуск веб-интерфейса (рекомендуется для большинства задач)
python beacon406/beacon_tester_dsp2web.py  # Открывается на http://127.0.0.1:8738/

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

# DSP-сервис с веб-интерфейсом (микросервисная архитектура)
python beacon406/beacon_dsp_service.py --pub tcp://127.0.0.1:8781 --rep tcp://127.0.0.1:8782
python beacon406/beacon_tester_web_dsp_only.py  # Клиент для DSP-сервиса
```

### Запуск приложений с визуализацией

```bash
# Основное приложение с RMS и PSK демодуляцией
python beacon406/beacon406-plot.py

# Версия с FM демодуляцией
python beacon406/beacon406_PSK_FM-plot.py
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
- `beacon_tester_gui.py` - PySide6/Qt GUI со спектром, водопадом, PSK-демодуляцией
- `beacon_tester_web.py` - Веб-интерфейс Flask (порт 8738) с загрузкой CF32 файлов (удален)
- `beacon_tester_dsp2web.py` - Монолитный веб-интерфейс с встроенной DSP обработкой (порт 8738)
- `beacon_tester_web_dsp_only.py` - Веб-клиент для отдельного DSP-сервиса (порт 8738)
- `beacon_dsp_service.py` - Headless DSP-движок с ZeroMQ IPC (порты 8781-8782)

**beacon406/lib/** - Библиотеки обработки:
- `backends.py` - Унифицированный слой SDR абстракции (RTL-SDR, HackRF, Airspy, SDRPlay, RSA306)
- `backends_sigio.py` - Альтернативный бэкенд с SigIO интеграцией
- `sigio.py` - Библиотека ввода-вывода сигналов (поддержка SigMF формата)
- `demod.py` - PSK-демодуляция с безопасным извлечением сообщений
- `metrics.py` - Фазовые метрики и обработка PSK импульсов
- `processing_fm.py` - FM демодуляция и обработка
- `config.py` - Настройка SDR бэкенда
- `hex_decoder.py` - Декодер EPIRB/COSPAS-SARSAT сообщений
- `logger.py` - Централизованная система логирования

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
  - `406_send_once.bat` - Однократная передача сигнала
  - `HackRF_tx.bat` - Прямая передача через hackrf_transfer
  - `ui_psk406.bat` - Запуск GUI передатчика
  - `ui_psk406_con.bat` - Запуск GUI с консольным режимом
  - `ui_psk406_con_tx.py` - Консольная версия передатчика

**captures/** - CF32 IQ записи (исключен из Git)
**captures/uploads/** - Файлы веб-загрузок

### Система бэкендов

#### Стандартный бэкенд (backends.py)
- Поддержка: RTL-SDR, HackRF, Airspy, SDRPlay, RSA306, воспроизведение файлов
- Автоопределение устройств в режиме `"auto"`
- Калибровочные смещения в `SDR_CALIB_OFFSETS_DB`
- Оптимальные частоты дискретизации в `SDR_DEFAULT_HW_SR`

#### SigIO бэкенд (sigio.py)
- Архитектура с улучшенной буферизацией
- Поддержка SigMF формата для записи/воспроизведения
- Оптимизированная обработка потоков данных с полифазной децимацией

### DSP-сервис архитектура

#### beacon_dsp_service.py
- Headless DSP-движок без GUI и Flask
- Кольцевой буфер IQ с NCO (baseband shift)
- Скользящее RMS (1 мс окно) с детекцией импульсов
- Детектор с OFF-HANG (защита от дробления)
- PSK демодуляция и метрики через lib.metrics/lib.demod
- ZeroMQ PUB для событий (status/pulse/psk) на порту 8781
- ZeroMQ REP для команд (start/stop/set_params) на порту 8782
- Опциональная запись JSONL-сессии

#### Клиент-серверное взаимодействие
- beacon_tester_web_dsp_only.py подключается к DSP-сервису через ZeroMQ
- Подписка на PUB канал для получения событий в реальном времени
- REP канал для управления DSP (старт/стоп, параметры)
- Веб-интерфейс обновляется асинхронно от DSP событий

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

### Маршруты API (beacon_tester_dsp2web.py)
- `GET /` - Главная страница
- `POST /api/upload` - Загрузка CF32 файлов
- `GET /api/status` - Текущие метрики и SDR информация
- `POST /api/run` - Запуск real-time SDR обработки
- `POST /api/cont` - Продолжение работы после паузы
- `POST /api/break` - Пауза обработки
- `POST /api/measure` - Измерение параметров
- `POST /api/load` - Загрузка конфигурации
- `POST /api/save` - Сохранение конфигурации

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
pip install pyzmq     # Для DSP-сервиса с ZeroMQ IPC (опционально)
```

## Важные параметры

### Основные константы обработки
- `SAMPLE_RATE`: 1 МГц (стандартная частота дискретизации)
- `RMS_THRESH_DBM`: -60 dBm (порог детекции импульсов)
- `RMS_WIN_MS`: 1.0 мс (окно RMS)
- `PSK_BASELINE_MS`: 2.0 мс (фазовая референция)
- `PULSE_THRESH_DBM`: -45.0 dBm (порог детекции импульса)
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
- Порты ZeroMQ DSP-сервиса: 8781 (PUB), 8782 (REP)
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

## Real-time обработка сигналов

### Потоковая обработка в веб-интерфейсе
Веб-интерфейс поддерживает real-time обработку SDR данных:
- Использует отдельный поток для чтения SDR
- Детекция импульсов по RMS порогу
- Автоматическая PSK демодуляция при обнаружении импульса
- Декодирование EPIRB сообщений в реальном времени

### Оптимизация производительности
- Децимация визуальных данных для веб-отображения
- Кольцевые буферы для истории RMS/времени
- Очередь импульсов для асинхронной обработки