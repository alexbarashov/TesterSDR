# Телеметрия времени (тайминг-логи) для SDR приложений

## Назначение

Система телеметрии позволяет замерить задержки между обнаружением импульса и его отображением на графиках ("импульс → график"). Телеметрия **выключена по умолчанию** и не влияет на производительность или логику обработки сигналов.

**Важно:** Телеметрия доступна только в файлах с суффиксом `_T`:
- `beacon406_PSK_FM_T-plot.py` - standalone приложение с встроенной DSP обработкой
- `beacon_dsp2plot_T.py` - UI клиент для DSP сервиса (ZeroMQ)
- `beacon_dsp_service_T.py` - DSP сервис (headless, ZeroMQ PUB/REP)

Оригинальные файлы (`beacon406_PSK_FM-plot.py`, `beacon_dsp2plot.py`, `beacon_dsp_service.py`) НЕ содержат телеметрии.

## Как включить

### Через переменную окружения (рекомендуется)

**Windows (CMD):**
```cmd
set BEACON_TELEMETRY=1
python beacon406/beacon406_PSK_FM_T-plot.py

# или для UI клиента DSP сервиса
python beacon406/beacon_dsp2plot_T.py
```

**Windows (PowerShell):**
```powershell
$env:BEACON_TELEMETRY="1"
python beacon406/beacon406_PSK_FM_T-plot.py

# или для UI клиента DSP сервиса
python beacon406/beacon_dsp2plot_T.py
```

**Linux/Mac:**
```bash
export BEACON_TELEMETRY=1
python beacon406/beacon406_PSK_FM_T-plot.py

# или для UI клиента DSP сервиса
python beacon406/beacon_dsp2plot_T.py
```

### Уровни детализации

- `BEACON_TELEM_LEVEL=basic` (по умолчанию) - записывает только ключевые метки времени в файл
- `BEACON_TELEM_LEVEL=verbose` - дополнительно выводит телеметрию в консоль

**Пример verbose режима:**
```cmd
set BEACON_TELEMETRY=1
set BEACON_TELEM_LEVEL=verbose
python beacon406/beacon406_PSK_FM_T-plot.py

# или для UI клиента DSP сервиса
python beacon406/beacon_dsp2plot_T.py
```

## Где смотреть логи

Логи телеметрии сохраняются в:
```
TesterSDR/logs/telemetry.log             - для beacon406_PSK_FM_T-plot.py
TesterSDR/logs/telemetry_dsp2plot.log    - для beacon_dsp2plot_T.py
TesterSDR/logs/telemetry_dsp_service.log - для beacon_dsp_service_T.py
```

Папка `logs/` создаётся автоматически при первом запуске с включённой телеметрией.

### Ротация файлов

- Максимальный размер файла: 2 МБ
- Количество бэкапов: 5
- Старые файлы: `telemetry.log.1`, `telemetry.log.2`, ... `telemetry.log.5`

## Формат логов

Каждая строка лога содержит:

```
HH:MM:SS [TM] tag=<TAG> t_ms=<время_от_старта> dt_ms=<время_от_предыдущей_метки> pulse_id=<ID_импульса> [контекстные_поля] note="<заметка>"
```

**Контекстные поля** зависят от приложения:
- `beacon406_PSK_FM_T-plot.py`: sr_sps, read_chunk, backend, mode
- `beacon_dsp2plot_T.py`: может не содержать контекстных полей (зависит от реализации)

### Метки (tags)

Логируются только события, связанные с импульсами.

**Для beacon406_PSK_FM_T-plot.py (standalone):**
1. **THRESH_FIRST** - момент первого пересечения порога (обнаружение импульса)
2. **ENQ** - постановка данных импульса в очередь для UI
3. **UI_TICK** - момент извлечения импульса из очереди UI-потоком
4. **UI_DRAWN** - завершение отрисовки графиков

**Для beacon_dsp2plot_T.py (UI клиент DSP сервиса):**
1. **RECV** - момент получения pulse события от DSP сервиса (ZeroMQ SUB)
2. **UI_TICK** - начало UI обработки (запрос get_last_pulse)
3. **UI_DRAWN** - завершение отрисовки графиков

**Для beacon_dsp_service_T.py (DSP сервис):**
1. **THRESH_FIRST** - момент первого пересечения порога (обнаружение импульса)
2. **PUB** - публикация pulse события через ZeroMQ PUB

### Пример лога

```
14:23:45 [TM] tag=THRESH_FIRST t_ms=1237.891 dt_ms=0.000 pulse_id=1 sr_sps=1000000 read_chunk=65536 backend=file mode=FILE
14:23:46 [TM] tag=ENQ t_ms=1789.456 dt_ms=551.565 pulse_id=1 sr_sps=1000000 read_chunk=65536 backend=file mode=FILE
14:23:46 [TM] tag=UI_TICK t_ms=1790.234 dt_ms=0.778 pulse_id=1 sr_sps=1000000 read_chunk=65536 backend=file mode=FILE
14:23:46 [TM] tag=UI_DRAWN t_ms=1798.567 dt_ms=8.333 pulse_id=1 sr_sps=1000000 read_chunk=65536 backend=file mode=FILE
```

## Анализ логов

### Ключевые метрики

**Для beacon406_PSK_FM_T-plot.py (standalone):**

1. **Latency обработка импульса (детекция → очередь):**
   ```
   ENQ.t_ms - THRESH_FIRST.t_ms
   ```
   Показывает время обработки импульса (демодуляция PSK, FM, расчёт метрик).

2. **Latency очередь → отрисовка:**
   ```
   UI_DRAWN.t_ms - ENQ.t_ms
   ```
   Показывает время доставки данных в UI и отрисовки графиков.

3. **Сквозная latency (обнаружение → график):**
   ```
   UI_DRAWN.t_ms - THRESH_FIRST.t_ms
   ```
   Полное время от обнаружения импульса до появления на графиках.

**Для beacon_dsp2plot_T.py (UI клиент):**

1. **Latency ZeroMQ доставка + обработка:**
   ```
   UI_TICK.t_ms - RECV.t_ms
   ```
   Показывает задержку от получения события до начала UI обработки (poll интервал + очередь).

2. **Latency запрос данных + отрисовка:**
   ```
   UI_DRAWN.t_ms - UI_TICK.t_ms
   ```
   Показывает время запроса get_last_pulse через ZeroMQ REP и отрисовки графиков.

3. **Сквозная latency UI (получение → график):**
   ```
   UI_DRAWN.t_ms - RECV.t_ms
   ```
   Полное время от получения pulse события до отображения графиков.

**Для beacon_dsp_service_T.py (DSP сервис):**

1. **Latency обработка импульса (детекция → публикация):**
   ```
   PUB.t_ms - THRESH_FIRST.t_ms
   ```
   Показывает время обработки импульса в DSP сервисе (PSK демодуляция, FM, метрики, подготовка данных).

**Сквозная latency DSP сервис + UI клиент:**

Для измерения полной сквозной latency от детекции до отображения на экране нужно коррелировать логи из разных файлов по pulse_id:

1. **DSP обработка**: `PUB.t_ms - THRESH_FIRST.t_ms` (из telemetry_dsp_service.log)
2. **ZeroMQ latency**: `RECV.t_ms(UI) - PUB.t_ms(DSP)` (требует синхронизации часов)
3. **UI latency**: `UI_DRAWN.t_ms - RECV.t_ms` (из telemetry_dsp2plot.log)

**Полная latency** = DSP обработка + ZeroMQ latency + UI latency

**Примечание:** Если DSP сервис и UI клиент запущены на одной машине, часы синхронизированы автоматически (используется `time.perf_counter_ns()` от старта каждого процесса).

### Анализ с grep/awk (Linux/Mac)

**Найти все импульсы:**
```bash
grep "THRESH_FIRST" logs/telemetry.log
```

**Извлечь pulse_id и время:**
```bash
awk '/THRESH_FIRST/ {print $4, $5}' logs/telemetry.log
```

**Показать только метки одного импульса (pulse_id=1):**
```bash
grep "pulse_id=1 " logs/telemetry.log
```

### Анализ в Windows (PowerShell)

**Найти все импульсы:**
```powershell
Select-String -Path logs\telemetry.log -Pattern "THRESH_FIRST"
```

**Извлечь данные импульса:**
```powershell
Select-String -Path logs\telemetry.log -Pattern "pulse_id=1 "
```

## Тестирование

### FILE-режим

1. Выберите тестовый CF32 файл с несколькими импульсами
2. Настройте config.py:
   ```python
   BACKEND_NAME = "file"
   BACKEND_ARGS = {"path": r"C:/work/TesterSDR/captures/test_signal.cf32"}
   ```
3. Запустите с телеметрией:
   ```cmd
   set BEACON_TELEMETRY=1
   python beacon406/beacon406_PSK_FM_T-plot.py
   ```
4. Проверьте `logs/telemetry.log` - должны быть все метки для каждого импульса

### RTL-SDR режим

1. Подключите RTL-SDR
2. Настройте config.py:
   ```python
   BACKEND_NAME = "auto"  # или "soapy_rtl"
   ```
3. Запустите с телеметрией (verbose для отладки):
   ```cmd
   set BEACON_TELEMETRY=1
   set BEACON_TELEM_LEVEL=verbose
   python beacon406/beacon406_PSK_FM_T-plot.py
   ```
4. Ожидайте реальный сигнал (406 МГц маяк)
5. Проверьте логи - сквозная latency будет выше из-за I/O SDR

## Выключение телеметрии

Просто запустите без переменной окружения:
```cmd
python beacon406/beacon406_PSK_FM_T-plot.py
```

Или явно выключите:
```cmd
set BEACON_TELEMETRY=0
python beacon406/beacon406_PSK_FM_T-plot.py
```

Или используйте обычную версию без телеметрии:
```cmd
python beacon406/beacon406_PSK_FM-plot.py
```

При выключенной телеметрии:
- Накладные расходы = 0 (пустая лямбда)
- Логи не пишутся
- Поведение приложения идентично версии без телеметрии

## Ограничения

- Телеметрия не логирует сырые IQ-данные
- В verbose режиме возможны краткие сводки (длина окна, min/max)
- Время замеряется с точностью `time.perf_counter_ns()` (~наносекунды)
- pulse_id стабилен только в рамках одной сессии (сбрасывается при перезапуске)

## Примеры использования

### Поиск узких мест

1. Запустите с телеметрией и соберите логи
2. Проанализируйте dt_ms для каждой пары меток одного импульса
3. Найдите стадию с максимальным dt_ms - это узкое место

**Пример:**
```
THRESH_FIRST → ENQ: dt_ms=450.0  <- Долгая обработка PSK
ENQ → UI_TICK: dt_ms=0.5
UI_TICK → UI_DRAWN: dt_ms=12.0
```
Узкое место: обработка импульса (450 мс).

### Сравнение FILE vs SDR режимов

Запустите тесты в обоих режимах и сравните:
- READ_END - READ_START (ожидается больше в SDR)
- THRESH_FIRST → UI_DRAWN (должна быть схожей)

## Дополнительная информация

См. `doc/prompts/TT_TELEMETRY.txt` для полной спецификации ТЗ.
