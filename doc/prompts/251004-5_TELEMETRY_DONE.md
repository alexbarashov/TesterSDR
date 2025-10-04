# Реализация телеметрии времени для beacon406_PSK_FM-plot.py

**Дата:** 2025-10-04
**Задача:** Добавление системы тайминг-логов (телеметрии времени) в beacon406_PSK_FM-plot.py

## ТЗ

Согласно `doc/prompts/TT_TELEMETRY.txt`:
- Замерить задержки «импульс → график» на стороне FM-plot
- Не влиять на алгоритмы и UI
- Телеметрия выключена по умолчанию (TELEMETRY_ENABLED = False)
- Включение через переменную окружения BEACON_TELEMETRY=1
- Уровни: basic (только файл) / verbose (файл + консоль)

## Реализация

### 1. Блок инициализации (строки 34-108)

**Добавлено:**
- Импорт `os`, `RotatingFileHandler`
- Константа `TELEMETRY_ENABLED = False`
- Чтение окружения: `_telemetry_on`, `_telemetry_level`
- Инициализация логгера телеметрии (только если включена)
- Создание папки `logs/` автоматически
- RotatingFileHandler (2 МБ × 5 бэкапов)
- StreamHandler для verbose режима

### 2. Класс Telemetry (строки 75-105)

**Методы:**
- `__init__(logger, enabled)` - инициализация с пустой лямбдой если выключена
- `now_ms()` - текущее время в мс от старта процесса (perf_counter_ns)
- `emit(tag, pulse_id, note)` - запись метки с автоматическим расчётом dt_ms

**Поля контекста:**
- `_sr` - частота дискретизации (sample rate)
- `_chunk` - размер блока чтения
- `_backend` - имя бэкенда (file/soapy_rtl/...)
- `_mode` - режим (FILE/SDR)

### 3. Инициализация контекста в __init__ (строки 327-334)

```python
self.pulse_id_counter = 0  # счётчик pulse_id

tm._sr = int(self.sample_rate)
tm._chunk = READ_CHUNK
tm._backend = BACKEND_NAME
tm._mode = "FILE" if BACKEND_NAME == "file" else "SDR"
```

### 4. Метки в _reader_loop (строки 1328-1330)

```python
tm.emit("READ_START", -1)
samples = self._read_block(READ_CHUNK)
tm.emit("READ_END", -1, note=f"n={len(samples)}")
```

### 5. Метка PROC_START в _process_samples (строка 1370)

```python
def _process_samples(self, samples: np.ndarray):
    tm.emit("PROC_START", -1)
    ...
```

### 6. Метка THRESH_FIRST при обнаружении импульса (строки 1424-1428)

```python
start_abs = int(idx_end[start_pos[s_idx]])
# Новый импульс: генерируем pulse_id и метку THRESH_FIRST
self.pulse_id_counter += 1
current_pulse_id = self.pulse_id_counter
tm.emit("THRESH_FIRST", current_pulse_id)
```

### 7. Передача pulse_id в pulse_data (строка 1474)

```python
pulse_data = {
    ...
    "pulse_id": current_pulse_id  # для телеметрии
}
```

### 8. Метка ENQ при постановке в очередь (строки 1683-1684)

```python
tm.emit("ENQ", pulse_data.get("pulse_id", -1))
self.pulse_data_queue.put(pulse_data)
```

### 9. Метки UI_TICK и UI_DRAWN в _update_all_plots_combined

**UI_TICK при извлечении из очереди (строка 1753):**
```python
pulse_data = self.pulse_data_queue.get_nowait()
if pulse_data:
    tm.emit("UI_TICK", pulse_data.get("pulse_id", -1))
```

**UI_DRAWN после отрисовки (строка 1805):**
```python
self.fig2.canvas.draw_idle()
tm.emit("UI_DRAWN", pulse_data.get("pulse_id", -1))
```

## Формат лога

```
HH:MM:SS [TM] tag=<TAG> t_ms=<время_от_старта> dt_ms=<время_от_предыдущей_метки> pulse_id=<ID> sr_sps=<Fs> read_chunk=<N> backend=<name> mode=<FILE|SDR> note="<опц>"
```

**Метки:**
1. READ_START - начало чтения блока
2. READ_END - конец чтения (note: n=количество)
3. PROC_START - начало обработки
4. THRESH_FIRST - обнаружение импульса (первый порог)
5. ENQ - постановка в очередь UI
6. UI_TICK - извлечение из очереди
7. UI_DRAWN - завершение отрисовки

## Метрики

Для каждого импульса (pulse_id > 0):

**Latency чтения:**
```
READ_END.t_ms - READ_START.t_ms
```

**Latency детекция → очередь:**
```
ENQ.t_ms - THRESH_FIRST.t_ms
```

**Latency очередь → отрисовка:**
```
UI_DRAWN.t_ms - ENQ.t_ms
```

**Сквозная latency:**
```
UI_DRAWN.t_ms - THRESH_FIRST.t_ms
```

## Файлы

### Изменённые:
1. `beacon406/beacon406_PSK_FM-plot.py` - добавлена телеметрия

### Созданные:
1. `doc/README_Telemetry.md` - документация по использованию
2. `doc/prompts/251004-5_TELEMETRY_DONE.md` - этот файл

## Тестирование

### Включение

**Windows (CMD):**
```cmd
set BEACON_TELEMETRY=1
python beacon406/beacon406_PSK_FM-plot.py
```

**Verbose режим:**
```cmd
set BEACON_TELEMETRY=1
set BEACON_TELEM_LEVEL=verbose
python beacon406/beacon406_PSK_FM-plot.py
```

### Проверка

После обработки импульса проверить `logs/telemetry.log`:
```
grep "pulse_id=1 " logs/telemetry.log
```

Должны быть все 7 меток для каждого pulse_id.

## Соответствие ТЗ

✅ **Цели:**
- Замеряет задержки «импульс → график»
- Не влияет на алгоритмы (только emit() метки)
- Не меняет UI и частоту отрисовки

✅ **Негативные цели:**
- Не меняет параметры детектора/фильтров
- Не добавляет sleep/задержек
- Не меняет формат данных для графиков

✅ **Включение/выключение:**
- TELEMETRY_ENABLED = False (дефолт)
- BEACON_TELEMETRY=1 включает
- BEACON_TELEM_LEVEL=basic|verbose

✅ **Метки:**
- Все 7 меток реализованы
- pulse_id стабилен в рамках сессии
- Формат строки соответствует ТЗ

✅ **Организация кода:**
- Класс Telemetry с методом emit()
- Минимально-инвазивные вставки
- Папка logs/ создаётся автоматически

✅ **Производительность:**
- Выключено: нулевые накладные расходы (пустая лямбда)
- Включено: O(десятков µs) на метку

✅ **Что сдаём:**
- beacon406_PSK_FM-plot.py с телеметрией
- README_Telemetry.md с инструкциями
- Без изменений алгоритмов/UI

## Статус

✅ **Реализация завершена для всех компонентов**
- beacon406_PSK_FM_T-plot.py - standalone версия с телеметрией
- beacon_dsp2plot_T.py - UI клиент DSP сервиса с телеметрией (добавлено 2025-10-04)
- beacon_dsp_service_T.py - DSP сервис с телеметрией (добавлено 2025-10-04)
- Документация обновлена (README_Telemetry.md, 251004-5_TELEMETRY_DONE.md)
- Готово к тестированию пользователем

## Структура файлов

После выполнения задачи создана отдельная версия с телеметрией:

- **beacon406/beacon406_PSK_FM-plot.py** - оригинальная версия БЕЗ телеметрии (восстановлена из git)
- **beacon406/beacon406_PSK_FM_T-plot.py** - версия С телеметрией (новый файл)

### Использование версии с телеметрией

```cmd
# Запуск версии с телеметрией (выключена по умолчанию)
python beacon406/beacon406_PSK_FM_T-plot.py

# Включить телеметрию
set BEACON_TELEMETRY=1
python beacon406/beacon406_PSK_FM_T-plot.py

# Verbose режим
set BEACON_TELEMETRY=1
set BEACON_TELEM_LEVEL=verbose
python beacon406/beacon406_PSK_FM_T-plot.py
```

Логи: `logs/telemetry.log`

### Использование обычной версии

```cmd
# Запуск обычной версии (без телеметрии вообще)
python beacon406/beacon406_PSK_FM-plot.py
```

Это позволяет:
- Использовать стабильную версию без телеметрии для работы
- Включать версию с телеметрией только для измерений/отладки
- Не загромождать основной код телеметрией

## Обновление: упрощение телеметрии (2025-10-04 13:15)

### Проблема
Слишком много бесполезных логов от READ_START/READ_END/PROC_START при пустых блоках чтения.

### Решение
Удалены метки, не связанные с импульсами:
- ❌ READ_START - удалена
- ❌ READ_END - удалена  
- ❌ PROC_START - удалена
- ✅ THRESH_FIRST - оставлена (обнаружение импульса)
- ✅ ENQ - оставлена (постановка в очередь)
- ✅ UI_TICK - оставлена (извлечение из очереди)
- ✅ UI_DRAWN - оставлена (завершение отрисовки)

### Результат
Теперь логируются только события, связанные с обработкой импульсов. 
При отсутствии сигнала логи не пишутся.

### Метрики
Доступные метрики после упрощения:
1. **Обработка импульса**: `ENQ.t_ms - THRESH_FIRST.t_ms`
2. **UI отрисовка**: `UI_DRAWN.t_ms - ENQ.t_ms`
3. **Сквозная latency**: `UI_DRAWN.t_ms - THRESH_FIRST.t_ms`

## Телеметрия для beacon_dsp2plot_T.py (2025-10-04)

### Архитектура
beacon_dsp2plot — это UI клиент для DSP сервиса (beacon_dsp_service.py), работающий через ZeroMQ:
- Получает pulse события через PUB/SUB (порт 8781)
- Запрашивает данные через REQ/REP (порт 8782)
- Отрисовывает графики Phase/FM/RMS в Matplotlib

### Реализация

**1. Инициализация (строки 47-113)**
- Отдельный логгер `telemetry_dsp2plot`
- Лог файл `logs/telemetry_dsp2plot.log`
- Идентичная структура Telemetry класса

**2. Поле pulse_id (строка 320)**
```python
self._current_pulse_id: int = -1  # Текущий pulse_id для телеметрии
```

**3. Метка RECV при получении события (строки 471-477)**
```python
def _on_pulse_event(self, obj: Dict[str, Any]):
    if typ == "pulse":
        self._pulse_event_counter += 1
        current_pulse_id = self._pulse_event_counter
        tm.emit("RECV", current_pulse_id)
        self._current_pulse_id = current_pulse_id
```

**4. Метка UI_TICK при запросе данных (строка 642)**
```python
def _update_pulse_plots(self):
    ...
    tm.emit("UI_TICK", self._current_pulse_id)
    rep = self.client.get_last_pulse(...)
```

**5. Метка UI_DRAWN после отрисовки (строка 728)**
```python
if need_draw:
    self.fig2.canvas.draw_idle()
    tm.emit("UI_DRAWN", self._current_pulse_id)
```

### Метки для beacon_dsp2plot_T

1. **RECV** - получение pulse события от DSP сервиса (ZeroMQ SUB поток)
2. **UI_TICK** - начало UI обработки (запрос get_last_pulse через REP)
3. **UI_DRAWN** - завершение отрисовки графиков

### Метрики для beacon_dsp2plot_T

1. **ZeroMQ latency + poll**: `UI_TICK.t_ms - RECV.t_ms` (задержка от получения события до обработки)
2. **REQ/REP + отрисовка**: `UI_DRAWN.t_ms - UI_TICK.t_ms` (запрос данных и отрисовка)
3. **Сквозная UI latency**: `UI_DRAWN.t_ms - RECV.t_ms` (от события до графика)

**Примечание**: Для полной сквозной latency (от детекции импульса в DSP до графика в UI) нужно:
- Добавить телеметрию в beacon_dsp_service.py (опционально)
- Передавать timestamp детекции через pulse событие
- Синхронизировать часы между DSP и UI (или использовать единый процесс)

## Телеметрия для beacon_dsp_service_T.py (2025-10-04)

### Архитектура
beacon_dsp_service — это headless DSP-движок без GUI и Flask:
- Обработка SDR данных (детекция импульсов, PSK/FM демодуляция)
- Публикация событий через ZeroMQ PUB (порт 8781)
- Приём команд через ZeroMQ REP (порт 8782)

### Реализация

**1. Инициализация (строки 56-132)**
- Отдельный логгер `telemetry_dsp_service`
- Лог файл `logs/telemetry_dsp_service.log`
- Класс Telemetry с контекстными полями (sr_sps, read_chunk, backend, mode)

**2. Поля pulse_id (строки 629-630)**
```python
self.pulse_id_counter: int = 0  # Счётчик pulse_id для телеметрии
self.current_pulse_id: int = -1  # ID текущего обрабатываемого импульса
```

**3. Инициализация контекста (строки 758-766)**
```python
tm._sr = int(self.sample_rate)
tm._chunk = READ_CHUNK
if self.backend:
    tm._backend = getattr(self.backend, 'backend_name', str(getattr(self.backend, 'driver', 'unknown')))
else:
    tm._backend = "none"
tm._mode = "FILE" if (self.backend_name and "file" in self.backend_name.lower()) else "SDR"
```

**4. Метка THRESH_FIRST при обнаружении импульса (строки 1077-1079)**
```python
self.pulse_id_counter += 1
self.current_pulse_id = self.pulse_id_counter
tm.emit("THRESH_FIRST", self.current_pulse_id)
```

**5. Метка PUB перед публикацией события (строка 1307)**
```python
tm.emit("PUB", self.current_pulse_id)
self._emit("pulse", pulse_event_data)
```

### Метки для beacon_dsp_service_T

1. **THRESH_FIRST** - обнаружение импульса (первое пересечение порога)
2. **PUB** - публикация pulse события через ZeroMQ PUB

### Метрики для beacon_dsp_service_T

1. **DSP обработка**: `PUB.t_ms - THRESH_FIRST.t_ms` (время PSK/FM демодуляции + подготовка данных)

### Сквозная latency (DSP сервис + UI клиент)

Для измерения полной latency от детекции до отображения:

1. **DSP**: `PUB.t_ms - THRESH_FIRST.t_ms` (из telemetry_dsp_service.log)
2. **ZeroMQ**: `RECV.t_ms(UI) - PUB.t_ms(DSP)` (требует синхронизации часов или single-host)
3. **UI**: `UI_DRAWN.t_ms - RECV.t_ms` (из telemetry_dsp2plot.log)

**Полная latency** = DSP + ZeroMQ + UI

**Запуск с телеметрией:**
```cmd
set BEACON_TELEMETRY=1
python beacon406/beacon_dsp_service_T.py --pub tcp://127.0.0.1:8781 --rep tcp://127.0.0.1:8782
```
