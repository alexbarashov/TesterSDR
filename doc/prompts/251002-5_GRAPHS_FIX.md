# 251002-5: Исправление отображения графиков в beacon_dsp2plot

**Дата:** 2025-10-02
**Статус:** ✅ ЗАВЕРШЕНО

## Проблема

Графики фазы, FM и RMS не отображались во втором окне `beacon_dsp2plot.py` при работе с DSP сервисом через ZeroMQ. Из 10 запусков графики появлялись только 2 раза.

### Симптомы
- Первое окно (Sliding RMS) работало корректно
- Второе окно (Pulse Analysis: RMS + Фаза + FM) оставалось пустым
- Параметры и сообщения декодировались правильно
- Иногда графики появлялись после нескольких нажатий кнопки "Параметры"

## Диагностика

### Инструменты
1. Создан тестовый скрипт `beacon406/test_get_pulse.py` для прямого тестирования DSP API
2. Добавлен `beacon406/start_ui_debug.bat` для запуска UI с консольным выводом
3. Добавлены диагностические print'ы в код UI

### Выявленные проблемы

#### 1. Таймаут ZeroMQ слишком короткий
```
[cmd=get_last_pulse] TIMEOUT after 2003.2 ms
```
- UI: установлен таймаут **2000 мс**
- Реальное время обработки: **~3000 мс** (2800-3200 мс)
- DSP сервис обрабатывает ~570k сэмплов IQ + вычисляет фазу/FM/RMS

Тест показал:
```
Ответ получен за 2779.5 мс
phase: x.len=142500, y.len=142500
fm: x.len=142500, y.len=142500
rms: x.len=569001, y.len=569001
```

#### 2. Графики не запрашивались в файловом режиме
- `get_last_pulse` вызывался только при получении pulse события из PUB канала
- В файловом режиме после загрузки файла pulse событие может не прийти сразу
- Графики оставались пустыми, даже если данные были доступны

#### 3. Фиксированный ylim для графика фазы
```python
self.ax_phase.set_ylim(-1.5, +1.5)  # Было
```
- Реальные данные: `phase.y: min=-49.336, max=44.190` (большой наклон!)
- График рисовался, но все данные были за пределами видимой области
- Это отдельная проблема, связанная с наклоном фазы (см. ниже)

## Решение

### Изменения в `beacon406/beacon_dsp2plot.py`

#### 1. Увеличен таймаут ZeroMQ (строка 95)
```python
# Было:
self.req.setsockopt(zmq.RCVTIMEO, 2000)  # 2s для ACK

# Стало:
self.req.setsockopt(zmq.RCVTIMEO, 5000)  # 5s для get_last_pulse
```

#### 2. Добавлена принудительная загрузка в файловом режиме (строки 253-254, 547-564)
```python
# Новые переменные состояния:
self._last_pulse_request_time: float = 0.0
self._pulse_request_interval_s: float = 1.0

# Логика в _update_pulse_plots():
force_request = False
if "file" in backend.lower() and acq_state == "running":
    if self._phase_x.size == 0:  # графики еще не загружены
        force_request = True

# Ограничение частоты запросов (не чаще 1 раз в секунду)
now = time.time()
if now - self._last_pulse_request_time < self._pulse_request_interval_s:
    return
```

#### 3. Автомасштабирование графика фазы (строки 621-628)
```python
# Было:
self.ax_phase.set_ylim(-1.5, +1.5)

# Стало:
ymin, ymax = np.nanmin(py), np.nanmax(py)
if np.isfinite(ymin) and np.isfinite(ymax):
    y_range = ymax - ymin
    margin = max(0.2, y_range * 0.1)
    self.ax_phase.set_ylim(ymin - margin, ymax + margin)
```

#### 4. Удалены диагностические print'ы
Оставлено только логирование ошибок

## Результат

✅ **Графики отображаются стабильно** при каждом запуске
✅ **Задержка загрузки** 2-3 секунды (нормально для обработки ~570k сэмплов)
✅ **Автомасштабирование** работает корректно
✅ **Коммит создан:** `6689bc6`
✅ **Отправлено на GitHub**

### Тестирование
```bash
# Запуск DSP сервиса
python beacon406/beacon_dsp_service.py

# Запуск UI с консольным выводом (для отладки)
beacon406/start_ui_debug.bat

# Или стандартный запуск
python beacon406/beacon_dsp2plot.py

# Тест API напрямую
python beacon406/test_get_pulse.py  # (не в git, только для диагностики)
```

## Открытые вопросы

### 1. ⚠️ Большой наклон графика фазы

**Проблема:**
График фазы имеет диапазон от -49 до +44 радиан (почти 30π!), что указывает на частотное расстройство или неправильный базовый уровень.

**Возможные причины:**
- `PSK_BASELINE_MS = 2.0` мс — слишком короткая несущая для вычисления базового уровня
- Частотное расстройство (IF offset)
- В начале импульса идет несущая, которая дает наклон

**Решения для тестирования:**

#### Вариант A: Увеличить PSK_BASELINE_MS
В `beacon406/beacon_dsp_service.py` строка 79:
```python
# Текущее:
PSK_BASELINE_MS = 2.0

# Попробовать:
PSK_BASELINE_MS = 10.0  # как в test_cf32_to_phase_msg_FFT.py
```

#### Вариант B: Обрезать начало импульса
В `beacon406/beacon_dsp_service.py` в двух местах (строки 1024 и 1373):

**Место 1:** При детекции импульса (после строки 1023):
```python
# Добавить перед process_psk_impulse:
START_DELAY_MS = 3.0  # пропускаем начало несущей
skip_samples = int(START_DELAY_MS * 1e-3 * self.sample_rate)
psk_seg = freq_seg[skip_samples:] if freq_seg.size > skip_samples else freq_seg

res = process_psk_impulse(
    iq_seg=psk_seg,  # вместо freq_seg
    fs=self.sample_rate,
    baseline_ms=PSK_BASELINE_MS,
    t0_offset_ms=START_DELAY_MS,  # вместо 0.0
    use_lpf_decim=True,
    remove_slope=True,
)
```

**Место 2:** В get_last_pulse (после строки 1370):
```python
# Если запрашивается полный срез (i0=0), применяем ту же обрезку
START_DELAY_MS = 3.0
phase_slice = iq_slice
if i0 == 0:
    skip_samples = int(START_DELAY_MS * 1e-3 * fs)
    if iq_slice.size > skip_samples:
        phase_slice = iq_slice[skip_samples:]
        t0_offset_for_phase = START_DELAY_MS

res_phase = process_psk_impulse(
    iq_seg=phase_slice,  # вместо iq_slice
    ...
)
```

**Примечание:** В файле `test_cf32_to_phase_msg_FFT.py` используется `START_DELAY_MS = 3.0` для обрезки начала и `PSK_BASELINE_MS = 10.0` для базовой фазы — это может быть правильным подходом.

### 2. 🔍 Оптимизация производительности get_last_pulse

**Текущее время:** ~3 секунды на обработку
**Узкие места:**
- Обработка ~570k сэмплов IQ
- Вычисление фазы с LPF/децимацией
- FM демодуляция
- Передача через ZeroMQ

**Возможные оптимизации:**
1. Кэширование результатов (если импульс не изменился)
2. Downsample по умолчанию для UI (сейчас `downsample="decimate"` не работает)
3. Передавать только изменившиеся данные
4. Асинхронная обработка в DSP сервисе

### 3. 📊 Проблема "из 10 запусков 2 раза работает"

**Возможная причина:** Race condition при старте
- Если `get_last_pulse` вызывается до `start_acquire` или до детекции импульса
- Возвращается "No pulse data available"
- Графики не загружаются

**Решение:** Уже реализовано через `force_request` для файлового режима

## План дальнейшей работы

### Высокий приоритет
1. ✅ Исправить отображение графиков (СДЕЛАНО)
2. ⚠️ **Убрать наклон фазы** (следующая задача)
   - Протестировать вариант A (PSK_BASELINE_MS = 10.0)
   - Если не поможет, применить вариант B (обрезка начала)
   - Сравнить с результатами из `test_cf32_to_phase_msg_FFT.py`

### Средний приоритет
3. Оптимизировать get_last_pulse (уменьшить время обработки)
4. Добавить индикатор загрузки в UI (progress bar/spinner)
5. Улучшить обработку ошибок (более информативные сообщения)

### Низкий приоритет
6. Рефакторинг диагностических print'ов (использовать logging)
7. Добавить тесты для ZeroMQ API
8. Документация по архитектуре DSP-сервис + UI

## Файлы изменены

### Коммит 6689bc6
- ✅ `beacon406/beacon_dsp2plot.py` - основные исправления
- ✅ `beacon406/start_ui_debug.bat` - новый файл для отладки

### Не в git (вспомогательные)
- `beacon406/test_get_pulse.py` - тестовый скрипт для диагностики API

## Команды для работы

```bash
# Стандартный запуск (весь стек)
app_dsp2web.bat

# Или раздельно:
# Терминал 1: DSP сервис
python beacon406/beacon_dsp_service.py

# Терминал 2: UI
python beacon406/beacon_dsp2plot.py

# Отладка UI
beacon406/start_ui_debug.bat

# Тест API
python beacon406/test_get_pulse.py
```

## Ссылки

- Коммит: https://github.com/alexbarashov/TesterSDR/commit/6689bc6
- Предыдущие задачи:
  - `doc/prompts/251002-3_CONTEX.md` - Реализация offset/span для DSP API
  - `doc/prompts/251002-4_CONTEX.md` - Исправление графиков в dsp2plot

---

**Следующая задача:** Убрать наклон графика фазы (тестирование вариантов A и B)
