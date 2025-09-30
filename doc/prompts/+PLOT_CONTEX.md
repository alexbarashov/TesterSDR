**ТЗ dsp_service + plot UI — AUTO/FILE, стабильный REQ/REP** 
  
# Цели

1. Добавить режимы backend: `"auto"`, `"soapy_rtl"`, `"soapy_hackrf"`, `"soapy_airspy"`, `"soapy_sdrplay"`, `"rsa306"`, `"file"`.
2. **AUTO**: если подключен RTL → выбрать `"soapy_rtl"`; иначе HackRF → Airspy → SDRplay → RSA; если SDR нет, но задан файл → `"file"`; иначе понятная ошибка.
3. **FILE**: источники `.sigmf-meta`/`.cf32`; в FILE **всегда** `BB_SHIFT_ENABLE = False`.
4. Починить REQ/REP в **plot-клиенте**: автопересоздание REQ при EFSM/timeout + ретрай; безопасный первичный `get_status`. (В веб-клиенте уже есть ретраи, **его не трогаем**.)  
5. **STRICT_COMPAT**: не ломаем старые контракты и старый UI/web.

---

# Файлы и объём правок

### 1) `beacon_dsp2plot.py`  — UI/клиент ZMQ

* Заменить реализацию REQ-клиента:

  * Таймауты `RCVTIMEO/SNDTIMEO ≈ 3000 ms`, `LINGER=0`.
  * `_cmd(...)`: при `EFSM/EAGAIN/timeout` **пересоздать** REQ и повторить один раз; вернуть понятную ошибку, если оба захода неудачны.
  * Убрать “ломающий” ранний `get_status()` или читать его **без** предположений о структуре (не крашить при пустом ответе). (Сейчас в коде есть ранний статус и короткие таймауты, из-за чего ловится EFSM). 
* Кнопки backend (Auto/RTL/HackRF/Airspy/SDRPlay/RSA/File):

  * Последовательность: `stop_acquire` → `set_sdr_config(...)` → при `ok` → `start_acquire`.
  * Для `"file"` при нажатии «Открыть…» — выбрать путь и передать его в `set_sdr_config`.
* Отображение статуса: не предполагать поля вне присланного (без падений, если `status`/ключи отсутствуют).
* **Без изменения внешнего UI/графиков и горячих клавиш.**

### 2) `beacon_dsp_service.py`  — сервис/движок

* **Новый выбор backend’а до вызова `safe_make_backend(...)`:**

  * Функция `detect_devices()` (через Soapy enumerate/доступные методы) → словарь `{"rtl":bool,"hackrf":...}`.
  * Если `backend_name == "auto"` → выбрать по приоритету; если ничего нет и передан `file_path` → `"file"`; иначе `RuntimeError("No SDR found and no file provided")`.
* **FILE режим:**

  * При выбранном `"file"`:

    * Отключить BB-shift внутри сервиса: `effective_bb_shift_enable=False`, `effective_bb_shift_hz=0`.
    * Загрузка: `.sigmf-meta` (по метаданным взять `sample_rate`, `center_freq`) или `.cf32` (дефолты/переданные параметры).
* **Совместимость с web-клиентом (`beacon_tester_dsp2web.py`):**

  * `get_sdr_config` → **возвращать** `{"ok":true, "config": {...}}` — веб ожидает поле `config`. 
  * `set_sdr_config` → принимать **оба** формата:

    * плоский: `{"cmd":"set_sdr_config", <поля>}` (для plot);
    * web-формат: `{"cmd":"set_sdr_config", "config": {<поля>}}` (для dsp2web). Возвращать `{"ok":true, "applied": {...}, "retuned": <bool>}`. 
  * Остальные команды/события **без изменений имён**: `start_acquire`, `stop_acquire`, `set_params`, `save_sigmf`, `get_status`, PUB-ивенты `status/pulse/psk`. (dsp2web ровно эти роуты вызывает/слушает.) 
* **Событие `status` (PUB):** оставить текущие ключи (например, `sdr`, `fs`, …), можно **добавить** (необязательно):

  * `backend_selected`, `source: "sdr"|"file"`, `file_path` (если file), `bb_shift_hz` (effective). Это улучшит отладку в обоих UI. (Сервис уже шлёт status с ключами по образцу.) 
* **Без правок в `lib/backends.py`**: используем существующий `safe_make_backend(...)`, при необходимости — только выбираем имя/аргументы снаружи. 

### 3) `beacon_tester_dsp2web.py`  — **учитываем**, но **не правим**

* Он шлёт: `{'cmd':'get_sdr_config'}` (ждёт `config`), `{'cmd':'set_sdr_config','config':{...}}`, `start/stop/set_params/save_sigmf/get_status`. Оставляем совместимость и ответы в ожидаемом формате. 

---

# Команды и поля (итоговый контракт)

### set_sdr_config (оба формата входа)

* **Вход (flat, для plot):**

```json
{
  "cmd": "set_sdr_config",
  "backend": "auto|soapy_rtl|soapy_hackrf|soapy_airspy|soapy_sdrplay|rsa306|file",
  "file_path": "C:/path/to/data.sigmf-meta|.cf32",
  "center_freq_hz": 406037000,
  "sample_rate_sps": 1000000,
  "bb_shift_enable": true,
  "bb_shift_hz": -37000,
  "freq_corr_hz": 0,
  "agc": false,
  "gain_db": 30.0
}
```

* **Вход (web):**

```json
{
  "cmd": "set_sdr_config",
  "config": { ... те же поля ... }
}
```

* **Выход (для обоих):**

```json
{
  "ok": true,
  "applied": {
    "backend_selected": "soapy_rtl|...|file",
    "source": "sdr|file",
    "file_path": "..." ,
    "center_freq_hz": 406037000,
    "sample_rate_sps": 1000000,
    "bb_shift_enable": false,   // если file
    "bb_shift_hz": 0            // если file
  },
  "retuned": true
}
```

### get_sdr_config

* **Выход (как ждёт web):**

```json
{
  "ok": true,
  "config": {
    "backend_name": "soapy_rtl|...|file|auto",
    "center_freq_hz": 406000000,
    "sample_rate_sps": 1000000,
    "bb_shift_enable": true,
    "bb_shift_hz": -37000,
    "freq_corr_hz": 0,
    "agc": false,
    "gain_db": 30.0,
    "bias_t": false,
    "antenna": "RX",
    "device": "..."  // строка статуса SDR
  }
}
```

(В plot можно и плоский ответ, но лучше единообразно вернуть именно `config`.)

### status (PUB)

Минимум сохраняем текущие поля, допускается добавить:

```json
{
  "type": "status",
  "sdr": "rtlsdr|hackrf|...|file_wait",
  "fs": 1000000,
  "bb_shift_hz": -37000,
  "target_signal_hz": 406037000,
  "thresh_dbm": -45,
  "queue_depth": 0,

  "backend_selected": "soapy_rtl|...|file",
  "source": "sdr|file",
  "file_path": "..."
}
```

(Веб-клиент и plot читают `status` — ничего не ломаем, добавления безопасны.) 

---

# Приёмочные кейсы

1. **AUTO + RTL подключён**
   В plot жмём **Auto** → `set_sdr_config` выбирает `"soapy_rtl"`, `start_acquire: ok`, `status.sdr =~ "rtlsdr"`. В web `/api/health` показывает активный ZMQ; `/api/sdr/get_config` отдаёт `config`. 

2. **AUTO без SDR + задан файл**
   В plot выбираем Auto, указываем файл → выбран `"file"`, `bb_shift_enable=false`, `bb_shift_hz=0`. Поток и события идут.

3. **FILE напрямую**
   В plot нажимаем **File**, открываем `.sigmf-meta` или `.cf32` → `status.source=file`, поле `file_path` отображается.

4. **Нет SDR и файла**
   Auto → `ok:false, error:"No SDR found and no file provided"`.

5. **Стабильность REQ**
   Холодный старт (сервис просыпается дольше): plot-клиент пересоздаёт REQ и повторяет запрос, без `EFSM`. (web-клиент уже имеет ретраи/таймауты, и мы их не меняем.)  

6. **STRICT_COMPAT**
   Старые вызовы без `backend` продолжают работать (используем текущий выбранный). Форматы ответов не ломаются; web-роуты/ожидания — без изменений. 


