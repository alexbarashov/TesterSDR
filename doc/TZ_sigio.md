
# ТЗ: модуль файлового I/O для IQ с поддержкой SigMF
**Имя файла (рабочее):** `lib/sigio.py`  
**Назначение:** Единая библиотека чтения/записи IQ-данных для проекта. Поддерживает **SigMF** (через `sigmf-python`) и «legacy» **.cf32**. На «верх» (backends/metrics/demod/GUI) всегда отдаёт **поток `np.complex64`** (interleaved I/Q).

---

## 1) Область применения и цели
- Прозрачно подменять источник данных: **SDR-live** ↔ **файл** (SigMF или cf32) без изменений в остальном коде.
- Сохранять **RAW** поток «как с SDR» в **SigMF** без раздувания (родная разрядность: `ci8/ci16/...`) + по флагу писать **чекпоинты** обработки в `cf32` (отдельными SigMF файлами).
- Минимальная зависимость от ОС/железа; фокус на **Raspberry Pi** (низкая накладная, mmap/батчи, без лишних копий).

## 2) Внешние зависимости
- **Обязательно:** `numpy` (>=1.20), `sigmf` (официальный пакет `sigmf-python`).
- **Опционально:** `mmap` (stdlib).

## 3) Режимы работы модуля

### 3.1 Чтение (Reader)
- **Вход:** путь к файлу:
  - `*.sigmf-meta` или `*.sigmf-data` → режим **SigMF**.
  - `*.cf32` → режим **legacy cf32**.
- **Определение формата:** по расширению; для SigMF валидируем связку `meta↔data` через `sigmf-python`.
- **Выход для backends:** батчи **`np.complex64`** + статус (см. §5).
- **Тайм-база:** `output_sample_rate_sps` = `global.sample_rate` (из SigMF). Если задан `target_fs`, выполняется **внутренняя децимация**; статус отражает `decim_factor`.
- **Поддерживаемые SigMF `datatype` (минимум):** `cf32_le`, `ci16_le`, `ci8`. При чтении `ci*` → конвертация в `cf32` (масштаб ±1.0).
- **Загрузка данных:**
  - Если `datatype=cf32_le` и нет децимации → **`numpy.memmap`** и батчевый `view` (минимум копий).
  - Если `ci*` или включена децимация → чтение батчами 64–256k компл. сэмплов, конверт/фильтр → `cf32`.
- **Аннотации:** по желанию вызывающего кода — доступ через `list_annotations()`/`iter_annotations()`; если включена децимация, индексы **рескейлятся**.

### 3.2 Запись (Writer)
- **RAW SigMF:** принимает массивы **родной разрядности с SDR** (ci8/ci16/… — предоставляемые backend’ом до конвертации) и пишет `*.sigmf-data` + `*.sigmf-meta` с `datatype=ci*`. Метаданные: `sample_rate`, `frequency`, `datetime`, `hw`, `gains`, `if_offset_hz`, `freq_correction_hz`, `pipeline="raw"`.
- **Чекпоинты SigMF (опционально):** принимает `cf32` (после децимации/CFO и т.п.), пишет SigMF с `datatype="cf32_le"`, `pipeline="post-decim"|"post-cfo"|"pre-demod"`.
- **Legacy .cf32:** отдельный простой writer (как сейчас). **Важно:** **.cf32** не связывается с SigMF; это самостоятельный бинарный формат.

## 4) Публичный API (без кода, сигнатуры)
```
open_iq(path: str, target_fs: float | None = None, strict: bool = True) -> Reader
Reader.read(max_complex: int) -> np.ndarray[np.complex64]        # ≤ max_complex
Reader.get_status() -> dict                                       # см. §5
Reader.stop() -> None
Reader.list_annotations() -> list[dict]                           # опц.
Reader.iter_annotations() -> Iterator[dict]                       # опц.

save_sigmf_raw(path_base: str,
               data_any: np.ndarray,                # ci8/ci16 interleaved I/Q
               sample_rate: float,
               center_freq_hz: float,
               *,
               datatype: Literal["ci8","ci16_le",...],
               hw: str | None = None,
               datetime_utc: str | None = None,
               extras: dict | None = None,
               captures: list[dict] | None = None,
               annotations: list[dict] | None = None) -> tuple[str, str]

save_sigmf_cf32(path_base: str,
                cf32: np.ndarray,                   # complex64 interleaved
                sample_rate: float,
                center_freq_hz: float,
                *,
                pipeline: Literal["post-decim","post-cfo","pre-demod"] | None,
                hw: str | None = None,
                datetime_utc: str | None = None,
                extras: dict | None = None,
                captures: list[dict] | None = None,
                annotations: list[dict] | None = None) -> tuple[str, str]

save_cf32_legacy(path: str, cf32: np.ndarray) -> str
```

## 5) Статус и контракт с backends
`Reader.get_status()`:
- `source`: `"file-sigmf" | "file-cf32"`
- `input_sample_rate_sps`: float (из SigMF или из параметров cf32)
- `output_sample_rate_sps`: float (после внутренней децимации или = input)
- `center_freq_hz`: float (SigMF: из `captures[0].frequency`; cf32: из параметров запуска)
- `decim_factor`: int (≥1)
- `total_samples_in`: int
- `total_samples_out`: int
- `eof`: bool
- `file_path`: str
- `sigmf_meta`: dict | None (минимальный срез)
- (опц.) `extras`: dict (if_offset_hz, freq_correction_hz, gain_profile, cal_db_offset, mode_hint, pipeline)

**Инварианты:** на «верх» всегда подаётся `np.complex64`; аннотации соответствуют `output_sample_rate_sps`.

## 6) Производительность и ограничения
- Raspberry Pi: минимум копий; memmap при cf32 без децимации; батч 64–256k; избегать `complex128`.
- Конверсия `ci* → cf32`: масштаб ±1.0 (`1/128` для int8, `1/32768` для int16).
- Децимация: polyphase FIR/half-band; подавление внеполосных ≥60 dB.
- Модуль не грузит весь файл в RAM.

## 7) Профиль метаданных SigMF (минимум)
**global:** `datatype`, `sample_rate`, `version`, `hw`, `author?`, `description?`,
`extras`: `if_offset_hz?`, `freq_correction_hz?`, `gain_profile?`, `cal_db_offset?`, `pipeline?`, `mode_hint?`  
**captures:** `[ { sample_start, frequency, datetime?, rf_bw_hz?, lna_state?, if_gain_db? } ]`  
**annotations (опц.):** `core:sample_start`, `core:sample_count`, `core:label`, `core:annotation`, `core:generator`, `core:uuid`,
`user:` поля: `baud`, `CFO_Hz`, `PosPhase`, `NegPhase`, `PhRise`, `PhFall`, `Asym`, `SNR_dB`, `bit_indexes[]`, `frame_id`

## 8) Политика ошибок и режим строгости
- strict=True: требуются `datatype`, `sample_rate`, хотя бы один `capture.frequency`; несоответствие размеров data↔meta → ошибка.
- strict=False: дефолты + предупреждения (freq из имени/параметров, неполные annotations пропускаем).
- Меняющийся `sample_rate` внутри одного SigMF — не поддерживается.

## 9) Интеграция
- В `lib.backends` режим **file** использует `sigio.open_iq(...)` вместо прямого чтения.
- API backends «наверх» не меняется (STRICT_COMPAT).
- Live-поток: модуль пишет RAW SigMF (ci8/ci16) + опц. чекпоинты cf32.

## 10) Тесты и приёмка
- SigMF `cf32_le` (короткий) → чтение без децимации, проверка RMS/длины.
- SigMF `ci16_le` → чтение + конвертация, сопоставление PSD.
- `.cf32` → чтение при заданных `Fs`/`Fcenter`.
- Децимация до `target_fs` → проверка длины и отсутствия alias.
- Запись RAW/чекпоинтов → обратное чтение, соответствие.
- Производительность Pi: memmap-путь ≥ 20 MB/s, без двойных копий.

## 11) Именование по умолчанию
- RAW SigMF: `name_YYYYMMDDThhmmssZ.raw.sigmf-{meta,data}`
- Чекпоинты SigMF (`cf32`): `name_...post-decim.sigmf-*`, `name_...pre-demod.sigmf-*`
- Legacy: `name.cf32`

## 12) Нефункциональные требования
- Кроссплатформенность: Windows / Linux (Pi).
- Понятные логи и ошибки.
- Конфигурируемость: `target_fs`, `strict`, децимация, пути сохранения.
