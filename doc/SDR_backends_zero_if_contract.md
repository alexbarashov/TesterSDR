# SDR Backends — Zero‑IF Contract (унифицировано)

**Правило:** из `backends` всегда выходим в **0 Hz baseband**. Любые IF‑смещения (аппаратные/цифровые) компенсируем **внутри backends**.

---

## 1) Единый контракт
- Выход `backends.read()/stream`: **несущая = 0 Hz** (DC ≈ 0).
- Внешние модули (демодуляторы, UI, метрики) **не учитывают IF**.
- Метаданные: `sample_rate`, `rf_center_hz`, `effective_if_hw_hz`, `effective_bb_shift_hz`.

## 2) Входные параметры `set_config(...)`
- `rf_center_hz` — RF‑центр наблюдения.
- `if_offset_hz` — желаемый аппаратный IF (может быть 0).
- Прочее: `sample_rate`, `gain`, `bandwidth`, `device`, `backend`.

## 3) Обязанности backends (унифицировано)
1. Настроить железо с учётом `rf_center_hz` и (если поддерживается) аппаратного `if_offset_hz`.
2. Рассчитать **цифровой BB‑сдвиг** так, чтобы `effective_if_hw_hz + effective_bb_shift_hz = 0` (Zero‑IF на выходе).
3. Для `file`: **жёстко** `if_offset_hz = 0`, BB‑сдвиг **выключен**.
4. Вернуть метаданные и залогировать итоговые значения.

### Псевдокод
```python
def set_config(cfg):
    hw_if = apply_hw_if_offset(cfg.if_offset_hz)     # может квантизоваться девайсом
    bb = -hw_if if hw_if else 0
    if cfg.backend == 'file':
        hw_if, bb = 0, 0
    state.bb_shift_enable = (bb != 0)
    state.bb_shift_hz = bb
    return {
        'effective_if_hw_hz': hw_if,
        'effective_bb_shift_hz': bb,
        'note': 'zero‑IF at output'
    }
```

## 4) Примеры настроек (все SDR одинаково)
- **file**: `if_offset_hz=0` (смещения **запрещены**), BB‑сдвиг **off**.
- **soapy_* (RTL/HackRF/Airspy/SDRplay) / rsa306**: `if_offset_hz` можно задавать для борьбы с DC/перегрузкой; backends **обязан** выставить BB‑сдвиг `= −effective_if_hw_hz`.

## 5) Диагностика/чек‑лист
- Лог при старте: `rf_center_hz`, `if_offset_hz`, `effective_if_hw_hz`, `effective_bb_shift_hz`, `sum = 0?`.
- Юнит‑тест Zero‑IF: подать тон на `rf_center_hz + Δ`; измеренная CFO на выходе должна быть `≈ Δ`.
- Режим `file`: убедиться, что BB‑сдвиг **off**, IF **0**.

## 6) TL;DR
- Всегда Zero‑IF на выходе.
- IF на железе — опционален; **обязательная цифровая компенсация внутри backends**.
- UI/демодуляторы работают в нулевой несущей, без знаний про IF.
