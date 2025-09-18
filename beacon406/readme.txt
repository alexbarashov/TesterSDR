
Коммит-месседж:
plot/backends: file one-shot + safe EOF; SDR underflow no-stop
- FilePlaybackBackend: single-pass playback, inverse IF applied only for file, stop() resets
- Plot: stop only when BACKEND_NAME=="file"; SDR empty blocks are skipped


-----------------------
TesterSDR/
  .vscode/
    settings.json
  .env
  beacon406/
    apps/
    lib/
  capture/


beacon406/               
  __init__.py
  beacon406-plot.py
lib/
  backends.py              ← было sdr_backends.py
  config.py                ← было config_backend.py
  demod.py                 ← (ваш demod_psk_message.py / *_safe)
  metrics.py               ← phase_metrics и processing_PSK .пр. 
  gen.py                   ← было epirb406_lib.py
  io_cf32.py               ← чтение/запись .cf32 (если нужно)
apps/                      ← «входные точки» (скрипты)
  beacon406-gui.py         ← было rtl_*RMS*_PSK*_safe*_ok*.py (GUI)
  beacon406-capture.py     ← захват в .cf32
  beacon406-analyze.py     ← офлайн-анализ .cf32
  beacon406-generate.py    ← генерация тест-сигналов 406 BPSK



> обработка сигнала

● Давайте посмотрим, какие инструменты обработки сигналов есть в проекте и как они работают.

● Read(beacon406\lib\demod.py)
  ⎿  Read 661 lines (ctrl+o to expand)

● Read(beacon406\lib\metrics.py)
  ⎿  Read 149 lines (ctrl+o to expand)

● Read(beacon406\beacon406-plot.py)
  ⎿  Read 100 lines (ctrl+o to expand)

● Обработка сигналов в TesterSDR

  В проекте реализована обработка PSK-сигналов аварийных радиобуев 406 МГц. Основные компоненты:

  1. Демодуляция PSK (lib/demod.py)

  - Детектор фронтов - detect_all_steps_by_mean_fast() находит перепады фазы методом скользящего окна
  - Извлечение данных - extract_half_bits() читает полубиты из фазы
  - Декодирование - halfbits_to_bytes_fast() преобразует полубиты в байты сообщения
  - Безопасный демодулятор - phase_demod_psk_msg_safe() с обработкой ошибок

  2. Фазовые метрики (lib/metrics.py)

  - НЧ-фильтрация - FIR-фильтр 12 кГц для подавления шума
  - Прореживание - децимация в 4 раза после фильтрации
  - Выделение фазы - process_psk_impulse() извлекает непрерывную фазу из IQ
  - Удаление тренда - компенсация частотного сдвига

  3. Анализ импульсов (lib/demod.py)

  - Расчёт параметров - calculate_pulse_params() измеряет:
    - PosPhase/NegPhase - уровни плато
    - PhRise/PhFall - длительности фронтов
    - Ass - асимметрия импульсов
    - Tmod - период модуляции

  Система обрабатывает как реальные сигналы с SDR, так и записанные CF32-файлы.
  
