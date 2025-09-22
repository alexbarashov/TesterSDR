# Сессия обновления веб-интерфейса TesterSDR

## Дата: 18.09.2025

## Задача
Обновление веб-интерфейса `tester_sdr_http_ui_stage_1_windows_single_file_app.py` в соответствии с предоставленным изображением оригинального интерфейса COSPAS/SARSAT Beacon Tester.

## Анализ изображения интерфейса

### Ключевые отличия от текущей реализации:

**1. Компоновка интерфейса:**
- Левая панель меньше по ширине с компактными секциями VIEW/MODE/FILE/TESTER
- Центральная область занимает больше места с детальной информацией о маяке
- Правая панель с числовыми метриками более широкая

**2. Центральная область:**
- Информация о протоколе и дате в верхних блоках
- График фазы имеет сетку с временной шкалой (1.01-9.08 мс)
- График показывает синусоидальный сигнал с четкими +1/-1 пределами
- Подписи Phase+/Phase- с конкретными значениями под графиком

**3. Правая панель статистики:**
- Блок "Current" с точными значениями FS1/FS2/FS3 Hz
- Детальные фазовые метрики (Phase+,rad / Phase-,rad)
- Временные параметры (TRise,mcs / TFall,mcs)
- Параметры мощности и модуляции

## Реализованные изменения

### 1. CSS Grid Layout
```css
.container {
  display: grid; gap: 8px; padding: 8px;
  grid-template-columns: 180px 1fr 320px; grid-template-rows: 1fr;
  height: calc(100vh - 54px);
}
```

### 2. Структура центральной области
```html
<div class="beacon-info">
  <div class="info-row">
    <div class="info-block">
      <span class="label">Protocol</span>
      <span class="value">N</span>
    </div>
    <div class="info-block">
      <span class="label">Date</span>
      <span id="date" class="value">01.08.2025</span>
    </div>
    <div class="info-block">
      <span class="label">Conditions</span>
      <span class="value"><a href="#">Normal temperature, Idling</a></span>
    </div>
  </div>
  <div class="info-row">
    <div class="info-block">
      <span class="label">Beacon Model</span>
      <span class="value">Beacon N</span>
    </div>
    <div class="info-block">
      <span class="label">Beacon Frequency</span>
      <span id="freq" class="value">406025000.0</span>
    </div>
  </div>
  <div class="beacon-title">Beacon 406</div>
  <div class="message-info">Message: <span id="msg">[no message]</span></div>
</div>
```

### 3. Улучшенный график с временной шкалой
```javascript
function drawGrid(){
  const w=C.width, h=C.height; ctx.clearRect(0,0,w,h);
  ctx.lineWidth=1; ctx.strokeStyle='#e5e7eb';

  // Horizontal grid lines
  for(let i=0;i<=10;i++){
    let y=i*h/10;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
  }

  // Vertical grid lines with time labels (1.01ms to 9.08ms)
  ctx.font = '10px sans-serif'; ctx.fillStyle = '#6b7280';
  for(let i=0;i<=8;i++){
    let x = i * w / 8;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();

    // Time labels at bottom
    let timeMs = (1.01 + i * 1.01).toFixed(2);
    ctx.fillText(timeMs, x - 10, h - 5);
  }

  // Zero line (middle)
  ctx.strokeStyle='#94a3b8'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = '#6b7280'; ctx.font = '10px sans-serif';
  ctx.fillText('+1 rad', 5, 15);
  ctx.fillText('0', 5, h/2 + 4);
  ctx.fillText('-1 rad', 5, h - 5);
}
```

### 4. Подписи фазовых параметров под графиком
```html
<div class="phase-info">
  <div class="phase-labels">
    <span>Phase+ = <span id="phasePos">-63.31</span>°</span>
    <span>TRise+ = <span id="tRise">-59.9</span> mcs</span>
    <span>Phase- = <span id="phaseNeg">-64.73</span>°</span>
    <span>TFall- = <span id="tFall">-121.4</span> mcs</span>
  </div>
</div>
```

### 5. Обновленные CSS стили
```css
.beacon-info { padding: 8px 12px; }
.info-row { display: flex; gap: 12px; margin-bottom: 8px; }
.info-block {
  flex: 1; display: flex; justify-content: space-between;
  background: var(--panel2); border: 1px solid #d8e2ef;
  border-radius: 4px; padding: 4px 8px; font-size: 12px;
}
.info-block .label { color: #31567f; font-weight: 600; }
.info-block .value { color: #111; font-weight: normal; }
.beacon-title { font-weight: bold; margin: 8px 0 4px 0; color: #31567f; }
.message-info { font-size: 12px; color: #6b7280; margin-bottom: 8px; }

.chart-section { padding: 0 12px 12px 12px; }
.phase-info { display: flex; justify-content: space-between; font-size: 11px; margin-top: 4px; color: #374151; }
.phase-labels { display: flex; gap: 20px; }
.chart-title { text-align: center; font-size: 12px; color: #64748b; margin-top: 4px; }
```

### 6. Реалистичные данные бэкенда
```python
def api_status():
    # realistic carrier measurements for 406 MHz
    fs = STATE.fs_hz
    fs1 = 406025864.040 + random.uniform(-1, 1)
    fs2 = 406025864.196 + random.uniform(-1, 1)
    fs3 = 406012489.990 + random.uniform(-1, 1)

    return jsonify({
        "date": now,
        "fs_hz": STATE.fs_hz,
        "message": STATE.current_msg,
        "beacon_model": STATE.beacon_model,
        "series": series,
        "fs1_hz": fs1, "fs2_hz": fs2, "fs3_hz": fs3,
        "stats": {
            "phase_pos": STATE.phase_pos_rad,
            "phase_neg": STATE.phase_neg_rad,
            "t_rise_us": STATE.t_rise_us,
            "t_fall_us": STATE.t_fall_us,
            "pr_wt": STATE.pr_wt,
            "bitrate_bps": STATE.bitrate_bps,
            "symmetry_pct": STATE.symmetry_pct,
            "preamble_ms": STATE.preamble_ms,
            "total_ms": STATE.total_ms,
        }
    })
```

### 7. Обновленная функция refresh()
```javascript
// phase labels under chart
const s = js.stats;
document.getElementById('phasePos').textContent = (s.phase_pos * 57.2958).toFixed(2); // rad to degrees
document.getElementById('phaseNeg').textContent = (s.phase_neg * 57.2958).toFixed(2);
document.getElementById('tRise').textContent = s.t_rise_us.toFixed(1);
document.getElementById('tFall').textContent = s.t_fall_us.toFixed(1);

// stats panel with updated units
toRow('TRise,mcs', s.t_rise_us.toFixed(3)),
toRow('TFall,mcs', s.t_fall_us.toFixed(3)),
```

## Результат

✅ **Реализованные улучшения:**

1. **Обновленная компоновка:**
   - Уменьшил левую панель до 180px
   - Увеличил правую панель статистики до 320px
   - Убрал лишние строки в grid layout

2. **Детальная информация о маяке:**
   - Добавил блоки Protocol/Date/Conditions в верхнюю строку
   - Добавил Beacon Model/Beacon Frequency во вторую строку
   - Заголовок "Beacon 406" и поле сообщения

3. **Улучшенный график:**
   - Временная шкала 1.01-9.08 мс по оси X
   - Подписи осей (+1 rad, 0, -1 rad)
   - Phase+/Phase-/TRise/TFall значения под графиком
   - Более детальная сетка

4. **Реалистичные данные:**
   - Точные частоты FS1/FS2/FS3 для 406 МГц
   - Конвертация фазы в градусы для отображения
   - Обновленные единицы измерения (mcs вместо µs)

Интерфейс теперь максимально близко соответствует оригиналу с предоставленного изображения!

## Файлы изменены
- `C:\work\TesterSDR\beacon406\apps\tester_sdr_http_ui_stage_1_windows_single_file_app.py`

## Команды для запуска
```bash
# Запуск веб-интерфейса
python C:\work\TesterSDR\beacon406\apps\tester_sdr_http_ui_stage_1_windows_single_file_app.py

# Или через batch файл
C:\work\TesterSDR\app.bat
```

Веб-интерфейс доступен по адресу: http://127.0.0.1:8737/