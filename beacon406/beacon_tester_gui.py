from lib.logger import get_logger
log = get_logger(__name__)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GUI приложение для тестирования сигналов маяков COSPAS-SARSAT 406 МГц
Интерфейс в стиле COSPAS Beacon Tester v2
"""

import sys
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import struct
from collections import deque

# Добавляем путь к модулям beacon406
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QGridLayout, QTextEdit, QSlider, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QLineEdit,
    QProgressBar, QFrame, QListWidget, QTreeWidget, QTreeWidgetItem,
    QMenuBar, QMenu, QStatusBar, QToolBar, QDockWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QObject, QDateTime
from PySide6.QtGui import QAction, QFont, QColor, QPalette, QIcon, QPen, QBrush

import pyqtgraph as pg
from pyqtgraph import PlotWidget

# Импорт модулей обработки сигналов
from lib.backends import make_backend, SoapyBackend, FilePlaybackBackend
from lib.demod import (
    phase_demod_psk_msg_safe,
    calculate_pulse_params,
    detect_all_steps_by_mean_fast,
    extract_half_bits,
    halfbits_to_bytes_fast
)
from lib.metrics import process_psk_impulse
from lib.config import BACKEND_NAME, BACKEND_ARGS
from lib.backends import SDR_DEFAULT_HW_SR


class SignalProcessor(QThread):
    """Поток обработки сигналов SDR"""

    # Сигналы для обновления GUI
    spectrum_update = Signal(np.ndarray, np.ndarray)  # частоты, мощность
    waterfall_update = Signal(np.ndarray)  # строка водопада
    beacon_detected = Signal(dict)  # информация о маяке
    status_update = Signal(str)  # статус обработки
    rms_update = Signal(float)  # уровень RMS
    phase_update = Signal(np.ndarray)  # фазовые данные

    def __init__(self):
        super().__init__()
        self.backend = None
        self.is_running = False
        self.sample_rate = 1000000  # 1 MHz
        self.center_freq = 406037000  # 406.037 MHz
        self.gain = 40
        self.if_offset = -37000  # -37 kHz
        self.backend_name = "file"
        self.file_path = None

        # Буферы для обработки
        self.iq_buffer = np.zeros(65536, dtype=np.complex64)
        self.waterfall_buffer = deque(maxlen=100)

    def setup_backend(self, backend_name: str, **kwargs):
        """Настройка SDR бэкенда"""
        try:
            self.backend_name = backend_name

            if backend_name == "file" and 'file_path' in kwargs:
                self.file_path = kwargs['file_path']
                self.backend = FilePlaybackBackend(
                    self.file_path,
                    self.sample_rate,
                    self.center_freq + self.if_offset
                )
            else:
                # Аппаратные SDR
                self.backend = make_backend(backend_name, None)
                if self.backend:
                    self.backend.set_sample_rate(self.sample_rate)
                    self.backend.set_center_freq(self.center_freq + self.if_offset)
                    if hasattr(self.backend, 'set_gain'):
                        self.backend.set_gain(self.gain)

            self.status_update.emit(f"Бэкенд {backend_name} инициализирован")
            return True

        except Exception as e:
            self.status_update.emit(f"Ошибка инициализации: {str(e)}")
            return False

    def run(self):
        """Основной цикл обработки"""
        if not self.backend:
            self.status_update.emit("Бэкенд не инициализирован")
            return

        self.is_running = True
        chunk_size = 65536

        while self.is_running:
            try:
                # Чтение данных
                samples = self.backend.read_samples(chunk_size)
                if samples is None or len(samples) == 0:
                    time.sleep(0.01)
                    continue

                # Расчет спектра (FFT)
                fft_data = np.fft.fft(samples)
                fft_shift = np.fft.fftshift(fft_data)
                magnitude = 20 * np.log10(np.abs(fft_shift) + 1e-10)

                # Частотная ось
                freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate))
                freqs = freqs + self.center_freq

                # Обновление спектра
                self.spectrum_update.emit(freqs, magnitude)

                # Добавление строки в водопад
                waterfall_line = magnitude[::len(magnitude)//1024] if len(magnitude) > 1024 else magnitude
                self.waterfall_update.emit(waterfall_line)

                # Расчет RMS
                rms_value = np.sqrt(np.mean(np.abs(samples)**2))
                rms_dbm = 20 * np.log10(rms_value + 1e-10)
                self.rms_update.emit(rms_dbm)

                # Демодуляция PSK (если сигнал достаточно сильный)
                if rms_dbm > -45:  # порог детекции
                    self.process_beacon(samples)

            except Exception as e:
                self.status_update.emit(f"Ошибка обработки: {str(e)}")

            time.sleep(0.01)  # небольшая задержка

    def process_beacon(self, samples):
        """Обработка и декодирование сигнала маяка"""
        try:
            # Демодуляция PSK
            result = phase_demod_psk_msg_safe(
                samples,
                self.sample_rate,
                lpf_cutoff_hz=12000,
                decim_factor=4
            )

            if result and result['msg_bytes']:
                # Маяк обнаружен
                beacon_info = {
                    'timestamp': datetime.now().isoformat(),
                    'frequency': self.center_freq,
                    'message': result['msg_bytes'].hex(),
                    'bit_count': result['bit_count'],
                    'params': result['params']
                }
                self.beacon_detected.emit(beacon_info)

                # Обновление фазовых данных
                if 'phase' in result:
                    self.phase_update.emit(result['phase'])

        except Exception as e:
            self.status_update.emit(f"Ошибка демодуляции: {str(e)}")

    def stop(self):
        """Остановка обработки"""
        self.is_running = False
        if self.backend:
            self.backend.close()
            self.backend = None


class SpectrumWidget(PlotWidget):
    """Виджет отображения спектра"""

    def __init__(self):
        super().__init__()

        self.setLabel('left', 'Мощность', units='dBm')
        self.setLabel('bottom', 'Частота', units='Hz')
        self.setTitle('Спектр сигнала')
        self.showGrid(x=True, y=True, alpha=0.3)

        # Кривая спектра
        self.spectrum_curve = self.plot(pen=pg.mkPen(color='cyan', width=1))

        # Маркеры
        self.peak_marker = self.plot(
            pen=None,
            symbol='o',
            symbolSize=10,
            symbolBrush='r'
        )

        # Настройки отображения
        self.setYRange(-120, -20)
        self.enableAutoRange(axis='x')

    def update_spectrum(self, freqs, magnitude):
        """Обновление данных спектра"""
        if len(freqs) > 0 and len(magnitude) > 0:
            self.spectrum_curve.setData(freqs, magnitude)

            # Поиск пика
            peak_idx = np.argmax(magnitude)
            if peak_idx >= 0:
                self.peak_marker.setData([freqs[peak_idx]], [magnitude[peak_idx]])


class WaterfallWidget(pg.ImageView):
    """Виджет водопада"""

    def __init__(self):
        super().__init__()

        # Инициализация буфера водопада
        self.waterfall_data = np.zeros((100, 1024))
        self.current_row = 0

        # Настройка цветовой карты
        self.setColorMap(pg.colormap.get('viridis'))

    def add_line(self, data):
        """Добавление новой строки в водопад"""
        # Преобразование к нужному размеру
        if len(data) != 1024:
            data = np.interp(
                np.linspace(0, len(data)-1, 1024),
                np.arange(len(data)),
                data
            )

        # Сдвиг данных вверх
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
        self.waterfall_data[-1, :] = data

        # Обновление изображения
        self.setImage(self.waterfall_data.T, autoLevels=False)


class BeaconTesterMainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()

        self.processor = SignalProcessor()
        self.setup_ui()
        self.connect_signals()

        # Таймер обновления статуса
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # каждую секунду

    def setup_ui(self):
        """Создание интерфейса"""
        self.setWindowTitle("Beacon Tester - Тестер маяков COSPAS-SARSAT")
        self.setGeometry(100, 100, 1400, 900)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QVBoxLayout(central_widget)

        # Панель управления
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Разделитель для основного контента
        splitter = QSplitter(Qt.Vertical)

        # Верхняя часть - спектр и водопад
        spectrum_container = QWidget()
        spectrum_layout = QHBoxLayout(spectrum_container)

        # Спектр
        self.spectrum_widget = SpectrumWidget()
        spectrum_layout.addWidget(self.spectrum_widget, 2)

        # Водопад
        self.waterfall_widget = WaterfallWidget()
        spectrum_layout.addWidget(self.waterfall_widget, 1)

        splitter.addWidget(spectrum_container)

        # Нижняя часть - вкладки с информацией
        self.tab_widget = self.create_tabs()
        splitter.addWidget(self.tab_widget)

        splitter.setSizes([500, 300])
        main_layout.addWidget(splitter)

        # Статусная строка
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

        # Меню
        self.create_menu()

    def create_control_panel(self) -> QGroupBox:
        """Создание панели управления"""
        panel = QGroupBox("Управление SDR")
        layout = QHBoxLayout()

        # Выбор бэкенда
        layout.addWidget(QLabel("Источник:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "file", "soapy_rtl", "soapy_hackrf",
            "soapy_airspy", "soapy_sdrplay", "rsa306"
        ])
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        layout.addWidget(self.backend_combo)

        # Кнопка выбора файла
        self.file_button = QPushButton("Выбрать файл...")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # Частота
        layout.addWidget(QLabel("Частота (МГц):"))
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(400, 410)
        self.freq_spin.setValue(406.037)
        self.freq_spin.setSingleStep(0.001)
        self.freq_spin.setDecimals(3)
        layout.addWidget(self.freq_spin)

        # Усиление
        layout.addWidget(QLabel("Усиление (дБ):"))
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 60)
        self.gain_spin.setValue(40)
        layout.addWidget(self.gain_spin)

        # Sample rate
        layout.addWidget(QLabel("Sample Rate:"))
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["1000000", "2000000", "2400000", "3000000"])
        self.sr_combo.setCurrentText("1000000")
        layout.addWidget(self.sr_combo)

        layout.addStretch()

        # Кнопки управления
        self.start_button = QPushButton("Старт")
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        # Индикатор RMS
        layout.addWidget(QLabel("RMS:"))
        self.rms_label = QLabel("-∞ dBm")
        self.rms_label.setMinimumWidth(80)
        layout.addWidget(self.rms_label)

        panel.setLayout(layout)
        return panel

    def create_tabs(self) -> QTabWidget:
        """Создание вкладок с информацией"""
        tabs = QTabWidget()

        # Вкладка декодированных сообщений
        self.message_tab = self.create_message_tab()
        tabs.addTab(self.message_tab, "Сообщения")

        # Вкладка параметров сигнала
        self.params_tab = self.create_params_tab()
        tabs.addTab(self.params_tab, "Параметры")

        # Вкладка статистики
        self.stats_tab = self.create_stats_tab()
        tabs.addTab(self.stats_tab, "Статистика")

        # Вкладка лога
        self.log_tab = self.create_log_tab()
        tabs.addTab(self.log_tab, "Лог")

        return tabs

    def create_message_tab(self) -> QWidget:
        """Вкладка декодированных сообщений"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Таблица сообщений
        self.message_table = QTableWidget()
        self.message_table.setColumnCount(5)
        self.message_table.setHorizontalHeaderLabels([
            "Время", "Частота", "Тип", "ID маяка", "Данные"
        ])
        self.message_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.message_table)

        # Детальная информация
        self.message_detail = QTextEdit()
        self.message_detail.setReadOnly(True)
        self.message_detail.setMaximumHeight(150)
        layout.addWidget(self.message_detail)

        widget.setLayout(layout)
        return widget

    def create_params_tab(self) -> QWidget:
        """Вкладка параметров сигнала"""
        widget = QWidget()
        layout = QGridLayout()

        # Параметры PSK
        params = [
            ("Фаза +:", "---"),
            ("Фаза -:", "---"),
            ("Время нарастания:", "---"),
            ("Время спада:", "---"),
            ("Асимметрия:", "---"),
            ("Период модуляции:", "---"),
            ("SNR:", "---"),
            ("Частота ошибок:", "---")
        ]

        self.param_labels = {}
        for i, (name, value) in enumerate(params):
            layout.addWidget(QLabel(name), i, 0)
            label = QLabel(value)
            self.param_labels[name] = label
            layout.addWidget(label, i, 1)

        # График фазы
        self.phase_plot = PlotWidget()
        self.phase_plot.setLabel('left', 'Фаза', units='рад')
        self.phase_plot.setLabel('bottom', 'Время', units='мс')
        self.phase_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.phase_plot, 0, 2, 8, 1)

        widget.setLayout(layout)
        return widget

    def create_stats_tab(self) -> QWidget:
        """Вкладка статистики"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Статистика приема
        stats_group = QGroupBox("Статистика приема")
        stats_layout = QGridLayout()

        self.stats_labels = {}
        stats_items = [
            ("Всего маяков:", "0"),
            ("Успешно декодировано:", "0"),
            ("Ошибок CRC:", "0"),
            ("Средний SNR:", "--- дБ"),
            ("Время работы:", "00:00:00")
        ]

        for i, (name, value) in enumerate(stats_items):
            stats_layout.addWidget(QLabel(name), i, 0)
            label = QLabel(value)
            self.stats_labels[name] = label
            stats_layout.addWidget(label, i, 1)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # График истории SNR
        self.snr_plot = PlotWidget()
        self.snr_plot.setLabel('left', 'SNR', units='dБ')
        self.snr_plot.setLabel('bottom', 'Время')
        self.snr_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.snr_plot)

        widget.setLayout(layout)
        return widget

    def create_log_tab(self) -> QWidget:
        """Вкладка лога"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)

        # Кнопки управления логом
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Очистить")
        clear_button.clicked.connect(self.log_text.clear)
        button_layout.addWidget(clear_button)

        save_button = QPushButton("Сохранить...")
        save_button.clicked.connect(self.save_log)
        button_layout.addWidget(save_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget

    def create_menu(self):
        """Создание меню"""
        menubar = self.menuBar()

        # Меню Файл
        file_menu = menubar.addMenu("Файл")

        open_action = QAction("Открыть запись...", self)
        open_action.triggered.connect(self.select_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Меню Вид
        view_menu = menubar.addMenu("Вид")

        # Меню Инструменты
        tools_menu = menubar.addMenu("Инструменты")

        # Меню Справка
        help_menu = menubar.addMenu("Справка")
        about_action = QAction("О программе", self)
        help_menu.addAction(about_action)

    def connect_signals(self):
        """Подключение сигналов"""
        self.processor.spectrum_update.connect(self.update_spectrum)
        self.processor.waterfall_update.connect(self.update_waterfall)
        self.processor.beacon_detected.connect(self.on_beacon_detected)
        self.processor.status_update.connect(self.update_log)
        self.processor.rms_update.connect(self.update_rms)
        self.processor.phase_update.connect(self.update_phase)

    def on_backend_changed(self, backend_name: str):
        """Обработка изменения бэкенда"""
        self.file_button.setEnabled(backend_name == "file")

    def select_file(self):
        """Выбор файла для воспроизведения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл записи",
            "C:/work/TesterSDR/captures",
            "CF32 Files (*.cf32);;All Files (*.*)"
        )

        if file_path:
            self.file_button.setText(Path(file_path).name)
            self.selected_file = file_path
            self.update_log(f"Выбран файл: {file_path}")

    def start_processing(self):
        """Запуск обработки"""
        backend_name = self.backend_combo.currentText()

        # Настройка параметров
        self.processor.sample_rate = int(self.sr_combo.currentText())
        self.processor.center_freq = int(self.freq_spin.value() * 1e6)
        self.processor.gain = self.gain_spin.value()

        # Настройка бэкенда
        kwargs = {}
        if backend_name == "file":
            if hasattr(self, 'selected_file'):
                kwargs['file_path'] = self.selected_file
            else:
                self.update_log("Ошибка: файл не выбран")
                return

        if self.processor.setup_backend(backend_name, **kwargs):
            self.processor.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_log("Обработка запущена")

    def stop_processing(self):
        """Остановка обработки"""
        self.processor.stop()
        self.processor.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_log("Обработка остановлена")

    @Slot(np.ndarray, np.ndarray)
    def update_spectrum(self, freqs, magnitude):
        """Обновление спектра"""
        self.spectrum_widget.update_spectrum(freqs, magnitude)

    @Slot(np.ndarray)
    def update_waterfall(self, data):
        """Обновление водопада"""
        self.waterfall_widget.add_line(data)

    @Slot(dict)
    def on_beacon_detected(self, beacon_info):
        """Обработка обнаруженного маяка"""
        # Добавление в таблицу
        row = self.message_table.rowCount()
        self.message_table.insertRow(row)

        self.message_table.setItem(row, 0, QTableWidgetItem(beacon_info['timestamp']))
        self.message_table.setItem(row, 1, QTableWidgetItem(f"{beacon_info['frequency']/1e6:.3f} MHz"))
        self.message_table.setItem(row, 2, QTableWidgetItem("406 EPIRB"))
        self.message_table.setItem(row, 3, QTableWidgetItem("---"))
        self.message_table.setItem(row, 4, QTableWidgetItem(beacon_info['message'][:32]))

        # Обновление деталей
        details = f"Маяк обнаружен:\n"
        details += f"Время: {beacon_info['timestamp']}\n"
        details += f"Частота: {beacon_info['frequency']/1e6:.3f} MHz\n"
        details += f"Сообщение: {beacon_info['message']}\n"
        details += f"Биты: {beacon_info['bit_count']}\n"

        if 'params' in beacon_info and beacon_info['params']:
            params = beacon_info['params']
            details += f"\nПараметры PSK:\n"
            details += f"  Фаза+: {params.get('PosPhase', 0):.2f}\n"
            details += f"  Фаза-: {params.get('NegPhase', 0):.2f}\n"
            details += f"  Асимметрия: {params.get('Ass', 0):.3f}\n"

        self.message_detail.setText(details)

        self.update_log(f"Маяк обнаружен: {beacon_info['message'][:16]}...")

    @Slot(str)
    def update_log(self, message):
        """Обновление лога"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    @Slot(float)
    def update_rms(self, rms_dbm):
        """Обновление индикатора RMS"""
        self.rms_label.setText(f"{rms_dbm:.1f} dBm")

        # Цветовая индикация
        if rms_dbm > -30:
            self.rms_label.setStyleSheet("color: red;")
        elif rms_dbm > -45:
            self.rms_label.setStyleSheet("color: yellow;")
        else:
            self.rms_label.setStyleSheet("color: green;")

    @Slot(np.ndarray)
    def update_phase(self, phase_data):
        """Обновление графика фазы"""
        if len(phase_data) > 0:
            time_axis = np.arange(len(phase_data)) / 1000  # в мс
            self.phase_plot.clear()
            self.phase_plot.plot(time_axis, phase_data, pen='y')

    def update_status(self):
        """Обновление статусной строки"""
        if self.processor.is_running:
            self.status_bar.showMessage("Обработка активна")
        else:
            self.status_bar.showMessage("Готов к работе")

    def save_log(self):
        """Сохранение лога в файл"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить лог",
            f"beacon_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.toPlainText())
            self.update_log(f"Лог сохранен: {file_path}")

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.processor.is_running:
            self.stop_processing()
        event.accept()


def main():
    """Точка входа"""
    app = QApplication(sys.argv)

    # Настройка стиля
    app.setStyle('Fusion')

    # Темная тема
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.black)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    # Создание и отображение главного окна
    window = BeaconTesterMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()