from __future__ import annotations
#
# backends.py
# v1.1 (2025-09-18)
# Единый тонкий адаптер для работы с разными SDR без правок основной логики анализа/GUI.
# Идея: основной код получает только np.complex64 буферы через единый интерфейс.
# Добавление нового SDR = +1 класс и регистрация в фабрике make_backend().
# Проверены все SDR и file
# в auto последовательность поиска rtl,hackrf,airspy,sdrplay,rsa (может быть потом добавить file)
# Добавлен статус SDR
# SDR работают со своим оптимальным SR (SAMPLE RATE) если он выше 1М идет простая децимация
# например 1024 идет без децимации а 2Ms / 2
# если ниже например RSA 825Ks идет без децимации (нужно проверить PSK /4 )
# нужно оптимизировать работу с буфером сейчас алгоритмы разные у  RSA, AIRspy
#
from lib.logger import get_logger
log = get_logger(__name__)
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

# Import SigMF support functions
try:
    from .sigio import open_iq as sigio_open_iq
except ImportError:
    sigio_open_iq = None
import ctypes as ct, threading, time
from collections import deque
import os, platform


# --- Calibration offsets for each SDR (in dB) ---
SDR_CALIB_OFFSETS_DB = {
    "rtlsdr":   0.0,   # подгоним позже
    "hackrf":   -10.0,
    "airspy":   35.0,  # Airspy обычно занижает на ~30 dB
    "sdrplay":  20.0,
    "rsa306":   12.0,  # Tektronix RSA306(A/B)
    "file":     0.0,
}

# --- Default HW sample rates per SDR ---
SDR_DEFAULT_HW_SR = {
    "rtlsdr": 1_024_000.0,
    "hackrf": 2_000_000.0,
    "sdrplay": 2_000_000.0,
    "airspy": 3_000_000.0,
    "rsa306": 1_000_000.0,  # ~Fs при BW=500 kHz, уточняется через DLL
}

# --- ДОБАВИТЬ: авто-настройка путей Soapy/SDRplay на Windows ---
def _ensure_soapy_env_on_windows():
    if platform.system() != "Windows":
        return
    # 1) Пути к плагинам Soapy
    if not os.environ.get("SOAPY_SDR_PLUGIN_PATH"):
        for p in (
            r"C:\Program Files\PothosSDR\lib\SoapySDR\modules0.8",
            r"C:\Program Files (x86)\PothosSDR\lib\SoapySDR\modules0.8",
        ):
            if os.path.isdir(p):
                os.environ["SOAPY_SDR_PLUGIN_PATH"] = p
                break
    # 2) Пути к DLL (Python 3.8+): добавим и в DLL search, и в PATH процесса
    for p in (r"C:\Program Files\PothosSDR\bin",
              r"C:\Program Files\SDRplay\API\x64"):
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)  # Windows 10+
            except Exception:
                pass
            if "PATH" in os.environ and p not in os.environ["PATH"]:
                os.environ["PATH"] = p + ";" + os.environ["PATH"]



class SDRBackend(ABC):
    """Единый интерфейс для всех SDR-источников IQ."""
    def __init__(self,
                 sample_rate: float,
                 center_freq: float,
                 gain_db: Optional[float] = None,
                 agc: bool = False,
                 corr_ppm: int = 0):
        # Храним параметры как float/int без сюрпризов
        self.sample_rate_sps = float(sample_rate)
        # NEW: фактический Fs, который наружу отдаёт read(); по умолчанию = запросу
        self.actual_sample_rate_sps = float(sample_rate)
        self.center_freq_hz  = float(center_freq)
        self.gain_db         = None if gain_db is None else float(gain_db)
        self.agc             = bool(agc)
        self.corr_ppm        = int(corr_ppm)

    # Унифицированный статус «что реально установилось/работает»
    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": self.__class__.__name__,
            "driver": None,
            "requested_sample_rate_sps": float(self.sample_rate_sps),
            "actual_sample_rate_sps": float(getattr(self, "actual_sample_rate_sps", self.sample_rate_sps)),
            "requested_center_freq_hz": float(self.center_freq_hz),
            "calib_offset_db": float(getattr(self, "calib_offset_db", getattr(self, "get_calib_offset_db", lambda: 0.0)())),
            "agc_on": bool(getattr(self, "agc", False)),
            "corr_ppm": int(getattr(self, "corr_ppm", 0)),
        }

    # Удобный печатный вид для логов/консоли
    def pretty_status(self) -> str:
        s = self.get_status()
        lines = []
        for k in sorted(s.keys()):
            lines.append(f"{k:28s}: {s[k]}")
        return "\n".join(lines)


    @abstractmethod
    def read(self, nsamps: int) -> np.ndarray:
        """Вернуть массив complex64 длиной <= nsamps. Пустой массив = временно нет данных."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Остановить/освободить устройство/ресурсы."""
        ...

class SoapyBackend(SDRBackend):
    """
    Универсальный Soapy-бэкенд с авто-децимацией.
    Любое SDR работает на своём «родном» sample_rate, а наружу выдаётся 1 MS/s.
    COMBAT/STRICT_COMPAT: внешний API не меняем. Изменения только во внутренней ветке Airspy.
    """

    def __init__(self, device_args: Dict[str, Any], **kw):
        # Читаем if_offset_hz для BB-shift (Zero-IF контракт)
        self._if_offset_hz = float(kw.pop("if_offset_hz", 0.0))

        super().__init__(**kw)
        try:
            # lazy import: optional dependency (SoapySDR подключаем только при создании Soapy-бэкенда)
            # _ensure_soapy_env_on_windows()  # оставим закомментированным, чтобы не менять текущее поведение
            import SoapySDR  # type: ignore
            from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32  # type: ignore
        except Exception as e:
            raise RuntimeError("SoapySDR не установлен: pip install SoapySDR") from e

        self._SoapySDR = SoapySDR
        self._SOAPY_SDR_RX = SOAPY_SDR_RX
        self._SOAPY_SDR_CF32 = SOAPY_SDR_CF32

        # Открываем устройство
        self.dev = SoapySDR.Device(device_args)
        drv_key = str(self.dev.getDriverKey()).lower()
        self._drv_key = drv_key  # COMBAT: сохраняем ключ драйвера для внутренней логики

        # Получаем информацию об устройстве
        try:
            hw_info = self.dev.getHardwareInfo()
            if hw_info:
                # Извлекаем основную информацию
                parts = []
                if 'product' in hw_info:
                    parts.append(hw_info['product'])
                if 'serial' in hw_info:
                    parts.append(f"S/N:{hw_info['serial']}")
                if 'tuner' in hw_info:
                    parts.append(f"tuner:{hw_info['tuner']}")
                self.device_info = ' '.join(parts) if parts else f"{drv_key} device"
            else:
                self.device_info = f"{drv_key} device"
        except Exception:
            self.device_info = f"{drv_key} device"

        fs_req = self.sample_rate_sps       # то, что запросил код (обычно 1e6)
        fs_hw = SDR_DEFAULT_HW_SR.get(drv_key, fs_req)

        self.decim = int(round(fs_hw / fs_req)) if fs_req > 0 else 1
        if self.decim < 1:
            self.decim = 1
        self.sample_rate_hw = fs_hw
        # NEW: реальный Fs после внутренней децимации (то, что вернёт read())
        self.actual_sample_rate_sps = float(self.sample_rate_hw / self.decim)

        if abs(fs_hw - fs_req) > 1e-3:
            log.info("%s: analysis Fs=%.2f MS/s, HW Fs=%.2f MS/s, decim=%d", drv_key, fs_req/1e6, fs_hw/1e6, self.decim)

        # Применяем частоту дискретизации
        self.dev.setSampleRate(self._SOAPY_SDR_RX, 0, self.sample_rate_hw)

        # Калибровка dB под драйвер
        self.calib_offset_db = SDR_CALIB_OFFSETS_DB.get(drv_key, 0.0)

        # Инициализация NCO для BB-shift (Zero-IF контракт)
        # Используем ту же логику, что старый NCO в beacon406_PSK_FM-plot.py:
        # BB_SHIFT_HZ = IF_OFFSET_HZ, mixer = exp(+1j × 2π × BB_SHIFT_HZ / Fs × n)
        # ВАЖНО: _mix_w вычисляется для actual_sample_rate_sps (ПОСЛЕ децимации)
        self._mix_phase = 0.0
        self._mix_shift_hz = self._if_offset_hz  # БЕЗ инверсии знака!
        self._mix_w = 2.0 * np.pi * self._mix_shift_hz / self.actual_sample_rate_sps if self._mix_shift_hz != 0 else 0.0

        # Логируем настройку BB-shift
        if self._if_offset_hz != 0:
            log.info("%s: if_offset_hz=%.0f Hz → BB-shift=%.0f Hz (Zero-IF контракт)",
                     drv_key, self._if_offset_hz, self._mix_shift_hz)
        else:
            log.debug("%s: if_offset_hz=0, BB-shift выключен (Zero-IF режим)", drv_key)

        # Частота
        self.dev.setFrequency(self._SOAPY_SDR_RX, 0, self.center_freq_hz)

        # AGC / Gain
        try:
            self.dev.setGainMode(self._SOAPY_SDR_RX, 0, bool(self.agc))
        except Exception:
            pass
        if (self.gain_db is not None) and (not self.agc):
            try:
                self.dev.setGain(self._SOAPY_SDR_RX, 0, float(self.gain_db))
            except Exception:
                try:
                    self.dev.setGain(self._SOAPY_SDR_RX, 0, "TUNER", float(self.gain_db))
                except Exception:
                    pass

        # Поток приёмника
        if "airspy" in drv_key:
            # Для Airspy пробуем компактные буферы — быстрее оборот, меньше риска underrun.
            try:
                self.stream = self.dev.setupStream(
                    self._SOAPY_SDR_RX, self._SOAPY_SDR_CF32, [], {"buffers": "8"}
                )
            except Exception:
                self.stream = self.dev.setupStream(self._SOAPY_SDR_RX, self._SOAPY_SDR_CF32)
        else:
            self.stream = self.dev.setupStream(self._SOAPY_SDR_RX, self._SOAPY_SDR_CF32)

        self.dev.activateStream(self.stream)

        # Настройка каскадов для HackRF / Airspy / RTL (как было)
        try:
            if "hackrf" in drv_key:  # HackRF gain control (AMP=0, LNA=24, VGA=40)
                # AMP (front-end amplifier) — включает +14 дБ усиление перед LNA
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "AMP", 0)        # 0 = выкл., 1 = вкл. (+14 дБ)
                # LNA (RF LNA) — малошумящий усилитель
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "LNA", 24)       # шаг 8 дБ: 0, 8, 16, 24, 32, 40 дБ
                # VGA (Baseband VGA) — основной регулятор после смесителя
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "VGA", 40)       # шаг 2 дБ: 0…62 дБ
                # Bandwidth (baseband filter) — можно ставить равным sample_rate_hw
                try:
                    self.dev.setBandwidth(self._SOAPY_SDR_RX, 0, self.sample_rate_hw)
                except Exception:
                    pass  # не все версии Soapy поддерживают setBandwidth для HackRF

            if "airspy" in drv_key:  # Airspy gain control
                # LNA gain
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "LNA", 12)       # 0…21 дБ, шаг 3 дБ
                # Mixer gain
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "MIXER", 2)      # 0…15 дБ, шаг 1 дБ
                # IF gain
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "IF", 0)         # 0…15 дБ, шаг 1 дБ

            if "rtlsdr" in drv_key:  # RTL-SDR gain control
                # RF Gain (фиксированные ступени в зависимости от тюнера, шаг нерегулярный)
                # Примеры доступных значений: [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4,
                # 15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4,
                # 37.2, 38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "TUNER", 28.0)   # пример: 28 дБ

            if "sdrplay" in drv_key:  # SDRplay gain control
                # LNAstate — дискретные состояния LNA (в зависимости от диапазона обычно 0…9).
                # Каждое значение соответствует фиксированному набору каскадов LNA (~2–10 дБ на шаг).
                # 0 = минимальное усиление (LNA выкл.), 1–9 = всё более высокое.
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "LNAstate", 1)

                # IFGRdB — IF Gain Reduction (шаг: 1 дБ, диапазон 0…59 дБ).
                # Чем больше значение, тем меньше итоговое усиление.
                # Типичное рабочее значение для −30 dBm входа: ~35 дБ.
                self.dev.setGain(self._SOAPY_SDR_RX, 0, "IFGRdB", 35)

                # AGC — автоматическое управление усилением.
                # "false" = AGC выключен, все усиления только вручную (рекомендуется для тестов).
                self.dev.writeSetting("AGC", "false")
                # Bandwidth - фиксированные значения: обычно 200 kHz, 300 kHz, 600 kHz, 1.536 MHz, 5 MHz, 6 MHz, 7 MHz, 8 MHz
                self.dev.setBandwidth(self._SOAPY_SDR_RX, 0, 200_000)

        except Exception:
            pass

    def get_calib_offset_db(self) -> float:
        return getattr(self, "calib_offset_db", 0.0)

    # Внутренний хелпер (удобно для диагностики; внешний контракт не меняем)
    def get_driver_key(self) -> str:
        return getattr(self, "_drv_key", "")

    def read(self, nsamps: int) -> np.ndarray:
        if nsamps <= 0:
            return np.empty(0, dtype=np.complex64)

        nsamps_hw = nsamps * self.decim

        # === AIRSPY: читаем «кусочками» с коротким таймаутом и накапливаем (COMBAT) ===
        if "airspy" in getattr(self, "_drv_key", ""):
            remaining = int(nsamps_hw)
            filled = 0
            # Шаг чтения небольшой — быстрее оборот, ниже шанс underrun
            step = int(min(16384 * max(1, self.decim), remaining))
            buff_hw = np.empty(nsamps_hw, dtype=np.complex64)
            while remaining > 0:
                chunk = step if remaining >= step else remaining
                tmp = np.empty(chunk, dtype=np.complex64)
                sr = self.dev.readStream(self.stream, [tmp], chunk, timeoutUs=100_000)  # 100 ms
                ret = int(getattr(sr, "ret", -999))
                if ret > 0:
                    buff_hw[filled:filled+ret] = tmp[:ret]
                    filled += ret
                    remaining -= ret
                    continue
                if ret in (-1, -2, -3, -4):
                    # частичный буфер лучше, чем пусто — выходим с тем, что успели
                    break
                raise RuntimeError(f"Soapy readStream error: {ret}")

            if filled == 0:
                return np.empty(0, dtype=np.complex64)
            x = buff_hw[:filled]

        else:
            # === Остальные драйверы — как было (STRICT_COMPAT) ===
            buff = np.empty(nsamps_hw, dtype=np.complex64)
            sr = self.dev.readStream(self.stream, [buff], nsamps_hw, timeoutUs=int(500_000))
            ret = int(getattr(sr, "ret", -999))
            if ret > 0:
                x = buff[:ret]
            elif ret in (-1, -2, -3, -4):
                return np.empty(0, dtype=np.complex64)
            else:
                raise RuntimeError(f"Soapy readStream error: {ret}")

        # === Единая децимация (усреднение блоками по decim) — логика не менялась ===
        if self.decim > 1:
            n_blocks = len(x) // self.decim
            if n_blocks > 0:
                x = x[:n_blocks * self.decim].reshape(n_blocks, self.decim)
                x = x.mean(axis=1).astype(np.complex64)
            else:
                x = x[::self.decim]  # fallback на простой downsample

        # === BB-shift для Zero-IF (применяется ПОСЛЕ децимации) ===
        # _mix_shift_hz = if_offset_hz, применяем exp(+1j×ph) как старый NCO
        if self._mix_w != 0 and len(x) > 0:
            n = np.arange(len(x), dtype=np.float64)
            ph = self._mix_phase + self._mix_w * n
            mixer = np.exp(1j * ph).astype(np.complex64)
            x = x * mixer
            self._mix_phase = float((self._mix_phase + self._mix_w * len(x)) % (2.0 * np.pi))

        return x

    def get_status(self) -> Dict[str, Any]:
        s = super().get_status()
        # Ключ драйвера (rtl, hackrf, airspy, sdrplay и т.п.)
        s.update({
            "driver": getattr(self, "_drv_key", None),
            "device_info": getattr(self, "device_info", None),
            "hw_sample_rate_sps": float(getattr(self, "sample_rate_hw", getattr(self, "sample_rate_sps", 0.0))),
            "decim": int(getattr(self, "decim", 1)),
            "if_offset_hz": float(getattr(self, "_if_offset_hz", 0.0)),  # BB-shift = -if_offset_hz (Zero-IF)
        })
        # Фактическая центральная частота
        try:
            s["actual_center_freq_hz"] = float(self.dev.getFrequency(self._SOAPY_SDR_RX, 0))
        except Exception:
            s["actual_center_freq_hz"] = None
        # Полоса (если поддерживается)
        try:
            s["bandwidth_hz"] = float(self.dev.getBandwidth(self._SOAPY_SDR_RX, 0))
        except Exception:
            s["bandwidth_hz"] = None
        # AGC (если поддерживается)
        try:
            s["agc_on"] = bool(self.dev.getGainMode(self._SOAPY_SDR_RX, 0))
        except Exception:
            pass
        # Общий gain
        try:
            s["overall_gain_db"] = float(self.dev.getGain(self._SOAPY_SDR_RX, 0))
        except Exception:
            s["overall_gain_db"] = None
        # По каскадам
        stage = {}
        try:
            names = list(self.dev.listGains(self._SOAPY_SDR_RX, 0))
        except Exception:
            names = []
        for nm in names:
            try:
                stage[str(nm)] = float(self.dev.getGain(self._SOAPY_SDR_RX, 0, nm))
            except Exception:
                try:
                    # Некоторые драйверы (SDRplay) возвращают «нестандартные» элементы через readSetting
                    val = self.dev.readSetting(nm)
                    stage[str(nm)] = float(val)
                except Exception:
                    pass
        # Попробуем явно вытащить популярные элементы
        for nm in ("LNA", "MIXER", "IF", "TUNER", "LNAstate", "IFGRdB", "AMP", "VGA"):
            if nm not in stage:
                try:
                    stage[nm] = float(self.dev.getGain(self._SOAPY_SDR_RX, 0, nm))
                except Exception:
                    pass
        s["stage_gains_db"] = stage or None
        return s


    def stop(self) -> None:
        try:
            self.dev.deactivateStream(self.stream)
            self.dev.closeStream(self.stream)
        except Exception:
            pass


class FilePlaybackBackend(SDRBackend):
    def __init__(self, path: str, **kw):
        # Вернём поддержку IF, как было в v2/v3_1
        self._if_offset_hz = float(kw.pop("if_offset_hz", 0.0) or 0.0)

        super().__init__(**kw)
        # NEW: для файла реальный Fs = заявленному sample_rate_sps
        self.actual_sample_rate_sps = float(self.sample_rate_sps)
        # грузим файл ОДИН раз в RAM
        self._path = str(path)
        self._data = np.fromfile(self._path, dtype=np.complex64)
        if self._data.size == 0:
            raise RuntimeError(f"Файл пуст или не найден: {path}")

        self._pos = 0
        self._eof = False

        # калибровка
        self.calib_offset_db = SDR_CALIB_OFFSETS_DB.get("file", 0.0)

        # обратная компенсация IF (к тому, что делает beacon406-plot)
        self._mix_shift_hz = self._if_offset_hz
        self._mix_phase = 0.0
        self._mix_w = (2.0*np.pi*self._mix_shift_hz / float(self.sample_rate_sps)
                       if self._mix_shift_hz else 0.0)


    def get_status(self) -> Dict[str, Any]:
        s = super().get_status()
        # Формируем удобочитаемую информацию о файле
        file_path = getattr(self, "_path", None)
        if file_path:

            filename = os.path.basename(file_path)
            device_info = f"File: {filename}"
        else:
            device_info = "File playback"

        s.update({
            "driver": "file",
            "device_info": device_info,
            "file_path": file_path,
            "if_offset_hz": float(getattr(self, "_if_offset_hz", 0.0)),
            "mix_shift_hz": float(getattr(self, "_mix_shift_hz", 0.0)),
            "eof": bool(getattr(self, "_eof", False)),
        })
        return s


    def get_calib_offset_db(self) -> float:
        return SDR_CALIB_OFFSETS_DB.get("file", 0.0)


    def read(self, nsamps: int) -> np.ndarray:
        # после конца файла — всегда возвращаем пусто
        if nsamps <= 0 or self._eof:
            return np.empty(0, dtype=np.complex64)

        N = self._data.size
        remain = N - self._pos
        if remain <= 0:
            self._eof = True
            return np.empty(0, dtype=np.complex64)

        take = nsamps if nsamps <= remain else remain
        out = self._data[self._pos:self._pos + take].astype(np.complex64, copy=False)
        self._pos += take
        if self._pos >= N:
            self._eof = True

        # применяем компенсацию IF только к тому, что реально отдали
        if self._mix_w:
            k = np.arange(out.size, dtype=np.float64)
            ph = self._mix_phase + self._mix_w * k
            out *= np.exp(-1j * ph).astype(np.complex64, copy=False)
            self._mix_phase = (self._mix_phase + self._mix_w * out.size) % (2*np.pi)

        return out

    def stop(self) -> None:
        # позволяет «перезапустить» файл вручную при повторном запуске графика
        self._pos = 0
        self._eof = False

    def close(self) -> None:
        # Ничего не делаем - нет ресурсов для закрытия
        pass


class FileWaitBackend(SDRBackend):
    """Пустой файловый бэкенд для режима file_wait - не читает данные, ожидает выбора файла в UI"""

    def __init__(self, **kw):
        # FIXED: для file бэкенда принудительно отключаем if_offset_hz
        _ = kw.pop("if_offset_hz", 0.0)  # удаляем из kwargs, но игнорируем значение
        self._if_offset_hz = 0.0  # принудительно устанавливаем в 0.0

        super().__init__(**kw)
        # NEW: для файла реальный Fs = заявленному sample_rate_sps
        self.actual_sample_rate_sps = float(self.sample_rate_sps)

        # калибровка
        self.calib_offset_db = SDR_CALIB_OFFSETS_DB.get("file", 0.0)

    def get_status(self) -> Dict[str, Any]:
        s = super().get_status()
        s.update({
            "driver": "file",
            "device_info": "Ожидание выбора файла",
            "file_path": None,
            "if_offset_hz": float(getattr(self, "_if_offset_hz", 0.0)),
            "eof": False,
        })
        return s

    def get_calib_offset_db(self) -> float:
        return SDR_CALIB_OFFSETS_DB.get("file", 0.0)

    def read(self, nsamps: int) -> np.ndarray:
        # Всегда возвращаем пустой массив - ожидаем выбора файла
        return np.empty(0, dtype=np.complex64)

    def stop(self) -> None:
        # Ничего не делаем - нет файла для остановки
        pass

    def close(self) -> None:
        # Ничего не делаем - нет ресурсов для закрытия
        pass


# --- NEW: safe_make_backend -----------------------------------------------
def safe_make_backend(
    name: str,
    *,
    on_fail: str = "raise",          # "raise" | "file_wait" | "none"
    fallback_args: dict | None = None,
    **kwargs,
):
    """
    Безопасный конструктор backend'ов.
    - on_fail="raise": поведение как у make_backend (по умолчанию).
    - on_fail="file_wait": при отсутствии SDR вернёт file-backend с пустым путём (ожидание выбора файла в UI).
    - on_fail="none": при ошибке вернёт None.

    Параметры kwargs пробрасываются в make_backend(...). Для file-backend
    из kwargs используются sample_rate и опционально if_offset_hz.
    """
    try:
        return make_backend(name, **kwargs)

    except RuntimeError as e:
        msg = str(e).lower()
        devices_not_found = ("устройства не найдены" in msg) or ("devices not found" in msg)

        if not devices_not_found:
            # другие ошибки не глотаем
            if on_fail == "raise":
                raise
            elif on_fail == "none":
                return None
            elif on_fail == "file_wait":
                # даже если ошибка не «нет устройств», пробуем мягкий фолбэк
                pass

        if on_fail == "raise":
            raise

        if on_fail == "none":
            return None

        if on_fail == "file_wait":
            # В режиме file_wait не пытаемся создавать backend с пустым путём,
            # а сразу возвращаем None для режима ожидания файла
            return None

        # На всякий случай (неизвестный on_fail) — ведём себя как raise
        raise
# --- END NEW --------------------------------------------------------------



def extract_sigmf_metadata(file_path: str):
    """
    Извлекает метаданные из .sigmf-meta файла для автоматической настройки.

    Args:
        file_path: Путь к .sigmf-meta файлу или связанному .sigmf-data файлу

    Returns:
        tuple: (sample_rate, center_freq, data_file_path) или (None, None, None) если не SigMF
    """
    if not file_path:
        return None, None, None

    # Normalize path and check for SigMF
    path_lower = file_path.lower()

    if path_lower.endswith('.sigmf-meta'):
        meta_path = file_path
        data_path = file_path.replace('.sigmf-meta', '.sigmf-data')
    elif path_lower.endswith('.sigmf-data'):
        meta_path = file_path.replace('.sigmf-data', '.sigmf-meta')
        data_path = file_path
    else:
        # Not a SigMF file
        return None, None, None

    # Check if both files exist
    if not (os.path.exists(meta_path) and os.path.exists(data_path)):
        return None, None, None

    try:
        if sigio_open_iq is not None:
            # Use sigio to parse SigMF
            reader = sigio_open_iq(meta_path, strict=False)
            status = reader.get_status()
            reader.stop()

            sample_rate = status.get('input_sample_rate_sps')
            center_freq = status.get('center_freq_hz')

            if sample_rate and center_freq:
                return float(sample_rate), float(center_freq), data_path
        else:
            # Fallback to manual JSON parsing
            import json

            with open(meta_path, 'r', encoding='utf-8') as f:
                sigmf_data = json.load(f)

            global_info = sigmf_data.get('global', {})
            captures = sigmf_data.get('captures', [])

            sample_rate = global_info.get('sample_rate')
            center_freq = captures[0].get('frequency', 0.0) if captures else 0.0

            if sample_rate and center_freq:
                return float(sample_rate), float(center_freq), data_path

    except Exception as e:
        print(f"Warning: Failed to parse SigMF metadata: {e}")

    return None, None, None



def make_backend(name: str,
                 *,
                 sample_rate: float,
                 center_freq: float,
                 gain_db: Optional[float] = None,
                 agc: bool = False,
                 corr_ppm: int = 0,
                 device_args: Optional[Dict[str, Any]] = None,
                 **extras) -> SDRBackend:
    """
    Фабрика бэкендов.

    name:
      - 'soapy_rtl' | 'soapy_hackrf' | 'soapy_airspy' | 'soapy_sdrplay'  -> SoapyBackend
      - 'file' -> FilePlaybackBackend (нужен extras['path'])
      - 'auto' -> autodetect_soapy_driver()  (если доступно)

    Пример:
        backend = make_backend(
            'soapy_rtl',
            sample_rate=1_000_000.0,
            center_freq=121_500_000.0,
            gain_db=44.0,
            agc=False,
            corr_ppm=0,
        )
    """
    name = (name or "").lower()

    if name == "auto":
        name = autodetect_soapy_driver()

    if name.startswith("soapy"):
        drv = {
            "soapy_rtl":     {"driver": "rtlsdr"},
            "soapy_hackrf":  {"driver": "hackrf"},
            "soapy_airspy":  {"driver": "airspy"},
            "soapy_sdrplay": {"driver": "sdrplay"},
        }.get(name, None)
        if drv is None and device_args is None:
            raise ValueError(f"Неизвестный Soapy-бэкенд '{name}'; задайте device_args={{...}}")
        return SoapyBackend(
            device_args=device_args or drv,
            sample_rate=sample_rate,
            center_freq=center_freq,
            gain_db=gain_db,
            agc=agc,
            corr_ppm=corr_ppm,
            **extras  # Передаём if_offset_hz и другие параметры
        )

    if name == "file":
        # 1) extras['path']  2) BACKEND_ARGS как str/dict  3) env IQ_FILE_PATH
        path = extras.get("path")
        if not path and device_args is not None:
            if isinstance(device_args, str):
                path = device_args
            elif isinstance(device_args, dict):
                path = (device_args.get("path")
                        or device_args.get("file")
                        or device_args.get("uri"))
        if not path:

            path = os.getenv("IQ_FILE_PATH")

        if not path:
            # Разрешаем пустой путь для режима file_wait - создаем "пустой" файловый бэкенд
            return FileWaitBackend(
                sample_rate=sample_rate,
                center_freq=center_freq,
                gain_db=gain_db,
                agc=agc,
                corr_ppm=corr_ppm,
                if_offset_hz=0.0  # принудительно отключено для file режима
            )

        # NEW: Check for SigMF files and extract metadata automatically
        sigmf_sr, sigmf_cf, sigmf_data_path = extract_sigmf_metadata(path)

        if sigmf_sr is not None and sigmf_cf is not None:
            # Use SigMF metadata values, override passed parameters
            actual_sample_rate = sigmf_sr
            actual_center_freq = sigmf_cf
            actual_path = sigmf_data_path
            print(f"SigMF detected: SR={actual_sample_rate}, CF={actual_center_freq}")
        else:
            # Regular .cf32 file - use passed parameters
            actual_sample_rate = sample_rate
            actual_center_freq = center_freq
            actual_path = path

        # Уважать if_offset_hz, если он передан (device_args/extras)
        if_offset_hz = 0.0
        if isinstance(device_args, dict):
            if_offset_hz = float(device_args.get("if_offset_hz",
                              device_args.get("IF_OFFSET_HZ", if_offset_hz)))
        if isinstance(extras, dict):
            if_offset_hz = float(extras.get("if_offset_hz", if_offset_hz))

        return FilePlaybackBackend(
            path=actual_path,
            sample_rate=actual_sample_rate,
            center_freq=actual_center_freq,
            gain_db=gain_db,
            agc=agc,
            corr_ppm=corr_ppm,
            if_offset_hz=if_offset_hz,
        )

    if name in ("tek_rsa", "rsa306", "rsa"):
        return TekRSABackend(
            sample_rate=sample_rate,
            center_freq=center_freq,
            gain_db=gain_db,
            agc=agc,
            corr_ppm=corr_ppm,
            ref_level_dbm=float(extras.get("ref_level_dbm", 0.0)),
            req_bw_hz=float(extras.get("req_bw_hz", 500e3)),
            if_offset_hz=float(extras.get("if_offset_hz", 0.0)),
            dll_path=extras.get("dll_path", r"C:/Tektronix/RSA_API/lib/x64/RSA_API.dll"),
            timeout_s=float(extras.get("timeout_s", 0.2)),
        )

    raise ValueError(f"Backend '{name}' не поддержан")

def autodetect_soapy_driver() -> str:
    """
    Робастный автодетект при BACKEND_NAME='auto'.

    Порядок по умолчанию:
      1) Soapy: rtlsdr -> hackrf -> airspy -> sdrplay
      2) Tektronix RSA306 через RSA_API (если Soapy-устройств нет)

    Можно переопределить порядок Soapy через BACKEND_PREFERRED_ORDER,
    например: "rtl,hackrf,airspy,sdrplay".
    """
    import os, subprocess, shlex # lazy imports: используются только в автодетекте

    # --- 1) Попытаться найти через SoapySDR ---
    try:
        import SoapySDR  # type: ignore

        # Карта имён Soapy -> имена фабрики
        mapping = {
            "rtlsdr": "soapy_rtl",
            "hackrf": "soapy_hackrf",
            "airspy": "soapy_airspy",
            "sdrplay": "soapy_sdrplay",
        }

        # 0) Общая enumeration()
        try:
            lst = SoapySDR.Device.enumerate()
            for dev in lst:
                drv = (dev.get("driver") or "").lower()
                if drv in mapping:
                    return mapping[drv]
        except Exception:
            pass

        # 1) Точечная enumeration({driver:...}) в заданном порядке
        preferred = os.getenv("BACKEND_PREFERRED_ORDER", "rtl,hackrf,airspy,sdrplay")
        order = [x.strip().lower() for x in preferred.split(",") if x.strip()]
        drv_map = {"rtl":"rtlsdr","rtlsdr":"rtlsdr","hackrf":"hackrf","airspy":"airspy","sdrplay":"sdrplay","rsp":"sdrplay"}

        for key in order:
            drv = drv_map.get(key)
            if not drv:
                continue
            try:
                if SoapySDR.Device.enumerate({"driver": drv}):
                    return mapping[drv]
            except Exception:
                continue

        # 2) Явное открытие (иногда enumerate пустой, а open работает)
        for key in order:
            drv = drv_map.get(key)
            if not drv:
                continue
            try:
                dev = SoapySDR.Device({"driver": drv})
                try:
                    del dev
                except Exception:
                    pass
                return mapping[drv]
            except Exception:
                continue

        # 3) Фолбэк: SoapySDRUtil --find
        try:
            out = subprocess.check_output(shlex.split("SoapySDRUtil --find"), text=True, timeout=5)
            found = [ln.split("=",1)[1].strip().lower()
                     for ln in out.splitlines() if ln.strip().lower().startswith("driver =")]
            for key in order:
                drv = drv_map.get(key)
                if drv and drv in found:
                    return mapping[drv]
        except Exception:
            pass
    except Exception:
        # SoapySDR не установлен/не грузится — перейдём к RSA
        pass

    # --- 2) Попытаться найти Tektronix RSA306 через RSA_API ---
    # dll_path можно переопределить через переменную окружения RSA_API_DLL
    dll_path = os.getenv("RSA_API_DLL", r"C:/Tektronix/RSA_API/lib/x64/RSA_API.dll")
    try:

        if os.path.exists(dll_path):
            L = ct.CDLL(dll_path)
            # DEVICE_SearchIntW: int *numDevs, int **ids, wchar_t **sn, wchar_t **type
            num = ct.c_int(0)
            ids = ct.POINTER(ct.c_int)()
            sns = ct.POINTER(ct.c_wchar_p)()
            tys = ct.POINTER(ct.c_wchar_p)()
            L.DEVICE_SearchIntW.argtypes = [ct.POINTER(ct.c_int),
                                            ct.POINTER(ct.POINTER(ct.c_int)),
                                            ct.POINTER(ct.POINTER(ct.c_wchar_p)),
                                            ct.POINTER(ct.POINTER(ct.c_wchar_p))]
            L.DEVICE_SearchIntW.restype = ct.c_int
            rc = L.DEVICE_SearchIntW(ct.byref(num), ct.byref(ids), ct.byref(sns), ct.byref(tys))
            if rc == 0 and num.value > 0:
                # Устройство RSA найдено
                return "rsa"
    except Exception:
        pass

    # --- Ничего не нашли ---
    raise RuntimeError("Soapy/RSA: устройства не найдены (rtlsdr/hackrf/airspy/sdrplay/rsa). "
                       "Укажи BACKEND_NAME явно или проверь PATH/плагины/драйверы.")


# === Tektronix RSA306 Backend (COMBAT: only addition) ===

class RSA_API_Wrapper:
    IQSOD_CLIENT = 0
    IQSODT_SINGLE = 0

    def __init__(self, dll_path: str):
        if not dll_path:
            raise FileNotFoundError("DLL path not specified")
        self.lib = ct.CDLL(dll_path)

        # --- сигнатуры (как в RSA306_RMS_stream_log_ok.py) ---
        L = self.lib
        L.DEVICE_SearchIntW.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.POINTER(ct.c_int)),
                                        ct.POINTER(ct.POINTER(ct.c_wchar_p)), ct.POINTER(ct.POINTER(ct.c_wchar_p))]
        L.DEVICE_SearchIntW.restype = ct.c_int
        L.DEVICE_Connect.argtypes, L.DEVICE_Connect.restype = [ct.c_int], ct.c_int
        L.DEVICE_Disconnect.argtypes, L.DEVICE_Disconnect.restype = [], ct.c_int
        L.DEVICE_Run.argtypes, L.DEVICE_Run.restype = [], ct.c_int
        L.DEVICE_Stop.argtypes, L.DEVICE_Stop.restype = [], ct.c_int
        L.CONFIG_SetCenterFreq.argtypes, L.CONFIG_SetCenterFreq.restype = [ct.c_double], ct.c_int
        L.CONFIG_SetReferenceLevel.argtypes, L.CONFIG_SetReferenceLevel.restype = [ct.c_double], ct.c_int
        L.IQSTREAM_SetAcqBandwidth.argtypes, L.IQSTREAM_SetAcqBandwidth.restype = [ct.c_double], ct.c_int
        L.IQSTREAM_GetAcqParameters.argtypes, L.IQSTREAM_GetAcqParameters.restype = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)], ct.c_int
        L.IQSTREAM_SetOutputConfiguration.argtypes, L.IQSTREAM_SetOutputConfiguration.restype = [ct.c_int, ct.c_int], ct.c_int
        L.IQSTREAM_SetIQDataBufferSize.argtypes, L.IQSTREAM_SetIQDataBufferSize.restype = [ct.c_int], ct.c_int
        L.IQSTREAM_GetIQDataBufferSize.argtypes, L.IQSTREAM_GetIQDataBufferSize.restype = [ct.POINTER(ct.c_int)], ct.c_int
        try:
            L.IQSTREAM_SetEnable.argtypes = [ct.c_int]
            L.IQSTREAM_SetEnable.restype = ct.c_int
            self.has_set_enable = True
        except AttributeError:
            self.has_set_enable = False
        L.IQSTREAM_Start.argtypes, L.IQSTREAM_Start.restype = [], ct.c_int
        L.IQSTREAM_Stop.argtypes, L.IQSTREAM_Stop.restype = [], ct.c_int
        L.IQSTREAM_GetIQData.argtypes, L.IQSTREAM_GetIQData.restype = [ct.c_void_p, ct.POINTER(ct.c_int), ct.c_void_p], ct.c_int

    def connect_first(self):
        n = ct.c_int(0)
        ids = ct.POINTER(ct.c_int)()
        sns = ct.POINTER(ct.c_wchar_p)()
        ty  = ct.POINTER(ct.c_wchar_p)()
        st = self.lib.DEVICE_SearchIntW(ct.byref(n), ct.byref(ids), ct.byref(sns), ct.byref(ty))
        if st != 0 or n.value <= 0:
            raise RuntimeError("RSA306 not found")
        did = ids[0]
        self.lib.DEVICE_Connect(ct.c_int(did))
        return did, sns[0], ty[0]

    def configure_common(self, center_hz: float, ref_dbm: float):
        self.lib.CONFIG_SetCenterFreq(ct.c_double(center_hz))
        self.lib.CONFIG_SetReferenceLevel(ct.c_double(ref_dbm))

    def iqstream_setup(self, bw_req_hz: float, req_block_sec: float = 0.05):
        self.lib.IQSTREAM_SetAcqBandwidth(ct.c_double(bw_req_hz))
        self.lib.IQSTREAM_SetOutputConfiguration(ct.c_int(self.IQSOD_CLIENT), ct.c_int(self.IQSODT_SINGLE))
        bw_act = ct.c_double(0.0); fs = ct.c_double(0.0)
        self.lib.IQSTREAM_GetAcqParameters(ct.byref(bw_act), ct.byref(fs))
        req_pairs = max(2048, int(fs.value * req_block_sec))
        self.lib.IQSTREAM_SetIQDataBufferSize(ct.c_int(req_pairs))
        buf_pairs = ct.c_int(0)
        self.lib.IQSTREAM_GetIQDataBufferSize(ct.byref(buf_pairs))
        return int(buf_pairs.value), bw_act.value, fs.value

    def iqstream_enable(self, enable=True):
        if self.has_set_enable:
            self.lib.IQSTREAM_SetEnable(ct.c_int(1 if enable else 0))

    def device_run(self):  self.lib.DEVICE_Run()
    def device_stop(self): self.lib.DEVICE_Stop()
    def iqstream_start(self): self.lib.IQSTREAM_Start()
    def iqstream_stop(self):  self.lib.IQSTREAM_Stop()

    def iqstream_get_block(self, buf_pairs: int):

        n = ct.c_int(0)
        buf = (ct.c_float * (2 * buf_pairs))()
        st = self.lib.IQSTREAM_GetIQData(ct.cast(buf, ct.c_void_p), ct.byref(n), None)
        if st != 0 or n.value <= 0:
            return np.empty(0, dtype=np.complex64)
        arr = np.frombuffer(buf, dtype=np.float32, count=2 * n.value).reshape(-1, 2)
        return (arr[:, 0] + 1j*arr[:, 1]).astype(np.complex64)

class TekRSABackend(SDRBackend):
    def __init__(self,
                 sample_rate: float,
                 center_freq: float,
                 gain_db=None,
                 agc=False,
                 corr_ppm=0,
                 *,
                 ref_level_dbm: float = -0.0,
                 req_bw_hz: float = 500e3,
                 if_offset_hz: float = 0.0,
                 dll_path: str = r"C:/Tektronix/RSA_API/lib/x64/RSA_API.dll",
                 timeout_s: float = 0.2):
        super().__init__(sample_rate, center_freq, gain_db, agc, corr_ppm)

        self._rsa = RSA_API_Wrapper(dll_path)
        did, sn, typ = self._rsa.connect_first()
        self.device_info = f"{typ} SN={sn}"
        self._rsa.configure_common(center_freq, ref_level_dbm)
        self.buf_pairs, bw_act, fs = self._rsa.iqstream_setup(req_bw_hz)
        self._target_sample_rate = float(sample_rate)

        self._timeout_s = float(timeout_s)
        self._stop_evt = threading.Event()
        self._queue = deque()
        self._lock = threading.Lock()

        # === Диагностика конфигурации ===
        freq = ct.c_double(0.0)
        self._rsa.lib.CONFIG_GetCenterFreq(ct.byref(freq))
        ref = ct.c_double(0.0)
        try:
            self._rsa.lib.CONFIG_GetReferenceLevel(ct.byref(ref))
        except Exception:
            ref.value = ref_level_dbm
        bw = ct.c_double(0.0)
        fs_val = ct.c_double(0.0)
        self._rsa.lib.IQSTREAM_GetAcqParameters(ct.byref(bw), ct.byref(fs_val))
        """
        print("=== RSA306 CONFIG ===")
        print(f" Center Freq set      : {center_freq/1e6:.6f} MHz (actual {freq.value/1e6:.6f} MHz)")
        print(f" Ref Level            : {ref.value:.1f} dBm")
        print(f" Requested BW         : {req_bw_hz/1e3:.1f} kHz")
        print(f" Actual BW            : {bw.value/1e3:.1f} kHz")
        print(f" Actual Sample Rate   : {fs_val.value:.2f} Sa/s")
        print(f" Buffer pairs         : {self.buf_pairs}")
        print("======================")
        """
        self._center_freq_act_hz = float(freq.value)
        self._ref_level_act_dbm = float(ref.value)
        self._bw_act_hz = float(bw.value)
        self.actual_sample_rate_sps = float(fs_val.value)  # наружу отдаём без ресемпла

        def _worker():
            self._rsa.iqstream_enable(True)
            self._rsa.device_run()
            self._rsa.iqstream_start()
            while not self._stop_evt.is_set():
                iq = self._rsa.iqstream_get_block(self.buf_pairs)
                if iq.size > 0:
                    with self._lock:
                        self._queue.append(iq)
                else:
                    time.sleep(self._timeout_s)
            try:
                self._rsa.iqstream_stop()
                self._rsa.device_stop()
                self._rsa.iqstream_enable(False)
            except Exception:
                pass

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def read(self, nsamps: int):

        if nsamps <= 0:
            return np.empty(0, dtype=np.complex64)
        with self._lock:
            if not self._queue:
                return np.empty(0, dtype=np.complex64)
            x = self._queue.popleft()
        return x[:nsamps].astype(np.complex64, copy=False)

    def get_status(self) -> Dict[str, Any]:
        s = super().get_status()
        s.update({
            "driver": "rsa306",
            "device_info": getattr(self, "device_info", None),
            "actual_center_freq_hz": float(getattr(self, "_center_freq_act_hz", self.center_freq_hz)),
            "ref_level_dbm": float(getattr(self, "_ref_level_act_dbm", 0.0)),
            "bandwidth_hz": float(getattr(self, "_bw_act_hz", 0.0)),
        })
        return s

    def stop(self):
        self._stop_evt.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._rsa.iqstream_stop()
        except Exception:
            pass
        try:
            self._rsa.device_stop()
        except Exception:
            pass
        try:
            self._rsa.iqstream_enable(False)
        except Exception:
            pass
        try:
            self._rsa.lib.DEVICE_Disconnect()
        except Exception:
            pass

    def close(self) -> None:
        # Ничего не делаем - нет ресурсов для закрытия
        pass

    def get_calib_offset_db(self) -> float:
        return SDR_CALIB_OFFSETS_DB.get("rsa306", 0.0)