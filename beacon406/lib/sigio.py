# lib/sigio.py
from __future__ import annotations
import os
import json
import math
import datetime as _dt
from typing import Iterator, Literal, Optional

import numpy as np

try:
    # pip install sigmf
    from sigmf import SigMFFile
except Exception:
    SigMFFile = None  # чтение/запись SigMF отключится, если пакет не установлен


# ----------------------------- helpers -----------------------------

def _now_utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _infer_sigmf_pair(path: str) -> tuple[str, str]:
    """Return (meta_path, data_path) for given .sigmf-meta or .sigmf-data."""
    base = path.replace(".sigmf-meta", "").replace(".sigmf-data", "")
    return base + ".sigmf-meta", base + ".sigmf-data"

def _dtype_scale_from_sigmf(datatype: str):
    """Map SigMF datatype to numpy dtype and scale -> complex64."""
    dt = datatype.lower()
    if dt == "cf32_le":
        return np.complex64, 1.0, "complex"
    if dt in ("ci16_le", "ci16"):
        return np.int16, 1.0 / 32768.0, "interleaved_iq"
    if dt in ("ci8", "ci8_le"):
        return np.int8, 1.0 / 128.0, "interleaved_iq"
    raise ValueError(f"Unsupported SigMF datatype: {datatype}")

def _make_lpf_kaiser(decim: int, taps_per_phase: int = 32, beta: float = 8.6) -> np.ndarray:
    """Kaiser-windowed sinc LPF for integer decimation; cutoff = 0.45/decim (Nyquist guard)."""
    if decim <= 1:
        return np.array([1.0], dtype=np.float32)
    num_taps = decim * taps_per_phase
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    fc = 0.45 / decim  # normalized to Fs=1.0
    h = np.sinc(2 * fc * n)
    w = np.kaiser(num_taps, beta)
    h *= w
    h /= np.sum(h)
    return h.astype(np.float32)

def _polyphase_decimate(x: np.ndarray, h: np.ndarray, decim: int, state: dict) -> np.ndarray:
    """Streaming polyphase decimation for complex64; carries state['tail']."""
    if decim <= 1:
        return x
    tail = state.get("tail", np.zeros(0, dtype=x.dtype))
    x_cat = np.concatenate([tail, x])
    # Convolve separately real/imag to avoid complex->float path surprises on some builds
    y_r = np.convolve(x_cat.real, h, mode="full")
    y_i = np.convolve(x_cat.imag, h, mode="full")
    y = y_r + 1j * y_i
    # Group delay
    gd = (len(h) - 1) // 2
    if y.size <= gd:
        # Not enough for one output; keep tail growing
        state["tail"] = x_cat
        return np.zeros(0, dtype=np.complex64)
    y = y[gd:]  # align
    out = y[::decim].astype(np.complex64, copy=False)
    # Save tail: last (len(h)-1) samples of x_cat
    keep = len(h) - 1
    if keep > 0:
        state["tail"] = x_cat[-keep:]
    else:
        state["tail"] = np.zeros(0, dtype=x.dtype)
    return out


# ----------------------------- Reader -----------------------------

class Reader:
    def __init__(
        self,
        *,
        source: str,
        input_sample_rate_sps: float,
        center_freq_hz: float,
        fmt: str,
        data_path: str,
        meta_min: Optional[dict],
        target_fs: Optional[float] = None,
    ) -> None:
        self._source = source
        self._in_fs = float(input_sample_rate_sps)
        self._cf = float(center_freq_hz)
        self._fmt = fmt  # "sigmf" | "cf32"
        self._data_path = data_path
        self._meta_min = meta_min
        self._total_in = 0
        self._total_out = 0
        self._eof = False

        # Decimation
        if target_fs and target_fs > 0 and target_fs < self._in_fs:
            decim_float = self._in_fs / target_fs
            decim = int(round(decim_float))
            if abs(decim - decim_float) / decim_float > 1e-6:
                # целочисленная децимация обязательна в этой версии
                self._decim = 1
                self._out_fs = self._in_fs
            else:
                self._decim = max(1, decim)
                self._out_fs = self._in_fs / self._decim
        else:
            self._decim = 1
            self._out_fs = self._in_fs

        self._lpf = _make_lpf_kaiser(self._decim)
        self._fir_state = {}  # carries 'tail'

        # Backend-specific init
        if self._fmt == "cf32":
            self._mm = np.memmap(self._data_path, dtype=np.complex64, mode="r")
            self._pos = 0  # complex index
        elif self._fmt == "sigmf_cf32":
            self._mm = np.memmap(self._data_path, dtype=np.complex64, mode="r")
            self._pos = 0
        elif self._fmt.startswith("sigmf_ci"):
            self._fh = open(self._data_path, "rb", buffering=0)
            self._raw_dtype = np.int16 if "ci16" in self._fmt else np.int8
            self._scale = 1.0 / (32768.0 if self._raw_dtype is np.int16 else 128.0)
        else:
            raise RuntimeError(f"Unknown reader fmt: {self._fmt}")

    # Public API
    def read(self, max_complex: int) -> np.ndarray:
        if self._eof or max_complex <= 0:
            return np.zeros(0, dtype=np.complex64)

        # Aim to output <= max_complex after decimation
        need_in = max_complex * self._decim

        if self._fmt in ("cf32", "sigmf_cf32"):
            # memmap path
            end = min(self._pos + int(need_in), self._mm.shape[0])
            chunk = self._mm[self._pos:end]
            self._pos = end
            if end >= self._mm.shape[0]:
                self._eof = True
            xin = chunk.astype(np.complex64, copy=False)
        else:
            # streaming CI* -> CF32
            # read interleaved I/Q of 2*need_in integers
            count = int(need_in) * 2
            raw = np.fromfile(self._fh, dtype=self._raw_dtype, count=count)
            if raw.size == 0:
                self._eof = True
                xin = np.zeros(0, dtype=np.complex64)
            else:
                if raw.size % 2 != 0:
                    raw = raw[:-1]
                iq = raw.astype(np.float32) * self._scale
                xin = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64, copy=False)
                if xin.size < need_in:
                    self._eof = True

        self._total_in += xin.size

        # decimate if required
        xout = _polyphase_decimate(xin, self._lpf, self._decim, self._fir_state)
        self._total_out += xout.size
        return xout

    def get_status(self) -> dict:
        return {
            "source": "file-sigmf" if self._fmt.startswith("sigmf") else "file-cf32",
            "input_sample_rate_sps": self._in_fs,
            "output_sample_rate_sps": self._out_fs,
            "center_freq_hz": self._cf,
            "decim_factor": int(self._decim),
            "total_samples_in": int(self._total_in),
            "total_samples_out": int(self._total_out),
            "eof": bool(self._eof),
            "file_path": self._data_path,
            "sigmf_meta": self._meta_min,
        }

    def stop(self) -> None:
        try:
            if hasattr(self, "_fh") and self._fh:
                self._fh.close()
        except Exception:
            pass


# ----------------------------- Factory -----------------------------

def open_iq(path: str, target_fs: float | None = None, strict: bool = True) -> Reader:
    p = path.lower()
    if p.endswith(".cf32"):
        # Legacy cf32: Fs/Fc должны приходить от вызова (через параметры backends)
        # Здесь мы не знаем Fs/Fc, поэтому читаем метаданные из соседнего .json (если есть)
        meta_min = None
        meta_guess = {}
        meta_sidecar = path + ".json"
        if os.path.exists(meta_sidecar):
            try:
                meta_guess = json.loads(open(meta_sidecar, "r", encoding="utf-8").read())
            except Exception:
                meta_guess = {}
        fs = float(meta_guess.get("sample_rate", 1_000_000.0))
        fc = float(meta_guess.get("center_freq_hz", 0.0))
        return Reader(
            source="file-cf32",
            input_sample_rate_sps=fs,
            center_freq_hz=fc,
            fmt="cf32",
            data_path=path,
            meta_min=None,
            target_fs=target_fs,
        )

    # SigMF pair
    meta_path, data_path = _infer_sigmf_pair(path)
    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        raise FileNotFoundError(f"SigMF pair not found: {meta_path} / {data_path}")

    # Load meta (prefer sigmf-python if present, else bare JSON)
    global_meta = {}
    captures = []
    try:
        if SigMFFile is not None:
            s = SigMFFile.fromfile(meta_path)
            s.validate()  # may raise
            global_meta = s.get_global_info()
            captures = s.get_captures()
        else:
            global_meta = json.loads(open(meta_path, "r", encoding="utf-8").read()).get("global", {})
            cap_all = json.loads(open(meta_path, "r", encoding="utf-8").read()).get("captures", [])
            captures = cap_all or []
    except Exception as e:
        if strict:
            raise
        # fallback: parse as JSON with minimal checks
        try:
            doc = json.loads(open(meta_path, "r", encoding="utf-8").read())
            global_meta = doc.get("global", {})
            captures = doc.get("captures", [])
        except Exception:
            global_meta = {}
            captures = []

    datatype = (global_meta.get("datatype") or "").lower()
    if strict and not datatype:
        raise ValueError("SigMF global.datatype required")

    fs = float(global_meta.get("sample_rate") or 0)
    if strict and fs <= 0:
        raise ValueError("SigMF global.sample_rate required and > 0")

    fc = 0.0
    if captures and isinstance(captures, list):
        fc = float(captures[0].get("frequency", 0.0))
    if strict and fc == 0.0:
        raise ValueError("SigMF captures[0].frequency required")

    # choose fmt
    np_dtype, scale, mode = _dtype_scale_from_sigmf(datatype)
    if datatype == "cf32_le":
        fmt = "sigmf_cf32"
    elif datatype.startswith("ci16"):
        fmt = "sigmf_ci16"
    elif datatype.startswith("ci8"):
        fmt = "sigmf_ci8"
    else:
        raise ValueError(f"Unsupported SigMF datatype: {datatype}")

    meta_min = {
        "datatype": datatype,
        "sample_rate": fs,
        "center_freq_hz": fc,
        "extras": global_meta.get("extras", {}),
    }

    return Reader(
        source="file-sigmf",
        input_sample_rate_sps=fs,
        center_freq_hz=fc,
        fmt=fmt,
        data_path=data_path,
        meta_min=meta_min,
        target_fs=target_fs,
    )


# ----------------------------- Writers -----------------------------

def save_sigmf_raw(
    path_base: str,
    data_any: np.ndarray,                # interleaved I/Q int8 or int16
    sample_rate: float,
    center_freq_hz: float,
    *,
    datatype: Literal["ci8", "ci16_le", "ci16", "ci8_le"],
    hw: Optional[str] = None,
    datetime_utc: Optional[str] = None,
    extras: Optional[dict] = None,
    captures: Optional[list[dict]] = None,
    annotations: Optional[list[dict]] = None,
) -> tuple[str, str]:
    """
    Write RAW SigMF with integer I/Q payload (no float expansion).
    """
    if SigMFFile is None:
        raise RuntimeError("sigmf-python not installed")

    dt = (datetime_utc or _now_utc_iso())
    meta_path = f"{path_base}.sigmf-meta"
    data_path = f"{path_base}.sigmf-data"

    # Write data (binary)
    arr = np.asarray(data_any)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("save_sigmf_raw expects integer I/Q array (interleaved)")
    arr.tofile(data_path)

    # Build SigMF
    s = SigMFFile(
        data_file=os.path.basename(data_path),
        global_info={
            "datatype": datatype,
            "sample_rate": float(sample_rate),
            "hw": hw or "",
            "version": "1.0.0",
            "extras": extras or {},
        },
    )
    s.add_capture(0, metadata={"frequency": float(center_freq_hz), "datetime": dt})
    if captures:
        for c in captures:
            if not isinstance(c, dict):
                continue
            s.add_capture(int(c.get("sample_start", 0)), metadata={k: v for k, v in c.items() if k != "sample_start"})
    if annotations:
        for a in annotations:
            s.add_annotation(a)

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(s.dumps())

    return meta_path, data_path


def save_sigmf_cf32(
    path_base: str,
    cf32: np.ndarray,                   # complex64
    sample_rate: float,
    center_freq_hz: float,
    *,
    pipeline: Optional[Literal["post-decim", "post-cfo", "pre-demod"]] = None,
    hw: Optional[str] = None,
    datetime_utc: Optional[str] = None,
    extras: Optional[dict] = None,
    captures: Optional[list[dict]] = None,
    annotations: Optional[list[dict]] = None,
) -> tuple[str, str]:
    """Write SigMF with cf32_le payload (robust ordering: datatype→data_file)."""
    if SigMFFile is None:
        raise RuntimeError("sigmf-python not installed")

    dt = (datetime_utc or _now_utc_iso())
    meta_path = f"{path_base}.sigmf-meta"
    data_path = f"{path_base}.sigmf-data"

    # 1) Пишем бинарные данные (cf32_le)
    arr = np.asarray(cf32).astype(np.complex64, copy=False)
    arr.tofile(data_path)

    # 2) Сборка extras (чтобы не класть None)
    ex = dict(extras or {})
    if pipeline:
        ex["pipeline"] = pipeline
    if "version" in ex and ex["version"] is None:
        del ex["version"]

    # 3) ВАЖНО: сначала задать global_info (включая datatype), затем data_file
    s = SigMFFile()  # не передаём data_file в конструктор — избежим ошибок с порядком

    # global_info с обязательным DATATYPE_KEY
    s.set_global_info({
        "datatype": "cf32_le",                    # <- ключевой порядок!
        "sample_rate": float(sample_rate),
        "hw": hw or "",
        "version": "1.0.0",
        "extras": ex,
    })

    # после того как datatype уже в global — указываем data_file
    s.set_data_file(os.path.basename(data_path))

    # capture c частотой и датой
    s.add_capture(0, metadata={"frequency": float(center_freq_hz), "datetime": dt})

    # дополнительные captures (если переданы)
    if captures:
        for c in captures:
            if not isinstance(c, dict):
                continue
            s.add_capture(int(c.get("sample_start", 0)),
                          metadata={k: v for k, v in c.items() if k != "sample_start"})

    # аннотации (если есть)
    if annotations:
        for a in annotations:
            s.add_annotation(a)

    # 4) Записываем .sigmf-meta
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(s.dumps())

    return meta_path, data_path


def save_cf32_legacy(path: str, cf32: np.ndarray) -> str:
    """Write legacy .cf32 (raw complex64)."""
    arr = np.asarray(cf32).astype(np.complex64, copy=False)
    arr.tofile(path)
    return path
