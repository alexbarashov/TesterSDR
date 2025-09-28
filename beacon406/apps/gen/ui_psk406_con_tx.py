import numpy as np
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.logger import get_logger
log = get_logger(__name__)
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont  # добавь вверху файла рядом с остальными импортами

sys.path.insert(0, str(Path(__file__).parent))  # Добавляем папку gen для локальных модулей
import psk406_msg_gen as gen
import backend_hackrf_tx as tx

PREFS_PATH = Path(__file__).with_name("ui_prefs.json")

DEF = {
    "target_signal_hz": 406_037_000,
    "if_offset_hz": -37_000,
    "freq_corr_hz": 750,
    "tx_gain_db": 40,
    "phase_low": -1.1,
    "phase_high": 1.1,
    "front": 100,
    "hex_message": "FFFED080020000007FDFFB0020B783E0F66C",
    "FILE_IQ": r"C:\work\TesterSDR\captures\UI_iq_1m.cf32",
    "repeat": "1",
    "gap_s": 4.0,
    "fs_gen": 1_000_000,
    "fs_tx": 2_000_000,
    "hw_amp": False,
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PSK-406 TX (HackRF) — minimal UI")
        self.resizable(False, False)
        
        # 1) общий масштаб (1.0 = 100%). На ноутбуках с HiDPI удобно 1.25–1.5
        self.tk.call("tk", "scaling", 1.5)   # увеличен масштаб для лучшей читаемости

        # 2) увеличить системные шрифты Tk
        default_font = tkfont.nametofont("TkDefaultFont")
        text_font    = tkfont.nametofont("TkTextFont")
        fixed_font   = tkfont.nametofont("TkFixedFont")
        menu_font    = tkfont.nametofont("TkMenuFont")
        small_font   = tkfont.nametofont("TkSmallCaptionFont") if "TkSmallCaptionFont" in tkfont.names() else default_font

        for f in (default_font, text_font, fixed_font, menu_font, small_font):
            try:
                f.configure(size=14)   # увеличено до 14 для лучшей читаемости
            except Exception:
                pass

        # 3) подсказать ttk, что по умолчанию используем увеличенный шрифт
        style = ttk.Style(self)
        style.configure(".", font=default_font)
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font)
        style.configure("TEntry", font=default_font)
        style.configure("TCheckbutton", font=default_font)
        style.configure("TSpinbox", font=default_font)

                
        
        self.vars = {}
        self._build()
        self._load_prefs()
        self._update_bit_samples()

    def _build(self):
        frm = ttk.Frame(self, padding=12)  # увеличен отступ
        frm.grid(row=0, column=0, sticky="nsew")

        def add_row(r, label, var, width=22, kind="entry", **kwargs):  # увеличена ширина по умолчанию
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", padx=(0,8), pady=4)  # увеличены отступы
            if kind == "entry":
                e = ttk.Entry(frm, textvariable=var, width=width)
                e.grid(row=r, column=1, sticky="ew", pady=4)  # увеличен отступ
                return e
            elif kind == "spin":
                e = ttk.Spinbox(frm, from_=kwargs.get("from_", 0), to=kwargs.get("to", 100), textvariable=var, width=width)
                e.grid(row=r, column=1, sticky="ew", pady=4)  # увеличен отступ
                return e
            elif kind == "check":
                e = ttk.Checkbutton(frm, variable=var)
                e.grid(row=r, column=1, sticky="w", pady=4)  # увеличен отступ
                return e
            else:
                raise ValueError(kind)

        # Variables
        self.vars["target_signal_hz"] = tk.StringVar(value=str(DEF["target_signal_hz"]))
        self.vars["if_offset_hz"]     = tk.StringVar(value=str(DEF["if_offset_hz"]))
        self.vars["freq_corr_hz"]    = tk.StringVar(value=str(DEF["freq_corr_hz"]))
        self.vars["tx_gain_db"]       = tk.StringVar(value=str(DEF["tx_gain_db"]))
        self.vars["phase_low"]        = tk.StringVar(value=str(DEF["phase_low"]))
        self.vars["phase_high"]       = tk.StringVar(value=str(DEF["phase_high"]))
        self.vars["front"]            = tk.StringVar(value=str(DEF["front"]))
        self.vars["hex_message"]      = tk.StringVar(value=DEF["hex_message"])
        self.vars["FILE_IQ"]          = tk.StringVar(value=DEF["FILE_IQ"])
        self.vars["repeat"]           = tk.StringVar(value=str(DEF["repeat"]))
        self.vars["gap_s"]            = tk.StringVar(value=str(DEF["gap_s"]))
        self.vars["fs_gen"]           = tk.StringVar(value=str(DEF["fs_gen"]))
        self.vars["fs_tx"]            = tk.StringVar(value=str(DEF["fs_tx"]))
        self.vars["hw_amp"]           = tk.BooleanVar(value=DEF["hw_amp"])

        r = 0
        add_row(r, "target_signal_hz", self.vars["target_signal_hz"]); r+=1
        add_row(r, "if_offset_hz",     self.vars["if_offset_hz"]); r+=1
        add_row(r, "freq_corr_hz",    self.vars["freq_corr_hz"]); r+=1
        add_row(r, "tx_gain_db (0..47)", self.vars["tx_gain_db"]); r+=1

        row = r
        ttk.Label(frm, text="phase_low / phase_high").grid(row=row, column=0, sticky="w", padx=(0,8), pady=4)  # увеличены отступы
        phase_frame = ttk.Frame(frm)
        phase_frame.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Entry(phase_frame, textvariable=self.vars["phase_low"], width=10).grid(row=0, column=0, padx=(0,6))  # увеличена ширина
        ttk.Entry(phase_frame, textvariable=self.vars["phase_high"], width=10).grid(row=0, column=1)
        r += 1

        add_row(r, "front (samples)", self.vars["front"]); r+=1
        add_row(r, "hex_message", self.vars["hex_message"], width=45); r+=1  # увеличена ширина для hex

        ttk.Label(frm, text="FILE_IQ").grid(row=r, column=0, sticky="w", padx=(0,8), pady=4)  # увеличены отступы
        file_frame = ttk.Frame(frm)
        file_frame.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Entry(file_frame, textvariable=self.vars["FILE_IQ"], width=32).grid(row=0, column=0, padx=(0,6))  # увеличена ширина
        ttk.Button(file_frame, text="Browse…", command=self._browse).grid(row=0, column=1)
        r += 1

        add_row(r, "repeat (N or 'loop')", self.vars["repeat"]); r+=1
        add_row(r, "gap_s", self.vars["gap_s"]); r+=1

        ttk.Label(frm, text="fs_gen / fs_tx").grid(row=r, column=0, sticky="w", padx=(0,8), pady=4)  # увеличены отступы
        fs_frame = ttk.Frame(frm)
        fs_frame.grid(row=r, column=1, sticky="ew", pady=4)
        self._fs_gen_entry = ttk.Entry(fs_frame, textvariable=self.vars["fs_gen"], width=14)
        self._fs_gen_entry.grid(row=0, column=0, padx=(0,6))  # увеличена ширина
        ttk.Entry(fs_frame, textvariable=self.vars["fs_tx"], width=14).grid(row=0, column=1)
        r += 1

        ttk.Label(frm, text="hw_amp_enabled").grid(row=r, column=0, sticky="w", padx=(0,8), pady=4)  # увеличены отступы
        ttk.Checkbutton(frm, variable=self.vars["hw_amp"]).grid(row=r, column=1, sticky="w", pady=4)
        r += 1

        self._bit_samples_lbl = ttk.Label(frm, text="bit_samples = 2500")
        self._bit_samples_lbl.grid(row=r, column=0, columnspan=2, sticky="w", pady=(6,8))  # увеличены отступы
        r += 1

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=r, column=0, columnspan=2, pady=(8,0))  # увеличен отступ
        self.btn_gen  = ttk.Button(btn_frame, text="Generate (cf32)", command=self._on_generate, width=18)  # увеличена ширина
        self.btn_tx_b = ttk.Button(btn_frame, text="TX from buffer",  command=self._on_tx_buffer, width=18)
        self.btn_tx_f = ttk.Button(btn_frame, text="TX from file",    command=self._on_tx_file, width=18)
        self.btn_gen.grid(row=0, column=0, padx=6)  # увеличены отступы между кнопками
        self.btn_tx_b.grid(row=0, column=1, padx=6)
        self.btn_tx_f.grid(row=0, column=2, padx=6)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor="w").grid(row=1, column=0, sticky="ew", padx=12, pady=(6,12))  # увеличены отступы

        self._fs_gen_entry.bind("<KeyRelease>", lambda e: self._update_bit_samples())

    def _browse(self):
        p = filedialog.askopenfilename(title="Select cf32 file", filetypes=[("cf32 IQ", "*.cf32;*.iq;*.bin;*.*")])
        if p:
            self.vars["FILE_IQ"].set(p)

    def _update_bit_samples(self):
        try:
            fs = float(self.vars["fs_gen"].get())
            bit_samples = int(round(fs / 400.0))
            self._bit_samples_lbl.config(text=f"bit_samples = {bit_samples}")
            try:
                front = int(float(self.vars['front'].get()))
                if front >= bit_samples // 2:
                    self.status.set("Warning: front >= bit_samples/2")
                else:
                    self.status.set("Ready.")
            except Exception:
                pass
        except Exception:
            self._bit_samples_lbl.config(text=f"bit_samples = ?")

    def _load_prefs(self):
        if PREFS_PATH.exists():
            try:
                data = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if k in self.vars:
                        if isinstance(self.vars[k], tk.BooleanVar):
                            self.vars[k].set(bool(v))
                        else:
                            self.vars[k].set(str(v))
            except Exception:
                pass

    def _save_prefs(self):
        data = {k: (self.vars[k].get() if not isinstance(self.vars[k], tk.BooleanVar) else self.vars[k].get())
                for k in self.vars}
        try:
            PREFS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _parse_common(self):
        try:
            target = float(self.vars["target_signal_hz"].get())
            if_off = float(self.vars["if_offset_hz"].get())
            hz    = float(self.vars["freq_corr_hz"].get())
            txg    = int(float(self.vars["tx_gain_db"].get()))
            phase_low  = float(self.vars["phase_low"].get())
            phase_high = float(self.vars["phase_high"].get())
            front  = int(float(self.vars["front"].get()))
            hexm   = self.vars["hex_message"].get().strip()
            rep_s  = self.vars["repeat"].get().strip()
            repeat = rep_s if rep_s.lower() == "loop" else int(float(rep_s))
            gap    = float(self.vars["gap_s"].get())
            fs_gen = int(float(self.vars["fs_gen"].get()))
            fs_tx  = int(float(self.vars["fs_tx"].get()))
            hw_amp = bool(self.vars["hw_amp"].get())
        except Exception as e:
            raise ValueError(f"Parse error: {e}")

        if any(c not in "0123456789abcdefABCDEF" for c in hexm) or (len(hexm) % 2 != 0):
            raise ValueError("HEX invalid (only 0-9A-F, even length)")
        if abs(if_off) > fs_tx/3:
            self.status.set("Warning: |if_offset_hz| > fs_tx/3")

        return dict(target=target, if_off=if_off, hz=hz, txg=txg,
                    phase_low=phase_low, phase_high=phase_high, front=front,
                    hexm=hexm, repeat=repeat, gap=gap, fs_gen=fs_gen, fs_tx=fs_tx, hw_amp=hw_amp)

    def _on_generate(self):
        try:
            p = self._parse_common()
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        path = filedialog.asksaveasfilename(title="Save cf32", defaultextension=".cf32",
                                            filetypes=[("cf32 IQ", "*.cf32")])
        if not path:
            return

        self.status.set("Generating…")
        self.update_idletasks()
        try:
            sig = gen.generate_psk406_cf32(
                sample_rate_sps=p["fs_gen"],
                bit_rate_bps=400.0,
                hex_message=p["hexm"],
                phase_low_high=(p["phase_low"], p["phase_high"]),
                front_samples=p["front"],
                save_path=path,
                return_array=True,
            )

            self.status.set(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Generate failed", str(e))

    def _build_ramped_sequence(self, sig: np.ndarray, fs_gen: int, gap_s: float, repeat) -> np.ndarray:
        """
        Делает мягкий вход/выход и собирает один длинный массив:
        [pre_silence + ramp_in + sig + ramp_out + gap_zeros] * repeat
        Если repeat='loop', возвращает один кадр (без повторов) — для -R.
        """
        # 1) параметры ramp и прелюдии
        ramp_ms = 2.0
        pre_ms  = 2.0
        ramp = max(1, int(round(fs_gen * ramp_ms / 1000.0)))
        pre  = max(1, int(round(fs_gen * pre_ms  / 1000.0)))
        gap  = max(0, int(round(fs_gen * gap_s)))

        # 2) окно Ханна для плавного входа/выхода
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(2 * ramp) / (2 * ramp - 1))
        win_in, win_out = w[:ramp].astype(np.float32), w[ramp:].astype(np.float32)

        sig = sig.astype(np.complex64, copy=True)
        if ramp < len(sig):
            sig[:ramp]      *= win_in
            sig[-ramp:]     *= win_out
        else:
            # очень короткий сигнал — нормализовать окно
            k = len(sig) // 2
            if k > 0:
                sig[:k]    *= win_in[:k]
                sig[-k:]   *= win_out[-k:]

        pre_zeros = np.zeros(pre, dtype=np.complex64)
        gap_zeros = np.zeros(gap, dtype=np.complex64)
        frame = np.concatenate([pre_zeros, sig, gap_zeros])

        if isinstance(repeat, str) and repeat.lower() == "loop":
            return frame  # один кадр; дальше backend запустит -R
        else:
            n = int(repeat)
            return np.tile(frame, n)



    def _on_tx_buffer(self):
        try:
            p = self._parse_common()
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        def worker():
            self.status.set("TX from buffer…")
            try:
                sig = gen.generate_psk406_cf32(
                    sample_rate_sps=p["fs_gen"],
                    bit_rate_bps=400.0,
                    hex_message=p["hexm"],
                    phase_low_high=(p["phase_low"], p["phase_high"]),
                    front_samples=p["front"],
                    return_array=True,
                    save_path=None,
                )

                # ВАЖНО: формируем непрерывный поток с ramp и паузой из нулей
                long_sig = self._build_ramped_sequence(sig, fs_gen=p["fs_gen"], gap_s=p["gap"], repeat=p["repeat"])

                # Один запуск передачи без «щёлканья» между посылками
                tx.hackrf_tx_from_array(
                    long_sig,
                    target_signal_hz=p["target"],
                    if_offset_hz=p["if_off"],
                    freq_correction_hz=p["hz"],
                    tx_sample_rate_sps=p["fs_tx"],
                    repeat=1,           # уже собрали повторы в массиве
                    gap_s=0.0,          # пауза уже внутри массива
                    tx_gain_db=p["txg"],
                    hw_amp_enabled=p["hw_amp"],
                    amp_scale=0.95,
                    transport="hackrf_transfer",
                    input_sample_rate_sps=p["fs_gen"],
                )
                self.status.set("TX done.")
            except Exception as e:
                messagebox.showerror("TX failed", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_tx_file(self):
        try:
            p = self._parse_common()
            path = self.vars["FILE_IQ"].get().strip()
            if not path:
                raise ValueError("FILE_IQ is empty")
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        def worker():
            self.status.set("TX from file…")
            try:
                tx.hackrf_tx_from_file(
                    path,
                    target_signal_hz=p["target"],
                    if_offset_hz=p["if_off"],
                    freq_correction_hz=p["hz"],
                    tx_sample_rate_sps=p["fs_tx"],
                    repeat=p["repeat"],
                    gap_s=p["gap"],
                    tx_gain_db=p["txg"],
                    hw_amp_enabled=p["hw_amp"],
                    amp_scale=0.95,
                    transport="hackrf_transfer",
                    assume_input_fs_sps=p["fs_gen"],
                )
                self.status.set("TX done.")
            except Exception as e:
                messagebox.showerror("TX failed", str(e))

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self):
        self._save_prefs()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
