import logging, os, sys, time
from typing import Optional, Union

# --- уровни ---
_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NONE": logging.CRITICAL + 10,
}

def _parse_level(level: Union[str, int, None], fallback: int) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return _LEVELS.get(level.upper(), fallback)
    return fallback

# --- Глобальная настройка рута ---
def setup_logging(level: Optional[Union[str, int]] = None, *, fallback: str = "INFO"):
    """
    Инициализирует root-логгер.
    Приоритет выбора уровня:
    1) явный аргумент level (например, 'DEBUG' или logging.DEBUG)
    2) переменная окружения BEACON_LOG
    3) fallback ('INFO' по умолчанию)
    """
    env = os.getenv("BEACON_LOG")
    resolved = _parse_level(level, _parse_level(env, _LEVELS[fallback]))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

def set_global_level(level: Union[str, int]):
    """Сменить уровень root-логгера после setup_logging()."""
    logging.getLogger().setLevel(_parse_level(level, logging.INFO))

# --- Локальный (модульный) логгер ---
def init_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Возвращает логгер модуля. Если задан level — фиксирует уровень
    только для этого логгера (root остаётся как есть).
    """
    lg = logging.getLogger(name)
    if level is not None:
        lg.setLevel(_parse_level(level, logging.INFO))
    return lg

# --- совместимость со старым API ---
def get_logger(name: str):
    return logging.getLogger(name)

# --- утилиты как были ---
class OnceEvery(logging.Filter):
    """Фильтр: пропускает запись не чаще 1 раза в interval_sec (на ключ)."""
    _last = {}
    def __init__(self, key: str, interval_sec: float):
        super().__init__()
        self.key = key
        self.interval = interval_sec
    def filter(self, record):
        now = time.time()
        last = self._last.get(self.key, 0.0)
        if now - last >= self.interval:
            self._last[self.key] = now
            return True
        return False

_seen_once = set()
def warn_once(logger: logging.Logger, key: str, message: str):
    if key not in _seen_once:
        _seen_once.add(key)
        logger.warning(message)
