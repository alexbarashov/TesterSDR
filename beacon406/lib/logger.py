import logging, os, sys, time

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NONE": logging.CRITICAL + 10,
}

def setup_logging(default_level="INFO"):
    level = _LEVELS.get(os.getenv("BEACON_LOG", default_level).upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

def get_logger(name: str):
    return logging.getLogger(name)

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