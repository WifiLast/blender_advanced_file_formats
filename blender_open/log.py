"""Logging for the importer. Logger 'cad_importer' (children: .nwd, .rvt, .build); INFO goes to the system console,
DEBUG (option 'Debug Log') additionally to a file in the temp folder."""
import logging
import tempfile
import time
from pathlib import Path

ROOT = "cad_importer"
_FMT = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
_handlers = {}


def get(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}" if name else ROOT)


def setup(debug: bool = False, tag: str = "import") -> str | None:
    """Configure logging for one import. Returns the log file path when debug is on."""
    root = get()
    root.propagate = False
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for h in list(root.handlers):          # also handlers left behind by a reloaded module (duplicate log lines)
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    _handlers.clear()
    con = logging.StreamHandler()
    con.setFormatter(_FMT)
    con.setLevel(logging.DEBUG if debug else logging.INFO)
    if not debug:                          # one warning per damaged block is too much for the console (summary stays)
        con.addFilter(lambda rec: not (rec.name.endswith("nwd_geometry") and rec.levelno < logging.ERROR
                                       and rec.getMessage().startswith("geometry block")))
    root.addHandler(con)
    _handlers["console"] = con
    path = None
    if debug:
        path = str(Path(tempfile.gettempdir()) / f"cad_importer_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(_FMT)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)
        _handlers["file"] = fh
    return path


def log_environment(logger):
    """Blender / Python / numpy / add-on versions, once per import (helps with bug reports)."""
    import platform
    import sys
    try:
        import bpy
        bl = bpy.app.version_string
    except Exception:
        bl = "?"
    try:
        import numpy
        npv = numpy.__version__
    except Exception:
        npv = "?"
    logger.info("environment: Blender %s, Python %s, numpy %s, %s", bl, sys.version.split()[0], npv, platform.platform())
    try:
        import tracemalloc  # noqa: F401
    except Exception:
        pass


class Timer:
    """with Timer(log, 'what'): ...  -> logs the duration at INFO."""

    def __init__(self, logger, what):
        self.logger, self.what = logger, what

    def __enter__(self):
        self.t = time.perf_counter()
        self.logger.info("%s ...", self.what)
        return self

    def __exit__(self, *exc):
        self.logger.info("%s done in %.2f s", self.what, time.perf_counter() - self.t)
