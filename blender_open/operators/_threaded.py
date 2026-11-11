"""Mixin: run a pure-Python reader in a worker thread, then build the Blender data in the main thread (modal timer)."""
import threading
import time
import traceback

from .. import log as cadlog

L = cadlog.get("op")


class ThreadedImportMixin:
    _timer = None
    _worker = None
    _result = None
    _error = None
    _log = None

    def _read(self):
        """Worker thread body: must not touch bpy. Return the reader result."""
        raise NotImplementedError

    def _build(self, context, data):
        """Main thread: create Blender data from the reader result."""
        raise NotImplementedError

    def _start(self, context):
        logfile = cadlog.setup(bool(getattr(self, "debug_log", False)), self.bl_idname.split(".")[-1])
        cadlog.log_environment(L)
        L.info("%s: %s", self.bl_label, getattr(self, "_path", ""))
        L.info("options: %s", {k: getattr(self, k) for k in self.__annotations__ if k != "filter_glob"})
        if logfile:
            L.info("debug log file: %s", logfile)
            self.report({"INFO"}, f"Debug log: {logfile}")
        self._log = []
        self._result = None
        self._error = None

        def run():
            t0 = time.perf_counter()
            L.info("worker started: %s", self.bl_label)
            try:
                self._result = self._read()
                L.info("worker finished in %.1f s", time.perf_counter() - t0)
            except Exception:
                self._error = traceback.format_exc()
                L.error("worker failed after %.1f s\n%s", time.perf_counter() - t0, self._error)

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _say(self, *a):
        L.info(" ".join(str(x) for x in a))
        if self._log is not None:
            self._log.append(" ".join(str(x) for x in a))

    def _stop(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass

    def modal(self, context, event):
        if event.type == "ESC":
            self._stop(context)
            self.report({"WARNING"}, "Import cancelled (the worker finishes in the background)")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._log:
            try:
                context.workspace.status_text_set(f"{self.bl_label}: {self._log[-1]}")
            except Exception:
                pass
        if self._worker.is_alive():
            return {"RUNNING_MODAL"}
        self._stop(context)
        if self._error:
            self.report({"ERROR"}, self._error.strip().splitlines()[-1])
            return {"CANCELLED"}
        t0 = time.perf_counter()
        try:
            L.info("building Blender data ...")
            self._build(context, self._result)
            L.info("build finished in %.1f s", time.perf_counter() - t0)
        except Exception:
            L.error("build failed\n%s", traceback.format_exc())
            self.report({"ERROR"}, "Building Blender data failed, see the system console")
            return {"CANCELLED"}
        return {"FINISHED"}
