from collections import deque
from PyQt5.QtCore import QThread, QTimer
from Threads.ThreadWorker import NeuroThreadWorker

class NeuroThreadController:
    def __init__(self):
        self.thread = None
        self.worker = None
        self.task_queue = deque()
        self.task_running = False
        self.external_signals = {'result': [], 'error': [], 'finished': []}
        self._cleaning_up = False  # Prevent re-entrant cleanup

    def setup_task(self, *args, **kwargs):
        """Sets up task into Queue"""
        self.task_queue.append((args, kwargs))
        self.Start_queued_task()
        
    def Start_queued_task(self):
        """Start the next queued task"""
        if self.task_running or not self.task_queue:
            return

        args, kwargs = self.task_queue.popleft()
        print(f"[DEBUG] Starting task: {kwargs.get('task')}")

        self.thread = QThread()
        self.worker = NeuroThreadWorker(*args, **kwargs)
        self.worker.moveToThread(self.thread)

        self.task_running = True
        self._cleaning_up = False

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(self._emit_result)
        self.worker.error.connect(self._emit_error)
        self.worker.finished.connect(self.Handle_finished_Task)

        # Safe cleanup signals
        self.worker.finished.connect(self.safe_cleanup)
        self.thread.finished.connect(self.safe_cleanup)

        self.thread.start()

    def connect(self, signal_name, callback):
        if signal_name in self.external_signals:
            if callback not in self.external_signals[signal_name]:
                self.external_signals[signal_name].append(callback)
                print(f"[DEBUG] Connected callback {callback.__name__} to signal '{signal_name}'")
            else:
                print(f"[DEBUG] Callback {callback.__name__} already connected to '{signal_name}'")
        else:
            print(f"[WARNING] Signal {signal_name} is not supported.")

    def connect_signals(self, signal_map: dict):
        for name, cb in signal_map.items():
            self.connect(name, cb)

    def _emit_result(self, value):
        for cb in self.external_signals.get('result', []):
            print(f"[DEBUG] Emitting to {cb.__name__} with value: {value} (type: {type(value)})")
            cb(value)

    def _emit_error(self, msg):
        for cb in self.external_signals.get('error', []):
            cb(msg)

    def Handle_finished_Task(self):
        """Handle task completion and cleanup"""
        if self._cleaning_up:
            # Prevent multiple cleanups if signals fire repeatedly
            return
        self._cleaning_up = True

        print("[DEBUG] Task finished, cleaning up...")

        # Disconnect signals safely
        if self.thread and self.worker:
            try:
                self.thread.started.disconnect(self.worker.run)
            except Exception:
                pass
            try:
                self.thread.finished.disconnect(self.safe_cleanup)
            except Exception:
                pass

            try:
                self.worker.result.disconnect(self._emit_result)
            except Exception:
                pass
            try:
                self.worker.error.disconnect(self._emit_error)
            except Exception:
                pass
            try:
                self.worker.finished.disconnect(self.Handle_finished_Task)
            except Exception:
                pass
            try:
                self.worker.finished.disconnect(self.safe_cleanup)
            except Exception:
                pass

        # Quit and wait thread
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        # Delete worker and thread
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()

        # Remove references
        self.worker = None
        self.thread = None
        self.task_running = False
        self._cleaning_up = False

        # Start next task if available
        if self.task_queue:
            QTimer.singleShot(0, self.Start_queued_task)
            
        # print(f"[DEBUG] {self.thread}, {self.worker}, {self.task_queue}, {self.task_running}, {self.external_signals}, ")

    def safe_cleanup(self):
        """Safely cleanup worker and thread"""
        # Using QTimer.singleShot to safely delete in event loop
        if self.worker:
            QTimer.singleShot(0, self.worker.deleteLater)
        if self.thread:
            QTimer.singleShot(0, self.thread.deleteLater)

    def clear_signal(self, signal_name):
        if signal_name in self.external_signals:
            self.external_signals[signal_name].clear()
            print(f"[DEBUG] Cleared callbacks for signal '{signal_name}'")
        else:
            print(f"[WARNING] Signal {signal_name} is not supported.")
