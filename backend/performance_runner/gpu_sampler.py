"""Background VRAM sampler (nvidia-smi based). No backend coupling."""

from __future__ import annotations

import threading
import time

import perf_config


class GpuSampler:
    """Samples used-VRAM in a background thread; supports windowed peak/floor.

    Usage:
        s = GpuSampler(hz=10).start()
        s.mark("gen-3-start")
        ... run work ...
        peak = s.window_peak(since="gen-3-start")
        s.stop()
    """

    def __init__(self, hz: float = 10.0, gpu_index: int = perf_config.GPU_INDEX) -> None:
        self._interval = 1.0 / hz
        self._gpu = gpu_index
        self._samples: list[tuple[float, int]] = []    # (t, vram_mib)  — fast, for peak/floor
        self._ram: list[tuple[float, float]] = []       # (t, host_ram_mib) — spill signal
        self._stats: list[tuple[float, float | None, float | None, float | None]] = []
        #                 (t, temp_c, clock_mhz, power_w) — slow (~1 Hz): thermal/throttle + energy
        self._last_stat = 0.0
        self._marks: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "GpuSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def mark(self, name: str) -> None:
        self._marks[name] = time.time()

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            try:
                self._samples.append((now, perf_config.gpu_used_mib(self._gpu)))
            except Exception:  # noqa: BLE001, S110
                pass
            ram = perf_config.host_ram().get("ram_used_mib")
            if ram is not None:
                self._ram.append((now, ram))
            # Temp/clock/power change slowly and the full nvidia-smi query is heavier, so
            # sample them ~1 Hz instead of at the fast VRAM cadence.
            if now - self._last_stat >= 1.0:
                self._last_stat = now
                try:
                    s = perf_config.gpu_stats(self._gpu)
                    self._stats.append((now, s.get("temp_c"), s.get("clock_mhz"), s.get("power_w")))
                except Exception:  # noqa: BLE001, S110
                    pass
            time.sleep(self._interval)

    def _win(self, data: list[tuple[float, float]], since: str | None, until: str | None) -> list[float]:
        t0 = self._marks.get(since, 0.0) if since else 0.0
        t1 = self._marks.get(until, time.time()) if until else time.time()
        return [m for (t, m) in data if t0 <= t <= t1]

    def window_peak(self, since: str | None = None, until: str | None = None) -> int:
        vals = self._win(self._samples, since, until)
        return int(max(vals)) if vals else 0

    def window_floor(self, since: str | None = None, until: str | None = None) -> int:
        vals = self._win(self._samples, since, until)
        return int(min(vals)) if vals else 0

    def window_ram_floor(self, since: str | None = None, until: str | None = None) -> float | None:
        """Min host-RAM (MiB) in the window — a rising per-gen RAM floor signals a spill/leak."""
        vals = self._win(self._ram, since, until)
        return min(vals) if vals else None

    def _stat_series(self, idx: int, since: str | None, until: str | None) -> list[float]:
        t0 = self._marks.get(since, 0.0) if since else 0.0
        t1 = self._marks.get(until, time.time()) if until else time.time()
        return [row[idx] for row in self._stats if t0 <= row[0] <= t1 and row[idx] is not None]

    def window_temp_max(self, since: str | None = None, until: str | None = None) -> float | None:
        vals = self._stat_series(1, since, until)
        return max(vals) if vals else None

    def window_clock_mean(self, since: str | None = None, until: str | None = None) -> float | None:
        vals = self._stat_series(2, since, until)
        return sum(vals) / len(vals) if vals else None

    def window_power_mean(self, since: str | None = None, until: str | None = None) -> float | None:
        vals = self._stat_series(3, since, until)
        return sum(vals) / len(vals) if vals else None

    def current(self) -> int:
        return self._samples[-1][1] if self._samples else 0
