from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class MMADGenerationProfiler:
    def __init__(self, config, settings: dict[str, object], torch, device: str):
        self.config = config
        self.settings = settings
        self.torch = torch
        self.device = str(device)
        self.cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        self.warmup_batches = max(0, int(settings.get("profile_warmup_batches", 0) or 0))
        self.power_interval = float(settings.get("profile_power_interval_s", 0.2) or 0.2)
        self.record_batches = bool(settings.get("profile_record_batches", True))
        self.batch_records: list[dict[str, Any]] = []
        self.power_records: list[dict[str, Any]] = []
        self._started_at = time.perf_counter()
        self._core_started_at: float | None = None
        self._core_finished_at: float | None = None
        self._sampler_stop = threading.Event()
        self._sampler_thread: threading.Thread | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._gpu_index = self._device_index()
        self._psutil_process = self._process_probe()
        self._rss_started_bytes = self._rss_bytes()
        self._error: str | None = None

    @classmethod
    def enabled(cls, settings: dict[str, object]) -> bool:
        return bool(settings.get("profile_generation", False))

    def set_metadata(self, **values: object) -> None:
        self.metadata = getattr(self, "metadata", {})
        self.metadata.update(values)

    def start_core(self) -> None:
        self._sync()
        self._core_started_at = time.perf_counter()
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats()
        self._start_power_sampler()

    def finish_core(self) -> None:
        self._sync()
        self._core_finished_at = time.perf_counter()
        self._stop_power_sampler()

    def measure_batch(self, sample_count: int, fn: Callable[[], Any]) -> Any:
        index = len(self.batch_records)
        included = index >= self.warmup_batches
        self._sync()
        gpu_start = gpu_end = None
        cuda_context = (
            self.torch.cuda.device(self._gpu_index)
            if self.cuda and self._gpu_index is not None
            else nullcontext()
        )
        with cuda_context:
            if self.cuda:
                gpu_start = self.torch.cuda.Event(enable_timing=True)
                gpu_end = self.torch.cuda.Event(enable_timing=True)
                gpu_start.record()
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            rss_start = self._rss_bytes()
            result = fn()
            if self.cuda and gpu_end is not None:
                gpu_end.record()
        self._sync()
        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        gpu_ms = None
        if gpu_start is not None and gpu_end is not None:
            gpu_ms = float(gpu_start.elapsed_time(gpu_end))
        record = {
            "batch_index": index,
            "sample_count": int(sample_count),
            "included_in_aggregate": included,
            "wall_s": wall_end - wall_start,
            "cpu_process_s": cpu_end - cpu_start,
            "gpu_event_ms": gpu_ms,
            "rss_start_bytes": rss_start,
            "rss_end_bytes": self._rss_bytes(),
            "gpu_memory_allocated_bytes": self._cuda_value("memory_allocated"),
            "gpu_memory_reserved_bytes": self._cuda_value("memory_reserved"),
            "gpu_max_memory_allocated_bytes": self._cuda_value("max_memory_allocated"),
            "gpu_max_memory_reserved_bytes": self._cuda_value("max_memory_reserved"),
        }
        self.batch_records.append(record)
        return result

    def save(self, *, error: Exception | None = None) -> Path:
        if error is not None:
            self._error = f"{type(error).__name__}: {error}"
        if self._core_started_at is not None and self._core_finished_at is None:
            self.finish_core()
        output = self._output_path()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload()
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def _payload(self) -> dict[str, Any]:
        included = [record for record in self.batch_records if record["included_in_aggregate"]]
        all_samples = sum(record["sample_count"] for record in self.batch_records)
        included_samples = sum(record["sample_count"] for record in included)
        wall_s = sum(record["wall_s"] for record in included)
        cpu_s = sum(record["cpu_process_s"] for record in included)
        gpu_ms_values = [record["gpu_event_ms"] for record in included if record["gpu_event_ms"] is not None]
        gpu_event_s = sum(gpu_ms_values) / 1000.0 if gpu_ms_values else None
        core_span_s = None
        if self._core_started_at is not None and self._core_finished_at is not None:
            core_span_s = self._core_finished_at - self._core_started_at
        energy_j = self._gpu_energy_j(core_span_s)
        payload = {
            "schema": "traffic_adv.mmad.generation_profile.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": self.config.name,
            "method": self.config.method,
            "attack_type": self.config.attack_type,
            "device": self.device,
            "cuda_available": self.cuda,
            "warmup_batches_excluded": self.warmup_batches,
            "notes": {
                "core_generation": "Measures attacker.attack(batch.to(device)) for each batch.",
                "end_to_end": "Covers MMADMethod.generate(), excluding final adversarial JSON serialization in pipeline.attack().",
                "gpu_energy": "Estimated from nvidia-smi power.draw samples when available.",
            },
            "metadata": getattr(self, "metadata", {}),
            "aggregate": {
                "batches_total": len(self.batch_records),
                "batches_included": len(included),
                "samples_total": all_samples,
                "samples_included": included_samples,
                "core_batch_wall_s": wall_s,
                "core_span_wall_s": core_span_s,
                "end_to_end_wall_s": time.perf_counter() - self._started_at,
                "cpu_process_s": cpu_s,
                "gpu_event_s": gpu_event_s,
                "wall_ms_per_sample": self._per_sample_ms(wall_s, included_samples),
                "gpu_event_ms_per_sample": self._per_sample_ms(gpu_event_s, included_samples),
                "throughput_samples_per_s": self._throughput(included_samples, wall_s),
                "gpu_energy_j": energy_j,
                "gpu_energy_j_per_sample": self._per_sample(energy_j, included_samples),
                "start_cpu_rss_bytes": self._rss_started_bytes,
                "end_cpu_rss_bytes": self._rss_bytes(),
                "peak_cpu_rss_bytes": self._peak_cpu_rss(),
                "peak_gpu_memory_allocated_bytes": self._cuda_value("max_memory_allocated"),
                "peak_gpu_memory_reserved_bytes": self._cuda_value("max_memory_reserved"),
            },
            "power_samples": self.power_records,
            "error": self._error,
        }
        if self.record_batches:
            payload["batches"] = self.batch_records
        return self._json_safe(payload)

    def _output_path(self) -> Path:
        configured = self.settings.get("profile_output")
        if configured:
            path = Path(str(configured))
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(f"generation_profile_{stamp}.json")
        if not path.is_absolute():
            path = self.config.output_dir / path
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        for index in range(2, 1000):
            candidate = path.with_name(f"{stem}_{index}{suffix}")
            if not candidate.exists():
                return candidate
        return path.with_name(f"{stem}_{int(time.time())}{suffix}")

    def _start_power_sampler(self) -> None:
        if not self.cuda or self._nvidia_smi is None:
            return
        self._sampler_stop.clear()
        self._sampler_thread = threading.Thread(target=self._sample_power_loop, daemon=True)
        self._sampler_thread.start()

    def _stop_power_sampler(self) -> None:
        self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=max(1.0, self.power_interval * 4))
            self._sampler_thread = None

    def _sample_power_loop(self) -> None:
        while not self._sampler_stop.is_set():
            started = time.perf_counter()
            record = self._read_power_sample()
            if record is not None:
                record["elapsed_s"] = started - (self._core_started_at or started)
                self.power_records.append(record)
            remaining = max(0.0, self.power_interval - (time.perf_counter() - started))
            self._sampler_stop.wait(remaining)

    def _read_power_sample(self) -> dict[str, Any] | None:
        command = [
            self._nvidia_smi,
            "--query-gpu=power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        if self._gpu_index is not None:
            command.extend(["-i", str(self._gpu_index)])
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1.0, self.power_interval),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0 or not result.stdout.strip():
            return None
        values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        if len(values) < 3:
            return None
        return {
            "power_w": self._float_or_none(values[0]),
            "gpu_memory_used_mb": self._float_or_none(values[1]),
            "gpu_utilization_percent": self._float_or_none(values[2]),
        }

    def _gpu_energy_j(self, duration_s: float | None) -> float | None:
        samples = [
            (record["elapsed_s"], record["power_w"])
            for record in self.power_records
            if record.get("power_w") is not None
        ]
        if not samples:
            return None
        if len(samples) == 1:
            return samples[0][1] * duration_s if duration_s is not None else None
        energy = samples[0][0] * samples[0][1]
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            if t1 > t0:
                energy += (t1 - t0) * (p0 + p1) / 2.0
        if duration_s is not None and duration_s > samples[-1][0]:
            energy += (duration_s - samples[-1][0]) * samples[-1][1]
        return energy

    def _device_index(self) -> int | None:
        if not self.cuda:
            return None
        if self.device == "cuda":
            return int(self.torch.cuda.current_device())
        if self.device.startswith("cuda:"):
            try:
                return int(self.device.split(":", 1)[1])
            except ValueError:
                return int(self.torch.cuda.current_device())
        return None

    def _sync(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize(self._gpu_index)

    def _cuda_value(self, name: str) -> int | None:
        if not self.cuda:
            return None
        return int(getattr(self.torch.cuda, name)())

    def _process_probe(self):
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process()

    def _rss_bytes(self) -> int | None:
        if self._psutil_process is None:
            return self._windows_rss_bytes()
        return int(self._psutil_process.memory_info().rss)

    def _peak_cpu_rss(self) -> int | None:
        values = [value for value in (self._rss_started_bytes, self._rss_bytes()) if value is not None]
        for record in self.batch_records:
            values.extend(
                value
                for value in (record.get("rss_start_bytes"), record.get("rss_end_bytes"))
                if value is not None
            )
        return max(values) if values else None

    @staticmethod
    def _windows_rss_bytes() -> int | None:
        if os.name != "nt":
            return None
        try:
            import ctypes
            from ctypes import wintypes
        except ImportError:
            return None

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.WorkingSetSize) if ok else None

    @staticmethod
    def _per_sample(value: float | None, samples: int) -> float | None:
        if value is None or samples <= 0:
            return None
        return value / samples

    @classmethod
    def _per_sample_ms(cls, value_s: float | None, samples: int) -> float | None:
        value = cls._per_sample(value_s, samples)
        return None if value is None else value * 1000.0

    @staticmethod
    def _throughput(samples: int, seconds: float | None) -> float | None:
        if seconds is None or seconds <= 0:
            return None
        return samples / seconds

    @staticmethod
    def _float_or_none(value: str) -> float | None:
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
