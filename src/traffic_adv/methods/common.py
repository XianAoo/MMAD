from __future__ import annotations

from typing import Iterable

import numpy as np

from traffic_adv.data.schema import TrafficFlow


def sample_flows(
    flows: Iterable[TrafficFlow],
    settings: dict[str, object],
    key: str,
    *,
    ratio_key: str | None = None,
    seed: int = 42,
) -> list[TrafficFlow]:
    values = list(flows)
    if not values:
        return values
    limit = sample_limit(len(values), settings, key, ratio_key)
    if limit is None or limit <= 0 or limit >= len(values):
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=limit, replace=False))
    return [values[int(index)] for index in indices]


def sample_limit(
    total: int,
    settings: dict[str, object],
    key: str,
    ratio_key: str | None = None,
) -> int | None:
    if ratio_key is not None and settings.get(ratio_key) is not None:
        ratio = float(settings[ratio_key])
        if ratio <= 0.0:
            return 0
        if ratio >= 1.0:
            return total
        return max(1, int(round(total * ratio)))
    limit = settings.get(key)
    return None if limit is None else int(limit)


def configured_attack_types(default_attack_type: str, settings: dict[str, object]) -> list[str]:
    configured = settings.get("attack_types")
    if configured is None:
        return [default_attack_type]
    if isinstance(configured, str):
        configured = [configured]
    attack_types = [str(value).lower() for value in configured]
    if not attack_types:
        raise ValueError("method_config.attack_types must not be empty")
    return attack_types


def load_attack_flows(
    dataset,
    default_attack_type: str,
    settings: dict[str, object],
    key: str,
    *,
    ratio_key: str | None = None,
    seed: int = 42,
) -> list[TrafficFlow]:
    output: list[TrafficFlow] = []
    for index, attack_type in enumerate(configured_attack_types(default_attack_type, settings)):
        flows = dataset.load_attack(attack_type)
        output.extend(
            sample_flows(
                flows,
                settings,
                key,
                ratio_key=ratio_key,
                seed=seed + index,
            )
        )
    if not output:
        raise ValueError("No malicious flows were loaded for the configured attack selection")
    return output


def maybe_progress(iterable, settings: dict[str, object], *, desc: str, total=None, leave=None):
    if not bool(settings.get("progress", True)):
        return iterable
    if leave is None:
        leave = bool(settings.get("progress_leave", True))
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)
