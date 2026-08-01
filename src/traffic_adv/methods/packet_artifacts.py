from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from traffic_adv.config import ExperimentConfig
from traffic_adv.data.io import save_flows
from traffic_adv.data.schema import TrafficFlow
from traffic_adv.methods.common import maybe_progress


def configured_artifact_path(
    config: ExperimentConfig, configured, default_name: str
) -> Path:
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else config.project_root / path
    return config.output_dir / default_name


def adversarial_pcap_path(config: ExperimentConfig, settings: dict[str, object]) -> Path:
    return configured_artifact_path(
        config,
        settings.get("adversarial_pcap_path"),
        f"{config.attack_type}_adv.pcap",
    )


def attack_display_name(attack_type: str) -> str:
    return attack_type[:1].upper() + attack_type[1:]


def write_packet_feature_artifacts(
    *,
    config: ExperimentConfig,
    flows: Iterable[TrafficFlow],
    settings: dict[str, object],
    method_label: str,
) -> dict[str, str]:
    from traffic_adv.traffic.afterimage.extract import pcap_to_npy
    from traffic_adv.traffic.flow_features.extract import extract_flow_features
    from traffic_adv.traffic.flow_features.extract_repaired import (
        extract_repaired_flow_features,
    )
    from traffic_adv.traffic.json_to_pcap import apply_json_to_pcap

    attack = config.dataset.attack(config.attack_type)
    if attack.pcap is None:
        raise ValueError(
            f"{method_label} requires an attack PCAP in the dataset config "
            "when extract_features is enabled"
        )

    display_name = attack_display_name(config.attack_type)
    json_path = configured_artifact_path(
        config,
        settings.get("adversarial_json_path"),
        f"{display_name}_adv.json",
    )
    pcap_path = adversarial_pcap_path(config, settings)
    flow_path = configured_artifact_path(
        config,
        settings.get("flow_features_path"),
        f"{display_name}_adv.csv",
    )
    repair_path = configured_artifact_path(
        config,
        settings.get("repair_features_path"),
        f"{display_name}_repair_adv.csv",
    )
    afterimage_path = configured_artifact_path(
        config,
        settings.get("afterimage_features_path"),
        f"{config.attack_type}_adv.npy",
    )

    save_flows(flows, json_path)
    steps = (
        ("JSON to PCAP", lambda: apply_json_to_pcap(json_path, attack.pcap, pcap_path)),
        ("CicFlowMeter CSV", lambda: extract_flow_features(pcap_path, flow_path)),
        (
            "CicFlowMeter-repair CSV",
            lambda: extract_repaired_flow_features(pcap_path, repair_path),
        ),
        ("AfterImage NPY", lambda: pcap_to_npy(str(pcap_path), str(afterimage_path))),
    )
    for _, action in maybe_progress(
        steps,
        settings,
        desc=f"[{method_label}] feature artifacts",
        total=len(steps),
        leave=bool(settings.get("progress_leave", True)),
    ):
        action()

    manifest = {
        "json": str(json_path),
        "pcap": str(pcap_path),
        "flow_features": str(flow_path),
        "repair_features": str(repair_path),
        "afterimage_features": str(afterimage_path),
    }
    (config.output_dir / "feature_artifacts.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
