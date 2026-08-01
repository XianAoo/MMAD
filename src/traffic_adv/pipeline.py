from __future__ import annotations

import json
import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from traffic_adv.config import ExperimentConfig
from traffic_adv.data.adapters import create_dataset_adapter
from traffic_adv.data.io import load_flows, save_flows
from traffic_adv.evaluation.metrics import compare_flows
from traffic_adv.evaluation.victim import evaluate_clean_victim, evaluate_victim
from traffic_adv.methods import create_method
from traffic_adv.methods.common import configured_attack_types


def train(config: ExperimentConfig) -> None:
    config.ensure_output_dirs()
    create_method(config).train(create_dataset_adapter(config.dataset))


def attack(config: ExperimentConfig):
    config.ensure_output_dirs()
    flows = create_method(config).generate(create_dataset_adapter(config.dataset))
    if flows is None:
        return config.output_dir
    destination = _adversarial_output_path(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_flows(flows, destination)
    return destination


def evaluate(config: ExperimentConfig):
    metrics = {}
    adversarial_json = _adversarial_output_path(config)
    if adversarial_json.exists() and config.method != "eta":
        original = _load_original_attack_flows(config)
        adversarial = load_flows(adversarial_json)
        metrics["traffic"] = compare_flows(original, adversarial)
    victims = config.victims or ((config.victim,) if config.victim is not None else ())
    victim_results = []
    for victim in victims:
        victim_results.append(evaluate_victim(replace(config, victim=victim)))
    if victim_results:
        metrics["victims"] = victim_results
        if len(victim_results) == 1:
            metrics["victim"] = victim_results[0]
    if not metrics:
        raise FileNotFoundError(
            f"No adversarial output found for evaluation in {config.output_dir}"
        )
    destination = config.output_dir / "metrics.json"
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_clean(config: ExperimentConfig):
    config.ensure_output_dirs()
    victims = config.victims or ((config.victim,) if config.victim is not None else ())
    if not victims:
        raise ValueError("Experiment does not define victim models")
    results = []
    for victim in victims:
        result = evaluate_clean_victim(replace(config, victim=victim))
        results.append(result)
    metrics = {"clean_victims": results}
    if len(results) == 1:
        metrics["clean_victim"] = results[0]
    destination = config.output_dir / "clean_metrics.json"
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_destination = config.output_dir / "clean_metrics.csv"
    _write_clean_metrics_csv(results, csv_destination)
    return metrics


def _write_clean_metrics_csv(results: list[dict[str, object]], destination):
    if not results:
        return
    columns = [
        "victim",
        "samples",
        "benign_samples",
        "attack_samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "model_path",
        "benign_path",
        "attack_path",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result.get(column, "") for column in columns})


def run(config: ExperimentConfig):
    train(config)
    attack(config)
    return evaluate(config)


def _adversarial_output_path(config: ExperimentConfig) -> Path:
    filename = str(config.method_config.get("adversarial_output_filename", "adversarial.json"))
    if "{timestamp}" in filename:
        filename = filename.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = Path(filename)
    destination = path if path.is_absolute() else config.output_dir / path
    if bool(config.method_config.get("avoid_adversarial_overwrite", False)):
        destination = _unique_path(destination)
    return destination


def _load_original_attack_flows(config: ExperimentConfig):
    flows = []
    for attack_type in configured_attack_types(config.attack_type, config.method_config):
        flows.extend(load_flows(config.dataset.attack(attack_type).json))
    return flows


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}")

