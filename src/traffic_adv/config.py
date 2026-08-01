from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Install PyYAML to read non-JSON YAML configuration files."
            ) from exc
    else:
        value = yaml.safe_load(text)

    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _title_slug(value: object) -> str:
    return "".join(part.capitalize() for part in _slug(value).split("_"))


def _template_vars(
    *, dataset: "DatasetConfig", attack_type: str, method: str, experiment_name: str
) -> dict[str, str]:
    dataset_key = _slug(dataset.options.get("namespace", dataset.name))
    attack_key = _slug(attack_type)
    method_key = _slug(method)
    return {
        "dataset": dataset_key,
        "dataset_name": dataset.name,
        "attack": attack_key,
        "attack_type": attack_key,
        "attack_title": _title_slug(attack_key),
        "method": method_key,
        "experiment": _slug(experiment_name),
        "namespace": f"{dataset_key}-{attack_key}",
    }


def _expand_templates(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(_SafeFormatDict(variables))
        except ValueError:
            return value
    if isinstance(value, list):
        return [_expand_templates(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_templates(item, variables) for key, item in value.items()}
    return value


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _find_project_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(f"Could not locate pyproject.toml above {path}")


@dataclass(frozen=True)
class TrafficSource:
    json: Path
    pcap: Path | None = None
    features: Path | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    adapter: str
    benign: TrafficSource
    attacks: dict[str, TrafficSource]
    sequence_length: int = 32
    protocol: str = "tcp"
    feature_fields: tuple[str, ...] = ("iat_ms", "packet_size")
    options: dict[str, Any] = field(default_factory=dict)

    def attack(self, name: str) -> TrafficSource:
        try:
            return self.attacks[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown attack type {name!r}; available: {sorted(self.attacks)}"
            ) from exc


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0


@dataclass(frozen=True)
class VictimConfig:
    name: str
    model_path: Path
    model_type: str = "auto"
    architecture: str = "auto"
    feature_source: str = "tabular_csv"
    feature_names_path: Path | None = None
    scaler_path: Path | None = None
    preprocessor_path: Path | None = None
    input_path: Path | None = None
    clean_benign_path: Path | None = None
    clean_attack_path: Path | None = None
    input_scaled: bool = False
    label: int = 1
    positive_label: int = 1
    batch_size: int = 512
    device: str = "auto"
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    method: str
    attack_type: str
    output_dir: Path
    dataset: DatasetConfig
    runtime: RuntimeConfig
    method_config: dict[str, Any]
    evaluation: dict[str, Any]
    victim: VictimConfig | None
    project_root: Path
    victims: tuple[VictimConfig, ...] = ()

    @property
    def dataset_key(self) -> str:
        return _slug(self.dataset.options.get("namespace", self.dataset.name))

    @property
    def attack_key(self) -> str:
        return _slug(self.attack_type)

    @property
    def namespace(self) -> str:
        return f"{self.dataset_key}-{self.attack_key}"

    @property
    def artifact_dir(self) -> Path:
        return self.project_root / "data" / "artifacts" / self.namespace

    @property
    def target_model_dir(self) -> Path:
        return self.project_root / "targetmodels" / self.namespace

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def load_dataset_config(path: str | Path, project_root: Path) -> DatasetConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    raw = _load_mapping(config_path)

    benign_raw = raw["benign"]
    attacks_raw = raw.get("attacks", {})
    benign = TrafficSource(
        json=_resolve(project_root, benign_raw["json"]),
        pcap=_resolve(project_root, benign_raw.get("pcap")),
        features=_resolve(project_root, benign_raw.get("features")),
    )
    attacks = {
        name: TrafficSource(
            json=_resolve(project_root, source["json"]),
            pcap=_resolve(project_root, source.get("pcap")),
            features=_resolve(project_root, source.get("features")),
        )
        for name, source in attacks_raw.items()
    }
    return DatasetConfig(
        name=raw["name"],
        adapter=raw.get("adapter", "json_flow"),
        benign=benign,
        attacks=attacks,
        sequence_length=int(raw.get("sequence_length", 32)),
        protocol=raw.get("protocol", "tcp"),
        feature_fields=tuple(raw.get("feature_fields", ["iat_ms", "packet_size"])),
        options=dict(raw.get("options", {})),
    )


def load_victim_config(
    value: str | Path | dict[str, Any] | None,
    project_root: Path,
    template_vars: dict[str, str] | None = None,
) -> VictimConfig | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        config_path = Path(value)
        if not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        raw = _load_mapping(config_path)
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise TypeError(f"Unsupported victim configuration: {type(value).__name__}")
    if template_vars is not None:
        raw = _expand_templates(raw, template_vars)

    return VictimConfig(
        name=str(raw.get("name", Path(raw["model_path"]).stem)),
        model_path=_resolve(project_root, raw["model_path"]),
        model_type=str(raw.get("model_type", "auto")).lower(),
        architecture=str(raw.get("architecture", "auto")),
        feature_source=str(raw.get("feature_source", "tabular_csv")).lower(),
        feature_names_path=_resolve(project_root, raw.get("feature_names_path")),
        scaler_path=_resolve(project_root, raw.get("scaler_path")),
        preprocessor_path=_resolve(project_root, raw.get("preprocessor_path")),
        input_path=_resolve(project_root, raw.get("input_path")),
        clean_benign_path=_resolve(project_root, raw.get("clean_benign_path")),
        clean_attack_path=_resolve(project_root, raw.get("clean_attack_path")),
        input_scaled=bool(raw.get("input_scaled", False)),
        label=int(raw.get("label", 1)),
        positive_label=int(raw.get("positive_label", 1)),
        batch_size=int(raw.get("batch_size", 512)),
        device=str(raw.get("device", "auto")),
        model_kwargs=dict(raw.get("model_kwargs", {})),
    )


def load_victim_configs(
    value: list[str | Path | dict[str, Any]] | None,
    project_root: Path,
    template_vars: dict[str, str] | None = None,
) -> tuple[VictimConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("victim_configs must be a list")
    victims = []
    for item in value:
        victim = load_victim_config(item, project_root, template_vars)
        if victim is not None:
            victims.append(victim)
    return tuple(victims)


def load_experiment(path: str | Path) -> ExperimentConfig:
    experiment_path = Path(path).resolve()
    raw = _load_mapping(experiment_path)
    project_root = _find_project_root(experiment_path)
    dataset = load_dataset_config(raw["dataset_config"], project_root)
    experiment_name = raw.get("name", experiment_path.stem)
    method_name = str(raw["method"]).lower()
    attack_type = str(raw["attack_type"]).lower()
    variables = _template_vars(
        dataset=dataset,
        attack_type=attack_type,
        method=method_name,
        experiment_name=experiment_name,
    )
    raw = _expand_templates(raw, variables)

    victim_value = raw.get("victim", raw.get("victim_config"))
    victims = load_victim_configs(
        raw.get("victims", raw.get("victim_configs")), project_root, variables
    )
    victim = load_victim_config(victim_value, project_root, variables)
    if victim is None and victims:
        victim = victims[0]
    if victim is not None and not victims:
        victims = (victim,)
    runtime_raw = raw.get("runtime", {})
    default_output_dir = f"outputs/{variables['namespace']}/{variables['method']}"
    output_dir_value = raw.get("output_dir", default_output_dir)
    output_dir = _resolve(project_root, output_dir_value)
    if output_dir is None:
        raise ValueError("output_dir is required")

    method_config = dict(raw.get("method_config", {}))
    config = ExperimentConfig(
        name=experiment_name,
        method=method_name,
        attack_type=attack_type,
        output_dir=output_dir,
        dataset=dataset,
        runtime=RuntimeConfig(
            device=runtime_raw.get("device", "auto"),
            seed=int(runtime_raw.get("seed", 42)),
            num_workers=int(runtime_raw.get("num_workers", 0)),
        ),
        method_config=method_config,
        evaluation=dict(raw.get("evaluation", {})),
        victim=victim,
        project_root=project_root,
        victims=victims,
    )
    _validate_attack_selection(config)
    return config


def _validate_attack_selection(config: ExperimentConfig) -> None:
    try:
        config.dataset.attack(config.attack_type)
        return
    except ValueError:
        attack_types = config.method_config.get("attack_types")
        if attack_types is None:
            raise
    if isinstance(attack_types, str):
        attack_types = [attack_types]
    if not attack_types:
        raise ValueError("method_config.attack_types must not be empty")
    for attack_type in attack_types:
        config.dataset.attack(str(attack_type).lower())
