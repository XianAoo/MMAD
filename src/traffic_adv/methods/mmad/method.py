from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from traffic_adv.data.mapping import apply_feature_sequences
from traffic_adv.methods.base import AttackMethod
from traffic_adv.methods.common import load_attack_flows, maybe_progress, sample_flows
from traffic_adv.methods.mmad.preprocessing import MMADPreprocessor
from traffic_adv.methods.mmad.profiling import MMADGenerationProfiler


class MMADMethod(AttackMethod):
    @property
    def preprocessor_path(self):
        return self.config.output_dir / "preprocessor.pkl"

    @property
    def model_path(self):
        return self.config.checkpoint_dir / "diffusion.pt"

    def generation_preprocessor_path(self):
        return self._artifact_path("preprocessor_path", self.preprocessor_path)

    def generation_model_path(self):
        return self._artifact_path("diffusion_model_path", self.model_path)

    def _surrogate_path(self, name: str):
        configured = self.config.method_config.get("surrogate_checkpoint_paths")
        if isinstance(configured, dict) and name in configured:
            return self._resolve_path(configured[name])
        return self.config.checkpoint_dir / f"surrogate_{name}.pt"

    def _artifact_path(self, key: str, default):
        value = self.config.method_config.get(key)
        return self._resolve_path(value) if value is not None else default

    def _resolve_path(self, value):
        path = Path(str(value))
        return path if path.is_absolute() else (self.config.project_root / path).resolve()

    def settings(self):
        defaults = {
            "sequence_length": self.config.dataset.sequence_length,
            "preprocessor_quantile": 0.99,
            "training_attack_types": None,
            "train_noisy_surrogates": True,
            "unet_dim": 64,
            "unet_dim_mults": [1, 2, 4],
            "unet_dropout": 0.1,
            "timesteps": 1000,
            "sampling_timesteps": 500,
            "beta_schedule": "cosine",
            "batch_size": 64,
            "learning_rate": 1e-4,
            "train_steps": 20000,
            "save_every": 5000,
            "surrogate_epochs": 30,
            "surrogate_learning_rate": 0.001,
            "surrogates": ["mlp", "cnn", "lstm", "transformer"],
            "lambda_all": 5.0,
            "lambda_target": 1.0,
            "budget": 0.0,
            "update_k": 0.02,
            "escape_threshold": 0.5,
            "training_benign_sample_ratio": None,
            "training_attack_sample_ratio": None,
            "attack_sample_ratio": None,
            "max_training_benign_flows": None,
            "max_training_attack_flows": None,
            "max_attack_flows": None,
            "sample_seed": self.config.runtime.seed,
            "progress": True,
            "progress_leave": True,
            "attack_step_progress": False,
            "attack_step_progress_leave": False,
            "profile_generation": False,
            "profile_warmup_batches": 0,
            "profile_power_interval_s": 0.2,
            "profile_record_batches": True,
            "profile_output": None,
        }
        defaults.update(self.config.method_config)
        return defaults

    def train(self, dataset) -> None:
        import torch

        from traffic_adv.data.torch_data import TrafficTensorDataset
        from traffic_adv.methods.mmad.diffusion import GaussianDiffusion1D
        from traffic_adv.methods.mmad.models.unet import Unet1D
        from traffic_adv.methods.mmad.trainer import Trainer1D

        self.config.ensure_output_dirs()
        settings = self.settings()
        device = self.device(torch)
        sequence_length = int(settings["sequence_length"])
        benign, malicious = self._load_training_flows(dataset, settings)
        preprocessor = MMADPreprocessor(
            sequence_length=sequence_length,
            quantile=float(settings["preprocessor_quantile"]),
        )
        preprocessor.fit([*benign, *malicious]).save(self.preprocessor_path)
        benign_values = preprocessor.transform(benign)
        malicious_values = preprocessor.transform(malicious) if malicious else None
        values = benign_values if malicious_values is None else np.concatenate([benign_values, malicious_values])
        labels = self._labels(len(benign_values), 0)
        if malicious_values is not None:
            labels = np.concatenate([labels, self._labels(len(malicious_values), 1)])

        model = Unet1D(
            dim=int(settings["unet_dim"]),
            dim_mults=tuple(settings["unet_dim_mults"]),
            channels=3,
            dropout=float(settings["unet_dropout"]),
        ).to(device)
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=sequence_length,
            timesteps=int(settings["timesteps"]),
            objective="pred_noise",
            beta_schedule=settings["beta_schedule"],
        ).to(device)
        train_dataset = TrafficTensorDataset(values, labels)
        trainer = Trainer1D(
            diffusion,
            train_dataset,
            train_batch_size=int(settings["batch_size"]),
            train_lr=float(settings["learning_rate"]),
            train_num_steps=int(settings["train_steps"]),
            results_folder=str(self.config.checkpoint_dir),
            save_and_sample_every=int(settings["save_every"]),
        )
        trainer.train()
        torch.save(diffusion.state_dict(), self.model_path)
        if bool(settings["train_noisy_surrogates"]):
            self._train_surrogates(torch, diffusion, train_dataset, settings, device)

    def generate(self, dataset):
        import torch
        from torch.utils.data import DataLoader

        from traffic_adv.data.torch_data import TrafficTensorDataset
        from traffic_adv.methods.mmad.attacker import DMADVAttacker
        from traffic_adv.methods.mmad.diffusion import GaussianDiffusion1D
        from traffic_adv.methods.mmad.models.unet import Unet1D

        settings = self.settings()
        device = self.device(torch)
        profiler = (
            MMADGenerationProfiler(self.config, settings, torch, device)
            if MMADGenerationProfiler.enabled(settings)
            else None
        )
        sequence_length = int(settings["sequence_length"])
        try:
            preprocessor = MMADPreprocessor.load(self.generation_preprocessor_path())
            malicious = load_attack_flows(
                dataset,
                self.config.attack_type,
                settings,
                "max_attack_flows",
                ratio_key="attack_sample_ratio",
                seed=int(settings["sample_seed"]) + 200,
            )
            values = preprocessor.transform(malicious)
            model = Unet1D(
                dim=int(settings["unet_dim"]),
                dim_mults=tuple(settings["unet_dim_mults"]),
                channels=3,
                dropout=float(settings["unet_dropout"]),
            ).to(device)
            diffusion = GaussianDiffusion1D(
                model,
                seq_length=sequence_length,
                timesteps=int(settings["timesteps"]),
                beta_schedule=settings["beta_schedule"],
            ).to(device)
            diffusion.load_state_dict(torch.load(self.generation_model_path(), map_location=device))
            diffusion.eval()
            surrogates = self._load_surrogates(torch, settings, device)
            attack_config = SimpleNamespace(
                DEVICE=device,
                TIMESTEPS=int(settings["timesteps"]),
                SAMPLING_TIMESTEPS=int(settings["sampling_timesteps"]),
                INIT_WEIGHTS={
                    name.upper() if name != "transformer" else "Transformer": 1.0
                    / len(surrogates)
                    for name in settings["surrogates"]
                },
                LAMBDA_ALL=float(settings["lambda_all"]),
                LAMBDA_TAR=float(settings["lambda_target"]),
                BUDGET=float(settings["budget"]),
                UPDATE_K=float(settings["update_k"]),
                ESCAPE_THRESHOLD=float(settings["escape_threshold"]),
                PROGRESS=bool(settings["progress"]),
                PROGRESS_LEAVE=bool(settings["progress_leave"]),
                ATTACK_STEP_PROGRESS=bool(settings["attack_step_progress"]),
                ATTACK_STEP_PROGRESS_LEAVE=bool(settings["attack_step_progress_leave"]),
            )
            attacker = DMADVAttacker(diffusion, surrogates, attack_config)
            loader = DataLoader(
                TrafficTensorDataset(values),
                batch_size=int(settings["batch_size"]),
                shuffle=False,
                num_workers=self.config.runtime.num_workers,
            )
            if profiler is not None:
                profiler.set_metadata(
                    attack_flows=len(malicious),
                    tensor_shape=list(values.shape),
                    batch_size=int(settings["batch_size"]),
                    sequence_length=sequence_length,
                    timesteps=int(settings["timesteps"]),
                    sampling_timesteps=int(settings["sampling_timesteps"]),
                    surrogates=list(settings["surrogates"]),
                    model_path=self.generation_model_path(),
                    preprocessor_path=self.generation_preprocessor_path(),
                )
            batches = maybe_progress(
                loader,
                settings,
                desc="[MMAD] attack",
                total=len(loader),
            )
            generated = []
            if profiler is not None:
                profiler.start_core()
            for batch_index, (batch, _) in enumerate(batches):
                step_desc = f"[MMAD] batch {batch_index + 1}/{len(loader)} steps"
                if profiler is None:
                    adversarial = attacker.attack(batch.to(device), desc=step_desc)
                else:
                    adversarial = profiler.measure_batch(
                        len(batch),
                        lambda batch=batch, step_desc=step_desc: attacker.attack(
                            batch.to(device), desc=step_desc
                        ),
                    )
                generated.append(adversarial.cpu().numpy())
            if profiler is not None:
                profiler.finish_core()
            normalized = np.concatenate(generated) if generated else values
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
            iats, sizes, _ = preprocessor.inverse_transform(normalized)
            flows = apply_feature_sequences(malicious, iats, sizes)
            if profiler is not None:
                profiler.set_metadata(generated_flows=len(flows))
                profiler.save()
            return flows
        except Exception as exc:
            if profiler is not None:
                profiler.save(error=exc)
            raise

    def device(self, torch):
        if self.config.runtime.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.runtime.device

    def _load_training_flows(self, dataset, settings):
        benign = sample_flows(
            dataset.load_benign(),
            settings,
            "max_training_benign_flows",
            ratio_key="training_benign_sample_ratio",
            seed=int(settings["sample_seed"]),
        )
        if not benign:
            raise ValueError("MMAD requires at least one benign training flow")
        malicious = []
        for index, attack_type in enumerate(self._training_attack_types(settings)):
            flows = dataset.load_attack(attack_type)
            malicious.extend(
                sample_flows(
                    flows,
                    settings,
                    "max_training_attack_flows",
                    ratio_key="training_attack_sample_ratio",
                    seed=int(settings["sample_seed"]) + 100 + index,
                )
            )
        return benign, malicious

    def _training_attack_types(self, settings) -> list[str]:
        configured = settings.get("training_attack_types")
        if configured is None:
            configured = settings.get("attack_types")
        if configured is None:
            return [self.config.attack_type]
        if isinstance(configured, str):
            configured = [configured]
        return [str(value).lower() for value in configured]

    @staticmethod
    def _labels(size: int, label: int) -> np.ndarray:
        return np.full(size, label, dtype=np.int64)

    def _train_surrogates(self, torch, diffusion, train_dataset, settings, device):
        from torch.utils.data import DataLoader

        from traffic_adv.methods.mmad.models.classifiers import get_classifier

        loader = DataLoader(
            train_dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            num_workers=self.config.runtime.num_workers,
        )
        diffusion.eval()
        sequence_length = int(settings["sequence_length"])
        for name in settings["surrogates"]:
            classifier = get_classifier(
                name,
                device=device,
                sequence_length=sequence_length,
                in_channels=3,
            )
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=float(settings["surrogate_learning_rate"]),
                weight_decay=1e-4,
            )
            criterion = torch.nn.CrossEntropyLoss()
            epochs = maybe_progress(
                range(int(settings["surrogate_epochs"])),
                settings,
                desc=f"[MMAD] surrogate {name}",
            )
            for _ in epochs:
                classifier.train()
                for values, labels in loader:
                    values, labels = values.to(device), labels.to(device)
                    times = torch.randint(
                        0, int(settings["timesteps"]), (len(values),), device=device
                    )
                    noisy = diffusion.q_sample(
                        x_start=values, t=times, noise=torch.randn_like(values)
                    )
                    optimizer.zero_grad()
                    loss = criterion(classifier(noisy, times), labels)
                    loss.backward()
                    optimizer.step()
            torch.save(
                classifier.state_dict(),
                self.config.checkpoint_dir / f"surrogate_{name}.pt",
            )

    def _load_surrogates(self, torch, settings, device):
        from traffic_adv.methods.mmad.models.classifiers import get_classifier

        models = {}
        display_names = {
            "mlp": "MLP",
            "cnn": "CNN",
            "lstm": "LSTM",
            "transformer": "Transformer",
        }
        sequence_length = int(settings["sequence_length"])
        for name in settings["surrogates"]:
            checkpoint = self._surrogate_path(name)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing MMAD noisy surrogate checkpoint: {checkpoint}. "
                    "Train with method_config.train_noisy_surrogates=true or provide this checkpoint."
                )
            model = get_classifier(
                name,
                device=device,
                sequence_length=sequence_length,
                in_channels=3,
            )
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            models[display_names[name]] = model
        return models
