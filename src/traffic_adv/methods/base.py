from __future__ import annotations

from abc import ABC, abstractmethod

from traffic_adv.config import ExperimentConfig
from traffic_adv.data.adapters.base import DatasetAdapter
from traffic_adv.data.schema import TrafficFlow


class AttackMethod(ABC):
    def __init__(self, config: ExperimentConfig):
        self.config = config

    @abstractmethod
    def train(self, dataset: DatasetAdapter) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, dataset: DatasetAdapter) -> list[TrafficFlow]:
        raise NotImplementedError

