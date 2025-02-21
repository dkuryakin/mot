import importlib

import numpy as np
from pydantic import BaseModel, ConfigDict


class ReIDResult(BaseModel):
    embeddings: np.ndarray  # n

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class BaseReIDModel:
    def __call__(self, image: list[np.ndarray]) -> list[ReIDResult]:
        raise NotImplementedError

    @classmethod
    def create(cls, name: str) -> "BaseReIDModel":
        model_module = importlib.import_module(f"models.reid.{name}")
        return model_module.ReIDModel()
