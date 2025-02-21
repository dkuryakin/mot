import importlib

import numpy as np
from pydantic import BaseModel, ConfigDict


class PoseResult(BaseModel):
    keypoints_xy: np.ndarray  # n x 17 x 2
    keypoints_scores: np.ndarray  # n x 17
    boxes_xyxy: np.ndarray  # n x 4

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class BasePoseModel:
    def __call__(self, image: np.ndarray) -> PoseResult:
        raise NotImplementedError

    @classmethod
    def create(cls, name: str) -> "BasePoseModel":
        model_module = importlib.import_module(f"models.pose.{name}")
        return model_module.PoseModel()
