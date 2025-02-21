import numpy as np

from models.reid.base import ReIDResult, BaseReIDModel
from models.reid.osnet.osnet import Osnet


class ReIDModel(BaseReIDModel):
    def __init__(self):
        self.model = Osnet(url="127.0.0.1:8001", model_name="osnet")

    def __call__(self, image: list[np.ndarray]) -> list[ReIDResult]:
        return [
            ReIDResult(embeddings=embeddings)
            for embeddings in self.model.predict_on_batch(image)
        ]
