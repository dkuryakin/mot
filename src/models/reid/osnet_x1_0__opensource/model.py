import numpy as np
from torchreid.reid.utils import FeatureExtractor

from models.reid.base import ReIDResult, BaseReIDModel
from settings import settings


class ReIDModel(BaseReIDModel):
    def __init__(self):
        self.model = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="osnet_x1_0_imagenet.pth",
            verbose=False,
            device=settings.device,
        )

    def __call__(self, image: list[np.ndarray]) -> list[ReIDResult]:
        return [
            ReIDResult(embeddings=embeddings)
            for embeddings in self.model(image).cpu().numpy()
        ]
