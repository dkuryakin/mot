import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from models.pose.base import BasePoseModel, PoseResult
from settings import settings


class PoseModel(BasePoseModel):
    def __init__(self):
        self.model = YOLO("yolo11m-pose.pt")

    def __call__(self, image: np.ndarray) -> PoseResult:
        results: Results = self.model(
            image,
            conf=settings.pose.conf_threshold,
            iou=settings.pose.iou_threshold,
            imgsz=settings.pose.img_size,
            classes=settings.pose.classes,
            device=settings.device,
        )[0]

        return PoseResult(
            keypoints_xy=results.keypoints.xy.cpu().numpy(),
            keypoints_scores=results.keypoints.conf.cpu().numpy(),
            boxes_xyxy=results.boxes.xyxy.cpu().numpy(),
        )
