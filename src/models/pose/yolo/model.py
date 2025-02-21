import numpy as np

from models.pose.base import BasePoseModel, PoseResult
from models.pose.yolo.yolo import (
    YoloTritonClient,
    Prediction,
    DEFAULT_IMAGE_INPUT_SIZE,
)


class PoseModel(BasePoseModel):
    def __init__(self):
        self.model = YoloTritonClient(
            triton_server_url="127.0.0.1:8001", model_name="yolo", model_version="1"
        )

    def __call__(self, image: np.ndarray) -> PoseResult:
        results: Prediction = self.model.inference(
            image,
            # conf=settings.pose.conf_threshold,
            # iou=settings.pose.iou_threshold,
            # imgsz=settings.pose.img_size,
            # classes=settings.pose.classes,
            # device=settings.device,
        )

        keypoints = results.keypoints.xy
        keypoints[:, :, 0] = (
            keypoints[:, :, 0] * image.shape[1] / DEFAULT_IMAGE_INPUT_SIZE[0]
        )
        keypoints[:, :, 1] = (
            keypoints[:, :, 1] * image.shape[0] / DEFAULT_IMAGE_INPUT_SIZE[1]
        )

        boxes = results.boxes.xyxy
        boxes[:, 0] = boxes[:, 0] * image.shape[1] / DEFAULT_IMAGE_INPUT_SIZE[0]
        boxes[:, 1] = boxes[:, 1] * image.shape[0] / DEFAULT_IMAGE_INPUT_SIZE[1]
        boxes[:, 2] = boxes[:, 2] * image.shape[1] / DEFAULT_IMAGE_INPUT_SIZE[0]
        boxes[:, 3] = boxes[:, 3] * image.shape[0] / DEFAULT_IMAGE_INPUT_SIZE[1]
        return PoseResult(
            keypoints_xy=keypoints,
            keypoints_scores=results.keypoints_scores,
            boxes_xyxy=boxes,
        )
