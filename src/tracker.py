import os.path

from loguru import logger
import sys
from pydantic_settings import BaseSettings
import cv2
import norfair
import numpy as np
from norfair import Detection, Tracker, Video
from norfair.distances import create_keypoints_voting_distance
from norfair.tracker import TrackedObject
from torchreid.reid.utils import FeatureExtractor
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Config(BaseSettings):
    # yolo detector
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: tuple[int, int] = (640, 640)
    classes: tuple[int, ...] = (0,)

    # tracker
    past_detections_length: int = 50
    init_delay: int = 8
    hit_counter_max: int = 16
    pointwise_hit_max: int = 4

    # tracker distance
    distance_threshold: float = 0.8
    keypoint_scale_factor: float = 40.0

    # reid distance
    max_dist_same_reid: float = 0.25


config = Config()


class PersonTracker:
    def __init__(self, input_path: str | int):
        self.input_path = input_path

        self.yolo = YOLO("./checkpoints/yolo11m-pose.pt")

        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="./checkpoints/osnet_x1_0_imagenet.pth",
            verbose=False,
        )

        if isinstance(input_path, int):
            self.video = Video(camera=input_path)
        else:
            self.video = Video(
                input_path=input_path, output_path=input_path + "__out.mp4"
            )
        self._init_tracker()

    def _init_tracker(self) -> None:
        keypoint_thresh = self.video.input_height / config.keypoint_scale_factor
        self.tracker = Tracker(
            distance_function=create_keypoints_voting_distance(
                keypoint_distance_threshold=keypoint_thresh,
                detection_threshold=config.conf_threshold,
            ),
            distance_threshold=config.distance_threshold,
            detection_threshold=config.conf_threshold,
            initialization_delay=config.init_delay,
            hit_counter_max=config.hit_counter_max,
            pointwise_hit_counter_max=config.pointwise_hit_max,
            reid_distance_function=self.reid_distance,
            reid_distance_threshold=config.max_dist_same_reid,
            reid_hit_counter_max=1000,
            past_detections_length=config.past_detections_length,
        )

    def process_video(self) -> None:
        for frame in self.video:
            processed_frame = self._process_frame(frame)
            self.video.write(processed_frame)
            cv2.imshow("Tracking", cv2.resize(processed_frame, None, fx=0.75, fy=0.75))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        yolo_results = self._run_yolo_detection(frame)
        detections = self._create_detections(frame, yolo_results)
        tracked_objects = self.tracker.update(detections=detections)
        return self._draw_results(frame, tracked_objects)

    def _run_yolo_detection(self, frame: np.ndarray) -> Results:
        return self.yolo(
            frame,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            imgsz=config.img_size,
            classes=config.classes,
        )[0]

    def _create_detections(
        self, frame: np.ndarray, results: Results
    ) -> list[Detection]:
        detections = []
        crops = {}

        if results.keypoints.conf is None:
            return []

        for idx, (kps, conf, bbox) in enumerate(
            zip(
                results.keypoints.xy.cpu().numpy(),
                results.keypoints.conf.cpu().numpy(),
                results.boxes.xyxy.cpu().numpy(),
            )
        ):
            conf[np.all(kps == [0, 0], axis=1)] = 0
            crop = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            crops[idx] = crop
            detections.append(
                Detection(
                    points=kps,
                    scores=conf,
                    embedding={
                        "reid": None,
                    },
                )
            )

        if crops:
            features = self.extractor(list(crops.values())).cpu().numpy()
            for idx, feat in zip(crops.keys(), features):
                detections[idx].embedding["reid"] = feat

        return detections

    @staticmethod
    def _draw_results(frame: np.ndarray, objects: list) -> np.ndarray:
        bboxes = []
        for obj in objects:
            points = obj.estimate[obj.live_points].astype(int)
            x, y, w, h = cv2.boundingRect(points)
            bboxes.append(
                Detection(
                    points=np.array([[x, y], [x + w, y + h]]),
                    label=obj.id,
                )
            )
        return norfair.draw_boxes(frame, bboxes, color_by_label=True, draw_labels=True)

    @staticmethod
    def reid_distance(a: TrackedObject, b: TrackedObject) -> float:
        distances = []
        a_embs = [d.embedding["reid"] for d in a.past_detections]
        b_embs = [d.embedding["reid"] for d in b.past_detections]

        for emb_a in a_embs:
            for emb_b in b_embs:
                distances.append(PersonTracker._cosine_distance(emb_a, emb_b))

        dist = min(distances) if distances else 1.0
        logger.info(f"REID DIST [{b.id}] {dist}")
        return dist

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        path: str | int = 0
    else:
        path = sys.argv[1]
    tracker = PersonTracker(path)
    tracker.process_video()
