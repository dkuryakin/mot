import json
import os.path
import sys
from collections import defaultdict

import cv2
import norfair
import numpy as np
import yaml
from loguru import logger
from norfair import Detection, Tracker, Video
from norfair.distances import create_keypoints_voting_distance
from norfair.tracker import TrackedObject

from models.pose.base import BasePoseModel, PoseResult
from models.reid.base import BaseReIDModel
from settings import settings


class PersonTracker:
    def __init__(self, input_path: str | int):
        self.input_path = input_path

        self.pose_model = BasePoseModel.create(settings.pose_model)
        self.reid_model = BaseReIDModel.create(settings.reid_model)

        if isinstance(input_path, int):
            self.video = Video(camera=input_path)
        else:
            self.video = Video(
                input_path=input_path,
                output_path=input_path + "__out.mp4" if settings.debug else None,
            )
        self._init_tracker()

        self._intervals: dict[str | int, list[list[int]]] = defaultdict(list[list[int]])

    def _init_tracker(self) -> None:
        keypoint_thresh = (
            self.video.input_height / settings.tracker.keypoint_scale_factor
        )
        self.tracker = Tracker(
            distance_function=create_keypoints_voting_distance(
                keypoint_distance_threshold=keypoint_thresh,
                detection_threshold=settings.pose.conf_threshold,
            ),
            distance_threshold=settings.tracker.distance_threshold,
            detection_threshold=settings.pose.conf_threshold,
            initialization_delay=settings.tracker.init_delay,
            hit_counter_max=settings.tracker.hit_counter_max,
            pointwise_hit_counter_max=settings.tracker.pointwise_hit_max,
            reid_distance_function=self.reid_distance,
            reid_distance_threshold=settings.tracker.max_dist_same_reid,
            reid_hit_counter_max=settings.tracker.reid_hit_counter_max,
            past_detections_length=settings.tracker.past_detections_length,
        )

    def process_video(self) -> None:
        try:
            for frame_number, frame in enumerate(self.video):
                processed_frame = self._process_frame(
                    frame_number=frame_number, frame=frame
                )
                if settings.debug:
                    self.video.write(processed_frame)
                    cv2.imshow(
                        "Tracking", cv2.resize(processed_frame, None, fx=0.75, fy=0.75)
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except:
            if isinstance(self.input_path, str) and os.path.exists(
                self.input_path + "__out.mp4"
            ):
                os.remove(self.input_path + "__out.mp4")
            raise

    def _process_frame(self, frame_number: int, frame: np.ndarray) -> np.ndarray:
        pose_results = self.pose_model(frame)
        detections = self._create_detections(frame, pose_results)
        tracked_objects = self.tracker.update(detections=detections)

        for obj in tracked_objects:
            obj_id = obj.id
            if (
                obj_id in self._intervals
                and self._intervals[obj_id]
                and self._intervals[obj_id][-1][1] >= frame_number - 1
            ):
                self._intervals[obj_id][-1][1] = frame_number
            else:
                self._intervals[obj_id].append([frame_number, frame_number])

        if settings.debug:
            return self._draw_results(frame, tracked_objects)

        return frame

    def _create_detections(
        self, frame: np.ndarray, results: PoseResult
    ) -> list[Detection]:
        detections = []
        crops = {}

        if results.keypoints_scores is None:
            return []

        for idx, (kps, conf, bbox) in enumerate(
            zip(
                results.keypoints_xy,
                results.keypoints_scores,
                results.boxes_xyxy,
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
            features = self.reid_model(list(crops.values()))
            for idx, feat in zip(crops.keys(), features):
                detections[idx].embedding["reid"] = feat.embeddings

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

    def save_intervals(self, filename: str) -> None:
        with open(filename, "w") as f:
            if filename.lower().endswith(".json"):
                json.dump(self._intervals, f)  # type: ignore
            elif filename.lower().endswith(".yml") or filename.lower().endswith(
                ".yaml"
            ):
                yaml.safe_dump(dict(self._intervals), f)
            else:
                raise ValueError("filename must be json or yaml")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        path: str | int = 0
    else:
        path = sys.argv[1]
    tracker = PersonTracker(path)
    tracker.process_video()
