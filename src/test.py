import os.path
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import git
import mlflow

from matcher import Track, match_tracks
from settings import settings
from tracker import PersonTracker


@dataclass
class Metrics:
    metrics_iou: dict[str, float]
    metrics_count: dict[str, float]
    score_iou: float
    score_count: float
    logs: str


def video_to_intervals(name: str) -> None:
    video_filename = f"videos/{name}.mp4"
    intervals_filename = f"markup/predict/{name}.yaml"
    if os.path.exists(intervals_filename):
        return
    tracker = PersonTracker(video_filename)
    tracker.process_video()
    tracker.save_intervals(intervals_filename)


def list_video_names() -> list[str]:
    names = []
    for fname in os.listdir("videos"):
        if "__out" in fname:
            continue
        name = fname.removesuffix(".mp4")
        names.append(name)
    return names


def calculate_metrics() -> Metrics:
    matches = []
    os.makedirs("markup/predict", exist_ok=True)
    metrics = Metrics(
        metrics_iou={}, metrics_count={}, score_iou=0, score_count=0, logs=""
    )
    logs = []
    for fname in os.listdir("markup/predict"):
        tracks_b = Track.from_file(f"markup/gt/{fname}")
        tracks_a = Track.from_file(f"markup/predict/{fname}")
        info = match_tracks(tracks_a, tracks_b)
        score_count = 1 - max(abs(len(tracks_a) - len(tracks_b)) / len(tracks_b), 1)
        metrics.metrics_iou[fname] = info.score
        metrics.metrics_count[fname] = score_count
        logs.append(f"{fname} -> iou={info.score:0.3f} count={score_count:0.3f}")
        logs.append("")
        matches.append(info)
        logs.append("    " + str(info).replace("\n", "\n    "))
        logs.append("")
        logs.append("-" * 40)
        logs.append("")

    score_iou = sum(metrics.metrics_iou.values()) / len(metrics.metrics_iou)
    metrics.score_iou = score_iou
    logs.append(f"Avg IOU score = {score_iou:0.3f}")

    score_count = sum(metrics.metrics_count.values()) / len(metrics.metrics_count)
    metrics.score_count = score_count
    logs.append(f"Avg COUNT score = {score_count:0.3f}")

    logs.append("")

    metrics.logs = "\n".join(logs)

    return metrics


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow.password

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    mlflow.set_tracking_uri(settings.mlflow.uri)

    with mlflow.start_run(
        experiment_id=str(settings.mlflow.experiment_id),
        run_name=f"{settings.mlflow.run_name}: {settings.pose_model} + {settings.reid_model}",
        tags={
            "mlflow.source.name": f"{settings.mlflow.git_base_url}/../../../tree/{sha}"
        },
    ):
        mlflow.log_param("pose_model", settings.pose_model)
        mlflow.log_param("reid_model", settings.reid_model)

        names = list_video_names()
        with ProcessPoolExecutor() as executor:
            executor.map(video_to_intervals, names)

        metrics = calculate_metrics()

        mlflow.log_metric("IOU: Overall Score", metrics.score_iou)
        mlflow.log_metric("COUNT: Overall Score", metrics.score_count)

        for metric_name, metric_value in metrics.metrics_iou.items():
            mlflow.log_metric("IOU: " + metric_name.removesuffix(".yaml"), metric_value)
        for metric_name, metric_value in metrics.metrics_count.items():
            mlflow.log_metric(
                "COUNT: " + metric_name.removesuffix(".yaml"), metric_value
            )

        mlflow.log_param("logs", metrics.logs)
