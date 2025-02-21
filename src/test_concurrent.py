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
    metrics: dict[str, float]
    score: float
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
    metrics = Metrics(metrics={}, score=0, logs="")
    logs = []
    for fname in os.listdir("markup/predict"):
        tracks_b = Track.from_file(f"markup/gt/{fname}")
        tracks_a = Track.from_file(f"markup/predict/{fname}")
        info = match_tracks(tracks_a, tracks_b)
        metrics.metrics[fname] = info.score
        logs.append(f"{fname} -> {info.score:0.3f}")
        logs.append("")
        matches.append(info)
        logs.append("    " + str(info).replace("\n", "\n    "))
        logs.append("")
        logs.append("-" * 40)
        logs.append("")

    score = sum([m.score for m in matches]) / len(matches)
    metrics.score = score
    logs.append(f"Avg score = {score:0.3f}")
    logs.append("")

    metrics.logs = "\n".join(logs)

    return metrics


def process_video(name: str) -> None:
    video_to_intervals(name)
    metrics = calculate_metrics()
    mlflow.log_metric("Overall Score", metrics.score)
    for metric_name, metric_value in metrics.metrics.items():
        mlflow.log_metric(metric_name.removesuffix(".yaml"), metric_value)
        mlflow.log_param("logs", metrics.logs)


if __name__ == "__main__":
    workers = int(sys.argv[1])

    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow.password

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    mlflow.set_tracking_uri(settings.mlflow.uri)

    with mlflow.start_run(
        experiment_id=str(settings.mlflow.experiment_id),
        run_name=f"{settings.mlflow.run_name}",
        tags={
            "mlflow.source.name": f"{settings.mlflow.git_base_url}/../../../tree/{sha}"
        },
    ):
        names = list_video_names()
        with ProcessPoolExecutor() as executor:
            executor.map(process_video, names)
