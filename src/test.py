import os.path
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


def video_to_intervals(video_filename: str, intervals_filename: str) -> None:
    if os.path.exists(intervals_filename):
        return
    tracker = PersonTracker(video_filename)
    tracker.process_video()
    tracker.save_intervals(intervals_filename)


def videos_to_intervals() -> None:
    for fname in os.listdir("videos"):
        if "__out" in fname:
            continue
        name = fname.removesuffix(".mp4")
        video_to_intervals(f"videos/{name}.mp4", f"markup/predict/{name}.yaml")


def calculate_metrics(fname: str) -> Metrics:
    matches = []
    os.makedirs("markup/predict", exist_ok=True)
    metrics = Metrics(metrics={}, score=0)
    with open(fname, "w") as f:
        for fname in os.listdir("markup/predict"):
            tracks_b = Track.from_file(f"markup/gt/{fname}")
            tracks_a = Track.from_file(f"markup/predict/{fname}")
            info = match_tracks(tracks_a, tracks_b)
            metrics.metrics[fname] = info.score
            print(f"{fname} -> {info.score:0.3f}", file=f)
            print(file=f)
            matches.append(info)
            print("    " + str(info).replace("\n", "\n    "), file=f)
            print(file=f)
            print("-" * 40, file=f)
            print(file=f)

        score = sum([m.score for m in matches]) / len(matches)
        metrics.score = score
        print(f"Avg score = {score:0.3f}", file=f)
        print(file=f)

        return metrics


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_password

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    mlflow.set_tracking_uri(settings.mlflow_uri)

    with mlflow.start_run(
        experiment_id=str(settings.mlflow_experiment_id),
        run_name=f"{settings.mlflow_run_prefix}",
        tags={"mlflow.source.name": f"{settings.git_base_url}/../../../tree/{sha}"},
    ):
        videos_to_intervals()
        metrics = calculate_metrics("metrics.log")

        mlflow.log_metric("Overall Score", metrics.score)
        for metric_name, metric_value in metrics.metrics.items():
            mlflow.log_metric(metric_name.removesuffix(".yaml"), metric_value)

        with open("metrics.log") as f:
            logs = f.read()
            mlflow.log_param("logs", logs)

        # mlflow.log_artifact("metrics.log")
