from pydantic_settings import BaseSettings, SettingsConfigDict


class MlflowSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MOT__MLFLOW__", env_nested_delimiter="__"
    )

    uri: str = "https://platform.clapeyron.digital/mlflow/"
    username: str = "deployer"
    password: str = "U0KjTPPDJE"
    experiment_id: int = 16
    run_name: str = "tracker"
    git_base_url: str = "https://github.com/dkuryakin/mot"


class PoseSettings(BaseSettings):
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: tuple[int, int] = (640, 640)
    classes: tuple[int, ...] = (0,)


class TrackerSettings(BaseSettings):
    # tracker
    past_detections_length: int = 50
    init_delay: int = 8
    hit_counter_max: int = 16
    pointwise_hit_max: int = 4
    reid_hit_counter_max: int = 1000

    # tracker distance
    distance_threshold: float = 0.8
    keypoint_scale_factor: float = 40.0

    # reid distance
    max_dist_same_reid: float = 0.25


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MOT__", env_nested_delimiter="__"
    )

    mlflow: MlflowSettings = MlflowSettings()
    pose: PoseSettings = PoseSettings()
    tracker: TrackerSettings = TrackerSettings()

    # pose_model: str = "yolo11m_pose__opensource"
    # reid_model: str = "osnet_x1_0__opensource"

    pose_model: str = "yolo"
    reid_model: str = "osnet"

    device: str = "cpu"
    debug: bool = True
    visible: bool = False


settings = Settings()
