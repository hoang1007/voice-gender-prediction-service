from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GENDER_", env_file=".env", extra="ignore"
    )

    model_id: str = "prithivMLmods/Common-Voice-Gender-Detection"
    onnx_dir: str = "onnx_model"
    precision: str = "fp16"  # "fp32" | "fp16"
    max_batch_size: int = 32
    batch_timeout: float = (
        0.005  # seconds — keep low for latency-first; raise for throughput
    )
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    target_sample_rate: int = 16000
    log_level: str = "INFO"
    log_dir: str | None = None  # directory for log files; None = stdout only


settings = Settings()
