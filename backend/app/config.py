from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VideoModelOption(BaseModel):
    id: str
    label: str
    description: str


def _find_env_file() -> str | None:
    """Locate the .env file relative to the backend directory."""
    for candidate in [
        Path(__file__).resolve().parent.parent.parent / ".env",  # backend/app -> root
        Path.cwd().parent / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.is_file():
            return str(candidate)
    return None


class Settings(BaseSettings):
    project_id: str = ""
    region: str = "global"
    gcs_bucket_name: str = ""

    gemini_model: str = "gemini-3-flash-preview"
    gemini_flash_model: str = "gemini-3-flash-preview"
    image_model: str = "gemini-3-pro-image-preview"
    imagen_model: str = "imagen-4.0-generate-001"

    veo_models: list[VideoModelOption] = [
        VideoModelOption(
            id="veo-3.1-generate-preview",
            label="Veo 3.1 Preview",
            description="Standard — Best quality",
        ),
        VideoModelOption(
            id="veo-3.1-fast-generate-preview",
            label="Veo 3.1 Fast Preview",
            description="Faster generation",
        ),
    ]

    # Seedance 2.0 (ByteDance) settings
    seedance_api_key: str = ""
    seedance_api_base_url: str = "https://api.byteplusapi.com"
    seedance_models: list[VideoModelOption] = [
        VideoModelOption(
            id="seedance-2-0-260128",
            label="Seedance 2.0",
            description="Standard — Best quality",
        ),
        VideoModelOption(
            id="seedance-2-0-fast-260128",
            label="Seedance 2.0 Fast",
            description="Faster generation",
        ),
    ]

    @property
    def default_video_model(self) -> str:
        """First Veo model ID, used as the default when no model is specified."""
        return self.veo_models[0].id if self.veo_models else ""

    @property
    def default_seedance_model(self) -> str:
        """First Seedance model ID."""
        return self.seedance_models[0].id if self.seedance_models else ""

    output_dir: str = "output"
    storyboard_qc_threshold: int = 60
    video_qc_threshold: int = 6
    max_regen_attempts: int = 3
    max_video_variants: int = 4
    max_avatar_variants: int = 4
    max_concurrent_scenes: int = 5

    # Video generation settings
    default_video_duration: int = 8
    default_video_compression: str = "optimized"
    max_video_qc_regen_attempts: int = 2

    # Script generation settings
    script_default_scene_count: int = 3
    script_max_scene_count: int = 6
    script_min_scene_count: int = 2
    script_default_total_duration: int = 30
    script_max_dialogue_words_per_scene: int = 25

    model_config = {
        "env_file": _find_env_file(),
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
