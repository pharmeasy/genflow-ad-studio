from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VideoGenerationOutput:
    """Provider-agnostic video generation output."""

    local_paths: list[str]  # Local file paths to downloaded video files
    # Remote URIs usable by QC (e.g. GCS URIs for Veo).
    # When empty, VideoService uploads local files to GCS for QC.
    remote_uris: list[str] = field(default_factory=list)


class VideoProvider(ABC):
    """Abstract base class for video generation providers.

    Each provider handles its own image upload mechanism (GCS for Veo,
    base64/HTTP for Seedance, etc.) and downloads generated videos locally.
    """

    @abstractmethod
    async def generate_videos(
        self,
        prompt: str,
        storyboard_image_path: str,
        asset_image_paths: list[str] | None,
        output_dir: str,
        num_variants: int = 1,
        seed: int | None = None,
        resolution: str = "720p",
        aspect_ratio: str = "9:16",
        duration_seconds: int = 8,
        generate_audio: bool = True,
        negative_prompt: str = "",
        model_id: str | None = None,
        compression_quality: str = "optimized",
        use_reference_images: bool = True,
    ) -> VideoGenerationOutput:
        """Generate video variants and download them locally.

        Args:
            prompt: Video generation prompt.
            storyboard_image_path: Local path to the storyboard/first-frame image.
            asset_image_paths: Local paths to reference images (avatar, product, prev frame).
            output_dir: Directory to save generated video files as variant_0.mp4, variant_1.mp4, etc.
            num_variants: Number of video variants to generate.
            seed: Random seed for reproducibility.
            resolution: Video resolution (e.g., "720p", "1080p").
            aspect_ratio: Video aspect ratio (e.g., "16:9", "9:16").
            duration_seconds: Video duration in seconds.
            generate_audio: Whether to generate audio.
            negative_prompt: Elements to avoid.
            model_id: Specific model ID override.
            compression_quality: Video compression quality.
            use_reference_images: Use reference images for consistency.

        Returns:
            VideoGenerationOutput with local file paths and optional remote URIs.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
        ...
