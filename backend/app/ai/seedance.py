"""Seedance 2.0 video generation provider (ByteDance).

Uses the BytePlus ModelArk REST API.  The base URL, API key, and model ID
are all configurable via environment variables so the same implementation
works with BytePlus, fal.ai, or compatible API proxies.

Env vars:
    SEEDANCE_API_KEY       – Bearer token for authentication.
    SEEDANCE_API_BASE_URL  – REST endpoint base (default: BytePlus ModelArk).
    SEEDANCE_MODEL         – Model ID (default: seedance-2-0-260128).
"""

import asyncio
import base64
import logging
import random
from pathlib import Path

import httpx

from app.ai.video_provider import VideoGenerationOutput, VideoProvider
from app.config import Settings

logger = logging.getLogger(__name__)

# Seedance accepts discrete duration values only.
_VALID_DURATIONS = {5, 10, 15}


def _snap_duration(seconds: int) -> int:
    """Map an arbitrary duration to the nearest valid Seedance value."""
    return min(_VALID_DURATIONS, key=lambda v: abs(v - seconds))


def _snap_resolution(resolution: str) -> str:
    """Seedance supports 480p and 720p only."""
    if resolution in ("480p", "720p"):
        return resolution
    return "720p"


def _image_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class SeedanceProvider(VideoProvider):
    """VideoProvider backed by the Seedance 2.0 REST API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.seedance_api_key
        self.base_url = settings.seedance_api_base_url.rstrip("/")
        self.default_model = settings.seedance_model
        self._poll_interval = 20  # seconds between status checks

    @property
    def provider_name(self) -> str:
        return "Seedance"

    # ------------------------------------------------------------------
    # VideoProvider interface
    # ------------------------------------------------------------------

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
        effective_model = model_id or self.default_model
        effective_duration = _snap_duration(duration_seconds)
        effective_resolution = _snap_resolution(resolution)

        if effective_duration != duration_seconds:
            logger.info(
                "Seedance: snapped duration %ds → %ds (valid: %s)",
                duration_seconds, effective_duration, sorted(_VALID_DURATIONS),
            )
        if effective_resolution != resolution:
            logger.info(
                "Seedance: snapped resolution %s → %s",
                resolution, effective_resolution,
            )

        # Encode first-frame image
        first_frame_b64 = _image_to_base64(storyboard_image_path)

        # Encode reference images (Seedance supports up to 9)
        ref_images_b64: list[str] = []
        if use_reference_images and asset_image_paths:
            for path in asset_image_paths[:9]:
                ref_images_b64.append(_image_to_base64(path))

        # Seedance generates 1 video per request — run concurrent tasks
        # for multiple variants (each with a different seed).
        base_seed = seed if seed is not None else random.randint(0, 2**31)
        tasks = [
            self._generate_single(
                prompt=prompt,
                first_frame_b64=first_frame_b64,
                ref_images_b64=ref_images_b64,
                model_id=effective_model,
                duration=effective_duration,
                resolution=effective_resolution,
                aspect_ratio=aspect_ratio,
                seed=base_seed + i,
                generate_audio=generate_audio,
                negative_prompt=negative_prompt,
            )
            for i in range(num_variants)
        ]
        video_urls = await asyncio.gather(*tasks)

        # Download all variants to output_dir
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        local_paths: list[str] = []
        for i, url in enumerate(video_urls):
            local_path = str(out / f"variant_{i}.mp4")
            await self._download_video(url, local_path)
            local_paths.append(local_path)
            logger.info("Seedance: downloaded variant %d → %s", i, local_path)

        return VideoGenerationOutput(local_paths=local_paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_single(
        self,
        prompt: str,
        first_frame_b64: str,
        ref_images_b64: list[str],
        model_id: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        seed: int,
        generate_audio: bool,
        negative_prompt: str,
    ) -> str:
        """Submit one generation task, poll until done, return video URL."""
        body: dict = {
            "model": model_id,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "seed": seed,
            "generate_audio": generate_audio,
            "first_frame_image": first_frame_b64,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if ref_images_b64:
            body["reference_images"] = ref_images_b64

        task_id = await self._submit_task(body)
        logger.info("Seedance task submitted: %s (model=%s)", task_id, model_id)
        return await self._poll_task(task_id)

    async def _submit_task(self, body: dict) -> str:
        """POST to the generation endpoint. Returns task ID."""
        url = f"{self.base_url}/v3/async/generate_video"
        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(3):
                resp = await client.post(
                    url,
                    json=body,
                    headers=self._headers(),
                )
                if resp.status_code == 429 and attempt < 2:
                    wait = 5 * (2 ** attempt) + random.uniform(0, 2)
                    logger.warning("Seedance rate-limited, retrying in %.1fs", wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                task_id = (
                    data.get("task_id")
                    or data.get("data", {}).get("task_id")
                    or data.get("request_id")
                )
                if not task_id:
                    raise ValueError(f"Seedance submit returned no task_id: {data}")
                return task_id
        raise RuntimeError("Seedance submit failed after retries")

    async def _poll_task(self, task_id: str) -> str:
        """Poll task status until done. Returns video download URL."""
        url = f"{self.base_url}/v3/async/query"
        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(180):  # ~60 minutes max
                await asyncio.sleep(self._poll_interval)
                resp = await client.get(
                    url,
                    params={"task_id": task_id},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()

                # Handle different response shapes from various providers
                inner = data.get("data", data)
                status = inner.get("status", "").lower()

                if status in ("done", "completed", "succeed", "success"):
                    video_url = self._extract_video_url(inner)
                    if video_url:
                        return video_url
                    raise ValueError(
                        f"Seedance task {task_id} completed but no video URL: {data}"
                    )
                if status in ("failed", "error", "cancelled"):
                    error = inner.get("error", inner.get("message", "unknown"))
                    raise RuntimeError(
                        f"Seedance task {task_id} failed: {error}"
                    )
                # Still processing — continue polling
                logger.debug("Seedance task %s status: %s", task_id, status)

        raise TimeoutError(f"Seedance task {task_id} timed out after polling")

    @staticmethod
    def _extract_video_url(data: dict) -> str | None:
        """Extract video URL from various response formats."""
        # Format: {"video_urls": ["https://..."]}
        urls = data.get("video_urls", [])
        if urls:
            return urls[0]
        # Format: {"video": {"url": "https://..."}}
        video = data.get("video", {})
        if isinstance(video, dict) and video.get("url"):
            return video["url"]
        # Format: {"output": {"video": "https://..."}}
        output = data.get("output", {})
        if isinstance(output, dict):
            return output.get("video") or output.get("video_url")
        return None

    async def _download_video(self, url: str, local_path: str) -> None:
        """Download video from HTTP URL to local file."""
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(resp.content)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
