import asyncio
import logging
import random
from typing import Callable

from app.ai.prompts import VIDEO_PROMPT_TEMPLATE_IMAGE, VIDEO_PROMPT_TEMPLATE_REFERENCE
from app.ai.video_provider import VideoProvider
from app.config import Settings
from app.models.script import AvatarProfile, Scene
from app.models.storyboard import StoryboardResult
from app.models.video import VideoQCReport, VideoResponse, VideoResult, VideoVariant
from app.services.qc_service import QCService
from app.storage.gcs import GCSStorage
from app.storage.local import LocalStorage
from app.utils.ffmpeg import extract_last_frame

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(
        self,
        video_provider: VideoProvider,
        gcs: GCSStorage,
        qc: QCService,
        storage: LocalStorage,
        settings: Settings,
    ):
        self.video_provider = video_provider
        self.gcs = gcs
        self.qc = qc
        self.storage = storage
        self.settings = settings

    async def generate_videos(
        self,
        run_id: str,
        scenes_data: list[StoryboardResult],
        script_scenes: list[Scene],
        avatar_profile: AvatarProfile,
        on_progress: Callable | None = None,
        num_variants: int | None = None,
        seed: int | None = None,
        resolution: str = "720p",
        video_model: str | None = None,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 8,
        compression_quality: str = "optimized",
        qc_threshold: int | None = None,
        max_qc_regen_attempts: int = 2,
        use_reference_images: bool = True,
        negative_prompt_extra: str = "",
        generate_audio: bool = True,
    ) -> VideoResponse:
        """Generate video variants for all scenes with QC and auto-selection.

        Uses bounded concurrency via asyncio.Semaphore.
        """
        # Validate avatar description consistency across scenes
        descriptions = {
            s.detailed_avatar_description
            for s in script_scenes
            if s.detailed_avatar_description
        }
        if len(descriptions) > 1:
            canonical = next(iter(descriptions))
            logger.warning(
                "Inconsistent avatar descriptions across %d scenes — normalizing to first",
                len(descriptions),
            )
            for s in script_scenes:
                if s.detailed_avatar_description:
                    s.detailed_avatar_description = canonical

        # Auto-generate seed for cross-scene consistency if none provided
        if seed is None:
            seed = random.randint(0, 2**31)
            logger.info("Auto-generated video seed: %d", seed)

        effective_variants = num_variants or self.settings.max_video_variants

        # Build a lookup from scene_number -> Scene
        scene_lookup = {s.scene_number: s for s in script_scenes}

        # Process scenes sequentially so we can extract the last frame of
        # each scene and pass it as an asset reference to the next scene,
        # maintaining visual continuity across the ad.
        sorted_scenes = sorted(scenes_data, key=lambda s: s.scene_number)
        results: list[VideoResult] = []
        prev_last_frame_path: str | None = None

        for sb_result in sorted_scenes:
            scene = scene_lookup[sb_result.scene_number]
            result = await self._process_single_scene(
                run_id=run_id,
                sb_result=sb_result,
                scene=scene,
                avatar_profile=avatar_profile,
                on_progress=on_progress,
                num_variants=effective_variants,
                seed=seed,
                resolution=resolution,
                video_model=video_model,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                compression_quality=compression_quality,
                qc_threshold=qc_threshold,
                max_qc_regen_attempts=max_qc_regen_attempts,
                use_reference_images=use_reference_images,
                negative_prompt_extra=negative_prompt_extra,
                prev_scene_last_frame_path=prev_last_frame_path,
                generate_audio=generate_audio,
            )
            results.append(result)

            # Extract last frame from the selected video for next scene
            try:
                selected_video_local = str(self.storage.get_path(
                    run_id,
                    f"variant_{result.selected_index}.mp4",
                    subdir=f"scenes/scene_{sb_result.scene_number}/video_variants",
                ))
                last_frame_local = str(self.storage.get_path(
                    run_id,
                    "last_frame.png",
                    subdir=f"scenes/scene_{sb_result.scene_number}",
                ))
                await extract_last_frame(selected_video_local, last_frame_local)
                prev_last_frame_path = last_frame_local
                logger.info(
                    "Scene %d: extracted last frame → %s",
                    sb_result.scene_number, last_frame_local,
                )
            except Exception as exc:
                logger.warning(
                    "Scene %d: failed to extract last frame: %s",
                    sb_result.scene_number, exc,
                )
                prev_last_frame_path = None

        return VideoResponse(results=results)

    async def _process_single_scene(
        self,
        run_id: str,
        sb_result: StoryboardResult,
        scene: Scene,
        avatar_profile: AvatarProfile,
        on_progress: Callable | None,
        num_variants: int | None = None,
        seed: int | None = None,
        resolution: str = "720p",
        video_model: str | None = None,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 8,
        compression_quality: str = "optimized",
        qc_threshold: int | None = None,
        max_qc_regen_attempts: int = 2,
        use_reference_images: bool = True,
        negative_prompt_extra: str = "",
        prev_scene_last_frame_path: str | None = None,
        generate_audio: bool = True,
        previous_qc_report: "VideoQCReport | None" = None,
    ) -> VideoResult:
        """Process a single scene: generate videos via provider, QC, select best."""
        effective_variants = num_variants or self.settings.max_video_variants
        scene_num = sb_result.scene_number

        # 1. Gather local paths for storyboard, product, and avatar images
        storyboard_local = str(self.storage.get_path(
            run_id, "storyboard.png", subdir=f"scenes/scene_{scene_num}",
        ))

        product_filename = self._find_product_image(run_id)
        product_local = str(self.storage.get_path(run_id, product_filename))
        avatar_local = str(self.storage.get_path(run_id, "avatar_selected.png"))

        # Build asset reference image list for character/product consistency
        asset_image_paths: list[str] | None = None
        if use_reference_images:
            asset_image_paths = [avatar_local, product_local]
            if prev_scene_last_frame_path:
                asset_image_paths.append(prev_scene_last_frame_path)
                logger.info(
                    "Scene %d: using previous scene last frame for continuity",
                    scene_num,
                )

        # 2. Build video prompt — use mode-aware template
        voice_style = (
            scene.voice_style
            or avatar_profile.voice_style
            or avatar_profile.tone_of_voice
        )
        detailed_desc = (
            scene.detailed_avatar_description
            or avatar_profile.visual_description
        )
        template = VIDEO_PROMPT_TEMPLATE_REFERENCE if use_reference_images else VIDEO_PROMPT_TEMPLATE_IMAGE
        prompt = template.format(
            detailed_avatar_description=detailed_desc,
            visual_background=scene.visual_background,
            lighting=scene.lighting,
            shot_type=scene.shot_type,
            avatar_action=scene.avatar_action,
            avatar_emotion=scene.avatar_emotion,
            camera_movement=scene.camera_movement,
            product_visual_integration=scene.product_visual_integration,
            voice_style=voice_style,
            script_dialogue=scene.script_dialogue,
            sound_design=scene.sound_design,
            audio_continuity=scene.audio_continuity or "",
        )

        # 2b. Pre-rewrite prompt using previous QC feedback (manual regen)
        qc_rewrite_context: str | None = None
        if previous_qc_report:
            feedback = self.qc.build_video_qc_feedback(previous_qc_report)
            qc_rewrite_context = f"Pre-generation rewrite from previous QC:\n{feedback}"
            prompt = await self.qc.rewrite_video_prompt(prompt, previous_qc_report)
            logger.info("Scene %d: prompt pre-rewritten using previous QC feedback", scene_num)

        # Build per-scene negative prompt from scene + request level
        scene_negative = scene.negative_elements or ""
        if negative_prompt_extra:
            scene_negative = f"{scene_negative}, {negative_prompt_extra}" if scene_negative else negative_prompt_extra

        # 3. Generate video variants via provider
        output_dir = str(self.storage.get_path(
            run_id, "", subdir=f"scenes/scene_{scene_num}/video_variants",
        ))

        gen_output = await self.video_provider.generate_videos(
            prompt=prompt,
            storyboard_image_path=storyboard_local,
            asset_image_paths=asset_image_paths,
            output_dir=output_dir,
            num_variants=effective_variants,
            seed=seed,
            resolution=resolution,
            negative_prompt=scene_negative,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            compression_quality=compression_quality,
            model_id=video_model,
            generate_audio=generate_audio,
            use_reference_images=use_reference_images,
        )

        variants: list[VideoVariant] = [
            VideoVariant(index=i, video_path=self.storage.to_url_path(lp))
            for i, lp in enumerate(gen_output.local_paths)
        ]

        # 4. Prepare GCS URIs for QC
        # If the provider returned remote URIs (e.g. Veo/GCS), use those.
        # Otherwise upload local videos to GCS for Gemini-based QC.
        video_qc_uris = gen_output.remote_uris
        if not video_qc_uris:
            video_qc_uris = await self._upload_videos_for_qc(
                gen_output.local_paths, run_id, scene_num,
            )

        # Upload product image to GCS for QC reference
        product_gcs_uri = await asyncio.to_thread(
            self.gcs.upload_file,
            product_local,
            f"pipeline/{run_id}/product_image.png",
        )

        # 5. Run QC on all variants in parallel
        qc_tasks = [
            self.qc.qc_video(video_uri=video_qc_uris[i], reference_uri=product_gcs_uri)
            for i in range(len(variants))
        ]
        qc_results = await asyncio.gather(*qc_tasks, return_exceptions=True)
        for i, result in enumerate(qc_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Video QC failed for scene %d variant %d: %s",
                    scene_num, i, result,
                )
            else:
                variants[i].qc_report = result

        # 6. Auto-select best variant
        selected_idx = self.qc.select_best_video_variant(variants)

        # 7. QC feedback loop: if best variant fails QC, rewrite prompt and regenerate
        regen_attempts = 0
        if max_qc_regen_attempts > 0:
            selected_variant = next((v for v in variants if v.index == selected_idx), variants[0])
            for regen_round in range(max_qc_regen_attempts):
                if selected_variant.qc_report and self.qc.video_passes_qc(
                    selected_variant.qc_report, threshold=qc_threshold
                ):
                    break
                regen_attempts += 1
                logger.info(
                    "Scene %d: video QC regen attempt %d/%d",
                    scene_num, regen_attempts, max_qc_regen_attempts,
                )
                # Rewrite prompt based on QC feedback
                if selected_variant.qc_report:
                    feedback = self.qc.build_video_qc_feedback(selected_variant.qc_report)
                    regen_ctx = f"QC regen attempt {regen_attempts}:\n{feedback}"
                    qc_rewrite_context = f"{qc_rewrite_context}\n\n{regen_ctx}" if qc_rewrite_context else regen_ctx
                    prompt = await self.qc.rewrite_video_prompt(prompt, selected_variant.qc_report)

                # Regenerate all variants with improved prompt
                gen_output = await self.video_provider.generate_videos(
                    prompt=prompt,
                    storyboard_image_path=storyboard_local,
                    asset_image_paths=asset_image_paths,
                    output_dir=output_dir,
                    num_variants=effective_variants,
                    seed=seed,
                    resolution=resolution,
                    negative_prompt=scene_negative,
                    aspect_ratio=aspect_ratio,
                    duration_seconds=duration_seconds,
                    compression_quality=compression_quality,
                    model_id=video_model,
                    generate_audio=generate_audio,
                    use_reference_images=use_reference_images,
                )

                variants = [
                    VideoVariant(index=i, video_path=self.storage.to_url_path(lp))
                    for i, lp in enumerate(gen_output.local_paths)
                ]

                # Prepare QC URIs for regenerated videos
                video_qc_uris = gen_output.remote_uris
                if not video_qc_uris:
                    video_qc_uris = await self._upload_videos_for_qc(
                        gen_output.local_paths, run_id, scene_num, suffix=f"_regen{regen_round + 1}",
                    )

                # Re-run QC in parallel
                regen_qc_tasks = [
                    self.qc.qc_video(video_uri=video_qc_uris[i], reference_uri=product_gcs_uri)
                    for i in range(len(variants))
                ]
                regen_qc_results = await asyncio.gather(*regen_qc_tasks, return_exceptions=True)
                for i, result in enumerate(regen_qc_results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Video QC failed for scene %d variant %d (regen %d): %s",
                            scene_num, i, regen_round + 1, result,
                        )
                    else:
                        variants[i].qc_report = result

                # Re-select best
                selected_idx = self.qc.select_best_video_variant(variants)
                selected_variant = next((v for v in variants if v.index == selected_idx), variants[0])

        # 8. Copy best to selected_video.mp4
        source_local = str(self.storage.get_path(
            run_id, f"variant_{selected_idx}.mp4",
            subdir=f"scenes/scene_{scene_num}/video_variants",
        ))
        selected_path = self.storage.save_file(
            run_id=run_id,
            filename="selected_video.mp4",
            source_path=source_local,
            subdir=f"scenes/scene_{scene_num}",
        )

        logger.info(
            "Scene %d: selected video variant %d of %d (%s)",
            scene_num, selected_idx, len(variants), self.video_provider.provider_name,
        )

        result = VideoResult(
            scene_number=scene_num,
            variants=variants,
            selected_index=selected_idx,
            selected_video_path=self.storage.to_url_path(selected_path),
            regen_attempts=regen_attempts,
            prompt_used=prompt,
            qc_rewrite_context=qc_rewrite_context,
        )

        if on_progress:
            on_progress({
                "scene_number": scene_num,
                "event": "video_completed",
                "result": result.model_dump(),
            })

        return result

    async def regenerate_single_scene(
        self,
        run_id: str,
        sb_result: StoryboardResult,
        scene: Scene,
        avatar_profile: AvatarProfile,
        on_progress: Callable | None = None,
        num_variants: int | None = None,
        seed: int | None = None,
        resolution: str = "720p",
        video_model: str | None = None,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 8,
        compression_quality: str = "optimized",
        qc_threshold: int | None = None,
        max_qc_regen_attempts: int = 2,
        use_reference_images: bool = True,
        negative_prompt_extra: str = "",
        generate_audio: bool = True,
        previous_qc_report: VideoQCReport | None = None,
    ) -> VideoResult:
        """Regenerate video for a single scene."""
        return await self._process_single_scene(
            run_id=run_id,
            sb_result=sb_result,
            scene=scene,
            avatar_profile=avatar_profile,
            on_progress=on_progress,
            num_variants=num_variants,
            seed=seed,
            resolution=resolution,
            video_model=video_model,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            compression_quality=compression_quality,
            qc_threshold=qc_threshold,
            max_qc_regen_attempts=max_qc_regen_attempts,
            use_reference_images=use_reference_images,
            negative_prompt_extra=negative_prompt_extra,
            generate_audio=generate_audio,
            previous_qc_report=previous_qc_report,
        )

    async def select_variant(
        self, run_id: str, scene_number: int, variant_index: int
    ) -> str:
        """Select a different video variant for a scene.

        Copies the chosen variant to selected_video.mp4 and returns its URL path.
        """
        source_local = str(self.storage.get_path(
            run_id,
            f"variant_{variant_index}.mp4",
            subdir=f"scenes/scene_{scene_number}/video_variants",
        ))
        if not self.storage.get_path(
            run_id, f"variant_{variant_index}.mp4",
            subdir=f"scenes/scene_{scene_number}/video_variants",
        ).exists():
            raise FileNotFoundError(
                f"Variant {variant_index} not found for scene {scene_number}"
            )

        selected_path = self.storage.save_file(
            run_id=run_id,
            filename="selected_video.mp4",
            source_path=source_local,
            subdir=f"scenes/scene_{scene_number}",
        )
        logger.info(
            "Scene %d: user selected variant %d", scene_number, variant_index
        )
        return self.storage.to_url_path(selected_path)

    def _find_product_image(self, run_id: str) -> str:
        """Find the product image file in the run directory."""
        for ext in ("png", "jpg", "webp"):
            path = self.storage.get_path(run_id, f"product_image.{ext}")
            if path.exists():
                return f"product_image.{ext}"
        raise FileNotFoundError(f"No product image found for run {run_id}")

    async def _upload_videos_for_qc(
        self,
        local_paths: list[str],
        run_id: str,
        scene_num: int,
        suffix: str = "",
    ) -> list[str]:
        """Upload local video files to GCS for Gemini-based QC."""
        uris = await asyncio.gather(*(
            asyncio.to_thread(
                self.gcs.upload_file,
                lp,
                f"pipeline/{run_id}/scenes/scene_{scene_num}{suffix}/qc_variant_{i}.mp4",
            )
            for i, lp in enumerate(local_paths)
        ))
        return list(uris)
