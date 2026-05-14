import logging

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["config"])

TRANSITION_TYPES = ["cut", "dissolve", "fade", "wipe", "zoom", "match_cut", "whip_pan"]
AD_TONES = ["energetic", "sophisticated", "playful", "authoritative", "warm"]


@router.post("/script")
async def get_script_config() -> dict:
    """Return script generation configuration defaults and ranges."""
    return {
        "scene_count": {
            "default": 3,
            "min": 2,
            "max": 6,
        },
        "ad_tones": AD_TONES,
        "transition_types": TRANSITION_TYPES,
    }


@router.post("/video")
async def get_video_config(settings: Settings = Depends(get_settings)) -> dict:
    """Return video generation configuration (all video models, defaults)."""
    all_models = [m.model_dump() for m in settings.veo_models]
    if settings.seedance_api_key:
        all_models.extend(m.model_dump() for m in settings.seedance_models)

    return {
        "video_models": all_models,
        "default_video_model": settings.default_video_model,
    }
