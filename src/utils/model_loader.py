"""
model_loader.py
===============
Utility module for downloading model weights from Hugging Face Hub.

- Downloads multiple model files safely
- Uses caching (no re-download if already present)
- Handles errors cleanly
"""

import logging

logger = logging.getLogger(__name__)

# ── Default Configuration ────────────────────────────────────────────────────
DEFAULT_REPO_ID = "Anuraaag17/mushroom-classifier-models"
PRIMARY_FILENAME = "efficientnet_b2_best.pth"
SECONDARY_FILENAME = "mobilenetv2_best.pth"


def download_models(
    repo_id: str = DEFAULT_REPO_ID,
    primary_filename: str = PRIMARY_FILENAME,
    secondary_filename: str = SECONDARY_FILENAME,
    cache_dir: str = None,
):
    """
    Download both primary and secondary models from Hugging Face Hub.

    Returns:
        tuple: (primary_model_path, secondary_model_path)
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )
        raise SystemExit(1)

    try:
        logger.info(f"Downloading primary model: {primary_filename}")
        primary_path = hf_hub_download(
            repo_id=repo_id,
            filename=primary_filename,
            repo_type="model",
            cache_dir=cache_dir,
        )

        logger.info(f"Downloading secondary model: {secondary_filename}")
        secondary_path = hf_hub_download(
            repo_id=repo_id,
            filename=secondary_filename,
            repo_type="model",
            cache_dir=cache_dir,
        )

        logger.info(f"Primary model ready at: {primary_path}")
        logger.info(f"Secondary model ready at: {secondary_path}")

        return primary_path, secondary_path

    except Exception as e:
        _handle_download_error(e, repo_id, primary_filename, secondary_filename)
        raise SystemExit(1)


def _handle_download_error(
    error: Exception,
    repo_id: str,
    primary_filename: str,
    secondary_filename: str,
):
    """Log a clean, useful error message."""

    error_type = type(error).__name__

    if "RepositoryNotFound" in error_type:
        logger.error(
            f"Repository not found: '{repo_id}'. "
            f"Check repo name (format: username/repo-name)."
        )

    elif "EntryNotFound" in error_type:
        logger.error(
            f"File not found in repo '{repo_id}'. "
            f"Check filenames:\n"
            f"- {primary_filename}\n"
            f"- {secondary_filename}"
        )

    elif "HfHubHTTPError" in error_type or "HTTPError" in error_type:
        logger.error(
            f"HTTP error while downloading: {error}. "
            f"Check internet or Hugging Face status."
        )

    elif "ConnectionError" in error_type or "Timeout" in error_type:
        logger.error(
            "Network error: Unable to reach Hugging Face Hub."
        )

    else:
        logger.error(
            f"Unexpected error: {error_type}: {error}"
        )