"""
model_loader.py
===============
Utility module for downloading model weights from Hugging Face Hub.

Uses `hf_hub_download` for reliable, cached model retrieval.
Designed for Streamlit Cloud deployment where model files
are NOT stored in the Git repository.

Testing Instructions (as comments):
    1. Delete local HF cache: rm -rf ~/.cache/huggingface/hub/models--Anuraaag17--mushroom-classifier-models
    2. Run: streamlit run app/app.py
    3. Verify: spinner appears → model downloads → prediction works
    4. Re-run: no spinner → model loads from cache instantly
"""

import os
import logging

logger = logging.getLogger(__name__)


# ── Default Configuration ────────────────────────────────────────────────────
DEFAULT_REPO_ID = "Anuraaag17/mushroom-classifier-models"
DEFAULT_FILENAME = "efficientnet_b2_best.pth"


def download_model_if_needed(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    cache_dir: str = None,
) -> str:
    """Download model weights from Hugging Face Hub if not already cached.

    Uses `huggingface_hub.hf_hub_download` which handles:
        - Automatic caching (won't re-download if cached)
        - ETag-based cache validation
        - Resumable downloads
        - Proper error types for different failure modes

    Args:
        repo_id:   Hugging Face repository ID (e.g. "username/repo-name").
                   Must exactly match the repo on huggingface.co.
        filename:  Name of the model file in the repository.
                   Must exactly match the uploaded filename.
        cache_dir: Optional override for the HF cache directory.
                   If None, uses the default HF cache (~/.cache/huggingface).

    Returns:
        str: Absolute local path to the downloaded (or cached) model file.

    Raises:
        SystemExit: If the download fails irrecoverably (logged before exit).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. "
            "Run: pip install huggingface_hub"
        )
        raise SystemExit(1)

    logger.info(f"Checking model: repo={repo_id}, file={filename}")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            cache_dir=cache_dir,
        )
        logger.info(f"Model ready at: {local_path}")
        return local_path

    except Exception as e:
        _handle_download_error(e, repo_id, filename)
        raise SystemExit(1)


def _handle_download_error(error: Exception, repo_id: str, filename: str):
    """Log a user-friendly error message based on the exception type."""
    error_type = type(error).__name__

    if "RepositoryNotFound" in error_type:
        logger.error(
            f"Repository not found: '{repo_id}'. "
            f"Check that the repo_id exactly matches your Hugging Face repository. "
            f"Expected format: 'username/repo-name'."
        )
    elif "EntryNotFound" in error_type:
        logger.error(
            f"File not found: '{filename}' in repo '{repo_id}'. "
            f"Check that the filename exactly matches the uploaded file on Hugging Face."
        )
    elif "HfHubHTTPError" in error_type or "HTTPError" in error_type:
        logger.error(
            f"HTTP error downloading model from Hugging Face: {error}. "
            f"Check your internet connection and try again."
        )
    elif "ConnectionError" in error_type or "Timeout" in error_type:
        logger.error(
            f"Network error: Could not reach Hugging Face Hub. "
            f"Check your internet connection."
        )
    else:
        logger.error(
            f"Unexpected error downloading model: {error_type}: {error}"
        )
