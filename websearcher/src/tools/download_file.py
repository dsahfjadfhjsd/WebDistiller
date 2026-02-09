import hashlib
import os
import re
from typing import Dict, Optional
from urllib.parse import urlparse, unquote

from ..utils.download_utils import download_to_path


_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_GENERIC_NAMES = {"download", "content", "file"}


def _sanitize_filename(name: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "download.bin"


def _infer_filename(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path or "")
    if name:
        name = unquote(name)
        safe = _sanitize_filename(name)
        lower = safe.lower()
        if lower and lower not in _GENERIC_NAMES and "." in lower:
            return safe
        ext = ".pdf" if ".pdf" in url.lower() else ".bin"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        base = safe if safe else "download"
        return f"{base}_{digest}{ext}"
    ext = ".pdf" if ".pdf" in url.lower() else ".bin"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"download_{digest}{ext}"


def _load_download_cache(cache_path: str) -> Dict[str, object]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _save_download_cache(cache_path: str, cache: Dict[str, object]) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def download_file_to_base(
    url: str,
    base_dir: str,
    file_name: Optional[str] = None,
    overwrite: bool = False,
    max_retries: int = 3,
    segment_size_mb: int = 5,
    large_file_threshold_mb: int = 20
) -> Dict[str, object]:
    if not url:
        return {"success": False, "error": "Missing URL"}

    cache_path = os.path.join(base_dir, "cache", "download_cache.json")
    cache = _load_download_cache(cache_path)
    if not overwrite:
        cached = cache.get(url)
        if isinstance(cached, dict):
            cached_name = cached.get("file_name")
            if cached_name:
                cached_path = os.path.join(base_dir, cached_name)
                if os.path.exists(cached_path):
                    size = os.path.getsize(cached_path)
                    return {
                        "success": True,
                        "file_name": cached_name.replace("\\", "/"),
                        "file_path": cached_path,
                        "size_bytes": size,
                        "final_url": cached.get("final_url", url),
                        "segmented": cached.get("segmented", False),
                        "skipped": True,
                        "cached": True
                    }

    safe_name = _sanitize_filename(file_name) if file_name else _infer_filename(url)
    download_dir = os.path.join(base_dir, "downloads")
    dest_path = os.path.join(download_dir, safe_name)

    result = download_to_path(
        url=url,
        dest_path=dest_path,
        overwrite=overwrite,
        max_retries=max_retries,
        segment_size_mb=segment_size_mb,
        large_file_threshold_mb=large_file_threshold_mb
    )

    if not result.get("success"):
        return result

    rel_name = os.path.relpath(dest_path, base_dir)
    cache[url] = {
        "file_name": rel_name.replace("\\", "/"),
        "file_path": dest_path,
        "size_bytes": result.get("size_bytes"),
        "final_url": result.get("final_url", url),
        "segmented": result.get("segmented", False)
    }
    _save_download_cache(cache_path, cache)

    return {
        "success": True,
        "file_name": rel_name.replace("\\", "/"),
        "file_path": dest_path,
        "size_bytes": result.get("size_bytes"),
        "final_url": result.get("final_url", url),
        "segmented": result.get("segmented", False),
        "skipped": result.get("skipped", False),
        "cached": False
    }
