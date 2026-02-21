import os
import re
import time
import tempfile
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote, quote, parse_qs, urlencode

import requests

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Connection": "keep-alive",
}

DEFAULT_SEGMENT_SIZE_MB = 5
DEFAULT_LARGE_FILE_THRESHOLD_MB = 20
DEFAULT_MAX_RETRIES = 3


try:
    from config.proxy_config import get_proxy_config, should_use_proxy
    PROXY_CONFIG_AVAILABLE = True
except ImportError:
    PROXY_CONFIG_AVAILABLE = False
    def get_proxy_config():
        return None
    def should_use_proxy(url: str) -> bool:
        return False


def _build_fallback_urls(url: str) -> List[str]:
    fallbacks: List[str] = []
    lower = url.lower()
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    query = parse_qs(parsed.query)
    if "archive.org" in lower:
        filename = os.path.basename(path)
        if filename.lower().endswith(".pdf"):
            decoded = unquote(filename).replace(" ", "_")
            file_path = quote(decoded)
            # Try Wikimedia Commons FilePath
            fallbacks.append(f"https://commons.wikimedia.org/wiki/Special:FilePath/{file_path}")
            # Try direct upload.wikimedia.org link (more reliable for PDFs)
            fallbacks.append(f"https://upload.wikimedia.org/wikipedia/commons/{file_path}")
            # Try alternative archive.org download format
            if "/download/" in path:
                # If it's already a download link, try the item page's direct download
                item_match = re.search(r"/details/([^/]+)", path)
                if item_match:
                    item_id = item_match.group(1)
                    fallbacks.append(f"https://archive.org/download/{item_id}/{filename}")
    bitstream_match = re.search(r"/bitstreams/([0-9a-fA-F-]{36})", url)
    if bitstream_match and "server/api/core/bitstreams" not in lower:
        uuid = bitstream_match.group(1)
        if host:
            fallbacks.append(f"https://{host}/server/api/core/bitstreams/{uuid}/content")

    # GitHub raw
    if host == "github.com" and "/blob/" in path:
        parts = path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, branch = parts[:4]
            rest = "/".join(parts[4:])
            fallbacks.append(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rest}")
    if host == "github.com" and "/raw/" in path:
        parts = path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "raw":
            owner, repo, _, branch = parts[:4]
            rest = "/".join(parts[4:])
            fallbacks.append(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rest}")

    # GitHub gists
    if host == "gist.github.com":
        parts = path.strip("/").split("/")
        if len(parts) >= 2:
            user, gist_id = parts[0], parts[1]
            file_param = query.get("file", [""])[0]
            if file_param:
                fallbacks.append(f"https://gist.githubusercontent.com/{user}/{gist_id}/raw/{file_param}")
            else:
                fallbacks.append(f"https://gist.githubusercontent.com/{user}/{gist_id}/raw")

    # GitLab raw
    if "gitlab" in host and "/-/blob/" in path:
        fallbacks.append(f"{parsed.scheme}://{host}{path.replace('/-/blob/', '/-/raw/', 1)}")
    elif "gitlab" in host and "/blob/" in path:
        fallbacks.append(f"{parsed.scheme}://{host}{path.replace('/blob/', '/raw/', 1)}")

    # Bitbucket raw
    if host == "bitbucket.org" and "/src/" in path:
        fallbacks.append(f"{parsed.scheme}://{host}{path.replace('/src/', '/raw/', 1)}")

    # Google Drive
    if host in {"drive.google.com", "docs.google.com"}:
        if path.startswith("/file/d/"):
            file_id = path.split("/")[3]
            fallbacks.append(f"https://drive.google.com/uc?export=download&id={file_id}")
        elif "id" in query:
            file_id = query.get("id", [""])[0]
            if file_id:
                fallbacks.append(f"https://drive.google.com/uc?export=download&id={file_id}")
        elif path.startswith("/uc"):
            q = {"export": "download"}
            if "id" in query:
                q["id"] = query.get("id", [""])[0]
            fallbacks.append(f"https://drive.google.com/uc?{urlencode(q)}")

    # Dropbox
    if host.endswith("dropbox.com"):
        q = {k: v for k, v in query.items()}
        q["dl"] = ["1"]
        fallbacks.append(f"{parsed.scheme}://{host}{path}?{urlencode(q, doseq=True)}")

    # OneDrive / SharePoint
    if host in {"onedrive.live.com", "1drv.ms"} or host.endswith("sharepoint.com"):
        q = {k: v for k, v in query.items()}
        q["download"] = ["1"]
        fallbacks.append(f"{parsed.scheme}://{host}{path}?{urlencode(q, doseq=True)}")

    # Box
    if host.endswith("box.com"):
        q = {k: v for k, v in query.items()}
        q["download"] = ["1"]
        fallbacks.append(f"{parsed.scheme}://{host}{path}?{urlencode(q, doseq=True)}")

    # Zenodo
    if host == "zenodo.org" and "/files/" in path and ("record" in path or "records" in path):
        if "download" not in query:
            fallbacks.append(f"{parsed.scheme}://{host}{path}?download=1")

    # Figshare
    if "figshare.com" in host:
        file_id = query.get("file", [""])[0] or query.get("file_id", [""])[0]
        if file_id:
            fallbacks.append(f"https://figshare.com/ndownloader/files/{file_id}")
        if "/ndownloader/files/" in path:
            fallbacks.append(f"{parsed.scheme}://{host}{path}")

    # OSF
    if host == "osf.io" and path.strip("/") and not path.endswith("/download"):
        fallbacks.append(f"{parsed.scheme}://{host}{path}/download")

    # arXiv
    if host == "arxiv.org" and path.startswith("/abs/"):
        arxiv_id = path.split("/abs/")[1]
        fallbacks.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

    return fallbacks


def _iter_candidate_urls(url: str) -> List[str]:
    seen = set()
    ordered = [url] + _build_fallback_urls(url)
    result = []
    for candidate in ordered:
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def configure_download_defaults(
    segment_size_mb: Optional[int] = None,
    large_file_threshold_mb: Optional[int] = None,
    max_retries: Optional[int] = None
) -> None:
    global DEFAULT_SEGMENT_SIZE_MB, DEFAULT_LARGE_FILE_THRESHOLD_MB, DEFAULT_MAX_RETRIES
    if isinstance(segment_size_mb, int) and segment_size_mb > 0:
        DEFAULT_SEGMENT_SIZE_MB = segment_size_mb
    if isinstance(large_file_threshold_mb, int) and large_file_threshold_mb > 0:
        DEFAULT_LARGE_FILE_THRESHOLD_MB = large_file_threshold_mb
    if isinstance(max_retries, int) and max_retries > 0:
        DEFAULT_MAX_RETRIES = max_retries


def _resolve_download_defaults(
    segment_size_mb: Optional[int],
    large_file_threshold_mb: Optional[int],
    max_retries: Optional[int]
) -> Tuple[int, int, int]:
    seg_mb = segment_size_mb if isinstance(segment_size_mb, int) and segment_size_mb > 0 else DEFAULT_SEGMENT_SIZE_MB
    large_mb = large_file_threshold_mb if isinstance(large_file_threshold_mb, int) and large_file_threshold_mb > 0 else DEFAULT_LARGE_FILE_THRESHOLD_MB
    retries = max_retries if isinstance(max_retries, int) and max_retries > 0 else DEFAULT_MAX_RETRIES
    return seg_mb, large_mb, retries


def _download_full(
    session: requests.Session,
    url: str,
    dest_path: str,
    timeout: int,
    proxies: Optional[dict]
) -> Tuple[bool, Optional[int], str]:
    try:
        with session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=timeout, proxies=proxies) as response:
            if response.status_code >= 400:
                return False, None, f"HTTP {response.status_code}"
            size = response.headers.get("Content-Length")
            total = int(size) if size and size.isdigit() else None
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            return True, total, ""
    except Exception as e:
        return False, None, str(e)


def _download_full_with_retries(
    session: requests.Session,
    url: str,
    dest_path: str,
    timeout: int,
    proxies: Optional[dict],
    max_retries: int
) -> Tuple[bool, Optional[int], str]:
    last_error = ""
    for attempt in range(max_retries):
        ok, size, err = _download_full(session, url, dest_path, timeout, proxies)
        if ok:
            return True, size, ""
        last_error = err or last_error
        time.sleep(0.5 * (attempt + 1))
    return False, None, last_error or "Download failed"


def _download_segmented(
    session: requests.Session,
    url: str,
    dest_path: str,
    total_size: int,
    segment_size: int,
    timeout: int,
    proxies: Optional[dict],
    max_retries: int
) -> Tuple[bool, str]:
    try:
        with open(dest_path, "wb") as f:
            downloaded = 0
            while downloaded < total_size:
                start = downloaded
                end = min(total_size - 1, start + segment_size - 1)
                headers = {**DEFAULT_HEADERS, "Range": f"bytes={start}-{end}"}
                success = False
                last_error = ""
                for attempt in range(max_retries):
                    try:
                        with session.get(url, headers=headers, stream=True, timeout=timeout, proxies=proxies) as response:
                            if response.status_code == 200 and start == 0:
                                # Server ignored range; treat this as full download
                                for chunk in response.iter_content(chunk_size=1024 * 64):
                                    if chunk:
                                        f.write(chunk)
                                return True, ""
                            if response.status_code != 206:
                                last_error = f"HTTP {response.status_code}"
                                time.sleep(0.5)
                                continue
                            for chunk in response.iter_content(chunk_size=1024 * 64):
                                if chunk:
                                    f.write(chunk)
                            success = True
                            break
                    except Exception as e:
                        last_error = str(e)
                        time.sleep(0.5)
                if not success:
                    return False, last_error or "Segment download failed"
                downloaded = end + 1
        return True, ""
    except Exception as e:
        return False, str(e)


def download_to_path(
    url: str,
    dest_path: str,
    overwrite: bool = False,
    timeout: int = 30,
    max_retries: Optional[int] = None,
    segment_size_mb: Optional[int] = None,
    large_file_threshold_mb: Optional[int] = None
) -> Dict[str, object]:
    if os.path.exists(dest_path) and not overwrite:
        size = os.path.getsize(dest_path)
        return {
            "success": True,
            "final_url": url,
            "size_bytes": size,
            "skipped": True,
            "segmented": False
        }

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    segment_size_mb, large_file_threshold_mb, max_retries = _resolve_download_defaults(
        segment_size_mb, large_file_threshold_mb, max_retries
    )
    segment_size = segment_size_mb * 1024 * 1024
    large_file_threshold = large_file_threshold_mb * 1024 * 1024

    session = requests.Session()
    last_error = ""
    attempted_urls = []
    for candidate in _iter_candidate_urls(url):
        attempted_urls.append(candidate)
        proxies = None
        if should_use_proxy(candidate):
            proxies = get_proxy_config()

        for proxy_attempt in range(2):
            try:
                head = session.head(candidate, headers=DEFAULT_HEADERS, allow_redirects=True, timeout=timeout, proxies=proxies)
                if head.status_code >= 400:
                    ok, reported_size, err = _download_full_with_retries(
                        session, candidate, dest_path, timeout, proxies, max_retries
                    )
                    if ok:
                        actual = os.path.getsize(dest_path)
                        return {
                            "success": True,
                            "final_url": candidate,
                            "size_bytes": actual,
                            "skipped": False,
                            "segmented": False,
                            "reported_size": reported_size
                        }
                    last_error = err or f"HTTP {head.status_code}"
                    if proxies:
                        proxies = None
                        continue
                    break

                length_header = head.headers.get("Content-Length", "")
                total_size = int(length_header) if length_header.isdigit() else None
                accept_ranges = "bytes" in head.headers.get("Accept-Ranges", "").lower()

                segmented = bool(total_size and total_size >= large_file_threshold and accept_ranges)
                if segmented and total_size:
                    ok, err = _download_segmented(
                        session, candidate, dest_path, total_size,
                        segment_size, timeout, proxies, max_retries
                    )
                    if ok:
                        actual = os.path.getsize(dest_path)
                        if actual < total_size:
                            last_error = "Downloaded size smaller than expected"
                            break
                        return {
                            "success": True,
                            "final_url": candidate,
                            "size_bytes": actual,
                            "skipped": False,
                            "segmented": True
                        }
                    last_error = err or "Segmented download failed"
                else:
                    ok, reported_size, err = _download_full_with_retries(
                        session, candidate, dest_path, timeout, proxies, max_retries
                    )
                    if ok:
                        actual = os.path.getsize(dest_path)
                        return {
                            "success": True,
                            "final_url": candidate,
                            "size_bytes": actual,
                            "skipped": False,
                            "segmented": False,
                            "reported_size": reported_size
                        }
                    last_error = err or "Download failed"

            except Exception as e:
                last_error = str(e)
                ok, reported_size, err = _download_full_with_retries(
                    session, candidate, dest_path, timeout, proxies, max_retries
                )
                if ok:
                    actual = os.path.getsize(dest_path)
                    return {
                        "success": True,
                        "final_url": candidate,
                        "size_bytes": actual,
                        "skipped": False,
                        "segmented": False,
                        "reported_size": reported_size
                    }
                last_error = err or last_error

            if proxies:
                proxies = None
                continue
            break

        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass

    # Provide detailed error message with all attempted URLs
    error_msg = last_error or "Download failed"
    if len(attempted_urls) > 1:
        error_msg = f"{error_msg}\n\nAttempted {len(attempted_urls)} URLs (including fallbacks):\n" + "\n".join(f"  {i+1}. {url}" for i, url in enumerate(attempted_urls))
    elif attempted_urls:
        error_msg = f"{error_msg}\n\nAttempted URL: {attempted_urls[0]}"
    
    return {
        "success": False,
        "error": error_msg,
        "attempted_urls": attempted_urls,
        "num_attempts": len(attempted_urls)
    }


def download_to_tempfile(
    url: str,
    suffix: str = "",
    timeout: int = 30,
    max_retries: Optional[int] = None,
    segment_size_mb: Optional[int] = None,
    large_file_threshold_mb: Optional[int] = None
) -> Tuple[Optional[str], Dict[str, object]]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    result = download_to_path(
        url=url,
        dest_path=tmp_path,
        overwrite=True,
        timeout=timeout,
        max_retries=max_retries,
        segment_size_mb=segment_size_mb,
        large_file_threshold_mb=large_file_threshold_mb
    )
    if not result.get("success"):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None, result
    return tmp_path, result
