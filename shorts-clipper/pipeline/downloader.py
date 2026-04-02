import os
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from api.models import VideoMetadata
from utils.logger import logger


class DownloaderError(RuntimeError):
    pass


class YDLLogger:
    def debug(self, msg: str) -> None:
        if msg.startswith("[debug] "):
            logger.debug(msg)
        else:
            self.info(msg)

    def info(self, msg: str) -> None:
        logger.info(msg)

    def warning(self, msg: str) -> None:
        logger.warning(msg)

    def error(self, msg: str) -> None:
        logger.error(msg)


def _extract_info(url: str, *, download: bool, outtmpl: str | None = None) -> dict[str, Any]:
    cookie_path = Path("cookies.txt").resolve()
    if cookie_path.exists():
        logger.info(f"🍪 Using cookies from: {cookie_path}")
        cookiefile = str(cookie_path)
    else:
        logger.warning("⚠️ No cookies.txt found in project root!")
        cookiefile = None

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": not download,
        "cookiefile": cookiefile,
        "logger": YDLLogger(),
        # Use Node.js (already installed) to solve YouTube's n-challenge.
        # yt-dlp Python API expects a dict: {runtime_name: {config}}.
        "js_runtimes": {"node": {}},
        # Download the official EJS challenge-solver script from GitHub.
        "remote_components": ["ejs:github"],
    }

    if download:
        ydl_opts["format"] = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]"
        ydl_opts["merge_output_format"] = "mp4"
        ydl_opts["nopart"] = True
        ydl_opts["hls_prefer_native"] = True
        if outtmpl is not None:
            ydl_opts["outtmpl"] = outtmpl

    try:
        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=download)
    except DownloadError as exc:
        logger.error(f"yt-dlp error for {url}: {exc}")
        raise DownloaderError(str(exc)) from exc


def get_metadata(url: str) -> VideoMetadata:
    logger.info(f"🔍 Extracting metadata for {url}...")
    try:
        info = _extract_info(url, download=False)
        automatic_captions = info.get("automatic_captions") or {}
        subtitles = info.get("subtitles") or {}

        metadata_dict = {
            "video_id": info.get("id"),
            "title": info.get("title") or "Untitled Video",
            "description": info.get("description") or "",
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
            "tags": info.get("tags") or [],
            "categories": info.get("categories") or [],
            "chapters": info.get("chapters") or [],
            "thumbnail": info.get("thumbnail"),
            "webpage_url": info.get("webpage_url") or url,
            "auto_caption_available": bool(automatic_captions or subtitles),
        }
    except Exception as e:
        logger.warning(f"⚠️ Metadata extraction failed: {e}. Switching to fallback...")
        # Extract basic ID from URL if possible
        video_id = url.split("v=")[1].split("&")[0] if "v=" in url else "unknown"
        metadata_dict = {
            "video_id": video_id,
            "title": f"Video {video_id}",
            "description": f"Metadata extraction failed: {str(e)[:100]}",
            "duration": 0,
            "uploader": "Unknown",
            "tags": [],
            "categories": [],
            "chapters": [],
            "thumbnail": None,
            "webpage_url": url,
            "auto_caption_available": False,
        }

    metadata = VideoMetadata(**metadata_dict)
    logger.info(f"✅ Metadata handled: {metadata.title}")
    return metadata


def download_video_via_proxy(url: str, output_dir: Path) -> str:
    """Fallback download method using a third-party proxy API."""
    import httpx
    import time

    proxy_url = "https://app.ytdown.to/proxy.php"
    headers = {
        "accept": "*/*",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    }
    cookies = {"PHPSESSID": "hqa5icv8587fcfqurqvh2o234h"}
    data = {"url": url}

    logger.info(f"🌐 Attempting proxy download via {proxy_url}...")
    try:
        with httpx.Client(timeout=60.0, verify=False) as client:
            resp = client.post(proxy_url, headers=headers, cookies=cookies, data=data)
            resp.raise_for_status()
            res_data = resp.json()

            media_items = res_data.get("api", {}).get("media", [])
            if not media_items:
                raise RuntimeError(f"Proxy returned no media items: {res_data}")

            # Prefer 720p
            target = next((m for m in media_items if m.get("mediaQuality") == "720p"), media_items[0])
            direct_url = target.get("mediaUrl")
            
            if not direct_url:
                raise RuntimeError("Proxy response missing direct link.")

            logger.info(f"🔗 Got direct link: {direct_url[:50]}...")

            output_path = output_dir / "proxy_download.mp4"
            with client.stream("GET", direct_url) as stream:
                stream.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in stream.iter_bytes():
                        f.write(chunk)

            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"✅ Proxy download successful: {output_path} ({output_path.stat().st_size} bytes)")
                return str(output_path.resolve())
            else:
                raise RuntimeError("Proxy download produced an empty file.")

    except Exception as e:
        logger.error(f"❌ Proxy download failed: {e}")
        raise DownloaderError(f"Proxy download failed: {e}")


def download_video(url: str, output_dir: Path) -> str:
    logger.info(f"📥 Starting video download: {url}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = None
    try:
        output_template = str(output_dir / "raw_download.%(ext)s")
        info = _extract_info(url, download=True, outtmpl=output_template)

        prepared_name = YoutubeDL(
            {"outtmpl": output_template, "merge_output_format": "mp4"}
        ).prepare_filename(info)
        
        potential_exts = [".mp4", ".mkv", ".webm", ".f137.mp4", ".f251.webm"]
        base_path = Path(prepared_name).with_suffix("")
        
        for ext in potential_exts:
            p = base_path.with_suffix(ext)
            if p.exists():
                final_path = p
                break
                
        if not final_path or not final_path.exists():
            files = list(output_dir.glob("raw_download.*"))
            if files:
                final_path = files[0]

        if not final_path or not final_path.exists():
            raise DownloaderError("Download produced no file.")

        if final_path.stat().st_size == 0:
            raise DownloaderError("The downloaded file is empty.")

    except Exception as exc:
        logger.warning(f"⚠️ Primary download (yt-dlp) failed: {exc}")
        if "bot" in str(exc).lower() or "empty" in str(exc).lower() or "no file" in str(exc).lower():
            logger.info("⚡ Switching to PROXY download fallback...")
            return download_video_via_proxy(url, output_dir)
        raise

    if final_path.suffix.lower() != ".mp4":
        logger.info(f"🔄 Converting {final_path.suffix} to mp4...")
        mp4_path = final_path.with_suffix(".mp4")
        cmd = f'ffmpeg -y -i "{final_path}" -c copy "{mp4_path}"'
        if os.system(cmd) == 0:
            final_path.unlink()
            final_path = mp4_path
            logger.info("✅ Conversion successful.")

    logger.info(f"✅ Video download complete: {final_path}")
    return str(final_path.resolve())
