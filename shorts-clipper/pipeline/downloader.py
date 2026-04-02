from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


class DownloaderError(RuntimeError):
    pass


def _extract_info(url: str, *, download: bool, outtmpl: str | None = None) -> dict[str, Any]:
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": not download,
    }

    if download:
        ydl_opts["format"] = (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=720][ext=mp4]"
        )
        if outtmpl is not None:
            ydl_opts["outtmpl"] = outtmpl
            ydl_opts["merge_output_format"] = "mp4"

    try:
        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=download)
    except DownloadError as exc:
        raise DownloaderError(str(exc)) from exc


def get_metadata(url: str) -> dict[str, Any]:
    info = _extract_info(url, download=False)
    automatic_captions = info.get("automatic_captions") or {}
    subtitles = info.get("subtitles") or {}

    return {
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


def download_video(url: str, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s [%(id)s].%(ext)s")
    info = _extract_info(url, download=True, outtmpl=output_template)

    requested_downloads = info.get("requested_downloads") or []
    if requested_downloads:
        candidate = requested_downloads[0].get("filepath")
        if candidate:
            return str(Path(candidate).resolve())

    prepared_name = YoutubeDL({}).prepare_filename(info)
    return str((output_dir / Path(prepared_name).name).resolve())
