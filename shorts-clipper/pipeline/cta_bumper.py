"""pipeline/cta_bumper.py

Stage 7 (optional): Generates a 3-second animated CTA bumper clip and appends
it to each final short clip.

Rendering: pure Pillow + ffmpeg (Option B — no headless browser required).

Public API
----------
    fetch_channel_info(video_url)  -> dict
    generate_cta_bumper(channel_name, subscriber_count, logo_path, accent_color) -> str
    append_cta_to_clip(clip_path, bumper_path, output_path) -> str
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Ensure ffmpeg is on PATH (mirrors what api/main.py does at startup)
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

from PIL import Image, ImageDraw, ImageFont

from utils.logger import logger


# ── Canvas constants ────────────────────────────────────────────────────────
W, H  = 1080, 1920
FPS   = 30
DURATION  = 3.0
N_FRAMES  = int(FPS * DURATION)  # 90

# Card geometry
CARD_W    = 860
CARD_H    = 165
CARD_X0   = (W - CARD_W) // 2        # 110
CARD_Y0   = (H - CARD_H) // 2        # 877
CARD_X1   = CARD_X0 + CARD_W         # 970
CARD_Y1   = CARD_Y0 + CARD_H         # 1042
CARD_CY   = (CARD_Y0 + CARD_Y1) // 2 # ~959
CARD_R    = 82                        # corner radius

# Logo circle — floats centred ABOVE the card.
# Bottom edge of circle overlaps card top by LOGO_OVERLAP px.
LOGO_R       = 70    # radius (smaller so it doesn't dominate)
LOGO_OVERLAP = 22    # px the circle dips below the card top edge
LOGO_RING_W  = 6     # white border ring width

# Subscribe button (right side of card)
BTN_W     = 210
BTN_H     = 70
BTN_X0    = CARD_X1 - BTN_W - 28
BTN_X1    = BTN_X0 + BTN_W

# Text area (left side of card) — full width now that logo is above the card
TEXT_X      = CARD_X0 + 32
TEXT_MAX_W  = BTN_X0 - TEXT_X - 20   # max px before hitting the button

# Animation timeline (seconds)
T_SLIDE_END  = 0.30  # card done sliding in
T_HOLD1_END  = 1.50  # first static hold ends
T_CLICK_END  = 1.80  # button click animation ends
T_HOLD2_END  = 2.50  # second hold ends → fade starts
T_FADE_END   = 3.00  # fully black

SLIDE_DIST   = 290   # px card slides up from
TICK_AMOUNT  = 50    # subscriber count added during tick animation


# ── Utilities ───────────────────────────────────────────────────────────────

def _ease_out_quad(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return 1.0 - (1.0 - p) ** 2


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _compact_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _parse_sub_count(s: str) -> tuple[str, int]:
    """'20K' → ('20K', 20000); '1.2M' → ('1.2M', 1200000)."""
    s = s.strip()
    u = s.upper()
    try:
        if u.endswith("M"):
            return s, int(float(u[:-1]) * 1_000_000)
        if u.endswith("K"):
            return s, int(float(u[:-1]) * 1_000)
        return s, int(float(s))
    except (ValueError, IndexError):
        return s, 0


def _format_tick(base: int, delta: int) -> str:
    """Format subscriber count during tick-up animation (e.g. 20050 → '20,050')."""
    val = base + delta
    if val >= 1_000_000:
        return f"{val:,}"
    return f"{val:,}"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    x0: int, y0: int, x1: int, y1: int,
    r: int,
    fill: tuple,
    border: tuple | None = None,
    border_w: int = 2,
) -> None:
    """Draw a rounded rectangle (pill shape) using Pillow primitives."""
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    draw.pieslice([x0, y0, x0 + 2 * r, y0 + 2 * r], 180, 270, fill=fill)
    draw.pieslice([x1 - 2 * r, y0, x1, y0 + 2 * r], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2 * r, x0 + 2 * r, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - 2 * r, y1 - 2 * r, x1, y1], 0, 90, fill=fill)
    if border:
        draw.arc([x0, y0, x0 + 2 * r, y0 + 2 * r], 180, 270, fill=border, width=border_w)
        draw.arc([x1 - 2 * r, y0, x1, y0 + 2 * r], 270, 360, fill=border, width=border_w)
        draw.arc([x0, y1 - 2 * r, x0 + 2 * r, y1], 90, 180, fill=border, width=border_w)
        draw.arc([x1 - 2 * r, y1 - 2 * r, x1, y1], 0, 90, fill=border, width=border_w)


def _make_background() -> Image.Image:
    """Precompute the dark gradient background (black + deep-red radial glow)."""
    if _HAS_NUMPY:
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        # Subtle vertical darkening (pure black → very dark)
        y_vals = np.arange(H).reshape(H, 1)
        base_r = (y_vals * 10 // H).astype(np.uint8)
        bg[:, :, 0] = np.broadcast_to(base_r, (H, W))
        # Deep-red radial glow centred in lower-left quadrant
        gy, gx = int(H * 0.62), int(W * 0.22)
        Ygrid, Xgrid = np.ogrid[:H, :W]
        dist = np.sqrt(
            (Xgrid.astype(float) - gx) ** 2 + (Ygrid.astype(float) - gy) ** 2
        )
        glow = np.clip(80.0 * np.exp(-dist / 480.0), 0, 80).astype(np.uint8)
        bg[:, :, 0] = np.clip(
            bg[:, :, 0].astype(np.int32) + glow, 0, 255
        ).astype(np.uint8)
        return Image.fromarray(bg, "RGB")
    else:
        import math as _math
        img = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(img)
        gy, gx = int(H * 0.62), int(W * 0.22)
        for y in range(0, H, 3):
            dist = _math.sqrt((gx - W // 2) ** 2 + (y - gy) ** 2)
            r = min(int(80 * _math.exp(-dist / 480)), 80)
            d.rectangle([0, y, W, y + 3], fill=(r, 0, 0))
        return img


def _make_logo_circle(logo_path: str, diameter: int) -> Image.Image:
    """Load an image, circular-crop it, and return an RGBA image."""
    src = Image.open(logo_path).convert("RGBA")
    src = src.resize((diameter, diameter), Image.LANCZOS)
    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, diameter, diameter], fill=255)
    result = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    result.paste(src, (0, 0), mask)
    return result


class _Fonts:
    """Pre-loaded font set — created once per bumper render, never per-frame."""
    __slots__ = ("name", "sub", "init", "btn_sub", "btn_done")

    def __init__(self) -> None:
        self.name     = _load_font(44, bold=True)
        self.sub      = _load_font(30, bold=False)
        self.init     = _load_font(36, bold=True)   # initials inside smaller logo circle
        self.btn_sub  = _load_font(26, bold=True)   # "subscribe"
        self.btn_done = _load_font(22, bold=True)   # "SUBSCRIBED"


def _render_frame(
    background: Image.Image,
    frame_idx: int,
    channel_name: str,
    sub_original: str,
    sub_base: int,
    logo_circle: Image.Image | None,
    logo_initials: str,
    accent_rgb: tuple[int, int, int],
    fonts: _Fonts,
) -> bytes:
    """Render one frame and return raw RGB24 bytes for ffmpeg."""
    t = frame_idx / FPS

    # ── Compute animation state ────────────────────────────────────────────
    card_alpha  = 255
    y_delta     = 0
    btn_scale   = 1.0
    btn_clicked = False
    sub_delta   = 0

    if t < T_SLIDE_END:
        p = _ease_out_quad(t / T_SLIDE_END)
        y_delta    = int(SLIDE_DIST * (1.0 - p))
        card_alpha = int(255 * p)
    elif t < T_HOLD1_END:
        pass
    elif t < T_CLICK_END:
        cp = (t - T_HOLD1_END) / (T_CLICK_END - T_HOLD1_END)
        if cp < 0.45:
            btn_scale = 1.0 - 0.08 * (cp / 0.45)
        elif cp < 0.65:
            btn_scale = 0.92
        else:
            btn_scale = 0.92 + 0.08 * ((cp - 0.65) / 0.35)
        btn_clicked = cp > 0.68
    elif t < T_HOLD2_END:
        btn_clicked = True
        hp = (t - T_CLICK_END) / (T_HOLD2_END - T_CLICK_END)
        sub_delta = int(TICK_AMOUNT * hp)
    else:
        btn_clicked = True
        fp = (t - T_HOLD2_END) / (T_FADE_END - T_HOLD2_END)
        card_alpha = int(255 * (1.0 - fp))

    # ── Compositing ────────────────────────────────────────────────────────
    # background is RGB; convert() always allocates a new image — no .copy() needed.
    frame = background.convert("RGBA")

    if card_alpha <= 0:
        data = frame.convert("RGB").tobytes()
        del frame
        return data

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    ccy = CARD_CY + y_delta
    cy0 = ccy - CARD_H // 2
    cy1 = ccy + CARD_H // 2

    # ── White card pill ───────────────────────────────────────────────────
    _draw_pill(draw, CARD_X0, cy0, CARD_X1, cy1, CARD_R, fill=(255, 255, 255, 238))

    # ── Channel name (truncated to fit before the subscribe button) ───────
    display_name = channel_name
    while display_name:
        nb = draw.textbbox((0, 0), display_name, font=fonts.name)
        if (nb[2] - nb[0]) <= TEXT_MAX_W:
            break
        display_name = display_name[:-1]
    if display_name != channel_name:
        display_name = display_name.rstrip() + "…"
    draw.text((TEXT_X, cy0 + 20), display_name, fill=(18, 18, 18, 255), font=fonts.name)

    # ── Subscriber count (only shown when a real count was provided) ──────
    if sub_base > 0:
        if sub_delta > 0:
            sub_label = _format_tick(sub_base, sub_delta) + " subscribers"
        else:
            sub_label = sub_original + " subscribers"
        draw.text((TEXT_X, ccy + 10), sub_label, fill=(105, 105, 105, 255), font=fonts.sub)

    # ── Logo circle — centred above the card, overlapping the top edge ────
    lcx = W // 2
    lcy = cy0 - LOGO_R + LOGO_OVERLAP   # bottom of circle is LOGO_OVERLAP px inside card top

    # White ring (drawn first, behind the logo fill)
    ring_r = LOGO_R + LOGO_RING_W
    draw.ellipse(
        [lcx - ring_r, lcy - ring_r, lcx + ring_r, lcy + ring_r],
        fill=(255, 255, 255, 255),
    )

    if logo_circle is not None:
        overlay.paste(logo_circle, (lcx - LOGO_R, lcy - LOGO_R), logo_circle)
    else:
        ar, ag, ab = accent_rgb
        draw.ellipse(
            [lcx - LOGO_R, lcy - LOGO_R, lcx + LOGO_R, lcy + LOGO_R],
            fill=(ar, ag, ab, 255),
        )
        ib = draw.textbbox((0, 0), logo_initials, font=fonts.init)
        iw, ih = ib[2] - ib[0], ib[3] - ib[1]
        draw.text(
            (lcx - iw // 2, lcy - ih // 2 - 3),
            logo_initials, fill=(255, 255, 255, 255), font=fonts.init,
        )

    # ── Subscribe button ──────────────────────────────────────────────────
    btn_cx = BTN_X0 + BTN_W // 2
    btn_cy = ccy
    sw = max(60, int(BTN_W * btn_scale))
    sh = max(30, int(BTN_H * btn_scale))
    bx0, by0 = btn_cx - sw // 2, btn_cy - sh // 2
    bx1, by1 = btn_cx + sw // 2, btn_cy + sh // 2
    br = max(4, sh // 2)

    if btn_clicked:
        _draw_pill(draw, bx0, by0, bx1, by1, br,
                   fill=(255, 255, 255, 255), border=(195, 195, 195, 255), border_w=2)
        btxt = "SUBSCRIBED"
        bb   = draw.textbbox((0, 0), btxt, font=fonts.btn_done)
        draw.text((btn_cx - (bb[2]-bb[0])//2, btn_cy - (bb[3]-bb[1])//2 - 2),
                  btxt, fill=(45, 45, 45, 255), font=fonts.btn_done)
    else:
        ar, ag, ab = accent_rgb
        _draw_pill(draw, bx0, by0, bx1, by1, br, fill=(ar, ag, ab, 255))
        btxt = "subscribe"
        bb   = draw.textbbox((0, 0), btxt, font=fonts.btn_sub)
        draw.text((btn_cx - (bb[2]-bb[0])//2, btn_cy - (bb[3]-bb[1])//2 - 2),
                  btxt, fill=(255, 255, 255, 255), font=fonts.btn_sub)

    # ── Apply card alpha ──────────────────────────────────────────────────
    if card_alpha < 255:
        r_ch, g_ch, b_ch, a_ch = overlay.split()
        a_ch = a_ch.point(lambda x: int(x * card_alpha / 255))
        overlay = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))

    result = Image.alpha_composite(frame, overlay)
    del frame, overlay

    if card_alpha < 255 and t >= T_HOLD2_END:
        black = Image.new("RGBA", (W, H), (0, 0, 0, 255 - card_alpha))
        merged = Image.alpha_composite(result, black)
        del result, black
        result = merged

    rgb = result.convert("RGB")
    del result
    data = rgb.tobytes()
    del rgb
    return data


# ── Click audio generation ───────────────────────────────────────────────────

def _generate_click_audio(output_path: str) -> bool:
    """
    Generate a 3-second AAC audio track with a soft UI click at t=1.5s.
    Returns True on success, False if ffmpeg fails (audio will be silent).
    """
    try:
        # 1. Render a 60ms sine click with fade in/out
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tf:
            click_tmp = tf.name

        r1 = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-t", "0.06",
                "-i", "sine=frequency=1000:sample_rate=44100",
                "-af", "afade=t=in:ss=0:d=0.008,afade=t=out:st=0.05:d=0.01,volume=0.35",
                "-c:a", "aac", click_tmp,
            ],
            capture_output=True,
        )
        if r1.returncode != 0:
            return False

        # 2. Build a 3s stereo track: silence + click delayed by 1500ms
        r2 = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-t", "3.0",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-i", click_tmp,
                "-filter_complex",
                "[1:a]adelay=1500|1500[delayed];[0:a][delayed]amix=inputs=2:duration=first[out]",
                "-map", "[out]",
                "-t", "3.0",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                output_path,
            ],
            capture_output=True,
        )
        Path(click_tmp).unlink(missing_ok=True)
        return r2.returncode == 0

    except Exception as exc:
        logger.warning(f"[CTA] Click audio generation failed (non-fatal): {exc}")
        return False


# ── Public: generate bumper ──────────────────────────────────────────────────

def generate_cta_bumper(
    channel_name: str,
    subscriber_count: str,
    logo_path: str | None = None,
    accent_color: str = "#CC0000",
) -> str:
    """
    Render a 3-second CTA bumper at 1080×1920 30fps.

    Parameters
    ----------
    channel_name      : displayed on the left of the pill card
    subscriber_count  : e.g. "20K", "1.2M"
    logo_path         : path to a logo image (any format); None → initials fallback
    accent_color      : hex color for the subscribe button and logo background

    Returns
    -------
    Path to the generated bumper.mp4 (in a system temp directory).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="cta_"))
    video_only = tmp_dir / "bumper_video.mp4"
    audio_path = tmp_dir / "bumper_audio.aac"
    output     = tmp_dir / "bumper.mp4"

    accent_rgb           = _hex_to_rgb(accent_color)
    sub_original, sub_base = _parse_sub_count(subscriber_count)

    # Initials fallback (up to 2 chars from channel name)
    words = channel_name.strip().split()
    if len(words) >= 2:
        logo_initials = (words[0][0] + words[1][0]).upper()
    else:
        logo_initials = channel_name[:2].upper() if channel_name else "CH"

    # Load + pre-crop logo
    logo_circle: Image.Image | None = None
    if logo_path and Path(logo_path).exists():
        try:
            logo_circle = _make_logo_circle(logo_path, LOGO_R * 2)
        except Exception as exc:
            logger.warning(f"[CTA] Failed to load logo '{logo_path}': {exc} — using initials.")

    # Pre-render background and fonts ONCE — never inside the frame loop
    background = _make_background()
    fonts      = _Fonts()

    logger.info(f"[CTA] Rendering {N_FRAMES} frames ({W}×{H} @ {FPS}fps)…")

    # ── Pipe raw RGB24 frames into ffmpeg ─────────────────────────────────
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}",
            "-pix_fmt", "rgb24",
            "-r", str(FPS),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            str(video_only),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,   # capture so we can report errors
    )

    import gc as _gc
    try:
        for i in range(N_FRAMES):
            raw_bytes = _render_frame(
                background, i,
                channel_name, sub_original, sub_base,
                logo_circle, logo_initials, accent_rgb,
                fonts,
            )
            ffmpeg_proc.stdin.write(raw_bytes)
            del raw_bytes
            # Release PIL image objects every 15 frames to prevent memory build-up
            if i % 15 == 14:
                _gc.collect()
        ffmpeg_proc.stdin.close()
    except Exception:
        ffmpeg_proc.kill()
        raise

    _, ffmpeg_err = ffmpeg_proc.communicate()
    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(
            f"[CTA] ffmpeg video encoding failed:\n{ffmpeg_err.decode(errors='replace')[:500]}"
        )

    logger.info("[CTA] Video encoded. Generating click audio…")

    # ── Audio: click at 1.5s ────────────────────────────────────────────
    has_audio = _generate_click_audio(str(audio_path))

    # ── Mux video + audio ───────────────────────────────────────────────
    if has_audio and audio_path.exists():
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_only),
            "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "copy",
            str(output),
        ]
    else:
        # Fallback: add silent audio track so concat is compatible
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_only),
            "-f", "lavfi", "-t", "3.0",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output),
        ]

    result = subprocess.run(mux_cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"[CTA] ffmpeg mux failed: {result.stderr.decode()[:300]}")

    logger.info(f"[CTA] Bumper ready: {output}")
    return str(output)


# ── Public: append bumper to clip ────────────────────────────────────────────

def append_cta_to_clip(clip_path: str, bumper_path: str, output_path: str) -> str:
    """
    Concatenate clip + bumper using the ffmpeg concat FILTER.

    The concat filter (not the concat demuxer) properly resets the video
    decoder at the file boundary.  The demuxer approach causes the last frame
    of the clip to freeze for the duration of the bumper because the decoder
    never gets a proper keyframe reset signal.

    Returns output_path on success.
    """
    # Normalize both inputs to the same resolution, SAR, fps and audio format
    # before concatenating so the filter has clean, compatible streams.
    filter_complex = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=fps=30[v0];"
        "[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=fps=30[v1];"
        "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0];"
        "[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[vout][aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-i", str(bumper_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"[CTA] ffmpeg concat filter failed:\n{result.stderr.decode(errors='replace')}"
        )
    return str(output_path)


# ── Public: fetch YouTube channel info ──────────────────────────────────────

def _extract_video_id(url: str) -> str | None:
    for pat in [
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def fetch_channel_info(video_url: str) -> dict:
    """
    Call the YouTube Data API v3 to retrieve channel name, subscriber count,
    and logo URL for the channel that owns *video_url*.

    Requires YOUTUBE_API_KEY to be set in the environment.

    Returns
    -------
    {
        "channel_name"      : str,
        "subscriber_count"  : str,   # e.g. "20K"
        "subscriber_count_raw": int,
        "logo_url"          : str,
        "channel_id"        : str,
    }
    """
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in the environment.")

    video_id = _extract_video_id(video_url)
    if not video_id:
        raise ValueError(f"Cannot extract video ID from URL: {video_url!r}")

    import httpx

    # 1. Resolve video → channel ID
    r1 = httpx.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "snippet", "id": video_id, "key": api_key},
        timeout=10.0,
    )
    r1.raise_for_status()
    items = r1.json().get("items", [])
    if not items:
        raise ValueError(f"YouTube video not found: {video_id}")

    channel_id = items[0]["snippet"]["channelId"]

    # 2. Fetch channel snippet + statistics
    r2 = httpx.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "snippet,statistics", "id": channel_id, "key": api_key},
        timeout=10.0,
    )
    r2.raise_for_status()
    ch_items = r2.json().get("items", [])
    if not ch_items:
        raise ValueError(f"YouTube channel not found: {channel_id}")

    ch       = ch_items[0]
    name     = ch["snippet"]["title"]
    raw_subs = int(ch["statistics"].get("subscriberCount", 0))
    logo_url = (
        ch["snippet"]
        .get("thumbnails", {})
        .get("medium", ch["snippet"].get("thumbnails", {}).get("default", {}))
        .get("url", "")
    )

    return {
        "channel_name":        name,
        "subscriber_count":    _compact_count(raw_subs),
        "subscriber_count_raw": raw_subs,
        "logo_url":            logo_url,
        "channel_id":          channel_id,
    }
