import httpx
from yt_dlp import YoutubeDL
from pathlib import Path

URL = "https://www.youtube.com/watch?v=s2EYIDY8wSM"
API_KEY = "sd_7776ff2edb112b80786eb2ce7876170d"

def test_ydl():
    print("Testing yt-dlp with cookies...")
    output_dir = Path("./data/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "cookiefile": "cookies.txt",
        "format": "best[ext=mp4]",
        "outtmpl": str(output_dir / "debug_video.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "verbose": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            print(f"Starting download to {output_dir}...")
            info = ydl.extract_info(URL, download=True)
            print(f"YDL info success, file: {info.get('filepath') or 'unknown'}")
    except Exception as e:
        print(f"YDL Error: {e}")

def test_supadata():
    print("\nTesting Supadata API...")
    headers = {"x-api-key": API_KEY}
    params = {"url": URL}
    url = "https://api.supadata.ai/v1/youtube/transcript"
    try:
        with httpx.Client(verify=False) as client: # Test if cert issue
            resp = client.get(url, headers=headers, params=params)
            print(f"Supadata resp: {resp.status_code}")
            print(f"Supadata body: {resp.text[:200]}")
    except Exception as e:
        print(f"Supadata Error: {e}")

if __name__ == "__main__":
    test_ydl()
    test_supadata()
