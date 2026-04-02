import asyncio
import time
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.downloader import get_metadata, download_video
from agents.classifier_agent import classify_content
from pipeline.transcript import fetch_transcript
from config import settings
from utils.logger import logger

TEST_URL = "https://www.youtube.com/watch?v=8Y9UTAA4ZTo"

def test_stage_1():
    logger.info("--- [TEST] Stage 1: Metadata & Classification ---")
    start_time = time.perf_counter()
    
    # 1. Get Metadata
    try:
        metadata = get_metadata(TEST_URL)
        logger.info(f"Metadata fetch: SUCCESS (Title: {metadata.title})")
    except Exception as e:
        logger.error(f"Metadata fetch: FAILED ({e})")
        return None

    # 2. Classify Content
    try:
        classification = classify_content(metadata)
        logger.info(f"Classification: SUCCESS (Type: {classification.content_type}, Source: {classification.source})")
    except Exception as e:
        logger.error(f"Classification: FAILED ({e})")
        return None

    duration = time.perf_counter() - start_time
    logger.info(f"Stage 1 Total Time: {duration:.2f} seconds")
    return metadata

def test_stage_2(metadata):
    logger.info("--- [TEST] Stage 2: Parallel Download & Transcript ---")
    start_time = time.perf_counter()
    
    output_dir = Path("./data/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        logger.info("Submitting parallel download and transcript tasks...")
        download_future = executor.submit(download_video, TEST_URL, output_dir)
        transcript_future = executor.submit(fetch_transcript, TEST_URL)
        
        # Wait for both
        transcript_result = transcript_future.result()
        logger.info(f"Transcript fetch: SUCCESS (Source: {transcript_result.source}, Segments: {len(transcript_result.segments)})")
        
        video_path = download_future.result()
        logger.info(f"Video download: SUCCESS (Path: {video_path})")

    duration = time.perf_counter() - start_time
    logger.info(f"Stage 2 Total Time: {duration:.2f} seconds")
    
    if Path(video_path).exists() and Path(video_path).stat().st_size > 0:
        logger.info(f"✅ FINAL VERIFICATION: PASS (Video exists, size: {Path(video_path).stat().st_size} bytes)")
    else:
        logger.error("❌ FINAL VERIFICATION: FAIL (Video file empty or missing)")

def test_cache_logic():
    logger.info("--- [TEST] Cache Logic ---")
    logger.info("Running transcription AGAIN to verify cache hit...")
    start_time = time.perf_counter()
    transcript_result = fetch_transcript(TEST_URL)
    duration = time.perf_counter() - start_time
    
    if transcript_result.source == "cache":
        logger.info(f"✅ Cache Logic: PASS (Fetched from cache in {duration:.4f}s)")
    else:
        logger.error(f"❌ Cache Logic: FAIL (Fetched from {transcript_result.source} instead of cache)")

if __name__ == "__main__":
    # Ensure logs dir exists
    Path("logs").mkdir(exist_ok=True)
    
    meta = test_stage_1()
    if meta:
        test_stage_2(meta)
        test_cache_logic()
    else:
        logger.error("Stage 1 failed, skipping Stage 2.")
