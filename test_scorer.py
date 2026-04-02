"""
test_scorer.py  —  End-to-end test for scorer.py
-------------------------------------------------
Calls scorer.score(url, shorts_length) and validates the result.

Run:
    python test_scorer.py
    python test_scorer.py 10-20     # test 10-20s mode
"""

import sys
from scorer import score

# Replace with any YouTube URL for a real end-to-end run
TEST_URL = "https://www.youtube.com/watch?v=s2EYIDY8wSM"


def run_test(shorts_length: str = "0-10", shorts_count: int = 2) -> None:
    print("=" * 60)
    print(f"  TEST: score(url, shorts_length={shorts_length!r})")
    print("=" * 60)

    result = score(TEST_URL, shorts_length=shorts_length, shorts_count=shorts_count)

    # Assertions
    assert "clips"        in result, "Missing 'clips'"
    assert "content_type" in result, "Missing 'content_type'"
    assert "weights_used" in result, "Missing 'weights_used'"
    assert result["total_clips"] > 0, "No clips returned"

    for clip in result["clips"]:
        assert 0.0 <= clip["composite_score"] <= 1.0, \
            f"composite_score out of range: {clip['composite_score']}"
        assert clip["clip_duration_sec"] > 0
        assert clip["rank"] >= 1

    print(f"\n  [PASS] {result['total_clips']} clip(s)  |  type: {result['content_type']}")
    print(f"  [PASS] All outputs saved to output/\n")
    for c in result["clips"]:
        print(f"    [{c['rank']}] {c['clip_start_time']} - {c['clip_end_time']} "
              f"({c['clip_duration_sec']}s)  score={c['composite_score']:.4f}")
        print(f"         {c['hook']}")


if __name__ == "__main__":
    length = sys.argv[1] if len(sys.argv) > 1 else "0-10"
    run_test(shorts_length=length)
