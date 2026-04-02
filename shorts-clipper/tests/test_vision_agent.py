import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.vision_agent import _region_selection_score
from api.models import Region


def test_region_selection_prefers_screenshare_layout():
    screenshare_regions = [
        Region(label="face", x1=50, y1=50, x2=250, y2=300),
        Region(label="screen", x1=200, y1=100, x2=950, y2=900),
    ]
    single_face_regions = [
        Region(label="face", x1=300, y1=100, x2=600, y2=700),
    ]

    assert _region_selection_score(screenshare_regions) > _region_selection_score(single_face_regions)


def test_region_selection_prefers_three_faces_over_two_faces():
    three_faces = [
        Region(label="face", x1=10, y1=10, x2=200, y2=300),
        Region(label="face", x1=250, y1=10, x2=450, y2=300),
        Region(label="face", x1=500, y1=10, x2=700, y2=300),
    ]
    two_faces = [
        Region(label="face", x1=10, y1=10, x2=200, y2=300),
        Region(label="face", x1=250, y1=10, x2=450, y2=300),
    ]

    assert _region_selection_score(three_faces) > _region_selection_score(two_faces)
