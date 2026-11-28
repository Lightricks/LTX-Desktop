"""The video dashboard leaves image runs out."""

from __future__ import annotations

from pathlib import Path

from services.features.image.feature_id import FEATURE as TEXT_TO_IMAGE_FEATURE
from tests.test_dashboard_stats import DAY, NOW, _insert, _open, _spec


def test_image_runs_are_left_out_of_the_video_dashboard(tmp_path: Path) -> None:
    # Bug caught: image runs count as video generations in the activity chart.
    store, conn = _open(tmp_path)
    for generation_id, feature, status in (
        ("video", "text-to-video", "succeeded"),
        ("image", TEXT_TO_IMAGE_FEATURE, "succeeded"),
        ("image-failed", TEXT_TO_IMAGE_FEATURE, "failed"),
    ):
        _insert(
            conn,
            generation_id=generation_id,
            feature=feature,
            status=status,
            finished=DAY,
            run_ms=1_000,
            spec=_spec(),
            error_code="IMAGE_BLOCKED_BY_SAFETY_CHECKER" if status == "failed" else None,
        )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    counts = snapshot.activity[-1].counts
    assert counts["text-to-video"] == 1
    assert sum(counts.values()) == 1
