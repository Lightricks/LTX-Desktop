"""Unit tests for ConditioningCache."""

from __future__ import annotations

from pathlib import Path

from state.conditioning_cache import (
    ConditioningCache,
    ConditioningCacheEntry,
    ConditioningCacheKey,
)


class TestConditioningCache:
    def test_get_returns_none_for_missing_key(self):
        cache = ConditioningCache()
        key = ConditioningCacheKey("video.mp4", "canny")
        assert cache.get(key) is None

    def test_put_then_get_returns_entry(self):
        cache = ConditioningCache()
        key = ConditioningCacheKey("video.mp4", "canny")
        entry = ConditioningCacheEntry("/tmp/control.mp4", 121, 24.0)
        cache.put(key, entry)
        assert cache.get(key) == entry

    def test_cleanup_removes_files_and_clears(self, tmp_path: Path):
        cache = ConditioningCache()
        file1 = tmp_path / "control1.mp4"
        file2 = tmp_path / "control2.mp4"
        file1.write_bytes(b"data1")
        file2.write_bytes(b"data2")

        cache.put(
            ConditioningCacheKey("a.mp4", "canny"),
            ConditioningCacheEntry(str(file1), 10, 24.0),
        )
        cache.put(
            ConditioningCacheKey("b.mp4", "depth"),
            ConditioningCacheEntry(str(file2), 20, 30.0),
        )

        cache.cleanup()

        assert not file1.exists()
        assert not file2.exists()
        assert cache.get(ConditioningCacheKey("a.mp4", "canny")) is None

    def test_cleanup_tolerates_already_deleted_files(self, tmp_path: Path):
        cache = ConditioningCache()
        missing = tmp_path / "gone.mp4"
        cache.put(
            ConditioningCacheKey("v.mp4", "canny"),
            ConditioningCacheEntry(str(missing), 5, 24.0),
        )
        cache.cleanup()  # should not raise

    def test_del_triggers_cleanup(self, tmp_path: Path):
        cache = ConditioningCache()
        file = tmp_path / "control.mp4"
        file.write_bytes(b"data")
        cache.put(
            ConditioningCacheKey("v.mp4", "canny"),
            ConditioningCacheEntry(str(file), 10, 24.0),
        )
        del cache
        assert not file.exists()
