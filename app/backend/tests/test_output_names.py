from datetime import datetime, timezone

from services.generation_queue.output_names import output_asset_name

CREATED = int(datetime(2026, 9, 9, 12, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
ASSET_ID = "abcdef12-3456-7890-abcd-ef1234567890"


def test_slugs_prompt_and_stamps_utc() -> None:
    assert (
        output_asset_name(
            prompt="A Fox, running!!",
            created_at_ms=CREATED,
            asset_id=ASSET_ID,
            mime_type="video/mp4",
        )
        == "a-fox-running-20260909-123000-abcdef.mp4"
    )


def test_empty_prompt_uses_generation() -> None:
    assert output_asset_name(
        prompt="   ",
        created_at_ms=CREATED,
        asset_id=ASSET_ID,
        mime_type="image/png",
    ).startswith("generation-20260909-123000-")


def test_non_string_prompt_uses_generation() -> None:
    name = output_asset_name(
        prompt=None,
        created_at_ms=CREATED,
        asset_id=ASSET_ID,
        mime_type="audio/wav",
    )
    assert name.startswith("generation-")
    assert name.endswith(".wav")
