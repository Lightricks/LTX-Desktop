"""colour-science must not replace matplotlib with mocks.

Without a real matplotlib, ``colour.plotting`` fills ``sys.modules`` with stand-in
objects. ``diffusers`` then crashes at import: ``find_spec("matplotlib")`` raises
``ValueError: matplotlib.__spec__ is not set``.
"""

from __future__ import annotations

import importlib.util


def test_matplotlib_stays_real_after_ltx_imports() -> None:
    import ltx_pipelines.utils.media_io  # noqa: F401  # pulls in colour

    import matplotlib

    assert matplotlib.__spec__ is not None
    assert importlib.util.find_spec("matplotlib") is not None
    assert isinstance(matplotlib.__version__, str)


def test_diffusers_imports_after_ltx_imports() -> None:
    import ltx_pipelines.utils.media_io  # noqa: F401
    from diffusers.pipelines.auto_pipeline import ZImagePipeline  # noqa: F401
