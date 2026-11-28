"""Unit tests for DW pose preprocessing/rendering helpers."""

from __future__ import annotations

from typing import cast

import numpy as np

from services.pose_processor_pipeline.dw_pose_pipeline import DWPosePipeline


def _pipeline_without_models() -> DWPosePipeline:
    return cast(DWPosePipeline, object.__new__(DWPosePipeline))


def test_apply_with_no_detected_persons_returns_black_frame() -> None:
    pipeline = _pipeline_without_models()

    def _detect_none(_frame):
        return np.empty((0, 4), dtype=np.float32)

    pipeline._detect_person_boxes = _detect_none  # type: ignore[attr-defined]

    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    output = DWPosePipeline.apply(pipeline, frame)

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not output.any()


def test_format_instances_inserts_neck_at_openpose_index_one() -> None:
    pipeline = _pipeline_without_models()

    keypoints = np.zeros((1, 133, 2), dtype=np.float32)
    scores = np.zeros((1, 133), dtype=np.float32)

    keypoints[0, 5] = np.array([10.0, 20.0], dtype=np.float32)
    keypoints[0, 6] = np.array([30.0, 40.0], dtype=np.float32)
    scores[0, 5] = 1.0
    scores[0, 6] = 1.0

    instances = pipeline._format_instances(keypoints, scores)  # type: ignore[attr-defined]

    assert len(instances) == 1
    neck = instances[0][1]
    assert np.allclose(neck[:2], np.array([20.0, 30.0], dtype=np.float32))
    assert float(neck[2]) == 1.0


def test_render_instances_draws_body_hand_and_face() -> None:
    pipeline = _pipeline_without_models()

    instance = np.zeros((134, 3), dtype=np.float32)

    # Body points used by limb (2,3) and eye fallback for face padding.
    instance[1] = np.array([20.0, 20.0, 1.0], dtype=np.float32)
    instance[2] = np.array([40.0, 20.0, 1.0], dtype=np.float32)
    instance[14] = np.array([24.0, 16.0, 1.0], dtype=np.float32)
    instance[15] = np.array([36.0, 16.0, 1.0], dtype=np.float32)

    # Face keypoint range [24:92].
    instance[24] = np.array([30.0, 30.0, 1.0], dtype=np.float32)

    # Left hand keypoints range [92:113], edge (0,1).
    instance[92] = np.array([50.0, 50.0, 1.0], dtype=np.float32)
    instance[93] = np.array([60.0, 50.0, 1.0], dtype=np.float32)

    canvas = pipeline._render_instances([instance], canvas_shape=(80, 80, 3))  # type: ignore[attr-defined]

    assert canvas.shape == (80, 80, 3)
    assert canvas.dtype == np.uint8
    assert int(canvas.sum()) > 0
