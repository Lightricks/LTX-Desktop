"""DW Pose pipeline with YOLOX person detection and OpenPose-style rendering."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from services.services_utils import FrameArray

_DETECTOR_INPUT_SIZE = (640, 640)
_POSE_INPUT_SIZE = (288, 384)  # (width, height)
_POSE_BATCH_SIZE = 5
_SIMCC_SPLIT_RATIO = 2.0
_CONFIDENCE_THRESHOLD = 0.3
_EPS = 0.01


class DWPosePipeline:
    @staticmethod
    def create(
        pose_model_path: str,
        person_detector_model_path: str,
        device: torch.device,
    ) -> "DWPosePipeline":
        return DWPosePipeline(
            pose_model_path=pose_model_path,
            person_detector_model_path=person_detector_model_path,
            device=device,
        )

    def __init__(
        self,
        pose_model_path: str,
        person_detector_model_path: str,
        device: torch.device,
    ) -> None:
        self._device = device
        jit_module = cast(Any, torch.jit)
        # dw-ll_ucoco_384_bs5 embeds float64 scalar constants in HardSigmoid; map_location=mps fails.
        self._pose_model = cast(torch.jit.ScriptModule, jit_module.load(pose_model_path, map_location="cpu")).to(device)
        self._pose_model.eval()

        self._detector_model = cast(torch.jit.ScriptModule, jit_module.load(person_detector_model_path, map_location="cpu")).to(device)
        self._detector_model.eval()

    def _module_device_dtype(self, module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
        for param in module.parameters():
            return param.device, param.dtype
        return self._device, torch.float32

    def _nms(self, boxes: Any, scores: Any, nms_threshold: float) -> list[int]:
        import numpy as np

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            overlap = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(overlap <= nms_threshold)[0]
            order = order[indices + 1]

        return keep

    def _multiclass_nms(self, boxes: Any, scores: Any, nms_threshold: float, score_threshold: float) -> Any:
        import numpy as np

        final_dets: list[Any] = []
        num_classes = int(scores.shape[1])

        for class_index in range(num_classes):
            class_scores = scores[:, class_index]
            valid_mask = class_scores > score_threshold
            if int(valid_mask.sum()) == 0:
                continue

            valid_scores = class_scores[valid_mask]
            valid_boxes = boxes[valid_mask]
            keep = self._nms(valid_boxes, valid_scores, nms_threshold)
            if not keep:
                continue

            class_indices = np.ones((len(keep), 1), dtype=np.float32) * float(class_index)
            detections = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], class_indices],
                axis=1,
            )
            final_dets.append(detections)

        if not final_dets:
            return None
        return np.concatenate(final_dets, axis=0)

    def _detector_preprocess(self, frame: FrameArray) -> tuple[Any, float]:
        import cv2
        import numpy as np

        if frame.ndim == 3:
            padded = np.ones((_DETECTOR_INPUT_SIZE[0], _DETECTOR_INPUT_SIZE[1], 3), dtype=np.uint8) * 114
        else:
            padded = np.ones(_DETECTOR_INPUT_SIZE, dtype=np.uint8) * 114

        ratio = min(_DETECTOR_INPUT_SIZE[0] / frame.shape[0], _DETECTOR_INPUT_SIZE[1] / frame.shape[1])
        resized = cv2.resize(
            frame,
            (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        padded[: resized.shape[0], : resized.shape[1]] = resized
        chw = padded.transpose(2, 0, 1)
        contiguous = np.ascontiguousarray(chw, dtype=np.float32)
        return contiguous, float(ratio)

    def _detector_postprocess(self, outputs: Any) -> Any:
        import numpy as np

        grids: list[Any] = []
        expanded_strides: list[Any] = []
        strides = [8, 16, 32]

        hsizes = [int(_DETECTOR_INPUT_SIZE[0] / stride) for stride in strides]
        wsizes = [int(_DETECTOR_INPUT_SIZE[1] / stride) for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            x, y = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((x, y), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride))

        all_grids = np.concatenate(grids, axis=1)
        all_expanded_strides = np.concatenate(expanded_strides, axis=1)

        outputs[..., :2] = (outputs[..., :2] + all_grids) * all_expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * all_expanded_strides
        return outputs

    def _detect_person_boxes(self, frame: FrameArray) -> Any:
        import numpy as np

        detector_input, ratio = self._detector_preprocess(frame)

        detector_device, detector_dtype = self._module_device_dtype(self._detector_model)
        input_tensor = torch.as_tensor(detector_input[None, :, :, :], dtype=torch.float32).to(
            detector_device,
            detector_dtype,
        )

        raw_output = self._detector_model(input_tensor)
        if not isinstance(raw_output, torch.Tensor):
            raise RuntimeError("YOLOX detector returned unexpected output type")

        predictions = raw_output.float().detach().cpu().numpy()
        decoded = self._detector_postprocess(predictions[0])

        boxes = decoded[:, :4]
        scores = decoded[:, 4:5] * decoded[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets = self._multiclass_nms(boxes_xyxy, scores, nms_threshold=0.45, score_threshold=0.1)
        if dets is None:
            return np.empty((0, 4), dtype=np.float32)

        final_boxes = dets[:, :4]
        final_scores = dets[:, 4]
        final_cls = dets[:, 5]

        keep_score = final_scores > _CONFIDENCE_THRESHOLD
        keep_person = np.isin(final_cls, [0.0])
        keep = np.logical_and(keep_score, keep_person)

        filtered = final_boxes[keep]
        if filtered.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return filtered.astype(np.float32)

    def _bbox_xyxy_to_center_scale(self, bbox: Any, padding: float = 1.0) -> tuple[Any, Any]:
        import numpy as np

        dim = int(bbox.ndim)
        box = bbox[None, :] if dim == 1 else bbox

        x1, y1, x2, y2 = np.hsplit(box, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            return center[0], scale[0]
        return center, scale

    def _fix_aspect_ratio(self, bbox_scale: Any, aspect_ratio: float) -> Any:
        import numpy as np

        width, height = np.hsplit(bbox_scale, [1])
        return np.where(
            width > height * aspect_ratio,
            np.hstack([width, width / aspect_ratio]),
            np.hstack([height * aspect_ratio, height]),
        )

    def _rotate_point(self, point: Any, angle_rad: float) -> Any:
        import numpy as np

        sin_v, cos_v = np.sin(angle_rad), np.cos(angle_rad)
        rotation_matrix = np.array([[cos_v, -sin_v], [sin_v, cos_v]])
        return rotation_matrix @ point

    def _third_point(self, point_a: Any, point_b: Any) -> Any:
        import numpy as np

        direction = point_a - point_b
        return point_b + np.r_[-direction[1], direction[0]]

    def _warp_matrix(
        self,
        center: Any,
        scale: Any,
        rotation_deg: float,
        output_size: tuple[int, int],
    ) -> Any:
        import cv2
        import numpy as np

        src_w = scale[0]
        dst_w, dst_h = output_size

        rotation_rad = np.deg2rad(rotation_deg)
        src_dir = self._rotate_point(np.array([0.0, src_w * -0.5]), rotation_rad)
        dst_dir = np.array([0.0, dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        src[2, :] = self._third_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._third_point(dst[0, :], dst[1, :])

        cv2_module = cast(Any, cv2)
        return cv2_module.getAffineTransform(np.float32(src), np.float32(dst))

    def _top_down_affine(self, img: FrameArray, bbox_scale: Any, bbox_center: Any) -> tuple[FrameArray, Any]:
        import cv2

        width, height = _POSE_INPUT_SIZE
        reshaped_scale = self._fix_aspect_ratio(bbox_scale, aspect_ratio=width / height)

        warp_size = (int(width), int(height))
        warp = self._warp_matrix(bbox_center, reshaped_scale[0], rotation_deg=0.0, output_size=(width, height))
        warped = cv2.warpAffine(img, warp, warp_size, flags=cv2.INTER_LINEAR)
        return cast(FrameArray, warped), reshaped_scale

    def _preprocess_pose(self, frame: FrameArray, boxes: Any) -> tuple[list[Any], Any, Any]:
        import numpy as np

        cropped_images: list[Any] = []
        centers: list[Any] = []
        scales: list[Any] = []

        for i in range(int(boxes.shape[0])):
            bbox = boxes[i]
            center, scale = self._bbox_xyxy_to_center_scale(np.asarray(bbox), padding=1.25)
            affine_image, affine_scale = self._top_down_affine(frame, np.asarray(scale)[None, ...], np.asarray(center))

            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            normalized = (affine_image.astype(np.float32) - mean) / std

            cropped_images.append(normalized)
            centers.append(center)
            scales.append(affine_scale[0])

        return cropped_images, np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)

    def _infer_pose_model(self, images: list[Any]) -> tuple[Any, Any]:
        import numpy as np

        if not images:
            return (
                np.empty((0, 133, 2), dtype=np.float32),
                np.empty((0, 133), dtype=np.float32),
            )

        original_count = len(images)
        remainder = original_count % _POSE_BATCH_SIZE
        pad_count = 0 if remainder == 0 else (_POSE_BATCH_SIZE - remainder)

        padded_images = list(images)
        if pad_count:
            padded_images.extend([np.zeros_like(images[0]) for _ in range(pad_count)])

        batch = np.stack(padded_images, axis=0).transpose(0, 3, 1, 2)
        pose_device, pose_dtype = self._module_device_dtype(self._pose_model)
        input_tensor = torch.as_tensor(batch, dtype=torch.float32).to(pose_device, pose_dtype)

        out_x_tensors: list[torch.Tensor] = []
        out_y_tensors: list[torch.Tensor] = []

        for i in range(0, int(input_tensor.shape[0]), _POSE_BATCH_SIZE):
            output_obj: object = self._pose_model(input_tensor[i : i + _POSE_BATCH_SIZE])
            if not isinstance(output_obj, (tuple, list)):
                raise RuntimeError("DW Pose model returned unexpected output structure")
            output_seq = cast(tuple[object, ...], output_obj)
            if len(output_seq) != 2:
                raise RuntimeError("DW Pose model returned unexpected output structure")
            simcc_x_obj, simcc_y_obj = output_seq[0], output_seq[1]
            if not isinstance(simcc_x_obj, torch.Tensor) or not isinstance(simcc_y_obj, torch.Tensor):
                raise RuntimeError("DW Pose model returned non-tensor outputs")
            out_x_tensors.append(simcc_x_obj.float())
            out_y_tensors.append(simcc_y_obj.float())

        simcc_x = torch.cat(out_x_tensors, dim=0)[:original_count].detach().cpu().numpy()
        simcc_y = torch.cat(out_y_tensors, dim=0)[:original_count].detach().cpu().numpy()

        keypoints, scores = self._decode_pose_outputs(simcc_x, simcc_y)
        return keypoints, scores

    def _simcc_maximum(self, simcc_x: Any, simcc_y: Any) -> tuple[Any, Any]:
        import numpy as np

        batch_size, keypoint_count, _ = simcc_x.shape

        x_flat = simcc_x.reshape(batch_size * keypoint_count, -1)
        y_flat = simcc_y.reshape(batch_size * keypoint_count, -1)

        x_locations = np.argmax(x_flat, axis=1)
        y_locations = np.argmax(y_flat, axis=1)
        locations = np.stack((x_locations, y_locations), axis=-1).astype(np.float32)

        max_val_x = np.amax(x_flat, axis=1)
        max_val_y = np.amax(y_flat, axis=1)

        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]

        values = max_val_x
        locations[values <= 0.0] = -1

        return (
            locations.reshape(batch_size, keypoint_count, 2),
            values.reshape(batch_size, keypoint_count),
        )

    def _decode_pose_outputs(self, simcc_x: Any, simcc_y: Any) -> tuple[Any, Any]:
        keypoints, scores = self._simcc_maximum(simcc_x, simcc_y)
        keypoints /= _SIMCC_SPLIT_RATIO
        return keypoints, scores

    def _rescale_keypoints(self, keypoints: Any, centers: Any, scales: Any) -> Any:
        import numpy as np

        input_size = np.asarray(_POSE_INPUT_SIZE, dtype=np.float32)
        return keypoints / input_size * scales[:, None, :] + centers[:, None, :] - scales[:, None, :] / 2.0

    def _to_optional_point(self, point_with_score: Any) -> tuple[float, float, float] | None:
        score = float(point_with_score[2])
        if score < _CONFIDENCE_THRESHOLD:
            return None
        return (float(point_with_score[0]), float(point_with_score[1]), score)

    def _draw_body_pose(self, canvas: FrameArray, keypoints: list[tuple[float, float, float] | None]) -> None:
        import cv2

        stick_width = 4
        limb_seq: tuple[tuple[int, int], ...] = (
            (2, 3),
            (2, 6),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (2, 9),
            (9, 10),
            (10, 11),
            (2, 12),
            (12, 13),
            (13, 14),
            (2, 1),
            (1, 15),
            (15, 17),
            (1, 16),
            (16, 18),
        )
        colors: tuple[tuple[int, int, int], ...] = (
            (255, 0, 0),
            (255, 85, 0),
            (255, 170, 0),
            (255, 255, 0),
            (170, 255, 0),
            (85, 255, 0),
            (0, 255, 0),
            (0, 255, 85),
            (0, 255, 170),
            (0, 255, 255),
            (0, 170, 255),
            (0, 85, 255),
            (0, 0, 255),
            (85, 0, 255),
            (170, 0, 255),
            (255, 0, 255),
            (255, 0, 170),
            (255, 0, 85),
        )

        for (start_idx, end_idx), color in zip(limb_seq, colors):
            point1 = keypoints[start_idx - 1]
            point2 = keypoints[end_idx - 1]
            if point1 is None or point2 is None:
                continue

            x1, y1 = point1[0], point1[1]
            x2, y2 = point2[0], point2[1]

            mean_x = (x1 + x2) * 0.5
            mean_y = (y1 + y2) * 0.5
            length = math.hypot(y1 - y2, x1 - x2)
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))

            polygon = cv2.ellipse2Poly(
                (int(mean_x), int(mean_y)),
                (int(length / 2), stick_width),
                int(angle),
                0,
                360,
                1,
            )
            cv2.fillConvexPoly(canvas, cast(Any, polygon), [int(float(c) * 0.6) for c in color])

        for keypoint, color in zip(keypoints, colors):
            if keypoint is None:
                continue
            cv2.circle(canvas, (int(keypoint[0]), int(keypoint[1])), 4, color, thickness=-1)

    def _edge_color_bgr(self, edge_index: int, total_edges: int) -> tuple[int, int, int]:
        import cv2
        import numpy as np

        denominator = float(total_edges) if total_edges > 0 else 1.0
        hue = int(round((float(edge_index) / denominator) * 179.0))
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _draw_hand_pose(self, canvas: FrameArray, keypoints: list[tuple[float, float, float] | None]) -> None:
        import cv2

        if not keypoints:
            return

        edges: tuple[tuple[int, int], ...] = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
        )

        for edge_index, (start_idx, end_idx) in enumerate(edges):
            point1 = keypoints[start_idx]
            point2 = keypoints[end_idx]
            if point1 is None or point2 is None:
                continue

            x1, y1 = int(point1[0]), int(point1[1])
            x2, y2 = int(point2[0]), int(point2[1])
            if x1 <= _EPS or y1 <= _EPS or x2 <= _EPS or y2 <= _EPS:
                continue

            edge_color = self._edge_color_bgr(edge_index, len(edges))
            cv2.line(canvas, (x1, y1), (x2, y2), edge_color, thickness=2)

        for keypoint in keypoints:
            if keypoint is None:
                continue
            x, y = int(keypoint[0]), int(keypoint[1])
            if x <= _EPS or y <= _EPS:
                continue
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    def _draw_face_pose(self, canvas: FrameArray, keypoints: list[tuple[float, float, float] | None]) -> None:
        import cv2

        if not keypoints:
            return

        for keypoint in keypoints:
            if keypoint is None:
                continue
            x, y = int(keypoint[0]), int(keypoint[1])
            if x <= _EPS or y <= _EPS:
                continue
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

    def _render_instances(self, instances: list[Any], canvas_shape: tuple[int, int, int]) -> FrameArray:
        import numpy as np

        canvas = np.zeros(canvas_shape, dtype=np.uint8)

        for instance in instances:
            body_raw = instance[:18]
            face_raw = instance[24:92]
            left_hand_raw = instance[92:113]
            right_hand_raw = instance[113:134]

            body_points = [self._to_optional_point(point) for point in body_raw]
            face_points = [self._to_optional_point(point) for point in face_raw]
            left_hand_points = [self._to_optional_point(point) for point in left_hand_raw]
            right_hand_points = [self._to_optional_point(point) for point in right_hand_raw]

            face_points.append(body_points[14])
            face_points.append(body_points[15])

            self._draw_body_pose(canvas, body_points)
            self._draw_hand_pose(canvas, left_hand_points)
            self._draw_hand_pose(canvas, right_hand_points)
            self._draw_face_pose(canvas, face_points)

        return cast(FrameArray, canvas)

    def _format_instances(self, keypoints: Any, scores: Any) -> list[Any]:
        import numpy as np

        if keypoints.size == 0 or scores.size == 0:
            return []

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

        neck_xy = np.mean(keypoints_info[:, [5, 6], :2], axis=1)
        neck_confidence = np.logical_and(
            keypoints_info[:, 5, 2] > _CONFIDENCE_THRESHOLD,
            keypoints_info[:, 6, 2] > _CONFIDENCE_THRESHOLD,
        ).astype(np.float32)
        neck = np.concatenate((neck_xy, neck_confidence[:, None]), axis=1)

        remapped = np.insert(keypoints_info, 17, neck, axis=1)

        mmpose_idx = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3], dtype=np.int64)
        openpose_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17], dtype=np.int64)
        remapped[:, openpose_idx] = remapped[:, mmpose_idx]

        return [remapped[i] for i in range(int(remapped.shape[0]))]

    @torch.inference_mode()
    def apply(self, frame: FrameArray) -> FrameArray:
        import numpy as np

        boxes = self._detect_person_boxes(frame)
        if boxes.size == 0:
            return cast(FrameArray, np.zeros(frame.shape, dtype=np.uint8))

        images, centers, scales = self._preprocess_pose(frame, boxes)
        keypoints, scores = self._infer_pose_model(images)
        if keypoints.size == 0:
            return cast(FrameArray, np.zeros(frame.shape, dtype=np.uint8))

        rescaled_keypoints = self._rescale_keypoints(keypoints, centers, scales)
        instances = self._format_instances(rescaled_keypoints, scores)
        return self._render_instances(instances, canvas_shape=frame.shape)
