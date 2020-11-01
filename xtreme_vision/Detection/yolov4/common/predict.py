"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


def DIoU_NMS(candidates, threshold):
    """
    Distance Intersection over Union(DIoU)
    Non-Maximum Suppression(NMS)

    @param candidates: [[center_x, center_y, w, h, class_id, propability], ...]
    """
    bboxes = []
    for class_id in set(candidates[:, 4]):
        class_bboxes = candidates[candidates[:, 4] == class_id]
        if class_bboxes.shape[0] == 1:
            # One candidate
            bboxes.append(class_bboxes)
            continue

        while True:
            half = class_bboxes[:, 2:4] * 0.5
            M_index = np.argmax(class_bboxes[:, 5])
            M_bbox = class_bboxes[M_index, :]
            M_half = half[M_index, :]
            # Max probability
            bboxes.append(M_bbox[np.newaxis, :])

            enclose_left = np.minimum(
                class_bboxes[:, 0] - half[:, 0],
                M_bbox[0] - M_half[0],
            )
            enclose_right = np.maximum(
                class_bboxes[:, 0] + half[:, 0],
                M_bbox[0] + M_half[0],
            )
            enclose_top = np.minimum(
                class_bboxes[:, 1] - half[:, 1],
                M_bbox[1] - M_half[1],
            )
            enclose_bottom = np.maximum(
                class_bboxes[:, 1] + half[:, 1],
                M_bbox[1] + M_half[1],
            )

            enclose_width = enclose_right - enclose_left
            enclose_height = enclose_bottom - enclose_top

            width_mask = enclose_width >= class_bboxes[:, 2] + M_bbox[2]
            height_mask = enclose_height >= class_bboxes[:, 3] + M_bbox[3]
            other_mask = np.logical_or(width_mask, height_mask)
            other_bboxes = class_bboxes[other_mask]

            mask = np.logical_not(other_mask)
            class_bboxes = class_bboxes[mask]
            if class_bboxes.shape[0] == 1:
                if other_bboxes.shape[0] == 1:
                    bboxes.append(other_bboxes)
                    break

                class_bboxes = other_bboxes
                continue

            half = half[mask]
            enclose_left = enclose_left[mask]
            enclose_right = enclose_right[mask]
            enclose_top = enclose_top[mask]
            enclose_bottom = enclose_bottom[mask]

            inter_left = np.maximum(
                class_bboxes[:, 0] - half[:, 0],
                M_bbox[0] - M_half[0],
            )
            inter_right = np.minimum(
                class_bboxes[:, 0] + half[:, 0],
                M_bbox[0] + M_half[0],
            )
            inter_top = np.maximum(
                class_bboxes[:, 1] - half[:, 1],
                M_bbox[1] - M_half[1],
            )
            inter_bottom = np.minimum(
                class_bboxes[:, 1] + half[:, 1],
                M_bbox[1] + M_half[1],
            )

            class_area = class_bboxes[:, 2] * class_bboxes[:, 3]
            M_area = M_bbox[2] * M_bbox[3]
            inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
            iou = inter_area / (class_area + M_area)

            c = (enclose_right - enclose_left) * (
                enclose_right - enclose_left
            ) + (enclose_bottom - enclose_top) * (enclose_bottom - enclose_top)
            d = (class_bboxes[:, 0] - M_bbox[0]) * (
                class_bboxes[:, 0] - M_bbox[0]
            ) + (class_bboxes[:, 1] - M_bbox[1]) * (
                class_bboxes[:, 1] - M_bbox[1]
            )

            # DIoU = IoU - d^2 / c^2
            other_mask = iou - d / c < threshold
            other2_bboxes = class_bboxes[other_mask]
            if other_bboxes.shape[0] != 0 and other2_bboxes.shape[0] != 0:
                class_bboxes = np.concatenate(
                    [other_bboxes, other2_bboxes], axis=0
                )
                continue

            if other_bboxes.shape[0] != 0:
                if other_bboxes.shape[0] == 1:
                    bboxes.append(other_bboxes)
                    break

                class_bboxes = other_bboxes
                continue

            if other2_bboxes.shape[0] != 0:
                if other2_bboxes.shape[0] == 1:
                    bboxes.append(other2_bboxes)
                    break

                class_bboxes = other2_bboxes
                continue

            break

    if len(bboxes) == 0:
        return np.zeros(shape=(1, 6))

    return np.concatenate(bboxes, axis=0)


def candidates_to_pred_bboxes(
    candidates,
    input_size,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.25,
):
    """
    @param candidates: Dim(-1, (x, y, w, h, obj_score, probabilities))

    @return Dim(-1, (x, y, w, h, class_id, class_probability))
    """
    # Remove low socre candidates
    # This step should be the first !!
    class_ids = np.argmax(candidates[:, 5:], axis=-1)
    # class_prob = obj_score * max_probability
    class_prob = (
        candidates[:, 4] * candidates[np.arange(len(candidates)), class_ids + 5]
    )
    candidates = candidates[class_prob > score_threshold, :]

    # Remove out of range candidates
    half = candidates[:, 2:4] * 0.5
    mask = candidates[:, 0] - half[:, 0] >= 0
    candidates = candidates[mask, :]
    half = half[mask, :]
    mask = candidates[:, 0] + half[:, 0] <= 1
    candidates = candidates[mask, :]
    half = half[mask, :]
    mask = candidates[:, 1] - half[:, 1] >= 0
    candidates = candidates[mask, :]
    half = half[mask, :]
    mask = candidates[:, 1] + half[:, 1] <= 1
    candidates = candidates[mask, :]

    # Remove small candidates
    candidates = candidates[
        np.logical_and(
            candidates[:, 2] > 2 / input_size,
            candidates[:, 3] > 2 / input_size,
        ),
        :,
    ]

    class_ids = np.argmax(candidates[:, 5:], axis=-1)
    class_prob = (
        candidates[:, 4] * candidates[np.arange(len(candidates)), class_ids + 5]
    )

    # x, y, w, h, class_id, class_probability
    candidates = np.concatenate(
        [
            candidates[:, :4],
            class_ids[:, np.newaxis],
            class_prob[:, np.newaxis],
        ],
        axis=-1,
    )

    return DIoU_NMS(candidates, iou_threshold)


def fit_pred_bboxes_to_original(bboxes, original_shape):
    """
    @param bboxes: Dim(-1, (x, y, w, h, class_id, probability))
    @param original_shape: (height, width, channels)
    """

    height, width, _ = original_shape

    bboxes = np.copy(bboxes)
    if width > height:
        w_h = width / height
        bboxes[:, 1] = w_h * (bboxes[:, 1] - 0.5) + 0.5
        bboxes[:, 3] = w_h * bboxes[:, 3]
    elif width < height:
        h_w = height / width
        bboxes[:, 0] = h_w * (bboxes[:, 0] - 0.5) + 0.5
        bboxes[:, 2] = h_w * bboxes[:, 2]

    return bboxes
