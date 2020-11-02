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
import colorsys

# Don't import tensorflow
import cv2
import numpy as np


_HSV = [(1.0 * x / 256, 1.0, 1.0) for x in range(256)]
_COLORS = list(map(lambda x: colorsys.hsv_to_rgb(*x), _HSV))
_COLORS = list(
    map(
        lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        _COLORS,
    )
)
BBOX_COLORS = []
_OFFSET = [0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]
for i in range(256):
    BBOX_COLORS.append(_COLORS[(i * 16) % 256 + _OFFSET[(i * 16) // 256]])


def resize_image(
    image: np.ndarray, target_size: int, ground_truth: np.ndarray = None
):
    """
    @param image:        Dim(height, width, channels)
    @param target_size
    @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

    @return resized_image or (resized_image, resized_ground_truth)

    Usage:
        image = media.resize_image(image, yolo.input_size)
        image, ground_truth = media.resize_image(image, yolo.input_size,
                                                                ground_truth)
    """
    height, width, _ = image.shape

    # Resize
    if max(height, width) != target_size:
        scale = min(target_size / width, target_size / height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        new_width = width
        new_height = height
        resized_image = np.copy(image)

    # Pad
    dw = int(target_size - new_width)
    dh = int(target_size - new_height)

    if dw != 0 or dh != 0:
        dw = dw // 2
        dh = dh // 2
        padded_image = np.full(
            (target_size, target_size, 3), 255, dtype=np.uint8
        )
        padded_image[
            dh : new_height + dh, dw : new_width + dw, :
        ] = resized_image
    else:
        padded_image = resized_image

    if ground_truth is None:
        return padded_image

    # Resize ground truth
    ground_truth = np.copy(ground_truth)

    if dw > dh:
        w_h = new_width / new_height
        ground_truth[:, 0] = w_h * (ground_truth[:, 0] - 0.5) + 0.5
        ground_truth[:, 2] = w_h * ground_truth[:, 2]
    elif dw < dh:
        h_w = new_height / new_width
        ground_truth[:, 1] = h_w * (ground_truth[:, 1] - 0.5) + 0.5
        ground_truth[:, 3] = h_w * ground_truth[:, 3]

    return padded_image, ground_truth


def draw_bboxes(image: np.ndarray, bboxes: np.ndarray, classes: dict):
    """
    @parma image:  Dim(height, width, channel)
    @param bboxes: (candidates, 4) or (candidates, 5)
            [[center_x, center_y, w, h, class_id], ...]
            [[center_x, center_y, w, h, class_id, propability], ...]
    @param classes: {0: 'person', 1: 'bicycle', 2: 'car', ...}

    @return drawn_image

    Usage:
        image = media.draw_bboxes(image, bboxes, classes)
    """
    image = np.copy(image)
    height, width, _ = image.shape

    # Set propability
    if bboxes.shape[-1] == 5:
        bboxes = np.concatenate(
            [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
        )
    else:
        bboxes = np.copy(bboxes)

    # Convert ratio to length
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height

    # Draw bboxes
    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        half_w = int(bbox[2] / 2)
        half_h = int(bbox[3] / 2)
        top_left = (c_x - half_w, c_y - half_h)
        bottom_right = (c_x + half_w, c_y + half_h)
        class_id = int(bbox[4])
        bbox_color = BBOX_COLORS[class_id]
        font_size = 0.4
        font_thickness = 1

        # Draw box
        cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)

        # Draw text box
        bbox_text = "{}: {:.1%}".format(classes[class_id], bbox[5])
        t_size = cv2.getTextSize(bbox_text, 0, font_size, font_thickness)[0]
        cv2.rectangle(
            image,
            top_left,
            (top_left[0] + t_size[0], top_left[1] - t_size[1] - 3),
            bbox_color,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            bbox_text,
            (top_left[0], top_left[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255 - bbox_color[0], 255 - bbox_color[1], 255 - bbox_color[2]),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return image


def read_classes_names(classes_name_path):
    """
    @return {id: class name}
    """
    classes = {}
    with open(classes_name_path, "r") as fd:
        index = 0
        for class_name in fd:
            class_name = class_name.strip()
            if len(class_name) != 0:
                classes[index] = class_name.replace(" ", "_")
                index += 1

    return classes


