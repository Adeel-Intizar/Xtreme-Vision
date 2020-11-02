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
from os import path
import time
from typing import Union

import cv2
import numpy as np

from xtreme_vision.Detection.yolov4.common import media, predict


class BaseClass:
    def __init__(self, tiny: bool = False, tpu: bool = False):
        """
        Default configuration
        """
        self.tiny = tiny
        self.tpu = tpu

        # properties
        if tiny:
            self.anchors = [
                [[23, 27], [37, 58], [81, 82]],
                [[81, 82], [135, 169], [344, 319]],
            ]
        else:
            self.anchors = [
                [[12, 16], [19, 36], [40, 28]],
                [[36, 75], [76, 55], [72, 146]],
                [[142, 110], [192, 243], [459, 401]],
            ]
        self._classes = None
        self._input_size = None
        if tiny:
            self.strides = [16, 32]
        else:
            self.strides = [8, 16, 32]
        if tiny:
            self.xyscales = [1.05, 1.05]
        else:
            self.xyscales = [1.2, 1.1, 1.05]

    @property
    def anchors(self):
        """
        Usage:
            yolo.anchors = [12, 16, 19, 36, 40, 28, 36, 75,
                            76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
            yolo.anchors = np.array([12, 16, 19, 36, 40, 28, 36, 75,
                            76, 55, 72, 146, 142, 110, 192, 243, 459, 401])
            print(yolo.anchors)
        """
        return self._anchors

    @anchors.setter
    def anchors(self, anchors: Union[list, tuple, np.ndarray]):
        if isinstance(anchors, (list, tuple)):
            self._anchors = np.array(anchors)
        elif isinstance(anchors, np.ndarray):
            self._anchors = anchors

        if self.tiny:
            self._anchors = self._anchors.astype(np.float32).reshape(2, 3, 2)
        else:
            self._anchors = self._anchors.astype(np.float32).reshape(3, 3, 2)

    @property
    def classes(self):
        """
        Usage:
            yolo.classes = {0: 'person', 1: 'bicycle', 2: 'car', ...}
            yolo.classes = "path/classes"
            print(len(yolo.classes))
        """
        return self._classes

    @classes.setter
    def classes(self, data: Union[str, dict]):
        if isinstance(data, str):
            self._classes = media.read_classes_names(data)
        elif isinstance(data, dict):
            self._classes = data
        else:
            raise TypeError("YOLOv4: Set classes path or dictionary")

    @property
    def input_size(self):
        """
        Usage:
            yolo.input_size = 608
            print(yolo.input_size)
        """
        return self._input_size

    @input_size.setter
    def input_size(self, size: int):
        if size % 32 == 0:
            self._input_size = size
        else:
            raise ValueError("YOLOv4: Set input_size to multiples of 32")

    @property
    def strides(self):
        """
        Usage:
            yolo.strides = [8, 16, 32]
            yolo.strides = np.array([8, 16, 32])
            print(yolo.strides)
        """
        return self._strides

    @strides.setter
    def strides(self, strides: Union[list, tuple, np.ndarray]):
        if isinstance(strides, (list, tuple)):
            self._strides = np.array(strides)
        elif isinstance(strides, np.ndarray):
            self._strides = strides

    @property
    def xyscales(self):
        """
        Usage:
            yolo.xyscales = [1.2, 1.1, 1.05]
            yolo.xyscales = np.array([1.2, 1.1, 1.05])
            print(yolo.xyscales)
        """
        return self._xyscales

    @xyscales.setter
    def xyscales(self, xyscales: Union[list, tuple, np.ndarray]):
        if isinstance(xyscales, (list, tuple)):
            self._xyscales = np.array(xyscales)
        elif isinstance(xyscales, np.ndarray):
            self._xyscales = xyscales

    def resize_image(self, image, ground_truth=None):
        """
        @param image:        Dim(height, width, channels)
        @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

        @return resized_image or (resized_image, resized_ground_truth)

        Usage:
            image = yolo.resize_image(image)
            image, ground_truth = yolo.resize_image(image, ground_truth)
        """
        return media.resize_image(
            image, target_size=self.input_size, ground_truth=ground_truth
        )

    def candidates_to_pred_bboxes(
        self, candidates, iou_threshold, score_threshold
    ):
        """
        @param candidates: Dim(-1, (x, y, w, h, conf, prob_0, prob_1, ...))

        @return Dim(-1, (x, y, w, h, class_id, probability))
        """
        return predict.candidates_to_pred_bboxes(
            candidates,
            self.input_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

    def fit_pred_bboxes_to_original(self, pred_bboxes, original_shape):
        """
        @param pred_bboxes:    Dim(-1, (x, y, w, h, class_id, probability))
        @param original_shape: (height, width, channels)
        """
        # pylint: disable=no-self-use
        return predict.fit_pred_bboxes_to_original(pred_bboxes, original_shape)

    def draw_bboxes(self, image, bboxes):
        """
        @parma image:  Dim(height, width, channel)
        @param bboxes: (candidates, 4) or (candidates, 5)
                [[center_x, center_y, w, h, class_id], ...]
                [[center_x, center_y, w, h, class_id, propability], ...]

        @return drawn_image

        Usage:
            image = yolo.draw_bboxes(image, bboxes)
        """
        return media.draw_bboxes(image, bboxes, self.classes)

    #############
    # Inference #
    #############

    def predict(
        self,
        frame: np.ndarray,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.25,
    ):
        # pylint: disable=unused-argument, no-self-use
        return [[0.0, 0.0, 0.0, 0.0, -1]]

    def inference(
        self,
        media_path,
        is_image: bool = True,
        cv_apiPreference=None,
        cv_frame_size: tuple = None,
        cv_fourcc: str = None,
        cv_waitKey_delay: int = 1,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.25,
    ):
        if not path.exists(media_path):
            raise FileNotFoundError("{} does not exist".format(media_path))

        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

        if is_image:
            frame = cv2.imread(media_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            bboxes = self.predict(
                frame,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            exec_time = time.time() - start_time
            print("time: {:.2f} ms".format(exec_time * 1000))

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = self.draw_bboxes(frame, bboxes)
            cv2.imshow("result", image)
        else:
            if cv_apiPreference is None:
                cap = cv2.VideoCapture(media_path)
            else:
                cap = cv2.VideoCapture(media_path, cv_apiPreference)

            if cv_frame_size is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv_frame_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv_frame_size[1])

            if cv_fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cv_fourcc))

            prev_time = time.time()
            if cap.isOpened():
                while True:
                    try:
                        is_success, frame = cap.read()
                    except cv2.error:
                        continue

                    if not is_success:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    predict_start_time = time.time()
                    bboxes = self.predict(
                        frame,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                    )
                    predict_exec_time = time.time() - predict_start_time

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    image = self.draw_bboxes(frame, bboxes)
                    curr_time = time.time()

                    cv2.putText(
                        image,
                        "preidct: {:.2f} ms, fps: {:.2f}".format(
                            predict_exec_time * 1000,
                            1 / (curr_time - prev_time),
                        ),
                        org=(5, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=(50, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    prev_time = curr_time

                    cv2.imshow("result", image)
                    if cv2.waitKey(cv_waitKey_delay) & 0xFF == ord("q"):
                        break

        print("YOLOv4: Inference is finished")
        while cv2.waitKey(10) & 0xFF != ord("q"):
            pass
        cv2.destroyWindow("result")
