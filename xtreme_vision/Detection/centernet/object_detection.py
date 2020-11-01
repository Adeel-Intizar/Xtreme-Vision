"""
MIT License

Copyright (c) 2020 Licht Takeuchi

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
import tensorflow as tf
import numpy as np
import PIL.Image
import cv2

from xtreme_vision.Detection.centernet.model.object_detection import ObjectDetectionModel
from xtreme_vision.Detection.centernet import util
from xtreme_vision.Detection.centernet.version import VERSION


class ObjectDetection:
    def __init__(self, num_classes: int = 80):
        self.mean = np.array([[[0.408, 0.447, 0.470]]], dtype=np.float32)
        self.std = np.array([[[0.289, 0.274, 0.278]]], dtype=np.float32)
        self.k = 100
        self.score_threshold = 0.3
        self.input_size = 512
        self.output_path = ""

        self.num_classes = num_classes

        self.model = None

        self.init_model()

        self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
                   9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
                   17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
                   26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                   41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                   50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                   59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                   68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                   77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    def init_model(self):
        self.model = ObjectDetectionModel(self.num_classes)
        self.model(tf.keras.Input((self.input_size, self.input_size, 3)))

    def load_model(self, weights_path: str = None):
        if weights_path is None:
            base_url = f'https://github.com/Licht-T/tf-centernet/releases/download/{VERSION}'
            if self.num_classes == 80:
                weights_path = tf.keras.utils.get_file(
                    f'centernet_pretrained_coco_{VERSION}.h5',
                    f'{base_url}/centernet_pretrained_coco.h5',
                    cache_subdir='tf-centernet'
                )
            else:
                raise RuntimeError('weights_path should not be None.')

        self.model.load_weights(weights_path)

    def predict(self, img: np.ndarray, output_path:str = None, debug=True, custom_objects:dict = None):
        if (img is None) or (output_path is None):
          raise RuntimeError ('Image_Path AND Output_path should not be None.')
       # img = np.array(PIL.Image.open(img))[..., ::-1]
       
        orig_wh = np.array(img.shape[:2])[::-1]
        resize_factor = self.input_size / orig_wh.max()
        centering = (self.input_size - orig_wh * resize_factor) / 2

        input_img = tf.image.resize_with_pad(img, self.input_size, self.input_size)
        input_img = (tf.dtypes.cast(input_img, tf.float32) / tf.constant(255, tf.float32) - self.mean) / self.std
        input_img = input_img[tf.newaxis, ...]

        predicted, _ = self.model(input_img)

        heatmap, offsets, whs = predicted

        heatmap = util.image.heatmap_non_max_surpression(heatmap)

        heatmap = np.squeeze(heatmap.numpy())
        offsets = np.squeeze(offsets.numpy())
        whs = np.squeeze(whs.numpy())

        idx = heatmap.flatten().argsort()[::-1][:self.k]
        scores = heatmap.flatten()[idx]
        idx = idx[scores > self.score_threshold]
        scores = scores[scores > self.score_threshold]

        rows, cols, classes = np.unravel_index(idx, heatmap.shape)

        xys = np.concatenate([cols[..., np.newaxis], rows[..., np.newaxis]], axis=-1) + offsets[rows, cols]
        boxes = np.concatenate([xys - whs[rows, cols]/2, xys + whs[rows, cols]/2], axis=1).reshape((-1, 2, 2))

        boxes = ((self.input_size / heatmap.shape[0]) * boxes - centering) / resize_factor
        boxes = boxes.reshape((-1, 4))

        names = []

        for i in classes:
          names.append(self.classes[i])
          
        self.output_path = output_path

        im = PIL.Image.fromarray(img[..., ::-1])
        im = util.image.draw_bounding_boxes(im, boxes, names, scores, custom_objects=custom_objects)
        if debug:
            im.save(self.output_path)
            
        return im
