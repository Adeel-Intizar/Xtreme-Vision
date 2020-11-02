"""
MIT License

Copyright (c) 2020 Adeel <kingadeel2017@outlook.com>

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



import cv2
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from xtreme_vision.Segmentation.maskrcnn import models
from xtreme_vision.Segmentation.maskrcnn.utils.visualization import draw_mask
from xtreme_vision.Detection.retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from xtreme_vision.Detection.retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from xtreme_vision.Detection.retinanet.utils.colors import label_color


class MaskRCNN:
    def __init__(self):
        self.model = None
        self.weights_path = ""
        self.output_path = ""
        self.min_prob = 0
        
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
        
    def load_model(self, Weights_Path:str = None):
        if Weights_Path is None:
            raise RuntimeError ('Weights_Path should not be None.')
        self.weights_path = Weights_Path
        self.model = models.load_model(self.weights_path, backbone_name = 'resnet50')
        
    def predict(self, img_path:np.ndarray = None, output_path:str = None, debug=True, custom_objects = None,
                min_prob:float = 0.25, show_names:bool=True):
        
        if (img_path is None) or (output_path is None):
            raise RuntimeError ('img_path & Outpu_path should not be None.')
        
        image = img_path
        self.output_path = output_path
        self.min_prob = min_prob
        self.custom_objects = custom_objects

        draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        copy = draw.copy()
        image = preprocess_image(image)
        image, scale = resize_image(image)
        outputs = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        
        boxes  = outputs[-4][0]
        scores = outputs[-3][0]
        labels = outputs[-2][0]
        masks  = outputs[-1][0]

        boxes /= scale


        for index, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
  
            if score < self.min_prob:
              continue

            elif (custom_objects != None):
              check_name = self.classes.get(label, 'invalid')
              check = custom_objects.get(check_name, 'invalid')
              if (check == "invalid"):
                continue
                
            index -= 1
            color = label_color(label+index)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            mask = mask[:, :, label]
            draw_mask(draw, b, mask, color=color)
            
            if show_names:
              caption = "{} {:.0f}%".format(self.classes[label], (score * 100))
              draw_caption(draw, b, caption)
        
        detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        
        if debug:
            cv2.imwrite(self.output_path, detected_img)
        
        return detected_img


