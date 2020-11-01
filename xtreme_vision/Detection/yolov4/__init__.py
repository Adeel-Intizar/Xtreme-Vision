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


from xtreme_vision.Detection.yolov4.tf import YOLOv4 as yolo_main
import numpy as np
import cv2

labels = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
            9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
            17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
            26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
            34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
            41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class YOLOv4:
    def __init__(self):
        self.weights_path = ""
        self.model = None
        self.yolo_classes = ""
        self.iou = 0
        self.score = 0
        self.input_shape = 0
        self.output_path = ""
    
    def load_model(self, weights_path:str = None, classes_path:str = None, input_shape:int = 608):
        if (weights_path  is None) or (classes_path is None):
            raise RuntimeError ('weights_path AND classes_path should not be None.')
        
        self.yolo_classes = classes_path
        self.weights_path = weights_path
        self.input_shape = input_shape
        
        self.model = yolo_main(shape = self.input_shape)
        self.model.classes = self.yolo_classes
        self.model.make_model()
        self.model.load_weights(self.weights_path, weights_type = 'yolo')
        
    def predict(self, img:np.ndarray, output_path:str, iou = 0.45, score = 0.25, custom_objects:dict = None, 
                debug=True):

        self.output_path = output_path
        self.iou = iou
        self.score = score  
        #img = np.array(Image.open(img))[..., ::-1] 
        pred_bboxes = self.model.predict(img, iou_threshold = self.iou, score_threshold = self.score)
        boxes = []
        if (custom_objects != None):
          for i in range(len(pred_bboxes)):
            check_name = labels[pred_bboxes[i][4]]
            check = custom_objects.get(check_name, 'invalid')
            if check == 'invalid':
              continue
            elif check == 'valid':
              boxes.append(list(pred_bboxes[i]))
          boxes = np.array(boxes)
          res = self.model.draw_bboxes(img, boxes)
          if debug:
            cv2.imwrite(self.output_path, res)

        else:
          res = self.model.draw_bboxes(img, pred_bboxes)
          if debug:
              cv2.imwrite(self.output_path, res)
        
        return res
  
class TinyYOLOv4:
    def __init__(self):
        self.weights_path = ""
        self.model = None
        self.yolo_classes = ""
        self.iou = 0
        self.score = 0
        self.input_shape = 0
        self.output_path = ""
    
    def load_model(self, weights_path:str = None, classes_path:str = None, input_shape:int = 0):
        if (weights_path  is None) or (classes_path is None):
            raise RuntimeError ('weights_path AND classes_path should not be None.')
        
        self.yolo_classes = classes_path
        self.weights_path = weights_path
        self.input_shape = input_shape
        self.model = yolo_main(tiny = True, shape = self.input_shape)
        self.model.classes = self.yolo_classes
        self.model.make_model()
        self.model.load_weights(self.weights_path, weights_type = 'yolo')
        
    def predict(self, img:np.ndarray, output_path:str, iou = 0.4, score = 0.07, custom_objects:dict = None,
                debug=True):

        self.output_path = output_path
        self.iou = iou
        self.score = score  
        #img = np.array(Image.open(img))[..., ::-1] 
        pred_bboxes = self.model.predict(img, iou_threshold = self.iou, score_threshold = self.score)
        
        boxes = []
        if (custom_objects != None):
          for i in range(len(pred_bboxes)):
            check_name = labels[pred_bboxes[i][4]]
            check = custom_objects.get(check_name, 'invalid')
            if check == 'invalid':
              continue
            elif check == 'valid':
              boxes.append(list(pred_bboxes[i]))
          boxes = np.array(boxes)
          res = self.model.draw_bboxes(img, boxes)
          if debug:
            cv2.imwrite(self.output_path, res)

        else:
          res = self.model.draw_bboxes(img, pred_bboxes)
          if debug:
              cv2.imwrite(self.output_path, res)
        
        return res
        