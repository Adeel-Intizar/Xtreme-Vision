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


from xtreme_vision.Detection.retinanet import RetinaNet
from xtreme_vision.Detection.yolov4 import YOLOv4, TinyYOLOv4
from xtreme_vision.Detection.centernet import ObjectDetection as CenterNet
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf

class Object_Detection:
    
    """
    This is Object Detection Class in Xtreme-Vision Library, it provides the support of State-Of-The-Art Models 
    like CenterNet, RetinaNet, YOLOv4 and TinyYOLOv4. After Instantiating this Class, you can set its
    properties and use pre-defined functions for performing Object Detection Tasks out of the box.


        Use_RetinaNet()                    |        # To Set which Model to Use
        Use_CenterNet()                    |        # To Set which Model to Use
        Use_YOLOv4()                       |        # To Set which Model to Use 
        Use_TinyYOLOv4()                   |        # To Set which Model to Use
        
        Detect_From_Image()                         # To Detect Objects from Images
        Detect_From_Video()                         # To Detect Objects from Videos
        Custom_Objects()                            # To set the desired objects to True e.g. Custom_Objects(car=True)
        Detect_Custom_Objects_From_Image()          # To Detect Custom Objects from Images
        Detect_Custom_Objects_From_Video()          # To Detect Custom Objects from Videos  
    """
    
    def __init__(self):
        self.model = None
        self.weights_path = ""
        self.modelLoaded = False
        self.modelType = ""
        
        self.iou = 0
        self.score = 0
        
        self.custom_objects = None
        
        self.input_path = ""
        self.output_path = ""
        
        # Variables for YOLO
        self.yolo_classes = ""
        self.input_shape = 0
        
    def Use_RetinaNet(self, weights_path:str = None, classes_path:str=None, backbone:str='resnet50'):
        
        """[This Function is Used to Set the Model Type to RetinaNet, Automatically downloads the model
        weights if weights_path is None and Loads the Model]

       Args:
          weights_path: path to the trained weights file
          classes_path: path to the txt file mapping classes to labels 
                    e.g. 
                        0 Cat
                        1 Dog
                        2 Monkey etc
                    If not Provided, Model will use 80 "Coco Dataset" classes by default
                    
          backbone: which backbone to use 
      
        Raises:
            RuntimeError: [If any other backbone is provided other than resnet50, resnet101 or resnet152]
            FileNotFoundError: [If Specified weights_path doesn't contain the weights file]
        """
        
        if 'resnet' not in backbone:
          raise RuntimeError ('Invalid BackBone: Valid BackBones are "resnet50"\t"resnet101"\t"resnet152".')
        
        if weights_path is None:
          path = 'xtreme_vision/weights/retinanet_weights.h5'
          if os.path.isfile(path):

            print('Found Existing Weights File...\nLoading Existing File...')
            self.weights_path = path
          else:

            if backbone == 'resnet50':
              print('-' * 20)
              print('Downloading Weights File...\nPlease Wait...')
              print('-' * 20)
              self.weights_path = tf.keras.utils.get_file('retinanet_weights.h5',
              						'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5',
                  					cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
          
            elif backbone == 'resnet101':
              print('-' * 20)
              print('Downloading Weights File...\nPlease Wait...')
              print('-' * 20)
              self.weights_path = tf.keras.utils.get_file('retinanet_weights.h5',
                                                        'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet101_oid_v1.0.0.h5',
                                                        cache_subdir = 'weights/', cache_dir = 'xtreme_vision')

            elif backbone == 'resnet152':
             print('-' * 20)
             print('Downloading Weights File...\nPlease Wait...')
             print('-' * 20)
             self.weights_path = tf.keras.utils.get_file('retinanet_weights.h5',
                                                        'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet152_oid_v1.0.0.h5',
                                                        cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
        else:
          if os.path.isfile(weights_path):
            self.weights_path = weights_path
          else:
            raise FileNotFoundError ("Weights File Doesn't Exist at Provided Path. Please Provide Valid Path.")
    
        self.classes = None
        if classes_path != None:
          self.classes = {}
          with open(classes_path, 'r') as f:
            for line in f:
              (key, val) = line.split()
              self.classes[int(key)] = val
        
        self.model = RetinaNet()
        self.model.load_model(self.weights_path, self.classes, backbone)
        self.modelLoaded = True
        self.modelType = 'retinanet'
    
    def Use_CenterNet(self, weights_path:str = None):
        
        """[This Function is used to set the Model Type to CenterNet, Automatically downloads the Model
        weights if weights_path is None and Loads the Model.]
      
        Args:
          weights_path: path to the trained weights_file

        Raises:
            FileNotFoundError: [If weights file doesn't exist at specified path]
        """
        
        if weights_path is None:
          path = 'xtreme_vision/weights/centernet_weights.h5'
          if os.path.isfile(path):
            print('Found Existing Weights File...\nLoading Existing File...')
            self.weights_path = path
          else:
            print('Downloading Weights File...\nPlease Wait...')
            self.weights_path = tf.keras.utils.get_file('centernet_weights.h5',
            'https://github.com/Licht-T/tf-centernet/releases/download/v1.0.6/centernet_pretrained_coco.h5',
            cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
        else:
          if os.path.isfile(weights_path):
            self.weights_path = weights_path
          else:
            raise FileNotFoundError ("Weights File Doesn't Exist at Provided Path. Please Provide Valid Path.")
        
        self.model = CenterNet()
        self.model.load_model(self.weights_path)
        self.modelLoaded = True
        self.modelType = 'centernet'
    
    def Use_YOLOv4(self, weights_path:str = None, classes_path:str = None, input_shape:int = 640,
                   iou = 0.45, score = 0.25):
        
        """[This Function is Used to set the Model Type to YOLOv4 and Loads the Model,
          Automatically downloads weights and classses files if weights_path or classes_path is None, 
          you can optionally set the input_shape as well. Dont change to values of iou and score, these
          values are recommended.]
        
        Args:
          weights_path: path to the trained weights file
          classes_path: path to the classes file with (.names) extension containg classes according to labels
                    e.g. classes.names
                    [
                      Cat
                      Dog
                      Monkey
                      etc
                    ]
                    If not Provided, Model will use 80 "Coco Classes" by default
                    
          input_shape: input_shape for the Model, it has to be multiple of 32

        Raises:
            FileNotFoundError: [If trained weights file doesn't exist at specified path]
            FileNotFoundError: [If classes_file doesn't exist at specified path]
        """
        
        if weights_path is None:
          path = 'xtreme_vision/weights/yolo.weights'
          if os.path.isfile(path):
            print('Found Existing weights file...\nLoading existing file...')
            self.weights_path = path
          else:
            print('Downloading weights file...\nPlease wait...')
            self.weights_path = tf.keras.utils.get_file('yolo.weights',
            'https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/yolov4.weights',
            cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
        else: 
          if os.path.isfile(weights_path):
            self.weights_path = weights_path
          else:
            raise FileNotFoundError ("Weights file doesn't exist at provided path. Please provide valid path.")

        if classes_path is None:
          path = 'xtreme_vision/weights/coco.names'
          if os.path.isfile(path):
            print('Found Existing Classes File...\nLoading Existing File...')
            self.yolo_classes = path
          else:
            print('Downloading Classes File...\nPlease wait...')
            self.yolo_classes = tf.keras.utils.get_file('coco.names',
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
            cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
        else:
          if os.path.isfile(classes_path):
            self.yolo_classes = classes_path
          else:
            raise FileNotFoundError ("Classes File Doesn't Exist at Provided Path. Please Provide Valid Path.")

        self.input_shape = input_shape
        self.iou = iou
        self.score = score
        
        self.model = YOLOv4()
        self.model.load_model(self.weights_path, self.yolo_classes, self.input_shape)
        self.modelLoaded = True
        self.modelType = 'yolo'
        
    def Use_TinyYOLOv4(self, weights_path:str = None, classes_path:str = None, input_shape:int = 960,
                       iou = 0.4, score = 0.1):
        
        """[This Function is Used to set the Model Type to TinyYOLOv4 and Loads the Model,
          Automatically downloads weights and classses files if weights_path or classes_path is None, 
          you can optionally set the input_shape as well. Dont change to values of iou and score, these
          values are recommended.]
        
        Args:
          weights_path: path to the trained weights file
          classes_path: path to the classes file with (.names) extension containg classes according to labels
                    e.g. classes.names
                    [
                      Cat
                      Dog
                      Monkey
                      etc
                    ]
                    If not Provided, Model will use 80 "Coco Classes" by default
                    
          input_shape: input_shape for the Model, it has to be multiple of 32

        Raises:
            FileNotFoundError: [If trained weights file doesn't exist at specified path]
            FileNotFoundError: [If classes_file doesn't exist at specified path]
        """
        
        if weights_path is None:
          path = 'xtreme_vision/weights/tinyyolo.weights'
          if os.path.isfile(path):
            print('Found Existing weights file...\nLoading existing file...')
            self.weights_path = path
          else:
            print('Downloading weights file...\nPlease wait...')
            self.weights_path = tf.keras.utils.get_file('tinyyolo.weights',
            'https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/yolov4-tiny.weights',
            cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
        else: 
          if os.path.isfile(weights_path):
            self.weights_path = weights_path
          else:
            raise FileNotFoundError ("Weights file doesn't exist at provided path. Please provide valid path.")

        if classes_path is None:
          path = 'xtreme_vision/weights/coco.names'
          if os.path.isfile(path):
            print('Found Existing Classes File...\nLoading Existing File...')
            self.yolo_classes = path
          else:
            print('Downloading Classes File...\nPlease wait...')
            self.yolo_classes = tf.keras.utils.get_file('coco.names',
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
            cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
        else:
          if os.path.isfile(classes_path):
            self.yolo_classes = classes_path
          else:
            raise FileNotFoundError ("Classes File Doesn't Exist at Provided Path. Please Provide Valid Path.")
        

        self.input_shape = input_shape
        self.iou = iou
        self.score = score
        
        self.model = TinyYOLOv4()
        self.model.load_model(self.weights_path, self.yolo_classes, self.input_shape)
        self.modelLoaded = True
        self.modelType = 'tinyyolo'
        
    def Detect_From_Image(self, input_path:str, output_path:str, extract_objects = False):
        
        """[This Function is used to Detect objects from Images]
      
        Args:
          input_path: path to the input image with jpg/jpeg/png extension
          output_path: path to save the output image with jpg/jpeg/png extension
          extract_objects: set it to True, if you want to extract detected objects, Only RetinaNet supports it

        Raises:
            RuntimeError: [If input_path or output_path is not specified]
            RuntimeError: [If Model is not Loaded before using this function]
        """
        
        if (input_path is None) or (output_path is None):
            raise RuntimeError ('Image_Path AND Output_Path Should Not Be None.')
            
        elif self.modelLoaded != True:
            raise RuntimeError ('First You have to specify which model you want to use.')
        
        else:
            
            self.input_path = input_path
            self.output_path = output_path
            
            if type(input_path) == 'str':
                img = np.array(Image.open(self.input_path))[..., ::-1]
            elif type(input_path) == np.ndarray:
                img = input_path
            else:
                raise RuntimeError("Invalid Input Image")
            
            if self.modelType == 'retinanet':
                _ = self.model.predict(img, self.output_path, debug = True, extract_objects = extract_objects)
                
            elif self.modelType == 'centernet':
                _ = self.model.predict(img, self.output_path, debug = True)
                
            elif self.modelType == 'yolo':
                _ = self.model.predict(img, self.output_path, debug = True, iou = self.iou, score = self.score)
                
            elif self.modelType == 'tinyyolo':
                _ = self.model.predict(img, self.output_path, debug = True, iou = self.iou, score = self.score)
                
                
    def Detect_From_Video(self, input_path:str, output_path:str, extract_objects = False, fps:int = 25):
        
        """[This Function is used to Detect objects from Videos.]
      
        Args:
          input_path: path to the input video with mp4/avi extension
          output_path: path to save the output video with mp4/avi extension
          extract_objects: set it to True, if you want to extract detected objects, Only RetinaNet supports it
          fps: frames per second for video processing

        Raises:
            RuntimeError: [If input_path or output_path is not specified]
            RuntimeError: [If Model is not Loaded before using this Function]
            RuntimeError: [If any other model type is specified other than RetinaNet/CenterNet/YOLOv4/TinyYOLOv4]
        """
        
        if (output_path is None) or (input_path is None):
            raise RuntimeError ('Output_Path should not be None, & One of Camera_input OR Input_Path must be specified.')
            
        elif self.modelLoaded != True:
            raise RuntimeError ('First You have to specify which model you want to use.')
        
        self.image_path = input_path
        self.output_path = output_path
        out = None
        
        cap = cv2.VideoCapture(self.image_path)
            
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'\nThere are {length} Frames in this video')
        print('-' * 20)
        print('Detecting Objects in the Video... Please Wait...')
        print('-' * 20)

        while(cap.isOpened()):
            retreive, frame = cap.read()
            if not retreive:
                break
            
            frame = np.array(frame)[..., ::-1]
            
            if self.modelType == 'retinanet':
                im = self.model.predict(frame, './', debug = False, extract_objects = extract_objects)
                
            elif self.modelType == 'yolo':
                im = self.model.predict(frame, './', debug = False, iou = self.iou, score = self.score)
                
            elif self.modelType == 'tinyyolo':
                im = self.model.predict(frame, './', debug = False, iou = self.iou, score = self.score)

            elif self.modelType == 'centernet':
                im = self.model.predict(frame, './', debug = False)
                im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            else:
              raise RuntimeError ('Invalid Model Type, For Video_ObjectDetection you can use \n "RetinaNet" \t "CenterNet" \t "YOLOv4" \t "TinyYOLOv4"')
            
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            
            out.write(im)
        print('Done. Processing has been Finished... Please Check Output Video.')
        out.release()
        cap.release()
        
        
    def Custom_Objects(self, person=False, bicycle=False, car=False, motorcycle=False, airplane=False,
                      bus=False, train=False, truck=False, boat=False, traffic_light=False, fire_hydrant=False,
                      stop_sign=False,
                      parking_meter=False, bench=False, bird=False, cat=False, dog=False, horse=False, sheep=False,
                      cow=False, elephant=False, bear=False, zebra=False,
                      giraffe=False, backpack=False, umbrella=False, handbag=False, tie=False, suitcase=False,
                      frisbee=False, skis=False, snowboard=False,
                      sports_ball=False, kite=False, baseball_bat=False, baseball_glove=False, skateboard=False,
                      surfboard=False, tennis_racket=False,
                      bottle=False, wine_glass=False, cup=False, fork=False, knife=False, spoon=False, bowl=False,
                      banana=False, apple=False, sandwich=False, orange=False,
                      broccoli=False, carrot=False, hot_dog=False, pizza=False, donut=False, cake=False, chair=False,
                      couch=False, potted_plant=False, bed=False,
                      dining_table=False, toilet=False, tv=False, laptop=False, mouse=False, remote=False,
                      keyboard=False, cell_phone=False, microwave=False,
                      oven=False, toaster=False, sink=False, refrigerator=False, book=False, clock=False, vase=False,
                      scissors=False, teddy_bear=False, hair_dryer=False,
                      toothbrush=False):

        """
                         The 'CustomObjects()' function allows you to handpick the type of objects you want to detect
                         from an image or video. The objects are pre-initiated in the function variables and predefined as 'False',
                         which you can easily set to true for any number of objects available.  This function
                         returns a dictionary which must be parsed into the 'Detect_Custom_Objects_From_Image()' and 
                         'Detect_Custom_Objects_From_Video()'. 
                         Detecting custom objects only happens when you call the function 'Detect_Custom_Objects_From_Image()'
                         or 'Detect_Custom_Objects_From_Video()'


                        * true_values_of_objects (array); Acceptable values are 'True' and False  for all object values present

                        :param boolean_values:
                        :return: custom_objects_dict
                """

        custom_objects_dict = {}
        input_values = [person, bicycle, car, motorcycle, airplane,
                        bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
                        parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
                        giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
                        sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
                        bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
                        broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed,
                        dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
                        oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
                        toothbrush]
        actual_labels = ["person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

        for input_value, actual_label in zip(input_values, actual_labels):
            if (input_value == True):
                custom_objects_dict[actual_label] = "valid"
            else:
                custom_objects_dict[actual_label] = "invalid"

        return custom_objects_dict
    
    
    def Detect_Custom_Objects_From_Image(self, custom_objects = None, input_path:str = None, output_path:str = None,
                                         minimum_percentage_probability:float = 0.5, extract_objects = False):
        
        """[This function is used to detect custom objects from Images, it will detect only those objects
        which are set to True in dictionary retruned from Custom_Objects() function]
      
        Args:
          custom_objects: dictionary returned from Custom_Objects()
          input_path: path to the input image with jpg/jpeg/png extension
          output_path: path to save the output image with jpg/jpeg/png extension
          minimum_percentage_probability: Anything detected with confidence less than its value will not be shown, Only RetinaNet supports it
          extract_objects: set it to True, if you want to extract detected objects, Only RetinaNet supports it

        Raises:
            RuntimeError: [If custom_objects/input_path/output_path is not specified]
            RuntimeError: [If any other model type is specified other than RetinaNet/CenterNet/YOLOv4/TinyYOLOv4]
        """
        
        
        if (custom_objects is None) or (input_path is None) or (output_path is None):
            raise RuntimeError ('Custom_Objects, Input_Path and Output_path should not be None.')
      
        else:
            self.image_path = input_path
            self.output_path = output_path
            self.custom_objects = custom_objects
            self.min_prob = minimum_percentage_probability
            
            img = np.array(Image.open(self.image_path))[..., ::-1]
            
            if self.modelLoaded:
                
                if (self.modelType == 'retinanet'):
                    
                    _ = self.model.predict(img, self.output_path, debug = True, custom_objects= self.custom_objects,
                                           min_prob = self.min_prob, extract_objects = extract_objects)

                elif (self.modelType == 'centernet'):
                    _ = self.model.predict(img, self.output_path, debug = True, custom_objects= self.custom_objects)

                elif (self.modelType == 'yolo'):
                    _ = self.model.predict(img, self.output_path, custom_objects= self.custom_objects)
                
                elif (self.modelType == 'tinyyolo'):
                    _ = self.model.predict(img, self.output_path, custom_objects= self.custom_objects)
                
                
                else:
                    raise RuntimeError ('Invalid Model Type: Supported Models are: Retinanet\tCenterNet\tYOLOv4\tTinyYOLOv4.')

    def Detect_Custom_Objects_From_Video(self, custom_objects = None, input_path:str = None, output_path:str = None,
                                         minimum_percentage_probability:float = 0.25, extract_objects = False, fps:int = 25):
        
        """[This function is used to detect custom objects from Videos, it will detect only those objects
        which are set to True in dictionary retruned from Custom_Objects() function]
      
        Args:
          custom_objects: dictionary returned from Custom_Objects()
          input_path: path to the Video image with mp4/avi extension
          output_path: path to save the output Video with mp4/avi extension
          minimum_percentage_probability: Anything detected with confidence less than its value will not be shown, Only RetinaNet supports it
          extract_objects: set it to True, if you want to extract detected objects, Only RetinaNet supports it
          fps: frames per second for video processing

        Raises:
            RuntimeError: [If custom_objects/input_path/output_path is not specified]
            RuntimeError: [If any other model type is specified other than RetinaNet/CenterNet/YOLOv4/TinyYOLOv4]
        """
        
        if (custom_objects is None) or (input_path is None) or (output_path is None):
            raise RuntimeError ('Custom_Objects, Input_Path and Output_path should not be None.')
        
        self.custom_objects = custom_objects
        self.image_path = input_path
        self.output_path = output_path
        self.min_prob = minimum_percentage_probability
        out = None
        
        cap = cv2.VideoCapture(self.image_path)
            
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'\nThere are {length} Frames in this video')
        print('-' * 20)
        print('Detecting Objects in the Video... Please Wait...')
        print('-' * 20)

        while(cap.isOpened()):
            retreive, frame = cap.read()
            if not retreive:
                break
            
            frame = np.array(frame)[..., ::-1]
            
            if self.modelType == 'retinanet':
                
                im = self.model.predict(frame, './', debug = False, custom_objects= self.custom_objects,
                                        min_prob = self.min_prob, extract_objects = extract_objects)
                
            elif self.modelType == 'centernet':
                im = self.model.predict(frame, './', debug = False, custom_objects= self.custom_objects)
                im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            elif self.modelType == 'yolo':
                im = self.model.predict(frame, './', debug = False, custom_objects = self.custom_objects)
                
            elif self.modelType == 'tinyyolo':
                im = self.model.predict(frame, './', debug = False, custom_objects = self.custom_objects)
            
            else:
                raise RuntimeError ('Invalid Model Type: Supported Models are: Retinanet\tCenterNet\tYOLOv4\tTinyYOLOv4.')

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            
            out.write(im)
        print('Done. Processing has been Finished... Please Check Output Video.')
        out.release()
        cap.release()
