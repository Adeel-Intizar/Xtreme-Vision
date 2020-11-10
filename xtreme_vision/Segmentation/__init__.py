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


from xtreme_vision.Segmentation.deeplab.semantic import semantic_segmentation
from xtreme_vision.Segmentation.maskrcnn import MaskRCNN
from xtreme_vision.Segmentation.maskrcnn import MaskRCNN
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf


class Segmentation:

    """
    This is Segmentation Class in Xtreme-Vision Library, it provides the support of State-Of-The-Art Models 
    like Mask-RCNN and DeepLabv3+. After Instantiating this Class, you can set its properties and use pre-defined
    functions for performing segmentation Tasks out of the box.

    Note: Custom Segmenation only Supports Mask-RCNN

        Use_MaskRCNN() or Use_DeepLabv3()           # To Specify which Model to Use
        Detect_From_Image()                         # To Segment from Images
        Detect_From_Video()                         # To Segment from Videos
        Custom_Objects()                            # To set the desired objects to True e.g. Custom_Objects(car=True)
        Detect_Custom_Objects_From_Image()          # To Segment Custom Objects from Images
        Detect_Custom_Objects_From_Video()          # To Segment Custom Objects from Videos  
    """

    def __init__(self):
        self.model = None
        self.weights_path = ""
        self.modelLoaded = False
        self.modelType = ""
        self.custom_objects = None

        self.input_path = ""
        self.output_path = ""

    def Use_MaskRCNN(self, weights_path: str = None):
        """[This Function is used to set the Model Type to Mask-RCNN, Automatically downloads the weights
        if set to None and Loads the Model]

        Args:
            weights_path (str, optional): [path to the trained weights file]. Defaults to None.

        Raises:
            FileNotFoundError: [If weights file doesn't exist at specified path]
        """

        if weights_path is None:
            path = 'xtreme_vision/weights/maskrcnn_weights.h5'
            if os.path.isfile(path):
                print('Found Existing Weights File...\nLoading Existing File...')
                self.weights_path = path
            else:
                print('Downloading Weights File...\nPlease Wait...')
                self.weights_path = tf.keras.utils.get_file('maskrcnn_weights.h5',
                                                            'https://github.com/fizyr/keras-maskrcnn/releases/download/0.2.2/resnet50_coco_v0.2.0.h5',
                                                            cache_subdir='weights/', cache_dir='xtreme_vision')
        else:
            if os.path.isfile(weights_path):
                self.weights_path = weights_path
            else:
                raise FileNotFoundError(
                    "Weights File Doesn't Exist at Provided Path. Please Provide Valid Path.")

        self.model = MaskRCNN()
        self.model.load_model(self.weights_path)
        self.modelLoaded = True
        self.modelType = 'maskrcnn'

    def Use_DeepLabv3(self, weights_path: str = None):
        """[This function is used to set the Model Type to DeepLabv3, Automatically downloads the weights
        if set to None and Loads the Model]

        Args:
            weights_path (str, optional): [path to the trained weights file]. Defaults to None.

        Raises:
            FileNotFoundError: [If weights file doesn't exist at specified path]
        """
        
        if weights_path is None:
            path = 'xtreme_vision/weights/deeplab_weights.h5'
            if os.path.isfile(path):
                print('Found Existing Weights File...\nLoading Existing File...')
                self.weights_path = path
            else:
                print('Downloading Weights File...\nPlease Wait...')
                self.weights_path = tf.keras.utils.get_file('deeplab_weights.h5',
                                                            'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.3/deeplabv3_xception65_ade20k.h5',
                                                            cache_subdir='weights/', cache_dir='xtreme_vision')
        else:
            if os.path.isfile(weights_path):
                self.weights_path = weights_path
            else:
                raise FileNotFoundError(
                    "Weights File Doesn't Exist at Provided Path. Please Provide Valid Path.")

        self.model = semantic_segmentation()
        self.model.load_ade20k_model(self.weights_path)
        self.modelLoaded = True
        self.modelType = 'deeplab'

    def Detect_From_Image(self, input_path: str, output_path: str, min_prob: float = 0.25, show_names: bool = False):
        
        """[This function is used to segment objects from Images]

        Args:
            input_path (str): [path to the input image with jpg/jpeg/png extension]
            output_path (str): [path to save the output image with jpg/jpeg/png extension]
            min_prob (float, optional): [anything detected with confidence less than this, will not be shown. Only Mask-RCNN supports it]. Defaults to 0.25.
            show_names (bool, optional): [wether to show the names of detected objects, Only Mask-RCNN supports it]. Defaults to False.

        Raises:
            RuntimeError: [If Model is not Loaded before Using this Function]
            RuntimeError: [If any other Model type is specified other than Mask-RCNN or DeepLabv3]
        """

        if self.modelLoaded != True:
            raise RuntimeError(
                'Before calling this function, you have to call Use_MaskRCNN().')

        else:

            self.input_path = input_path
            self.output_path = output_path
            self.min_prob = min_prob
            self.show_names = show_names

            img = np.array(Image.open(self.input_path))[..., ::-1]

            if self.modelType == 'maskrcnn':
                _ = self.model.predict(
                    img, self.output_path, min_prob=self.min_prob, show_names=self.show_names)

            elif self.modelType == 'deeplab':
                raw_labels, img = self.model.segmentAsAde20k(
                    self.input_path, self.output_path, overlay=True)

            else:
                raise RuntimeError(
                    'Invalid ModelType: Valid Types are "MaskRCNN"\t"DeepLab".')

    def Detect_From_Video(self, input_path: str, output_path: str, min_prob: float = 0.25, show_names: bool = False):

        """[This function is used to segment objects from Videos]

        Args:
            input_path (str): [path to the input video with mp4/avi extension]
            output_path (str): [path to save the output video with mp4/avi extension]
            min_prob (float, optional): [anything detected with confidence less than this, will not be shown. Only Mask-RCNN supports it]. Defaults to 0.25.
            show_names (bool, optional): [wether to show the names of detected objects, Only Mask-RCNN supports it]. Defaults to False.

        Raises:
            RuntimeError: [If Model is not Loaded before Using this Function]
            RuntimeError: [If any other Model type is specified other than Mask-RCNN or DeepLabv3]
        """

        if self.modelLoaded != True:
            raise RuntimeError(
                'Before calling this function, you have to call Use_MaskRCNN().')

        self.input_path = input_path
        self.output_path = output_path
        self.min_prob = min_prob
        self.show_names = show_names
        out = None

        cap = cv2.VideoCapture(self.input_path)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'\nThere are {length} Frames in this video')
        print('-' * 20)
        print('Detecting Objects in the Video... Please Wait...')
        print('-' * 20)

        while(cap.isOpened()):
            retreive, frame = cap.read()
            if not retreive:
                break

            fr = np.array(frame)[..., ::-1]

            if self.modelType == 'maskrcnn':
                im = self.model.predict(
                    fr, './', False, min_prob=self.min_prob, show_names=self.show_names)

            elif self.modelType == 'deeplab':
                _, im = self.model.segmentFrameAsAde20k(frame, overlay=True)

            else:
                raise RuntimeError(
                    'Invalid ModelType: Valid Types are  "MaskRCNN"\t"DeepLab".')

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(
                    self.output_path, fourcc, 30, (frame.shape[1], frame.shape[0]))

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
                         from an image. The objects are pre-initiated in the function variables and predefined as 'False',
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

    def Detect_Custom_Objects_From_Image(self, custom_objects=None, input_path: str = None, output_path: str = None,
                                         min_prob: float = 0.25, show_names: bool = False):
        
        """[This function is used to detect custom objects from Images, it will only detect those objects which
        are set to True in dictionary returned from Custom_Objects() function.]
        
        Args:
            custom_objects: (dict) [dictionary returned from Custom_Objects() function]
            input_path: (str) [path to the input Image with jpg/jpeg/png extension]
            output_path: (str) [path to save the output image with jpg/jpeg/png extension]
            min_prob: (float) [anything detected with confidence less than this will not be shown]
            show_names: (bool) [wether to show the names of detected objects]
 
        Raises:
            RuntimeError: [If custom_objects/input_path/output_path is not specified]
            RuntimeError: [If Model is not Loaded before calling this function]
            RuntimeError: [If any other Model Type is Specified other than Mask-RCNN]
        """

        if (custom_objects is None) or (input_path is None) or (output_path is None):
            raise RuntimeError(
                'Custom_Objects, Input_Path and Output_path should not be None.')

        else:
            self.input_path = input_path
            self.output_path = output_path
            self.custom_objects = custom_objects
            self.min_prob = min_prob
            self.show_names = show_names

            img = np.array(Image.open(self.input_path))[..., ::-1]

            if self.modelLoaded:

                if (self.modelType == 'maskrcnn'):

                    _ = self.model.predict(img, self.output_path, custom_objects=self.custom_objects,
                                           min_prob=self.min_prob, show_names=self.show_names)
                else:
                    raise RuntimeError(
                        'Invalid ModelType: Valid Type is "MaskRCNN".')

            else:
                raise RuntimeError(
                    'Before calling this function, you have to call Use_MaskRCNN().')

    def Detect_Custom_Objects_From_Video(self, custom_objects=None, input_path: str = None, output_path: str = None,
                                         min_prob: float = 0.25, show_names: bool = False):
       
        """[This function is used to detect custom objects from Videos, it will only detect those objects which
        are set to True in dictionary returned from Custom_Objects() function.]
        
        Args:
            custom_objects: (dict) [dictionary returned from Custom_Objects() function]
            input_path: (str) [path to the input Video with mp4/avi extension]
            output_path: (str) [path to save the output Video with mp4/avi extension]
            min_prob: (float) [anything detected with confidence less than this will not be shown]
            show_names: (bool) [wether to show the names of detected objects]
 
        Raises:
            RuntimeError: [If custom_objects/input_path/output_path is not specified]
            RuntimeError: [If Model is not Loaded before calling this function]
            RuntimeError: [If any other Model Type is Specified other than Mask-RCNN]
        """

        if (custom_objects is None) or (input_path is None) or (output_path is None):
            raise RuntimeError(
                'Custom_Objects, Input_Path and Output_path should not be None.')

        if self.modelLoaded != True:
            raise RuntimeError(
                'Before calling this function, you have to call Use_MaskRCNN().')

        self.custom_objects = custom_objects
        self.input_path = input_path
        self.output_path = output_path
        self.min_prob = min_prob
        self.show_names = show_names
        out = None

        cap = cv2.VideoCapture(self.input_path)

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

            if self.modelType == 'maskrcnn':

                im = self.model.predict(frame, './', False, custom_objects=self.custom_objects,
                                        min_prob=self.min_prob, show_names=self.show_names)
            else:
                raise RuntimeError(
                    'Invalid ModelType: Valid Type is "MaskRCNN".')

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(
                    self.output_path, fourcc, 30, (frame.shape[1], frame.shape[0]))

            out.write(im)
        print('Done. Processing has been Finished... Please Check Output Video.')
        out.release()
        cap.release()
