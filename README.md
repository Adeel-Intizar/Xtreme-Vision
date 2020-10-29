# Xtreme-Vision

[![Build Status](https://camo.githubusercontent.com/6446a7907a4d4f8de024ec85750feb07d7914658/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f70617472656f6e2d646f6e6174652d79656c6c6f772e737667)](https://patreon.com/adeelintizar) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

![](assets/output.png)

Xtreme-Vision is a Python Library which is built with simplicity in mind for Computer Vision Tasks, Currently it provides the solution for only Object-Detection Tasks, it provides the support of a list of state-of-the-art algorithms for Object Detection, Video Object Detection and Training on Custom Datasets. Currently it supports 4 different algorithms. For Detection with pre-trained models it provides:

  - RetinaNet
  - CenterNet
  - YOLOv4
  - TinyYOLOv4

For Custom Training It Provides:
  - YOLOv4
  - TinyYOLOv4

>In Future it will not be limited to just Object-Detection, it will provide solution for a wide variety of Computer-Vision Tasks such as Image-Segmentation, Image-Prediction, Auto-Encoders and GANs.

>If You Like this Project Please do support it by donating [here](https://patreon.com/adeelintizar)


### Dependencies:
  - Tensorflow >= 2.3.0
  - Keras
  - Opencv-python
  - Numpy
  - Pillow
  - Matplotlib
  - Pandas
  - Scikit-learn
  - Progressbar2
  - Scipy
  - H5Py

### Installation:
To Install Xtreme-Vision, run the following command in command line
 ```bash
pip install xtreme-vision
```
### Get Started:

You can use any of provided Models by calling the following functions:
  - Use_RetinaNet()
  - Use_CenterNet()
  - Use_YOLOv4()
  - Use_TinyYOLOv4()

And you can Perform Tasks by calling the following functions:
  - Detect_From_Image()
  - Detect_From_Video()
  - Detect_Custom_Objects_From_Image()
  - Detect_Custom_Objects_From_Video()

**Note**:
>Detect_From_Image() Supports RetinaNet, YOLOv4, TinyYOLOv4, CenterNet
Detect_From_Video() Supports RetinaNet, YOLOv4, TinyYOLOv4
Detect_Custom_Objects_From_Image() Supports RetinaNet
Detect_Custom_Objects_From_Video() Supports RetinaNet
 
 Use the Following Code to start detecting objects in Images.
 ```bash
from xtreme_vision.Detection import Object_Detection
from PIL import Image

model = Object_Detection()
model.Use_CenterNet(weights_path = None)
model.Detect_From_Image(input_path='', output_path='out.jpg', extract_objects=False)
Image.open('out.jpg')
```

For Detecting Custom Objects in Images, Run the following Code.
For Detecting Custom Objects in Videos, replace with **Detect_Custom_Objects_From_Video()**

 ```bash
model = Object_Detection()
model.Use_RetinaNet(weights_path = None)
custom_objects = model.Custom_Objects(Car=False, person=True)

model.Detect_Custom_Objects_From_Image(custom_objects, 
                                       input_path = 'image.jpg',
                                       output_path = './output.jpg', 
                                       minimum_percentage_probability = 0.2,
                                       extract_objects=False)

Image.open('output.jpg')
```

Use the Following Code to detect Objects in Videos.
 ```bash
from xtreme_vision.Detection import Object_Detection
from PIL import Image

model = Object_Detection()
model.Use_TinyYOLOv4(weights_path = None)
model.Detect_From_Video(input_path= 'video.mp4', 
                        output_path= './tinyyolo.mp4',
                        extract_objects=True)
```
![](assets/out.gif)

For Training on Custom Dataset, Dataset should be in following format
annotations.txt
```
<img.jpg> class_label,x,y,width,height
```
class_labels: **should be greater than or equal to 0**
x, y, width, height: **should be between 0 and 1**

img_dir
```
path to the folder containing images
```
For Training on Custom Dataset, Use the Following Code
to use tinyyolo replace with **Use_TinyYOLOv4()**
```
from xtreme_vision.Detection.Custom import Train_Custom_Detector

clf = Train_Custom_Detector()
clf.Use_YOLOv4('classes.names')
clf.load_data('training.txt', './', 'validation.txt', './')
clf.train(epochs = 10, lr = 0.001)
```

Then Use the following code to load the trained model
```
from xtreme_vision.Detection import Object_Detection
model = Object_Detection()
model.Use_YOLOv4(classes_path = 'classes.names',
                weights_path = 'trainedmodel.weights'
                )
model.Detect_From_Image(input_path = 'input.jpg',
                        output_path = 'output.jpg',
                        extract_objects = True
                        )
```
