# Xtreme-Vision

[![Build Status](https://camo.githubusercontent.com/6446a7907a4d4f8de024ec85750feb07d7914658/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f70617472656f6e2d646f6e6174652d79656c6c6f772e737667)](https://patreon.com/adeelintizar) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

![](assets/intro.gif)

`Go to PyPI page`> [Here](https://pypi.org/project/xtreme-vision/)

This is the Official Repository of Xtreme-Vision. Xtreme-Vision is a High Level Python Library which is built with simplicity in mind for Computer Vision Tasks, such as Object-Detection, Human-Pose-Estimation, Segmentation Tasks, it provides the support of a list of state-of-the-art algorithms, You can Start Detecting with Pretrained Weights as well as You can train the Models On Custom Dataset and with Xtreme-Vision you have the Power to detect/segment only the Objects of your interest

Currently, It Provides the Solution for the following Tasks:
   - Object Detection
   - Pose Estimation
   - Object Segmentation
   - Human Part Segmentation


For Detection with pre-trained models it provides:
  - RetinaNet
  - CenterNet
  - YOLOv4
  - TinyYOLOv4
  - Mask-RCNN
  - DeepLabv3+ (Ade20k)
  - CDCL (Cross Domain Complementary Learning)

For Custom Training It Provides:
  - YOLOv4
  - TinyYOLOv4
  - RetinaNet with (resnet50, resnet101, resnet152) 

![](assets/pose.gif)

>In Future it will provide solution for a wide variety of Computer-Vision Tasks such as Object-Detection, Pose-Estimation, Object Segmentation, Image-Prediction, Auto-Encoders and GANs with **2d and 3D Models** and it will support More State-Of-the-Art Algorithms.

>If You Like this Project Please do support it by donating here [![Build Status](https://camo.githubusercontent.com/6446a7907a4d4f8de024ec85750feb07d7914658/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f70617472656f6e2d646f6e6174652d79656c6c6f772e737667)](https://patreon.com/adeelintizar)


### Dependencies:
  - tensorflow >= 2.3.0
  - keras
  - opencv-python
  - numpy
  - pillow
  - matplotlib
  - pandas
  - scikit-learn
  - scikit-image
  - imgaug
  - labelme2coco
  - progressbar2
  - scipy
  - h5py
  - configobj


## **`Get Started:`**
```python
!pip install xtreme-vision
```
 >### `For More Tutorials of Xtreme-Vision, Click` [Here](https://github.com/Adeel-Intizar/Xtreme-Vision/tree/master/Tutorials)
# **`YOLOv4` Example** 


### **`Image Object Detection` Using `YOLOv4`** 

```python
from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_YOLOv4()
model.Detect_From_Image(input_path='kite.jpg',
                        output_path='./output.jpg')

from PIL import Image
Image.open('output.jpg')
```
