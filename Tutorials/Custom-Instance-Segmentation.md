# **`Custom Instance Segmentation From Images & Videos`**
```python
!pip install xtreme-vision
```


### **`Custom Instance Segmentation From Images`** 

```python
from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
objects = model.Custom_Objects(car = True)
model.Detect_Custom_Objects_From_Image(custom_objects = objects,
                                       input_path='image.jpg',
                                       output_path='./out.jpg',
                                       show_boxes = True)

from PIL import Image
Image.open('out.jpg')
```

### **`Custom Instance Segmentation From Videos`** 

```python
from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
objects = model.Custom_Objects(person=True)
model.Detect_Custom_Objects_From_Video(custom_objects = objects,
                                       input_path='video.mp4',
                                       output_path='./mask_custom.mp4',
                                       show_boxes = True,
                                       fps=25)
```

### Details:
```python
from xtreme_vision.Segmentation import Segmentation
model = Segmentation()
model.Use_MaskRCNN()
```
The First Line imports Segmentation Class from Xtreme-Vision Library and
second line Instantiates the this Class.
Third line specifies which model you want to Use, currently only maskrcnn is supported for instance segmentation

```python
objects = model.Custom_Objects(person=True)
```
The above line generates a dictionary, set those objects to True which you want to segment, 80 different objects are available in this function which are set to False by default, which you can easily set to True.

```python
model.Detect_From_Image()
model.Detect_From_Video()
```
First line specifies that, you want to segment custom objects from Images.
Second line specifies that, you want to segment custom objects from Videos.


**Detect_From_Image** Function accepts the following Arguments:
  - custom_objects (dict) -> **[dictionary returned from Custom_Objects() function]**
  - input_path (str) -> **[path to the input image in which you want to segment objects]**
  - output_path (str) -> **[output path at which you want to save the output image]**
  - show_boxes (bool) -> **[wether to show boxes around segmented objects or not]**

**Detect_From_Video** Function accepts the following Arguments:
  - custom_objects (dict) -> **[dictionary returned from Custom_Objects() function]**
  - input_path (str) -> **[path to the input video in which you want to segment objects]**
  - output_path (str) -> **[output path at which you want to save the output video]**
  - show_boxes (bool) -> **[wether to show boxes around segmented objects]**
  - fps (int) -> **[Frames Per Second for Output video]**

```python
from PIL import Image
Image.open('out.jpg')
```
The Above lines are used to open the output image from the Model to show the results