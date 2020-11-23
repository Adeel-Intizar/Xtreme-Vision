# **`Instance Segmentation From Images & Videos`**
```python
!pip install xtreme-vision
```


### **`Instance Segmentation From Images`** 

```python
from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
model.Detect_From_Image(input_path='pose.jpg',
                        output_path='./mask.jpg',
                        show_boxes = True)

from PIL import Image
Image.open('mask.jpg')
```

### **`Instance Segmentation From Videos`** 

```python
from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
model.Detect_From_Video(input_path='video.mp4',
                        output_path='./mask.mp4',
                        show_boxes=False,
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
model.Detect_From_Image()
model.Detect_From_Video()
```
First line specifies that, you want to segment objects from Images.
Second line specifies that, you want to segment objects from Videos.


**Detect_From_Image** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input image in which you want to segment objects]**
  - output_path (str) -> **[output path at which you want to save the output image]**
  - show_boxes (bool) -> **[wether to show boxes around segmented objects or not]**

**Detect_From_Video** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input video in which you want to segment objects]**
  - output_path (str) -> **[output path at which you want to save the output video]**
  - show_boxes (bool) -> **[wether to show boxes around segmented objects]**
  - fps (int) -> **[Frames Per Second for Output video]**

```python
from PIL import Image
Image.open('mask.jpg')
```
The Above lines are used to open the output image from the Model to show the results