# **`Semantic Segmentation From Images & Videos`**
```python
!pip install xtreme-vision
```


### **`Semantic Segmentation From Images`** 

```python
from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_DeepLabv3()
model.Detect_From_Image(input_path='image.jpg',
                        output_path='./out.jpg')

from PIL import Image
Image.open('out.jpg')
```

### **`Semantic Segmentation From Videos`** 

```python
from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_DeepLabv3()
model.Detect_From_Video(input_path='video.mp4',
                        output_path='./deeplab.mp4',
                        fps=25)
```

### Details:
```python
from xtreme_vision.Segmentation import Segmentation
model = Segmentation()
model.Use_DeepLabv3()
```
The First Line imports Segmentation Class from Xtreme-Vision Library and
second line Instantiates the this Class.
Third line specifies which model you want to Use, currently only deeplabv3+ is supported for semantic segmentation

```python
model.Detect_From_Image()
model.Detect_From_Video()
```
First line specifies that, you want to segment objects from Images.
Second line specifies that, you want to segment objects from Videos.


**Detect_From_Image** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input image in which you want to segment objects]**
  - output_path (str) -> **[output path at which you want to save the output image]**

**Detect_From_Video** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input video in which you want to segment objects]**
  - output_path (str) -> **[output path at which you want to save the output video]**
  - fps (int) -> **[Frames Per Second for Output video]**

```python
from PIL import Image
Image.open('out.jpg')
```
The Above lines are used to open the output image from the Model to show the results