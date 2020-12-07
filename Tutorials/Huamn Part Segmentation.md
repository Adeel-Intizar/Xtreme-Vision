# **`Human Part Segmentation From Images & Videos`**
```python
!pip install xtreme-vision
```


### **`Part Segmentation From Images`** 

```python
from xtreme_vision.Segmentation import Segmentation
model = Segmentation()
model.Use_PersonPart()
model.Detect_From_Image('image3.jpg', 'out.jpg')

from PIL import Image
Image.open('out.jpg')
```

### **`Part Segmentation From Videos`** 

```python
from xtreme_vision.Segmentation import Segmentation
model = Segmentation()
model.Use_PersonPart()
model.Detect_From_Video('svideo1.mp4', 'part-seg1.mp4')
```

### Details:
```python
from xtreme_vision.Segmentation import Segmentation
model = Segmentation()
model.Use_PersonPart()
```
The First Line imports Segmentation Class from Xtreme-Vision Library and
second line Instantiates the this Class.
Third line specifies which model you want to Use

```python
model.Detect_From_Image()
model.Detect_From_Video()
```
First line specifies that, you want to Detect from Images.
Second line specifies that, you want to Detect from Videos.


**Detect_From_Image** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input image in which you want to detect]**
  - output_path (str) -> **[output path at which you want to save the output image]**

**Detect_From_Video** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input video in which you want to detect]**
  - output_path (str) -> **[output path at which you want to save the output video]**
  - fps (int) -> **[Frames Per Second for Output video]**

```python
from PIL import Image
Image.open('output.jpg')
```
The Above lines are used to open the output image from the Model to show the results