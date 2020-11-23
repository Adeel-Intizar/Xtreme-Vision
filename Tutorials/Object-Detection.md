# **`Object Detection From Images & Videos`**
```python
!pip install xtreme-vision
```


### **`Image Object Detection` Using `RetinaNet`** 

> **Note**: You can use any of the Following Models:
  - Use_CenterNet()
  - Use_RetinaNet()
  - Use_YOLOv4()
  - Use_TinyYOLOv4()
  
```python
from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_RetinaNet()
model.Detect_From_Image(input_path='kite.jpg',
                        output_path='./retinanet.jpg', 
                        extract_objects=True)

from PIL import Image
Image.open('retinanet.jpg')
```

### **`Video Object Detection` Using `YOLOv4`** 

> **Note**: You can Use Any of the following Models:
  - Use_RetinaNet()
  - Use_CenterNet()
  - Use_YOLOv4()
  - Use_TinyYOLOv4()
  
```python
from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_YOLOv4()
model.Detect_From_Video(input_path='video.mp4',
                        output_path='./output.mp4')
```

### Details:
```python
from xtreme_vision.Detection import Object_Detection
model = Object_Detection()
```
The First Line imports Object_Detection Class from Xtreme-Vision Library and
second line Instantiates the Object Detection Class.
```python
model.Use_RetinaNet()
model.Use_YOLOv4()
```
The above lines specify which Model you want to Use, 
You can use any of above mentioned models.
```python
model.Detect_From_Image()
model.Detect_From_Video()
```
First line specifies that, you want to Detect Objects from Images.
Second line specifies that, you want to Detect Objects from Videos.

**Detect_From_Image** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input image in which you want to detect objects]**
  - output_path (str) -> **[output path at which you want to save the output image]**
  - extract_objects (bool) -> **[wether you want to extract detected objects or not,Only RetinaNet supports it]**

**Detect_From_Video** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input video in which you want to detect objects]**
  - output_path (str) -> **[output path at which you want to save the output video]**
  - extract_objects (bool) -> **[wether you want to extract detected objects or not,Only RetinaNet supports it]**
  - fps (int) -> **[frames per second for output video]**

```python
from PIL import Image
Image.open('output.jpg')
```
The Above lines are used to open the output image from the Model to show the results