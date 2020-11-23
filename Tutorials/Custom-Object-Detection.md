# **`Custom Object Detection From Images & Videos`**
```python
!pip install xtreme-vision
```


### **`Custom Image Object Detection` Using `RetinaNet`** 

> **Note**: You can use any of the Following Models:
  - Use_CenterNet()
  - Use_RetinaNet()
  - Use_YOLOv4()
  - Use_TinyYOLOv4()
  
```python
from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_RetinaNet()
custom_objects = model.Custom_Objects(kite=True)

model.Detect_Custom_Objects_From_Image(custom_objects, 
                                       input_path = 'kite.jpg',
                                       output_path = './custom_detection.jpg', 
                                       minimum_percentage_probability = 0.2,
                                       extract_objects=False)

Image.open('custom_detection.jpg')
```

### **`Custom Video Object Detection` Using `CenterNet`** 

> **Note**: You can Use Any of the following Models:
  - Use_RetinaNet()
  - Use_CenterNet()
  - Use_YOLOv4()
  - Use_TinyYOLOv4()
  
```python
from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_CenterNet()
custom_obj = model.Custom_Objects(car=True)

model.Detect_Custom_Objects_From_Video(custom_obj, 
                                       input_path = 'traffic-mini.mp4',
                                       output_path = './centernet_custom.mp4')
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
model.Use_CenterNet()
```
The above lines specify which Model you want to Use, 
You can use any of above mentioned models.
```python
custom_obj = model.Custom_Objects(car=True)
model.Detect_Custom_Objects_From_Image()
model.Detect_Custom_Objects_From_Video()
```
First line generates a dictionary, set those Objects to True which you want to Detect, it provides 80 different objects that are set to False, you can easily set them to True.
Second line specifies that, you want to Detect Custom Objects from Images.
Third line specifies that, you want to Detect Custom Objects from Videos.


**Detect_Custom_Objects_From_Image** Function accepts the following Arguments:
  - custom_objects (dict) -> **[dictionary returned from Custom_Objects Function]**
  - input_path (str) -> **[path to the input image in which you want to detect objects]**
  - output_path (str) -> **[output path at which you want to save the output image]**
  - extract_objects (bool) -> **[wether you want to extract detected objects or not,Only RetinaNet supports it]**
  - minimum_percentage_probability (float) -> **[Any thing detected with confidence below this value, will not be shown in the output, Only RetinaNet supports it]**

**Detect_Custom_Objects_From_Video** Function accepts the following Arguments:
  - custom_objects (dict) -> **[dictionary returned from Custom_Objects Function]**
  - input_path (str) -> **[path to the input video in which you want to detect objects]**
  - output_path (str) -> **[output path at which you want to save the output video]**
  - extract_objects (bool) -> **[wether you want to extract detected objects or not,Only RetinaNet supports it]**
  - minimum_percentage_probability (float) -> **[Any thing detected with confidence below this value, will not be shown in the output, Only RetinaNet supports it]**
  - fps (int) -> **[Frames Per Second for Output video]**

```python
from PIL import Image
Image.open('output.jpg')
```
The Above lines are used to open the output image from the Model to show the results