# **`Human Pose Estimation From Images & Videos`**
```python
!pip install xtreme-vision
```


### **`Pose Estimation From Images`** 

```python
from xtreme_vision.Estimation import Pose_Estimation

model = Pose_Estimation()
model.Use_CenterNet()
model.Detect_From_Image(input_path = 'pose.jpg', 
                        output_path = './out.jpg')

from PIL import Image
Image.open('out.jpg')
```

### **`Pose Estimation From Videos`** 

```python
from xtreme_vision.Estimation import Pose_Estimation

model = Pose_Estimation()
model.Use_CenterNet()
model.Detect_From_Video(input_path = 'video.mp4', 
                        output_path = 'pose.mp4')
```

### Details:
```python
from xtreme_vision.Estimation import Pose_Estimation
model = Pose_Estimation()
model.Use_CenterNet()
```
The First Line imports Pose_Estimation Class from Xtreme-Vision Library and
second line Instantiates the this Class.
Third line specifies which model you want to Use, currently only centernet is supported

```python
model.Detect_From_Image()
model.Detect_From_Video()
```
First line specifies that, you want to Detect Pose from Images.
Second line specifies that, you want to Detect Pose from Videos.


**Detect_From_Image** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input image in which you want to detect pose]**
  - output_path (str) -> **[output path at which you want to save the output image]**

**Detect_From_Video** Function accepts the following Arguments:
  - input_path (str) -> **[path to the input video in which you want to detect pose]**
  - output_path (str) -> **[output path at which you want to save the output video]**
  - fps (int) -> **[Frames Per Second for Output video]**

```python
from PIL import Image
Image.open('output.jpg')
```
The Above lines are used to open the output image from the Model to show the results