# **`Training & Using TinyYOLOv4`**
```python
!pip install xtreme-vision
```
For Training YOLO Models,
**train_images_file** and **val_images_file** are txt format files **containing image names**, one name per line in following format

```bash
img_001.jpg
img_002.jpg
img_003.jpg
img_004.jpg
```
**train_img_dir** and **val_img_dir** are path to the **directory containing training images and annotations**.
```bash
img_001.jpg
img_001.txt
img_002.jpg
img_002.txt
img_003.jpg
img_003.txt
```
Annotations must be in following format:
class_label x-min y-min x-max y-max

class_label (int) -> greater or equal to zero
x-min, x-max, y-min, y-max has to be b/w (0 and 1)

if your annotations are not scaled, you can scale the annotations by the following method:
x-min/width-of-img 
x-max/width-of-img 
y-min/height-of-img 
y-max/height-of-img

img_001.txt:
```bash
0 0.123 0.234 0.345 0.456
```
img_002.txt:
```bash
1 0.1000 0.2340 0.3450 0.4560
```

**classes.names** file contains class_labels on label per line:
```bash
Cat
Dog
Cow
Bird
```

**Note: val_images_file and val_img_dir are optional** if you don't specify these, the training will still work
### **`Train TinyYOLOv4 Model`** 
```python
from xtreme_vision.Detection.Custom import Train_TinyYOLOv4
model = Train_TinyYOLOv4()
model.create_model(classes_path='classes.names',
                   input_size=640,
                   batch_size=32)

model.load_data(train_images_file = 'names.txt',
                train_img_dir='train/')

model.train(epochs=20,
            lr=1e-4,
            steps_per_epoch=400)
```

### Details:
```python
from xtreme_vision.Detection.Custom import Train_TinyYOLOv4
model = Train_TinyYOLOv4()
```
The First Line imports Train_TinyYOLOv4 Class from Xtreme-Vision Library and
second line Instantiates this Class.

```python
model.create_model()
```
The above line creates model and loads the model, it takes following arguments:
  - classes_path (str) -> **[.names file in above mentioned format]**
  - input_size (int) -> **[input image size for Model, it has to be multiple of 32]**
  - batch_size (int) -> **[batch_size for Model, if RAM crashes, decrease batch size]**

```python
model.load_data()
```
The above line loads the data for training and it takes following arguments:
  - train_images_file (str) -> **[txt file containing image names mentioned in above format]**
  - val_images_file (str) -> **[txt file containing image names mentioned in above format]**
  - train_img_dir (str) -> **[path to directory containing images and annotations]**
  - val_img_dir (str) -> **[path to directory containing images and annotations]**

```python
model.train()
```
The Above line starts training and it saves the model after every epoch and it takes following arguments:
  - epochs (int) -> **[Total number of epochs to train the Model]**
  - lr (float) -> **[Learning Rate for the Model]**
  - steps_per_epoch (int) -> **[steps per epoch to train single epoch]**

### **`Using Trained Model`**


```python
from xtreme_vision.Detection import Object_Detection
clf = Object_Detection()
clf.Use_TinyYOLOv4(weights_path='model_weights/yolov4-tiny-9.weights',
                   classes_path='classes.names',
                   input_shape=640,
                   iou = 0.2)
clf.Detect_From_Image(input_path='/content/train/01809061f4e2db2b.jpg', 
                      output_path='out.jpg')
from PIL import Image
Image.open('out.jpg')
```
### Details:
```python
from xtreme_vision.Detection import Object_Detection
clf = Object_Detection()
```
First line imports Object_Detection class from Xtreme_Vision Library.
Second line instatiates this class.

```python
clf.Use_TinyYOLOv4()
```
The above line specifies that you want to use YOLOv4, and you have to provide the following arguments:
  - weights_path (str) -> **[path to the directory containing trained weights]**
  - classes_path (str) -> **[path to the .names file mentioned in above format]**
  - input_shape (int) -> **[same shape on which you trained the Model]**
  - iou (float) -> **[Intersection Over Union threshold]**

Rest of the code is Similar to [This](Object-Detection.md)

