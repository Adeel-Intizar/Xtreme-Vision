# **`Training & Using RetinaNet`**
```python
!pip install xtreme-vision
```
For Training RetinaNet Models,
**train_annot_file** and **val_annot_file** are csv format files with one annotation per line in following format
> path/to/image.jpg,x1,y1,x2,y2,class_name

Example:
```bash
img_001.jpg,837,346,981,456,cow
img_002.jpg,215,312,279,391,cat
img_002.jpg,117,49,287,395,cat
img_002.jpg,22,5,89,84,bird
img_003.jpg,21,3,97,24,dog
```
**csv_classes_file:** is csv file in following format
>class_name,id

Example:
```bash
cow,0
cat,1
bird,2
dog,3
```
**train_data_dir** and **val_data_dir** are the relative paths to the images
**Example:**
if you have annotation files like this
```bash
img_001.jpg,837,346,981,456,cow
img_002.jpg,215,312,279,391,cat
img_002.jpg,22,5,89,84,bird
img_003.jpg,21,3,97,24,dog
```
and your images are in
```bash
data/train/img_001.jpg
data/train/img_002.jpg
data/val/img_009.jpg
```
then **train_data_dir = "data/train"**
and **val_data_dir = "data/val"**

**Note: val_annot_file and val_data_dir are optional** if you don't specify these, the training will still work
### **`Train RetinaNet Models`** 

```python
from xtreme_vision.Detection.Custom import Train_RetinaNet

model = Train_RetinaNet()
model.create_model(num_classes=4, 
                   backbone_retinanet = 'resnet101')

model.load_data(train_annot_file='train_annotation.csv', 
                val_annot_file='val_annotation.csv', 
                csv_classes_file='classes_retinanet.csv', 
                train_data_dir='data/train', 
                val_data_dir='data/test')

model.train(epochs=30, 
            lr=1e-4, 
            steps_per_epoch=400,
            save_path = 'resnet101.h5',
            nms = 0.2)
```

### Details:
```python
from xtreme_vision.Detection.Custom import Train_RetinaNet
model = Train_RetinaNet()
```
The First Line imports Train_RetinaNet Class from Xtreme-Vision Library and
second line Instantiates this Class.

```python
model.create_model(num_classes=4, 
                   backbone_retinanet = 'resnet101')
```
The above line creates model and loads the model, it takes following arguments:
  - num_classes (int) -> **[total Number of Classes in your training data]**
  - freeze_backbone (bool) -> **[wether to freeze the backbone and not train it or not]**
  - backbone_retinanet (str) -> **[Available options: resnet50, resnet101, resnet152]**

```python
model.load_data()
```
The above line loads the data for training and it takes following arguments:
  - train_annot_file (str) -> **[path to training annotation file in above mentioned format]**
  - val_annot_file (str) -> **[path to validation annotation file in above mentioned format]**
  - csv_classes_file (str) -> **[path to classes file in above mentioned format]**
  - train_data_dir (str) -> **[training data directory]**
  - val_data_dir (str) -> **[validation data directory]**

```python
model.train()
```
The Above line starts training the Model and after training, it automatically converts it to inference Model and it takes following arguments:
  - epochs (int) **[Total number of epochs to train the Model]**
  - lr (float) **[Learning Rate for the Model]**
  - steps_per_epoch (int) **[steps per epoch ro train single epoch]**
  - save_path (str) **[path to save the converted model]**
  - nms (float) **[Non Maximum Supperession thershold]**

### **`Using Trained Model`**

**classes_path** is txt file in following format
Example

```bash
0 cow
1 cat
2 bird
3 dog
```
```python
from xtreme_vision.Detection import Object_Detection

clf = Object_Detection()
clf.Use_RetinaNet(weights_path='resnet101.h5', 
                  classes_path='classes.txt',
                  backbone = 'resnet101')

clf.Detect_From_Image(input_path='data/train/img_000.jpg', 
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
clf.Use_RetinaNet()
```
The above line specifies that you want to use RetinaNet, and you have to provide the following arguments:
  - weights_path (str) -> **[path to the trained model]**
  - classes_path (str) -> **[path to the txt classes file in above mentioned format]**
  - backbone (str) -> **[backbone on which you trained the Model]**

Rest of the code is Similar to [This](Object-Detection.md)

