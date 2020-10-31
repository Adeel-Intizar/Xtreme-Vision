# **`Xtreme-Vision`**


```python
!pip install xtreme-vision
```

## `Training Object-Detection Models`

Note: You can use any of the Following Functions, Rest of the code remains same


*   `Use_YOLOv4()`
*   `Use_TinyYOLOv4()`




```python
from xtreme_vision.Detection.Custom import Train_Custom_Detector

model = Train_Custom_Detector()
model.Use_YOLOv4(classes_path = 'classes.names',
                 input_size = 608,
                 batch_size = 4)

model.load_data(train_annot_path = 'training.txt',
                train_img_dir = './',
                val_annot_path = 'validation.txt',
                val_img_dir = './',
                weights_path = None)

model.train(epochs=1,
            lr = 0.001)
```

    Downloading weights file...
    Please wait...
    Downloading data from https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/yolov4.conv.137
    170041344/170038676 [==============================] - 8s 0us/step
    grid: 76 iou_loss: 0 conf_loss: 11794.4941 prob_loss: 0 total_loss 11794.4941
    grid: 38 iou_loss: 0 conf_loss: 2977.81519 prob_loss: 0 total_loss 2977.81519
    grid: 19 iou_loss: 10.1079569 conf_loss: 724.619141 prob_loss: 22.491045 total_loss 757.21814
    grid: 76 iou_loss: 0 conf_loss: 11781.2041 prob_loss: 0 total_loss 11781.2041
    grid: 38 iou_loss: 0 conf_loss: 2962.00684 prob_loss: 0 total_loss 2962.00684
    grid: 19 iou_loss: 9.96237659 conf_loss: 714.340332 prob_loss: 23.7497559 total_loss 748.05249
