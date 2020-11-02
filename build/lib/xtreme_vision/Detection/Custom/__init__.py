"""
MIT License

Copyright (c) 2020 Adeel <kingadeel2017@outlook.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from tensorflow.keras import callbacks, optimizers
from xtreme_vision.Detection.yolov4.tf import YOLOv4, SaveWeightsCallback
import os
import tensorflow as tf

class Train_Custom_Detector:
    
    """
    This is the Train-Custom-Detector class in the xtreme_vision library, it provides support of state-of-the-art
    (SOTA) models (YOLOv4 and TinyYOLOv4) for training on your custom dataset. After Instantiating this class 
    you can use pre-defined functions to start training the model on your custom dataset.
    
    First you have to specify which model you want to use, you can do it by calling one of the following functions:
        
        Use_YOLOv4()
        Use_TinyYOLOv4()
        
    After spcifying the model you have to load the data, you can do it by calling the following function:
        
        load_data()
        
    Then you can start training by calling the following function:
        
        train()
    """
    
    def __init__(self):
        
        self.modelLoaded = False
        self.modelType = ""
        
        self.classes = None
        self.model = None
        
        
    def Use_YOLOv4(self, classes_path = None, input_size:int = 608, batch_size:int = 4):
        
        """
        This function is used to specify the modelType to yolov4 and it instantiates the model.
        You can set the following parameters as well
        
        param: classes_path (path to the classes file containing the labels and it has to be (.names) file) required
        param: *input_size (size of the input image, it has to be mutiple of 32) optional
        param: *batch_size (batch size for the model, if your RAM exausted, decrease the batch_size) optional
        """
        
        if classes_path is not None:
            if os.path.isfile(classes_path):
              self.classes = classes_path
            else:
              raise FileNotFoundError ('File Does not exist. Please Provide Valid Classes_Path')
        else:
          raise FileNotFoundError ('Classes_Path is Mandatory. Please Provide Valid Classes_Path')

        self.model = YOLOv4()
        self.model.classes = self.classes
        self.model.input_size = input_size
        self.model.batch_size = batch_size
        
        self.modelType = 'yolov4'
        self.modelLoaded = True
        
    def Use_TinyYOLOv4(self, classes_path = None, input_size:int = 960, batch_size:int = 32):
        
        """
        This function is used to specify the modelType to tinyyolov4 and it instantiates the model.
        You can optionally set the following parameters as well
        
        param: classes_path (path to the classes file containing the labels and it has to be (.names) file) required
        param: *input_size (size of the input image, it has to be mutiple of 32) optional
        param: *batch_size (batch size for the model, if your RAM exausted, decrease the batch_size) optional
        """
        
        if classes_path is not None:
            if os.path.isfile(classes_path):
              self.classes = classes_path
            else:
              raise FileNotFoundError ('File Does not exist. Please Provide Valid Classes_Path')
        else:
          raise FileNotFoundError ('Classes_Path is Mandatory. Please Provide Valid Classes_Path')
        
        self.model = YOLOv4(tiny=True)
        self.model.classes = self.classes
        self.model.input_size = input_size
        self.model.batch_size = batch_size
        
        self.modelType = 'tinyyolov4'
        self.modelLoaded = True
        
        
        
    def load_data(self, train_annot_path:str, train_img_dir:str, val_annot_path:str, val_img_dir:str, weights_path:str=None):
        
        """
        This function is used to load the data for traininig the model, if you have pretrained weights file, set its path
        to the weights_path, or if you leave it to None, it will automatically download the pretrained weights for 
        retraining.
        
        Data should be in the following pattern
        
        annotation_file.txt:
            
            {
                
            <relative/absolute path to img> class_label,x,y,width,height class_label,x,y,width,height ...
            <relative/absolute path to img> class_label,x,y,width,height ...
            .
            .
            
            }
            
            class_labels should be greater or equal to 0
            x, y, width, height has to be b/w (0 and 1)
            
            if your annots are not scaled, you can scale the annotations by the following method:
                x/width-of-img, y/height-of-img, width/width-of-img, height/height-of-img
            
            
        img_dir:
            
            {
            img-001.jpg
            img-002.jpg
            img-oo3.jpg
            .
            .
            }
        
        
        You must specify the following parameters:
            
        param: train_annot_path (path to annotation file for training images)
        param: train_img_dir (path to the directory containing images for training)
        param: val_annot_path (path to annotations file for validation images)
        param: val_img_dir (path to the directory containng images for validation)
        param: weights_path (path to downloaded weights for retraining, if you don't have the file leave it to None.
                             It will automatically download the file.)
        """
        
        self.train_dataset = self.model.load_dataset(train_annot_path, image_path_prefix = train_img_dir)
        self.val_dataset = self.model.load_dataset(val_annot_path, image_path_prefix = val_img_dir, training = False)
        
        self.model.make_model()

        if weights_path is None:
          if (self.modelType == 'yolov4'):
            path = 'xtreme_vision/weights/yolotrain.weights'
            name = 'yolotrain.weights'
            url = 'https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/yolov4.conv.137'
          elif (self.modelType == 'tinyyolov4'):
            path = 'xtreme_vision/weights/tinyyolotrain.weights'
            name = 'tinyyolotrain.weights'
            url = 'https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/yolov4-tiny.conv.29'
          else:
            pass

          if os.path.isfile(path):
            print('Found Existing File...\nLoading existing file...')
            self.weights_path = path
          else:
            print('Downloading weights file...\nPlease wait...')
            self.weights_path = tf.keras.utils.get_file(name, url, cache_dir = 'xtreme_vision',cache_subdir='weights/')

        else:
          if os.path.isfile(weights_path):
            self.weights_path = weights_path
          else:
            raise FileNotFoundError ("Weights File Doesn't exist at provided path. Please provide valid path.")
          
        self.model.load_weights(self.weights_path, weights_type = 'yolo')
        
    
    def train(self, epochs:int = 1, lr:float = 0.001):
        
        """
        This function is used to Train the model, it uses Adam Optimizer to train, and it saves the weights of every 
        epoch in 'model_weights' dir, training steps_per_epoch=100 and val_steps=50 by default.
        
        You can optionally set the following parameters:
        
        param: epochs (NO of epochs to train the model)
        param: lr (learning rate for the model)
        """
        
        self.optimizer = optimizers.Adam(learning_rate = lr)
        self.model.compile(optimizer = self.optimizer, loss_iou_type = 'ciou')
        
        
        def lr_scheduler(epoch):
            if epoch < int(epochs * 0.5):
                return lr 
            elif epoch < int(epochs * 0.7):
                return lr * 0.1
            return lr * 0.01

        self.model.fit(
            self.train_dataset,
            epochs = epochs,
            callbacks=[
                callbacks.LearningRateScheduler(lr_scheduler),
                callbacks.TerminateOnNaN(),
                callbacks.TensorBoard(
                    histogram_freq=1,
                    log_dir="./logs"),
                    
                SaveWeightsCallback(
                    yolo= self.model,
                    dir_path= "./model_weights",
                    weights_type= "yolo",
                    epoch_per_save= 1),
                ],
            validation_data=self.val_dataset,
            validation_steps = 50,
            steps_per_epoch = 100)
            