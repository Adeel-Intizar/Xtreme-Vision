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

class Train_TinyYOLOv4:
    
    """
    This is the Train_TinyYOLOv4 class in the xtreme_vision library, it provides support of state-of-the-art
    (SOTA) models (TinyYOLOv4) for training on your custom dataset. After Instantiating this class 
    you can use pre-defined functions to start training the model on your custom dataset.
    
        create_model()                                  # to create the model
        load_data()                                     # to load the data
        train()                                         # to start training
    """
    
    def __init__(self):
        
        self.modelType = ""
        
        self.classes = None
        self.model = None
        
        
    def create_model(self, classes_path = None, input_size:int = 640, batch_size:int = 32):
        
        """
        This function is used to specify the modelType to yolov4 and it instantiates the model.
        You can set the following parameters as well
        
        param: classes_path (path to the classes file containing the labels and it has to be (.names) file)
        param: input_size (size of the input image, it has to be mutiple of 32)
        param: batch_size (batch size for the model, if your RAM exausted, decrease the batch_size)
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

        
    def load_data(self, train_images_file:str, train_img_dir:str, val_images_file:str=None, val_img_dir:str=None,
                  weights_path:str=None):
        
        """
        This function is used to load the data for traininig the YOLO models, if you have pretrained weights file, set its path
        to the weights_path, or if you leave it to None, it will automatically download the pretrained weights for 
        retraining.
        
        param: train_images_file (path to txt file containing images names one/name/per/line)
        param: train_img_dir (path to the directory containing images for training and annotations)
        param: val_images_file (path to txt file containing images names one/name/per/line)
        param: val_img_dir (path to the directory containng images for validation and annotations)
        param: weights_path (path to downloaded weights for retraining, if you don't have the file leave it to None.
                             It will automatically download the file.)
        
        Data should be in the following pattern
        
        images_file.txt:
            
            {
                
            <img 1.jpg>
            <img 2.jpg>
            ...
            }
            
            class_labels should be greater or equal to 0
            x-min, x-max, y-min, y-max has to be b/w (0 and 1)
            
            if your annots are not scaled, you can scale the annotations by the following method:
                x/width-of-img, y/height-of-img
            
            
        img_dir:
            
            {
            img-001.jpg
            img-001.txt
            img-002.jpg
            img-002.txt
            img-oo3.jpg
            img-003.txt
            ...
            }
            
        image-001.txt:
            {
            class_label x-min y-min x-max y-max
            }

        """
        
        self.train_dataset = self.model.load_dataset(train_images_file, image_path_prefix = train_img_dir, dataset_type="yolo")
        
        if (val_images_file != None) and (val_img_dir != None):
            self.val_dataset = self.model.load_dataset(val_images_file, image_path_prefix = val_img_dir, training = False)
            self.val_steps = 5
        else:
            self.val_dataset = None
            self.val_steps = None
        
        self.model.make_model()

        if weights_path is None:
          if (self.modelType == 'tinyyolov4'):
            path = 'xtreme_vision/weights/tinyyolotrain.weights'
            name = 'tinyyolotrain.weights'
            url = 'https://github.com/Adeel-Intizar/Xtreme-Vision/releases/download/1.0/yolov4-tiny.conv.29'
          else:
            raise RuntimeError ('Invalid ModelType: Valid type is YOLOv4.')

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
        

    def train(self, epochs:int, lr:float, steps_per_epoch:int=1):
        
        """
        This function is used to Train the model, it uses Adam Optimizer to train, and it saves the weights of every 
        epoch in 'model_weights' dir, training steps_per_epoch=1 and val_steps=5 by default.
        
        You can optionally set the following parameters:
        
        param: epochs (NO of epochs to train the model)
        param: lr (learning rate for the model)
        param: steps_per_epoch (it defines steps per epoch for training data)
        """
        
        if (self.modelType=='tinyyolov4'):
            self.optimizer = optimizers.Adam(learning_rate = lr)
            self.model.compile(optimizer = self.optimizer, loss_iou_type = 'ciou', loss_verbose=0)
        
        
            def lr_scheduler(epoch, lr):
                return lr * tf.math.exp(-0.1)

            self.model.fit(
                self.train_dataset,
                epochs = epochs,
                callbacks=[
                    callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
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
                validation_steps = self.val_steps,
                steps_per_epoch = steps_per_epoch)
        else:
            raise RuntimeError ('Invalid ModelType: Valid Type is YOLOv4')
            