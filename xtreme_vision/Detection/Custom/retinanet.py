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



from xtreme_vision.Detection.retinanet.preprocessing.csv_generator import CSVGenerator
from xtreme_vision.Detection.retinanet import models
from xtreme_vision.Detection.retinanet.losses import smooth_l1, focal
from xtreme_vision.Detection.retinanet.models import convert_model, check_training_model
from xtreme_vision.Detection.retinanet.utils.config import parse_anchor_parameters, parse_pyramid_levels
from xtreme_vision.Detection.retinanet.utils.model import freeze as freeze_model

import tensorflow as tf




class Train_RetinaNet:
    
    def __init__(self):
        
        self.model = None
        self.num_classes = 0
        self.backbone = ''
    
    
    def create_model(self, num_classes:int, freeze_backbone=True, backbone_retinanet:str='resnet50'):
        """ Creates model

    Args
        backbone : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        """
        
        def model_with_weights(model, weights):
            if weights is not None:
                model.load_weights(weights, by_name=True, skip_mismatch=True)
            return model

        def download_weights(backbone):

          if backbone == 'resnet50':
            print('-' * 20)
            print('Downloading Weights File...\nPlease Wait...')
            print('-' * 20)
            self.weights = tf.keras.utils.get_file('retinanet_weights.h5',
                                                        'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5',
                                                        cache_subdir = 'weights/', cache_dir = 'xtreme_vision')
          
          elif backbone == 'resnet101':
            print('-' * 20)
            print('Downloading Weights File...\nPlease Wait...')
            print('-' * 20)
            self.weights = tf.keras.utils.get_file('retinanet_weights.h5',
                                                        'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet101_oid_v1.0.0.h5',
                                                        cache_subdir = 'weights/', cache_dir = 'xtreme_vision')

          elif backbone == 'resnet152':
            print('-' * 20)
            print('Downloading Weights File...\nPlease Wait...')
            print('-' * 20)
            self.weights = tf.keras.utils.get_file('retinanet_weights.h5',
                                                        'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet152_oid_v1.0.0.h5',
                                                        cache_subdir = 'weights/', cache_dir = 'xtreme_vision')

          else:
            self.weights = None
        
        self.num_classes= num_classes
        self.backbone = backbone_retinanet
        self.back = models.backbone(self.backbone)
        modifier = freeze_model if freeze_backbone else None

        # load anchor parameters, or pass None (so that defaults will be used)
        num_anchors   = None
        pyramid_levels = None

        download_weights(self.backbone)
        self.model = model_with_weights(self.back.retinanet(self.num_classes, num_anchors=num_anchors, modifier=modifier, pyramid_levels=pyramid_levels), self.weights)
    
    
    def load_data(self, train_annot_file:str, val_annot_file:str, csv_classes_file:str, train_data_dir:str, val_data_dir:str):
        
        """
        This Function is Used to Load the Data For Training the RetinaNet Model.
        
        param: num_classes (Total Number of classes on which you are training the model)
        param: csv_annot_file (Path to the CSV Annotations File)
        param: csv_classes_file (Path to the CSV Classes File, mapping names to labels)
        param: data_dir (Relative path to directory with respect to annotations file)
        """
        
        common_args = {
            'batch_size'       : 1,
            'image_min_side'   : 800,
            'image_max_side'   : 1333,
            'no_resize'        : True,
            'preprocess_image' : self.back.preprocess_image,
            'group_method'     : 'ratio'
            }
        
        self.train_gen = CSVGenerator(train_annot_file, csv_classes_file, train_data_dir, **common_args)
        self.val_gen = CSVGenerator(val_annot_file, csv_classes_file, val_data_dir, shuffle_groups=False, **common_args)

      

    def train(self, epochs:int, lr:float, steps_per_epoch:int, save_path:str='model.h5', restore_best_weights:bool=True):
        
        
        self.model.compile(
            loss={
                'regression'    : smooth_l1(),
                'classification': focal() },
            
            optimizer=tf.keras.optimizers.Adam(lr=lr, clipnorm=0.001))  

        def sched(epoch, lr):
          return lr * tf.math.exp(-0.1)

        callbacks = [
          tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                          patience=31, 
                                          verbose=1, 
                                          restore_best_weights=restore_best_weights),
          tf.keras.callbacks.TerminateOnNaN(),
          tf.keras.callbacks.LearningRateScheduler(sched, verbose=1)
          ]
        
        self.model.fit(self.train_gen, 
                       steps_per_epoch = steps_per_epoch, 
                       epochs=epochs, 
                       validation_data = self.val_gen,
                       validation_steps = 1,
                       callbacks=callbacks,
                       verbose = 2)
       
        check_training_model(self.model)
        
        converted = convert_model(self.model, 
                          parallel_iterations=32, 
                          max_detections=300, 
                          score_threshold=0.05, 
                          nms_threshold=0.5)
        
        converted.save(save_path)
        