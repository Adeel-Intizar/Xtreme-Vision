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
from xtreme_vision.Detection.retinanet.utils.config import read_config_file, parse_anchor_parameters, parse_pyramid_levels
from xtreme_vision.Detection.retinanet.utils.model import freeze as freeze_model
from xtreme_vision.Detection.retinanet.models.retinanet import retinanet_bbox

import tensorflow as tf






def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, optimizer_clipnorm=0.001, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    pyramid_levels = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()
    if config and 'pyramid_levels' in config:
        pyramid_levels = parse_pyramid_levels(config)

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, pyramid_levels=pyramid_levels), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, pyramid_levels=pyramid_levels), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, pyramid_levels=pyramid_levels)

    # compile model
    training_model.compile(
        loss={
            'regression'    : smooth_l1(),
            'classification': focal()
        },
        optimizer=tf.keras.optimizers.Adam(lr=lr, clipnorm=optimizer_clipnorm)
    )

    return model, training_model, prediction_model




class Train_RetinaNet:
    
    def __init__(self):
        
        self.model = None
        self.num_classes = 0
        self.backbone = ''
        
    def load_data(self, num_classes:int, train_annot_file:str, val_annot_file:str, csv_classes_file:str,
                  data_dir:str, backbone:str='resnet50'):
        
        """
        This Function is Used to Load the Data For Training the RetinaNet Model.
        
        param: num_classes (Total Number of classes on which you are training the model)
        param: csv_annot_file (Path to the CSV Annotations File)
        param: csv_classes_file (Path to the CSV Classes File, mapping names to labels)
        param: data_dir (Relative path to directory with respect to annotations file)
        """
        if 'resnet' not in backbone:
            raise RuntimeError ('Available Backbones: ResNet50\tResNet101\tResNet152')
            
        else:
            self.backbone = backbone
        
        back = models.backbone(self.backbone)
        
        common_args = {
            'batch_size'       : 50,
            'image_min_side'   : 800,
            'image_max_side'   : 1333,
            'no_resize'        : True,
            'preprocess_image' : back.preprocess_image,
            'group_method'     : 'ratio'
            }
        
        self.num_classes = num_classes
        self.train_gen = CSVGenerator(train_annot_file, csv_classes_file, data_dir, **common_args)
        self.val_gen = CSVGenerator(val_annot_file, csv_classes_file, data_dir, shuffle_groups=False, **common_args)

        
    def train(self, epochs:int, lr:float, steps_per_epoch:int, save_path:str='model.h5'):
        
        self.model = models.backbone(self.backbone).retinanet(num_classes=self.num_classes)
        
        self.model.compile(
            loss={
                'regression'    : smooth_l1(),
                'classification': focal() },
            
            optimizer=tf.keras.optimizers.Adam(lr=lr, clipnorm=0.001))  

        callbacks = [
	     tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                          patience=21, 
                                          verbose=1, 
                                          restore_best_weights=True),
             tf.keras.callbacks.TerminateOnNaN(),
             tf.keras.callbacks.ReduceLROnPlateau('loss', 
                                                  patience=3, 
                                                  verbose=1, 
                                                  min_lr=1e-8)]
        
        self.model.fit(self.train_gen, 
                       steps_per_epoch = steps_per_epoch, 
                       epochs=epochs, 
                       validation_data = self.val_gen,
                       validation_steps = 1,
                       callbacks=callbacks)
       
        check_training_model(self.model)
        
        converted = convert_model(self.model, 
                          parallel_iterations=32, 
                          max_detections=300, 
                          score_threshold=0.05, 
                          nms_threshold=0.5)
        
        converted.save(save_path)
        