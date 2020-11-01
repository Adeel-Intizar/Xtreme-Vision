"""
MIT License

Copyright (c) 2020 Licht Takeuchi

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
import tensorflow as tf

from xtreme_vision.Detection.centernet.model.convolution import Convolution


class CenterNetHeadPart(tf.keras.Model):
    def __init__(self, n: int):
        super(CenterNetHeadPart, self).__init__()

        self.conv1 = Convolution(256, 3, batch_normalization=False)
        self.conv2 = Convolution(n, 1, batch_normalization=False, activation=False)

    def call(self, inputs, training=None, mask=None):
        return self.conv2(self.conv1(inputs))


class ObjectDetectionHead(tf.keras.Model):
    def __init__(self, num_classes: int):
        super(ObjectDetectionHead, self).__init__()

        self.class_heatmap_predictor = CenterNetHeadPart(num_classes)
        self.offset_predictor = CenterNetHeadPart(2)
        self.wh_predictor = CenterNetHeadPart(2)

    def call(self, inputs, training=None, mask=None):
        return [
            tf.sigmoid(self.class_heatmap_predictor(inputs)),
            self.offset_predictor(inputs),
            self.wh_predictor(inputs)
        ]


class PoseEstimationHead(tf.keras.Model):
    def __init__(self, num_joints: int):
        super(PoseEstimationHead, self).__init__()

        self.joint_heatmap_predictor = CenterNetHeadPart(num_joints)
        self.joint_locations_predictor = CenterNetHeadPart(2 * num_joints)
        self.joint_offset_predictor = CenterNetHeadPart(2)

        self.class_heatmap_predictor = CenterNetHeadPart(1)
        self.offset_predictor = CenterNetHeadPart(2)
        self.wh_predictor = CenterNetHeadPart(2)

    def call(self, inputs, training=None, mask=None):
        return [
            tf.sigmoid(self.joint_heatmap_predictor(inputs)),
            self.joint_locations_predictor(inputs),
            self.joint_offset_predictor(inputs),
            tf.sigmoid(self.class_heatmap_predictor(inputs)),
            self.offset_predictor(inputs),
            self.wh_predictor(inputs)
        ]
