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

from xtreme_vision.Detection.centernet.model.hourglass import Hourglass104
from xtreme_vision.Detection.centernet.model.head import PoseEstimationHead


class PoseEstimationModel(tf.keras.Model):
    def __init__(self, num_joints: int):
        super(PoseEstimationModel, self).__init__()

        self.hourglass104 = Hourglass104()

        self.head1 = PoseEstimationHead(num_joints)
        self.head2 = PoseEstimationHead(num_joints)

    def call(self, inputs, training=None, mask=None):
        hourglass2_output, hourglass1_output = self.hourglass104(inputs)

        output1 = self.head1(hourglass1_output)
        output2 = self.head2(hourglass2_output)

        return [output2, output1]
