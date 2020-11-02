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


class ResidualBlock(tf.keras.Model):
    def __init__(self, input_channels: int, output_channels: int, strides=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Convolution(output_channels, 3, strides)

        self.conv2 = Convolution(output_channels, 3, activation=False)

        self.shortcut = tf.keras.Sequential()

        if strides > 1 or input_channels != output_channels:
            self.shortcut.add(Convolution(output_channels, 1, strides, activation=False))

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)

        y = self.shortcut(inputs)

        return tf.keras.activations.relu(x + y)
