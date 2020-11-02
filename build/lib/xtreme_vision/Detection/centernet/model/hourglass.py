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
from xtreme_vision.Detection.centernet.model.residual import ResidualBlock
from xtreme_vision.Detection.centernet.model.convolution import Convolution


class HourglassModule(tf.keras.Model):
    def __init__(self, current_channels: int, next_channels: int, inner_module: tf.keras.Model):
        super(HourglassModule, self).__init__()

        self.current_channels = current_channels

        self.res_shortcut1 = ResidualBlock(current_channels, current_channels)
        self.res_shortcut2 = ResidualBlock(current_channels, current_channels)

        self.res_in1 = ResidualBlock(current_channels, next_channels, strides=2)
        self.res_in2 = ResidualBlock(next_channels, next_channels)

        self.inner_module = inner_module

        self.res_out1 = ResidualBlock(next_channels, next_channels)
        self.res_out2 = ResidualBlock(next_channels, current_channels)
        self.upsample = tf.keras.layers.UpSampling2D()

    def call(self, inputs, training=None, mask=None):
        x = self.res_in2(self.res_in1(inputs))
        x = self.inner_module(x)
        x = self.res_out2(self.res_out1(x))
        x = self.upsample(x)

        y = self.res_shortcut2(self.res_shortcut1(inputs))

        return x + y


class Hourglass(tf.keras.Model):
    def __init__(self):
        super(Hourglass, self).__init__()

        module = tf.keras.Sequential()
        for _ in range(4):
            module.add(ResidualBlock(512, 512))

        channels = [512, 384, 384, 384, 256, 256]

        for next_channels, current_channels in zip(channels, channels[1:]):
            module = HourglassModule(current_channels, next_channels, module)

        self.module = module
        self.conv = Convolution(256, 3)

    def call(self, inputs, training=None, mask=None):
        return self.conv(self.module(inputs))


class Hourglass104(tf.keras.Model):
    def __init__(self):
        super(Hourglass104, self).__init__()

        self.pre_sequential = tf.keras.Sequential()
        self.pre_sequential.add(Convolution(128, 7, 2))
        self.pre_sequential.add(ResidualBlock(128, 256, 2))

        self.hourglass1 = Hourglass()

        self.conv_hourglass1 = Convolution(256, 1, activation=False)
        self.conv_shortcut = Convolution(256, 1, activation=False)
        self.residual_before_hourglass2 = ResidualBlock(256, 256)

        self.hourglass2 = Hourglass()

    def call(self, inputs, training=None, mask=None):
        pre_output = self.pre_sequential(inputs)

        hourglass1_output = self.hourglass1(pre_output)

        hourglass2_input = self.residual_before_hourglass2(
            tf.keras.activations.relu(
                self.conv_hourglass1(hourglass1_output) + self.conv_shortcut(pre_output)
            )
        )

        hourglass2_output = self.hourglass2(hourglass2_input)

        return [hourglass2_output, hourglass1_output]
