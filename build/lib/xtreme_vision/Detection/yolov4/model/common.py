"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

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
from typing import Union

import tensorflow as tf
from tensorflow.keras import backend, layers, Sequential
from tensorflow.keras.layers import Layer


class Mish(Layer):
    def call(self, x):
        # pylint: disable=no-self-use
        return x * backend.tanh(backend.softplus(x))


class YOLOConv2D(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, tuple],
        activation: str = "mish",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        strides: Union[int, tuple] = 1,
        **kwargs
    ):
        super(YOLOConv2D, self).__init__(**kwargs)
        self.activation = activation
        self.filters = filters
        self.input_dim = None
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = strides

        self.sequential = Sequential()

        if self.strides[0] == 2:
            self.sequential.add(layers.ZeroPadding2D(((1, 0), (1, 0))))

        self.sequential.add(
            layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding="same" if self.strides[0] == 1 else "valid",
                strides=self.strides,
                use_bias=not self.activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
            )
        )

        if self.activation is not None:
            self.sequential.add(layers.BatchNormalization())

        if self.activation == "mish":
            self.sequential.add(Mish())
        elif self.activation == "leaky":
            self.sequential.add(layers.LeakyReLU(alpha=0.1))
        elif self.activation == "relu":
            self.sequential.add(layers.ReLU())

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x):
        return self.sequential(x)
