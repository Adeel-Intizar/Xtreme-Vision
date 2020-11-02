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


class Convolution(tf.keras.Model):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 strides: int = 1,
                 batch_normalization: bool = True,
                 activation: bool = True
                 ):
        super(Convolution, self).__init__()

        self.sequential = tf.keras.Sequential()

        if kernel_size > 1:
            padding_size = kernel_size // 2
            self.sequential.add(tf.keras.layers.ZeroPadding2D((padding_size, padding_size)))

        self.sequential.add(tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides,
            padding='VALID', use_bias=not batch_normalization,
            kernel_initializer='he_normal'
        ))

        if batch_normalization:
            self.sequential.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))

        if activation:
            self.sequential.add(tf.keras.layers.ReLU())

    def call(self, x, **kwargs):
        return self.sequential(x)
