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
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, backend, layers, Model


class YOLOv3Head(Model):
    def __init__(self, anchors, num_classes, xysclaes):
        super(YOLOv3Head, self).__init__(name="YOLOv3Head")
        self.a_half = None
        self.anchors = anchors
        self.grid_coord = []
        self.grid_size = None
        self.image_size = None
        self.num_classes = num_classes
        self.scales = xysclaes

        self.reshape0 = layers.Reshape((-1,))
        self.reshape1 = layers.Reshape((-1,))
        self.reshape2 = layers.Reshape((-1,))

        self.concat0 = layers.Concatenate(axis=-1)
        self.concat1 = layers.Concatenate(axis=-1)
        self.concat2 = layers.Concatenate(axis=-1)

    def build(self, input_shape):
        grid = (input_shape[0][1], input_shape[1][1], input_shape[2][1])

        self.reshape0.target_shape = (grid[0], grid[0], 3, 5 + self.num_classes)
        self.reshape1.target_shape = (grid[1], grid[1], 3, 5 + self.num_classes)
        self.reshape2.target_shape = (grid[2], grid[2], 3, 5 + self.num_classes)

        self.a_half = [
            tf.constant(
                0.5,
                dtype=tf.float32,
                shape=(1, grid[i], grid[i], 3, 2),
            )
            for i in range(3)
        ]

        for i in range(3):
            xy_grid = tf.meshgrid(tf.range(grid[i]), tf.range(grid[i]))
            xy_grid = tf.stack(xy_grid, axis=-1)
            xy_grid = xy_grid[tf.newaxis, :, :, tf.newaxis, :]
            xy_grid = tf.tile(xy_grid, [1, 1, 1, 3, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)
            self.grid_coord.append(xy_grid)

        self.grid_size = grid
        self.image_size = grid[0] * 8

    def call(self, x):
        raw_s, raw_m, raw_l = x

        raw_s = self.reshape0(raw_s)
        raw_m = self.reshape1(raw_m)
        raw_l = self.reshape2(raw_l)

        txty_s, twth_s, conf_s, prob_s = tf.split(
            raw_s, (2, 2, 1, self.num_classes), axis=-1
        )
        txty_m, twth_m, conf_m, prob_m = tf.split(
            raw_m, (2, 2, 1, self.num_classes), axis=-1
        )
        txty_l, twth_l, conf_l, prob_l = tf.split(
            raw_l, (2, 2, 1, self.num_classes), axis=-1
        )

        txty_s = activations.sigmoid(txty_s)
        txty_s = (txty_s - self.a_half[0]) * self.scales[0] + self.a_half[0]
        bxby_s = (txty_s + self.grid_coord[0]) / self.grid_size[0]
        txty_m = activations.sigmoid(txty_m)
        txty_m = (txty_m - self.a_half[1]) * self.scales[1] + self.a_half[1]
        bxby_m = (txty_m + self.grid_coord[1]) / self.grid_size[1]
        txty_l = activations.sigmoid(txty_l)
        txty_l = (txty_l - self.a_half[2]) * self.scales[2] + self.a_half[2]
        bxby_l = (txty_l + self.grid_coord[2]) / self.grid_size[2]

        conf_s = activations.sigmoid(conf_s)
        conf_m = activations.sigmoid(conf_m)
        conf_l = activations.sigmoid(conf_l)

        prob_s = activations.sigmoid(prob_s)
        prob_m = activations.sigmoid(prob_m)
        prob_l = activations.sigmoid(prob_l)

        bwbh_s = (self.anchors[0] / self.image_size) * backend.exp(twth_s)
        bwbh_m = (self.anchors[1] / self.image_size) * backend.exp(twth_m)
        bwbh_l = (self.anchors[2] / self.image_size) * backend.exp(twth_l)

        pred_s = self.concat0([bxby_s, bwbh_s, conf_s, prob_s])
        pred_m = self.concat1([bxby_m, bwbh_m, conf_m, prob_m])
        pred_l = self.concat2([bxby_l, bwbh_l, conf_l, prob_l])

        return pred_s, pred_m, pred_l


class YOLOv3HeadTiny(Model):
    def __init__(self, anchors, num_classes, xysclaes):
        super(YOLOv3HeadTiny, self).__init__(name="YOLOv3HeadTiny")
        self.a_half = []
        self.anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        self.grid_coord = []
        self.grid_size = None
        self.image_size = None
        self.num_classes = num_classes
        self.scales = xysclaes

    def build(self, input_shape):
        _size = [shape[1] for shape in input_shape]

        for i in range(2):
            xy_grid = np.meshgrid(np.arange(_size[i]), np.arange(_size[i]))
            xy_grid = np.stack(xy_grid, axis=-1)
            xy_grid = xy_grid[np.newaxis, ...]
            self.grid_coord.append(
                tf.convert_to_tensor(xy_grid, dtype=tf.float32)
            )

        self.grid_size = tf.convert_to_tensor(_size, dtype=tf.float32)
        self.image_size = tf.convert_to_tensor(
            _size[0] * 16.0, dtype=tf.float32
        )

    def call(self, x):
        raw_m, raw_l = x

        sig_m = activations.sigmoid(raw_m)
        sig_l = activations.sigmoid(raw_l)

        # Dim(batch, grid, grid, 5 + num_classes)
        sig_m = tf.split(sig_m, 3, axis=-1)
        raw_m = tf.split(raw_m, 3, axis=-1)
        sig_l = tf.split(sig_l, 3, axis=-1)
        raw_l = tf.split(raw_l, 3, axis=-1)

        for i in range(3):
            txty_m, _, conf_prob_m = tf.split(sig_m[i], (2, 2, -1), axis=-1)
            _, twth_m, _ = tf.split(raw_m[i], (2, 2, -1), axis=-1)
            txty_m = (txty_m - 0.5) * self.scales[0] + 0.5
            bxby_m = (txty_m + self.grid_coord[0]) / self.grid_size[0]
            bwbh_m = (self.anchors[0][i] / self.image_size) * backend.exp(
                twth_m
            )
            sig_m[i] = tf.concat([bxby_m, bwbh_m, conf_prob_m], axis=-1)

            txty_l, _, conf_prob_l = tf.split(sig_l[i], (2, 2, -1), axis=-1)
            _, twth_l, _ = tf.split(raw_l[i], (2, 2, -1), axis=-1)
            txty_l = (txty_l - 0.5) * self.scales[1] + 0.5
            bxby_l = (txty_l + self.grid_coord[1]) / self.grid_size[1]
            bwbh_l = (self.anchors[1][i] / self.image_size) * backend.exp(
                twth_l
            )
            sig_l[i] = tf.concat([bxby_l, bwbh_l, conf_prob_l], axis=-1)

        # Dim(batch, grid, grid, 3 * (5 + num_classes))
        pred_m = tf.concat(sig_m, axis=-1)
        pred_l = tf.concat(sig_l, axis=-1)

        return pred_m, pred_l
