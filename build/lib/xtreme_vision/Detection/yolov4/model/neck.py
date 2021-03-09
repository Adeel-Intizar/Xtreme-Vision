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
from tensorflow.keras import layers, Model

from xtreme_vision.Detection.yolov4.model.common import YOLOConv2D


class PANet(Model):
    def __init__(
        self,
        num_classes,
        activation: str = "leaky",
        kernel_regularizer=None,
    ):
        super(PANet, self).__init__(name="PANet")
        self.conv78 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.upSampling78 = layers.UpSampling2D(interpolation="bilinear")
        self.conv79 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat78_79 = layers.Concatenate(axis=-1)

        self.conv80 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv81 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv82 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv83 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv84 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv85 = YOLOConv2D(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.upSampling85 = layers.UpSampling2D(interpolation="bilinear")
        self.conv86 = YOLOConv2D(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat85_86 = layers.Concatenate(axis=-1)

        self.conv87 = YOLOConv2D(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv88 = YOLOConv2D(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv89 = YOLOConv2D(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv90 = YOLOConv2D(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv91 = YOLOConv2D(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv92 = YOLOConv2D(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv93 = YOLOConv2D(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv94 = YOLOConv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat84_94 = layers.Concatenate(axis=-1)

        self.conv95 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv96 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv97 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv98 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv99 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv100 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv101 = YOLOConv2D(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv102 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            strides=2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat77_102 = layers.Concatenate(axis=-1)

        self.conv103 = YOLOConv2D(
            filters=512,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv104 = YOLOConv2D(
            filters=1024,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv105 = YOLOConv2D(
            filters=512,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv106 = YOLOConv2D(
            filters=1024,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv107 = YOLOConv2D(
            filters=512,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv108 = YOLOConv2D(
            filters=1024,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv109 = YOLOConv2D(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, x):
        route1, route2, route3 = x

        x1 = self.conv78(route3)
        part2 = self.upSampling78(x1)
        part1 = self.conv79(route2)
        x1 = self.concat78_79([part1, part2])

        x1 = self.conv80(x1)
        x1 = self.conv81(x1)
        x1 = self.conv82(x1)
        x1 = self.conv83(x1)
        x1 = self.conv84(x1)

        x2 = self.conv85(x1)
        part2 = self.upSampling85(x2)
        part1 = self.conv86(route1)
        x2 = self.concat85_86([part1, part2])

        x2 = self.conv87(x2)
        x2 = self.conv88(x2)
        x2 = self.conv89(x2)
        x2 = self.conv90(x2)
        x2 = self.conv91(x2)

        pred_s = self.conv92(x2)
        pred_s = self.conv93(pred_s)

        x2 = self.conv94(x2)
        x2 = self.concat84_94([x2, x1])

        x2 = self.conv95(x2)
        x2 = self.conv96(x2)
        x2 = self.conv97(x2)
        x2 = self.conv98(x2)
        x2 = self.conv99(x2)

        pred_m = self.conv100(x2)
        pred_m = self.conv101(pred_m)

        x2 = self.conv102(x2)
        x2 = self.concat77_102([x2, route3])

        x2 = self.conv103(x2)
        x2 = self.conv104(x2)
        x2 = self.conv105(x2)
        x2 = self.conv106(x2)
        x2 = self.conv107(x2)

        pred_l = self.conv108(x2)
        pred_l = self.conv109(pred_l)

        return pred_s, pred_m, pred_l


class PANetTiny(Model):
    def __init__(
        self, num_classes, activation: str = "leaky", kernel_regularizer=None
    ):
        super(PANetTiny, self).__init__(name="PANetTiny")
        self.conv15 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv16 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv17 = YOLOConv2D(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv18 = YOLOConv2D(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.upSampling18 = layers.UpSampling2D(interpolation="bilinear")
        self.concat13_18 = layers.Concatenate(axis=-1)

        self.conv19 = YOLOConv2D(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv20 = YOLOConv2D(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, x):
        route1, route2 = x

        x1 = self.conv15(route2)

        x2 = self.conv16(x1)
        pred_l = self.conv17(x2)

        x1 = self.conv18(x1)
        x1 = self.upSampling18(x1)
        x1 = self.concat13_18([x1, route1])

        x1 = self.conv19(x1)
        pred_m = self.conv20(x1)

        return pred_m, pred_l
