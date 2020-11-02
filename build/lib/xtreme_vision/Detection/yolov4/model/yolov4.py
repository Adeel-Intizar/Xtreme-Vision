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
from tensorflow.keras import Model

from xtreme_vision.Detection.yolov4.model.backbone import CSPDarknet53, CSPDarknet53Tiny
from xtreme_vision.Detection.yolov4.model.head import YOLOv3Head, YOLOv3HeadTiny
from xtreme_vision.Detection.yolov4.model.neck import PANet, PANetTiny


class YOLOv4(Model):
    """
    Path Aggregation Network(PAN)
    Spatial Attention Module(SAM)
    Bounding Box(BBox)
    """

    def __init__(
        self,
        anchors,
        num_classes: int,
        xyscales,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer=None,
    ):
        super(YOLOv4, self).__init__(name="YOLOv4")
        self.csp_darknet53 = CSPDarknet53(
            activation0=activation0,
            activation1=activation1,
            kernel_regularizer=kernel_regularizer,
        )
        self.panet = PANet(
            num_classes=num_classes,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )
        self.yolov3_head = YOLOv3Head(
            anchors=anchors, num_classes=num_classes, xysclaes=xyscales
        )

    def call(self, x):
        """
        @param x: Dim(batch, input_size, input_size, channels)
            The element has a value between 0.0 and 1.0.

        @return (s_pred, m_pred, l_pred)
            downsampling_size = [8, 16, 32]
            output_size = input_size // downsampling_szie
            s_pred = Dim(batch, output_size[0], output_size[0],
                                        num_anchors, (5 + num_classes))
            l_pred = Dim(batch, output_size[1], output_size[1],
                                        num_anchors, (5 + num_classes))
            m_pred = Dim(batch, output_size[2], output_size[2],
                                        num_anchors, (5 + num_classes))

        Ref: https://arxiv.org/abs/1612.08242 - YOLOv2

        5 + num_classes = (t_x, t_y, t_w, t_h, t_o, c_0, c_1, ...)
                            => (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...)

        A top-left coordinate of a grid is (c_x, c_y).
        A dimension prior is (p_w, p_h).(== anchor size)
                [[[12, 16],   [19, 36],   [40, 28]  ],
                 [[36, 75],   [76, 55],   [72, 146] ],
                 [[142, 110], [192, 243], [459, 401]]]
        Pr == Probability.

        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * exp(t_w)
        b_h = p_h * exp(t_h)
        sigmoid(t_o) == confidence == Pr(Object) ∗ IoU(b, Object)
        sigmoid(c_i) == conditional class probability == Pr(Class_i|Object)
        """
        x = self.csp_darknet53(x)
        x = self.panet(x)
        x = self.yolov3_head(x)
        return x


class YOLOv4Tiny(Model):
    def __init__(
        self,
        anchors,
        num_classes: int,
        xyscales,
        activation: str = "leaky",
        kernel_regularizer=None,
    ):
        super(YOLOv4Tiny, self).__init__(name="YOLOv4Tiny")
        self.csp_darknet53_tiny = CSPDarknet53Tiny(
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.panet_tiny = PANetTiny(
            num_classes=num_classes,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.yolov3_head_tiny = YOLOv3HeadTiny(
            anchors=anchors, num_classes=num_classes, xysclaes=xyscales
        )

    def call(self, x):
        x = self.csp_darknet53_tiny(x)
        x = self.panet_tiny(x)
        x = self.yolov3_head_tiny(x)
        return x
