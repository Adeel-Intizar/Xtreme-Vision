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


########
# Load #
########


def load_weights(model, weights_file, tiny: bool = False):
    with open(weights_file, "rb") as fd:
        # major, minor, revision, seen, _
        _np_fromfile(fd, dtype=np.int32, count=5)

        if tiny:
            ret = yolov4_tiny_load_weignts(model, fd)
        else:
            ret = yolov4_load_weights(model, fd)

        if len(fd.read()) != 0:
            raise ValueError("Model and weights file do not match.")

    return ret


def _np_fromfile(fd, dtype, count):
    data = np.fromfile(fd, dtype=dtype, count=count)
    if len(data) != count:
        if len(data) == 0:
            return None
        raise ValueError("Model and weights file do not match.")
    return data


def yolo_conv2d_load_weights(yolo_conv2d, fd):
    if yolo_conv2d.strides[0] == 1:
        conv_index = 0
    else:
        conv_index = 1
    filters = yolo_conv2d.filters

    if yolo_conv2d.activation is not None:
        bn = yolo_conv2d.sequential.get_layer(index=conv_index + 1)

        # darknet weights: [beta, gamma, mean, variance]
        bn_weights = _np_fromfile(fd, dtype=np.float32, count=4 * filters)
        if bn_weights is None:
            return False
        # tf weights: [gamma, beta, mean, variance]
        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

        bn.set_weights(bn_weights)
        conv_bias = None
    else:
        conv_bias = _np_fromfile(fd, dtype=np.float32, count=filters)
        if conv_bias is None:
            return False

    conv = yolo_conv2d.sequential.get_layer(index=conv_index)

    # darknet shape (out_dim, in_dim, height, width)
    conv_shape = (filters, yolo_conv2d.input_dim, *yolo_conv2d.kernel_size)
    conv_weights = _np_fromfile(
        fd, dtype=np.float32, count=np.product(conv_shape)
    )
    if conv_weights is None:
        return False
    # tf shape (height, width, in_dim, out_dim)
    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

    if conv_bias is None:
        conv.set_weights([conv_weights])
    else:
        conv.set_weights([conv_weights, conv_bias])

    return True


def res_block_load_weights(model, fd):
    for i in range(model.iteration):
        _res_block = model.sequential.get_layer(index=i)
        if not yolo_conv2d_load_weights(_res_block.get_layer(index=0), fd):
            return False
        if not yolo_conv2d_load_weights(_res_block.get_layer(index=1), fd):
            return False

    return True


def csp_res_net_load_weights(model, fd):
    for i in range(3):
        if not yolo_conv2d_load_weights(model.get_layer(index=i), fd):
            return False
    if not res_block_load_weights(model.get_layer(index=3), fd):
        return False
    for i in (4, 6):
        if not yolo_conv2d_load_weights(model.get_layer(index=i), fd):
            return False

    return True


def csp_darknet53_load_weights(csp_darknet53, fd):
    if not yolo_conv2d_load_weights(csp_darknet53.get_layer(index=0), fd):
        return False

    for i in range(1, 1 + 5):
        if not csp_res_net_load_weights(csp_darknet53.get_layer(index=i), fd):
            return False

    for i in range(6, 6 + 3):
        if not yolo_conv2d_load_weights(csp_darknet53.get_layer(index=i), fd):
            return False

    # index 9 is SPP

    for i in range(10, 10 + 3):
        if not yolo_conv2d_load_weights(csp_darknet53.get_layer(index=i), fd):
            return False

    return True


def csp_darknet53_tiny_load_weights(csp_darknet53_tiny, fd):
    if not yolo_conv2d_load_weights(csp_darknet53_tiny.get_layer(index=0), fd):
        return False

    for i in range(1, 15):
        layer_name = "yolo_conv2d_%d" % i

        yolo_conv2d = csp_darknet53_tiny.get_layer(layer_name)
        if not yolo_conv2d_load_weights(yolo_conv2d, fd):
            return False

    return True


def panet_load_weights(panet, fd):
    for i in range(78, 110):
        layer_name = "yolo_conv2d_%d" % i

        yolo_conv2d = panet.get_layer(layer_name)
        if not yolo_conv2d_load_weights(yolo_conv2d, fd):
            return False

    return True


def panet_tiny_load_weights(panet_tiny, fd):
    for i in range(15, 21):
        layer_name = "yolo_conv2d_%d" % i

        yolo_conv2d = panet_tiny.get_layer(layer_name)
        if not yolo_conv2d_load_weights(yolo_conv2d, fd):
            return False

    return True


def yolov4_load_weights(yolov4, fd):
    csp_darknet53 = yolov4.get_layer("CSPDarknet53")

    if not csp_darknet53_load_weights(csp_darknet53, fd):
        return False

    panet = yolov4.get_layer("PANet")

    if not panet_load_weights(panet, fd):
        return False

    return True


def yolov4_tiny_load_weignts(yolov4_tiny, fd):
    csp_darknet53_tiny = yolov4_tiny.get_layer("CSPDarknet53Tiny")

    if not csp_darknet53_tiny_load_weights(csp_darknet53_tiny, fd):
        return False

    panet_tiny = yolov4_tiny.get_layer("PANetTiny")

    if not panet_tiny_load_weights(panet_tiny, fd):
        return False

    return True


########
# Save #
########


def save_weights(model, weights_file, tiny: bool = False):
    with open(weights_file, "wb") as fd:
        # major, minor, revision, seen, _
        np.array([0, 2, 5, 32032000, 0], dtype=np.int32).tofile(fd)

        if tiny:
            yolov4_tiny_save_weignts(model, fd)
        else:
            yolov4_save_weights(model, fd)


def yolo_conv2d_save_weights(yolo_conv2d, fd):
    if yolo_conv2d.strides[0] == 1:
        conv_index = 0
    else:
        conv_index = 1

    if yolo_conv2d.activation is not None:
        bn = yolo_conv2d.sequential.get_layer(index=conv_index + 1)

        # tf weights: [gamma, beta, mean, variance]
        bn_weights = np.stack(bn.get_weights())
        # darknet weights: [beta, gamma, mean, variance]
        bn_weights[[1, 0, 2, 3]].reshape((-1,)).tofile(fd)

        conv_bias = False
    else:
        conv_bias = True

    conv = yolo_conv2d.sequential.get_layer(index=conv_index)

    # tf shape (height, width, in_dim, out_dim)
    if conv_bias:
        conv_weights, conv_bias = conv.get_weights()
        conv_bias.tofile(fd)
    else:
        conv_weights = conv.get_weights()[0]

    # darknet shape (out_dim, in_dim, height, width)
    conv_weights.transpose([3, 2, 0, 1]).reshape((-1,)).tofile(fd)


def res_block_save_weights(model, fd):
    for i in range(model.iteration):
        _res_block = model.sequential.get_layer(index=i)
        yolo_conv2d_save_weights(_res_block.get_layer(index=0), fd)
        yolo_conv2d_save_weights(_res_block.get_layer(index=1), fd)


def csp_res_net_save_weights(model, fd):
    for i in range(3):
        yolo_conv2d_save_weights(model.get_layer(index=i), fd)

    res_block_save_weights(model.get_layer(index=3), fd)

    for i in (4, 6):
        yolo_conv2d_save_weights(model.get_layer(index=i), fd)


def csp_darknet53_save_weights(csp_darknet53, fd):
    yolo_conv2d_save_weights(csp_darknet53.get_layer(index=0), fd)

    for i in range(1, 1 + 5):
        csp_res_net_save_weights(csp_darknet53.get_layer(index=i), fd)

    for i in range(6, 6 + 3):
        yolo_conv2d_save_weights(csp_darknet53.get_layer(index=i), fd)

    # index 9 is SPP

    for i in range(10, 10 + 3):
        yolo_conv2d_save_weights(csp_darknet53.get_layer(index=i), fd)


def csp_darknet53_tiny_save_weights(csp_darknet53_tiny, fd):
    yolo_conv2d_save_weights(csp_darknet53_tiny.get_layer(index=0), fd)

    for i in range(1, 15):
        layer_name = "yolo_conv2d_%d" % i

        yolo_conv2d = csp_darknet53_tiny.get_layer(layer_name)
        yolo_conv2d_save_weights(yolo_conv2d, fd)


def panet_save_weights(panet, fd):
    for i in range(78, 110):
        layer_name = "yolo_conv2d_%d" % i

        yolo_conv2d = panet.get_layer(layer_name)
        yolo_conv2d_save_weights(yolo_conv2d, fd)


def panet_tiny_save_weights(panet_tiny, fd):
    for i in range(15, 21):
        layer_name = "yolo_conv2d_%d" % i

        yolo_conv2d = panet_tiny.get_layer(layer_name)
        yolo_conv2d_save_weights(yolo_conv2d, fd)


def yolov4_save_weights(yolov4, fd):
    csp_darknet53 = yolov4.get_layer("CSPDarknet53")
    csp_darknet53_save_weights(csp_darknet53, fd)

    panet = yolov4.get_layer("PANet")
    panet_save_weights(panet, fd)


def yolov4_tiny_save_weignts(yolov4_tiny, fd):
    csp_darknet53_tiny = yolov4_tiny.get_layer("CSPDarknet53Tiny")
    csp_darknet53_tiny_save_weights(csp_darknet53_tiny, fd)

    panet_tiny = yolov4_tiny.get_layer("PANetTiny")
    panet_tiny_save_weights(panet_tiny, fd)
