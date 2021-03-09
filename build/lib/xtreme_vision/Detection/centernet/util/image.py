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
import PIL.Image
import PIL.ImageDraw
import PIL.ImageColor
import numpy as np
import tensorflow as tf
import cv2


def draw_bounding_boxes(im: PIL.Image, bboxes: np.ndarray, classes: np.ndarray,
                        scores: np.ndarray, custom_objects: dict = None):
    im = im.copy()
    num_classes = len(set(classes))
    class_to_color_id = {cls: i for i, cls in enumerate(set(classes))}

    colors = [PIL.ImageColor.getrgb(f'hsv({int(360 * x / num_classes)},100%,100%)') for x in range(num_classes)]

    draw = PIL.ImageDraw.Draw(im, 'RGBA')
    for index, (bbox, cls, score) in enumerate(zip(bboxes, classes, scores)):
        
        if (custom_objects != None):
          check = custom_objects.get(cls, 'invalid')
          if (check == "invalid"):
              continue

        color = colors[class_to_color_id[cls]]
        draw.rectangle((*bbox.astype(np.int64),), fill=color+(100,), outline=color)

        text = f'{cls}: {int(100 * score)}%'
        text_w, text_h = draw.textsize(text)
        draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=color, outline=color)
        draw.text((bbox[0], bbox[1]), text, fill=(0, 0, 0))

    return im
    
def draw_keypoints(img, kps):
    """Draw the pose like https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/debugger.py#L191
    Arguments
      img: uint8 BGR
      kps: (17, 2) keypoint [[x, y]] coordinates
    """
    
    colors = [PIL.ImageColor.getrgb(f'hsv({int(360 * x / 17)},100%,100%)') for x in range(18)]

    edges = [[0, 1], [0, 2], [1, 3], [2, 4],
              [3, 5], [4, 6], [5, 6],
              [5, 7], [7, 9], [6, 8], [8, 10],
              [5, 11], [6, 12], [11, 12],
              [11, 13], [13, 15], [12, 14], [14, 16]]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for i in range(kps.shape[0]):
      kp = np.array(kps[i], dtype=np.int32).reshape(17, 2)
      for j in range(17):
        cv2.circle(img, (kp[j, 0], kp[j, 1]), 3, colors[j], -1)
      for j, e in enumerate(edges):
        if kp[e].min() > 0:
          cv2.line(img, (kp[e[0], 0], kp[e[0], 1]), (kp[e[1], 0], kp[e[1], 1]), colors[j], 2,
                   lineType=cv2.LINE_AA)
    return img


def apply_exif_orientation(img: PIL.Image) -> PIL.Image:
    methods = {
        1: tuple(),
        2: (PIL.Image.FLIP_LEFT_RIGHT,),
        3: (PIL.Image.ROTATE_180,),
        4: (PIL.Image.FLIP_TOP_BOTTOM,),
        5: (PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.ROTATE_90),
        6: (PIL.Image.ROTATE_270,),
        7: (PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.ROTATE_270),
        8: (PIL.Image.ROTATE_90,),
    }

    exif = img._getexif()

    if exif is None:
        return img

    for method in methods[exif.get(0x112, 1)]:
        img = img.transpose(method)

    return img


def heatmap_non_max_surpression(heatmap: tf.Tensor) -> tf.Tensor:
    return tf.dtypes.cast(heatmap == tf.nn.max_pool2d(heatmap, 3, 1, 'SAME'), tf.float32) * heatmap
