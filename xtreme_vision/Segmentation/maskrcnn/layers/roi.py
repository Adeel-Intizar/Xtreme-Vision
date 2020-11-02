# import keras.backend
# import keras.layers
from xtreme_vision.Detection import retinanet
from .. import backend

import tensorflow as tf


class RoiAlign(tf.keras.layers.Layer):
    def __init__(self, crop_size=(14, 14), parallel_iterations=32, **kwargs):
        self.crop_size = crop_size
        self.parallel_iterations = parallel_iterations

        super(RoiAlign, self).__init__(**kwargs)

    def map_to_level(self, boxes, canonical_size=224, canonical_level=1, min_level=0, max_level=4):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        w = x2 - x1
        h = y2 - y1

        size = tf.keras.backend.sqrt(w * h)

        levels = backend.floor(canonical_level + backend.log2(size / canonical_size + tf.keras.backend.epsilon()))
        levels = tf.keras.backend.clip(levels, min_level, max_level)

        return levels

    def call(self, inputs, **kwargs):
        image_shape = tf.keras.backend.cast(inputs[0], tf.keras.backend.floatx())
        boxes       = tf.keras.backend.stop_gradient(inputs[1])
        scores      = tf.keras.backend.stop_gradient(inputs[2])
        fpn         = [tf.keras.backend.stop_gradient(i) for i in inputs[3:]]

        def _roi_align(args):
            boxes  = args[0]
            scores = args[1]
            fpn    = args[2]

            # compute from which level to get features from
            target_levels = self.map_to_level(boxes)

            # process each pyramid independently
            rois           = []
            ordered_indices = []
            for i in range(len(fpn)):
                # select the boxes and classification from this pyramid level
                indices = tf.where(tf.keras.backend.equal(target_levels, i))
                ordered_indices.append(indices)

                level_boxes = tf.gather_nd(boxes, indices)
                fpn_shape   = tf.keras.backend.cast(tf.keras.backend.shape(fpn[i]), dtype=tf.keras.backend.floatx())

                # convert to expected format for crop_and_resize
                x1 = level_boxes[:, 0]
                y1 = level_boxes[:, 1]
                x2 = level_boxes[:, 2]
                y2 = level_boxes[:, 3]
                level_boxes = tf.keras.backend.stack([
                    (y1 / image_shape[1] * fpn_shape[0]) / (fpn_shape[0] - 1),
                    (x1 / image_shape[2] * fpn_shape[1]) / (fpn_shape[1] - 1),
                    (y2 / image_shape[1] * fpn_shape[0] - 1) / (fpn_shape[0] - 1),
                    (x2 / image_shape[2] * fpn_shape[1] - 1) / (fpn_shape[1] - 1),
                ], axis=1)

                # append the rois to the list of rois
                rois.append(backend.crop_and_resize(
                    tf.keras.backend.expand_dims(fpn[i], axis=0),
                    level_boxes,
                    tf.zeros((tf.keras.backend.shape(level_boxes)[0],), dtype='int32'),  # TODO: Remove this workaround (https://github.com/tensorflow/tensorflow/issues/33787).
                    self.crop_size
                ))

            # concatenate rois to one blob
            rois = tf.keras.backend.concatenate(rois, axis=0)

            # reorder rois back to original order
            indices = tf.keras.backend.concatenate(ordered_indices, axis=0)
            rois    = tf.scatter_nd(indices, rois, tf.keras.backend.cast(tf.keras.backend.shape(rois), 'int64'))

            return rois

        roi_batch = retinanet.backend.map_output(
            _roi_align,
            elems=[boxes, scores, fpn],
            dtype=tf.keras.backend.floatx(),
            parallel_iterations=self.parallel_iterations
        )

        return roi_batch

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], None, self.crop_size[0], self.crop_size[1], input_shape[3][-1])

    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config.update({
            'crop_size' : self.crop_size,
        })

        return config
