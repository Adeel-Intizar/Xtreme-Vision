
from xtreme_vision.Detection import retinanet
from . import backend

import tensorflow as tf

def mask(iou_threshold=0.5, mask_size=(28, 28), parallel_iterations=32):
    def _mask_conditional(y_true, y_pred):
        # if there are no masks annotations, return 0; else, compute the masks loss
        loss = backend.cond(
            tf.keras.backend.any(tf.keras.backend.equal(tf.keras.backend.shape(y_true), 0)),
            lambda: tf.keras.backend.cast_to_floatx(0.0),
            lambda: _mask_batch(y_true, y_pred, iou_threshold=iou_threshold, mask_size=mask_size, parallel_iterations=parallel_iterations)
        )
        return loss

    def _mask_batch(y_true, y_pred, iou_threshold=0.5, mask_size=(28, 28), parallel_iterations=32):
        # split up the different predicted blobs
        boxes = y_pred[:, :, :4]
        masks = y_pred[:, :, 4:]

        # split up the different blobs
        annotations  = y_true[:, :, :5]
        width        = tf.keras.backend.cast(y_true[0, 0, 5], dtype='int32')
        height       = tf.keras.backend.cast(y_true[0, 0, 6], dtype='int32')
        masks_target = y_true[:, :, 7:]

        # reshape the masks back to their original size
        masks_target = tf.keras.backend.reshape(masks_target, (tf.keras.backend.shape(masks_target)[0], tf.keras.backend.shape(masks_target)[1], height, width))
        masks        = tf.keras.backend.reshape(masks, (tf.keras.backend.shape(masks)[0], tf.keras.backend.shape(masks)[1], mask_size[0], mask_size[1], -1))

        def _mask(args):
            boxes = args[0]
            masks = args[1]
            annotations = args[2]
            masks_target = args[3]

            return compute_mask_loss(
                boxes,
                masks,
                annotations,
                masks_target,
                width,
                height,
                iou_threshold = iou_threshold,
                mask_size     = mask_size,
            )

        mask_batch_loss = retinanet.backend.map_fn(
            _mask,
            elems=[boxes, masks, annotations, masks_target],
            dtype=tf.keras.backend.floatx(),
            parallel_iterations=parallel_iterations
        )

        return tf.keras.backend.mean(mask_batch_loss)

    return _mask_conditional


def compute_mask_loss(
    boxes,
    masks,
    annotations,
    masks_target,
    width,
    height,
    iou_threshold=0.5,
    mask_size=(28, 28)
):
    # compute overlap of boxes with annotations
    iou                  = backend.overlap(boxes, annotations)
    argmax_overlaps_inds = tf.keras.backend.argmax(iou, axis=1)
    max_iou              = tf.keras.backend.max(iou, axis=1)

    # filter those with IoU > 0.5
    indices              = tf.where(tf.keras.backend.greater_equal(max_iou, iou_threshold))
    boxes                = tf.gather_nd(boxes, indices)
    masks                = tf.gather_nd(masks, indices)
    argmax_overlaps_inds = tf.keras.backend.cast(tf.gather_nd(argmax_overlaps_inds, indices), 'int32')
    labels               = tf.keras.backend.cast(tf.keras.backend.gather(annotations[:, 4], argmax_overlaps_inds), 'int32')

    # make normalized boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    boxes = tf.keras.backend.stack([
        y1 / (tf.keras.backend.cast(height, dtype=tf.keras.backend.floatx()) - 1),
        x1 / (tf.keras.backend.cast(width, dtype=tf.keras.backend.floatx()) - 1),
        (y2 - 1) / (tf.keras.backend.cast(height, dtype=tf.keras.backend.floatx()) - 1),
        (x2 - 1) / (tf.keras.backend.cast(width, dtype=tf.keras.backend.floatx()) - 1),
    ], axis=1)

    # crop and resize masks_target
    masks_target = tf.keras.backend.expand_dims(masks_target, axis=3)  # append a fake channel dimension
    masks_target = backend.crop_and_resize(
        masks_target,
        boxes,
        argmax_overlaps_inds,
        mask_size
    )
    masks_target = masks_target[:, :, :, 0]  # remove fake channel dimension

    # gather the predicted masks using the annotation label
    masks = backend.transpose(masks, (0, 3, 1, 2))
    label_indices = tf.keras.backend.stack([
        tf.keras.backend.arange(tf.keras.backend.shape(labels)[0]),
        labels
    ], axis=1)
    masks = tf.gather_nd(masks, label_indices)

    # compute mask loss
    mask_loss  = tf.keras.backend.binary_crossentropy(masks_target, masks)
    normalizer = tf.keras.backend.shape(masks)[0] * tf.keras.backend.shape(masks)[1] * tf.keras.backend.shape(masks)[2]
    normalizer = tf.keras.backend.maximum(tf.keras.backend.cast(normalizer, tf.keras.backend.floatx()), 1)
    mask_loss  = tf.keras.backend.sum(mask_loss) / normalizer

    return mask_loss
