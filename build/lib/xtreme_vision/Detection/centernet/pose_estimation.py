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
import numpy as np
import PIL.Image
import cv2

from xtreme_vision.Detection.centernet.model.pose_estimation import PoseEstimationModel
from xtreme_vision.Detection.centernet import util
from xtreme_vision.Detection.centernet.version import VERSION


class PoseEstimation:
    def __init__(self, num_joints: int = 17):
        self.mean = np.array([[[0.408, 0.447, 0.470]]], dtype=np.float32)
        self.std = np.array([[[0.289, 0.274, 0.278]]], dtype=np.float32)
        self.k = 100
        self.score_threshold = 0.3
        self.input_size = 512

        self.num_joints = num_joints

        self.num_joints_arange = None
        self.model = None

        self.init_model()

    def init_model(self):
        self.num_joints_arange = np.arange(self.num_joints)[:, np.newaxis]
        self.model = PoseEstimationModel(self.num_joints)
        self.model(tf.keras.Input((self.input_size, self.input_size, 3)))

    def load_model(self, weights_path: str = None):
        if weights_path is None:
            base_url = f'https://github.com/Licht-T/tf-centernet/releases/download/{VERSION}'
            if self.num_joints == 17:
                weights_path = tf.keras.utils.get_file(
                    f'centernet_pretrained_pose_{VERSION}.h5',
                    f'{base_url}/centernet_pretrained_pose.h5',
                    cache_subdir='tf-centernet'
                )
            else:
                raise RuntimeError('weights_path should not be None.')

        self.model.load_weights(weights_path)

    def predict(self, img:np.ndarray, output_path:str, debug=True):

        # img = np.array(PIL.Image.open(input_path))[..., ::-1]
        
        orig_wh = np.array(img.shape[:2])[::-1]
        resize_factor = self.input_size / orig_wh.max()
        centering = (self.input_size - orig_wh * resize_factor) / 2

        input_img = tf.image.resize_with_pad(img, self.input_size, self.input_size)
        input_img = (tf.dtypes.cast(input_img, tf.float32) / tf.constant(255, tf.float32) - self.mean) / self.std
        input_img = input_img[tf.newaxis, ...]

        predicted, _ = self.model(input_img)

        joint_heatmap, joint_locations, joint_offsets, heatmap, offsets, whs = predicted

        joint_heatmap = util.image.heatmap_non_max_surpression(joint_heatmap)
        heatmap = util.image.heatmap_non_max_surpression(heatmap)

        joint_heatmap = np.squeeze(joint_heatmap.numpy())
        joint_locations = np.squeeze(joint_locations.numpy())
        joint_offsets = np.squeeze(joint_offsets.numpy())
        heatmap = np.squeeze(heatmap.numpy())
        offsets = np.squeeze(offsets.numpy())
        whs = np.squeeze(whs.numpy())

        idx = heatmap.flatten().argsort()[::-1][:self.k]
        scores = heatmap.flatten()[idx]
        idx = idx[scores > self.score_threshold]
        scores = scores[scores > self.score_threshold]

        rows, cols = np.unravel_index(idx, heatmap.shape)

        xys = np.concatenate([cols[..., np.newaxis], rows[..., np.newaxis]], axis=-1)
        keypoints = joint_locations[rows, cols].reshape((-1, self.num_joints, 2)) + xys[:, np.newaxis, :]
        keypoints = keypoints.transpose([1, 0, 2])

        xys = xys + offsets[rows, cols]
        boxes = np.concatenate([xys - whs[rows, cols]/2, xys + whs[rows, cols]/2], axis=1).reshape((-1, 2, 2))

        joint_idx = joint_heatmap.transpose([2, 0, 1]).reshape((self.num_joints, -1)).argsort(1)[:, ::-1][:, :self.k]
        joint_rows, joint_cols = np.unravel_index(joint_idx, joint_heatmap.shape[:2])
        joint_classes = np.broadcast_to(self.num_joints_arange, (self.num_joints, self.k))
        joint_scores = joint_heatmap[joint_rows, joint_cols, joint_classes]

        joint_xys = np.concatenate([joint_cols[..., np.newaxis], joint_rows[..., np.newaxis]], axis=-1)
        joint_xys = joint_xys + joint_offsets[joint_rows, joint_cols]
        joint_xys[0.1 >= joint_scores] = np.inf  # CxKx2

        joint_xys_matrix = np.tile(joint_xys[:, np.newaxis, :, :], (1, keypoints.shape[1], 1, 1))
        box_upperlefts = boxes[np.newaxis, :, np.newaxis, 0, :]
        box_lowerrights = boxes[np.newaxis, :, np.newaxis, 1, :]
        joint_xys_matrix[((joint_xys_matrix < box_upperlefts) | (box_lowerrights < joint_xys_matrix)).any(-1)] = np.inf

        distance_matrix = ((keypoints[:, :, np.newaxis, :] - joint_xys_matrix) ** 2).sum(axis=-1) ** 0.5  # CxJxK
        nearest_joint_for_keypoints = distance_matrix.argmin(-1)  # CxJ
        nearest_joints_conditions = np.isfinite(distance_matrix.min(-1))
        keypoints[nearest_joints_conditions] = \
            joint_xys[self.num_joints_arange, nearest_joint_for_keypoints, :][nearest_joints_conditions]

        boxes = ((self.input_size / heatmap.shape[0]) * boxes - centering) / resize_factor
        boxes = boxes.reshape((-1, 4))

        keypoints = ((self.input_size / heatmap.shape[0]) * keypoints - centering) / resize_factor
        keypoints = keypoints.transpose([1, 0, 2])

        im = PIL.Image.fromarray(img[..., ::-1])
        im = util.image.draw_keypoints(img = im, kps = keypoints)
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        if debug:
            cv2.imwrite(output_path, im)

        return im
