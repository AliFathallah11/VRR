# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for iou_similarity.py."""

import tensorflow as tf, tf_keras

from official.vision.ops import iou_similarity


class BoxMatcherTest(tf.test.TestCase):

  def test_similarity_unbatched(self):
    boxes = tf.constant(
        [
            [0, 0, 1, 1],
            [5, 0, 10, 5],
        ],
        dtype=tf.float32)

    gt_boxes = tf.constant(
        [
            [0, 0, 5, 5],
            [0, 5, 5, 10],
            [5, 0, 10, 5],
            [5, 5, 10, 10],
        ],
        dtype=tf.float32)

    sim_calc = iou_similarity.IouSimilarity()
    sim_matrix = sim_calc(boxes, gt_boxes)

    self.assertAllClose(
        sim_matrix.numpy(),
        [[0.04, 0, 0, 0],
         [0, 0, 1., 0]])

  def test_similarity_batched(self):
    boxes = tf.constant(
        [[
            [0, 0, 1, 1],
            [5, 0, 10, 5],
        ]],
        dtype=tf.float32)

    gt_boxes = tf.constant(
        [[
            [0, 0, 5, 5],
            [0, 5, 5, 10],
            [5, 0, 10, 5],
            [5, 5, 10, 10],
        ]],
        dtype=tf.float32)

    sim_calc = iou_similarity.IouSimilarity()
    sim_matrix = sim_calc(boxes, gt_boxes)

    self.assertAllClose(
        sim_matrix.numpy(),
        [[[0.04, 0, 0, 0],
          [0, 0, 1., 0]]])


if __name__ == '__main__':
  tf.test.main()
