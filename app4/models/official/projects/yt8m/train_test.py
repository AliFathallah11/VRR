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

import json
import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.yt8m import train as train_lib
from official.projects.yt8m.dataloaders import utils
from official.vision.dataloaders import tfexample_utils

FLAGS = flags.FLAGS


class TrainTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._model_dir = os.path.join(self.get_temp_dir(), 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)

    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    examples = [utils.make_yt8m_example() for _ in range(8)]
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  @parameterized.named_parameters(
      dict(
          testcase_name='segment_with_avg_precison',
          use_segment_level_labels=True,
          use_average_precision_metric=True,
          num_sample_frames=24,
      ),
      dict(
          testcase_name='video_with_avg_precison',
          use_segment_level_labels=False,
          use_average_precision_metric=True,
          num_sample_frames=24,
      ),
      dict(
          testcase_name='segment',
          use_segment_level_labels=True,
          use_average_precision_metric=False,
          num_sample_frames=24,
      ),
      dict(
          testcase_name='video',
          use_segment_level_labels=False,
          use_average_precision_metric=False,
          num_sample_frames=24,
      ),
      dict(
          testcase_name='segment_without_sampling_frames',
          use_segment_level_labels=True,
          use_average_precision_metric=False,
          num_sample_frames=None,
      ),
      dict(
          testcase_name='video_without_sampling_frames',
          use_segment_level_labels=False,
          use_average_precision_metric=False,
          num_sample_frames=None,
      ),
  )
  def test_train_and_eval(
      self,
      use_segment_level_labels,
      use_average_precision_metric,
      num_sample_frames,
  ):
    saved_flag_values = flagsaver.save_flag_values()
    train_lib.tfm_flags.define_flags()
    FLAGS.mode = 'train'
    FLAGS.model_dir = self._model_dir
    FLAGS.experiment = 'yt8m_experiment'
    FLAGS.tpu = ''

    average_precision = {'top_k': 20} if use_average_precision_metric else None
    params_override = json.dumps({
        'runtime': {
            'distribution_strategy': 'mirrored',
            'mixed_precision_dtype': 'float32',
        },
        'trainer': {
            'train_steps': 2,
            'validation_steps': 2,
        },
        'task': {
            'model': {
                'backbone': {
                    'type': 'dbof',
                    'dbof': {
                        'cluster_size': 16,
                        'hidden_size': 16,
                        'use_context_gate_cluster_layer': True,
                    },
                },
                'head': {
                    'type': 'moe',
                    'moe': {
                        'use_input_context_gate': True,
                        'use_output_context_gate': True,
                    },
                },
            },
            'train_data': {
                'input_path': self._data_path,
                'global_batch_size': 4,
                'num_sample_frames': num_sample_frames,
            },
            'validation_data': {
                'input_path': self._data_path,
                'segment_labels': use_segment_level_labels,
                'global_batch_size': 4,
                'num_sample_frames': num_sample_frames,
            },
            'evaluation': {
                'average_precision': average_precision,
            },
        },
    })
    FLAGS.params_override = params_override

    with train_lib.train.gin.unlock_config():
      train_lib.train.main('unused_args')

    FLAGS.mode = 'eval'
    with train_lib.train.gin.unlock_config():
      train_lib.train.main('unused_args')

    flagsaver.restore_flag_values(saved_flag_values)


if __name__ == '__main__':
  tf.config.set_soft_device_placement(True)
  tf.test.main()
