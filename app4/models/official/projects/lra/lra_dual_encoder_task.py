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

"""Trainer network for dual encoder style models."""
# pylint: disable=g-classes-have-attributes
import dataclasses
from typing import List, Union, Optional

from absl import logging
import numpy as np
import orbit
from scipy import stats
from sklearn import metrics as sklearn_metrics
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.tasks import utils

from official.projects.lra import lra_dual_encoder

METRIC_TYPES = frozenset(
    ['accuracy', 'f1', 'matthews_corrcoef', 'pearson_spearman_corr']
)


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A classifier/regressor configuration."""

  num_classes: int = 2
  use_encoder_pooler: bool = False
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  max_seq_length: int = 512


@dataclasses.dataclass
class DualEncoderConfig(cfg.TaskConfig):
  """The model config."""

  # At most one of `init_checkpoint` and `hub_module_url` can
  # be specified.
  init_checkpoint: str = ''
  init_cls_pooler: bool = False
  hub_module_url: str = ''
  metric_type: str = 'accuracy'
  # Defines the concrete model config at instantiation time.
  model: ModelConfig = ModelConfig()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@task_factory.register_task_cls(DualEncoderConfig)
class DualEncoderTask(base_task.Task):
  """Task object for DualEncoderTask."""

  def __init__(self, params: cfg.TaskConfig, logging_dir=None, name=None):
    super().__init__(params, logging_dir, name=name)
    if params.metric_type not in METRIC_TYPES:
      raise ValueError('Invalid metric_type: {}'.format(params.metric_type))
    self.metric_type = params.metric_type
    if hasattr(params.train_data, 'label_field'):
      self.label_field = params.train_data.label_field
    else:
      self.label_field = 'label_ids'

  def build_model(self):
    if self.task_config.hub_module_url and self.task_config.init_checkpoint:
      raise ValueError(
          'At most one of `hub_module_url` and '
          '`init_checkpoint` can be specified.'
      )
    if self.task_config.hub_module_url:
      encoder_network = utils.get_encoder_from_hub(
          self.task_config.hub_module_url
      )
    else:
      encoder_network = encoders.build_encoder(self.task_config.model.encoder)
    encoder_cfg = self.task_config.model.encoder.get()

    return lra_dual_encoder.LRADualEncoder(
        network=encoder_network,
        max_seq_length=self.task_config.model.max_seq_length,
        num_classes=self.task_config.model.num_classes,
        initializer=tf_keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range
        ),
        use_encoder_pooler=self.task_config.model.use_encoder_pooler,
        inner_dim=encoder_cfg.hidden_size * 2,
    )

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    label_ids = labels[self.label_field]
    if self.task_config.model.num_classes == 1:
      loss = tf_keras.losses.mean_squared_error(label_ids, model_outputs)
    else:
      loss = tf_keras.losses.sparse_categorical_crossentropy(
          label_ids, tf.cast(model_outputs, tf.float32), from_logits=True
      )

    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf_utils.safe_mean(loss)

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':

      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        x = dict(
            left_word_ids=dummy_ids,
            left_mask=dummy_ids,
            right_word_ids=dummy_ids,
            right_mask=dummy_ids,
        )

        if self.task_config.model.num_classes == 1:
          y = tf.zeros((1,), dtype=tf.float32)
        else:
          y = tf.zeros((1, 1), dtype=tf.int32)
        x[self.label_field] = y
        return x

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
      )
      return dataset

    return data_loader_factory.get_data_loader(params).load(input_context)

  def build_metrics(self, training=None):
    del training
    if self.task_config.model.num_classes == 1:
      metrics = [tf_keras.metrics.MeanSquaredError()]
    elif self.task_config.model.num_classes == 2:
      metrics = [
          tf_keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy'),
          tf_keras.metrics.AUC(name='auc', curve='PR'),
      ]
    else:
      metrics = [
          tf_keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy'),
      ]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    for metric in metrics:
      if metric.name == 'auc':
        # Convert the logit to probability and extract the probability of True..
        metric.update_state(
            labels[self.label_field],
            tf.expand_dims(tf.nn.softmax(model_outputs)[:, 1], axis=1),
        )
      if metric.name == 'cls_accuracy':
        metric.update_state(labels[self.label_field], model_outputs)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    compiled_metrics.update_state(labels[self.label_field], model_outputs)

  def validation_step(self, inputs, model: tf_keras.Model, metrics=None):
    features, labels = inputs, inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
        labels=labels, model_outputs=outputs, aux_losses=model.losses
    )
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    if model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics or []})
      logs.update({m.name: m.result() for m in model.metrics})
    if self.metric_type == 'matthews_corrcoef':
      logs.update({
          'sentence_prediction': (
              tf.expand_dims(  # Ensure one prediction along batch dimension.
                  tf.math.argmax(outputs, axis=1), axis=1
              )
          ),
          'labels': labels[self.label_field],
      })
    else:
      logs.update({
          'sentence_prediction': outputs,
          'labels': labels[self.label_field],
      })
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if self.metric_type == 'accuracy':
      return None
    if state is None:
      state = {'sentence_prediction': [], 'labels': []}
    state['sentence_prediction'].append(
        np.concatenate(
            [v.numpy() for v in step_outputs['sentence_prediction']], axis=0
        )
    )
    state['labels'].append(
        np.concatenate([v.numpy() for v in step_outputs['labels']], axis=0)
    )
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    if self.metric_type == 'accuracy':
      return None

    preds = np.concatenate(aggregated_logs['sentence_prediction'], axis=0)
    labels = np.concatenate(aggregated_logs['labels'], axis=0)
    if self.metric_type == 'f1':
      preds = np.argmax(preds, axis=1)
      return {self.metric_type: sklearn_metrics.f1_score(labels, preds)}
    elif self.metric_type == 'matthews_corrcoef':
      preds = np.reshape(preds, -1)
      labels = np.reshape(labels, -1)
      return {
          self.metric_type: sklearn_metrics.matthews_corrcoef(preds, labels)
      }
    elif self.metric_type == 'pearson_spearman_corr':
      preds = np.reshape(preds, -1)
      labels = np.reshape(labels, -1)
      pearson_corr = stats.pearsonr(preds, labels)[0]
      spearman_corr = stats.spearmanr(preds, labels)[0]
      corr_metric = (pearson_corr + spearman_corr) / 2
      return {self.metric_type: corr_metric}

  def initialize(self, model):
    """Load a pretrained checkpoint (if exists) and then train from iter 0."""
    ckpt_dir_or_file = self.task_config.init_checkpoint
    logging.info(
        'Trying to load pretrained checkpoint from %s', ckpt_dir_or_file
    )
    if ckpt_dir_or_file and tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      logging.info(
          'No checkpoint file found from %s. Will not load.', ckpt_dir_or_file
      )
      return

    pretrain2finetune_mapping = {
        'encoder': model.checkpoint_items['encoder'],
    }
    if self.task_config.init_cls_pooler:
      # This option is valid when use_encoder_pooler is false.
      pretrain2finetune_mapping['next_sentence.pooler_dense'] = (
          model.checkpoint_items['sentence_prediction.pooler_dense']
      )
    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info(
        'Finished loading pretrained checkpoint from %s', ckpt_dir_or_file
    )


def predict(
    task: DualEncoderTask,
    params: cfg.DataConfig,
    model: tf_keras.Model,
    params_aug: Optional[cfg.DataConfig] = None,
    test_time_aug_wgt: float = 0.3,
) -> List[Union[int, float]]:
  """Predicts on the input data.

  Args:
    task: A `DualEncoderTask` object.
    params: A `cfg.DataConfig` object.
    model: A keras.Model.
    params_aug: A `cfg.DataConfig` object for augmented data.
    test_time_aug_wgt: Test time augmentation weight. The prediction score will
      use (1. - test_time_aug_wgt) original prediction plus test_time_aug_wgt
      augmented prediction.

  Returns:
    A list of predictions with length of `num_examples`. For regression task,
      each element in the list is the predicted score; for classification task,
      each element is the predicted class id.
  """

  def predict_step(inputs):
    """Replicated prediction calculation."""
    x = inputs
    example_id = x.pop('example_id')
    outputs = task.inference_step(x, model)
    return dict(example_id=example_id, predictions=outputs)

  def aggregate_fn(state, outputs):
    """Concatenates model's outputs."""
    if state is None:
      state = []

    for per_replica_example_id, per_replica_batch_predictions in zip(
        outputs['example_id'], outputs['predictions']
    ):
      state.extend(zip(per_replica_example_id, per_replica_batch_predictions))
    return state

  dataset = orbit.utils.make_distributed_dataset(
      tf.distribute.get_strategy(), task.build_inputs, params
  )
  outputs = utils.predict(predict_step, aggregate_fn, dataset)

  # When running on TPU POD, the order of output cannot be maintained,
  # so we need to sort by example_id.
  outputs = sorted(outputs, key=lambda x: x[0])
  is_regression = task.task_config.model.num_classes == 1
  if params_aug is not None:
    dataset_aug = orbit.utils.make_distributed_dataset(
        tf.distribute.get_strategy(), task.build_inputs, params_aug
    )
    outputs_aug = utils.predict(predict_step, aggregate_fn, dataset_aug)
    outputs_aug = sorted(outputs_aug, key=lambda x: x[0])
    if is_regression:
      return [
          (1.0 - test_time_aug_wgt) * x[1] + test_time_aug_wgt * y[1]
          for x, y in zip(outputs, outputs_aug)
      ]
    else:
      return [
          tf.argmax(
              (1.0 - test_time_aug_wgt) * x[1] + test_time_aug_wgt * y[1],
              axis=-1,
          )
          for x, y in zip(outputs, outputs_aug)
      ]
  if is_regression:
    return [x[1] for x in outputs]
  else:
    return [tf.argmax(x[1], axis=-1) for x in outputs]
