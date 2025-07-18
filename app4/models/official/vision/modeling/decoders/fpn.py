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

"""Contains the definitions of Feature Pyramid Networks (FPN)."""
from typing import Any, Mapping, Optional

# Import libraries
from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.decoders import factory
from official.vision.ops import spatial_transform_ops


@tf_keras.utils.register_keras_serializable(package='Vision')
class FPN(tf_keras.Model):
  """Creates a Feature Pyramid Network (FPN).

  This implements the paper:
  Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and
  Serge Belongie.
  Feature Pyramid Networks for Object Detection.
  (https://arxiv.org/pdf/1612.03144)
  """

  def __init__(
      self,
      input_specs: Mapping[str, tf.TensorShape],
      min_level: int = 3,
      max_level: int = 7,
      num_filters: int = 256,
      fusion_type: str = 'sum',
      use_separable_conv: bool = False,
      use_keras_layer: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a Feature Pyramid Network (FPN).

    Args:
      input_specs: A `dict` of input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      min_level: An `int` of minimum level in FPN output feature maps.
      max_level: An `int` of maximum level in FPN output feature maps.
      num_filters: An `int` number of filters in FPN layers.
      fusion_type: A `str` of `sum` or `concat`. Whether performing sum or
        concat for feature fusion.
      use_separable_conv: A `bool`.  If True use separable convolution for
        convolution in FPN layers.
      use_keras_layer: A `bool`. If Ture use keras layers as many as possible.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._config_dict = {
        'input_specs': input_specs,
        'min_level': min_level,
        'max_level': max_level,
        'num_filters': num_filters,
        'fusion_type': fusion_type,
        'use_separable_conv': use_separable_conv,
        'use_keras_layer': use_keras_layer,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    conv2d = (
        tf_keras.layers.SeparableConv2D
        if use_separable_conv
        else tf_keras.layers.Conv2D
    )
    norm = tf_keras.layers.BatchNormalization
    activation_fn = tf_utils.get_activation(activation, use_keras_layer=True)

    # Build input feature pyramid.
    bn_axis = (
        -1 if tf_keras.backend.image_data_format() == 'channels_last' else 1
    )

    # Get input feature pyramid from backbone.
    logging.info('FPN input_specs: %s', input_specs)
    inputs = self._build_input_pyramid(input_specs, min_level)
    backbone_max_level = min(int(max(inputs.keys())), max_level)

    # Build lateral connections.
    feats_lateral = {}
    for level in range(min_level, backbone_max_level + 1):
      feats_lateral[str(level)] = conv2d(
          filters=num_filters,
          kernel_size=1,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          name=f'lateral_{level}')(
              inputs[str(level)])

    # Build top-down path.
    feats = {str(backbone_max_level): feats_lateral[str(backbone_max_level)]}
    for level in range(backbone_max_level - 1, min_level - 1, -1):
      feat_a = spatial_transform_ops.nearest_upsampling(
          feats[str(level + 1)], 2, use_keras_layer=use_keras_layer)
      feat_b = feats_lateral[str(level)]

      if fusion_type == 'sum':
        if use_keras_layer:
          feats[str(level)] = tf_keras.layers.Add()([feat_a, feat_b])
        else:
          feats[str(level)] = feat_a + feat_b
      elif fusion_type == 'concat':
        if use_keras_layer:
          feats[str(level)] = tf_keras.layers.Concatenate(axis=-1)(
              [feat_a, feat_b])
        else:
          feats[str(level)] = tf.concat([feat_a, feat_b], axis=-1)
      else:
        raise ValueError('Fusion type {} not supported.'.format(fusion_type))

    # TODO(fyangf): experiment with removing bias in conv2d.
    # Build post-hoc 3x3 convolution kernel.
    for level in range(min_level, backbone_max_level + 1):
      feats[str(level)] = conv2d(
          filters=num_filters,
          strides=1,
          kernel_size=3,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          name=f'post_hoc_{level}')(
              feats[str(level)])

    # TODO(fyangf): experiment with removing bias in conv2d.
    # Build coarser FPN levels introduced for RetinaNet.
    for level in range(backbone_max_level + 1, max_level + 1):
      feats_in = feats[str(level - 1)]
      if level > backbone_max_level + 1:
        feats_in = activation_fn(feats_in)
      feats[str(level)] = conv2d(
          filters=num_filters,
          strides=2,
          kernel_size=3,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          name=f'coarser_{level}')(
              feats_in)

    # Apply batch norm layers.
    for level in range(min_level, max_level + 1):
      feats[str(level)] = norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          synchronized=use_sync_bn,
          name=f'norm_{level}')(
              feats[str(level)])

    self._output_specs = {
        str(level): feats[str(level)].get_shape()
        for level in range(min_level, max_level + 1)
    }

    super().__init__(inputs=inputs, outputs=feats, **kwargs)

  def _build_input_pyramid(self, input_specs: Mapping[str, tf.TensorShape],
                           min_level: int):
    assert isinstance(input_specs, dict)
    if min(input_specs.keys()) > str(min_level):
      raise ValueError(
          'Backbone min level should be less or equal to FPN min level')

    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf_keras.Input(shape=spec[1:])
    return inputs

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_decoder_builder('fpn')
def build_fpn_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None
) -> tf_keras.Model:
  """Builds FPN decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: A `tf_keras.regularizers.Regularizer` instance. Default to
      None.

  Returns:
    A `tf_keras.Model` instance of the FPN decoder.

  Raises:
    ValueError: If the model_config.decoder.type is not `fpn`.
  """
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  if decoder_type != 'fpn':
    raise ValueError(f'Inconsistent decoder type {decoder_type}. '
                     'Need to be `fpn`.')
  norm_activation_config = model_config.norm_activation
  return FPN(
      input_specs=input_specs,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_filters=decoder_cfg.num_filters,
      fusion_type=decoder_cfg.fusion_type,
      use_separable_conv=decoder_cfg.use_separable_conv,
      use_keras_layer=decoder_cfg.use_keras_layer,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
