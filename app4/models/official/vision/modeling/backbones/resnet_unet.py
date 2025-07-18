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

"""Contains definitions of ResNet UNet style."""

from typing import Callable, Optional, List

# Import libraries
import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.backbones import factory
from official.vision.modeling.layers import nn_blocks
from official.vision.modeling.layers import nn_layers
from official.vision.ops import spatial_transform_ops

layers = tf_keras.layers


# Specifications for different ResNet variants.
# Each entry specifies block configurations of the particular ResNet variant.
# Each element in the block configuration is in the following format:
# (block_fn, num_filters, block_repeats)
RESNET_SPECS = {
    10: [
        ('residual', 64, 1),
        ('residual', 128, 1),
        ('residual', 256, 1),
        ('residual', 512, 1),
    ],
    18: [
        ('residual', 64, 2),
        ('residual', 128, 2),
        ('residual', 256, 2),
        ('residual', 512, 2),
    ],
    34: [
        ('residual', 64, 3),
        ('residual', 128, 4),
        ('residual', 256, 6),
        ('residual', 512, 3),
    ],
    50: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 6),
        ('bottleneck', 512, 3),
    ],
    101: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 23),
        ('bottleneck', 512, 3),
    ],
    152: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 8),
        ('bottleneck', 256, 36),
        ('bottleneck', 512, 3),
    ],
    200: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 24),
        ('bottleneck', 256, 36),
        ('bottleneck', 512, 3),
    ],
    270: [
        ('bottleneck', 64, 4),
        ('bottleneck', 128, 29),
        ('bottleneck', 256, 53),
        ('bottleneck', 512, 4),
    ],
    350: [
        ('bottleneck', 64, 4),
        ('bottleneck', 128, 36),
        ('bottleneck', 256, 72),
        ('bottleneck', 512, 4),
    ],
    420: [
        ('bottleneck', 64, 4),
        ('bottleneck', 128, 44),
        ('bottleneck', 256, 87),
        ('bottleneck', 512, 4),
    ],
}


def conv_2d(*args, **kwargs):
  return tf_keras.layers.Conv2D(
      kernel_initializer=tf_keras.initializers.truncated_normal(stddev=0.02),
      bias_initializer='zeros',
      *args,
      **kwargs,
  )


def dense(*args, **kwargs):
  return tf_keras.layers.Dense(
      kernel_initializer=tf_keras.initializers.truncated_normal(stddev=0.02),
      bias_initializer='zeros',
      *args,
      **kwargs,
  )


class ConvNeXtBlock(tf_keras.Model):
  """ConvNeXt block."""

  def __init__(
      self,
      dim,
      drop_rate=0.0,
      layer_scale_init_value=1e-6,
      norm_fn=None,
      kernel_size=7,
      se_ratio=0.0625,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.depthwise_conv = tf_keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        padding='same',
        depthwise_initializer=tf_keras.initializers.truncated_normal(
            stddev=0.02
        ),
        bias_initializer='zeros',
    )
    if norm_fn:
      self.norm = norm_fn()
    else:
      self.norm = tf_keras.layers.LayerNormalization(epsilon=1e-6)
    self.pointwise_conv1 = dense(4 * dim)
    self.act = tf_keras.layers.Activation('gelu')
    self.pointwise_conv2 = dense(dim)
    if layer_scale_init_value > 0:
      self.gamma = self.add_weight(
          name='layer_scale',
          shape=(1, 1, 1, dim),
          initializer=tf_keras.initializers.Constant(layer_scale_init_value))
    else:
      self.gamma = None

    self.drop_path = nn_layers.StochasticDepth(
        drop_rate
    ) if drop_rate > 0 else tf_keras.layers.Activation('linear')
    if se_ratio and se_ratio > 0 and se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          activation='gelu',
          in_filters=4 * dim,
          out_filters=4 * dim,
          se_ratio=se_ratio,)
    else:
      self._squeeze_excitation = None

  def call(self, x, training=None):
    inputs = x

    x = self.depthwise_conv(x)
    x = self.norm(x)
    x = self.pointwise_conv1(x)
    x = self.act(x)
    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)
    x = self.pointwise_conv2(x)

    if self.gamma is not None:
      x = self.gamma * x

    x = inputs + self.drop_path(x, training=training)
    return x


@tf_keras.utils.register_keras_serializable(package='Vision')
class ResNetUNet(tf_keras.Model):
  """Creates ResNet and ResNet-RS family models.

  This implements the Deep Residual Network from:
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition.
    (https://arxiv.org/pdf/1512.03385) and
    Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas,
    Tsung-Yi Lin, Jonathon Shlens, Barret Zoph.
    Revisiting ResNets: Improved Training and Scaling Strategies.
    (https://arxiv.org/abs/2103.07579).
  """

  def __init__(
      self,
      model_id: int,
      input_specs: tf_keras.layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, 3]),
      depth_multiplier: float = 1.0,
      stem_type: str = 'v0',
      resnetd_shortcut: bool = False,
      replace_stem_max_pool: bool = False,
      se_ratio: Optional[float] = None,
      init_stochastic_depth_rate: float = 0.0,
      upsample_repeats: Optional[List[int]] = None,
      upsample_filters: Optional[List[int]] = None,
      upsample_kernel_sizes: Optional[List[int]] = None,
      scale_stem: bool = True,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bn_trainable: bool = True,
      classification_output: bool = False,
      **kwargs):
    """Initializes a ResNet model.

    Args:
      model_id: An `int` of the depth of ResNet backbone model.
      input_specs: A `tf_keras.layers.InputSpec` of the input tensor.
      depth_multiplier: A `float` of the depth multiplier to uniformaly scale up
        all layers in channel size. This argument is also referred to as
        `width_multiplier` in (https://arxiv.org/abs/2103.07579).
      stem_type: A `str` of stem type of ResNet. Default to `v0`. If set to
        `v1`, use ResNet-D type stem (https://arxiv.org/abs/1812.01187).
      resnetd_shortcut: A `bool` of whether to use ResNet-D shortcut in
        downsampling blocks.
      replace_stem_max_pool: A `bool` of whether to replace the max pool in stem
        with a stride-2 conv,
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
      init_stochastic_depth_rate: A `float` of initial stochastic depth rate.
      upsample_repeats: A `list` for upsample repeats of the ConvNext blocks for
        each level starting from L5, then L4, and so on.
      upsample_filters: A `list` for the upsample filter sizes for the ConvNext
        blocks for each level.
      upsample_kernel_sizes: A `list` for upsample kernel sizes for the ConvNext
        blocks for each level.
      scale_stem: A `bool` of whether to scale stem layers.
      activation: A `str` name of the activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A small `float` added to variance to avoid dividing by zero.
      kernel_initializer: A str for kernel initializer of convolutional layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      bn_trainable: A `bool` that indicates whether batch norm layers should be
        trainable. Default to True.
      classification_output: A `bool` to output the correct level needed for
        classification (L3), only set to True for pretraining.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._model_id = model_id
    self._input_specs = input_specs
    self._depth_multiplier = depth_multiplier
    self._stem_type = stem_type
    self._resnetd_shortcut = resnetd_shortcut
    self._replace_stem_max_pool = replace_stem_max_pool
    self._se_ratio = se_ratio
    self._init_stochastic_depth_rate = init_stochastic_depth_rate
    self._upsample_repeats = upsample_repeats
    self._upsample_filters = upsample_filters
    self._upsample_kernel_sizes = upsample_kernel_sizes
    self._scale_stem = scale_stem
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._bn_trainable = bn_trainable
    self._classification_output = classification_output

    if tf_keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build ResNet.
    inputs = tf_keras.Input(shape=input_specs.shape[1:])

    stem_depth_multiplier = self._depth_multiplier if scale_stem else 1.0
    if stem_type == 'v0':
      x = layers.Conv2D(
          filters=int(64 * stem_depth_multiplier),
          kernel_size=7,
          strides=2,
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              inputs)
      x = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          trainable=bn_trainable)(
              x)
      x = tf_utils.get_activation(activation, use_keras_layer=True)(x)
    elif stem_type == 'v1':
      x = layers.Conv2D(
          filters=int(32 * stem_depth_multiplier),
          kernel_size=3,
          strides=2,
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              inputs)
      x = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          trainable=bn_trainable)(
              x)
      x = tf_utils.get_activation(activation, use_keras_layer=True)(x)
      x = layers.Conv2D(
          filters=int(32 * stem_depth_multiplier),
          kernel_size=3,
          strides=1,
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              x)
      x = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          trainable=bn_trainable)(
              x)
      x = tf_utils.get_activation(activation, use_keras_layer=True)(x)
      x = layers.Conv2D(
          filters=int(64 * stem_depth_multiplier),
          kernel_size=3,
          strides=1,
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              x)
      x = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          trainable=bn_trainable)(
              x)
      x = tf_utils.get_activation(activation, use_keras_layer=True)(x)
    else:
      raise ValueError('Stem type {} not supported.'.format(stem_type))

    if replace_stem_max_pool:
      x = layers.Conv2D(
          filters=int(64 * self._depth_multiplier),
          kernel_size=3,
          strides=2,
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              x)
      x = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          trainable=bn_trainable)(
              x)
      x = tf_utils.get_activation(activation, use_keras_layer=True)(x)
    else:
      x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    endpoints = {}
    for i, spec in enumerate(RESNET_SPECS[model_id]):
      if spec[0] == 'residual':
        block_fn = nn_blocks.ResidualBlock
      elif spec[0] == 'bottleneck':
        block_fn = nn_blocks.BottleneckBlock
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(spec[0]))
      x = self._block_group(
          inputs=x,
          filters=int(spec[1] * self._depth_multiplier),
          strides=(1 if i == 0 else 2),
          block_fn=block_fn,
          block_repeats=spec[2],
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 2, 8),
          name='block_group_l{}'.format(i + 2))
      endpoints[str(i + 2)] = x

    norm_layer = lambda: tf_keras.layers.LayerNormalization(epsilon=1e-6)
    for i in range(len(upsample_filters)):
      backbone_feature = layers.Conv2D(
          filters=int(upsample_filters[i] * stem_depth_multiplier),
          kernel_size=1,
          strides=1,
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              endpoints['{}'.format(5 - i)])
      backbone_feature = norm_layer()(backbone_feature)

      if i == 0:
        x = backbone_feature
      else:
        x = layers.Conv2D(
            filters=int(upsample_filters[i] * stem_depth_multiplier),
            kernel_size=1,
            strides=1,
            use_bias=False,
            padding='same',
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)(
                x)
        x = norm_layer()(x)

        x = spatial_transform_ops.nearest_upsampling(
            x, scale=2,
            use_keras_layer=True) + backbone_feature

      for _ in range(upsample_repeats[i]):
        x = ConvNeXtBlock(
            int(upsample_filters[i] * self._depth_multiplier),
            drop_rate=nn_layers.get_stochastic_depth_rate(
                self._init_stochastic_depth_rate, i + 6, 8),
            kernel_size=upsample_kernel_sizes[i])(x)
      x = tf_utils.get_activation(activation, use_keras_layer=True)(x)
      endpoints[str(5 - i)] = x

    if classification_output:
      endpoints['6'] = endpoints[str(5 - len(upsample_repeats) + 1)]

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super().__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs: tf.Tensor,
                   filters: int,
                   strides: int,
                   block_fn: Callable[..., tf_keras.layers.Layer],
                   block_repeats: int = 1,
                   stochastic_depth_drop_rate: float = 0.0,
                   name: str = 'block_group'):
    """Creates one group of blocks for the ResNet model.

    Args:
      inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
      filters: An `int` number of filters for the first convolution of the
        layer.
      strides: An `int` stride to use for the first convolution of the layer.
        If greater than 1, this layer will downsample the input.
      block_fn: The type of block group. Either `nn_blocks.ResidualBlock` or
        `nn_blocks.BottleneckBlock`.
      block_repeats: An `int` number of blocks contained in the layer.
      stochastic_depth_drop_rate: A `float` of drop rate of the current block
        group.
      name: A `str` name for the block.

    Returns:
      The output `tf.Tensor` of the block layer.
    """
    x = block_fn(
        filters=filters,
        strides=strides,
        use_projection=True,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate,
        se_ratio=self._se_ratio,
        resnetd_shortcut=self._resnetd_shortcut,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon,
        bn_trainable=self._bn_trainable)(
            inputs)

    for _ in range(1, block_repeats):
      x = block_fn(
          filters=filters,
          strides=1,
          use_projection=False,
          stochastic_depth_drop_rate=stochastic_depth_drop_rate,
          se_ratio=self._se_ratio,
          resnetd_shortcut=self._resnetd_shortcut,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon,
          bn_trainable=self._bn_trainable)(
              x)

    return tf_keras.layers.Activation('linear', name=name)(x)

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'depth_multiplier': self._depth_multiplier,
        'stem_type': self._stem_type,
        'resnetd_shortcut': self._resnetd_shortcut,
        'replace_stem_max_pool': self._replace_stem_max_pool,
        'activation': self._activation,
        'se_ratio': self._se_ratio,
        'init_stochastic_depth_rate': self._init_stochastic_depth_rate,
        'upsample_repeats': self._upsample_repeats,
        'upsample_filters': self._upsample_filters,
        'upsample_kernel_sizes': self._upsample_kernel_sizes,
        'scale_stem': self._scale_stem,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'bn_trainable': self._bn_trainable,
        'classification_output': self._classification_output,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('resnet_unet')
def build_resnet(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer = None) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds ResNet ConvNext Unet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'resnet_unet', (
      f'Inconsistent backbone type {backbone_type}')

  return ResNetUNet(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs,
      depth_multiplier=backbone_cfg.depth_multiplier,
      upsample_repeats=backbone_cfg.upsample_repeats,
      upsample_filters=backbone_cfg.upsample_filters,
      upsample_kernel_sizes=backbone_cfg.upsample_kernel_sizes,
      stem_type=backbone_cfg.stem_type,
      resnetd_shortcut=backbone_cfg.resnetd_shortcut,
      replace_stem_max_pool=backbone_cfg.replace_stem_max_pool,
      se_ratio=backbone_cfg.se_ratio,
      init_stochastic_depth_rate=backbone_cfg.stochastic_depth_drop_rate,
      scale_stem=backbone_cfg.scale_stem,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      bn_trainable=backbone_cfg.bn_trainable,
      classification_output=backbone_cfg.classification_output)
