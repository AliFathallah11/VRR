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

"""Semantic segmentation configuration definition."""
import dataclasses
import os
from typing import List, Optional, Sequence, Union

import numpy as np

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones
from official.vision.ops import preprocess_ops


@dataclasses.dataclass
class DenseFeatureConfig(hyperparams.Config):
  """Config for dense features, such as RGB pixels, masks, heatmaps.

  The dense features are encoded images in TF examples. Thus they are
  1-, 3- or 4-channel. For features with another channel number (e.g.
  optical flow), they could be encoded in multiple 1-channel features.
  The default config is for RGB input, with mean and stddev from ImageNet
  datasets. Only supports 8-bit encoded features with the maximum value = 255.

  Attributes:
    feature_name: The key of the feature in TF examples.
    num_channels: An `int` specifying the number of channels of the feature.
    mean: A list of floats in the range of [0, 255] representing the mean value
      of each channel. The length of the list should match num_channels.
    stddev: A list of floats in the range of [0, 255] representing the standard
      deviation of each channel. The length should match num_channels.
  """
  feature_name: str = 'image/encoded'
  num_channels: int = 3
  mean: List[float] = dataclasses.field(
      default_factory=lambda: preprocess_ops.MEAN_RGB
  )
  stddev: List[float] = dataclasses.field(
      default_factory=lambda: preprocess_ops.STDDEV_RGB
  )


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  image_feature: DenseFeatureConfig = dataclasses.field(
      default_factory=DenseFeatureConfig
  )
  output_size: List[int] = dataclasses.field(default_factory=list)
  # If crop_size is specified, image will be resized first to
  # output_size, then crop of size crop_size will be cropped.
  crop_size: List[int] = dataclasses.field(default_factory=list)
  input_path: Union[Sequence[str], str, hyperparams.Config] = None
  weights: Optional[hyperparams.Config] = None
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 1000
  cycle_length: int = 10
  # If resize_eval_groundtruth is set to False, original image sizes are used
  # for eval. In that case, groundtruth_padded_size has to be specified too to
  # allow for batching the variable input sizes of images.
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_hflip: bool = True
  preserve_aspect_ratio: bool = True
  aug_policy: Optional[str] = None
  drop_remainder: bool = True
  file_type: str = 'tfrecord'
  decoder: Optional[common.DataDecoder] = dataclasses.field(
      default_factory=common.DataDecoder
  )
  additional_dense_features: List[DenseFeatureConfig] = dataclasses.field(
      default_factory=list)
  # If `centered_crop` is set to True, then resized crop
  # (if smaller than padded size) is place in the center of the image.
  # Default behaviour is to place it at left top corner.
  centered_crop: bool = False


@dataclasses.dataclass
class SegmentationHead(hyperparams.Config):
  """Segmentation head config."""
  level: int = 3
  num_convs: int = 2
  num_filters: int = 256
  use_depthwise_convolution: bool = False
  prediction_kernel_size: int = 1
  upsample_factor: int = 1
  logit_activation: Optional[str] = None  # None, 'sigmoid', or 'softmax'.
  feature_fusion: Optional[
      str] = None  # None, deeplabv3plus, panoptic_fpn_fusion or pyramid_fusion
  # deeplabv3plus feature fusion params
  low_level: Union[int, str] = 2
  low_level_num_filters: int = 48
  # panoptic_fpn_fusion params
  decoder_min_level: Optional[Union[int, str]] = None
  decoder_max_level: Optional[Union[int, str]] = None


@dataclasses.dataclass
class MaskScoringHead(hyperparams.Config):
  """Mask Scoring head config."""
  num_convs: int = 4
  num_filters: int = 128
  fc_input_size: List[int] = dataclasses.field(default_factory=list)
  num_fcs: int = 2
  fc_dims: int = 1024
  use_depthwise_convolution: bool = False


@dataclasses.dataclass
class SemanticSegmentationModel(hyperparams.Config):
  """Semantic segmentation model config."""
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  max_level: int = 6
  head: SegmentationHead = dataclasses.field(default_factory=SegmentationHead)
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='resnet', resnet=backbones.ResNet()
      )
  )
  decoder: decoders.Decoder = dataclasses.field(
      default_factory=lambda: decoders.Decoder(type='identity')
  )
  mask_scoring_head: Optional[MaskScoringHead] = None
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=common.NormActivation
  )


@dataclasses.dataclass
class Losses(hyperparams.Config):
  """Loss function config."""
  loss_weight: float = 1.0
  label_smoothing: float = 0.0
  ignore_label: int = 255
  gt_is_matting_map: bool = False
  class_weights: List[float] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 0.0
  use_groundtruth_dimension: bool = True
  # If true, use binary cross entropy (sigmoid) in loss, otherwise, use
  # categorical cross entropy (softmax).
  use_binary_cross_entropy: bool = False
  top_k_percent_pixels: float = 1.0
  mask_scoring_weight: float = 1.0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  """Evaluation config."""
  report_per_class_iou: bool = True
  report_train_mean_iou: bool = True  # Turning this off can speed up training.


@dataclasses.dataclass
class ExportConfig(hyperparams.Config):
  """Model export config."""
  # Whether to rescale the predicted mask to the original image size.
  rescale_output: bool = False


@dataclasses.dataclass
class SemanticSegmentationTask(cfg.TaskConfig):
  """The model config."""
  model: SemanticSegmentationModel = dataclasses.field(
      default_factory=SemanticSegmentationModel
  )
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=False)
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  evaluation: Evaluation = dataclasses.field(default_factory=Evaluation)
  train_input_partition_dims: List[int] = dataclasses.field(
      default_factory=list)
  eval_input_partition_dims: List[int] = dataclasses.field(default_factory=list)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder
  export_config: ExportConfig = dataclasses.field(default_factory=ExportConfig)
  allow_image_summary: bool = True


@exp_factory.register_config_factory('semantic_segmentation')
def semantic_segmentation() -> cfg.ExperimentConfig:
  """Semantic segmentation general."""
  return cfg.ExperimentConfig(
      task=SemanticSegmentationTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


# PASCAL VOC 2012 Dataset
PASCAL_TRAIN_EXAMPLES = 10582
PASCAL_VAL_EXAMPLES = 1449
PASCAL_INPUT_PATH_BASE = 'gs://**/pascal_voc_seg'


@exp_factory.register_config_factory('seg_deeplabv3_pascal')
def seg_deeplabv3_pascal() -> cfg.ExperimentConfig:
  """Image segmentation on pascal voc with resnet deeplabv3."""
  train_batch_size = 16
  eval_batch_size = 8
  steps_per_epoch = PASCAL_TRAIN_EXAMPLES // train_batch_size
  output_stride = 16
  aspp_dilation_rates = [12, 24, 36]  # [6, 12, 18] if output_stride = 16
  multigrid = [1, 2, 4]
  stem_type = 'v1'
  level = int(np.math.log2(output_stride))
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              input_size=[None, None, 3],
              backbone=backbones.Backbone(
                  type='dilated_resnet',
                  dilated_resnet=backbones.DilatedResNet(
                      model_id=101,
                      output_stride=output_stride,
                      multigrid=multigrid,
                      stem_type=stem_type)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level, dilation_rates=aspp_dilation_rates)),
              head=SegmentationHead(level=level, num_convs=0),
              norm_activation=common.NormActivation(
                  activation='swish',
                  norm_momentum=0.9997,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              # TODO(arashwan): test changing size to 513 to match deeplab.
              output_size=[512, 512],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              output_size=[512, 512],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False),
          # resnet101
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/deeplab/deeplab_resnet101_imagenet/ckpt-62400',
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=45 * steps_per_epoch,
          validation_steps=PASCAL_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.007,
                      'decay_steps': 45 * steps_per_epoch,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('seg_deeplabv3plus_pascal')
def seg_deeplabv3plus_pascal() -> cfg.ExperimentConfig:
  """Image segmentation on pascal voc with resnet deeplabv3+."""
  train_batch_size = 16
  eval_batch_size = 8
  steps_per_epoch = PASCAL_TRAIN_EXAMPLES // train_batch_size
  output_stride = 16
  aspp_dilation_rates = [6, 12, 18]
  multigrid = [1, 2, 4]
  stem_type = 'v1'
  level = int(np.math.log2(output_stride))
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              input_size=[None, None, 3],
              backbone=backbones.Backbone(
                  type='dilated_resnet',
                  dilated_resnet=backbones.DilatedResNet(
                      model_id=101,
                      output_stride=output_stride,
                      stem_type=stem_type,
                      multigrid=multigrid)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level, dilation_rates=aspp_dilation_rates)),
              head=SegmentationHead(
                  level=level,
                  num_convs=2,
                  feature_fusion='deeplabv3plus',
                  low_level=2,
                  low_level_num_filters=48),
              norm_activation=common.NormActivation(
                  activation='swish',
                  norm_momentum=0.9997,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              output_size=[512, 512],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              output_size=[512, 512],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False),
          # resnet101
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/deeplab/deeplab_resnet101_imagenet/ckpt-62400',
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=45 * steps_per_epoch,
          validation_steps=PASCAL_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.007,
                      'decay_steps': 45 * steps_per_epoch,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('seg_resnetfpn_pascal')
def seg_resnetfpn_pascal() -> cfg.ExperimentConfig:
  """Image segmentation on pascal voc with resnet-fpn."""
  train_batch_size = 256
  eval_batch_size = 32
  steps_per_epoch = PASCAL_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              input_size=[512, 512, 3],
              min_level=3,
              max_level=7,
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)),
              decoder=decoders.Decoder(type='fpn', fpn=decoders.FPN()),
              head=SegmentationHead(level=3, num_convs=3),
              norm_activation=common.NormActivation(
                  activation='swish', use_sync_bn=True)),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.2,
              aug_scale_max=1.5),
          validation_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False),
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=450 * steps_per_epoch,
          validation_steps=PASCAL_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.007,
                      'decay_steps': 450 * steps_per_epoch,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('mnv2_deeplabv3_pascal')
def mnv2_deeplabv3_pascal() -> cfg.ExperimentConfig:
  """Image segmentation on pascal with mobilenetv2 deeplabv3."""
  train_batch_size = 16
  eval_batch_size = 16
  steps_per_epoch = PASCAL_TRAIN_EXAMPLES // train_batch_size
  output_stride = 16
  aspp_dilation_rates = []
  level = int(np.math.log2(output_stride))
  pool_kernel_size = []

  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              input_size=[None, None, 3],
              backbone=backbones.Backbone(
                  type='mobilenet',
                  mobilenet=backbones.MobileNet(
                      model_id='MobileNetV2', output_stride=output_stride)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level,
                      dilation_rates=aspp_dilation_rates,
                      pool_kernel_size=pool_kernel_size)),
              head=SegmentationHead(level=level, num_convs=0),
              norm_activation=common.NormActivation(
                  activation='relu',
                  norm_momentum=0.99,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=4e-5),
          train_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              output_size=[512, 512],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              output_size=[512, 512],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False),
          # mobilenetv2
          init_checkpoint='gs://tf_model_garden/cloud/vision-2.0/deeplab/deeplabv3_mobilenetv2_coco/best_ckpt-63',
          init_checkpoint_modules=['backbone', 'decoder']),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=30000,
          validation_steps=PASCAL_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          best_checkpoint_eval_metric='mean_iou',
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_metric_comp='higher',
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.007 * train_batch_size / 16,
                      'decay_steps': 30000,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


# Cityscapes Dataset (Download and process the dataset yourself)
CITYSCAPES_TRAIN_EXAMPLES = 2975
CITYSCAPES_VAL_EXAMPLES = 500
CITYSCAPES_INPUT_PATH_BASE = 'cityscapes'


@exp_factory.register_config_factory('seg_deeplabv3plus_cityscapes')
def seg_deeplabv3plus_cityscapes() -> cfg.ExperimentConfig:
  """Image segmentation on cityscapes with resnet deeplabv3+."""
  train_batch_size = 16
  eval_batch_size = 16
  steps_per_epoch = CITYSCAPES_TRAIN_EXAMPLES // train_batch_size
  output_stride = 16
  aspp_dilation_rates = [6, 12, 18]
  multigrid = [1, 2, 4]
  stem_type = 'v1'
  level = int(np.math.log2(output_stride))
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              # Cityscapes uses only 19 semantic classes for train/evaluation.
              # The void (background) class is ignored in train and evaluation.
              num_classes=19,
              input_size=[None, None, 3],
              backbone=backbones.Backbone(
                  type='dilated_resnet',
                  dilated_resnet=backbones.DilatedResNet(
                      model_id=101,
                      output_stride=output_stride,
                      stem_type=stem_type,
                      multigrid=multigrid)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level,
                      dilation_rates=aspp_dilation_rates,
                      pool_kernel_size=[512, 1024])),
              head=SegmentationHead(
                  level=level,
                  num_convs=2,
                  feature_fusion='deeplabv3plus',
                  low_level=2,
                  low_level_num_filters=48),
              norm_activation=common.NormActivation(
                  activation='swish',
                  norm_momentum=0.99,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=1e-4),
          train_data=DataConfig(
              input_path=os.path.join(CITYSCAPES_INPUT_PATH_BASE,
                                      'train_fine**'),
              crop_size=[512, 1024],
              output_size=[1024, 2048],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=DataConfig(
              input_path=os.path.join(CITYSCAPES_INPUT_PATH_BASE, 'val_fine*'),
              output_size=[1024, 2048],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=True,
              drop_remainder=False),
          # resnet101
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/deeplab/deeplab_resnet101_imagenet/ckpt-62400',
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=500 * steps_per_epoch,
          validation_steps=CITYSCAPES_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.01,
                      'decay_steps': 500 * steps_per_epoch,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('mnv2_deeplabv3_cityscapes')
def mnv2_deeplabv3_cityscapes() -> cfg.ExperimentConfig:
  """Image segmentation on cityscapes with mobilenetv2 deeplabv3."""
  train_batch_size = 16
  eval_batch_size = 16
  steps_per_epoch = CITYSCAPES_TRAIN_EXAMPLES // train_batch_size
  output_stride = 16
  aspp_dilation_rates = []
  pool_kernel_size = [512, 1024]

  level = int(np.math.log2(output_stride))
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              # Cityscapes uses only 19 semantic classes for train/evaluation.
              # The void (background) class is ignored in train and evaluation.
              num_classes=19,
              input_size=[None, None, 3],
              backbone=backbones.Backbone(
                  type='mobilenet',
                  mobilenet=backbones.MobileNet(
                      model_id='MobileNetV2', output_stride=output_stride)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level,
                      dilation_rates=aspp_dilation_rates,
                      pool_kernel_size=pool_kernel_size)),
              head=SegmentationHead(level=level, num_convs=0),
              norm_activation=common.NormActivation(
                  activation='relu',
                  norm_momentum=0.99,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=Losses(l2_weight_decay=4e-5),
          train_data=DataConfig(
              input_path=os.path.join(CITYSCAPES_INPUT_PATH_BASE,
                                      'train_fine**'),
              crop_size=[512, 1024],
              output_size=[1024, 2048],
              is_training=True,
              global_batch_size=train_batch_size,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=DataConfig(
              input_path=os.path.join(CITYSCAPES_INPUT_PATH_BASE, 'val_fine*'),
              output_size=[1024, 2048],
              is_training=False,
              global_batch_size=eval_batch_size,
              resize_eval_groundtruth=True,
              drop_remainder=False),
          # Coco pre-trained mobilenetv2 checkpoint
          init_checkpoint='gs://tf_model_garden/cloud/vision-2.0/deeplab/deeplabv3_mobilenetv2_coco/best_ckpt-63',
          init_checkpoint_modules='backbone'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=100000,
          validation_steps=CITYSCAPES_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          best_checkpoint_eval_metric='mean_iou',
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_metric_comp='higher',
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.01,
                      'decay_steps': 100000,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('mnv2_deeplabv3plus_cityscapes')
def mnv2_deeplabv3plus_cityscapes() -> cfg.ExperimentConfig:
  """Image segmentation on cityscapes with mobilenetv2 deeplabv3plus."""
  config = mnv2_deeplabv3_cityscapes()
  config.task.model.head = SegmentationHead(
      level=4,
      num_convs=2,
      feature_fusion='deeplabv3plus',
      use_depthwise_convolution=True,
      low_level='2/depthwise',
      low_level_num_filters=48)
  config.task.model.backbone.mobilenet.output_intermediate_endpoints = True
  return config
