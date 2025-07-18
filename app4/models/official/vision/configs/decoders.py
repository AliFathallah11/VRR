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

"""Decoders configurations."""
import dataclasses
from typing import List, Optional

# Import libraries

from official.modeling import hyperparams


@dataclasses.dataclass
class Identity(hyperparams.Config):
  """Identity config."""
  pass


@dataclasses.dataclass
class FPN(hyperparams.Config):
  """FPN config."""
  num_filters: int = 256
  fusion_type: str = 'sum'
  use_separable_conv: bool = False
  use_keras_layer: bool = False


@dataclasses.dataclass
class NASFPN(hyperparams.Config):
  """NASFPN config."""
  num_filters: int = 256
  num_repeats: int = 5
  use_separable_conv: bool = False


@dataclasses.dataclass
class ASPP(hyperparams.Config):
  """ASPP config."""
  level: int = 4
  dilation_rates: List[int] = dataclasses.field(default_factory=list)
  dropout_rate: float = 0.0
  num_filters: int = 256
  use_depthwise_convolution: bool = False
  pool_kernel_size: Optional[List[int]] = None  # Use global average pooling.
  spp_layer_version: str = 'v1'
  output_tensor: bool = False


@dataclasses.dataclass
class Decoder(hyperparams.OneOfConfig):
  """Configuration for decoders.

  Attributes:
    type: 'str', type of decoder be used, one of the fields below.
    fpn: fpn config.
  """
  type: Optional[str] = None
  fpn: FPN = dataclasses.field(default_factory=FPN)
  nasfpn: NASFPN = dataclasses.field(default_factory=NASFPN)
  identity: Identity = dataclasses.field(default_factory=Identity)
  aspp: ASPP = dataclasses.field(default_factory=ASPP)
