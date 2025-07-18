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

"""The encoder used for FFFNER."""
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.projects.fffner import fffner_encoder

FFFNerEncoderConfig = encoders.BertEncoderConfig


@base_config.bind(FFFNerEncoderConfig)
def get_encoder(encoder_cfg: FFFNerEncoderConfig):
  """Gets the FFNerEncoder from the configurations."""
  encoder = fffner_encoder.FFFNerEncoder(
      vocab_size=encoder_cfg.vocab_size,
      hidden_size=encoder_cfg.hidden_size,
      num_layers=encoder_cfg.num_layers,
      num_attention_heads=encoder_cfg.num_attention_heads,
      inner_dim=encoder_cfg.intermediate_size,
      inner_activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      output_dropout=encoder_cfg.dropout_rate,
      attention_dropout=encoder_cfg.attention_dropout_rate,
      max_sequence_length=encoder_cfg.max_position_embeddings,
      type_vocab_size=encoder_cfg.type_vocab_size,
      initializer=tf_keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=encoder_cfg.embedding_size,
      norm_first=encoder_cfg.norm_first)
  return encoder
