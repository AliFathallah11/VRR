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

"""Transformer-based text encoder network."""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf, tf_keras

from official.modeling import activations
from official.modeling import tf_utils
from official.nlp import modeling
from official.nlp.modeling import layers
from official.projects.bigbird import recompute_grad
from official.projects.bigbird import recomputing_dropout


_MAX_SEQ_LEN = 4096


class RecomputeTransformerLayer(layers.TransformerScaffold):
  """Transformer layer that recomputes the forward pass during backpropagation."""

  def call(self, inputs, training=None):
    emb, mask = inputs
    def f(*args):
      # recompute_grad can only handle tensor inputs. so we enumerate the
      # nested input [emb, mask] as follows:
      # args[0]: emb
      # args[1]: mask[0] = band_mask
      # args[2]: mask[1] = encoder_from_mask
      # args[3]: mask[2] = encoder_to_mask
      # args[4]: mask[3] = blocked_encoder_mask
      x = super(RecomputeTransformerLayer,
                self).call([args[0], [args[1], args[2], args[3], args[4]]],
                           training=training)
      return x

    f = recompute_grad.recompute_grad(f)

    return f(emb, *mask)


@tf_keras.utils.register_keras_serializable(package='Text')
class BigBirdEncoder(tf_keras.Model):
  """Transformer-based encoder network with BigBird attentions.

  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_position_embeddings: The maximum length of position embeddings that this
      encoder can consume. If None, max_position_embeddings uses the value from
      sequence length. This determines the variable shape for positional
      embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    intermediate_size: The intermediate size for the transformer layers.
    block_size: int. A BigBird Attention parameter: size of block in from/to
      sequences.
    num_rand_blocks: int. A BigBird Attention parameter: number of random chunks
      per row.
    activation: The activation to use for the transformer layers.
    dropout_rate: The dropout rate to use for the transformer layers.
    attention_dropout_rate: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    use_gradient_checkpointing: Use gradient checkpointing to trade-off compute
      for memory.
  """

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_layers=12,
               num_attention_heads=12,
               max_position_embeddings=_MAX_SEQ_LEN,
               type_vocab_size=16,
               intermediate_size=3072,
               block_size=64,
               num_rand_blocks=3,
               activation=activations.gelu,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02),
               embedding_width=None,
               use_gradient_checkpointing=False,
               **kwargs):
    activation = tf_keras.activations.get(activation)
    initializer = tf_keras.initializers.get(initializer)

    if use_gradient_checkpointing:
      tf_keras.layers.Dropout = recomputing_dropout.RecomputingDropout
      layer_cls = RecomputeTransformerLayer
    else:
      layer_cls = layers.TransformerScaffold

    self._self_setattr_tracking = False
    self._config_dict = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'max_position_embeddings': max_position_embeddings,
        'type_vocab_size': type_vocab_size,
        'intermediate_size': intermediate_size,
        'block_size': block_size,
        'num_rand_blocks': num_rand_blocks,
        'activation': tf_utils.serialize_activation(
            activation, use_legacy_format=True
        ),
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'initializer': tf_utils.serialize_initializer(
            initializer, use_legacy_format=True
        ),
        'embedding_width': embedding_width,
    }

    word_ids = tf_keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    mask = tf_keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')
    type_ids = tf_keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')

    if embedding_width is None:
      embedding_width = hidden_size
    self._embedding_layer = modeling.layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        name='word_embeddings')
    word_embeddings = self._embedding_layer(word_ids)

    # Always uses dynamic slicing for simplicity.
    self._position_embedding_layer = modeling.layers.PositionEmbedding(
        initializer=initializer,
        max_length=max_position_embeddings,
        name='position_embedding')
    position_embeddings = self._position_embedding_layer(word_embeddings)
    self._type_embedding_layer = modeling.layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        use_one_hot=True,
        name='type_embeddings')
    type_embeddings = self._type_embedding_layer(type_ids)

    embeddings = tf_keras.layers.Add()(
        [word_embeddings, position_embeddings, type_embeddings])

    self._embedding_norm_layer = tf_keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = tf_keras.layers.Dropout(rate=dropout_rate)(embeddings)

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    if embedding_width != hidden_size:
      self._embedding_projection = tf_keras.layers.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=initializer,
          name='embedding_projection')
      embeddings = self._embedding_projection(embeddings)

    self._transformer_layers = []
    data = embeddings
    masks = layers.BigBirdMasks(block_size=block_size)(
        data, mask)
    encoder_outputs = []
    attn_head_dim = hidden_size // num_attention_heads
    for i in range(num_layers):
      layer = layer_cls(
          num_attention_heads,
          intermediate_size,
          activation,
          attention_cls=layers.BigBirdAttention,
          attention_cfg=dict(
              num_heads=num_attention_heads,
              key_dim=attn_head_dim,
              kernel_initializer=initializer,
              from_block_size=block_size,
              to_block_size=block_size,
              num_rand_blocks=num_rand_blocks,
              max_rand_mask_length=max_position_embeddings,
              seed=i),
          dropout_rate=dropout_rate,
          attention_dropout_rate=dropout_rate,
          kernel_initializer=initializer)
      self._transformer_layers.append(layer)
      data = layer([data, masks])
      encoder_outputs.append(data)

    outputs = dict(
        sequence_output=encoder_outputs[-1], encoder_outputs=encoder_outputs)
    super().__init__(
        inputs=[word_ids, mask, type_ids], outputs=outputs, **kwargs)

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_embedding_layer(self):
    return self._embedding_layer

  def get_config(self):
    return self._config_dict

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
