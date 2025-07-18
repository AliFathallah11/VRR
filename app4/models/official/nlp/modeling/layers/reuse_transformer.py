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

"""Keras-based TransformerEncoder block layer."""
import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.nlp.modeling.layers import reuse_attention as attention


class ReuseTransformer(tf_keras.layers.Layer):
  """Transformer layer.

  This layer implements the ReuseTransformer Encoder from
  "Leveraging redundancy in attention with Reuse Transformers".
  (https://arxiv.org/abs/2110.06821)
  """

  def __init__(self,
               num_attention_heads,
               inner_dim,
               inner_activation,
               head_size=None,
               output_range=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_bias=True,
               norm_first=False,
               norm_epsilon=1e-12,
               output_dropout=0.0,
               attention_dropout=0.0,
               inner_dropout=0.0,
               attention_initializer=None,
               attention_axes=None,
               reuse_attention=0,
               use_relative_pe=False,
               pe_max_seq_length=512,
               layer_idx=None,
               max_reuse_layer_idx=None,
               **kwargs):
    """Initializes `ReuseTransformer`.

    Args:
      num_attention_heads: Number of attention heads.
      inner_dim: The output dimension of the first Dense layer in a two-layer
        feedforward network.
      inner_activation: The activation for the first Dense layer in a two-layer
        feedforward network.
      head_size: Projection size of heads.
      output_range: the sequence output range, [0, output_range) for slicing the
        target sequence. `None` means the target sequence is not sliced.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
      use_bias: Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
      norm_first: Whether to normalize inputs to attention and intermediate
        dense layers. If set False, output of attention and intermediate dense
        layers is normalized.
      norm_epsilon: Epsilon value to initialize normalization layers.
      output_dropout: Dropout probability for the post-attention and output
        dropout.
      attention_dropout: Dropout probability for within the attention layer.
      inner_dropout: Dropout probability for the first Dense layer in a
        two-layer feedforward network.
      attention_initializer: Initializer for kernels of attention layers. If set
        `None`, attention layers use kernel_initializer as initializer for
        kernel.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      reuse_attention: reuse_attention: An integer specifying number of heads
        to reuse. -1 for all heads.
      use_relative_pe: whether to use relative position bias.
      pe_max_seq_length: used to set the size of the relative positin encodings.
      layer_idx: the idx of this layer.
      max_reuse_layer_idx: layer idx (if passed) greater than this value will
        not reuse attention scores from previous layers.
      **kwargs: keyword arguments.
    """
    super().__init__(**kwargs)

    self._num_heads = num_attention_heads
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
    self._head_size = head_size
    self._attention_dropout = attention_dropout
    self._attention_dropout_rate = attention_dropout
    self._output_dropout = output_dropout
    self._output_dropout_rate = output_dropout
    self._output_range = output_range
    self._kernel_initializer = tf_keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf_keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf_keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf_keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf_keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf_keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf_keras.constraints.get(bias_constraint)
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._inner_dropout = inner_dropout
    self._reuse_attention = reuse_attention
    self._use_relative_pe = use_relative_pe
    self._pe_max_seq_length = pe_max_seq_length
    self._layer_idx = layer_idx
    self._max_reuse_layer_idx = max_reuse_layer_idx
    # Overwrite for the first layer and layers greater than max_reuse_layer_idx.
    if self._layer_idx is not None and (
        self._layer_idx == 0 or (self._max_reuse_layer_idx is not None and
                                 self._max_reuse_layer_idx < self._layer_idx)):
      self._reuse_attention = 0
    if attention_initializer:
      self._attention_initializer = tf_keras.initializers.get(
          attention_initializer)
    else:
      self._attention_initializer = tf_utils.clone_initializer(
          self._kernel_initializer)
    self._attention_axes = attention_axes

  def build(self, input_shape):
    if isinstance(input_shape, tf.TensorShape):
      input_tensor_shape = input_shape
    elif isinstance(input_shape, (list, tuple)):
      input_tensor_shape = tf.TensorShape(input_shape[0])
    else:
      raise ValueError(
          "The type of input shape argument is not supported, got: %s" %
          type(input_shape))
    einsum_equation = "abc,cd->abd"
    if len(input_tensor_shape.as_list()) > 3:
      einsum_equation = "...bc,cd->...bd"
    hidden_size = input_tensor_shape[-1]
    if self._head_size is None:
      if hidden_size % self._num_heads != 0:
        raise ValueError(
            "The input size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, self._num_heads))
      self._attention_head_size = int(hidden_size // self._num_heads)
    else:
      self._attention_head_size = self._head_size
    common_kwargs = dict(
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
    self._attention_layer = attention.ReuseMultiHeadAttention(
        num_heads=self._num_heads,
        key_dim=self._attention_head_size,
        dropout=self._attention_dropout,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        attention_axes=self._attention_axes,
        reuse_attention=self._reuse_attention,
        use_relative_pe=self._use_relative_pe,
        pe_max_seq_length=self._pe_max_seq_length,
        name="self_attention",
        **common_kwargs)
    self._attention_dropout = tf_keras.layers.Dropout(
        rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = (
        tf_keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32))
    self._intermediate_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=(None, self._inner_dim),
        bias_axes="d",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        name="intermediate",
        **common_kwargs)
    policy = tf_keras.mixed_precision.global_policy()
    if policy.name == "mixed_bfloat16":
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      # TODO(b/154538392): Investigate this.
      policy = tf.float32
    self._intermediate_activation_layer = tf_keras.layers.Activation(
        self._inner_activation, dtype=policy)
    self._inner_dropout_layer = tf_keras.layers.Dropout(
        rate=self._inner_dropout)
    self._output_dense = tf_keras.layers.EinsumDense(
        einsum_equation,
        output_shape=(None, hidden_size),
        bias_axes="d",
        name="output",
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
        **common_kwargs)
    self._output_dropout = tf_keras.layers.Dropout(rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf_keras.layers.LayerNormalization(
        name="output_layer_norm",
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32)

    super(ReuseTransformer, self).build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads":
            self._num_heads,
        "inner_dim":
            self._inner_dim,
        "inner_activation":
            self._inner_activation,
        "head_size":
            self._head_size,
        "output_dropout":
            self._output_dropout_rate,
        "attention_dropout":
            self._attention_dropout_rate,
        "output_range":
            self._output_range,
        "reuse_attention":
            self._reuse_attention,
        "use_relative_pe": self._use_relative_pe,
        "pe_max_seq_length": self._pe_max_seq_length,
        "max_reuse_layer_idx": self._max_reuse_layer_idx,
        "kernel_initializer":
            tf_keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf_keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf_keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf_keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf_keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf_keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf_keras.constraints.serialize(self._bias_constraint),
        "use_bias":
            self._use_bias,
        "norm_first":
            self._norm_first,
        "norm_epsilon":
            self._norm_epsilon,
        "inner_dropout":
            self._inner_dropout,
        "attention_initializer":
            tf_keras.initializers.serialize(self._attention_initializer),
        "attention_axes": self._attention_axes,
    }
    base_config = super(ReuseTransformer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Transformer self-attention encoder block call.

    Args:
      inputs: a single tensor or a list of tensors.
        `input tensor` as the single sequence of embeddings.
        [`input tensor`, `attention mask`] to have the additional attention
          mask.
        [`query tensor`, `attention mask`, `attention scores`] to have
        additional attention scores for reuse computation. If `attention scores`
        is None, the reuse_attention flag will be ignored.
    Returns:
      An output tensor with the same dimensions as input/query tensor.
      Attention scores if return_attention_scores is true.
    """
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        input_tensor, attention_mask = inputs
        reuse_attention_scores = None
      elif len(inputs) == 3:
        input_tensor, attention_mask, reuse_attention_scores = inputs
      else:
        raise ValueError("Unexpected inputs to %s with length at %d" %
                         (self.__class__, len(inputs)))
    else:
      input_tensor, attention_mask, reuse_attention_scores = (inputs, None,
                                                              None)

    key_value = None

    if self._reuse_attention != 0 and reuse_attention_scores is None:
      raise ValueError(
          "reuse_attention_scores cannot be None when reuse_attention != 0.")

    if self._output_range:
      if self._norm_first:
        source_tensor = input_tensor[:, 0:self._output_range, :]
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor[:, 0:self._output_range, :]
      if attention_mask is not None:
        attention_mask = attention_mask[:, 0:self._output_range, :]
      if reuse_attention_scores is not None:
        reuse_attention_scores = reuse_attention_scores[:, :,
                                                        0:self._output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm(key_value)
      target_tensor = input_tensor

    if key_value is None:
      key_value = input_tensor
    attention_output = self._attention_layer(
        query=target_tensor, value=key_value, attention_mask=attention_mask,
        reuse_attention_scores=reuse_attention_scores,
        return_attention_scores=True)
    attention_output, attention_scores = attention_output
    attention_output = self._attention_dropout(attention_output)
    if self._norm_first:
      attention_output = source_tensor + attention_output
    else:
      attention_output = self._attention_layer_norm(target_tensor +
                                                    attention_output)
    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)

    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout(layer_output)

    if self._norm_first:
      return source_attention_output + layer_output, attention_scores

    # During mixed precision training, layer norm output is always fp32 for now.
    # Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    layer_output = self._output_layer_norm(layer_output + attention_output)
    return layer_output, attention_scores
