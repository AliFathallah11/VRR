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

"""Keras Layers for BERT-specific preprocessing."""
# pylint: disable=g-import-not-at-top
from typing import Any, Dict, List, Mapping, Optional, Text, Union

from absl import logging
import tensorflow as tf, tf_keras

try:
  # pytype: disable=import-error
  import tensorflow_text as text
  from tensorflow_text.python.ops import bert_tokenizer
  # pytype: enable=import-error
except ImportError:
  text = None
  bert_tokenizer = None
except tf.errors.NotFoundError as e:
  logging.warn("Encountered error when importing tensorflow_text: %s", e)
  text = None
  bert_tokenizer = None


def _check_if_tf_text_installed():
  if text is None:
    raise ImportError("import tensorflow_text failed, please install "
                      "'tensorflow-text-nightly'.")


def _truncate_row_lengths(ragged_tensor: tf.RaggedTensor,
                          new_lengths: tf.Tensor) -> tf.RaggedTensor:
  """Truncates the rows of `ragged_tensor` to the given row lengths."""
  new_lengths = tf.broadcast_to(new_lengths,
                                ragged_tensor.bounding_shape()[0:1])
  def fn(x):
    row, new_length = x
    return row[0:new_length]
  fn_dtype = tf.RaggedTensorSpec(dtype=ragged_tensor.dtype,
                                 ragged_rank=ragged_tensor.ragged_rank - 1)
  result = tf.map_fn(fn, (ragged_tensor, new_lengths), dtype=fn_dtype)
  # Work around broken shape propagation: without this, result has unknown rank.
  flat_values_shape = [None] * ragged_tensor.flat_values.shape.rank
  result = result.with_flat_values(
      tf.ensure_shape(result.flat_values, flat_values_shape))

  return result


class BertTokenizer(tf_keras.layers.Layer):
  """Wraps TF.Text's BertTokenizer with pre-defined vocab as a Keras Layer.

  Attributes:
    tokenize_with_offsets: If true, calls
      `text.BertTokenizer.tokenize_with_offsets()` instead of plain
      `text.BertTokenizer.tokenize()` and outputs a triple of
      `(tokens, start_offsets, limit_offsets)`.
    raw_table_access: An object with methods `.lookup(keys) and `.size()`
      that operate on the raw lookup table of tokens. It can be used to
      look up special token synbols like `[MASK]`.
  """

  def __init__(self, *,
               vocab_file: str,
               lower_case: Optional[bool] = None,
               tokenize_with_offsets: bool = False,
               tokenizer_kwargs: Optional[Mapping[Text, Any]] = None,
               **kwargs):
    """Initialize a `BertTokenizer` layer.

    Args:
      vocab_file: A Python string with the path of the vocabulary file.
        This is a text file with newline-separated wordpiece tokens.
        This layer initializes a lookup table from it that gets used with
        `text.BertTokenizer`.
      lower_case: Optional boolean forwarded to `text.BertTokenizer`.
        If true, input text is converted to lower case (where applicable)
        before tokenization. This must be set to match the way in which
        the `vocab_file` was created. If passed, this overrides whatever value
        may have been passed in `tokenizer_kwargs`.
      tokenize_with_offsets: A Python boolean. If true, this layer calls
        `text.BertTokenizer.tokenize_with_offsets()` instead of plain
        `text.BertTokenizer.tokenize()` and outputs a triple of
        `(tokens, start_offsets, limit_offsets)`
        insead of just tokens.
      tokenizer_kwargs: Optional mapping with keyword arguments to forward to
        `text.BertTokenizer`'s constructor.
      **kwargs: Standard arguments to `Layer()`.

    Raises:
      ImportError: If importing `tensorflow_text` failed.
    """
    _check_if_tf_text_installed()

    self.tokenize_with_offsets = tokenize_with_offsets
    # TODO(b/177326279): Stop storing the vocab table initializer as an
    # attribute when https://github.com/tensorflow/tensorflow/issues/46456
    # has been fixed in the TensorFlow versions of the TF Hub users that load
    # a SavedModel created from this layer. Due to that issue, loading such a
    # SavedModel forgets to add .vocab_table._initializer as a trackable
    # dependency of .vocab_table, so that saving it again to a second SavedModel
    # (e.g., the final model built using TF Hub) does not properly track
    # the ._vocab_table._initializer._filename as an Asset.
    self._vocab_table, self._vocab_initializer_donotuse = (
        self._create_vocab_table_and_initializer(vocab_file))
    self._special_tokens_dict = self._create_special_tokens_dict(
        self._vocab_table, vocab_file)
    super().__init__(**kwargs)
    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    if lower_case is not None:
      tokenizer_kwargs["lower_case"] = lower_case
    self._bert_tokenizer = text.BertTokenizer(self._vocab_table,
                                              **tokenizer_kwargs)

  @property
  def vocab_size(self):
    return self._vocab_table.size()

  def _create_vocab_table_and_initializer(self, vocab_file):
    vocab_initializer = tf.lookup.TextFileInitializer(
        vocab_file,
        key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
    vocab_table = tf.lookup.StaticHashTable(vocab_initializer, default_value=-1)
    return vocab_table, vocab_initializer

  def call(self, inputs: tf.Tensor):
    """Calls `text.BertTokenizer` on inputs.

    Args:
      inputs: A string Tensor of shape `(batch_size,)`.

    Returns:
      One or three of `RaggedTensors` if `tokenize_with_offsets` is False or
      True, respectively. These are
        tokens: A `RaggedTensor` of shape
          `[batch_size, (words), (pieces_per_word)]`
          and type int32. `tokens[i,j,k]` contains the k-th wordpiece of the
          j-th word in the i-th input.
        start_offsets, limit_offsets: If `tokenize_with_offsets` is True,
          RaggedTensors of type int64 with the same indices as tokens.
          Element `[i,j,k]` contains the byte offset at the start, or past the
          end, resp., for the k-th wordpiece of the j-th word in the i-th input.
    """
    # Prepare to reshape the result to work around broken shape inference.
    batch_size = tf.shape(inputs)[0]
    def _reshape(rt):
      values = rt.values
      row_splits = rt.row_splits
      row_splits = tf.reshape(row_splits, [batch_size + 1])
      return tf.RaggedTensor.from_row_splits(values, row_splits)

    # Call the tokenizer.
    if self.tokenize_with_offsets:
      tokens, start_offsets, limit_offsets = (
          self._bert_tokenizer.tokenize_with_offsets(inputs))
      tokens = tf.cast(tokens, dtype=tf.int32)
      return _reshape(tokens), _reshape(start_offsets), _reshape(limit_offsets)
    else:
      tokens = self._bert_tokenizer.tokenize(inputs)
      tokens = tf.cast(tokens, dtype=tf.int32)
      return _reshape(tokens)

  def get_config(self):
    # Skip in tf.saved_model.save(); fail if called direcly.
    raise NotImplementedError("TODO(b/170480226): implement")

  def get_special_tokens_dict(self):
    """Returns dict of token ids, keyed by standard names for their purpose.

    Returns:
      A dict from Python strings to Python integers. Each key is a standard
      name for a special token describing its use. (For example, "padding_id"
      is what BERT traditionally calls "[PAD]" but others may call "<pad>".)
      The corresponding value is the integer token id. If a special token
      is not found, its entry is omitted from the dict.

      The supported keys and tokens are:
        * start_of_sequence_id: looked up from "[CLS]"
        * end_of_segment_id: looked up from "[SEP]"
        * padding_id: looked up form "[PAD]"
        * mask_id: looked up from "[MASK]"
        * vocab_size: one past the largest token id used
    """
    return self._special_tokens_dict

  def _create_special_tokens_dict(self, vocab_table, vocab_file):
    special_tokens = dict(start_of_sequence_id="[CLS]",
                          end_of_segment_id="[SEP]",
                          padding_id="[PAD]",
                          mask_id="[MASK]")
    with tf.init_scope():
      if tf.executing_eagerly():
        special_token_ids = vocab_table.lookup(
            tf.constant(list(special_tokens.values()), tf.string))
        vocab_size = vocab_table.size()
      else:
        # A blast from the past: non-eager init context while building Model.
        # This can happen with Estimator or tf.compat.v1.disable_v2_behavior().
        logging.warning(
            "Non-eager init context; computing "
            "BertTokenizer's special_tokens_dict in tf.compat.v1.Session")
        with tf.Graph().as_default():
          local_vocab_table, _ = self._create_vocab_table_and_initializer(
              vocab_file)
          special_token_ids_tensor = local_vocab_table.lookup(
              tf.constant(list(special_tokens.values()), tf.string))
          vocab_size_tensor = local_vocab_table.size()
          init_ops = [tf.compat.v1.initialize_all_tables()]
          with tf.compat.v1.Session() as sess:
            sess.run(init_ops)
            special_token_ids, vocab_size = sess.run(
                [special_token_ids_tensor, vocab_size_tensor])
      result = dict(
          vocab_size=int(vocab_size)  # Numpy to Python.
      )
      for k, v in zip(special_tokens, special_token_ids):
        v = int(v)
        if v >= 0:
          result[k] = v
        else:
          logging.warning("Could not find %s as token \"%s\" in vocab file %s",
                          k, special_tokens[k], vocab_file)
    return result


class SentencepieceTokenizer(tf_keras.layers.Layer):
  """Wraps `tf_text.SentencepieceTokenizer` as a Keras Layer.

  Attributes:
    tokenize_with_offsets: If true, calls
      `SentencepieceTokenizer.tokenize_with_offsets()`
      instead of plain `.tokenize()` and outputs a triple of
      `(tokens, start_offsets, limit_offsets)`.
  """

  def __init__(self,
               *,
               lower_case: bool,
               model_file_path: Optional[str] = None,
               model_serialized_proto: Optional[str] = None,
               tokenize_with_offsets: bool = False,
               nbest_size: int = 0,
               alpha: float = 1.0,
               strip_diacritics: bool = False,
               **kwargs):
    """Initializes a SentencepieceTokenizer layer.

    Args:
      lower_case: A Python boolean indicating whether to lowercase the string
        before tokenization. NOTE: New models are encouraged to build `*_cf`
        (case folding) normalization into the Sentencepiece model itself and
        avoid this extra step.
      model_file_path: A Python string with the path of the sentencepiece model.
        Exactly one of `model_file_path` and `model_serialized_proto` can be
        specified. In either case, the Keras model config for this layer will
        store the actual proto (not a filename passed here).
      model_serialized_proto: The sentencepiece model serialized proto string.
      tokenize_with_offsets: A Python boolean. If true, this layer calls
        `SentencepieceTokenizer.tokenize_with_offsets()` instead of
        plain `.tokenize()` and outputs a triple of
        `(tokens, start_offsets, limit_offsets)` insead of just tokens.
        Note that when following `strip_diacritics` is set to True, returning
        offsets is not supported now.
      nbest_size: A scalar for sampling:
        nbest_size = {0,1}: No sampling is performed. (default)
        nbest_size > 1: samples from the nbest_size results.
        nbest_size < 0: assuming that nbest_size is infinite and samples
           from the all hypothesis (lattice) using
           forward-filtering-and-backward-sampling algorithm.
      alpha: A scalar for a smoothing parameter. Inverse temperature for
        probability rescaling.
      strip_diacritics: Whether to strip diacritics or not. Note that stripping
        diacritics requires additional text normalization and dropping bytes,
        which makes it impossible to keep track of the offsets now. Hence
        when `strip_diacritics` is set to True, we don't yet support
        `tokenize_with_offsets`. NOTE: New models are encouraged to put this
        into custom normalization rules for the Sentencepiece model itself to
        avoid this extra step and the limitation regarding offsets.
      **kwargs: standard arguments to `Layer()`.

    Raises:
      ImportError: if importing tensorflow_text failed.
    """
    _check_if_tf_text_installed()
    super().__init__(**kwargs)
    if bool(model_file_path) == bool(model_serialized_proto):
      raise ValueError("Exact one of `model_file_path` and "
                       "`model_serialized_proto` can be specified.")
    # TODO(b/181866850): Support tokenize_with_offsets for strip_diacritics=True
    if tokenize_with_offsets and strip_diacritics:
      raise ValueError("`tokenize_with_offsets` is not supported when "
                       "`strip_diacritics` is set to True.")
    if model_file_path:
      self._model_serialized_proto = tf.io.gfile.GFile(model_file_path,
                                                       "rb").read()
    else:
      self._model_serialized_proto = model_serialized_proto

    self._lower_case = lower_case
    self.tokenize_with_offsets = tokenize_with_offsets
    self._nbest_size = nbest_size
    self._alpha = alpha
    self._strip_diacritics = strip_diacritics
    self._tokenizer = self._create_tokenizer()
    self._special_tokens_dict = self._create_special_tokens_dict()

  def _create_tokenizer(self):
    return text.SentencepieceTokenizer(
        model=self._model_serialized_proto,
        out_type=tf.int32,
        nbest_size=self._nbest_size,
        alpha=self._alpha)

  @property
  def vocab_size(self):
    return self._tokenizer.vocab_size()

  def call(self, inputs: tf.Tensor):
    """Calls `text.SentencepieceTokenizer` on inputs.

    Args:
      inputs: A string Tensor of shape `(batch_size,)`.

    Returns:
      One or three of RaggedTensors if tokenize_with_offsets is False or True,
      respectively. These are
      tokens: A RaggedTensor of shape `[batch_size, (pieces)]` and type `int32`.
        `tokens[i,j]` contains the j-th piece in the i-th input.
      start_offsets, limit_offsets: If `tokenize_with_offsets` is True,
        RaggedTensors of type `int64` with the same indices as tokens.
        Element `[i,j]` contains the byte offset at the start, or past the
        end, resp., for the j-th piece in the i-th input.
    """
    if self._strip_diacritics:
      if self.tokenize_with_offsets:
        raise ValueError("`tokenize_with_offsets` is not supported yet when "
                         "`strip_diacritics` is set to True (b/181866850).")
      inputs = text.normalize_utf8(inputs, "NFD")
      inputs = tf.strings.regex_replace(inputs, r"\p{Mn}", "")

    if self._lower_case:
      inputs = text.case_fold_utf8(inputs)

    # Prepare to reshape the result to work around broken shape inference.
    batch_size = tf.shape(inputs)[0]
    def _reshape(rt):
      values = rt.values
      row_splits = rt.row_splits
      row_splits = tf.reshape(row_splits, [batch_size + 1])
      return tf.RaggedTensor.from_row_splits(values, row_splits)

    # Call the tokenizer.
    if self.tokenize_with_offsets:
      tokens, start_offsets, limit_offsets = (
          self._tokenizer.tokenize_with_offsets(inputs))
      return _reshape(tokens), _reshape(start_offsets), _reshape(limit_offsets)
    else:
      tokens = self._tokenizer.tokenize(inputs)
      return _reshape(tokens)

  def get_config(self):
    # Skip in tf.saved_model.save(); fail if called direcly.
    raise NotImplementedError("TODO(b/170480226): implement")

  def get_special_tokens_dict(self):
    """Returns dict of token ids, keyed by standard names for their purpose.

    Returns:
      A dict from Python strings to Python integers. Each key is a standard
      name for a special token describing its use. (For example, "padding_id"
      is what Sentencepiece calls "<pad>" but others may call "[PAD]".)
      The corresponding value is the integer token id. If a special token
      is not found, its entry is omitted from the dict.

      The supported keys and tokens are:
        * start_of_sequence_id: looked up from "[CLS]"
        * end_of_segment_id: looked up from "[SEP]"
        * padding_id: looked up from "<pad>"
        * mask_id: looked up from "[MASK]"
        * vocab_size: one past the largest token id used
    """
    return self._special_tokens_dict

  def _create_special_tokens_dict(self):
    special_tokens = dict(
        start_of_sequence_id=b"[CLS]",
        end_of_segment_id=b"[SEP]",
        padding_id=b"<pad>",
        mask_id=b"[MASK]")
    with tf.init_scope():
      if tf.executing_eagerly():
        special_token_ids = self._tokenizer.string_to_id(
            tf.constant(list(special_tokens.values()), tf.string))
        inverse_tokens = self._tokenizer.id_to_string(special_token_ids)
        vocab_size = self._tokenizer.vocab_size()
      else:
        # A blast from the past: non-eager init context while building Model.
        # This can happen with Estimator or tf.compat.v1.disable_v2_behavior().
        logging.warning(
            "Non-eager init context; computing SentencepieceTokenizer's "
            "special_tokens_dict in tf.compat.v1.Session")
        with tf.Graph().as_default():
          local_tokenizer = self._create_tokenizer()
          special_token_ids_tensor = local_tokenizer.string_to_id(
              tf.constant(list(special_tokens.values()), tf.string))
          inverse_tokens_tensor = local_tokenizer.id_to_string(
              special_token_ids_tensor)
          vocab_size_tensor = local_tokenizer.vocab_size()
          with tf.compat.v1.Session() as sess:
            special_token_ids, inverse_tokens, vocab_size = sess.run(
                [special_token_ids_tensor, inverse_tokens_tensor,
                 vocab_size_tensor])
      result = dict(
          vocab_size=int(vocab_size)  # Numpy to Python.
      )
      for name, token_id, inverse_token in zip(special_tokens,
                                               special_token_ids,
                                               inverse_tokens):
        if special_tokens[name] == inverse_token:
          result[name] = int(token_id)
        else:
          logging.warning(
              "Could not find %s as token \"%s\" in sentencepiece model, "
              "got \"%s\"", name, special_tokens[name], inverse_token)
    return result


class BertPackInputs(tf_keras.layers.Layer):
  """Packs tokens into model inputs for BERT."""

  def __init__(self,
               seq_length,
               *,
               start_of_sequence_id=None,
               end_of_segment_id=None,
               padding_id=None,
               special_tokens_dict=None,
               truncator="round_robin",
               **kwargs):
    """Initializes with a target `seq_length`, relevant token ids and truncator.

    Args:
      seq_length: The desired output length. Must not exceed the max_seq_length
        that was fixed at training time for the BERT model receiving the inputs.
      start_of_sequence_id: The numeric id of the token that is to be placed
        at the start of each sequence (called "[CLS]" for BERT).
      end_of_segment_id: The numeric id of the token that is to be placed
        at the end of each input segment (called "[SEP]" for BERT).
      padding_id: The numeric id of the token that is to be placed into the
        unused positions after the last segment in the sequence
        (called "[PAD]" for BERT).
      special_tokens_dict: Optionally, a dict from Python strings to Python
        integers that contains values for `start_of_sequence_id`,
        `end_of_segment_id` and `padding_id`. (Further values in the dict are
        silenty ignored.) If this is passed, separate *_id arguments must be
        omitted.
      truncator: The algorithm to truncate a list of batched segments to fit a
        per-example length limit. The value can be either `round_robin` or
        `waterfall`:
          (1) For "round_robin" algorithm, available space is assigned
          one token at a time in a round-robin fashion to the inputs that still
          need some, until the limit is reached. It currently only supports
          one or two segments.
          (2) For "waterfall" algorithm, the allocation of the budget is done
            using a "waterfall" algorithm that allocates quota in a
            left-to-right manner and fills up the buckets until we run out of
            budget. It support arbitrary number of segments.

      **kwargs: standard arguments to `Layer()`.

    Raises:
      ImportError: if importing `tensorflow_text` failed.
    """
    _check_if_tf_text_installed()
    super().__init__(**kwargs)
    self.seq_length = seq_length
    if truncator not in ("round_robin", "waterfall"):
      raise ValueError("Only 'round_robin' and 'waterfall' algorithms are "
                       "supported, but got %s" % truncator)
    self.truncator = truncator
    self._init_token_ids(
        start_of_sequence_id=start_of_sequence_id,
        end_of_segment_id=end_of_segment_id,
        padding_id=padding_id,
        special_tokens_dict=special_tokens_dict)

  def _init_token_ids(
      self, *,
      start_of_sequence_id,
      end_of_segment_id,
      padding_id,
      special_tokens_dict):
    usage = ("Must pass either all of start_of_sequence_id, end_of_segment_id, "
             "padding_id as arguments, or else a special_tokens_dict "
             "with those keys.")
    special_tokens_args = [start_of_sequence_id, end_of_segment_id, padding_id]
    if special_tokens_dict is None:
      if any(x is None for x in special_tokens_args):
        return ValueError(usage)
      self.start_of_sequence_id = int(start_of_sequence_id)
      self.end_of_segment_id = int(end_of_segment_id)
      self.padding_id = int(padding_id)
    else:
      if any(x is not None for x in special_tokens_args):
        return ValueError(usage)
      self.start_of_sequence_id = int(
          special_tokens_dict["start_of_sequence_id"])
      self.end_of_segment_id = int(special_tokens_dict["end_of_segment_id"])
      self.padding_id = int(special_tokens_dict["padding_id"])

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config["seq_length"] = self.seq_length
    config["start_of_sequence_id"] = self.start_of_sequence_id
    config["end_of_segment_id"] = self.end_of_segment_id
    config["padding_id"] = self.padding_id
    config["truncator"] = self.truncator
    return config

  def call(self, inputs: Union[tf.RaggedTensor, List[tf.RaggedTensor]]):
    """Adds special tokens to pack a list of segments into BERT input Tensors.

    Args:
      inputs: A Python list of one or two RaggedTensors, each with the batched
        values one input segment. The j-th segment of the i-th input example
        consists of slice `inputs[j][i, ...]`.

    Returns:
      A nest of Tensors for use as input to the BERT TransformerEncoder.
    """
    # BertPackInputsSavedModelWrapper relies on only calling bert_pack_inputs()
    return BertPackInputs.bert_pack_inputs(
        inputs, self.seq_length,
        start_of_sequence_id=self.start_of_sequence_id,
        end_of_segment_id=self.end_of_segment_id,
        padding_id=self.padding_id,
        truncator=self.truncator)

  @staticmethod
  def bert_pack_inputs(inputs: Union[tf.RaggedTensor, List[tf.RaggedTensor]],
                       seq_length: Union[int, tf.Tensor],
                       start_of_sequence_id: Union[int, tf.Tensor],
                       end_of_segment_id: Union[int, tf.Tensor],
                       padding_id: Union[int, tf.Tensor],
                       truncator="round_robin"):
    """Freestanding equivalent of the BertPackInputs layer."""
    _check_if_tf_text_installed()
    # Sanitize inputs.
    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]
    if not inputs:
      raise ValueError("At least one input is required for packing")
    input_ranks = [rt.shape.rank for rt in inputs]
    if None in input_ranks or len(set(input_ranks)) > 1:
      raise ValueError("All inputs for packing must have the same known rank, "
                       "found ranks " + ",".join(input_ranks))
    # Flatten inputs to [batch_size, (tokens)].
    if input_ranks[0] > 2:
      inputs = [rt.merge_dims(1, -1) for rt in inputs]
    # In case inputs weren't truncated (as they should have been),
    # fall back to some ad-hoc truncation.
    num_special_tokens = len(inputs) + 1
    if truncator == "round_robin":
      trimmed_segments = text.RoundRobinTrimmer(seq_length -
                                                num_special_tokens).trim(inputs)
    elif truncator == "waterfall":
      trimmed_segments = text.WaterfallTrimmer(
          seq_length - num_special_tokens).trim(inputs)
    else:
      raise ValueError("Unsupported truncator: %s" % truncator)
    # Combine segments.
    segments_combined, segment_ids = text.combine_segments(
        trimmed_segments,
        start_of_sequence_id=start_of_sequence_id,
        end_of_segment_id=end_of_segment_id)
    # Pad to dense Tensors.
    input_word_ids, _ = text.pad_model_inputs(segments_combined, seq_length,
                                              pad_value=padding_id)
    input_type_ids, input_mask = text.pad_model_inputs(segment_ids, seq_length,
                                                       pad_value=0)
    # Work around broken shape inference.
    output_shape = tf.stack([
        inputs[0].nrows(out_type=tf.int32),  # batch_size
        tf.cast(seq_length, dtype=tf.int32)])
    def _reshape(t):
      return tf.reshape(t, output_shape)
    # Assemble nest of input tensors as expected by BERT TransformerEncoder.
    return dict(input_word_ids=_reshape(input_word_ids),
                input_mask=_reshape(input_mask),
                input_type_ids=_reshape(input_type_ids))


class FastWordpieceBertTokenizer(tf_keras.layers.Layer):
  """A bert tokenizer keras layer using text.FastWordpieceTokenizer.

  See details: "Fast WordPiece Tokenization" (https://arxiv.org/abs/2012.15524)
  """

  def __init__(self,
               *,
               vocab_file: str,
               lower_case: bool,
               tokenize_with_offsets: bool = False,
               **kwargs):
    """Initializes a FastWordpieceBertTokenizer layer.

    Args:
      vocab_file: A Python string with the path of the vocabulary file. This is
        a text file with newline-separated wordpiece tokens. This layer loads
        a list of tokens from it to create text.FastWordpieceTokenizer.
      lower_case: A Python boolean forwarded to text.BasicTokenizer. If true,
        input text is converted to lower case (where applicable) before
        tokenization. This must be set to match the way in which the vocab_file
        was created.
      tokenize_with_offsets: A Python boolean. If true, this layer calls
        FastWordpieceTokenizer.tokenize_with_offsets() instead of plain
        .tokenize() and outputs a triple of (tokens, start_offsets,
        limit_offsets) insead of just tokens.
      **kwargs: standard arguments to Layer().
    """
    super().__init__(**kwargs)
    logging.info("Initialize a FastWordpieceBertTokenizer.")
    self.tokenize_with_offsets = tokenize_with_offsets
    self._basic_tokenizer = bert_tokenizer.BasicTokenizer(lower_case=lower_case)

    # Read the vocab file into a list of tokens to create `fast_wp_tokenizer`.
    self._vocab = [line.rstrip() for line in tf.io.gfile.GFile(vocab_file)]
    self._fast_wp_tokenizer = text.FastWordpieceTokenizer(
        vocab=self._vocab, token_out_type=tf.int32, no_pretokenization=True)
    self._special_tokens_dict = self._create_special_tokens_dict()

  @property
  def vocab_size(self):
    return len(self._vocab)

  def get_config(self):
    # Skip in tf.saved_model.save(); fail if called direcly.
    # We cannot just put the original, user-supplied vocab file name into
    # the config, because the path has to change as the SavedModel is copied
    # around.
    raise NotImplementedError("Not implemented yet.")

  def get_special_tokens_dict(self):
    """Returns dict of token ids, keyed by standard names for their purpose.

    Returns:
      A dict from Python strings to Python integers. Each key is a standard
      name for a special token describing its use. (For example, "padding_id"
      is what BERT traditionally calls "[PAD]" but others may call "<pad>".)
      The corresponding value is the integer token id. If a special token
      is not found, its entry is omitted from the dict.

      The supported keys and tokens are:
        * start_of_sequence_id: looked up from "[CLS]"
        * end_of_segment_id: looked up from "[SEP]"
        * padding_id: looked up form "[PAD]"
        * mask_id: looked up from "[MASK]"
        * vocab_size: one past the largest token id used
    """
    return self._special_tokens_dict

  def _create_special_tokens_dict(self):
    """Creates dict of token ids, keyed by standard names for their purpose."""
    special_tokens = {"vocab_size": self.vocab_size}

    def add_special_token(key, token):
      try:
        token_id = self._vocab.index(token)
        special_tokens[key] = token_id
      except ValueError:
        # Similar as nlp.modeling.layers.BertTokenizer, if a special token
        # is not found, its entry is omitted from the dict.
        logging.warning("Could not find %s as token \"%s\" in vocab file", key,
                        token)

    add_special_token("start_of_sequence_id", "[CLS]")
    add_special_token("end_of_segment_id", "[SEP]")
    add_special_token("padding_id", "[PAD]")
    add_special_token("mask_id", "[MASK]")
    return special_tokens

  def _tokenize_with_offsets(self, text_input: tf.Tensor):
    tokens, begin, _ = self._basic_tokenizer.tokenize_with_offsets(text_input)
    wordpieces, wp_begin, wp_end = (
        self._fast_wp_tokenizer.tokenize_with_offsets(tokens))
    begin_expanded = tf.expand_dims(begin, axis=2)
    final_begin = begin_expanded + wp_begin
    final_end = begin_expanded + wp_end
    return wordpieces, final_begin, final_end

  def _tokenize(self, text_input: tf.Tensor):
    tokens = self._basic_tokenizer.tokenize(text_input)
    return self._fast_wp_tokenizer.tokenize(tokens)

  def call(self, inputs: tf.Tensor):
    """Calls text.BertTokenizer on inputs.

    Args:
      inputs: A string Tensor of shape [batch_size].

    Returns:
      One or three of RaggedTensors if tokenize_with_offsets is False or True,
      respectively. These are
      tokens: A RaggedTensor of shape [batch_size, (words), (pieces_per_word)]
        and type int32. tokens[i,j,k] contains the k-th wordpiece of the
        j-th word in the i-th input.
      start_offsets, limit_offsets: If tokenize_with_offsets is True,
        RaggedTensors of type int64 with the same indices as tokens.
        Element [i,j,k] contains the byte offset at the start, or past the
        end, resp., for the k-th wordpiece of the j-th word in the i-th input.
    """
    # Prepare to reshape the result to work around broken shape inference.
    batch_size = tf.shape(inputs)[0]

    def _reshape(rt):
      values = rt.values
      row_splits = rt.row_splits
      row_splits = tf.reshape(row_splits, [batch_size + 1])
      return tf.RaggedTensor.from_row_splits(values, row_splits)

    if self.tokenize_with_offsets:
      tokens, start_offsets, limit_offsets = self._tokenize_with_offsets(inputs)
      return _reshape(tokens), _reshape(start_offsets), _reshape(limit_offsets)
    else:
      tokens = self._tokenize(inputs)
      return _reshape(tokens)
