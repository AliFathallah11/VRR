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

"""Ops package definition."""
from official.nlp.modeling.ops.beam_search import sequence_beam_search
from official.nlp.modeling.ops.beam_search import SequenceBeamSearch
from official.nlp.modeling.ops.sampling_module import SamplingModule
from official.nlp.modeling.ops.segment_extractor import get_next_sentence_labels
from official.nlp.modeling.ops.segment_extractor import get_sentence_order_labels
