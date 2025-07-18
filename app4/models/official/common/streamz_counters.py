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

"""Global streamz counters."""

from tensorflow.python.eager import monitoring


progressive_policy_creation_counter = monitoring.Counter(
    "/tensorflow/training/fast_training/progressive_policy_creation",
    "Counter for the number of ProgressivePolicy creations.")


stack_vars_to_vars_call_counter = monitoring.Counter(
    "/tensorflow/training/fast_training/tf_vars_to_vars",
    "Counter for the number of low-level stacking API calls.")
