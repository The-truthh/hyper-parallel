# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Thin launchers for full-head DSA CP/FSDP kernel parity."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_dsa_full_heads_cp2():
    """Compare CP2 to the native full-sequence DSA/FSDP gradient oracle."""
    torchrun_case(str(Path(__file__).with_name("_test_dsa_full_head_cp.py")),
                  "test_dsa_cp_fsdp_full_heads", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_dsa_full_heads_cp4():
    """Cover the index-head geometry that previously exposed the KL tail bug."""
    torchrun_case(str(Path(__file__).with_name("_test_dsa_full_head_cp.py")),
                  "test_dsa_cp_fsdp_full_heads", num_proc=4)
