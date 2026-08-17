# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from pathlib import Path

from evalplus_utils import (
    assemble_humaneval_code,
    extract_completion,
    evaluate_humaneval_plus,
    evaluate_mbpp_plus,
    humaneval_plus_query,
    mbpp_plus_query,
)


PROMPT = '''def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
'''

CANONICAL = "    return a + b\n"

TEST = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0
"""

MBPP_TEST = """
def check():
    assert add(1, 2) == 3
    assert add(0, 0) == 0

check()
"""


def test_extract_completion_strips_prompt_echo_and_fences():
    echoed = PROMPT + CANONICAL
    assert extract_completion(echoed, PROMPT) == CANONICAL

    fenced = "Here is the solution:\n```python\n    return a + b\n```\n"
    assert extract_completion(fenced, PROMPT) == "    return a + b\n"


def test_assemble_humaneval_code_accepts_full_function_rewrite():
    rewrite = "def add(a: int, b: int) -> int:\n    return a + b\n"
    assert assemble_humaneval_code(PROMPT, rewrite) == rewrite
    assert assemble_humaneval_code(PROMPT, CANONICAL) == PROMPT + CANONICAL


def test_humaneval_plus_query_is_completion_style():
    line = {"prompt": PROMPT, "entry_point": "add", "test": TEST}
    assert humaneval_plus_query(line) == PROMPT


def test_mbpp_plus_query_includes_problem_and_visible_tests():
    line = {
        "prompt": "Write a function to add two numbers.",
        "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    }
    query = mbpp_plus_query(line)
    assert "Write a function to add two numbers." in query
    assert "assert add(1, 2) == 3" in query
    assert "expert Python programmer" in query


def test_evaluate_humaneval_plus_accepts_canonical_and_rejects_wrong():
    assert evaluate_humaneval_plus(PROMPT, CANONICAL, TEST, "add")
    assert not evaluate_humaneval_plus(PROMPT, "    return a - b\n", TEST, "add")


def test_evaluate_humaneval_plus_accepts_fenced_completion():
    prediction = "```python\n    return a + b\n```"
    assert evaluate_humaneval_plus(PROMPT, prediction, TEST, "add")


def test_evaluate_humaneval_plus_times_out_on_infinite_loop():
    looping = "    while True:\n        pass\n"
    assert not evaluate_humaneval_plus(PROMPT, looping, TEST, "add", timeout=1.0)


def test_evaluate_mbpp_plus_accepts_canonical_and_rejects_wrong():
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    assert evaluate_mbpp_plus(good, MBPP_TEST)
    assert not evaluate_mbpp_plus(bad, MBPP_TEST)


def test_smollm3_base_task_list_includes_evalplus_tasks():
    task_list = Path(__file__).with_name("smollm3_base.txt").read_text(encoding="utf-8")
    assert "custom|humaneval_plus|0|0" in task_list
    assert "custom|mbpp_plus|0|0" in task_list


def test_readme_points_at_existing_base_task_file():
    readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
    assert "smollm3_base.txt" in readme
    assert "smollm3_base_test.txt" not in readme
    assert "humaneval_plus" in readme
    assert "mbpp_plus" in readme
