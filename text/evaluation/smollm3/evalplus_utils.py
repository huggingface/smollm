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

"""Helpers for HumanEval+ / MBPP+ LightEval tasks.

These utilities are intentionally independent of LightEval so they can be unit
tested without downloading models or the EvalPlus datasets. Generated code is
executed in a short-lived subprocess; only use this with trusted eval data.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Iterable

HUMAN_EVAL_STOP_SEQUENCES = [
    "\nclass",
    "\ndef",
    "\n#",
    "\n@",
    "\nprint",
    "\nif",
    "\n```",
]

MBPP_STOP_SEQUENCES = [
    "\nclass",
    "\nassert",
    "\n```",
]

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_completion(prediction: str, prompt: str | None = None) -> str:
    """Return the model completion, stripping prompt echo and markdown fences."""
    text = prediction or ""
    if prompt and text.startswith(prompt):
        text = text[len(prompt) :]

    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1)

    return text


def assemble_humaneval_code(prompt: str, prediction: str) -> str:
    """Build a HumanEval program from a function prefix and a model completion."""
    completion = extract_completion(prediction, prompt)
    stripped = completion.lstrip()
    if stripped.startswith(("def ", "from ", "import ", "class ")):
        return completion
    return f"{prompt}{completion}"


def assemble_mbpp_code(prediction: str) -> str:
    """Build an MBPP program from a free-form model completion."""
    return extract_completion(prediction)


def humaneval_plus_query(line: dict) -> str:
    """Completion-style HumanEval+ prompt used for SmolLM3-3B-Base."""
    return line["prompt"]


def mbpp_plus_query(line: dict) -> str:
    """0-shot MBPP+ prompt used by EvalPlus-style base-model evals."""
    tests = "\n".join(line.get("test_list") or [])
    return (
        f"You are an expert Python programmer, and here is your task: {line['prompt']} "
        f"Your code should pass these tests:\n\n{tests}\n"
    )


def build_humaneval_program(prompt: str, prediction: str, test: str, entry_point: str) -> str:
    code = assemble_humaneval_code(prompt, prediction)
    return f"{code}\n{test}\ncheck({entry_point})\n"


def build_mbpp_program(prediction: str, test: str, test_imports: Iterable[str] | None = None) -> str:
    imports = "\n".join(test_imports or [])
    prefix = f"{imports}\n" if imports else ""
    return f"{prefix}{assemble_mbpp_code(prediction)}\n{test}\n"


def run_program(program: str, timeout: float = 15.0) -> bool:
    """Execute a program in a subprocess and return True if it exits 0."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(program)
            tmp_path = handle.name
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            check=False,
        )
        return completed.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def evaluate_humaneval_plus(
    prompt: str,
    prediction: str,
    test: str,
    entry_point: str,
    timeout: float = 15.0,
) -> bool:
    return run_program(
        build_humaneval_program(prompt, prediction, test, entry_point),
        timeout=timeout,
    )


def evaluate_mbpp_plus(
    prediction: str,
    test: str,
    test_imports: Iterable[str] | None = None,
    timeout: float = 15.0,
) -> bool:
    return run_program(
        build_mbpp_program(prediction, test, test_imports=test_imports),
        timeout=timeout,
    )
