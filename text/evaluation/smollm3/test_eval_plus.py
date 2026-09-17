import pytest
import re

# --- Implementation of Utilities under test (Simulating SmolLM3 EvalPlus utilities) ---

def extract_code_block(text: str) -> str:
    """
    Extracts Python code from markdown blocks or returns cleaned raw text.
    Handles empty/whitespace strings, unclosed blocks, and non-string inputs gracefully.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Check for markdown code fences (e.g., ```python ... ``` or just ``` ... ```)
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the first detected valid block
        return matches[0].strip()
    
    # Fallback: if no markdown fences exist, return stripped raw text if it looks like code,
    # or empty string if it's just whitespace or conversational text without code.
    stripped = text.strip()
    if not stripped:
        return ""
        
    return stripped


def process_eval_results(results: dict) -> dict:
    """
    Parses and sanitizes EvalPlus benchmark payload dictionaries.
    Ensures missing keys or malformed structures default safely without crashing.
    """
    if not isinstance(results, dict):
        raise TypeError("Evaluation results payload must be a dictionary.")
        
    base_score = results.get("pass_at_1", None)
    eval_plus_details = results.get("eval_plus_details", {})
    
    if not isinstance(eval_plus_details, dict):
        eval_plus_details = {}
        
    # Extract sub-metrics safely with robust defaults
    base_metric = eval_plus_details.get("base", 0.0)
    plus_metric = eval_plus_details.get("plus", 0.0)
    
    if base_metric is None:
        base_metric = 0.0
    if plus_metric is None:
        plus_metric = 0.0

    return {
        "pass_at_1": base_score if base_score is not None else 0.0,
        "base": base_metric,
        "plus": plus_metric,
        "score": plus_metric  # primary evaluated metric fallback
    }


# --- Unit Test Suite ---

def test_extract_code_block_empty():
    """Test code extraction when model output is empty, whitespace, or invalid types."""
    empty_outputs = ["", "   ", "\n\n", None, 12345]
    
    for output in empty_outputs:
        # Graceful handling returns empty string for any empty or non-string input
        result = extract_code_block(output) # type: ignore
        assert result == ""


def test_extract_code_block_markdown_variants():
    """Test code extraction across different markdown block styles and multiple blocks."""
    
    # Case 1: Standard python code block with explicit specifier
    sample_1 = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
    expected_1 = "def add(a, b):\n    return a + b"
    assert extract_code_block(sample_1) == expected_1

    # Case 2: Code block without explicit language specifier
    sample_2 = "```\nprint('hello')\n```"
    expected_2 = "print('hello')"
    assert extract_code_block(sample_2) == expected_2

    # Case 3: Multiple code blocks (should extract the first intended solution block)
    sample_3 = "```python\n# first block\n```\n```python\n# second block\n```"
    assert extract_code_block(sample_3) == "# first block"


def test_extract_code_block_malformed():
    """Test handling of unclosed markdown blocks or fallback raw text behavior."""
    unclosed_code = "```python\ndef solution():\n    pass"
    result = extract_code_block(unclosed_code)
    assert isinstance(result, str)


def test_unexpected_evalplus_results():
    """Test resilience against unexpected or malformed EvalPlus benchmark payloads."""
    
    # Simulate an unexpected structure returned from EvalPlus evaluation metrics
    malformed_results = {
        "pass_at_1": None,
        "eval_plus_details": {}  # Missing expected subkeys or containing Nones
    }
    
    parsed = process_eval_results(malformed_results)
    
    # Verify defaults handle missing payload data gracefully without raising exceptions
    assert parsed["pass_at_1"] == 0.0
    assert parsed["base"] == 0.0
    assert parsed["plus"] == 0.0
    assert parsed["score"] == 0.0

    # Test handling of completely invalid payload data types for result processor
    with pytest.raises(TypeError):
        process_eval_results("not_a_dict") # type: ignore