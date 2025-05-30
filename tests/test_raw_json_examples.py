#!/usr/bin/env python3
"""
Test Raw JSON Examples - Demonstrating robust parser with raw JSON responses.

Shows how the parser handles LLMs that just return straight JSON without any formatting.
"""

import pytest
import json
import sys
import os

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.json_parser import extract_json_robust


def test_raw_json_responses():
    """Test that the parser handles raw JSON responses (no code blocks)."""

    # Example 1: LLM just returns raw JSON
    raw_response_1 = '{"question_text": "What is 2+2?", "marks": 1, "answer": "4"}'

    result = extract_json_robust(raw_response_1)
    assert result.success == True
    assert result.data["question_text"] == "What is 2+2?"
    assert result.extraction_method in ["largest_json_object", "brackets"]

    # Example 2: Raw JSON with unicode characters
    raw_response_2 = '{"formula": "A = π × r²", "answer": "78.54 cm²", "symbols": ["π", "²", "×"]}'

    result = extract_json_robust(raw_response_2)
    assert result.success == True
    assert result.data["formula"] == "A = π × r²"
    assert "π" in result.data["symbols"]

    # Example 3: Raw JSON with nested structure
    raw_response_3 = '''{"question": {"text": "Calculate area", "marks": 3}, "solution": {"steps": [{"step": 1, "desc": "Apply formula"}]}}'''

    result = extract_json_robust(raw_response_3)
    assert result.success == True
    assert result.data["question"]["text"] == "Calculate area"
    assert result.data["solution"]["steps"][0]["step"] == 1


def test_mixed_format_responses():
    """Test responses that mix text with raw JSON."""

    # Example 1: Explanation before JSON
    mixed_response_1 = '''Here's the generated question:

    {"question_id": "Q123", "text": "Calculate the perimeter", "marks": 2}

    This should meet the requirements.'''

    result = extract_json_robust(mixed_response_1)
    assert result.success == True
    assert result.data["question_id"] == "Q123"
    assert result.data["marks"] == 2

    # Example 2: JSON with leading/trailing text
    mixed_response_2 = '''I'll create the question now: {"question": "Find the area of a circle with radius 7 cm", "marks": 3, "topic": "geometry"} - this covers the required topic.'''

    result = extract_json_robust(mixed_response_2)
    assert result.success == True
    assert "circle" in result.data["question"]
    assert result.data["topic"] == "geometry"


def test_malformed_raw_json_fixes():
    """Test that malformed raw JSON gets fixed automatically."""

    # Example 1: Trailing comma in raw JSON
    malformed_1 = '{"question": "What is 5×3?", "marks": 1,}'

    result = extract_json_robust(malformed_1)
    assert result.success == True
    assert result.data["question"] == "What is 5×3?"
    assert result.extraction_method == "fixed_common_issues"

    # Example 2: Single quotes in raw JSON
    malformed_2 = "{'question_text': 'Calculate the area', 'marks': 2}"

    result = extract_json_robust(malformed_2)
    assert result.success == True
    assert result.data["question_text"] == "Calculate the area"

    # Example 3: Unquoted keys in raw JSON
    malformed_3 = '{question: "Find the volume", marks: 4, topic: "3D shapes"}'

    result = extract_json_robust(malformed_3)
    assert result.success == True
    assert result.data["question"] == "Find the volume"
    assert result.data["marks"] == 4


def test_real_world_llm_variations():
    """Test actual variations we might see from different LLMs."""

    # ChatGPT style - often wraps in explanations
    gpt_response = '''I'll generate a question following your requirements:

{"question_id_local": "Gen_Q5678", "raw_text_content": "A circle has a diameter of 14 cm. Calculate its area.", "marks": 3, "command_word": "Calculate"}

This question tests circle area calculation at the appropriate level.'''

    result = extract_json_robust(gpt_response)
    assert result.success == True
    assert "diameter" in result.data["raw_text_content"]

    # Claude style - sometimes returns just JSON
    claude_response = '{"question_id_local": "Ref_Q9876", "raw_text_content": "Find the circumference of a circle with radius 8.5 cm. Give your answer to 1 decimal place.", "marks": 2}'

    result = extract_json_robust(claude_response)
    assert result.success == True
    assert "circumference" in result.data["raw_text_content"]

    # Gemini style - sometimes adds reasoning
    gemini_response = '''Based on the requirements, here's an appropriate question:

```json
{"question_text": "A rectangular garden is 12m long and 8m wide. Calculate its perimeter.", "marks": 2, "topic": "measurement"}
```

This tests basic perimeter calculation.'''

    result = extract_json_robust(gemini_response)
    assert result.success == True
    assert "rectangular" in result.data["question_text"]
    assert result.extraction_method == "json_code_block"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
