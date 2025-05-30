#!/usr/bin/env python3
"""
Test Robust JSON Parser - Testing various LLM output formats.

Tests the robust JSON extraction capabilities for different formats
that LLMs might output.
"""

import pytest
import json
import sys
import os

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.json_parser import (
    RobustJSONParser,
    JSONExtractionResult,
    extract_json_robust,
    extract_json_or_none,
    extract_json_with_fallback
)


class TestRobustJSONParser:
    """Test cases for the robust JSON parser."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return RobustJSONParser(debug=True)

    @pytest.fixture
    def sample_data(self):
        """Sample JSON data for testing."""
        return {
            "question_id": "Q123",
            "text": "Calculate the area of a circle with radius 5 cm.",
            "marks": 3,
            "answer": "78.54 cm²",
            "symbols": "π × r²"
        }

    def test_json_code_block_extraction(self, parser, sample_data):
        """Test extraction from ```json code blocks."""

        response = f"""
        Here's the refined question:

        ```json
        {json.dumps(sample_data)}
        ```

        This should work well!
        """

        result = parser.extract_json(response)

        assert result.success == True
        assert result.data == sample_data
        assert result.extraction_method == "json_code_block"

    def test_any_code_block_extraction(self, parser, sample_data):
        """Test extraction from any ``` code blocks."""

        response = f"""
        Here's the response:

        ```
        {json.dumps(sample_data)}
        ```
        """

        result = parser.extract_json(response)

        assert result.success == True
        assert result.data == sample_data
        assert result.extraction_method == "any_code_block"

    def test_raw_json_extraction(self, parser, sample_data):
        """Test extraction of raw JSON mixed with text."""

        response = f"""
        The refined question is: {json.dumps(sample_data)}

        Let me know if you need any changes!
        """

        result = parser.extract_json(response)

        assert result.success == True
        assert result.data == sample_data

    def test_multiple_json_objects(self, parser):
        """Test handling of multiple JSON objects."""

        obj1 = {"type": "question", "id": 1}
        obj2 = {"type": "answer", "id": 2}

        response = f"""
        First object: {json.dumps(obj1)}
        Second object: {json.dumps(obj2)}
        """

        # Should extract the largest/first one
        result = parser.extract_json(response)
        assert result.success == True

        # Test multiple extraction
        results = parser.extract_multiple_json(response)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_trailing_comma_fix(self, parser):
        """Test fixing trailing commas."""

        malformed_json = '''
        {
            "question": "What is 2+2?",
            "answer": 4,
            "marks": 1,
        }
        '''

        result = parser.extract_json(malformed_json)

        assert result.success == True
        assert result.data["question"] == "What is 2+2?"
        assert result.extraction_method == "fixed_common_issues"

    def test_single_quotes_fix(self, parser):
        """Test fixing single quotes."""

        malformed_json = '''
        {
            'question': 'What is the area?',
            'answer': '78.54',
            'unit': 'cm²'
        }
        '''

        result = parser.extract_json(malformed_json)

        assert result.success == True
        assert result.data["question"] == "What is the area?"

    def test_unquoted_keys_fix(self, parser):
        """Test fixing unquoted keys."""

        malformed_json = '''
        {
            question: "What is the area?",
            answer: "78.54",
            marks: 3
        }
        '''

        result = parser.extract_json(malformed_json)

        assert result.success == True
        assert result.data["question"] == "What is the area?"

    def test_complex_nested_structure(self, parser):
        """Test complex nested JSON structures."""

        complex_data = {
            "question": {
                "text": "Calculate area",
                "marks": 3,
                "metadata": {
                    "topic": "geometry",
                    "difficulty": "medium"
                }
            },
            "solution": {
                "steps": [
                    {"step": 1, "description": "Apply formula"},
                    {"step": 2, "description": "Calculate result"}
                ],
                "answer": "78.54 cm²"
            }
        }

        response = f"""
        Here's the complex question structure:

        ```json
        {json.dumps(complex_data, indent=2)}
        ```
        """

        result = parser.extract_json(response)

        assert result.success == True
        assert result.data == complex_data

    def test_special_characters_handling(self, parser):
        """Test handling of special characters."""

        data_with_special_chars = {
            "text": "Calculate π × r² for radius 5 cm",
            "formula": "A = πr²",
            "answer": "≈ 78.54 cm²",
            "symbols": ["π", "²", "×", "≈"]
        }

        response = f"""
        ```json
        {json.dumps(data_with_special_chars)}
        ```
        """

        result = parser.extract_json(response)

        assert result.success == True
        assert result.data == data_with_special_chars

    def test_no_json_found(self, parser):
        """Test handling when no JSON is found."""

        response = "This is just regular text with no JSON content at all."

        result = parser.extract_json(response)

        assert result.success == False
        assert result.error is not None

    def test_empty_response(self, parser):
        """Test handling of empty responses."""

        result = parser.extract_json("")

        assert result.success == False
        assert result.error == "Empty response"

    def test_malformed_json_recovery(self, parser):
        """Test recovery from various malformed JSON."""

        malformed_cases = [
            # Trailing comma
            '{"key": "value",}',
            # Single quotes
            "{'key': 'value'}",
            # Unquoted keys
            '{key: "value"}',
            # Mixed issues
            "{question: 'What is 2+2?', answer: 4,}",
        ]

        for malformed in malformed_cases:
            result = parser.extract_json(malformed)
            # Should either succeed with fixes or fail gracefully
            assert isinstance(result, JSONExtractionResult)

    def test_backticks_extraction(self, parser, sample_data):
        """Test extraction from single backticks."""

        response = f"The result is `{json.dumps(sample_data)}` which looks good."

        result = parser.extract_json(response)

        assert result.success == True
        assert result.data == sample_data

    def test_convenience_functions(self, sample_data):
        """Test convenience functions."""

        response = f"```json\n{json.dumps(sample_data)}\n```"

        # Test extract_json_or_none
        data = extract_json_or_none(response)
        assert data == sample_data

        # Test extract_json_with_fallback
        fallback = {"default": "value"}
        data = extract_json_with_fallback(response, fallback)
        assert data == sample_data

        # Test with invalid response
        data = extract_json_or_none("no json here")
        assert data is None

        data = extract_json_with_fallback("no json here", fallback)
        assert data == fallback

    def test_largest_json_object_selection(self, parser):
        """Test that the largest JSON object is selected when multiple exist."""

        small_obj = {"id": 1}
        large_obj = {
            "id": 2,
            "data": {
                "nested": "value",
                "more_data": ["a", "b", "c"]
            }
        }

        response = f"Small: {json.dumps(small_obj)} Large: {json.dumps(large_obj)}"

        result = parser.extract_json(response)

        assert result.success == True
        assert result.data == large_obj

    def test_debug_mode(self):
        """Test debug mode functionality."""

        debug_parser = RobustJSONParser(debug=True)
        response = '```json\n{"test": "value"}\n```'

        result = debug_parser.extract_json(response)

        assert result.success == True
        assert result.extraction_method == "json_code_block"


def test_integration_with_refinement_scenarios():
    """Test scenarios that might occur in refinement agent usage."""

    parser = RobustJSONParser()

    # Scenario 1: LLM wraps response in explanation
    llm_response_1 = """
    I'll help you refine this question. Here's the improved version:

    ```json
    {
        "question_id_local": "Ref_Q1234",
        "raw_text_content": "Calculate the area of a circle with radius 5 cm. Give your answer to 2 decimal places.",
        "marks": 3
    }
    ```

    This version is clearer and more specific about the required precision.
    """

    result = parser.extract_json(llm_response_1)
    assert result.success == True
    assert "Ref_Q1234" in result.data["question_id_local"]

    # Scenario 2: LLM outputs raw JSON
    llm_response_2 = '{"question_text": "What is the area of a circle?", "marks": 2}'

    result = parser.extract_json(llm_response_2)
    assert result.success == True
    assert result.data["marks"] == 2

    # Scenario 3: LLM has formatting issues
    llm_response_3 = """
    {
        "question_text": "Calculate the area",
        "marks": 3,
    }
    """

    result = parser.extract_json(llm_response_3)
    assert result.success == True
    assert result.data["marks"] == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
