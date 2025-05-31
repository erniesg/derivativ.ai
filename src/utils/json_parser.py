"""
Robust JSON Parser Utility for LLM Responses.

This module provides robust JSON extraction capabilities for various formats
that LLMs might output, including code blocks, mixed text, and malformed JSON.
Includes special handling for thinking models like Qwen and DeepSeek.
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class JSONExtractionResult:
    """Result of JSON extraction attempt."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    raw_json: Optional[str] = None
    extraction_method: Optional[str] = None
    error: Optional[str] = None
    thinking_tokens_removed: bool = False
    original_length: int = 0
    cleaned_length: int = 0


def is_thinking_model(model_name: str) -> bool:
    """Detect if a model is a thinking model that generates reasoning tokens."""
    thinking_model_patterns = [
        'qwen', 'deepseek', 'thinking', 'reasoning', 'o1-', 'o3-'
    ]
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in thinking_model_patterns)


def strip_thinking_tokens(text: str) -> tuple[str, bool]:
    """
    Remove thinking tokens from LLM response.

    Returns:
        (cleaned_text, thinking_tokens_found)
    """
    original_text = text

    # Remove various thinking token formats
    thinking_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<thought>.*?</thought>',
        r'<analysis>.*?</analysis>',
        r'<reasoning>.*?</reasoning>',
        r'<internal>.*?</internal>'
    ]

    for pattern in thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove common thinking prefixes before JSON
    prefixes_to_remove = [
        r'^.*?(?=\{)',  # Remove everything before first {
    ]

    first_brace = text.find('{')
    if first_brace > 0:
        prefix_text = text[:first_brace].strip()
        # Only remove if prefix looks like thinking/explanation
        if any(word in prefix_text.lower() for word in ['let', 'okay', 'sure', 'here', 'i need', 'first', 'to solve']):
            text = text[first_brace:]

    thinking_tokens_found = len(original_text) != len(text)
    return text.strip(), thinking_tokens_found


class RobustJSONParser:
    """
    Robust JSON parser that can handle various LLM output formats.

    Supports:
    - JSON in ```json code blocks
    - JSON in ``` code blocks
    - Raw JSON mixed with text
    - Multiple JSON objects
    - Malformed JSON with common issues
    - Thinking models with reasoning tokens
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def extract_json(self, response: str, model_name: str = None) -> JSONExtractionResult:
        """
        Extract and parse JSON from LLM response using multiple strategies.

        Args:
            response: Raw LLM response text
            model_name: Name of the model (for thinking model detection)

        Returns:
            JSONExtractionResult with parsed data or error info
        """
        if not response or not response.strip():
            return JSONExtractionResult(
                success=False,
                error="Empty response",
                original_length=0,
                cleaned_length=0
            )

        original_length = len(response)

        # Log raw response for debugging (truncated for readability)
        if self.debug:
            preview = response[:500] + "..." if len(response) > 500 else response
            logger.debug(f"Raw LLM response ({len(response)} chars): {preview}")

        # Handle thinking models
        cleaned_response = response
        thinking_tokens_removed = False

        if model_name and is_thinking_model(model_name):
            cleaned_response, thinking_tokens_removed = strip_thinking_tokens(response)
            if self.debug and thinking_tokens_removed:
                logger.debug(f"Stripped thinking tokens: {original_length} -> {len(cleaned_response)} chars")

        # Try different extraction strategies in order of preference
        strategies = [
            self._extract_from_json_code_block,
            self._extract_from_any_code_block,
            self._extract_from_backticks,
            self._extract_largest_json_object,
            self._extract_from_brackets,
            self._extract_and_fix_common_issues
        ]

        for strategy in strategies:
            try:
                result = strategy(cleaned_response)
                if result.success:
                    # Add metadata about thinking token handling
                    result.thinking_tokens_removed = thinking_tokens_removed
                    result.original_length = original_length
                    result.cleaned_length = len(cleaned_response)

                    if self.debug:
                        logger.debug(f"JSON extracted using: {result.extraction_method}")
                    return result
            except Exception as e:
                if self.debug:
                    logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue

        return JSONExtractionResult(
            success=False,
            error="No valid JSON found using any extraction method",
            thinking_tokens_removed=thinking_tokens_removed,
            original_length=original_length,
            cleaned_length=len(cleaned_response)
        )

    def _extract_from_json_code_block(self, response: str) -> JSONExtractionResult:
        """Extract JSON from ```json code blocks."""
        pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                data = json.loads(match.strip())
                return JSONExtractionResult(
                    success=True,
                    data=data,
                    raw_json=match.strip(),
                    extraction_method="json_code_block"
                )
            except json.JSONDecodeError:
                continue

        return JSONExtractionResult(success=False)

    def _extract_from_any_code_block(self, response: str) -> JSONExtractionResult:
        """Extract JSON from any ``` code blocks."""
        pattern = r'```(?:\w+)?\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            cleaned = match.strip()
            if cleaned.startswith('{') and cleaned.endswith('}'):
                try:
                    data = json.loads(cleaned)
                    return JSONExtractionResult(
                        success=True,
                        data=data,
                        raw_json=cleaned,
                        extraction_method="any_code_block"
                    )
                except json.JSONDecodeError:
                    continue

        return JSONExtractionResult(success=False)

    def _extract_from_backticks(self, response: str) -> JSONExtractionResult:
        """Extract JSON from single backticks."""
        pattern = r'`([^`]*{[^`]*}[^`]*)`'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match.strip())
                return JSONExtractionResult(
                    success=True,
                    data=data,
                    raw_json=match.strip(),
                    extraction_method="backticks"
                )
            except json.JSONDecodeError:
                continue

        return JSONExtractionResult(success=False)

    def _extract_largest_json_object(self, response: str) -> JSONExtractionResult:
        """Extract the largest valid JSON object from response."""
        # Find all potential JSON start positions
        candidates = []

        for i, char in enumerate(response):
            if char == '{':
                # Try to find matching closing brace
                brace_count = 0
                for j in range(i, len(response)):
                    if response[j] == '{':
                        brace_count += 1
                    elif response[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON candidate
                            candidate = response[i:j+1]
                            candidates.append((len(candidate), candidate))
                            break

        # Sort by length (largest first) and try to parse
        candidates.sort(reverse=True, key=lambda x: x[0])

        for length, candidate in candidates:
            try:
                data = json.loads(candidate)
                return JSONExtractionResult(
                    success=True,
                    data=data,
                    raw_json=candidate,
                    extraction_method="largest_json_object"
                )
            except json.JSONDecodeError:
                continue

        return JSONExtractionResult(success=False)

    def _extract_from_brackets(self, response: str) -> JSONExtractionResult:
        """Simple extraction using first complete bracket pair."""
        start_idx = response.find('{')
        if start_idx == -1:
            return JSONExtractionResult(success=False)

        brace_count = 0
        end_idx = start_idx

        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if brace_count == 0:
            candidate = response[start_idx:end_idx + 1]
            try:
                data = json.loads(candidate)
                return JSONExtractionResult(
                    success=True,
                    data=data,
                    raw_json=candidate,
                    extraction_method="brackets"
                )
            except json.JSONDecodeError:
                pass

        return JSONExtractionResult(success=False)

    def _extract_and_fix_common_issues(self, response: str) -> JSONExtractionResult:
        """Try to fix common JSON issues and extract."""
        # Find JSON-like content
        start_idx = response.find('{')
        if start_idx == -1:
            return JSONExtractionResult(success=False)

        # Extract potential JSON
        brace_count = 0
        end_idx = start_idx

        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if brace_count != 0:
            return JSONExtractionResult(success=False)

        candidate = response[start_idx:end_idx + 1]

        # Try common fixes
        fixes = [
            candidate,  # Original
            self._fix_trailing_commas(candidate),
            self._fix_single_quotes(candidate),
            self._fix_unquoted_keys(candidate),
            self._fix_control_characters(candidate)
        ]

        for fixed_json in fixes:
            if fixed_json:
                try:
                    data = json.loads(fixed_json)
                    return JSONExtractionResult(
                        success=True,
                        data=data,
                        raw_json=fixed_json,
                        extraction_method="fixed_common_issues"
                    )
                except json.JSONDecodeError:
                    continue

        return JSONExtractionResult(success=False)

    def _fix_trailing_commas(self, json_str: str) -> str:
        """Remove trailing commas that break JSON parsing."""
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str

    def _fix_single_quotes(self, json_str: str) -> str:
        """Convert single quotes to double quotes."""
        # Simple replacement - may not handle all edge cases
        return json_str.replace("'", '"')

    def _fix_unquoted_keys(self, json_str: str) -> str:
        """Add quotes around unquoted keys."""
        # Add quotes around unquoted object keys
        pattern = r'(\w+)(\s*:\s*)'
        return re.sub(pattern, r'"\1"\2', json_str)

    def _fix_control_characters(self, json_str: str) -> str:
        """Remove or escape control characters."""
        # Remove common problematic characters
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        return json_str

    def extract_multiple_json(self, response: str) -> List[JSONExtractionResult]:
        """Extract multiple JSON objects from a response."""
        results = []

        # Find all potential JSON objects
        start_positions = [i for i, char in enumerate(response) if char == '{']

        for start_idx in start_positions:
            brace_count = 0
            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = response[start_idx:i+1]
                        try:
                            data = json.loads(candidate)
                            results.append(JSONExtractionResult(
                                success=True,
                                data=data,
                                raw_json=candidate,
                                extraction_method="multiple_json"
                            ))
                        except json.JSONDecodeError:
                            pass
                        break

        return results


# Create singleton instance for easy imports
robust_json_parser = RobustJSONParser(debug=False)


def extract_json_robust(response: str, model_name: str = None) -> JSONExtractionResult:
    """
    Convenience function for robust JSON extraction.

    Args:
        response: Raw LLM response text
        model_name: Name of the model (for thinking model detection)

    Returns:
        JSONExtractionResult with parsed data or error info
    """
    return robust_json_parser.extract_json(response, model_name)


def extract_json_or_none(response: str, model_name: str = None) -> Optional[Dict[str, Any]]:
    """
    Extract JSON and return data or None.

    Args:
        response: Raw LLM response text
        model_name: Name of the model (for thinking model detection)

    Returns:
        Parsed JSON data or None if extraction failed
    """
    result = robust_json_parser.extract_json(response, model_name)
    return result.data if result.success else None


def extract_json_with_fallback(response: str, fallback: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
    """
    Extract JSON with fallback value.

    Args:
        response: Raw LLM response text
        fallback: Default value if extraction fails
        model_name: Name of the model (for thinking model detection)

    Returns:
        Parsed JSON data or fallback value
    """
    result = robust_json_parser.extract_json(response, model_name)
    return result.data if result.success else fallback
