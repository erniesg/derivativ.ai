"""
Enhanced JSON Parser for LLM Responses.

Robust JSON extraction with support for thinking models, code blocks,
mixed text, and various malformed JSON patterns. Async-enabled with caching.
"""

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JSONExtractionResult(BaseModel):
    """Result of JSON extraction attempt with detailed metadata"""

    success: bool = Field(..., description="Whether extraction succeeded")
    data: Optional[dict[str, Any]] = Field(None, description="Extracted JSON data")
    raw_json: Optional[str] = Field(None, description="Raw JSON string found")
    extraction_method: Optional[str] = Field(None, description="Method used for extraction")
    error: Optional[str] = Field(None, description="Error message if failed")
    thinking_tokens_removed: bool = Field(
        False, description="Whether thinking tokens were found and removed"
    )
    original_length: int = Field(0, description="Original response length")
    cleaned_length: int = Field(0, description="Length after cleaning")
    confidence_score: float = Field(0.0, description="Confidence in extraction (0-1)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional extraction metadata"
    )


class JSONParser:
    """
    Enhanced JSON parser for LLM responses.

    Features:
    - Multiple extraction strategies
    - Thinking model support
    - Code block detection
    - Malformed JSON repair
    - Async operations with caching
    - Confidence scoring
    """

    def __init__(self, enable_cache: bool = True, cache_size: int = 100):
        """
        Initialize JSON parser.

        Args:
            enable_cache: Whether to cache extraction results
            cache_size: Maximum cache entries
        """
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache: dict[str, JSONExtractionResult] = {}
        self._extraction_stats = {
            "total_attempts": 0,
            "successful_extractions": 0,
            "cache_hits": 0,
            "method_success_counts": {},
        }

    async def extract_json(
        self,
        response: str,
        model_name: Optional[str] = None,
        expected_schema: Optional[dict[str, Any]] = None,
    ) -> JSONExtractionResult:
        """
        Extract and parse JSON from LLM response using multiple strategies.

        Args:
            response: Raw LLM response text
            model_name: Model name for thinking model detection
            expected_schema: Expected JSON schema for validation

        Returns:
            JSONExtractionResult with extraction details
        """
        self._extraction_stats["total_attempts"] += 1

        # Check cache first
        cache_key = self._create_cache_key(response, model_name)
        if self.enable_cache and cache_key in self._cache:
            self._extraction_stats["cache_hits"] += 1
            return self._cache[cache_key]

        # Clean response for thinking models
        cleaned_response, thinking_tokens_removed = self._clean_thinking_tokens(
            response, model_name
        )

        # Try multiple extraction strategies
        extraction_strategies = [
            self._extract_from_code_blocks,
            self._extract_from_raw_json,
            self._extract_first_json_object,
            self._extract_with_repair,
            self._extract_from_markdown_json,
        ]

        best_result = JSONExtractionResult(
            success=False,
            original_length=len(response),
            cleaned_length=len(cleaned_response),
            thinking_tokens_removed=thinking_tokens_removed,
        )

        for strategy in extraction_strategies:
            try:
                result = await strategy(cleaned_response)
                if result.success:
                    # Validate against expected schema if provided
                    if expected_schema and not self._validate_schema(result.data, expected_schema):
                        continue

                    # Update result with metadata
                    result.original_length = len(response)
                    result.cleaned_length = len(cleaned_response)
                    result.thinking_tokens_removed = thinking_tokens_removed
                    result.confidence_score = self._calculate_confidence(result, response)

                    # Cache successful result
                    if self.enable_cache:
                        self._add_to_cache(cache_key, result)

                    # Update stats
                    self._extraction_stats["successful_extractions"] += 1
                    method_name = strategy.__name__
                    self._extraction_stats["method_success_counts"][method_name] = (
                        self._extraction_stats["method_success_counts"].get(method_name, 0) + 1
                    )

                    return result

                # Keep track of best attempt even if not successful
                if result.confidence_score > best_result.confidence_score:
                    best_result = result

            except Exception as e:
                logger.debug(f"Extraction strategy {strategy.__name__} failed: {e}")
                continue

        # Return best attempt if all strategies failed
        best_result.error = "All extraction strategies failed"
        return best_result

    def _clean_thinking_tokens(self, text: str, model_name: Optional[str]) -> tuple[str, bool]:
        """Remove thinking tokens from LLM response"""
        if not model_name or not self._is_thinking_model(model_name):
            return text, False

        original_text = text

        # Remove various thinking token formats
        thinking_patterns = [
            r"<think>.*?</think>",
            r"<thinking>.*?</thinking>",
            r"<thought>.*?</thought>",
            r"<analysis>.*?</analysis>",
            r"<reasoning>.*?</reasoning>",
            r"<internal>.*?</internal>",
            r"<planning>.*?</planning>",
        ]

        for pattern in thinking_patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove common thinking prefixes before JSON
        first_brace = text.find("{")
        if first_brace > 0:
            prefix_text = text[:first_brace].strip().lower()
            thinking_indicators = [
                "let me",
                "i need to",
                "first",
                "okay",
                "sure",
                "here's",
                "to solve",
                "let's",
                "i'll",
                "thinking",
                "reasoning",
            ]
            if any(indicator in prefix_text for indicator in thinking_indicators):
                text = text[first_brace:]

        thinking_tokens_found = len(original_text) != len(text)
        return text.strip(), thinking_tokens_found

    def _is_thinking_model(self, model_name: str) -> bool:
        """Detect if a model generates thinking tokens"""
        thinking_patterns = ["qwen", "deepseek", "thinking", "reasoning", "o1-", "o3-", "o1", "o3"]
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in thinking_patterns)

    async def _extract_from_code_blocks(self, text: str) -> JSONExtractionResult:
        """Extract JSON from ```json or ``` code blocks"""
        patterns = [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    return JSONExtractionResult(
                        success=True,
                        data=data,
                        raw_json=match.strip(),
                        extraction_method="code_block",
                        confidence_score=0.9,
                    )
                except json.JSONDecodeError:
                    continue

        return JSONExtractionResult(success=False, extraction_method="code_block")

    async def _extract_from_raw_json(self, text: str) -> JSONExtractionResult:
        """Try to parse the entire text as JSON"""
        try:
            data = json.loads(text.strip())
            return JSONExtractionResult(
                success=True,
                data=data,
                raw_json=text.strip(),
                extraction_method="raw_json",
                confidence_score=1.0,
            )
        except json.JSONDecodeError:
            return JSONExtractionResult(success=False, extraction_method="raw_json")

    async def _extract_first_json_object(self, text: str) -> JSONExtractionResult:
        """Extract the first complete JSON object from text"""
        brace_count = 0
        start_idx = -1

        for i, char in enumerate(text):
            if char == "{":
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_str = text[start_idx : i + 1]
                    try:
                        data = json.loads(json_str)
                        return JSONExtractionResult(
                            success=True,
                            data=data,
                            raw_json=json_str,
                            extraction_method="first_object",
                            confidence_score=0.8,
                        )
                    except json.JSONDecodeError:
                        continue

        return JSONExtractionResult(success=False, extraction_method="first_object")

    async def _extract_with_repair(self, text: str) -> JSONExtractionResult:
        """Attempt to repair common JSON formatting issues"""
        # Common repairs
        repairs = [
            (r",\s*}", "}"),  # Remove trailing commas
            (r",\s*]", "]"),  # Remove trailing commas in arrays
            (r"}\s*{", "},{"),  # Fix missing commas between objects
            (r'"\s*:\s*"([^"]*)"', r'": "\1"'),  # Fix spacing in key-value pairs
        ]

        repaired_text = text
        for pattern, replacement in repairs:
            repaired_text = re.sub(pattern, replacement, repaired_text)

        try:
            data = json.loads(repaired_text)
            return JSONExtractionResult(
                success=True,
                data=data,
                raw_json=repaired_text,
                extraction_method="repaired",
                confidence_score=0.7,
            )
        except json.JSONDecodeError:
            return JSONExtractionResult(success=False, extraction_method="repaired")

    async def _extract_from_markdown_json(self, text: str) -> JSONExtractionResult:
        """Extract JSON from markdown-style formatting"""
        # Look for JSON in various markdown contexts
        patterns = [
            r"JSON:\s*(.*?)(?:\n\n|\Z)",  # JSON: label
            r"Response:\s*(.*?)(?:\n\n|\Z)",  # Response: label
            r"Output:\s*(.*?)(?:\n\n|\Z)",  # Output: label
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    return JSONExtractionResult(
                        success=True,
                        data=data,
                        raw_json=match.strip(),
                        extraction_method="markdown",
                        confidence_score=0.75,
                    )
                except json.JSONDecodeError:
                    continue

        return JSONExtractionResult(success=False, extraction_method="markdown")

    def _validate_schema(self, data: dict[str, Any], expected_schema: dict[str, Any]) -> bool:
        """Basic schema validation (can be enhanced with jsonschema)"""
        if not isinstance(data, dict):
            return False

        # Check required keys
        required_keys = expected_schema.get("required", [])
        for key in required_keys:
            if key not in data:
                return False

        return True

    def _calculate_confidence(self, result: JSONExtractionResult, original_text: str) -> float:
        """Calculate confidence score for extraction"""
        base_confidence = 0.5

        # Higher confidence for certain extraction methods
        method_confidence = {
            "raw_json": 1.0,
            "code_block": 0.9,
            "first_object": 0.8,
            "repaired": 0.7,
            "markdown": 0.75,
        }

        if result.extraction_method in method_confidence:
            base_confidence = method_confidence[result.extraction_method]

        # Adjust based on JSON complexity and completeness
        if result.data:
            # More complex JSON structures are more likely to be correct
            if isinstance(result.data, dict) and len(result.data) > 3:
                base_confidence += 0.1

            # Penalize very short extractions from long text
            if result.raw_json and len(result.raw_json) < len(original_text) * 0.1:
                base_confidence -= 0.2

        return min(1.0, max(0.0, base_confidence))

    def _create_cache_key(self, response: str, model_name: Optional[str]) -> str:
        """Create cache key for response"""
        import hashlib

        key_data = f"{response}{model_name or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _add_to_cache(self, key: str, result: JSONExtractionResult):
        """Add result to cache with size limit"""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result

    def get_extraction_stats(self) -> dict[str, Any]:
        """Get statistics about extraction performance"""
        success_rate = self._extraction_stats["successful_extractions"] / max(
            1, self._extraction_stats["total_attempts"]
        )

        return {
            **self._extraction_stats,
            "success_rate": success_rate,
            "cache_hit_rate": (
                self._extraction_stats["cache_hits"]
                / max(1, self._extraction_stats["total_attempts"])
            ),
        }

    def clear_cache(self):
        """Clear extraction cache"""
        self._cache.clear()
