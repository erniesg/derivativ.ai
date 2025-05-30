#!/usr/bin/env python3
"""
Test script for robust JSON parsing from DeepSeek and Qwen3 outputs.
"""

import json
import re
from typing import Optional, Dict, Any

def strip_thinking_tokens(text: str) -> str:
    """Remove <think>...</think> blocks from the response"""
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

def extract_json_from_code_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from ```json ... ``` code blocks"""
    # Look for ```json ... ``` blocks
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Also try generic ``` blocks
    generic_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(generic_pattern, text, re.DOTALL)

    for match in matches:
        try:
            # Skip if it looks like code (contains keywords)
            if any(keyword in match.lower() for keyword in ['def ', 'import ', 'class ', 'function']):
                continue
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    return None

def extract_raw_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract raw JSON object from text"""
    # Try to find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1

    if start_idx == -1 or end_idx == 0:
        return None

    json_str = text[start_idx:end_idx]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing raw JSON: {e}")
        return None

def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM response to extract JSON question data"""
    try:
        # Step 1: Remove <think>...</think> reasoning tokens
        content = strip_thinking_tokens(response)

        # Step 2: Try to extract JSON from code blocks first
        json_from_code_block = extract_json_from_code_block(content)
        if json_from_code_block:
            return json_from_code_block

        # Step 3: Try to find raw JSON in the response
        json_from_raw = extract_raw_json(content)
        if json_from_raw:
            return json_from_raw

        print("No valid JSON found in LLM response")
        return None

    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return None

# Test data from actual model outputs
DEEPSEEK_OUTPUT = '''<think>
We are to generate a simple math question in JSON format. The example provided is an addition problem.
 We can create a similar one, perhaps with different numbers or a different operation, but keeping it simple and easy.

 Let's choose: "What is 7 multiplied by 8?"

 The answer is 56.

 We'll set the difficulty to "easy".

 The JSON structure will be:
 {
   "question": "What is 7 multiplied by 8?",
   "answer": 56,
   "difficulty": "easy"
 }
</think>


Here's a simple math question in the requested JSON format:

```json
{
  "question": "What is 9 multiplied by 5?",
  "answer": 45,
  "difficulty": "easy"
}
```

Explanation of the elements:
- `question`: A clear multiplication problem (9 × 5)
- `answer`: Correct numerical result (45)
- `difficulty`: Classified as "easy" (basic multiplication fact)

This follows the same structure as your example, using multiplication instead of addition while maintaining the "easy" difficulty level appropriate for beginners.'''

QWEN3_OUTPUT = '''<think>
Okay, the user wants me to generate a simple math question in JSON format. Let me start by understanding the example they provided. The example has a question, answer, and difficulty level. The question is "What is 11 + 6?", the answer is 17, and the difficulty is easy.

First, I need to make sure that the JSON structure matches the example. The keys should be "question", "answer", and "difficulty". The question should be a simple arithmetic problem, addition in the example. The answer must be correct, so I need to double-check my calculations. The difficulty should remain easy, so the numbers shouldn't be too large or complex.

[... more thinking ...]

Yes, that works. Let me confirm that 9+5 is 14. Correct. The difficulty is easy. The question is a simple addition problem. The JSON structure matches the example. So that's a valid response.
</think>

{
  "question": "What is 9 + 5?",
  "answer": 14,
  "difficulty": "easy"
}'''

def test_parsing():
    print("=== Testing DeepSeek Output (with code block) ===")
    deepseek_result = parse_llm_response(DEEPSEEK_OUTPUT)
    print(f"Extracted JSON: {deepseek_result}")

    print("\n=== Testing Qwen3 Output (raw JSON) ===")
    qwen3_result = parse_llm_response(QWEN3_OUTPUT)
    print(f"Extracted JSON: {qwen3_result}")

    print("\n=== Validation ===")
    if deepseek_result and qwen3_result:
        print("✅ Both parsing methods work!")
        print(f"DeepSeek question: {deepseek_result.get('question')}")
        print(f"Qwen3 question: {qwen3_result.get('question')}")
    else:
        print("❌ Parsing failed for one or both outputs")

if __name__ == "__main__":
    test_parsing()
