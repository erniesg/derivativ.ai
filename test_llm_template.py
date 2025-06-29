#!/usr/bin/env python3
"""Test the exact template and LLM response."""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from src.models.llm_models import LLMRequest  # noqa: E402
from src.services.llm_factory import LLMFactory  # noqa: E402


async def test_template():
    """Test the exact template that's failing."""

    # Minimal test prompt
    prompt = """You are a math teacher. Generate content for a worksheet.

You MUST respond with a valid JSON object with this structure:
{
  "enhanced_title": "string",
  "introduction": "optional string",
  "blocks": [
    {
      "block_type": "string",
      "content": {},
      "estimated_minutes": 5,
      "reasoning": "string"
    }
  ],
  "total_estimated_minutes": 30,
  "actual_detail_level": 5,
  "generation_reasoning": "string",
  "coverage_notes": "optional string",
  "personalization_applied": []
}

Generate content for these blocks:
1. practice_questions - Create 3 math questions
2. learning_objectives - List 2 objectives

Respond ONLY with the JSON object, no markdown formatting."""

    llm_factory = LLMFactory()
    llm_service = llm_factory.get_service("openai")

    request = LLMRequest(
        model="gpt-4o-mini",
        prompt=prompt,
        temperature=0.3,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    print("📤 Sending test prompt...")
    response = await llm_service.generate_non_stream(request)

    print(f"\n📥 Raw response ({len(response.content)} chars):")
    print(response.content[:500] + "...")

    # Check if it starts with markdown
    if response.content.startswith("```"):
        print("\n⚠️ Response starts with markdown code block!")

    # Try parsing
    try:
        # Handle markdown wrapping
        content = response.content
        if content.startswith("```json"):
            content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        elif content.startswith("```"):
            content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)
        print("\n✅ JSON parsed successfully!")
        print(f"Keys: {list(parsed.keys())}")

        # Validate required fields
        required = [
            "blocks",
            "total_estimated_minutes",
            "actual_detail_level",
            "generation_reasoning",
        ]
        missing = [f for f in required if f not in parsed]
        if missing:
            print(f"❌ Missing required fields: {missing}")
        else:
            print("✅ All required fields present!")

    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parse error: {e}")


if __name__ == "__main__":
    asyncio.run(test_template())
