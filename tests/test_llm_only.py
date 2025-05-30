#!/usr/bin/env python3
"""
Test script for LLM integration only, without database dependencies.
"""

import os
import asyncio
from dotenv import load_dotenv
from smolagents import LiteLLMModel

async def test_llm_integration():
    """Test LLM integration with JSON mode"""

    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False

    print("🚀 Testing LLM Integration")
    print(f"OpenAI Key: {openai_key[:10]}..." if openai_key else "None")

    try:
        # Create model with JSON mode
        model = LiteLLMModel(
            model_id="gpt-4o-mini",  # Using mini for faster testing
            api_key=openai_key,
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        print("✅ Model created successfully")

        # Test simple JSON generation
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON. You must respond with valid JSON only."},
            {"role": "user", "content": """Generate a simple math question in JSON format:
            {
              "question": "What is 5 + 3?",
              "answer": 8,
              "difficulty": "easy"
            }"""}
        ]

        print("📤 Sending test request...")
        response = model(messages)

        print(f"📥 Response received (length: {len(str(response))})")
        print(f"Response: {response}")

        # Extract content if response is a ChatMessage object
        content = getattr(response, "content", response)

        # Try to parse as JSON
        import json
        try:
            parsed = json.loads(content)
            print("✅ Valid JSON response!")
            print(f"Parsed: {parsed}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_llm_integration())
