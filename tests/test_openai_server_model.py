#!/usr/bin/env python3
"""
Test script for OpenAIServerModel integration with OpenAI and Gemini.
"""

import os
import asyncio
from dotenv import load_dotenv
from smolagents import OpenAIServerModel

async def test_openai_model():
    """Test OpenAI model using OpenAIServerModel"""

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return False

    print("🚀 Testing OpenAI with OpenAIServerModel")

    try:
        model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key=openai_key,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON. You must respond with valid JSON only."},
            {"role": "user", "content": """Generate a simple math question in JSON format:
            {
              "question": "What is 7 + 5?",
              "answer": 12,
              "difficulty": "easy"
            }"""}
        ]

        print("📤 Sending request to OpenAI...")
        response = model(messages)
        print(f"📥 OpenAI response: {response}")

        # Extract content if response is a ChatMessage object
        content = getattr(response, "content", response)

        # Validate JSON
        import json
        parsed = json.loads(content)
        print("✅ OpenAI test successful!")
        return True

    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

async def test_gemini_model():
    """Test Gemini model using OpenAIServerModel"""

    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("❌ GOOGLE_API_KEY not found")
        return False

    print("🚀 Testing Gemini with OpenAIServerModel")

    try:
        model = OpenAIServerModel(
            model_id="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON. You must respond with valid JSON only."},
            {"role": "user", "content": """Generate a simple math question in JSON format:
            {
              "question": "What is 9 - 4?",
              "answer": 5,
              "difficulty": "easy"
            }"""}
        ]

        print("📤 Sending request to Gemini...")
        response = model(messages)
        print(f"📥 Gemini response: {response}")

        content = getattr(response, "content", response)
        import json
        parsed = json.loads(content)
        print("✅ Gemini test successful!")
        return True

    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

async def main():
    """Run all tests"""
    load_dotenv()

    print("=== Testing OpenAIServerModel Integration ===\n")

    openai_success = await test_openai_model()
    print()
    gemini_success = await test_gemini_model()

    print("\n=== Test Summary ===")
    print(f"OpenAI: {'✅ PASS' if openai_success else '❌ FAIL'}")
    print(f"Gemini: {'✅ PASS' if gemini_success else '❌ FAIL'}")

    if openai_success or gemini_success:
        print("\n🎉 At least one model is working! Ready to proceed.")
    else:
        print("\n💥 All models failed. Please check API keys and connections.")

if __name__ == "__main__":
    asyncio.run(main())
