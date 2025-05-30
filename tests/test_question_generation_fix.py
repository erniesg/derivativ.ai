#!/usr/bin/env python3
"""
Test QuestionGeneratorAgent in isolation to debug generation issues.
"""

import asyncio
import os
import sys

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents import QuestionGeneratorAgent
from src.models import GenerationConfig, CalculatorPolicy, CommandWord, LLMModel
from smolagents import OpenAIServerModel

from dotenv import load_dotenv
load_dotenv()


async def test_question_generation_debug():
    """Test question generation with detailed debugging"""

    print("🧪 Testing QuestionGeneratorAgent in isolation...")

    # Create model
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create agent with NO database client (to simulate orchestrator conditions)
    agent = QuestionGeneratorAgent(model=model, db_client=None, debug=True)

    # Create test config
    config = GenerationConfig(
        target_grade=5,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED,
        desired_marks=3,
        subject_content_references=["C1.6", "C1.11"],
        command_word_override=CommandWord.CALCULATE,  # Explicit command word
        llm_model_generation=LLMModel.GPT_4O_MINI,
        llm_model_marking_scheme=LLMModel.GPT_4O_MINI,
        llm_model_review=LLMModel.GPT_4O_MINI,
        temperature=0.7,
        max_tokens=2000
    )

    print(f"\n📋 Test Configuration:")
    print(f"   Target Grade: {config.target_grade}")
    print(f"   Subject Refs: {config.subject_content_references}")
    print(f"   Command Word: {config.command_word_override}")
    print(f"   Calculator Policy: {config.calculator_policy}")
    print(f"   Desired Marks: {config.desired_marks}")

    try:
        # Generate question
        print(f"\n🔄 Generating question...")
        question = await agent.generate_question(config)

        if question:
            print(f"✅ Question generated successfully!")
            print(f"   Question ID: {question.question_id_local}")
            print(f"   Content: {question.raw_text_content[:100]}...")
            print(f"   Marks: {question.marks}")
            print(f"   Command Word: {question.command_word}")
            print(f"   Validation Errors: {question.validation_errors if hasattr(question, 'validation_errors') else 'None'}")
            return True
        else:
            print(f"❌ Question generation returned None")
            return False

    except Exception as e:
        print(f"❌ Error during question generation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the debug test"""

    print("🚀 Starting QuestionGeneratorAgent Debug Test\n")

    success = await test_question_generation_debug()

    if success:
        print("\n🎉 Question generation test passed!")
    else:
        print("\n💥 Question generation test failed!")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
