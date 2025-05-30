#!/usr/bin/env python3
"""
Test the comprehensive validation system with generated questions.
"""

import asyncio
import os
import sys

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents import QuestionGeneratorAgent
from src.models import GenerationConfig, CalculatorPolicy, CommandWord, LLMModel
from src.validation import validate_question, CambridgeQuestionValidator
from smolagents import OpenAIServerModel

from dotenv import load_dotenv
load_dotenv()


async def test_validation_system():
    """Test the validation system with generated questions"""

    print("🔍 Testing Comprehensive Validation System")
    print("=" * 60)

    # Create model and agent
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = QuestionGeneratorAgent(model=model, db_client=None, debug=False)

    # Test different scenarios
    test_cases = [
        {
            "name": "Valid Question Test",
            "config": GenerationConfig(
                target_grade=5,
                calculator_policy=CalculatorPolicy.NOT_ALLOWED,
                desired_marks=2,
                subject_content_references=["A1.1", "A1.2"],  # Valid refs from syllabus
                command_word_override=CommandWord.CALCULATE,
                llm_model_generation=LLMModel.GPT_4O_MINI,
                llm_model_marking_scheme=LLMModel.GPT_4O_MINI,
                llm_model_review=LLMModel.GPT_4O_MINI,
                temperature=0.7,
                max_tokens=2000
            )
        },
        {
            "name": "Invalid Subject Refs Test",
            "config": GenerationConfig(
                target_grade=5,
                calculator_policy=CalculatorPolicy.NOT_ALLOWED,
                desired_marks=2,
                subject_content_references=["INVALID_REF", "ANOTHER_BAD_REF"],  # Invalid refs
                command_word_override=CommandWord.WORK_OUT,
                llm_model_generation=LLMModel.GPT_4O_MINI,
                llm_model_marking_scheme=LLMModel.GPT_4O_MINI,
                llm_model_review=LLMModel.GPT_4O_MINI,
                temperature=0.7,
                max_tokens=2000
            )
        }
    ]

    validator = CambridgeQuestionValidator()

    print(f"📋 Loaded syllabus data:")
    print(f"   • Valid subject refs: {len(validator.valid_subject_refs)}")
    print(f"   • Valid topics: {len(validator.valid_topic_paths)}")
    print(f"   • Command words: {len(validator.command_words)}")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 40)

        try:
            # Generate question
            question = await agent.generate_question(test_case['config'])

            if question:
                print(f"✅ Question generated: {question.question_id_local}")
                print(f"   Content: {question.raw_text_content[:100]}...")

                # Validate question
                validation_result = validate_question(question, verbose=True)

                # Show insert decision
                print(f"\n🎯 Insert Decision: {'✅ CAN INSERT' if validation_result.can_insert else '❌ CANNOT INSERT'}")

            else:
                print("❌ Question generation failed")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n🎉 Validation system testing complete!")


async def test_subject_ref_validation():
    """Test subject reference validation specifically"""
    print(f"\n📚 Testing Subject Reference Validation")
    print("-" * 40)

    validator = CambridgeQuestionValidator()

    # Show some valid subject refs
    sample_refs = list(validator.valid_subject_refs)[:20]
    print(f"✅ Sample valid subject refs: {', '.join(sample_refs)}")

    # Show valid topic paths
    sample_topics = list(validator.valid_topic_paths.keys())[:10]
    print(f"✅ Sample valid topics: {', '.join(sample_topics)}")

    # Show command words
    print(f"✅ Valid command words: {', '.join(sorted(validator.command_words))}")


async def main():
    """Run all validation tests"""

    print("🚀 Starting Validation System Tests\n")

    # Test subject reference loading
    await test_subject_ref_validation()

    # Test full validation pipeline
    await test_validation_system()

    print("\n🎯 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
