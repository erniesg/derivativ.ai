#!/usr/bin/env python3
"""
Test validation with VALID subject references.
"""

import asyncio
import os
import sys

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents import QuestionGeneratorAgent
from src.models import GenerationConfig, CalculatorPolicy, CommandWord, LLMModel
from src.validation import validate_question
from smolagents import OpenAIServerModel

from dotenv import load_dotenv
load_dotenv()


async def test_valid_question():
    """Test with valid Cambridge subject references"""

    print("✅ Testing with VALID Subject References")

    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = QuestionGeneratorAgent(model=model, db_client=None, debug=False)

    # Use VALID subject references from the validation output
    config = GenerationConfig(
        target_grade=5,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED,
        desired_marks=2,
        subject_content_references=["C1.1", "C1.9"],  # VALID refs
        command_word_override=CommandWord.CALCULATE,
        llm_model_generation=LLMModel.GPT_4O_MINI,
        llm_model_marking_scheme=LLMModel.GPT_4O_MINI,
        llm_model_review=LLMModel.GPT_4O_MINI,
        temperature=0.7,
        max_tokens=2000
    )

    question = await agent.generate_question(config)

    if question:
        print(f"📝 Generated question: {question.question_id_local}")
        print(f"   Content: {question.raw_text_content}")

        # Validate
        result = validate_question(question, verbose=True)

        if result.can_insert:
            print(f"\n🎉 SUCCESS: Question can be inserted into database!")
        else:
            print(f"\n❌ FAILED: Question still has validation issues")
    else:
        print("❌ Question generation failed")


if __name__ == "__main__":
    asyncio.run(test_valid_question())
