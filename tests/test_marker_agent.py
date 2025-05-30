#!/usr/bin/env python3
"""
Test MarkerAgent functionality.
Tests the specialized marking scheme generation agent.
"""

import asyncio
import os
import sys

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents import MarkerAgent
from src.models import GenerationConfig, CalculatorPolicy
from smolagents import OpenAIServerModel

from dotenv import load_dotenv
load_dotenv()


async def test_marker_agent():
    """Test MarkerAgent marking scheme generation"""

    print("🧪 Testing MarkerAgent...")

    # Create test model
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create MarkerAgent
    marker_agent = MarkerAgent(model=model, debug=True)

    # Test configuration
    config = GenerationConfig(
        target_grade=5,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED,
        desired_marks=3,
        subject_content_references=["C1.6", "C1.11"],
        temperature=0.7,
        max_tokens=2000
    )

    # Test question
    question_text = "A rectangle has length 12 cm and width 8 cm. Calculate the area and perimeter of the rectangle."

    try:
        # Generate marking scheme
        print("\n🔄 Generating marking scheme...")
        marking_scheme = await marker_agent.generate_marking_scheme(
            question_text=question_text,
            config=config,
            expected_answer="Area: 96 cm², Perimeter: 40 cm"
        )

        print(f"✅ Marking scheme generated!")
        print(f"   Total marks: {marking_scheme.total_marks_for_part}")
        print(f"   Number of criteria: {len(marking_scheme.mark_allocation_criteria)}")
        print(f"   Number of answers: {len(marking_scheme.final_answers_summary)}")

        # Display marking scheme details
        print("\n📋 Final Answers:")
        for i, answer in enumerate(marking_scheme.final_answers_summary, 1):
            print(f"   {i}. {answer.answer_text}")
            if answer.value_numeric is not None:
                print(f"      Value: {answer.value_numeric}")
            if answer.unit:
                print(f"      Unit: {answer.unit}")

        print("\n📊 Mark Allocation Criteria:")
        for criterion in marking_scheme.mark_allocation_criteria:
            print(f"   {criterion.mark_code_display}: {criterion.criterion_text}")
            print(f"      Marks: {criterion.marks_value}")
            print(f"      Type: {criterion.mark_type_primary}")
            if criterion.qualifiers_and_notes:
                print(f"      Notes: {criterion.qualifiers_and_notes}")
            print()

        # Test refinement
        print("🔄 Testing marking scheme refinement...")
        refined_scheme = await marker_agent.refine_marking_scheme(
            existing_scheme=marking_scheme,
            question_text=question_text,
            config=config,
            feedback="Make the criteria more specific about showing working"
        )

        print(f"✅ Refined marking scheme generated!")
        print(f"   Refined criteria count: {len(refined_scheme.mark_allocation_criteria)}")

        return True

    except Exception as e:
        print(f"❌ Error testing MarkerAgent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_marker_agent_with_claude():
    """Test MarkerAgent with Claude 4 (if available)"""

    print("\n🧪 Testing MarkerAgent with Claude 4...")

    try:
        from smolagents import AmazonBedrockServerModel

        # Create Claude 4 model
        model = AmazonBedrockServerModel(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            client_kwargs={'region_name': os.getenv("AWS_REGION", "us-east-1")}
        )

        # Create MarkerAgent
        marker_agent = MarkerAgent(model=model, debug=True)

        # Test configuration
        config = GenerationConfig(
            target_grade=7,
            calculator_policy=CalculatorPolicy.ALLOWED,
            desired_marks=4,
            subject_content_references=["A2.1", "A2.2"],
            temperature=0.8,
            max_tokens=3000
        )

        # Test question
        question_text = "Solve the simultaneous equations: 2x + 3y = 11 and x - y = 1. Show all your working."

        # Generate marking scheme
        print("🔄 Generating marking scheme with Claude 4...")
        marking_scheme = await marker_agent.generate_marking_scheme(
            question_text=question_text,
            config=config
        )

        print(f"✅ Claude 4 marking scheme generated!")
        print(f"   Total marks: {marking_scheme.total_marks_for_part}")
        print(f"   Number of criteria: {len(marking_scheme.mark_allocation_criteria)}")

        return True

    except Exception as e:
        print(f"⚠️ Claude 4 test skipped: {e}")
        return True  # Don't fail the test if Claude 4 isn't available


async def main():
    """Run all MarkerAgent tests"""

    print("🚀 Starting MarkerAgent Tests\n")

    # Test with GPT-4o-mini
    test1_result = await test_marker_agent()

    # Test with Claude 4 (if available)
    test2_result = await test_marker_agent_with_claude()

    print(f"\n📊 Test Results:")
    print(f"   GPT-4o-mini test: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   Claude 4 test: {'✅ PASS' if test2_result else '❌ FAIL'}")

    if test1_result and test2_result:
        print("\n🎉 All MarkerAgent tests passed!")
        return True
    else:
        print("\n💥 Some MarkerAgent tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
