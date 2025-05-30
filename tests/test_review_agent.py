#!/usr/bin/env python3
"""
Test ReviewAgent functionality.
Tests the specialized question review and quality assurance agent.
"""

import asyncio
import os
import sys
import uuid

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents import ReviewAgent, ReviewOutcome
from src.models import (
    CandidateQuestion, GenerationConfig, CalculatorPolicy,
    SolutionAndMarkingScheme, MarkAllocationCriterion, AnswerSummary,
    SolverAlgorithm, SolverStep, QuestionTaxonomy
)
from smolagents import OpenAIServerModel

from dotenv import load_dotenv
load_dotenv()


def create_test_question() -> CandidateQuestion:
    """Create a test question for review"""

    # Create answer summary
    answer = AnswerSummary(
        answer_text="Area: 96 cm², Perimeter: 40 cm",
        value_numeric=None,
        unit=None
    )

    # Create marking criteria
    criteria = [
        MarkAllocationCriterion(
            criterion_id="crit_1",
            criterion_text="Correct calculation of area using length × width",
            mark_code_display="M1",
            marks_value=1.0,
            mark_type_primary="M",
            qualifiers_and_notes="oe"
        ),
        MarkAllocationCriterion(
            criterion_id="crit_2",
            criterion_text="Correct calculation of perimeter using 2(length + width)",
            mark_code_display="M1",
            marks_value=1.0,
            mark_type_primary="M",
            qualifiers_and_notes="oe"
        ),
        MarkAllocationCriterion(
            criterion_id="crit_3",
            criterion_text="Both correct answers with appropriate units",
            mark_code_display="A1",
            marks_value=1.0,
            mark_type_primary="A",
            qualifiers_and_notes="cao"
        )
    ]

    # Create marking scheme
    marking_scheme = SolutionAndMarkingScheme(
        final_answers_summary=[answer],
        mark_allocation_criteria=criteria,
        total_marks_for_part=3
    )

    # Create solver steps
    steps = [
        SolverStep(
            step_number=1,
            description_text="Calculate area using formula",
            mathematical_expression_latex="Area = length \\times width = 12 \\times 8 = 96",
            justification_or_reasoning="Apply rectangle area formula"
        ),
        SolverStep(
            step_number=2,
            description_text="Calculate perimeter using formula",
            mathematical_expression_latex="Perimeter = 2(length + width) = 2(12 + 8) = 40",
            justification_or_reasoning="Apply rectangle perimeter formula"
        )
    ]

    solver_algorithm = SolverAlgorithm(steps=steps)

    # Create taxonomy
    taxonomy = QuestionTaxonomy(
        topic_path=["Geometry", "Mensuration", "Rectangles"],
        subject_content_references=["G1.1", "G1.2"],
        skill_tags=["AREA_CALCULATION", "PERIMETER_CALCULATION", "RECTANGLE_PROPERTIES"],
        cognitive_level="ProceduralFluency",
        difficulty_estimate_0_to_1=0.4
    )

    # Create the complete question
    question = CandidateQuestion(
        question_id_local="Test_Q001",
        question_id_global="test_review_q001",
        question_number_display="Test Question 1",
        marks=3,
        command_word="Calculate",
        raw_text_content="A rectangle has length 12 cm and width 8 cm. Calculate the area and perimeter of the rectangle.",
        formatted_text_latex=None,
        taxonomy=taxonomy,
        solution_and_marking_scheme=marking_scheme,
        solver_algorithm=solver_algorithm,
        assets=[],
        generation_id=str(uuid.uuid4()),
        target_grade_input=5,
        llm_model_used_generation="gpt-4o-mini",
        llm_model_used_marking_scheme="gpt-4o-mini",
        prompt_template_version_generation="v1.1",
        prompt_template_version_marking_scheme="v1.0"
    )

    return question


async def test_review_agent():
    """Test ReviewAgent question review functionality"""

    print("🧪 Testing ReviewAgent...")

    # Create test model
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create ReviewAgent
    review_agent = ReviewAgent(model=model, debug=True)

    # Test configuration
    config = GenerationConfig(
        target_grade=5,
        calculator_policy=CalculatorPolicy.NOT_ALLOWED,
        desired_marks=3,
        subject_content_references=["G1.1", "G1.2"],
        temperature=0.7,
        max_tokens=2000
    )

    # Create test question
    question = create_test_question()

    try:
        # Review the question
        print("\n🔄 Reviewing question...")
        feedback = await review_agent.review_question(
            question=question,
            config=config
        )

        print(f"✅ Review completed!")
        print(f"   Outcome: {feedback.outcome.value}")
        print(f"   Overall Score: {feedback.overall_score:.2f}")
        print(f"   Quality Grade: {review_agent._calculate_quality_grade(feedback.overall_score)}")

        # Display detailed feedback
        print(f"\n📝 Feedback Summary:")
        print(f"   {feedback.feedback_summary}")

        print(f"\n📊 Scores:")
        print(f"   Syllabus Compliance: {feedback.syllabus_compliance:.2f}")
        print(f"   Difficulty Alignment: {feedback.difficulty_alignment:.2f}")
        print(f"   Marking Quality: {feedback.marking_quality:.2f}")

        print(f"\n🔍 Specific Feedback:")
        for category, detail in feedback.specific_feedback.items():
            print(f"   {category.replace('_', ' ').title()}: {detail}")

        print(f"\n💡 Suggested Improvements:")
        for i, improvement in enumerate(feedback.suggested_improvements, 1):
            print(f"   {i}. {improvement}")

        return True

    except Exception as e:
        print(f"❌ Error testing ReviewAgent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_review():
    """Test ReviewAgent batch review functionality"""

    print("\n🧪 Testing ReviewAgent batch review...")

    # Create test model
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create ReviewAgent
    review_agent = ReviewAgent(model=model, debug=True)

    # Create test questions (2 questions for batch)
    questions = [
        create_test_question(),
        create_test_question()  # Same question for simplicity
    ]

    # Update second question to be different
    questions[1].question_id_local = "Test_Q002"
    questions[1].raw_text_content = "A rectangle has length 15 cm and width 10 cm. Calculate the area and perimeter."
    questions[1].marks = 2  # Different mark allocation

    # Test configurations
    configs = [
        GenerationConfig(
            target_grade=5,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED,
            desired_marks=3,
            subject_content_references=["G1.1", "G1.2"],
            temperature=0.7,
            max_tokens=2000
        ),
        GenerationConfig(
            target_grade=4,
            calculator_policy=CalculatorPolicy.ALLOWED,
            desired_marks=2,
            subject_content_references=["G1.1"],
            temperature=0.7,
            max_tokens=2000
        )
    ]

    try:
        # Batch review
        print(f"🔄 Batch reviewing {len(questions)} questions...")
        feedbacks = await review_agent.batch_review(questions, configs)

        print(f"✅ Batch review completed!")
        print(f"   Reviewed {len(feedbacks)} questions")

        # Generate summary
        summary = review_agent.get_review_summary(feedbacks)

        print(f"\n📈 Review Summary:")
        print(f"   Total Questions: {summary['total_questions']}")
        print(f"   Quality Grade: {summary['quality_grade']}")

        print(f"\n📊 Outcome Distribution:")
        for outcome, count in summary['outcome_distribution'].items():
            print(f"   {outcome.replace('_', ' ').title()}: {count}")

        print(f"\n🎯 Average Scores:")
        for metric, score in summary['average_scores'].items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")

        print(f"\n🔧 Top Improvement Suggestions:")
        for i, suggestion in enumerate(summary['top_improvement_suggestions'][:3], 1):
            print(f"   {i}. {suggestion}")

        return True

    except Exception as e:
        print(f"❌ Error testing batch review: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all ReviewAgent tests"""

    print("🚀 Starting ReviewAgent Tests\n")

    # Test individual review
    test1_result = await test_review_agent()

    # Test batch review
    test2_result = await test_batch_review()

    print(f"\n📊 Test Results:")
    print(f"   Individual review test: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   Batch review test: {'✅ PASS' if test2_result else '❌ FAIL'}")

    if test1_result and test2_result:
        print("\n🎉 All ReviewAgent tests passed!")
        return True
    else:
        print("\n💥 Some ReviewAgent tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
