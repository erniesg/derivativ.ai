#!/usr/bin/env python3
"""
Test MultiAgentOrchestrator functionality.
Tests the complete question generation pipeline with quality control.
"""

import asyncio
import os
import sys
import json

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services import MultiAgentOrchestrator, InsertionCriteria, InsertionStatus
from src.agents import ReviewOutcome, ReviewFeedback
from smolagents import OpenAIServerModel

from dotenv import load_dotenv
load_dotenv()


async def test_orchestrator_basic():
    """Test basic orchestrator functionality"""

    print("🧪 Testing MultiAgentOrchestrator...")

    # Create models (using same model for simplicity)
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(
        generator_model=model,
        marker_model=model,
        reviewer_model=model,
        debug=True
    )

    try:
        # Generate questions with quality control
        print("\n🚀 Starting orchestrated generation...")
        session = await orchestrator.generate_questions_with_quality_control(
            config_id="mixed_review_gpt4o_mini",
            num_questions=2,
            auto_insert=False  # Don't auto-insert for testing
        )

        print(f"✅ Generation session completed!")
        print(f"   Session ID: {session.session_id}")
        print(f"   Status: {session.status}")
        print(f"   Questions Generated: {session.questions_generated}")
        print(f"   Questions Approved: {session.questions_approved}")
        print(f"   Error Count: {session.error_count}")

        # Display session summary
        summary = orchestrator.get_session_summary(session)
        print(f"\n📊 Session Summary:")
        print(json.dumps(summary, indent=2))

        # Display LLM interactions
        print(f"\n🔄 LLM Interactions ({len(session.llm_interactions)}):")
        for i, interaction in enumerate(session.llm_interactions):
            print(f"   {i+1}. {interaction.agent_type} ({interaction.model_used})")
            print(f"      Success: {interaction.success}")
            print(f"      Time: {interaction.processing_time_ms}ms")
            if interaction.error_message:
                print(f"      Error: {interaction.error_message}")

        # Display review feedbacks
        print(f"\n🔍 Review Feedbacks ({len(session.review_feedbacks)}):")
        for i, feedback in enumerate(session.review_feedbacks):
            print(f"   {i+1}. Outcome: {feedback.outcome.value}")
            print(f"      Score: {feedback.overall_score:.2f}")
            print(f"      Summary: {feedback.feedback_summary}")

        # Display insertion decisions
        print(f"\n🚦 Insertion Decisions ({len(session.insertion_decisions)}):")
        for i, decision in enumerate(session.insertion_decisions):
            print(f"   {i+1}. Status: {decision['insertion_status']}")
            print(f"      Score: {decision['review_score']:.2f}")
            print(f"      Question ID: {decision['question_id']}")

        return True

    except Exception as e:
        print(f"❌ Error testing orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insertion_criteria():
    """Test automated insertion criteria"""

    print("\n🧪 Testing Insertion Criteria...")

    # Test case 1: High quality question (should auto-approve)
    high_quality_feedback = ReviewFeedback(
        outcome=ReviewOutcome.APPROVE,
        overall_score=0.90,
        feedback_summary="Excellent question",
        specific_feedback={},
        suggested_improvements=[],
        syllabus_compliance=0.95,
        difficulty_alignment=0.88,
        marking_quality=0.92
    )

    status = InsertionCriteria.evaluate(high_quality_feedback)
    print(f"✅ High quality question: {status.value} (Expected: auto_approved)")
    assert status == InsertionStatus.AUTO_APPROVED

    # Test case 2: Medium quality question (should manual review)
    medium_quality_feedback = ReviewFeedback(
        outcome=ReviewOutcome.MINOR_REVISIONS,
        overall_score=0.75,
        feedback_summary="Good question with minor issues",
        specific_feedback={},
        suggested_improvements=[],
        syllabus_compliance=0.82,
        difficulty_alignment=0.78,
        marking_quality=0.80
    )

    status = InsertionCriteria.evaluate(medium_quality_feedback)
    print(f"✅ Medium quality question: {status.value} (Expected: manual_review)")
    assert status == InsertionStatus.MANUAL_REVIEW

    # Test case 3: Low quality question (should auto-reject)
    low_quality_feedback = ReviewFeedback(
        outcome=ReviewOutcome.MAJOR_REVISIONS,
        overall_score=0.55,
        feedback_summary="Significant issues found",
        specific_feedback={},
        suggested_improvements=[],
        syllabus_compliance=0.60,
        difficulty_alignment=0.50,
        marking_quality=0.65
    )

    status = InsertionCriteria.evaluate(low_quality_feedback)
    print(f"✅ Low quality question: {status.value} (Expected: auto_rejected)")
    assert status == InsertionStatus.AUTO_REJECTED

    # Test case 4: Rejected question (should auto-reject)
    rejected_feedback = ReviewFeedback(
        outcome=ReviewOutcome.REJECT,
        overall_score=0.30,
        feedback_summary="Fundamentally flawed",
        specific_feedback={},
        suggested_improvements=[],
        syllabus_compliance=0.40,
        difficulty_alignment=0.20,
        marking_quality=0.35
    )

    status = InsertionCriteria.evaluate(rejected_feedback)
    print(f"✅ Rejected question: {status.value} (Expected: auto_rejected)")
    assert status == InsertionStatus.AUTO_REJECTED

    # Test case 5: Low syllabus compliance (should manual review)
    low_compliance_feedback = ReviewFeedback(
        outcome=ReviewOutcome.APPROVE,
        overall_score=0.88,
        feedback_summary="High score but low syllabus compliance",
        specific_feedback={},
        suggested_improvements=[],
        syllabus_compliance=0.70,  # Below threshold
        difficulty_alignment=0.85,
        marking_quality=0.90
    )

    status = InsertionCriteria.evaluate(low_compliance_feedback)
    print(f"✅ Low compliance question: {status.value} (Expected: manual_review)")
    assert status == InsertionStatus.MANUAL_REVIEW

    print("✅ All insertion criteria tests passed!")
    return True


async def main():
    """Run all orchestrator tests"""

    print("🚀 Starting MultiAgentOrchestrator Tests\n")

    # Test insertion criteria (synchronous)
    criteria_test = test_insertion_criteria()

    # Test orchestrator functionality (asynchronous)
    orchestrator_test = await test_orchestrator_basic()

    print(f"\n📊 Test Results:")
    print(f"   Insertion Criteria: {'✅ PASS' if criteria_test else '❌ FAIL'}")
    print(f"   Orchestrator Basic: {'✅ PASS' if orchestrator_test else '❌ FAIL'}")

    if criteria_test and orchestrator_test:
        print("\n🎉 All orchestrator tests passed!")
        return True
    else:
        print("\n💥 Some orchestrator tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
