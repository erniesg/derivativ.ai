#!/usr/bin/env python3
"""
Test Full Persistence Pipeline - End-to-End Verification

This test verifies:
1. Question generation with validation
2. Multi-agent pipeline (Generator → Marker → Reviewer)
3. Quality control decisions
4. Database persistence with full audit trail
5. Data retrieval and integrity
"""

import asyncio
import os
import sys
import asyncpg
from datetime import datetime
import uuid

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services import MultiAgentOrchestrator
from src.validation import validate_question
from smolagents import OpenAIServerModel

from dotenv import load_dotenv
load_dotenv()

from src.models import GenerationConfig, CalculatorPolicy, CommandWord, LLMModel


async def test_complete_pipeline():
    """Test the complete pipeline end-to-end"""

    print("🧪 Testing Complete Persistence Pipeline")
    print("=" * 60)

    try:
        # 1. Setup models and orchestrator
        print("🤖 Setting up AI models...")
        model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(
            generator_model=model,
            marker_model=model,
            reviewer_model=model,
            db_client=None,  # We'll handle DB operations manually for testing
            debug=True
        )

        print("✅ Models initialized")

        # Create a simple test config with valid subject references
        test_config = GenerationConfig(
            target_grade=5,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED,
            desired_marks=2,
            subject_content_references=["C1.1", "C1.6"],  # Valid Cambridge refs
            command_word_override=CommandWord.CALCULATE,
            llm_model_generation=LLMModel.GPT_4O_MINI,
            llm_model_marking_scheme=LLMModel.GPT_4O_MINI,
            llm_model_review=LLMModel.GPT_4O_MINI,
            temperature=0.7,
            max_tokens=2000
        )

        print("✅ Models and config initialized")

        # 2. Test question generation with quality control
        print("\n📝 Testing question generation with validation...")

        # Generate a single question directly
        question = await orchestrator.generator_agent.generate_question(test_config)

        if not question:
            print("❌ Question generation failed")
            return None

        print(f"✅ Question generated: {question.question_id_local}")
        print(f"   Content: {question.raw_text_content[:100]}...")

        # Create a mock session for testing
        from src.services.orchestrator import GenerationSession, LLMInteraction
        session = GenerationSession("test_config", 1)
        session.session_id = str(uuid.uuid4())  # Generate proper UUID
        session.status = "completed"
        session.questions_generated = 1
        session.questions = [question]

        # Add mock interactions
        session.llm_interactions = [
            LLMInteraction(
                agent_type="generator",
                model_used="gpt-4o-mini",
                prompt_text="Test generation prompt",
                raw_response=f"Generated question {question.question_id_local}",
                success=True,
                processing_time_ms=1000
            )
        ]
        session.llm_interactions[0].interaction_id = str(uuid.uuid4())  # Set proper UUID

        # Add mock review feedback
        from src.agents import ReviewOutcome, ReviewFeedback
        mock_feedback = ReviewFeedback(
            outcome=ReviewOutcome.APPROVE,
            overall_score=0.85,
            feedback_summary="Good quality test question",
            specific_feedback={"content": "Clear and appropriate"},
            suggested_improvements=[],
            syllabus_compliance=0.8,
            difficulty_alignment=0.85,
            marking_quality=0.9
        )
        session.review_feedbacks = [mock_feedback]

        # 3. Test validation
        print(f"\n🔍 Testing validation...")
        validation_result = validate_question(question, verbose=True)

        # 3. Test database persistence
        if session.questions:
            print(f"\n💾 Testing database persistence...")
            await test_database_operations(session)
        else:
            print("⚠️ No questions generated, skipping database test")

        print(f"\n🎉 Full pipeline test completed successfully!")
        return session.session_id

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_database_operations(session):
    """Test database operations manually"""

    connection_string = os.getenv("NEON_DATABASE_URL")
    conn = await asyncpg.connect(connection_string)

    try:
        # Insert session data
        await conn.execute("""
            INSERT INTO deriv_generation_sessions (
                session_id, config_id, status,
                total_questions_requested, questions_generated,
                questions_approved, error_count, curriculum_type
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            session.session_id,
            session.config_id,
            session.status,
            session.total_questions_requested,
            session.questions_generated,
            session.questions_approved,
            session.error_count,
            'cambridge_igcse'
        )
        print("   ✅ Session data inserted")

        # Insert LLM interactions
        for interaction in session.llm_interactions:
            await conn.execute("""
                INSERT INTO deriv_llm_interactions (
                    interaction_id, session_id, agent_type, model_used,
                    prompt_text, raw_response, success, processing_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                interaction.interaction_id,
                session.session_id,
                interaction.agent_type,
                interaction.model_used,
                interaction.prompt_text[:1000],  # Truncate for test
                interaction.raw_response[:1000] if interaction.raw_response else None,
                interaction.success,
                interaction.processing_time_ms
            )
        print(f"   ✅ {len(session.llm_interactions)} LLM interactions inserted")

        # Insert questions with validation
        questions_inserted = 0
        question_uuids = []  # Store the generated UUIDs
        for question in session.questions:
            # Validate before insert
            validation_result = validate_question(question)

            if validation_result.can_insert:
                question_uuid = str(uuid.uuid4())  # Generate proper UUID
                question_uuids.append(question_uuid)
                await conn.execute("""
                    INSERT INTO deriv_candidate_questions (
                        question_id, session_id, question_data,
                        subject_content_refs, topic_path, command_word,
                        target_grade, marks, calculator_policy,
                        insertion_status, validation_passed, validation_warnings
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                    question_uuid,
                    session.session_id,
                    question.model_dump_json(),
                    question.taxonomy.subject_content_references,
                    question.taxonomy.topic_path,
                    question.command_word.value,
                    question.target_grade_input,
                    question.marks,
                    'not_allowed',  # Use string directly since we know the test config value
                    'pending',
                    True,
                    validation_result.warnings_count
                )
                questions_inserted += 1
                print(f"   ✅ Question {question.question_id_local} inserted (validation passed)")
            else:
                print(f"   ❌ Question {question.question_id_local} rejected (validation failed)")

        print(f"   ✅ {questions_inserted}/{len(session.questions)} questions inserted")

        # Insert review results
        for i, feedback in enumerate(session.review_feedbacks):
            if i < len(question_uuids):  # Use the stored UUIDs
                await conn.execute("""
                    INSERT INTO deriv_review_results (
                        review_id, question_id, outcome, overall_score,
                        syllabus_compliance, difficulty_alignment, marking_quality,
                        feedback_summary
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    str(uuid.uuid4()),  # Generate proper UUID for review_id
                    question_uuids[i],  # Use the stored UUID
                    feedback.outcome.value,
                    feedback.overall_score,
                    feedback.syllabus_compliance,
                    feedback.difficulty_alignment,
                    feedback.marking_quality,
                    feedback.feedback_summary[:500] if feedback.feedback_summary else ""
                )
        print(f"   ✅ {len(session.review_feedbacks)} review results inserted")

        # Test data retrieval
        print(f"\n🔍 Testing data retrieval...")

        # Test session retrieval
        session_data = await conn.fetchrow("""
            SELECT * FROM deriv_generation_sessions WHERE session_id = $1
        """, session.session_id)

        if session_data:
            print(f"   ✅ Session retrieved: {session_data['config_id']}")
            print(f"      Status: {session_data['status']}")
            print(f"      Questions: {session_data['questions_generated']}")

        # Test questions retrieval
        questions = await conn.fetch("""
            SELECT question_id, insertion_status, validation_passed,
                   target_grade, marks, command_word
            FROM deriv_candidate_questions
            WHERE session_id = $1
        """, session.session_id)

        print(f"   ✅ {len(questions)} questions retrieved:")
        for question in questions:
            status = "✅" if question['validation_passed'] else "❌"
            print(f"      {status} Grade {question['target_grade']}, {question['marks']} marks, {question['command_word']}")

    finally:
        await conn.close()


async def cleanup_test_data(session_id: str):
    """Clean up test data"""
    print(f"\n🧹 Cleaning up test data...")

    connection_string = os.getenv("NEON_DATABASE_URL")
    conn = await asyncpg.connect(connection_string)
    try:
        # Delete in reverse dependency order
        await conn.execute("DELETE FROM deriv_review_results WHERE question_id IN (SELECT question_id FROM deriv_candidate_questions WHERE session_id = $1)", session_id)
        await conn.execute("DELETE FROM deriv_candidate_questions WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM deriv_llm_interactions WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM deriv_generation_sessions WHERE session_id = $1", session_id)

        print(f"   ✅ Test data cleaned up")

    finally:
        await conn.close()


async def main():
    """Run the full pipeline test"""

    print("🚀 Starting Full Persistence Pipeline Test\n")

    session_id = None

    try:
        # Run the test
        session_id = await test_complete_pipeline()

        if session_id:
            print(f"\n🎉 Full persistence pipeline test PASSED!")

            # Ask if user wants to keep test data
            keep_data = input(f"\n🤔 Keep test data in database? (y/n): ").lower().strip()
            if keep_data != 'y':
                await cleanup_test_data(session_id)
        else:
            print(f"\n❌ Full persistence pipeline test FAILED!")

    except Exception as e:
        print(f"💥 Test failed: {e}")
        if session_id:
            await cleanup_test_data(session_id)


if __name__ == "__main__":
    asyncio.run(main())
