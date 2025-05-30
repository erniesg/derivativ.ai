#!/usr/bin/env python3
"""
Direct test of PayloadPublisher functionality.
"""

from dotenv import load_dotenv
load_dotenv()

from src.services.payload_publisher import PayloadPublisher
import asyncio
import uuid
from src.models.question_models import *
from datetime import datetime

async def test_payload_direct():
    """Test PayloadPublisher directly."""
    print("🧪 Testing PayloadPublisher directly")
    print("=" * 40)

    # Create sample question
    unique_id = str(uuid.uuid4())[:8]
    question = CandidateQuestion(
        question_id_local=f'test_q_{unique_id}',
        question_id_global=f'test_global_{unique_id}',
        question_number_display='Direct Test Question',
        marks=3,
        command_word=CommandWord.CALCULATE,
        raw_text_content=f'Direct test question {unique_id}',
        formatted_text_latex=None,
        taxonomy=QuestionTaxonomy(
            topic_path=['Test'],
            subject_content_references=['C1.1'],
            skill_tags=['TEST_SKILL'],
            cognitive_level='ProceduralFluency',
            difficulty_estimate_0_to_1=0.7
        ),
        solution_and_marking_scheme=SolutionAndMarkingScheme(
            final_answers_summary=[AnswerSummary(answer_text='42', value_numeric=42, unit='cm')],
            mark_allocation_criteria=[MarkAllocationCriterion(
                criterion_id=f'test_crit_{unique_id}',
                criterion_text='Test criterion',
                mark_code_display='M3',
                marks_value=3.0,
                mark_type_primary='M'
            )],
            total_marks_for_part=3
        ),
        solver_algorithm=SolverAlgorithm(
            steps=[SolverStep(
                step_number=1,
                description_text='Test step',
                mathematical_expression_latex='x = 42',
                skill_applied_tag='TEST_SKILL'
            )]
        ),
        generation_id=uuid.uuid4(),
        target_grade_input=7,
        llm_model_used_generation=LLMModel.GPT_4O.value,
        llm_model_used_marking_scheme=LLMModel.GPT_4O.value,
        llm_model_used_review=LLMModel.CLAUDE_4_SONNET.value,
        prompt_template_version_generation='v1.0',
        prompt_template_version_marking_scheme='v1.0',
        prompt_template_version_review='v1.0',
        generation_timestamp=datetime.utcnow(),
        status=GenerationStatus.CANDIDATE
    )

    publisher = PayloadPublisher()
    print(f'✅ Publisher enabled: {publisher.is_enabled()}')

    if not publisher.is_enabled():
        print("❌ Publisher not enabled - check environment variables")
        return

    try:
        print(f"📤 Publishing question: {question.question_id_global}")

        result = await publisher.publish_question(question)
        print(f'✅ Published question ID: {result}')

        if result:
            # Verify
            exists = await publisher.verify_question_exists(question.question_id_global)
            print(f'✅ Question exists in Payload: {exists}')

            # Cleanup
            deleted = await publisher.delete_question(result)
            print(f'✅ Cleaned up: {deleted}')
        else:
            print("❌ Failed to publish question")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_payload_direct())
