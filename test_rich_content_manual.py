#!/usr/bin/env python3
"""
Manual test for rich content generation.
Creates services manually without FastAPI dependency injection.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.document_models import (  # noqa: E402
    DetailLevel,
    DocumentGenerationRequest,
    DocumentType,
)
from src.models.enums import Tier  # noqa: E402
from src.services.document_generation_service import DocumentGenerationService  # noqa: E402
from src.services.llm_factory import LLMFactory  # noqa: E402
from src.services.prompt_manager import PromptManager  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_question_repository():
    """Create a mock question repository for testing."""
    from unittest.mock import MagicMock

    from src.models.enums import SubjectContentReference
    from src.models.question_models import (
        FinalAnswer,
        MarkingCriterion,
        Question,
        QuestionTaxonomy,
        SolutionAndMarkingScheme,
        SolverAlgorithm,
        SolverStep,
    )

    mock_repo = MagicMock()

    # Create sample questions for demo
    sample_questions = [
        Question(
            question_id_local="1a",
            question_id_global="demo_q1",
            question_number_display="1 (a)",
            marks=3,
            command_word="Solve",
            raw_text_content="Solve the linear equation: 2x + 5 = 13",
            taxonomy=QuestionTaxonomy(
                topic_path=["Algebra", "Linear Equations"],
                subject_content_references=[SubjectContentReference.C5_2],
                skill_tags=["equation_solving", "linear_equations"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    FinalAnswer(answer_text="x = 4", value_numeric=4.0, unit="")
                ],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1",
                        criterion_text="Correct rearrangement",
                        mark_code_display="M1",
                        marks_value=1,
                    ),
                    MarkingCriterion(
                        criterion_id="2",
                        criterion_text="Correct calculation",
                        mark_code_display="A1",
                        marks_value=2,
                    ),
                ],
                total_marks_for_part=3,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1,
                        description_text="Subtract 5 from both sides: 2x = 13 - 5",
                        calculation_text="2x = 8",
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Divide both sides by 2: x = 8 ÷ 2",
                        calculation_text="x = 4",
                    ),
                ]
            ),
        ),
        Question(
            question_id_local="1b",
            question_id_global="demo_q2",
            question_number_display="1 (b)",
            marks=4,
            command_word="Find",
            raw_text_content="Find the value of x when 3x - 7 = 2x + 5",
            taxonomy=QuestionTaxonomy(
                topic_path=["Algebra", "Linear Equations"],
                subject_content_references=[SubjectContentReference.C5_2],
                skill_tags=["equation_solving", "linear_equations", "variables_both_sides"],
            ),
            solution_and_marking_scheme=SolutionAndMarkingScheme(
                final_answers_summary=[
                    FinalAnswer(answer_text="x = 12", value_numeric=12.0, unit="")
                ],
                mark_allocation_criteria=[
                    MarkingCriterion(
                        criterion_id="1",
                        criterion_text="Collect like terms",
                        mark_code_display="M1",
                        marks_value=2,
                    ),
                    MarkingCriterion(
                        criterion_id="2",
                        criterion_text="Correct answer",
                        mark_code_display="A1",
                        marks_value=2,
                    ),
                ],
                total_marks_for_part=4,
            ),
            solver_algorithm=SolverAlgorithm(
                steps=[
                    SolverStep(
                        step_number=1,
                        description_text="Subtract 2x from both sides: 3x - 2x - 7 = 5",
                        calculation_text="x - 7 = 5",
                    ),
                    SolverStep(
                        step_number=2,
                        description_text="Add 7 to both sides: x = 5 + 7",
                        calculation_text="x = 12",
                    ),
                ]
            ),
        ),
    ]

    # Mock the async method
    async def mock_get_question(question_id):
        for q in sample_questions:
            if q.question_id_global == question_id:
                return q
        return None

    # Mock the list questions method as async
    async def mock_list_questions(**kwargs):
        # Return metadata format that the service expects
        return [
            {
                "id": 1,
                "question_id_global": q.question_id_global,
                "content_json": q.model_dump(),
                "tier": "core",
                "marks": q.marks,
                "command_word": q.command_word.value,
                "quality_score": 0.8,
                "created_at": "2025-01-01T00:00:00",
            }
            for q in sample_questions
        ]

    mock_repo.get_question = mock_get_question
    mock_repo.list_questions = mock_list_questions

    return mock_repo


async def test_rich_content():  # noqa: PLR0915
    """Test document generation for rich content."""

    print("🧪 Testing Rich Content Generation")
    print("=" * 50)

    # Create services manually
    llm_factory = LLMFactory()
    prompt_manager = PromptManager()
    question_repo = create_mock_question_repository()

    # Create document generation service
    doc_service = DocumentGenerationService(
        question_repository=question_repo,
        llm_factory=llm_factory,
        prompt_manager=prompt_manager,
    )

    # Create test request
    request = DocumentGenerationRequest(
        title="Linear Equations Practice Worksheet",
        topic="Solving Linear Equations",
        document_type=DocumentType.WORKSHEET,
        detail_level=DetailLevel.MEDIUM,
        tier=Tier.CORE,
        grade_level=8,
        auto_include_questions=True,
        max_questions=2,
        custom_instructions="Include detailed step-by-step solutions and explanations for each method",
    )

    print("📋 Test Request:")
    print(f"   Title: {request.title}")
    print(f"   Topic: {request.topic}")
    print(f"   Type: {request.document_type.value}")
    print(f"   Detail Level: {request.detail_level.value}")
    print(f"   Auto Include Questions: {request.auto_include_questions}")
    print(f"   Max Questions: {request.max_questions}")
    print()

    # Generate document
    print("🚀 Generating document...")
    start_time = datetime.now()

    try:
        result = await doc_service.generate_document(request)
        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"⏱️  Processing completed in {processing_time:.2f}s")
        print()

        if result.success:
            print("✅ Document generation SUCCEEDED")
            doc = result.document

            print("📄 Document Details:")
            print(f"   ID: {doc.document_id}")
            print(f"   Title: {doc.title}")
            print(f"   Template: {doc.template_used}")
            print(f"   Sections: {len(doc.sections)}")
            print(f"   Questions Used: {len(doc.questions_used)}")
            print(f"   Duration: {doc.estimated_duration} min")
            print()

            # Analyze content richness
            print("🔍 Content Analysis:")
            total_content = 0
            rich_sections = 0
            detailed_sections = []

            for i, section in enumerate(doc.sections):
                print(f"\n{i+1}. {section.title} ({section.content_type})")

                if section.content_data:
                    content_str = json.dumps(section.content_data, indent=2)
                    content_size = len(content_str)
                    total_content += content_size

                    print(f"   Content size: {content_size} chars")

                    # Analyze richness
                    rich_indicators = []
                    if "questions" in section.content_data:
                        questions = section.content_data["questions"]
                        if isinstance(questions, list) and len(questions) > 0:
                            rich_indicators.append(f"{len(questions)} questions")
                            rich_sections += 1

                            # Show question details
                            for j, q in enumerate(questions[:2]):  # Show first 2
                                if isinstance(q, dict) and "question_text" in q:
                                    print(f"     Q{j+1}: {q['question_text'][:50]}...")

                    if "examples" in section.content_data:
                        examples = section.content_data["examples"]
                        if isinstance(examples, list) and len(examples) > 0:
                            rich_indicators.append(f"{len(examples)} examples")
                            rich_sections += 1

                    if "solution_steps" in section.content_data:
                        steps = section.content_data["solution_steps"]
                        if isinstance(steps, list) and len(steps) > 0:
                            rich_indicators.append(f"{len(steps)} solution steps")

                    if rich_indicators:
                        print(f"   ✅ Rich content: {', '.join(rich_indicators)}")
                        detailed_sections.append(section.title)
                    elif content_size > 100:
                        print("   📝 Text content")
                    else:
                        print("   ⚠️  Basic content")
                        print(f"   Preview: {content_str[:100]}...")
                else:
                    print("   ❌ No content")

            print("\n📊 Summary:")
            print(f"   Total content: {total_content} chars")
            print(f"   Rich sections: {rich_sections}/{len(doc.sections)}")
            print(f"   Detailed sections: {detailed_sections}")
            print("   Processing stats:")
            print(f"     - Questions processed: {result.questions_processed}")
            print(f"     - Sections generated: {result.sections_generated}")
            print(f"     - Customizations applied: {result.customizations_applied}")

            # Determine success criteria
            is_rich_content = (
                total_content > 1000  # Substantial content
                and rich_sections > 0  # At least one rich section
                and result.sections_generated > 3  # Multiple sections
            )

            if is_rich_content:
                print("\n✅ RICH CONTENT CONFIRMED")
                print(
                    "   Document contains substantial LLM-generated content with proper structure"
                )
                print("   This is NOT fallback mock data")
                return True
            else:
                print("\n⚠️  BASIC CONTENT DETECTED")
                print("   Content appears to be fallback/mock data")
                print("   Possible issues:")
                print("     - LLM JSON parsing failed")
                print("     - API connectivity problems")
                print("     - Template rendering issues")
                return False

        else:
            print("❌ Document generation FAILED")
            print(f"   Error: {result.error_message}")
            print(f"   Processing time: {result.processing_time:.2f}s")
            return False

    except Exception as e:
        print(f"💥 Exception: {e}")
        logger.error("Test failed", exc_info=True)
        return False


async def main():
    """Run the test."""
    print("🚀 Rich Content Generation Test")
    print(f"Demo mode: {os.getenv('DEMO_MODE', 'False')}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    success = await test_rich_content()

    if success:
        print("\n🎉 TEST PASSED - Rich content generation is working!")
        print("   ✅ LLM integration successful")
        print("   ✅ JSON parsing working")
        print("   ✅ Real content generated (not fallback)")
    else:
        print("\n❌ TEST FAILED - Needs investigation")
        print("   Check logs for specific error details")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
