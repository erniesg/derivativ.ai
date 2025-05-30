#!/usr/bin/env python3
"""
Test script for the question generation system.
Validates the end-to-end pipeline from configuration to database storage.
"""

import asyncio
import os
import json
import sys
from dotenv import load_dotenv

# Add parent directory to Python path to find src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import QuestionGenerationService, GenerationRequest, CalculatorPolicy, CommandWord


async def test_basic_generation():
    """Test basic question generation functionality"""

    # Load environment variables
    load_dotenv()

    database_url = os.getenv("NEON_DATABASE_URL")
    if not database_url:
        print("❌ NEON_DATABASE_URL not found in environment")
        print("Please set up your .env file with database credentials")
        return False

    print("🚀 Starting Question Generation Test")

    # Initialize service
    service = QuestionGenerationService(database_url)

    try:
        await service.initialize()
        print("✅ Service initialized successfully")

        # Test 1: Generate basic questions for different grades
        print("\n📝 Test 1: Generating basic questions...")

        request = GenerationRequest(
            target_grades=[2, 4, 6],
            count_per_grade=1,
            calculator_policy=CalculatorPolicy.NOT_ALLOWED
        )

        response = await service.generate_questions(request)

        print(f"📊 Generation Results:")
        print(f"   - Requested: {response.total_requested}")
        print(f"   - Generated: {response.total_generated}")
        print(f"   - Failed: {len(response.failed_generations)}")
        print(f"   - Time: {response.generation_time_seconds:.2f}s")

        if response.failed_generations:
            print("\n❌ Failed generations:")
            for failure in response.failed_generations:
                print(f"   - {failure}")

        # Test 2: Generate questions by topic
        print("\n📝 Test 2: Generating questions by topic...")

        topic_response = await service.generate_by_topic(
            subject_content_references=["C1.6", "C1.4"],
            target_grades=[3, 5],
            count_per_grade=1,
            command_word=CommandWord.WORK_OUT
        )

        print(f"📊 Topic-based Generation Results:")
        print(f"   - Generated: {topic_response.total_generated}")
        print(f"   - Failed: {len(topic_response.failed_generations)}")

        # Test 3: Check database storage
        print("\n📝 Test 3: Checking database storage...")

        all_questions = await service.get_candidate_questions(limit=10)
        print(f"📊 Questions in database: {len(all_questions)}")

        if all_questions:
            sample_question = all_questions[0]
            print(f"\n📋 Sample Generated Question:")
            print(f"   - ID: {sample_question.get('question_id_global', 'N/A')}")
            print(f"   - Grade: {sample_question.get('target_grade_input', 'N/A')}")
            print(f"   - Marks: {sample_question.get('marks', 'N/A')}")
            print(f"   - Command: {sample_question.get('command_word', 'N/A')}")
            print(f"   - Text: {sample_question.get('raw_text_content', 'N/A')[:100]}...")
            print(f"   - Status: {sample_question.get('status', 'N/A')}")

        # Test 4: Generation statistics
        print("\n📝 Test 4: Getting generation statistics...")

        stats = await service.get_generation_statistics()
        if stats.get('grade_distribution'):
            print("📊 Grade Distribution:")
            for grade_stat in stats['grade_distribution']:
                print(f"   - Grade {grade_stat.get('target_grade_input', 'N/A')}: {grade_stat.get('count_per_grade', 0)} questions")

        print("\n✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await service.shutdown()
        print("🔚 Service shutdown complete")


async def test_individual_question():
    """Test generation of a single question with detailed output"""

    load_dotenv()
    database_url = os.getenv("NEON_DATABASE_URL")

    if not database_url:
        print("❌ Database URL not found")
        return

    print("\n🔍 Detailed Single Question Test")

    service = QuestionGenerationService(database_url)

    try:
        await service.initialize()

        # Generate a single Grade 4 question
        request = GenerationRequest(
            target_grades=[4],
            count_per_grade=1,
            subject_content_references=["C1.6"],  # The four operations
            calculator_policy=CalculatorPolicy.NOT_ALLOWED
        )

        response = await service.generate_questions(request)

        if response.generated_questions:
            question = response.generated_questions[0]

            print(f"\n📋 Generated Question Details:")
            print(f"{'='*50}")
            print(f"ID: {question.question_id_global}")
            print(f"Target Grade: {question.target_grade_input}")
            print(f"Marks: {question.marks}")
            print(f"Command Word: {question.command_word.value}")
            print(f"\nQuestion Text:")
            print(f"{question.raw_text_content}")

            if question.formatted_text_latex:
                print(f"\nLaTeX Format:")
                print(f"{question.formatted_text_latex}")

            print(f"\nTaxonomy:")
            print(f"- Topic Path: {' > '.join(question.taxonomy.topic_path)}")
            print(f"- Subject Refs: {', '.join(question.taxonomy.subject_content_references)}")
            print(f"- Skills: {', '.join(question.taxonomy.skill_tags)}")
            print(f"- Cognitive Level: {question.taxonomy.cognitive_level}")
            print(f"- Difficulty: {question.taxonomy.difficulty_estimate_0_to_1}")

            print(f"\nSolution & Marking:")
            for i, answer in enumerate(question.solution_and_marking_scheme.final_answers_summary):
                print(f"- Answer {i+1}: {answer.answer_text}")
                if answer.value_numeric is not None:
                    print(f"  Numeric: {answer.value_numeric}")
                if answer.unit:
                    print(f"  Unit: {answer.unit}")

            print(f"\nMark Allocation:")
            for criterion in question.solution_and_marking_scheme.mark_allocation_criteria:
                print(f"- {criterion.mark_code_display}: {criterion.criterion_text} ({criterion.marks_value} mark(s))")

            print(f"\nSolver Algorithm:")
            for step in question.solver_algorithm.steps:
                print(f"- Step {step.step_number}: {step.description_text}")
                if step.mathematical_expression_latex:
                    print(f"  Math: {step.mathematical_expression_latex}")

            print(f"\nGeneration Metadata:")
            print(f"- Model Used: {question.llm_model_used_generation}")
            print(f"- Prompt Version: {question.prompt_template_version_generation}")
            print(f"- Validation Errors: {len(question.validation_errors)}")
            if question.validation_errors:
                for error in question.validation_errors:
                    print(f"  ⚠️  {error}")

            print(f"{'='*50}")

        else:
            print("❌ No questions generated")
            if response.failed_generations:
                print("Failures:")
                for failure in response.failed_generations:
                    print(f"  - {failure}")

    finally:
        await service.shutdown()


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Test question generation system")
    parser.add_argument("--mode", choices=["basic", "detailed"], default="basic",
                       help="Test mode: basic (multiple questions) or detailed (single question)")

    args = parser.parse_args()

    if args.mode == "detailed":
        asyncio.run(test_individual_question())
    else:
        asyncio.run(test_basic_generation())


if __name__ == "__main__":
    main()
