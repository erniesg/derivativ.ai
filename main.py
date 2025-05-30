#!/usr/bin/env python3
"""
Main entry point for the IGCSE Mathematics Question Generation System.
Provides CLI interface for generating candidate questions.
"""

import asyncio
import argparse
import os
from typing import List
from dotenv import load_dotenv

from src.services.generation_service import QuestionGenerationService
from src.models.question_models import (
    GenerationRequest, CalculatorPolicy, CommandWord, LLMModel
)


async def generate_questions_cli(
    target_grades: List[int],
    count_per_grade: int,
    subject_refs: List[str] = None,
    calculator_policy: str = "not_allowed",
    command_word: str = None,
    seed_question_id: str = None,
    model: str = "gpt-4o",
    debug: bool = False
):
    """Generate questions via CLI interface"""

    # Load environment
    load_dotenv()
    database_url = os.getenv("NEON_DATABASE_URL")

    if not database_url:
        print("❌ Error: NEON_DATABASE_URL not found in environment variables")
        print("Please set up your .env file with database credentials")
        return

    print(f"🚀 IGCSE Mathematics Question Generator")
    print(f"Target Grades: {target_grades}")
    print(f"Questions per Grade: {count_per_grade}")
    print(f"Model: {model}")
    print(f"Calculator Policy: {calculator_policy}")
    if debug:
        print("🐛 Debug mode: ENABLED")

    if subject_refs:
        print(f"Subject References: {', '.join(subject_refs)}")
    if command_word:
        print(f"Command Word: {command_word}")
    if seed_question_id:
        print(f"Seed Question: {seed_question_id}")

    print("-" * 50)

    # Initialize service with debug flag
    service = QuestionGenerationService(database_url, debug=debug)

    try:
        await service.initialize()
        print("✅ Service initialized")

        # Prepare request
        calc_policy = CalculatorPolicy(calculator_policy)
        cmd_word = CommandWord(command_word) if command_word else None

        if subject_refs:
            # Generate by topic
            response = await service.generate_by_topic(
                subject_content_references=subject_refs,
                target_grades=target_grades,
                count_per_grade=count_per_grade,
                command_word=cmd_word
            )
        elif seed_question_id:
            # Generate from seed
            response = await service.generate_batch_from_seed(
                seed_question_id=seed_question_id,
                target_grades=target_grades,
                count_per_grade=count_per_grade
            )
        else:
            # Basic generation
            request = GenerationRequest(
                target_grades=target_grades,
                count_per_grade=count_per_grade,
                calculator_policy=calc_policy
            )
            response = await service.generate_questions(request)

        # Display results
        print(f"\n📊 Generation Results:")
        print(f"   Requested: {response.total_requested}")
        print(f"   Generated: {response.total_generated}")
        print(f"   Failed: {len(response.failed_generations)}")
        print(f"   Time: {response.generation_time_seconds:.2f} seconds")

        if response.generated_questions:
            print(f"\n📋 Generated Questions:")
            for i, question in enumerate(response.generated_questions, 1):
                print(f"\n{i}. {question.question_id_global} (Grade {question.target_grade_input}, {question.marks} marks)")
                print(f"   Command: {question.command_word.value}")
                print(f"   Topic: {' > '.join(question.taxonomy.topic_path)}")
                print(f"   Text: {question.raw_text_content[:100]}...")

                if question.validation_errors:
                    print(f"   ⚠️  Validation issues: {len(question.validation_errors)}")

        if response.failed_generations:
            print(f"\n❌ Failed Generations:")
            for failure in response.failed_generations:
                print(f"   - {failure.get('error', 'Unknown error')}")

        print(f"\n✅ Generation complete! Questions saved to database.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await service.shutdown()


async def list_questions_cli(grade: int = None, limit: int = 20, debug: bool = False):
    """List generated questions"""

    load_dotenv()
    database_url = os.getenv("NEON_DATABASE_URL")

    if not database_url:
        print("❌ Error: NEON_DATABASE_URL not found")
        return

    service = QuestionGenerationService(database_url, debug=debug)

    try:
        await service.initialize()

        questions = await service.get_candidate_questions(
            target_grade=grade,
            limit=limit
        )

        print(f"📋 Candidate Questions (showing {len(questions)} of max {limit})")
        if grade:
            print(f"Filtered by Grade: {grade}")
        print("-" * 70)

        for question in questions:
            print(f"ID: {question.get('question_id_global', 'N/A')}")
            print(f"Grade: {question.get('target_grade_input', 'N/A')} | "
                  f"Marks: {question.get('marks', 'N/A')} | "
                  f"Status: {question.get('status', 'N/A')}")
            print(f"Text: {question.get('raw_text_content', 'N/A')[:80]}...")
            print(f"Created: {question.get('created_at', 'N/A')}")
            print("-" * 70)

    finally:
        await service.shutdown()


async def stats_cli(debug: bool = False):
    """Show generation statistics"""

    load_dotenv()
    database_url = os.getenv("NEON_DATABASE_URL")

    if not database_url:
        print("❌ Error: NEON_DATABASE_URL not found")
        return

    service = QuestionGenerationService(database_url, debug=debug)

    try:
        await service.initialize()

        stats = await service.get_generation_statistics()

        print("📊 Question Generation Statistics")
        print("=" * 40)

        if stats.get('grade_distribution'):
            print("\nBy Grade:")
            for grade_stat in stats['grade_distribution']:
                grade = grade_stat.get('target_grade_input', 'N/A')
                total = grade_stat.get('total_questions', 0)
                pending = grade_stat.get('pending_review', 0)
                accepted = grade_stat.get('accepted', 0)
                rejected = grade_stat.get('rejected', 0)

                print(f"  Grade {grade}: {total} total "
                      f"({pending} pending, {accepted} accepted, {rejected} rejected)")

            overall_stats = stats['grade_distribution'][0] if stats['grade_distribution'] else {}
            print(f"\nOverall:")
            print(f"  Total Questions: {overall_stats.get('total_questions', 0)}")
            print(f"  Pending Review: {overall_stats.get('pending_review', 0)}")
            print(f"  Accepted: {overall_stats.get('accepted', 0)}")
            print(f"  Rejected: {overall_stats.get('rejected', 0)}")

            avg_confidence = overall_stats.get('avg_confidence')
            if avg_confidence:
                print(f"  Average Confidence: {avg_confidence:.3f}")

        print(f"\nLast Updated: {stats.get('timestamp', 'N/A')}")

    finally:
        await service.shutdown()


def main():
    """Main CLI interface"""

    parser = argparse.ArgumentParser(
        description="IGCSE Mathematics Question Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 2 questions each for grades 3, 5, 7
  python main.py generate --grades 3 5 7 --count 2

  # Generate questions on specific topics
  python main.py generate --grades 4 6 --subject-refs C1.6 C1.4 --count 1

  # Generate with specific command word and debug mode
  python main.py generate --grades 5 --command-word "Calculate" --count 3 --debug

  # Generate from a seed question
  python main.py generate --grades 4 5 6 --seed-question 0580_s15_qp_01_q1a --count 2

  # List generated questions
  python main.py list --grade 5 --limit 10

  # Show statistics
  python main.py stats
        """
    )

    # Global debug flag
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new questions")
    gen_parser.add_argument("--grades", type=int, nargs="+", required=True,
                           help="Target grades (1-9)")
    gen_parser.add_argument("--count", type=int, default=1,
                           help="Number of questions per grade (default: 1)")
    gen_parser.add_argument("--subject-refs", nargs="+",
                           help="Subject content references (e.g., C1.6 C1.4)")
    gen_parser.add_argument("--calculator-policy",
                           choices=["allowed", "not_allowed", "varies_by_question"],
                           default="not_allowed",
                           help="Calculator usage policy")
    gen_parser.add_argument("--command-word",
                           choices=[cw.value for cw in CommandWord],
                           help="Specific command word to use")
    gen_parser.add_argument("--seed-question",
                           help="Past paper question ID to use as inspiration")
    gen_parser.add_argument("--model",
                           choices=[model.value for model in LLMModel],
                           default="gpt-4o",
                           help="LLM model to use")

    # List command
    list_parser = subparsers.add_parser("list", help="List generated questions")
    list_parser.add_argument("--grade", type=int, help="Filter by grade")
    list_parser.add_argument("--limit", type=int, default=20, help="Max questions to show")

    # Stats command
    subparsers.add_parser("stats", help="Show generation statistics")

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(generate_questions_cli(
            target_grades=args.grades,
            count_per_grade=args.count,
            subject_refs=args.subject_refs,
            calculator_policy=args.calculator_policy,
            command_word=args.command_word,
            seed_question_id=args.seed_question,
            model=args.model,
            debug=args.debug
        ))
    elif args.command == "list":
        asyncio.run(list_questions_cli(args.grade, args.limit, debug=args.debug))
    elif args.command == "stats":
        asyncio.run(stats_cli(debug=args.debug))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
