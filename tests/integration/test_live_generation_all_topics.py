#!/usr/bin/env python3
"""
Live test: Generate one question per main Cambridge IGCSE topic.
Uses dev tables and prompts for cleanup at the end.
"""

import asyncio
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from src.agents.question_generator import QuestionGeneratorAgent
from src.core.config import get_settings
from src.database.supabase_repository import (
    GenerationSessionRepository,
    QuestionRepository,
    get_supabase_client,
)
from src.models.enums import CalculatorPolicy, CommandWord, Tier
from src.models.question_models import GenerationRequest, GenerationSession, GenerationStatus
from src.services.llm_factory import LLMFactory

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Main Cambridge IGCSE Mathematics topics (9 categories)
MAIN_TOPICS = [
    "algebra",
    "geometry",
    "trigonometry",
    "statistics",
    "probability",
    "number",
    "coordinate geometry",
    "mensuration",
    "vectors",
]


def setup_dev_mode():
    """Force dev mode for this test."""
    os.environ["DB_MODE"] = "dev"


def generate_random_params():
    """Generate randomized parameters for question generation."""
    return {
        "tier": random.choice([Tier.CORE, Tier.EXTENDED]),
        "grade_level": random.randint(7, 11),
        "marks": random.choice([2, 3, 4, 5, 6]),
        "command_word": random.choice(
            [
                CommandWord.CALCULATE,
                CommandWord.SOLVE,
                CommandWord.FIND,
                CommandWord.SHOW,
                CommandWord.EXPLAIN,
                CommandWord.DETERMINE,
            ]
        ),
        "calculator_policy": random.choice(
            [
                CalculatorPolicy.ALLOWED,
                CalculatorPolicy.NOT_ALLOWED,
                CalculatorPolicy.REQUIRED,
            ]
        ),
        "include_diagrams": False,  # No diagrams as requested
    }


async def generate_question_for_topic(agent, topic, index, total):
    """Generate a single question for a given topic."""
    params = generate_random_params()
    params["topic"] = topic

    print(f"\n[{index}/{total}] Generating: {topic}")
    print(
        f"  Parameters: {params['tier'].value}, Grade {params['grade_level']}, {params['marks']} marks"
    )
    print(
        f"  Command: {params['command_word'].value}, Calculator: {params['calculator_policy'].value}"
    )

    try:
        result = await agent.process(params)

        if result.success and result.output and "question" in result.output:
            quality_score = result.output.get("quality_score", "N/A")
            print(f"  ✅ Success! Quality: {quality_score}")
            return topic, result.output["question"], True
        else:
            error = result.error if hasattr(result, "error") else "Unknown error"
            print(f"  ❌ Failed: {error}")
            return topic, None, False

    except Exception as e:
        print(f"  ❌ Exception: {e!s}")
        return topic, None, False


async def main():  # noqa: PLR0915
    """Main test function."""
    print("=" * 80)
    print("LIVE QUESTION GENERATION TEST - ALL CAMBRIDGE IGCSE TOPICS")
    print("=" * 80)

    # Setup dev mode
    setup_dev_mode()
    settings = get_settings()

    print(f"\nDatabase mode: {settings.db_mode}")
    print(f"Table prefix: '{settings.table_prefix}'")
    print(
        f"Using tables: {settings.table_prefix}generated_questions, {settings.table_prefix}generation_sessions"
    )

    # Check credentials
    if not settings.supabase_url or not settings.supabase_anon_key:
        print("\n❌ ERROR: Missing Supabase credentials!")
        print("Set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file")
        return

    print("\n🚀 Initializing services...")

    # Create services
    supabase_client = get_supabase_client(settings.supabase_url, settings.supabase_anon_key)
    question_repo = QuestionRepository(supabase_client)
    session_repo = GenerationSessionRepository(supabase_client)

    llm_factory = LLMFactory()
    llm_service = llm_factory.get_service(provider="openai")
    agent = QuestionGeneratorAgent(llm_service=llm_service)

    # Create generation session
    session = GenerationSession(
        session_id=uuid4(),
        request=GenerationRequest(
            topic="multi-topic",
            tier=Tier.CORE,
            marks=4,
            command_word=CommandWord.CALCULATE,
        ),
        questions=[],
        quality_decisions=[],
        agent_results=[],
        status=GenerationStatus.CANDIDATE,
        metadata={
            "test_type": "live_all_topics_generation",
            "total_topics": len(MAIN_TOPICS),
            "include_diagrams": False,
        },
    )

    print("\n💾 Creating generation session...")
    try:
        session_db_id = session_repo.save_session(session)
        print(f"✅ Session saved: {session_db_id}")
    except Exception as e:
        print(f"❌ Failed to save session: {e}")
        return

    # Generate questions
    print(f"\n🎯 Generating {len(MAIN_TOPICS)} questions (one per topic)...")
    print("This may take a few minutes...\n")

    start_time = datetime.utcnow()
    generated_questions = []
    results = []

    for index, topic in enumerate(MAIN_TOPICS, 1):
        topic_name, question, success = await generate_question_for_topic(
            agent, topic, index, len(MAIN_TOPICS)
        )

        results.append((topic_name, success))

        # Save successful questions
        if success and question:
            try:
                question_id = question_repo.save_question(question)
                generated_questions.append((question_id, topic_name))
                print(f"  💾 Saved to dev table: {question_id}")
            except Exception as e:
                print(f"  ⚠️  Failed to save: {e}")

        # Small delay to avoid rate limiting
        await asyncio.sleep(1)

    # Calculate stats
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    successful = sum(1 for _, success in results if success)

    # Update session
    try:
        session_repo.update_session_status(str(session.session_id), GenerationStatus.CANDIDATE)
        print("\n✅ Session marked as completed")
    except Exception as e:
        print(f"\n⚠️  Failed to update session: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"Topics attempted: {len(MAIN_TOPICS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(MAIN_TOPICS) - successful}")
    print(f"Saved to dev tables: {len(generated_questions)}")
    print(f"Total time: {duration:.1f} seconds")
    print(f"Average per question: {duration/len(MAIN_TOPICS):.1f} seconds")

    # Show sample questions
    if generated_questions:
        print("\n" + "-" * 80)
        print("SAMPLE GENERATED QUESTIONS:")
        print("-" * 80)

        sample_count = min(3, len(generated_questions))
        for i, (question_id, topic) in enumerate(generated_questions[:sample_count], 1):
            print(f"{i}. Topic: {topic}")
            print(f"   ID: {question_id}")

    # Cleanup prompt
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    if generated_questions:
        print(f"\n{len(generated_questions)} questions were saved to dev tables.")
        response = input("Remove all generated questions from dev tables? (y/n): ").strip().lower()

        if response in ["y", "yes"]:
            print("\n🧹 Cleaning up...")
            removed = 0

            for question_id, topic in generated_questions:
                try:
                    supabase_client.table("dev_generated_questions").delete().eq(
                        "id", question_id
                    ).execute()
                    print(f"  ✅ Removed: {topic}")
                    removed += 1
                except Exception as e:
                    print(f"  ❌ Failed to remove {topic}: {e}")

            # Remove session
            try:
                supabase_client.table("dev_generation_sessions").delete().eq(
                    "id", session_db_id
                ).execute()
                print("  ✅ Removed session")
            except Exception as e:
                print(f"  ❌ Failed to remove session: {e}")

            print(f"\n🗑️  Cleaned up {removed}/{len(generated_questions)} questions")
        else:
            print("\n📋 Questions retained in dev tables")
            print(f"Session ID: {session.session_id}")
    else:
        print("\nNo questions to clean up.")

    print("\n✨ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
