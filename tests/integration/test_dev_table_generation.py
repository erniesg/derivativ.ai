"""
Integration test for dev table question generation across main Cambridge IGCSE topics.
Tests dev/prod table switching and saves questions to dev tables.
"""

import os
import random
from typing import ClassVar
from uuid import uuid4

import pytest

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


class TestDevTableGeneration:
    """Test question generation with dev table mode."""

    # Main Cambridge IGCSE Mathematics topic categories (9 topics)
    MAIN_TOPICS: ClassVar[list[str]] = [
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

    @pytest.fixture(autouse=True)
    def setup_dev_mode(self):
        """Force dev mode for all tests."""
        original_mode = os.environ.get("DB_MODE")
        os.environ["DB_MODE"] = "dev"
        yield
        # Restore original mode
        if original_mode:
            os.environ["DB_MODE"] = original_mode
        else:
            os.environ.pop("DB_MODE", None)

    @pytest.fixture
    def settings(self):
        """Get settings with dev mode enabled."""
        return get_settings()

    @pytest.fixture
    def supabase_client(self, settings):
        """Create Supabase client."""
        if not settings.supabase_url or not settings.supabase_anon_key:
            pytest.skip("Supabase credentials not configured")

        return get_supabase_client(settings.supabase_url, settings.supabase_anon_key)

    @pytest.fixture
    def repositories(self, supabase_client):
        """Create repository instances."""
        return {
            "question": QuestionRepository(supabase_client),
            "session": GenerationSessionRepository(supabase_client),
        }

    @pytest.fixture
    def agent(self):
        """Create question generator agent."""
        llm_factory = LLMFactory()
        llm_service = llm_factory.get_service(provider="openai")
        return QuestionGeneratorAgent(llm_service=llm_service)

    def _generate_random_params(self):
        """Generate random parameters for question generation."""
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

    @pytest.mark.asyncio
    async def test_dev_table_configuration(self, settings):
        """Test that dev mode is properly configured."""
        assert settings.db_mode == "dev"
        assert settings.table_prefix == "dev_"

    @pytest.mark.asyncio
    async def test_generate_questions_all_topics(self, agent, repositories, settings):
        """Generate one question per main topic and save to dev tables."""
        print(f"\n🧪 Testing question generation across {len(self.MAIN_TOPICS)} main topics")
        print(
            f"Using tables: {settings.table_prefix}generated_questions, {settings.table_prefix}generation_sessions"
        )

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
                "test_type": "dev_table_integration",
                "topics": self.MAIN_TOPICS,
                "include_diagrams": False,
            },
        )

        # Save session
        session_id = repositories["session"].save_session(session)
        assert session_id
        print(f"✅ Created session: {session_id}")

        generated_questions = []
        results = []

        # Generate one question per topic
        for i, topic in enumerate(self.MAIN_TOPICS, 1):
            params = self._generate_random_params()
            params["topic"] = topic

            print(f"\n[{i}/{len(self.MAIN_TOPICS)}] Topic: {topic}")
            print(
                f"  Params: {params['tier'].value}, Grade {params['grade_level']}, {params['marks']} marks"
            )

            try:
                result = await agent.process(params)

                if result.success and result.output and "question" in result.output:
                    question = result.output["question"]

                    # Save to dev tables
                    question_id = repositories["question"].save_question(question)
                    generated_questions.append((question_id, topic))
                    results.append((topic, True))

                    print(f"  ✅ Generated and saved: {question_id}")
                else:
                    print(f"  ❌ Failed: {result.error}")
                    results.append((topic, False))

            except Exception as e:
                print(f"  ❌ Exception: {e!s}")
                results.append((topic, False))

        # Statistics
        successful = sum(1 for _, success in results if success)
        print(
            f"\n📊 Results: {successful}/{len(self.MAIN_TOPICS)} questions generated successfully"
        )

        # Verify questions were saved to dev tables
        assert len(generated_questions) > 0, "No questions were successfully generated"
        print(f"💾 Saved {len(generated_questions)} questions to dev tables")

        # Update session status
        repositories["session"].update_session_status(
            str(session.session_id), GenerationStatus.CANDIDATE
        )

        # Store question IDs for cleanup
        self._cleanup_questions = generated_questions
        self._cleanup_session_id = session_id

    def teardown_method(self):
        """Clean up generated questions after test."""
        if hasattr(self, "_cleanup_questions") and hasattr(self, "_cleanup_session_id"):
            print(f"\n🧹 Cleaning up {len(self._cleanup_questions)} generated questions...")

            # Note: Cleanup would require additional repository methods
            # For now, questions remain in dev tables for manual inspection
            print("Note: Questions retained in dev tables for inspection")
            print(f"Session ID: {getattr(self, '_cleanup_session_id', 'N/A')}")


if __name__ == "__main__":
    """Run test manually for development."""
    pytest.main([__file__, "-v", "-s"])
