"""
Supabase repository implementation for question and session management.
Provides hybrid storage: flattened fields for querying + JSON for full fidelity.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from src.core.config import get_settings
from src.models.enums import CommandWord, QuestionOrigin, Tier
from src.models.question_models import GenerationSession, GenerationStatus, Question
from supabase import Client, create_client

logger = logging.getLogger(__name__)


def get_supabase_client(url: str, key: str) -> Client:
    """
    Create and configure Supabase client.

    Args:
        url: Supabase project URL
        key: Supabase API key

    Returns:
        Configured Supabase client
    """
    return create_client(url, key)


class QuestionRepository:
    """Repository for managing questions in Supabase."""

    def __init__(self, supabase_client: Client):
        """Initialize with Supabase client."""
        self.supabase = supabase_client
        settings = get_settings()
        self.table_name = f"{settings.table_prefix}generated_questions"

    def save_question(
        self, question: Question, origin: QuestionOrigin = QuestionOrigin.GENERATED
    ) -> str:
        """
        Save question to Supabase with hybrid storage approach.

        Args:
            question: Question object to save
            origin: Source of the question

        Returns:
            Database ID of saved question
        """
        try:
            # Prepare data with flattened fields for querying + full JSON
            data = {
                "question_id_global": question.question_id_global,
                "question_id_local": question.question_id_local,
                "question_number_display": question.question_number_display,
                "marks": question.marks,
                "tier": getattr(question.taxonomy, "tier", Tier.CORE).value,
                "command_word": question.command_word.value,
                "raw_text_content": question.raw_text_content,
                "quality_score": None,  # Will be set by review agent
                "origin": origin.value,
                "source_reference": None,  # For past papers/textbooks
                "content_json": question.model_dump(),
            }

            # Insert into Supabase
            response = self.supabase.table(self.table_name).insert(data).execute()

            if response.data and len(response.data) > 0:
                return response.data[0]["id"]
            else:
                raise Exception("No data returned from insert operation")

        except Exception as e:
            logger.error(f"Failed to save question {question.question_id_global}: {e}")
            raise

    def get_question(self, question_id_global: str) -> Optional[Question]:
        """
        Retrieve question by global ID.

        Args:
            question_id_global: Global question identifier

        Returns:
            Question object if found, None otherwise
        """
        try:
            response = (
                self.supabase.table(self.table_name)
                .select("*")
                .eq("question_id_global", question_id_global)
                .execute()
            )

            if response.data and len(response.data) > 0:
                question_data = response.data[0]["content_json"]
                return Question.model_validate(question_data)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get question {question_id_global}: {e}")
            raise

    def search_questions_by_content(
        self,
        subject_content_refs: list[str],
        tier: Optional[Tier] = None,
        min_quality_score: Optional[float] = None,
        limit: int = 20,
        topic_keywords: list[str] = None,
    ) -> list[Question]:
        """
        Search questions by subject content references and topic keywords.

        Args:
            subject_content_refs: List of syllabus references to match (e.g., ['C2.1', 'C2.2'])
            tier: Filter by tier
            min_quality_score: Minimum quality score
            limit: Maximum number of results
            topic_keywords: Additional topic keywords to match against skill_tags/topic_path

        Returns:
            List of Question objects matching the criteria
        """
        try:
            questions = []

            # Build query with basic filters
            query = self.supabase.table(self.table_name).select("*")

            if tier:
                query = query.eq("tier", tier.value)
            if min_quality_score:
                query = query.gte("quality_score", min_quality_score)

            # Get more records to filter by content
            query = query.order("created_at", desc=True).limit(limit * 3)

            response = query.execute()

            if response.data:
                for row in response.data:
                    try:
                        # First check if this question matches our content criteria
                        question_data = row["content_json"]
                        if self._matches_content_criteria(
                            question_data, subject_content_refs, topic_keywords
                        ):
                            # Only try to parse if it matches our criteria
                            try:
                                question = Question.model_validate(question_data)
                                questions.append(question)

                                # Stop when we have enough matches
                                if len(questions) >= limit:
                                    break
                            except Exception as parse_error:
                                logger.warning(
                                    f"Failed to parse matching question from DB: {parse_error}"
                                )
                                # For now, skip questions that don't parse properly
                                continue

                    except Exception as e:
                        logger.warning(f"Error processing question from DB: {e}")

            logger.info(
                f"Found {len(questions)} questions matching content criteria from {len(response.data or [])} total questions"
            )
            return questions

        except Exception as e:
            logger.error(f"Failed to search questions by content: {e}")
            return []  # Return empty list on error

    def _matches_content_criteria(
        self,
        question_data: dict,
        subject_content_refs: list[str],
        topic_keywords: list[str] = None,
    ) -> bool:
        """
        Check if a question matches the content criteria using its rich JSON structure.

        Args:
            question_data: The content_json from database
            subject_content_refs: Required syllabus references
            topic_keywords: Additional topic keywords to match

        Returns:
            True if question matches criteria
        """
        try:
            # Extract taxonomy data
            taxonomy = question_data.get("taxonomy", {})

            # 1. Check subject_content_references match
            question_refs = taxonomy.get("subject_content_references", [])
            if subject_content_refs:
                # Check if any of the required refs are in the question's refs
                if any(ref in question_refs for ref in subject_content_refs):
                    logger.debug(
                        f"Question matches syllabus refs: {question_refs} ∩ {subject_content_refs}"
                    )
                else:
                    return False

            # 2. Check topic keywords against skill_tags and topic_path
            if topic_keywords:
                skill_tags = taxonomy.get("skill_tags", [])
                topic_path = taxonomy.get("topic_path", [])

                # Convert to lowercase for matching
                all_tags = [tag.lower() for tag in skill_tags + topic_path]
                keywords_lower = [kw.lower() for kw in topic_keywords]

                # Check if any keyword matches
                if not any(kw in all_tags for kw in keywords_lower):
                    return False

            return True

        except Exception as e:
            logger.warning(f"Error checking content criteria: {e}")
            return False

    def list_questions(
        self,
        tier: Optional[Tier] = None,
        min_quality_score: Optional[float] = None,
        command_word: Optional[CommandWord] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List questions with filtering.

        Args:
            tier: Filter by tier
            min_quality_score: Minimum quality score
            command_word: Filter by command word
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of question metadata dictionaries
        """
        try:
            query = self.supabase.table(self.table_name).select(
                "id, question_id_global, tier, marks, command_word, quality_score, created_at"
            )

            # Apply filters
            if tier:
                query = query.eq("tier", tier.value)
            if min_quality_score:
                query = query.gte("quality_score", min_quality_score)
            if command_word:
                query = query.eq("command_word", command_word.value)

            # Apply ordering and pagination
            query = query.order("created_at", desc=True).limit(limit)
            if offset > 0:
                query = query.offset(offset)

            response = query.execute()
            return response.data or []

        except Exception as e:
            logger.error(f"Failed to list questions: {e}")
            raise

    def delete_question(self, question_id_global: str) -> bool:
        """
        Delete question by global ID.

        Args:
            question_id_global: Global question identifier

        Returns:
            True if deleted successfully
        """
        try:
            response = (
                self.supabase.table(self.table_name)
                .delete()
                .eq("question_id_global", question_id_global)
                .execute()
            )

            return response.data and len(response.data) > 0

        except Exception as e:
            logger.error(f"Failed to delete question {question_id_global}: {e}")
            raise

    def update_quality_score(self, question_id_global: str, quality_score: float) -> bool:
        """
        Update quality score for a question.

        Args:
            question_id_global: Global question identifier
            quality_score: New quality score

        Returns:
            True if updated successfully
        """
        try:
            response = (
                self.supabase.table(self.table_name)
                .update(
                    {"quality_score": quality_score, "updated_at": datetime.utcnow().isoformat()}
                )
                .eq("question_id_global", question_id_global)
                .execute()
            )

            return response.data and len(response.data) > 0

        except Exception as e:
            logger.error(f"Failed to update quality score for {question_id_global}: {e}")
            raise


class GenerationSessionRepository:
    """Repository for managing generation sessions in Supabase."""

    def __init__(self, supabase_client: Client):
        """Initialize with Supabase client."""
        self.supabase = supabase_client
        settings = get_settings()
        self.table_name = f"{settings.table_prefix}generation_sessions"

    def save_session(self, session: GenerationSession) -> str:
        """
        Save generation session to Supabase.

        Args:
            session: GenerationSession object to save

        Returns:
            Database ID of saved session
        """
        try:
            # Prepare session data according to production table structure
            data = {
                "session_id": str(session.session_id),
                "topic": session.request.topic,
                "tier": session.request.tier.value,
                "grade_level": getattr(session.request, "grade_level", 8),
                "marks": session.request.marks,
                "count": getattr(session.request, "count", 1),
                "calculator_policy": getattr(session.request, "calculator_policy", "allowed").value
                if hasattr(getattr(session.request, "calculator_policy", None), "value")
                else "allowed",
                "command_word": session.request.command_word.value if session.request.command_word else "Calculate",
                "status": session.status.value,
                "total_processing_time": session.total_processing_time,
                "questions_generated": len(session.questions),
                "request_json": session.request.model_dump(mode="json"),
                "questions_json": [q.model_dump(mode="json") for q in session.questions],
                "quality_decisions_json": [
                    qd.model_dump(mode="json") for qd in session.quality_decisions
                ],
                "agent_results_json": [ar.model_dump(mode="json") for ar in session.agent_results],
                "created_at": session.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Insert into Supabase
            response = self.supabase.table(self.table_name).insert(data).execute()

            if response.data and len(response.data) > 0:
                return response.data[0]["id"]
            else:
                raise Exception("No data returned from session insert")

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            raise

    def get_session(self, session_id: str) -> Optional[GenerationSession]:
        """
        Retrieve generation session by ID.

        Args:
            session_id: Session identifier

        Returns:
            GenerationSession object if found, None otherwise
        """
        try:
            response = (
                self.supabase.table(self.table_name)
                .select("*")
                .eq("session_id", session_id)
                .execute()
            )

            if response.data and len(response.data) > 0:
                # Reconstruct session from production table structure
                row = response.data[0]

                # Build the full session from separate JSON columns
                session_data = {
                    "session_id": row["session_id"],
                    "request": row["request_json"],
                    "questions": row.get("questions_json", []),
                    "quality_decisions": row.get("quality_decisions_json", []),
                    "agent_results": row.get("agent_results_json", []),
                    "status": row.get("status", "pending"),
                    "total_processing_time": row.get("total_processing_time", 0.0),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }

                return GenerationSession.model_validate(session_data)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            raise

    def update_session_status(self, session_id: str, status: GenerationStatus) -> bool:
        """
        Update session status.

        Args:
            session_id: Session identifier
            status: New status

        Returns:
            True if updated successfully
        """
        try:
            response = (
                self.supabase.table(self.table_name)
                .update({"status": status.value, "updated_at": datetime.utcnow().isoformat()})
                .eq("session_id", session_id)
                .execute()
            )

            return response.data and len(response.data) > 0

        except Exception as e:
            logger.error(f"Failed to update session status for {session_id}: {e}")
            raise

    def list_sessions(
        self, status: Optional[GenerationStatus] = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        List generation sessions with filtering.

        Args:
            status: Filter by status
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of session metadata dictionaries
        """
        try:
            query = self.supabase.table(self.table_name).select(
                "id, session_id, status, total_processing_time, created_at"
            )

            if status:
                query = query.eq("status", status.value)

            query = query.order("created_at", desc=True).limit(limit)
            if offset > 0:
                query = query.offset(offset)

            response = query.execute()
            return response.data or []

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            raise
