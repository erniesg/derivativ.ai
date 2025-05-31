"""
Database Manager for Multi-Agent Orchestrator.
Implements the complete database schema and operations for session tracking.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncpg
from dataclasses import asdict

from .orchestrator import GenerationSession, LLMInteraction, InsertionStatus
from ..models import CandidateQuestion
from ..models.database_schema import TableNames, DatabaseSchemas, DatabaseIndexes
from ..agents import ReviewFeedback
from ..validation import validate_question, ValidationSeverity


class DatabaseManager:
    """Manages database operations for orchestrator data"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        await self.create_tables()

    async def create_tables(self):
        """Create all required tables if they don't exist"""

        async with self.pool.acquire() as conn:
            # Use centralized schema definitions
            schemas = DatabaseSchemas.get_all_schemas()

            # Create tables in dependency order
            for table_name, schema in schemas.items():
                await conn.execute(schema)
                print(f"✅ Created/verified table: {table_name}")

            # Create all indexes
            indexes = DatabaseIndexes.get_all_indexes()
            for index_sql in indexes:
                await conn.execute(index_sql)

            print(f"✅ Created/verified {len(indexes)} indexes")

    async def save_session(self, session: GenerationSession) -> bool:
        """Save complete generation session to database"""

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Save session metadata
                    await conn.execute(f"""
                        INSERT INTO {TableNames.GENERATION_SESSIONS} (
                            session_id, config_id, timestamp, status,
                            total_questions_requested, questions_generated,
                            questions_approved, error_count, summary_metrics
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (session_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            questions_generated = EXCLUDED.questions_generated,
                            questions_approved = EXCLUDED.questions_approved,
                            error_count = EXCLUDED.error_count,
                            summary_metrics = EXCLUDED.summary_metrics
                    """,
                        session.session_id,
                        session.config_id,
                        session.timestamp,
                        session.status,
                        session.total_questions_requested,
                        session.questions_generated,
                        session.questions_approved,
                        session.error_count,
                        json.dumps({})  # TODO: Add summary metrics
                    )

                    # 2. Save all LLM interactions
                    for interaction in session.llm_interactions:
                        await self.save_interaction(conn, interaction, session.session_id)

                    # 3. Save questions with lineage tracking
                    for i, question in enumerate(session.candidate_questions):
                        # Map interactions to questions (assuming 3 interactions per question)
                        gen_interaction = session.llm_interactions[i*3] if len(session.llm_interactions) > i*3 else None
                        mark_interaction = session.llm_interactions[i*3+1] if len(session.llm_interactions) > i*3+1 else None
                        review_interaction = session.llm_interactions[i*3+2] if len(session.llm_interactions) > i*3+2 else None

                        await self.save_question(
                            conn, question, session.session_id,
                            gen_interaction.interaction_id if gen_interaction else None,
                            mark_interaction.interaction_id if mark_interaction else None,
                            review_interaction.interaction_id if review_interaction else None
                        )

                    # 4. Save review results and errors
                    for feedback in session.review_feedbacks:
                        # Find corresponding question
                        for question in session.candidate_questions:
                            await self.save_review_result(conn, feedback, question.question_id_global)

                    for error in session.errors:
                        await self.save_error(conn, error, session.session_id)

                    return True

        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    async def save_interaction(self, conn, interaction: LLMInteraction, session_id: str):
        """Save LLM interaction"""
        await conn.execute(f"""
            INSERT INTO {TableNames.LLM_INTERACTIONS} (
                interaction_id, session_id, agent_type, model_used,
                prompt_text, raw_response, parsed_response,
                processing_time_ms, success, error_message, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
            interaction.interaction_id,
            session_id,
            interaction.agent_type,
            interaction.model_used,
            interaction.prompt_text,
            interaction.raw_response,
            json.dumps(interaction.parsed_response) if interaction.parsed_response else None,
            interaction.processing_time_ms,
            interaction.success,
            interaction.error_message,
            interaction.timestamp
        )

    async def save_question(
        self,
        conn,
        question: CandidateQuestion,
        session_id: str,
        gen_interaction_id: str = None,
        mark_interaction_id: str = None,
        review_interaction_id: str = None
    ):
        """Save candidate question with lineage tracking"""

        # Extract metadata for indexing
        subject_content_refs = question.taxonomy.subject_content_references if question.taxonomy else []
        topic_path = question.taxonomy.topic_path if question.taxonomy else []

        await conn.execute(f"""
            INSERT INTO {TableNames.CANDIDATE_QUESTIONS} (
                question_id, session_id, generation_interaction_id,
                marking_interaction_id, review_interaction_id,
                question_data, subject_content_refs, topic_path,
                command_word, target_grade, marks, calculator_policy
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
            str(question.question_id_global),
            session_id,
            gen_interaction_id,
            mark_interaction_id,
            review_interaction_id,
            question.model_dump_json(),
            subject_content_refs,
            topic_path,
            question.command_word.value,
            question.target_grade_input,
            question.marks,
            question.calculator_policy.value if question.calculator_policy else 'not_allowed'
        )

    async def save_review_result(self, conn, feedback: ReviewFeedback, question_id: str):
        """Save review result"""
        await conn.execute(f"""
            INSERT INTO {TableNames.REVIEW_RESULTS} (
                review_id, question_id, outcome, overall_score,
                syllabus_compliance, difficulty_alignment, marking_quality,
                feedback_summary, specific_feedback, suggested_improvements
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
            str(uuid.uuid4()),
            question_id,
            feedback.outcome.value,
            feedback.overall_score,
            feedback.syllabus_compliance,
            feedback.difficulty_alignment,
            feedback.marking_quality,
            feedback.feedback_summary,
            json.dumps(asdict(feedback.specific_feedback)) if feedback.specific_feedback else None,
            json.dumps(feedback.suggested_improvements) if feedback.suggested_improvements else None
        )

    async def save_error(self, conn, error: Dict[str, Any], session_id: str):
        """Save error log"""
        await conn.execute(f"""
            INSERT INTO {TableNames.ERROR_LOGS} (
                error_id, session_id, error_type, error_severity,
                error_message, stack_trace, context_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            str(uuid.uuid4()),
            session_id,
            error.get('type', 'unknown'),
            error.get('severity', 'medium'),
            str(error.get('message', '')),
            error.get('stack_trace'),
            json.dumps(error.get('context', {}))
        )

    async def get_questions_for_manual_review(self) -> List[Dict]:
        """Get questions that need manual review"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT q.*, r.overall_score, r.outcome
                FROM {TableNames.CANDIDATE_QUESTIONS} q
                JOIN {TableNames.REVIEW_RESULTS} r ON q.question_id = r.question_id
                WHERE q.insertion_status = 'manual_review'
                ORDER BY r.overall_score DESC
            """)
            return [dict(row) for row in rows]

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()

    async def get_candidate_question(self, question_id: str) -> Optional[CandidateQuestion]:
        """Retrieve a specific candidate question"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT question_data FROM {TableNames.CANDIDATE_QUESTIONS}
                WHERE question_id = $1
            """, question_id)

            if row:
                question_data = row['question_data']
                return CandidateQuestion.model_validate(question_data)
            return None

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        async with self.pool.acquire() as conn:
            # Get session data
            session_row = await conn.fetchrow(f"""
                SELECT * FROM {TableNames.GENERATION_SESSIONS}
                WHERE session_id = $1
            """, session_id)

            if not session_row:
                return {}

            # Get interaction counts
            interaction_counts = await conn.fetchrow(f"""
                SELECT
                    COUNT(*) as total_interactions,
                    COUNT(*) FILTER (WHERE success = true) as successful_interactions,
                    COUNT(*) FILTER (WHERE agent_type = 'generator') as generation_interactions,
                    COUNT(*) FILTER (WHERE agent_type = 'marker') as marking_interactions,
                    COUNT(*) FILTER (WHERE agent_type = 'reviewer') as review_interactions
                FROM {TableNames.LLM_INTERACTIONS}
                WHERE session_id = $1
            """, session_id)

            # Get question counts
            question_counts = await conn.fetchrow(f"""
                SELECT
                    COUNT(*) as total_questions,
                    COUNT(*) FILTER (WHERE insertion_status = 'auto_approved') as auto_approved,
                    COUNT(*) FILTER (WHERE insertion_status = 'manual_review') as manual_review,
                    COUNT(*) FILTER (WHERE insertion_status = 'auto_rejected') as auto_rejected
                FROM {TableNames.CANDIDATE_QUESTIONS}
                WHERE session_id = $1
            """, session_id)

            return {
                "session": dict(session_row),
                "interactions": dict(interaction_counts) if interaction_counts else {},
                "questions": dict(question_counts) if question_counts else {},
            }
