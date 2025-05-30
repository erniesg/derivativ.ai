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
from ..agents import ReviewFeedback


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

        # TODO: Implement table creation
        # This would create:
        # - generation_sessions
        # - llm_interactions
        # - candidate_questions_extended
        # - review_results
        # - error_logs
        # - manual_review_queue

        async with self.pool.acquire() as conn:
            # Generation Sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_sessions (
                    session_id UUID PRIMARY KEY,
                    config_id VARCHAR NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    status VARCHAR CHECK (status IN ('running', 'completed', 'failed')),
                    total_questions_requested INT NOT NULL,
                    questions_generated INT DEFAULT 0,
                    questions_approved INT DEFAULT 0,
                    error_count INT DEFAULT 0,
                    summary_metrics JSONB
                )
            """)

            # LLM Interactions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_interactions (
                    interaction_id UUID PRIMARY KEY,
                    session_id UUID REFERENCES generation_sessions,
                    agent_type VARCHAR NOT NULL,
                    model_used VARCHAR NOT NULL,
                    prompt_text TEXT NOT NULL,
                    raw_response TEXT,
                    parsed_response JSONB,
                    processing_time_ms INT,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    temperature FLOAT,
                    max_tokens INT,
                    token_usage JSONB
                )
            """)

            # Extended Candidate Questions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS candidate_questions_extended (
                    question_id UUID PRIMARY KEY,
                    session_id UUID REFERENCES generation_sessions,
                    generation_interaction_id UUID REFERENCES llm_interactions,
                    marking_interaction_id UUID REFERENCES llm_interactions,
                    review_interaction_id UUID REFERENCES llm_interactions,

                    -- Question data (would include all CandidateQuestion fields)
                    question_data JSONB NOT NULL,

                    -- Quality control
                    insertion_status VARCHAR CHECK (insertion_status IN (
                        'pending', 'auto_approved', 'manual_review',
                        'auto_rejected', 'manually_approved', 'manually_rejected'
                    )),
                    insertion_timestamp TIMESTAMP,
                    approved_by VARCHAR,
                    rejection_reason TEXT,

                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    version INT DEFAULT 1
                )
            """)

            # Review Results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS review_results (
                    review_id UUID PRIMARY KEY,
                    question_id UUID REFERENCES candidate_questions_extended,
                    interaction_id UUID REFERENCES llm_interactions,

                    outcome VARCHAR NOT NULL,
                    overall_score DECIMAL(3,2) CHECK (overall_score BETWEEN 0 AND 1),
                    syllabus_compliance DECIMAL(3,2) CHECK (syllabus_compliance BETWEEN 0 AND 1),
                    difficulty_alignment DECIMAL(3,2) CHECK (difficulty_alignment BETWEEN 0 AND 1),
                    marking_quality DECIMAL(3,2) CHECK (marking_quality BETWEEN 0 AND 1),

                    feedback_summary TEXT,
                    specific_feedback JSONB,
                    suggested_improvements JSONB,

                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)

            # Error Logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    error_id UUID PRIMARY KEY,
                    session_id UUID REFERENCES generation_sessions,
                    interaction_id UUID REFERENCES llm_interactions,

                    error_type VARCHAR NOT NULL,
                    error_severity VARCHAR CHECK (error_severity IN ('low', 'medium', 'high', 'critical')),
                    error_message TEXT NOT NULL,
                    stack_trace TEXT,
                    context_data JSONB,

                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_notes TEXT,
                    resolved_by VARCHAR,
                    resolved_at TIMESTAMP,

                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)

            # Manual Review Queue table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_review_queue (
                    queue_id UUID PRIMARY KEY,
                    question_id UUID REFERENCES candidate_questions_extended,
                    review_id UUID REFERENCES review_results,

                    priority INT DEFAULT 1,
                    assigned_to VARCHAR,
                    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'in_review', 'completed')),

                    review_started_at TIMESTAMP,
                    review_completed_at TIMESTAMP,
                    reviewer_notes TEXT,
                    final_decision VARCHAR,

                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def save_session(self, session: GenerationSession) -> bool:
        """Save complete generation session to database"""

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Save session metadata
                    await conn.execute("""
                        INSERT INTO generation_sessions (
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

                    # 3. Save all questions with lineage
                    for i, question in enumerate(session.questions):
                        # Find corresponding interactions
                        gen_interaction = session.llm_interactions[i*3] if len(session.llm_interactions) > i*3 else None
                        mark_interaction = session.llm_interactions[i*3+1] if len(session.llm_interactions) > i*3+1 else None
                        review_interaction = session.llm_interactions[i*3+2] if len(session.llm_interactions) > i*3+2 else None

                        await self.save_question(
                            conn, question, session.session_id,
                            gen_interaction.interaction_id if gen_interaction else None,
                            mark_interaction.interaction_id if mark_interaction else None,
                            review_interaction.interaction_id if review_interaction else None
                        )

                    # 4. Save review results
                    for i, feedback in enumerate(session.review_feedbacks):
                        if i < len(session.questions):
                            await self.save_review_result(conn, feedback, session.questions[i].question_id_local)

                    # 5. Save errors
                    for error in session.errors:
                        await self.save_error(conn, error, session.session_id)

            return True

        except Exception as e:
            print(f"❌ Error saving session to database: {e}")
            return False

    async def save_interaction(self, conn, interaction: LLMInteraction, session_id: str):
        """Save LLM interaction"""
        await conn.execute("""
            INSERT INTO llm_interactions (
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
        """Save question with lineage"""
        await conn.execute("""
            INSERT INTO candidate_questions_extended (
                question_id, session_id, generation_interaction_id,
                marking_interaction_id, review_interaction_id,
                question_data, insertion_status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            question.question_id_local,
            session_id,
            gen_interaction_id,
            mark_interaction_id,
            review_interaction_id,
            json.dumps(question.model_dump(mode='json')),
            'pending'  # Default status
        )

    async def save_review_result(self, conn, feedback: ReviewFeedback, question_id: str):
        """Save review feedback"""
        review_id = str(uuid.uuid4())
        await conn.execute("""
            INSERT INTO review_results (
                review_id, question_id, outcome, overall_score,
                syllabus_compliance, difficulty_alignment, marking_quality,
                feedback_summary, specific_feedback, suggested_improvements
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
            review_id,
            question_id,
            feedback.outcome.value,
            feedback.overall_score,
            feedback.syllabus_compliance,
            feedback.difficulty_alignment,
            feedback.marking_quality,
            feedback.feedback_summary,
            json.dumps(feedback.specific_feedback),
            json.dumps(feedback.suggested_improvements)
        )

    async def save_error(self, conn, error: Dict[str, Any], session_id: str):
        """Save error log"""
        error_id = str(uuid.uuid4())
        await conn.execute("""
            INSERT INTO error_logs (
                error_id, session_id, error_type, error_severity,
                error_message, context_data
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
            error_id,
            session_id,
            error.get('step', 'unknown'),
            'medium',  # Default severity
            error.get('error', ''),
            json.dumps(error)
        )

    async def get_questions_for_manual_review(self) -> List[Dict]:
        """Get questions that need manual review"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT q.*, r.overall_score, r.feedback_summary, r.suggested_improvements
                FROM candidate_questions_extended q
                JOIN review_results r ON q.question_id = r.question_id
                WHERE q.insertion_status = 'manual_review'
                ORDER BY r.overall_score DESC
            """)
            return [dict(row) for row in rows]

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
