"""
Neon Database client for question generation system.
Handles database operations for candidate questions, past papers, and syllabus data.
"""

import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncpg
from datetime import datetime

from ..models.question_models import CandidateQuestion, GenerationStatus


class NeonDBClient:
    """Client for interacting with Neon PostgreSQL database"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("NEON_DATABASE_URL")
        if not self.database_url:
            raise ValueError("NEON_DATABASE_URL must be provided")
        self.pool = None

    async def connect(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(self.database_url)

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()

    async def create_candidate_questions_table(self):
        """Create the candidate questions table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS candidate_questions (
            id SERIAL PRIMARY KEY,
            generation_id UUID UNIQUE NOT NULL,
            question_id_local VARCHAR(50) NOT NULL,
            question_id_global VARCHAR(100) UNIQUE NOT NULL,
            question_number_display VARCHAR(20),
            marks INTEGER NOT NULL,
            command_word VARCHAR(50) NOT NULL,
            raw_text_content TEXT NOT NULL,
            formatted_text_latex TEXT,

            -- Taxonomy (stored as JSONB)
            taxonomy JSONB NOT NULL,

            -- Solution and marking scheme (stored as JSONB)
            solution_and_marking_scheme JSONB NOT NULL,

            -- Solver algorithm (stored as JSONB)
            solver_algorithm JSONB NOT NULL,

            -- Generation metadata
            seed_question_id VARCHAR(100),
            target_grade_input INTEGER NOT NULL CHECK (target_grade_input >= 1 AND target_grade_input <= 9),

            -- Model tracking
            llm_model_used_generation VARCHAR(100) NOT NULL,
            llm_model_used_marking_scheme VARCHAR(100) NOT NULL,
            llm_model_used_review VARCHAR(100),

            -- Prompt tracking
            prompt_template_version_generation VARCHAR(50) NOT NULL,
            prompt_template_version_marking_scheme VARCHAR(50) NOT NULL,
            prompt_template_version_review VARCHAR(50),

            -- Status and review
            generation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            status VARCHAR(50) DEFAULT 'candidate' CHECK (status IN (
                'candidate',
                'human_reviewed_accepted',
                'human_reviewed_rejected',
                'llm_reviewed_needs_human',
                'auto_rejected'
            )),
            reviewer_notes TEXT,

            -- Quality metrics
            confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
            validation_errors JSONB DEFAULT '[]'::jsonb,

            -- Indexing
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_candidate_questions_generation_id ON candidate_questions(generation_id);
        CREATE INDEX IF NOT EXISTS idx_candidate_questions_status ON candidate_questions(status);
        CREATE INDEX IF NOT EXISTS idx_candidate_questions_target_grade ON candidate_questions(target_grade_input);
        CREATE INDEX IF NOT EXISTS idx_candidate_questions_created_at ON candidate_questions(created_at);
        CREATE INDEX IF NOT EXISTS idx_candidate_questions_seed_id ON candidate_questions(seed_question_id);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)

    async def save_candidate_question(self, question: CandidateQuestion) -> bool:
        """Save a candidate question to the database"""
        insert_sql = """
        INSERT INTO candidate_questions (
            generation_id, question_id_local, question_id_global, question_number_display,
            marks, command_word, raw_text_content, formatted_text_latex,
            taxonomy, solution_and_marking_scheme, solver_algorithm,
            seed_question_id, target_grade_input,
            llm_model_used_generation, llm_model_used_marking_scheme, llm_model_used_review,
            prompt_template_version_generation, prompt_template_version_marking_scheme,
            prompt_template_version_review,
            generation_timestamp, status, reviewer_notes, confidence_score, validation_errors
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
        )
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    question.generation_id,
                    question.question_id_local,
                    question.question_id_global,
                    question.question_number_display,
                    question.marks,
                    question.command_word.value,
                    question.raw_text_content,
                    question.formatted_text_latex,
                    question.taxonomy.model_dump(),
                    question.solution_and_marking_scheme.model_dump(),
                    question.solver_algorithm.model_dump(),
                    question.seed_question_id,
                    question.target_grade_input,
                    question.llm_model_used_generation,
                    question.llm_model_used_marking_scheme,
                    question.llm_model_used_review,
                    question.prompt_template_version_generation,
                    question.prompt_template_version_marking_scheme,
                    question.prompt_template_version_review,
                    question.generation_timestamp,
                    question.status.value,
                    question.reviewer_notes,
                    question.confidence_score,
                    question.validation_errors
                )
            return True
        except Exception as e:
            print(f"Error saving candidate question: {e}")
            return False

    async def get_past_paper_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a past paper question by ID for use as seed"""
        # This assumes you have a past_papers table with the 2015p1 data
        query_sql = """
        SELECT * FROM past_paper_questions
        WHERE question_id_global = $1 OR question_id_local = $1
        LIMIT 1
        """

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query_sql, question_id)
                if row:
                    return dict(row)
                return None
        except Exception as e:
            print(f"Error fetching past paper question: {e}")
            return None

    async def get_syllabus_content(self, content_refs: List[str]) -> List[Dict[str, Any]]:
        """Fetch syllabus content for given references"""
        # Load from local file for now (could be moved to DB later)
        try:
            with open("data/syllabus_command.json", "r") as f:
                syllabus_data = json.load(f)

            results = []
            for ref in content_refs:
                # Search through core and extended content
                for topic in syllabus_data.get("core_subject_content", []):
                    for sub_topic in topic.get("sub_topics", []):
                        if sub_topic.get("subject_content_ref") == ref:
                            results.append({
                                "ref": ref,
                                "topic": topic.get("topic_name"),
                                "title": sub_topic.get("title"),
                                "details": sub_topic.get("details", []),
                                "notes_and_examples": sub_topic.get("notes_and_examples", [])
                            })
                            break

            return results
        except Exception as e:
            print(f"Error fetching syllabus content: {e}")
            return []

    async def get_command_word_definition(self, command_word: str) -> Optional[str]:
        """Get definition for a command word"""
        # Load from local file for now
        try:
            with open("data/syllabus_command.json", "r") as f:
                syllabus_data = json.load(f)

            command_words = syllabus_data.get("command_words", [])
            for cw in command_words:
                if cw.get("command_word", "").lower() == command_word.lower():
                    return cw.get("definition", "")

            return None
        except Exception as e:
            print(f"Error fetching command word definition: {e}")
            return None

    async def update_question_status(self, generation_id: UUID, status: GenerationStatus, reviewer_notes: str = None) -> bool:
        """Update the status and reviewer notes for a candidate question"""
        update_sql = """
        UPDATE candidate_questions
        SET status = $1, reviewer_notes = $2, updated_at = NOW()
        WHERE generation_id = $3
        """

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(update_sql, status.value, reviewer_notes, generation_id)
                return "UPDATE 1" in result
        except Exception as e:
            print(f"Error updating question status: {e}")
            return False

    async def get_candidate_questions(
        self,
        status: Optional[GenerationStatus] = None,
        target_grade: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch candidate questions with optional filters"""
        conditions = []
        params = []
        param_count = 0

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status.value)

        if target_grade:
            param_count += 1
            conditions.append(f"target_grade_input = ${param_count}")
            params.append(target_grade)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        param_count += 1
        query_sql = f"""
        SELECT * FROM candidate_questions
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_count}
        """
        params.append(limit)

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching candidate questions: {e}")
            return []

    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated questions"""
        stats_sql = """
        SELECT
            COUNT(*) as total_questions,
            COUNT(*) FILTER (WHERE status = 'candidate') as pending_review,
            COUNT(*) FILTER (WHERE status = 'human_reviewed_accepted') as accepted,
            COUNT(*) FILTER (WHERE status = 'human_reviewed_rejected') as rejected,
            COUNT(*) FILTER (WHERE status = 'auto_rejected') as auto_rejected,
            AVG(confidence_score) as avg_confidence,
            target_grade_input,
            COUNT(*) as count_per_grade
        FROM candidate_questions
        GROUP BY target_grade_input
        ORDER BY target_grade_input
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(stats_sql)
                return {
                    "grade_distribution": [dict(row) for row in rows],
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            print(f"Error fetching generation stats: {e}")
            return {}
