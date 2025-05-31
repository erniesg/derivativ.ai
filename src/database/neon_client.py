"""
Neon Database client for question generation system.
Handles database operations for candidate questions, past papers, and syllabus data.
Works with existing Payload CMS collections.
"""

import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncpg
from datetime import datetime
import uuid

# Import with error handling for when used as standalone
try:
    from ..models.question_models import CandidateQuestion, GenerationStatus
except ImportError:
    # Fallback for standalone usage
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.question_models import CandidateQuestion, GenerationStatus

from ..validation.question_validator import validate_question, ValidationSeverity


class NeonDBClient:
    """Client for interacting with Neon PostgreSQL database with Payload CMS collections"""

    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        self.pool = None

    async def connect(self):
        """Create connection pool"""
        if self.connection_string and self.connection_string != "postgresql://dummy:dummy@dummy/dummy":
            try:
                self.pool = await asyncpg.create_pool(self.connection_string)
                print("✅ Database connection established")
            except Exception as e:
                print(f"❌ Database connection failed: {e}")
                self.pool = None
        else:
            print("⚠️ Using dummy database URL - no actual database operations")
            self.pool = None

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()

    async def create_candidate_questions_table(self):
        """Create the deriv_candidate_questions table if it doesn't exist (updated schema)"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS deriv_candidate_questions (
            question_id UUID PRIMARY KEY,
            session_id UUID,  -- Can be null for standalone questions

            -- Lineage tracking
            generation_interaction_id UUID,
            marking_interaction_id UUID,
            review_interaction_id UUID,

            -- Question content (stored as JSONB for flexibility)
            question_data JSONB NOT NULL,

            -- Educational metadata (extracted for indexing)
            subject_content_refs TEXT[],
            topic_path TEXT[],
            command_word VARCHAR(50),
            target_grade INT CHECK (target_grade BETWEEN 1 AND 12),
            marks INT CHECK (marks > 0),
            calculator_policy VARCHAR(20) CHECK (calculator_policy IN ('allowed', 'not_allowed', 'assumed')),
            curriculum_type VARCHAR(50) DEFAULT 'cambridge_igcse',

            -- Quality control
            insertion_status VARCHAR(30) CHECK (insertion_status IN (
                'pending', 'auto_approved', 'manual_review',
                'auto_rejected', 'manually_approved', 'manually_rejected',
                'needs_revision', 'archived'
            )) DEFAULT 'pending',

            -- Validation results
            validation_passed BOOLEAN,
            validation_warnings INT DEFAULT 0,
            validation_errors JSONB,

            -- Review workflow
            review_score DECIMAL(3,2) CHECK (review_score BETWEEN 0 AND 1),
            insertion_timestamp TIMESTAMP,
            approved_by VARCHAR(100),
            rejection_reason TEXT,
            manual_notes TEXT,

            -- Version control
            version INT DEFAULT 1,
            parent_question_id UUID REFERENCES deriv_candidate_questions(question_id),

            -- Timestamps
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );

        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_deriv_candidate_questions_session_id ON deriv_candidate_questions(session_id);
        CREATE INDEX IF NOT EXISTS idx_deriv_candidate_questions_insertion_status ON deriv_candidate_questions(insertion_status);
        CREATE INDEX IF NOT EXISTS idx_deriv_candidate_questions_target_grade ON deriv_candidate_questions(target_grade);
        CREATE INDEX IF NOT EXISTS idx_deriv_candidate_questions_created_at ON deriv_candidate_questions(created_at);
        CREATE INDEX IF NOT EXISTS idx_deriv_candidate_questions_validation_passed ON deriv_candidate_questions(validation_passed);
        CREATE INDEX IF NOT EXISTS idx_deriv_candidate_questions_subject_content ON deriv_candidate_questions USING GIN(subject_content_refs);
        """

        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(create_table_sql)

    async def save_candidate_question(self, question: CandidateQuestion, session_id: str = None) -> bool:
        """Save a candidate question to the deriv_candidate_questions table with validation"""
        if not self.pool:
            print("⚠️ No database pool available - skipping save")
            return False

        # Validate question before saving
        validation_result = validate_question(question)

        # Log validation results
        if validation_result.critical_errors_count > 0:
            print(f"❌ Question validation failed: {validation_result.critical_errors_count} critical errors")
            for issue in validation_result.issues:
                if issue.severity == ValidationSeverity.CRITICAL:
                    print(f"   • {issue.field}: {issue.message}")
            print("⚠️ Question not saved due to validation failures")
            return False

        if validation_result.warnings_count > 0:
            print(f"⚠️ Question has {validation_result.warnings_count} warnings but will be saved")
            for issue in validation_result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    print(f"   • {issue.field}: {issue.message}")

        insert_sql = """
        INSERT INTO deriv_candidate_questions (
            question_id, session_id, question_data,
            subject_content_refs, topic_path, command_word,
            target_grade, marks, calculator_policy,
            insertion_status, validation_passed, validation_warnings, validation_errors
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
        )
        """

        try:
            # Generate UUID for question_id
            question_uuid = str(uuid.uuid4())

            # Extract metadata for indexing
            subject_content_refs = question.taxonomy.subject_content_references if question.taxonomy else []
            topic_path = question.taxonomy.topic_path if question.taxonomy else []

            # Convert calculator policy enum to string
            if hasattr(question, 'calculator_policy'):
                calculator_policy = question.calculator_policy.value if question.calculator_policy else 'not_allowed'
            else:
                calculator_policy = 'not_allowed'

            # Prepare validation data for storage
            validation_errors_json = json.dumps([{
                "field": issue.field,
                "issue_type": issue.issue_type,
                "message": issue.message,
                "severity": issue.severity.value
            } for issue in validation_result.issues])

            async with self.pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    question_uuid,  # question_id
                    session_id,  # session_id (can be None)
                    question.model_dump_json(),  # question_data as JSON
                    subject_content_refs,  # subject_content_refs array
                    topic_path,  # topic_path array
                    question.command_word.value,  # command_word
                    question.target_grade_input,  # target_grade
                    question.marks,  # marks
                    calculator_policy,  # calculator_policy
                    'pending',  # insertion_status
                    validation_result.is_valid,  # validation_passed
                    validation_result.warnings_count,  # validation_warnings
                    validation_errors_json  # validation_errors as JSON
                )

            if validation_result.is_valid:
                print(f"✅ Question {question.question_id_local} saved successfully (validation passed)")
            else:
                print(f"⚠️ Question {question.question_id_local} saved with validation warnings")

            return True
        except Exception as e:
            print(f"❌ Error saving candidate question: {e}")
            return False

    async def get_past_paper_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a past paper question by ID from Payload CMS questions collection"""
        query_sql = """
        SELECT
            q.question_id_local,
            q.question_id_global,
            q.question_number_display,
            q.marks,
            q.command_word,
            q.raw_text_content,
            q.formatted_text_latex,
            q.taxonomy,
            COUNT(qa.assets_id) as asset_count
        FROM questions q
        LEFT JOIN questions_rels qa ON q.id = qa.parent_id AND qa.path = 'assets'
        WHERE q.question_id_global = $1 OR q.question_id_local = $1
        AND q.origin = 'past_year'
        GROUP BY q.id, q.question_id_local, q.question_id_global, q.question_number_display,
                 q.marks, q.command_word, q.raw_text_content, q.formatted_text_latex, q.taxonomy
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

    async def get_questions_without_assets(
        self,
        limit: int = 10,
        grade_range: Optional[tuple] = None,
        command_words: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch past paper questions that don't have assets (diagrams)"""
        conditions = ["q.origin = 'past_year'"]
        params = []
        param_count = 0

        # Filter by grade range using difficulty estimate or marks as proxy
        if grade_range:
            param_count += 1
            conditions.append(f"((q.taxonomy->>'difficulty_estimate_0_to_1')::float >= ${param_count}")
            params.append(grade_range[0] / 9.0)  # Convert grade to 0-1 scale

            param_count += 1
            conditions.append(f"(q.taxonomy->>'difficulty_estimate_0_to_1')::float <= ${param_count})")
            params.append(grade_range[1] / 9.0)

        # Filter by command words
        if command_words:
            param_count += 1
            conditions.append(f"q.command_word = ANY(${param_count})")
            params.append(command_words)

        where_clause = " AND ".join(conditions)
        param_count += 1

        query_sql = f"""
        SELECT
            q.id,
            q.question_id_local,
            q.question_id_global,
            q.question_number_display,
            q.marks,
            q.command_word,
            q.raw_text_content,
            q.formatted_text_latex,
            q.taxonomy,
            COUNT(qa.assets_id) as asset_count
        FROM questions q
        LEFT JOIN questions_rels qa ON q.id = qa.parent_id AND qa.path = 'assets'
        WHERE {where_clause}
        GROUP BY q.id, q.question_id_local, q.question_id_global, q.question_number_display,
                 q.marks, q.command_word, q.raw_text_content, q.formatted_text_latex, q.taxonomy
        HAVING COUNT(qa.assets_id) = 0
        ORDER BY RANDOM()
        LIMIT ${param_count}
        """
        params.append(limit)

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching questions without assets: {e}")
            return []

    async def get_full_question_set(self, question_id: str) -> List[Dict[str, Any]]:
        """Fetch all parts of a question (e.g., 1a, 1b, 1c) given any part"""
        # Extract the base question number from question_id
        # e.g., "0580_SP_25_P1_q1a" -> get all q1* questions

        if "_q" in question_id:
            base_pattern = question_id.split("_q")[0] + "_q" + question_id.split("_q")[1][0]  # Get just the number part
        else:
            # Fallback - just return the single question
            single_question = await self.get_past_paper_question(question_id)
            return [single_question] if single_question else []

        query_sql = """
        SELECT
            q.id,
            q.question_id_local,
            q.question_id_global,
            q.question_number_display,
            q.marks,
            q.command_word,
            q.raw_text_content,
            q.formatted_text_latex,
            q.taxonomy,
            COUNT(qa.assets_id) as asset_count
        FROM questions q
        LEFT JOIN questions_rels qa ON q.id = qa.parent_id AND qa.path = 'assets'
        WHERE q.question_id_global LIKE $1
        AND q.origin = 'past_year'
        GROUP BY q.id, q.question_id_local, q.question_id_global, q.question_number_display,
                 q.marks, q.command_word, q.raw_text_content, q.formatted_text_latex, q.taxonomy
        ORDER BY q.question_id_global
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query_sql, f"{base_pattern}%")
                questions = [dict(row) for row in rows]

                # If we found multiple parts, return all; if just one, check if it's standalone
                if len(questions) > 1:
                    return questions
                elif len(questions) == 1:
                    # Check if this is a standalone question or part of a series
                    return questions
                else:
                    # Fallback to original single question lookup
                    single_question = await self.get_past_paper_question(question_id)
                    return [single_question] if single_question else []

        except Exception as e:
            print(f"Error fetching full question set: {e}")
            return []

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

    async def update_question_status(self, question_id: str, status: str, reviewer_notes: str = None) -> bool:
        """Update the status and reviewer notes for a candidate question"""
        if not self.pool:
            return False

        update_sql = """
        UPDATE deriv_candidate_questions
        SET insertion_status = $1, manual_notes = $2, updated_at = NOW()
        WHERE question_id = $3
        """

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(update_sql, status, reviewer_notes, question_id)
                return "UPDATE 1" in result
        except Exception as e:
            print(f"Error updating question status: {e}")
            return False

    async def get_candidate_questions(
        self,
        status: Optional[str] = None,
        target_grade: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch candidate questions with optional filters"""
        if not self.pool:
            return []

        conditions = []
        params = []
        param_count = 0

        if status:
            param_count += 1
            conditions.append(f"insertion_status = ${param_count}")
            params.append(status)

        if target_grade:
            param_count += 1
            conditions.append(f"target_grade = ${param_count}")
            params.append(target_grade)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        param_count += 1
        query_sql = f"""
        SELECT question_id, question_data, insertion_status, target_grade,
               marks, command_word, validation_passed, created_at
        FROM deriv_candidate_questions
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_count}
        """
        params.append(limit)

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)
                result = []
                for row in rows:
                    # Parse the JSON data and add metadata
                    question_data = json.loads(row['question_data'])
                    question_data.update({
                        'question_id': str(row['question_id']),
                        'insertion_status': row['insertion_status'],
                        'target_grade': row['target_grade'],
                        'marks': row['marks'],
                        'command_word': row['command_word'],
                        'validation_passed': row['validation_passed'],
                        'created_at': row['created_at']
                    })
                    result.append(question_data)
                return result
        except Exception as e:
            print(f"Error fetching candidate questions: {e}")
            return []

    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated questions"""
        if not self.pool:
            return {}

        try:
            async with self.pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_questions,
                        COUNT(CASE WHEN validation_passed = true THEN 1 END) as validated_questions,
                        COUNT(CASE WHEN insertion_status = 'auto_approved' THEN 1 END) as approved_questions,
                        COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as questions_today,
                        COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as questions_this_week
                    FROM deriv_candidate_questions
                """)
                return dict(stats)
        except Exception as e:
            print(f"Error fetching generation stats: {e}")
            return {}

    def _safe_json_load(self, file_path: str) -> Dict[str, Any]:
        """Safely load JSON file with potential LaTeX escape sequences"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # First try normal JSON loading
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                # If that fails, try to fix common LaTeX escape issues
                import re

                print(f"[DEBUG] JSON decode error, attempting to fix LaTeX escapes...")

                # More comprehensive LaTeX escape fixing
                # Fix common LaTeX commands that need double backslashes
                latex_commands = [
                    'frac', 'text', 'le', 'times', 'cap', 'xi', 'quad', 'circ',
                    'ge', 'ne', 'pm', 'sqrt', 'cdot', 'div', 'sum', 'int', 'lim',
                    'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'min', 'max'
                ]

                for cmd in latex_commands:
                    # Replace \command with \\command, but not \\command (already escaped)
                    pattern = rf'(?<!\\)\\{cmd}(?![a-zA-Z])'
                    content = re.sub(pattern, rf'\\\\{cmd}', content)

                # Fix other common LaTeX patterns
                # Fix single backslashes before special characters in LaTeX contexts
                content = re.sub(r'(?<!\\)\\(?=[{}\[\]()^_])', r'\\\\', content)

                # Fix degree symbols and other special cases
                content = re.sub(r'(?<!\\)\\degree', r'\\\\degree', content)

                # Try parsing again
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e2:
                    print(f"[DEBUG] Still failing after LaTeX fixes. Error at position {e2.pos}")
                    # Extract context around the error
                    start = max(0, e2.pos - 100)
                    end = min(len(content), e2.pos + 100)
                    error_context = content[start:end]
                    print(f"[DEBUG] Error context: ...{error_context}...")

                    # More aggressive fix - escape all single backslashes in string values
                    # This is risky but may be needed
                    pattern = r'"([^"]*(?:\\.[^"]*)*)"'

                    def fix_string_escapes(match):
                        string_content = match.group(1)
                        # Double any single backslashes that aren't part of valid escape sequences
                        fixed = re.sub(r'(?<!\\)\\(?!["\\\/nrtfb])', r'\\\\', string_content)
                        return f'"{fixed}"'

                    content = re.sub(pattern, fix_string_escapes, content)

                    return json.loads(content)

        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return {}

    async def get_questions_from_local_file(
        self,
        file_path: str = "data/processed/2025p1.json",
        limit: int = 10,
        exclude_with_assets: bool = True
    ) -> List[Dict[str, Any]]:
        """Load questions from local JSON file for testing (fallback method)"""
        try:
            paper_data = self._safe_json_load(file_path)

            questions = paper_data.get("questions", [])

            # Filter out questions with assets if requested
            if exclude_with_assets:
                questions = [q for q in questions if not q.get("assets", [])]

            # Return random sample up to limit
            import random
            if len(questions) > limit:
                questions = random.sample(questions, limit)

            return questions

        except Exception as e:
            print(f"Error loading questions from local file: {e}")
            return []

    def _analyze_question_relationships(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze questions to identify which ones belong to the same set"""
        # Group questions by potential sets
        question_sets = {}

        for q in questions:
            local_id = q.get('question_id_local', '')
            global_id = q.get('question_id_global', '')

            # Strategy 1: Group by question number pattern in local_id
            if local_id.startswith('Q') and len(local_id) > 1:
                # Extract base number (handles Q1a, Q10b, Q15ai, etc.)
                base_match = ''
                for i, char in enumerate(local_id[1:], 1):
                    if char.isdigit():
                        base_match += char
                    else:
                        break

                if base_match:
                    set_key = f"Q{base_match}"
                    if set_key not in question_sets:
                        question_sets[set_key] = []
                    question_sets[set_key].append(q)

            # Strategy 2: Group by global_id pattern for database queries
            elif '_q' in global_id:
                # Extract pattern like "0580_SP_25_P1_q10" from "0580_SP_25_P1_q10a"
                base_pattern = global_id.split('_q')[0] + '_q'
                if len(global_id.split('_q')) > 1:
                    question_part = global_id.split('_q')[1]
                    # Extract just the number part
                    base_num = ''
                    for char in question_part:
                        if char.isdigit():
                            base_num += char
                        else:
                            break

                    if base_num:
                        set_key = f"{base_pattern}{base_num}"
                        if set_key not in question_sets:
                            question_sets[set_key] = []
                        question_sets[set_key].append(q)

        # Filter out single-question "sets" and return only true multi-part questions
        return {k: v for k, v in question_sets.items() if len(v) > 1}

    def _find_question_set_key(self, question_id: str, all_questions: List[Dict[str, Any]]) -> Optional[str]:
        """Find the set key for a given question ID"""
        question_sets = self._analyze_question_relationships(all_questions)

        for set_key, questions in question_sets.items():
            for q in questions:
                if (q.get('question_id_local') == question_id or
                    q.get('question_id_global') == question_id or
                    question_id in q.get('question_id_local', '') or
                    question_id in q.get('question_id_global', '')):
                    return set_key

        return None

    async def get_intelligent_question_set(self, question_id: str, source: str = "auto") -> List[Dict[str, Any]]:
        """Intelligently find all questions that belong to the same set as the given question"""
        if source == "auto":
            # Try database first, fallback to local
            try:
                return await self._get_intelligent_question_set_from_db(question_id)
            except Exception:
                return await self._get_intelligent_question_set_from_local(question_id)
        elif source == "database":
            return await self._get_intelligent_question_set_from_db(question_id)
        elif source == "local":
            return await self._get_intelligent_question_set_from_local(question_id)
        else:
            return []

    async def _get_intelligent_question_set_from_local(self, question_id: str) -> List[Dict[str, Any]]:
        """Get question set from local file using intelligent analysis"""
        try:
            paper_data = self._safe_json_load("data/processed/2025p1.json")
            all_questions = paper_data.get("questions", [])

            # Find which set this question belongs to
            set_key = self._find_question_set_key(question_id, all_questions)

            if not set_key:
                # If no set found, try to find the individual question
                target_question = next((q for q in all_questions if
                                      q.get("question_id_global") == question_id or
                                      q.get("question_id_local") == question_id), None)
                return [target_question] if target_question else []

            # Get all questions in the identified set
            question_sets = self._analyze_question_relationships(all_questions)
            question_set = question_sets.get(set_key, [])

            # Sort by local ID for proper ordering
            question_set.sort(key=lambda x: x.get("question_id_local", ""))

            return question_set

        except Exception as e:
            print(f"Error in intelligent question set retrieval from local: {e}")
            return []

    async def _get_intelligent_question_set_from_db(self, question_id: str) -> List[Dict[str, Any]]:
        """Get question set from database using intelligent analysis"""
        try:
            # First, get the target question to understand its pattern
            target_query = """
            SELECT q.*, COUNT(qa.assets_id) as asset_count
            FROM questions q
            LEFT JOIN questions_rels qa ON q.id = qa.parent_id AND qa.path = 'assets'
            WHERE (q.question_id_global = $1 OR q.question_id_local = $1)
            AND q.origin = 'past_year'
            GROUP BY q.id
            LIMIT 1
            """

            async with self.pool.acquire() as conn:
                target_row = await conn.fetchrow(target_query, question_id)
                if not target_row:
                    return []

                target_question = dict(target_row)
                global_id = target_question.get('question_id_global', '')
                local_id = target_question.get('question_id_local', '')

                # Strategy 1: Extract base pattern from global_id
                if '_q' in global_id:
                    base_pattern = global_id.split('_q')[0] + '_q'
                    question_part = global_id.split('_q')[1]

                    # Extract base number
                    base_num = ''
                    for char in question_part:
                        if char.isdigit():
                            base_num += char
                        else:
                            break

                    if base_num:
                        # Find all questions with the same base pattern
                        set_query = """
                        SELECT q.*, COUNT(qa.assets_id) as asset_count
                        FROM questions q
                        LEFT JOIN questions_rels qa ON q.id = qa.parent_id AND qa.path = 'assets'
                        WHERE q.question_id_global LIKE $1
                        AND q.origin = 'past_year'
                        GROUP BY q.id
                        ORDER BY q.question_id_global
                        """

                        pattern = f"{base_pattern}{base_num}%"
                        rows = await conn.fetch(set_query, pattern)

                        question_set = [dict(row) for row in rows]

                        # If we found multiple questions, return the set
                        if len(question_set) > 1:
                            return question_set

                # Strategy 2: Try local_id pattern if global_id didn't work
                if local_id.startswith('Q'):
                    base_num = ''
                    for char in local_id[1:]:
                        if char.isdigit():
                            base_num += char
                        else:
                            break

                    if base_num:
                        # This would require a more complex query for local_id patterns
                        # For now, fallback to the original method
                        return await self.get_full_question_set(question_id)

                # If no set found, return just the target question
                return [target_question]

        except Exception as e:
            print(f"Error in intelligent question set retrieval from database: {e}")
            return []

    async def get_question_set_from_local_file(
        self,
        question_id: str,
        file_path: str = "data/processed/2025p1.json"
    ) -> List[Dict[str, Any]]:
        """Get full question set from local file for testing - now uses intelligent analysis"""
        return await self._get_intelligent_question_set_from_local(question_id)
