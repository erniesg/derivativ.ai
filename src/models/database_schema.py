"""
Database Schema Definitions
==========================

Centralized definitions for all database tables, preventing inconsistencies
between different modules and ensuring proper naming conventions.
"""

from typing import Dict, List, Set
from enum import Enum


class TableNames:
    """Centralized table name definitions"""

    # Core tables with deriv_ prefix (for Neon shared database)
    GENERATION_SESSIONS = "deriv_generation_sessions"
    LLM_INTERACTIONS = "deriv_llm_interactions"
    CANDIDATE_QUESTIONS = "deriv_candidate_questions"  # Note: This is the actual table
    REVIEW_RESULTS = "deriv_review_results"
    ERROR_LOGS = "deriv_error_logs"
    MANUAL_REVIEW_QUEUE = "deriv_manual_review_queue"

    # Legacy table names that should be migrated
    LEGACY_CANDIDATE_QUESTIONS_EXTENDED = "candidate_questions_extended"
    LEGACY_GENERATION_SESSIONS = "generation_sessions"
    LEGACY_LLM_INTERACTIONS = "llm_interactions"
    LEGACY_REVIEW_RESULTS = "review_results"
    LEGACY_ERROR_LOGS = "error_logs"
    LEGACY_MANUAL_REVIEW_QUEUE = "manual_review_queue"

    @classmethod
    def get_all_table_names(cls) -> List[str]:
        """Get all current table names"""
        return [
            cls.GENERATION_SESSIONS,
            cls.LLM_INTERACTIONS,
            cls.CANDIDATE_QUESTIONS,
            cls.REVIEW_RESULTS,
            cls.ERROR_LOGS,
            cls.MANUAL_REVIEW_QUEUE
        ]

    @classmethod
    def get_legacy_table_names(cls) -> List[str]:
        """Get legacy table names that need migration"""
        return [
            cls.LEGACY_CANDIDATE_QUESTIONS_EXTENDED,
            cls.LEGACY_GENERATION_SESSIONS,
            cls.LEGACY_LLM_INTERACTIONS,
            cls.LEGACY_REVIEW_RESULTS,
            cls.LEGACY_ERROR_LOGS,
            cls.LEGACY_MANUAL_REVIEW_QUEUE
        ]


class DatabaseSchemas:
    """SQL schema definitions for all tables"""

    GENERATION_SESSIONS = f"""
    CREATE TABLE IF NOT EXISTS {TableNames.GENERATION_SESSIONS} (
        session_id UUID PRIMARY KEY,
        config_id VARCHAR(100) NOT NULL,
        timestamp TIMESTAMP DEFAULT NOW(),
        status VARCHAR(20) CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
        total_questions_requested INT NOT NULL CHECK (total_questions_requested > 0),
        questions_generated INT DEFAULT 0 CHECK (questions_generated >= 0),
        questions_approved INT DEFAULT 0 CHECK (questions_approved >= 0),
        error_count INT DEFAULT 0 CHECK (error_count >= 0),

        -- Session metadata
        user_id VARCHAR(100),
        session_notes TEXT,
        configuration_snapshot JSONB,
        summary_metrics JSONB,
        curriculum_type VARCHAR(50) DEFAULT 'cambridge_igcse',

        -- Timestamps
        started_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """

    LLM_INTERACTIONS = f"""
    CREATE TABLE IF NOT EXISTS {TableNames.LLM_INTERACTIONS} (
        interaction_id UUID PRIMARY KEY,
        session_id UUID REFERENCES {TableNames.GENERATION_SESSIONS}(session_id) ON DELETE CASCADE,

        -- Agent info
        agent_type VARCHAR(50) NOT NULL CHECK (agent_type IN ('generator', 'marker', 'reviewer', 'refiner')),
        model_used VARCHAR(100) NOT NULL,

        -- Interaction data
        prompt_text TEXT NOT NULL,
        raw_response TEXT,
        parsed_response JSONB,

        -- Performance metrics
        processing_time_ms INT CHECK (processing_time_ms >= 0),
        token_usage JSONB,
        cost_estimate DECIMAL(10,6),

        -- Status
        success BOOLEAN DEFAULT TRUE,
        error_message TEXT,
        retry_count INT DEFAULT 0,

        -- Model parameters
        temperature FLOAT CHECK (temperature >= 0 AND temperature <= 2),
        max_tokens INT CHECK (max_tokens > 0),

        -- Timestamps
        timestamp TIMESTAMP DEFAULT NOW(),
        started_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP
    )
    """

    CANDIDATE_QUESTIONS = f"""
    CREATE TABLE IF NOT EXISTS {TableNames.CANDIDATE_QUESTIONS} (
        question_id UUID PRIMARY KEY,
        session_id UUID REFERENCES {TableNames.GENERATION_SESSIONS}(session_id) ON DELETE CASCADE,

        -- Lineage tracking
        generation_interaction_id UUID REFERENCES {TableNames.LLM_INTERACTIONS}(interaction_id),
        marking_interaction_id UUID REFERENCES {TableNames.LLM_INTERACTIONS}(interaction_id),
        review_interaction_id UUID REFERENCES {TableNames.LLM_INTERACTIONS}(interaction_id),

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
        parent_question_id UUID REFERENCES {TableNames.CANDIDATE_QUESTIONS}(question_id),

        -- Timestamps
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """

    REVIEW_RESULTS = f"""
    CREATE TABLE IF NOT EXISTS {TableNames.REVIEW_RESULTS} (
        review_id UUID PRIMARY KEY,
        question_id UUID REFERENCES {TableNames.CANDIDATE_QUESTIONS}(question_id) ON DELETE CASCADE,
        interaction_id UUID REFERENCES {TableNames.LLM_INTERACTIONS}(interaction_id),

        -- Review outcome
        outcome VARCHAR(20) NOT NULL CHECK (outcome IN ('approve', 'minor_revisions', 'major_revisions', 'reject')),

        -- Scores (0.0 to 1.0)
        overall_score DECIMAL(3,2) CHECK (overall_score BETWEEN 0 AND 1),
        mathematical_accuracy DECIMAL(3,2) CHECK (mathematical_accuracy BETWEEN 0 AND 1),
        syllabus_compliance DECIMAL(3,2) CHECK (syllabus_compliance BETWEEN 0 AND 1),
        difficulty_alignment DECIMAL(3,2) CHECK (difficulty_alignment BETWEEN 0 AND 1),
        marking_quality DECIMAL(3,2) CHECK (marking_quality BETWEEN 0 AND 1),
        pedagogical_soundness DECIMAL(3,2) CHECK (pedagogical_soundness BETWEEN 0 AND 1),
        technical_quality DECIMAL(3,2) CHECK (technical_quality BETWEEN 0 AND 1),

        -- Feedback
        feedback_summary TEXT,
        specific_feedback JSONB,
        suggested_improvements JSONB,

        -- Quality grades
        quality_grade VARCHAR(20) CHECK (quality_grade IN ('Excellent', 'Good', 'Satisfactory', 'Needs Improvement', 'Poor')),

        -- Review metadata
        reviewer_type VARCHAR(20) DEFAULT 'ai' CHECK (reviewer_type IN ('ai', 'human', 'hybrid')),
        reviewer_id VARCHAR(100),
        review_duration_ms INT,

        -- Timestamps
        timestamp TIMESTAMP DEFAULT NOW(),
        reviewed_at TIMESTAMP DEFAULT NOW()
    )
    """

    ERROR_LOGS = f"""
    CREATE TABLE IF NOT EXISTS {TableNames.ERROR_LOGS} (
        error_id UUID PRIMARY KEY,
        session_id UUID REFERENCES {TableNames.GENERATION_SESSIONS}(session_id) ON DELETE CASCADE,
        interaction_id UUID REFERENCES {TableNames.LLM_INTERACTIONS}(interaction_id) ON DELETE SET NULL,
        question_id UUID REFERENCES {TableNames.CANDIDATE_QUESTIONS}(question_id) ON DELETE SET NULL,

        -- Error details
        error_type VARCHAR(50) NOT NULL,
        error_severity VARCHAR(20) CHECK (error_severity IN ('low', 'medium', 'high', 'critical')) DEFAULT 'medium',
        error_code VARCHAR(20),
        error_message TEXT NOT NULL,
        stack_trace TEXT,

        -- Context
        context_data JSONB,
        step_name VARCHAR(50),
        agent_type VARCHAR(20),
        model_used VARCHAR(100),

        -- Resolution tracking
        resolved BOOLEAN DEFAULT FALSE,
        resolution_notes TEXT,
        resolved_by VARCHAR(100),
        resolved_at TIMESTAMP,

        -- Timestamps
        timestamp TIMESTAMP DEFAULT NOW(),
        reported_at TIMESTAMP DEFAULT NOW()
    )
    """

    MANUAL_REVIEW_QUEUE = f"""
    CREATE TABLE IF NOT EXISTS {TableNames.MANUAL_REVIEW_QUEUE} (
        queue_id UUID PRIMARY KEY,
        question_id UUID REFERENCES {TableNames.CANDIDATE_QUESTIONS}(question_id) ON DELETE CASCADE,
        review_id UUID REFERENCES {TableNames.REVIEW_RESULTS}(review_id) ON DELETE CASCADE,

        -- Queue management
        priority INT DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
        status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'assigned', 'in_review', 'completed', 'escalated')),

        -- Assignment
        assigned_to VARCHAR(100),
        assigned_at TIMESTAMP,
        due_date TIMESTAMP,

        -- Review process
        review_started_at TIMESTAMP,
        review_completed_at TIMESTAMP,
        estimated_time_minutes INT,
        actual_time_minutes INT,

        -- Results
        reviewer_notes TEXT,
        final_decision VARCHAR(20) CHECK (final_decision IN ('approved', 'rejected', 'needs_revision', 'escalated')),
        admin_override BOOLEAN DEFAULT FALSE,

        -- Metadata
        queue_reason TEXT,
        complexity_score DECIMAL(3,2),
        tags TEXT[],

        -- Timestamps
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """

    @classmethod
    def get_all_schemas(cls) -> Dict[str, str]:
        """Get all table schemas"""
        return {
            TableNames.GENERATION_SESSIONS: cls.GENERATION_SESSIONS,
            TableNames.LLM_INTERACTIONS: cls.LLM_INTERACTIONS,
            TableNames.CANDIDATE_QUESTIONS: cls.CANDIDATE_QUESTIONS,
            TableNames.REVIEW_RESULTS: cls.REVIEW_RESULTS,
            TableNames.ERROR_LOGS: cls.ERROR_LOGS,
            TableNames.MANUAL_REVIEW_QUEUE: cls.MANUAL_REVIEW_QUEUE,
        }


class DatabaseIndexes:
    """Index definitions for all tables"""

    @staticmethod
    def get_generation_sessions_indexes() -> List[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS idx_deriv_sessions_config ON {TableNames.GENERATION_SESSIONS}(config_id)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_sessions_status ON {TableNames.GENERATION_SESSIONS}(status)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_sessions_timestamp ON {TableNames.GENERATION_SESSIONS}(timestamp)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_sessions_curriculum ON {TableNames.GENERATION_SESSIONS}(curriculum_type)",
        ]

    @staticmethod
    def get_llm_interactions_indexes() -> List[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS idx_deriv_interactions_session ON {TableNames.LLM_INTERACTIONS}(session_id)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_interactions_agent ON {TableNames.LLM_INTERACTIONS}(agent_type)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_interactions_model ON {TableNames.LLM_INTERACTIONS}(model_used)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_interactions_success ON {TableNames.LLM_INTERACTIONS}(success)",
        ]

    @staticmethod
    def get_candidate_questions_indexes() -> List[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_session ON {TableNames.CANDIDATE_QUESTIONS}(session_id)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_status ON {TableNames.CANDIDATE_QUESTIONS}(insertion_status)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_grade ON {TableNames.CANDIDATE_QUESTIONS}(target_grade)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_marks ON {TableNames.CANDIDATE_QUESTIONS}(marks)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_refs ON {TableNames.CANDIDATE_QUESTIONS} USING GIN(subject_content_refs)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_topic ON {TableNames.CANDIDATE_QUESTIONS} USING GIN(topic_path)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_validation ON {TableNames.CANDIDATE_QUESTIONS}(validation_passed)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_questions_curriculum ON {TableNames.CANDIDATE_QUESTIONS}(curriculum_type)",
        ]

    @staticmethod
    def get_review_results_indexes() -> List[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS idx_deriv_reviews_question ON {TableNames.REVIEW_RESULTS}(question_id)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_reviews_outcome ON {TableNames.REVIEW_RESULTS}(outcome)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_reviews_score ON {TableNames.REVIEW_RESULTS}(overall_score)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_reviews_grade ON {TableNames.REVIEW_RESULTS}(quality_grade)",
        ]

    @staticmethod
    def get_error_logs_indexes() -> List[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS idx_deriv_errors_session ON {TableNames.ERROR_LOGS}(session_id)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_errors_severity ON {TableNames.ERROR_LOGS}(error_severity)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_errors_type ON {TableNames.ERROR_LOGS}(error_type)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_errors_resolved ON {TableNames.ERROR_LOGS}(resolved)",
        ]

    @staticmethod
    def get_manual_review_queue_indexes() -> List[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS idx_deriv_queue_status ON {TableNames.MANUAL_REVIEW_QUEUE}(status)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_queue_priority ON {TableNames.MANUAL_REVIEW_QUEUE}(priority)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_queue_assigned ON {TableNames.MANUAL_REVIEW_QUEUE}(assigned_to)",
            f"CREATE INDEX IF NOT EXISTS idx_deriv_queue_due ON {TableNames.MANUAL_REVIEW_QUEUE}(due_date)",
        ]

    @classmethod
    def get_all_indexes(cls) -> List[str]:
        """Get all index creation statements"""
        indexes = []
        indexes.extend(cls.get_generation_sessions_indexes())
        indexes.extend(cls.get_llm_interactions_indexes())
        indexes.extend(cls.get_candidate_questions_indexes())
        indexes.extend(cls.get_review_results_indexes())
        indexes.extend(cls.get_error_logs_indexes())
        indexes.extend(cls.get_manual_review_queue_indexes())
        return indexes


# Migration helpers
def get_table_migration_map() -> Dict[str, str]:
    """Mapping from legacy table names to current table names"""
    return {
        TableNames.LEGACY_GENERATION_SESSIONS: TableNames.GENERATION_SESSIONS,
        TableNames.LEGACY_LLM_INTERACTIONS: TableNames.LLM_INTERACTIONS,
        TableNames.LEGACY_CANDIDATE_QUESTIONS_EXTENDED: TableNames.CANDIDATE_QUESTIONS,
        TableNames.LEGACY_REVIEW_RESULTS: TableNames.REVIEW_RESULTS,
        TableNames.LEGACY_ERROR_LOGS: TableNames.ERROR_LOGS,
        TableNames.LEGACY_MANUAL_REVIEW_QUEUE: TableNames.MANUAL_REVIEW_QUEUE,
    }


def identify_table_inconsistencies() -> Dict[str, List[str]]:
    """Identify files that might be using inconsistent table names"""
    return {
        "current_deriv_tables": TableNames.get_all_table_names(),
        "legacy_tables_to_migrate": TableNames.get_legacy_table_names(),
        "migration_map": get_table_migration_map(),
    }
