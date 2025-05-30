#!/usr/bin/env python3
"""
Database Setup Script for Derivativ.ai Question Generation System.

This script safely creates tables with 'deriv_' prefix to avoid conflicts
with existing Payload CMS tables in the shared Neon database.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncpg
from dotenv import load_dotenv

load_dotenv()


class DerivDatabaseSetup:
    """Safe database setup for Derivativ.ai question generation system"""

    def __init__(self):
        self.connection_string = os.getenv("NEON_DATABASE_URL")
        if not self.connection_string:
            raise ValueError("NEON_DATABASE_URL not found in environment")

    async def inspect_existing_tables(self):
        """Inspect what tables already exist in the database"""
        print("🔍 Inspecting existing database tables...")

        conn = await asyncpg.connect(self.connection_string)
        try:
            # Get all tables
            tables = await conn.fetch("""
                SELECT table_name, table_schema
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)

            print(f"\n📋 Found {len(tables)} existing tables:")
            for table in tables:
                print(f"   • {table['table_name']}")

            # Check for any Deriv-related tables
            deriv_tables = [t for t in tables if 'deriv' in t['table_name'].lower()]
            if deriv_tables:
                print(f"\n⚠️  Found {len(deriv_tables)} existing Deriv tables:")
                for table in deriv_tables:
                    print(f"   • {table['table_name']}")

            return [t['table_name'] for t in tables]

        finally:
            await conn.close()

    async def create_deriv_tables(self, force: bool = False):
        """Create Deriv tables with proper prefixes"""

        existing_tables = await self.inspect_existing_tables()

        # Define our tables with deriv_ prefix
        deriv_table_names = [
            'deriv_generation_sessions',
            'deriv_llm_interactions',
            'deriv_candidate_questions',
            'deriv_review_results',
            'deriv_error_logs',
            'deriv_manual_review_queue'
        ]

        # Check for conflicts
        conflicts = [t for t in deriv_table_names if t in existing_tables]
        if conflicts and not force:
            print(f"\n❌ Table conflicts found: {conflicts}")
            print("Use force=True to recreate tables")
            return False

        print(f"\n🛠️  Creating Deriv tables...")

        conn = await asyncpg.connect(self.connection_string)
        try:
            # Drop existing Deriv tables if force mode
            if force and conflicts:
                print("🗑️  Dropping existing Deriv tables...")
                for table in reversed(deriv_table_names):  # Reverse order for FK constraints
                    if table in existing_tables:
                        await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                        print(f"   ✅ Dropped {table}")

            # Create tables in dependency order
            await self._create_generation_sessions_table(conn)
            await self._create_llm_interactions_table(conn)
            await self._create_candidate_questions_table(conn)
            await self._create_review_results_table(conn)
            await self._create_error_logs_table(conn)
            await self._create_manual_review_queue_table(conn)

            print("✅ All Deriv tables created successfully!")
            return True

        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
        finally:
            await conn.close()

    async def _create_generation_sessions_table(self, conn):
        """Create generation sessions table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deriv_generation_sessions (
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

                -- Indexes
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_sessions_config ON deriv_generation_sessions(config_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_sessions_status ON deriv_generation_sessions(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_sessions_timestamp ON deriv_generation_sessions(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_sessions_curriculum ON deriv_generation_sessions(curriculum_type)")

        print("   ✅ deriv_generation_sessions")

    async def _create_llm_interactions_table(self, conn):
        """Create LLM interactions table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deriv_llm_interactions (
                interaction_id UUID PRIMARY KEY,
                session_id UUID REFERENCES deriv_generation_sessions(session_id) ON DELETE CASCADE,

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
        """)

        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_interactions_session ON deriv_llm_interactions(session_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_interactions_agent ON deriv_llm_interactions(agent_type)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_interactions_model ON deriv_llm_interactions(model_used)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_interactions_success ON deriv_llm_interactions(success)")

        print("   ✅ deriv_llm_interactions")

    async def _create_candidate_questions_table(self, conn):
        """Create candidate questions table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deriv_candidate_questions (
                question_id UUID PRIMARY KEY,
                session_id UUID REFERENCES deriv_generation_sessions(session_id) ON DELETE CASCADE,

                -- Lineage tracking
                generation_interaction_id UUID REFERENCES deriv_llm_interactions(interaction_id),
                marking_interaction_id UUID REFERENCES deriv_llm_interactions(interaction_id),
                review_interaction_id UUID REFERENCES deriv_llm_interactions(interaction_id),

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
            )
        """)

        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_session ON deriv_candidate_questions(session_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_status ON deriv_candidate_questions(insertion_status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_grade ON deriv_candidate_questions(target_grade)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_marks ON deriv_candidate_questions(marks)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_refs ON deriv_candidate_questions USING GIN(subject_content_refs)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_topic ON deriv_candidate_questions USING GIN(topic_path)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_validation ON deriv_candidate_questions(validation_passed)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_questions_curriculum ON deriv_candidate_questions(curriculum_type)")

        print("   ✅ deriv_candidate_questions")

    async def _create_review_results_table(self, conn):
        """Create review results table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deriv_review_results (
                review_id UUID PRIMARY KEY,
                question_id UUID REFERENCES deriv_candidate_questions(question_id) ON DELETE CASCADE,
                interaction_id UUID REFERENCES deriv_llm_interactions(interaction_id),

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
        """)

        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_reviews_question ON deriv_review_results(question_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_reviews_outcome ON deriv_review_results(outcome)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_reviews_score ON deriv_review_results(overall_score)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_reviews_grade ON deriv_review_results(quality_grade)")

        print("   ✅ deriv_review_results")

    async def _create_error_logs_table(self, conn):
        """Create error logs table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deriv_error_logs (
                error_id UUID PRIMARY KEY,
                session_id UUID REFERENCES deriv_generation_sessions(session_id) ON DELETE CASCADE,
                interaction_id UUID REFERENCES deriv_llm_interactions(interaction_id) ON DELETE SET NULL,
                question_id UUID REFERENCES deriv_candidate_questions(question_id) ON DELETE SET NULL,

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
        """)

        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_errors_session ON deriv_error_logs(session_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_errors_severity ON deriv_error_logs(error_severity)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_errors_type ON deriv_error_logs(error_type)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_errors_resolved ON deriv_error_logs(resolved)")

        print("   ✅ deriv_error_logs")

    async def _create_manual_review_queue_table(self, conn):
        """Create manual review queue table"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deriv_manual_review_queue (
                queue_id UUID PRIMARY KEY,
                question_id UUID REFERENCES deriv_candidate_questions(question_id) ON DELETE CASCADE,
                review_id UUID REFERENCES deriv_review_results(review_id) ON DELETE CASCADE,

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
        """)

        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_queue_status ON deriv_manual_review_queue(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_queue_priority ON deriv_manual_review_queue(priority)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_queue_assigned ON deriv_manual_review_queue(assigned_to)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_queue_due ON deriv_manual_review_queue(due_date)")

        print("   ✅ deriv_manual_review_queue")

    async def test_connection(self):
        """Test database connection"""
        print("🔌 Testing database connection...")
        try:
            conn = await asyncpg.connect(self.connection_string)
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            if result == 1:
                print("✅ Database connection successful!")
                return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    async def get_table_stats(self):
        """Get statistics about Deriv tables"""
        print("\n📊 Deriv Table Statistics:")

        conn = await asyncpg.connect(self.connection_string)
        try:
            tables = [
                'deriv_generation_sessions',
                'deriv_llm_interactions',
                'deriv_candidate_questions',
                'deriv_review_results',
                'deriv_error_logs',
                'deriv_manual_review_queue'
            ]

            for table in tables:
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    print(f"   • {table}: {count} rows")
                except:
                    print(f"   • {table}: table does not exist")

        finally:
            await conn.close()


async def main():
    """Main setup process"""
    print("🚀 Derivativ.ai Database Setup")
    print("=" * 50)

    try:
        setup = DerivDatabaseSetup()

        # Test connection
        if not await setup.test_connection():
            return

        # Inspect existing tables
        await setup.inspect_existing_tables()

        # Ask user for confirmation
        print(f"\n🤔 Ready to create Deriv tables with 'deriv_' prefix?")
        print("   This is safe and won't affect your Payload CMS tables.")
        print("   Tables will be generic for any educational content.")

        response = input("   Continue? (y/n): ").lower().strip()
        if response != 'y':
            print("   Setup cancelled.")
            return

        # Create tables
        success = await setup.create_deriv_tables(force=False)

        if success:
            print(f"\n🎉 Database setup complete!")
            await setup.get_table_stats()
            print(f"\n✅ Ready for full persistence pipeline testing!")
        else:
            print(f"\n❌ Database setup failed.")

    except Exception as e:
        print(f"💥 Setup error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
