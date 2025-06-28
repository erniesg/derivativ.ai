-- Migration: Create dev_ prefixed tables for development testing
-- Only creates tables needed for question generation testing

-- Drop dev tables if they exist
DROP TABLE IF EXISTS dev_generated_questions CASCADE;
DROP TABLE IF EXISTS dev_generation_sessions CASCADE;

-- Create dev_generation_sessions table (copy of production structure)
CREATE TABLE IF NOT EXISTS public.dev_generation_sessions (
    -- Primary key
    id uuid default gen_random_uuid() primary key,

    -- Session identifier
    session_id uuid not null unique,

    -- Generation request metadata
    topic text not null,
    tier text not null check (tier in ('Core', 'Extended')),
    grade_level integer check (grade_level >= 1 and grade_level <= 12),
    marks integer not null check (marks > 0),
    count integer not null default 1 check (count > 0),
    calculator_policy text check (calculator_policy in ('required', 'allowed', 'not_allowed')),
    command_word text check (command_word in ('Calculate', 'Solve', 'Find', 'Determine', 'Show', 'Prove', 'Explain', 'Describe', 'State', 'Write')),

    -- Session status and metrics
    status text not null default 'pending' check (status in ('pending', 'in_progress', 'candidate', 'approved', 'rejected', 'failed')),
    total_processing_time numeric(8,3),
    questions_generated integer not null default 0,

    -- Complete session data as JSONB
    request_json jsonb not null,
    questions_json jsonb not null default '[]'::jsonb,
    quality_decisions_json jsonb not null default '[]'::jsonb,
    agent_results_json jsonb not null default '[]'::jsonb,

    -- Timestamps
    created_at timestamptz default now() not null,
    updated_at timestamptz default now() not null,
    completed_at timestamptz
);

-- Create dev_generated_questions table (copy of production structure)
CREATE TABLE IF NOT EXISTS public.dev_generated_questions (
    -- Primary key
    id uuid default gen_random_uuid() primary key,

    -- Flattened fields for efficient querying
    question_id_global uuid not null unique,
    question_id_local text not null,
    question_number_display text not null,
    marks integer not null check (marks > 0),
    tier text not null check (tier in ('Core', 'Extended')),
    command_word text check (command_word in ('Calculate', 'Solve', 'Find', 'Determine', 'Show', 'Prove', 'Explain', 'Describe', 'State', 'Write')),

    -- Question content and metadata
    raw_text_content text not null,
    quality_score numeric(3,2) check (quality_score >= 0 and quality_score <= 1),

    -- Origin tracking
    origin text not null default 'generated' check (origin in ('generated', 'past_paper', 'textbook', 'manual')),
    source_reference text,

    -- Full Pydantic model as JSONB for complete data fidelity
    content_json jsonb not null,

    -- Timestamps
    created_at timestamptz default now() not null,
    updated_at timestamptz default now() not null
);

-- Create indexes for dev_generation_sessions
CREATE INDEX IF NOT EXISTS idx_dev_sessions_session_id ON public.dev_generation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_dev_sessions_status ON public.dev_generation_sessions(status);
CREATE INDEX IF NOT EXISTS idx_dev_sessions_topic ON public.dev_generation_sessions(topic);
CREATE INDEX IF NOT EXISTS idx_dev_sessions_created_at ON public.dev_generation_sessions(created_at);

-- Create indexes for dev_generated_questions
CREATE INDEX IF NOT EXISTS idx_dev_generated_questions_tier ON public.dev_generated_questions(tier);
CREATE INDEX IF NOT EXISTS idx_dev_generated_questions_marks ON public.dev_generated_questions(marks);
CREATE INDEX IF NOT EXISTS idx_dev_generated_questions_command_word ON public.dev_generated_questions(command_word);
CREATE INDEX IF NOT EXISTS idx_dev_generated_questions_quality_score ON public.dev_generated_questions(quality_score);
CREATE INDEX IF NOT EXISTS idx_dev_generated_questions_created_at ON public.dev_generated_questions(created_at);

-- Update triggers for updated_at timestamps
CREATE TRIGGER update_dev_generation_sessions_updated_at
    BEFORE UPDATE ON public.dev_generation_sessions
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_dev_generated_questions_updated_at
    BEFORE UPDATE ON public.dev_generated_questions
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Completed_at trigger for dev sessions
CREATE TRIGGER update_dev_sessions_completed_at
    BEFORE UPDATE ON public.dev_generation_sessions
    FOR EACH ROW EXECUTE FUNCTION public.update_completed_at_column();

-- Row Level Security
ALTER TABLE public.dev_generation_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dev_generated_questions ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users
CREATE POLICY "Enable all operations for authenticated users" ON public.dev_generation_sessions
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Enable all operations for authenticated users" ON public.dev_generated_questions
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');
