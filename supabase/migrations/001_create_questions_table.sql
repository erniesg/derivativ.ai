-- Migration: Create questions table with hybrid storage pattern
-- Date: 2025-06-21
-- Description: Questions table with flattened fields for querying + full JSONB for data fidelity

-- Create questions table
create table if not exists public.questions (
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

-- Create indexes for efficient querying
create index if not exists idx_questions_tier on public.questions(tier);
create index if not exists idx_questions_marks on public.questions(marks);
create index if not exists idx_questions_command_word on public.questions(command_word);
create index if not exists idx_questions_quality_score on public.questions(quality_score);
create index if not exists idx_questions_origin on public.questions(origin);
create index if not exists idx_questions_created_at on public.questions(created_at);

-- GIN index for JSONB content search
create index if not exists idx_questions_content_json on public.questions using gin(content_json);

-- Composite indexes for common query patterns
create index if not exists idx_questions_tier_marks on public.questions(tier, marks);
create index if not exists idx_questions_tier_quality on public.questions(tier, quality_score);

-- Update trigger for updated_at timestamp
create or replace function public.update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger update_questions_updated_at
    before update on public.questions
    for each row execute function public.update_updated_at_column();

-- Row Level Security (RLS) - enable but allow all for now
alter table public.questions enable row level security;

-- Allow all operations for authenticated users (can be refined later)
create policy "Enable all operations for authenticated users" on public.questions
    for all using (auth.role() = 'authenticated' or auth.role() = 'anon');

-- Comments for documentation
comment on table public.questions is 'Cambridge IGCSE Mathematics questions with hybrid storage pattern';
comment on column public.questions.question_id_global is 'Global unique identifier for questions across all sources';
comment on column public.questions.content_json is 'Complete Pydantic Question model serialized as JSONB';
comment on column public.questions.quality_score is 'AI-assessed quality score from 0.0 to 1.0';
comment on column public.questions.origin is 'Source of the question: generated, past_paper, textbook, or manual';
