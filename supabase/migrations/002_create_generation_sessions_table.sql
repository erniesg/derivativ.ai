-- Migration: Create generation_sessions table
-- Date: 2025-06-21
-- Description: Track multi-agent question generation sessions with agent results

-- Create generation_sessions table
create table if not exists public.generation_sessions (
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

-- Create indexes for efficient querying
create index if not exists idx_sessions_session_id on public.generation_sessions(session_id);
create index if not exists idx_sessions_status on public.generation_sessions(status);
create index if not exists idx_sessions_topic on public.generation_sessions(topic);
create index if not exists idx_sessions_tier on public.generation_sessions(tier);
create index if not exists idx_sessions_created_at on public.generation_sessions(created_at);
create index if not exists idx_sessions_processing_time on public.generation_sessions(total_processing_time);

-- GIN indexes for JSONB content search
create index if not exists idx_sessions_request_json on public.generation_sessions using gin(request_json);
create index if not exists idx_sessions_questions_json on public.generation_sessions using gin(questions_json);
create index if not exists idx_sessions_agent_results_json on public.generation_sessions using gin(agent_results_json);

-- Composite indexes for common query patterns
create index if not exists idx_sessions_status_created on public.generation_sessions(status, created_at);
create index if not exists idx_sessions_tier_status on public.generation_sessions(tier, status);

-- Update trigger for updated_at timestamp
create trigger update_generation_sessions_updated_at
    before update on public.generation_sessions
    for each row execute function public.update_updated_at_column();

-- Trigger to update completed_at when status changes to completed states
create or replace function public.update_completed_at_column()
returns trigger as $$
begin
    if new.status in ('approved', 'rejected', 'failed') and old.status != new.status then
        new.completed_at = now();
    end if;
    return new;
end;
$$ language plpgsql;

create trigger update_sessions_completed_at
    before update on public.generation_sessions
    for each row execute function public.update_completed_at_column();

-- Row Level Security (RLS) - enable but allow all for now
alter table public.generation_sessions enable row level security;

-- Allow all operations for authenticated users (can be refined later)
create policy "Enable all operations for authenticated users" on public.generation_sessions
    for all using (auth.role() = 'authenticated' or auth.role() = 'anon');

-- Comments for documentation
comment on table public.generation_sessions is 'Multi-agent question generation sessions with full workflow tracking';
comment on column public.generation_sessions.session_id is 'Unique session identifier for client tracking';
comment on column public.generation_sessions.request_json is 'Complete GenerationRequest Pydantic model as JSONB';
comment on column public.generation_sessions.questions_json is 'Array of generated Question models as JSONB';
comment on column public.generation_sessions.agent_results_json is 'Array of AgentResult models tracking multi-agent workflow';
comment on column public.generation_sessions.total_processing_time is 'Total generation time in seconds for performance monitoring';
