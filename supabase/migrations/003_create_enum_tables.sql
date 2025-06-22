-- Migration: Create enum tables for referential integrity
-- Date: 2025-06-21
-- Description: Enum tables for tiers, command words, calculator policies, etc.

-- Create tiers enum table
create table if not exists public.tiers (
    value text primary key,
    description text not null,
    display_order integer not null
);

insert into public.tiers (value, description, display_order) values
    ('Core', 'Core tier - standard difficulty level', 1),
    ('Extended', 'Extended tier - higher difficulty level', 2)
on conflict (value) do nothing;

-- Create command_words enum table
create table if not exists public.command_words (
    value text primary key,
    description text not null,
    category text not null,
    display_order integer not null
);

insert into public.command_words (value, description, category, display_order) values
    ('Calculate', 'Work out from given facts, figures or information', 'Computational', 1),
    ('Solve', 'Work out the answer to a problem', 'Computational', 2),
    ('Find', 'Identify or locate something', 'Computational', 3),
    ('Determine', 'Establish exactly by research or calculation', 'Computational', 4),
    ('Show', 'Provide structured evidence that leads to a given result', 'Proof', 5),
    ('Prove', 'Provide a valid mathematical argument', 'Proof', 6),
    ('Explain', 'Set out purposes or reasons', 'Explanation', 7),
    ('Describe', 'Give a detailed account', 'Explanation', 8),
    ('State', 'Express in clear terms', 'Recall', 9),
    ('Write', 'Express in writing', 'Recall', 10)
on conflict (value) do nothing;

-- Create calculator_policies enum table
create table if not exists public.calculator_policies (
    value text primary key,
    description text not null,
    symbol text,
    display_order integer not null
);

insert into public.calculator_policies (value, description, symbol, display_order) values
    ('required', 'Calculator required for this question', '🧮', 1),
    ('allowed', 'Calculator allowed but not required', '✓', 2),
    ('not_allowed', 'Calculator not allowed for this question', '✗', 3)
on conflict (value) do nothing;

-- Create generation_statuses enum table
create table if not exists public.generation_statuses (
    value text primary key,
    description text not null,
    is_final boolean not null default false,
    display_order integer not null
);

insert into public.generation_statuses (value, description, is_final, display_order) values
    ('pending', 'Generation request received, not yet started', false, 1),
    ('in_progress', 'Multi-agent generation workflow is running', false, 2),
    ('candidate', 'Generation complete, awaiting quality review', false, 3),
    ('approved', 'Question approved for use', true, 4),
    ('rejected', 'Question rejected, needs regeneration', true, 5),
    ('failed', 'Generation failed due to error', true, 6)
on conflict (value) do nothing;

-- Create question_origins enum table
create table if not exists public.question_origins (
    value text primary key,
    description text not null,
    is_ai_generated boolean not null default false,
    display_order integer not null
);

insert into public.question_origins (value, description, is_ai_generated, display_order) values
    ('generated', 'AI-generated question using multi-agent workflow', true, 1),
    ('past_paper', 'Question from Cambridge IGCSE past papers', false, 2),
    ('textbook', 'Question adapted from educational textbooks', false, 3),
    ('manual', 'Manually created question by educators', false, 4)
on conflict (value) do nothing;

-- Add foreign key constraints to main tables (optional, for referential integrity)
-- Note: These are commented out to avoid breaking existing data, but can be enabled for strict validation

-- alter table public.questions
--     add constraint fk_questions_tier
--     foreign key (tier) references public.tiers(value);

-- alter table public.questions
--     add constraint fk_questions_command_word
--     foreign key (command_word) references public.command_words(value);

-- alter table public.questions
--     add constraint fk_questions_origin
--     foreign key (origin) references public.question_origins(value);

-- alter table public.generation_sessions
--     add constraint fk_sessions_tier
--     foreign key (tier) references public.tiers(value);

-- alter table public.generation_sessions
--     add constraint fk_sessions_command_word
--     foreign key (command_word) references public.command_words(value);

-- alter table public.generation_sessions
--     add constraint fk_sessions_calculator_policy
--     foreign key (calculator_policy) references public.calculator_policies(value);

-- alter table public.generation_sessions
--     add constraint fk_sessions_status
--     foreign key (status) references public.generation_statuses(value);

-- Row Level Security for enum tables (read-only for all users)
alter table public.tiers enable row level security;
alter table public.command_words enable row level security;
alter table public.calculator_policies enable row level security;
alter table public.generation_statuses enable row level security;
alter table public.question_origins enable row level security;

-- Allow read access to enum tables for all users
create policy "Enable read access for all users" on public.tiers
    for select using (true);

create policy "Enable read access for all users" on public.command_words
    for select using (true);

create policy "Enable read access for all users" on public.calculator_policies
    for select using (true);

create policy "Enable read access for all users" on public.generation_statuses
    for select using (true);

create policy "Enable read access for all users" on public.question_origins
    for select using (true);

-- Comments for documentation
comment on table public.tiers is 'Cambridge IGCSE tier levels (Core/Extended)';
comment on table public.command_words is 'Cambridge assessment command words with categories';
comment on table public.calculator_policies is 'Calculator usage policies for questions';
comment on table public.generation_statuses is 'Status values for generation workflow tracking';
comment on table public.question_origins is 'Source types for questions (AI-generated, past papers, etc.)';
