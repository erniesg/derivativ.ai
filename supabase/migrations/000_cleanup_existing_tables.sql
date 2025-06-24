-- Cleanup script: Remove existing tables if they exist
-- Run this BEFORE the main migrations if you have conflicting table structures

-- Drop existing tables in reverse dependency order
DROP TABLE IF EXISTS public.generation_sessions CASCADE;
DROP TABLE IF EXISTS public.questions CASCADE;

-- Drop enum tables if they exist
DROP TABLE IF EXISTS public.tiers CASCADE;
DROP TABLE IF EXISTS public.command_words CASCADE;
DROP TABLE IF EXISTS public.calculator_policies CASCADE;
DROP TABLE IF EXISTS public.generation_statuses CASCADE;
DROP TABLE IF EXISTS public.question_origins CASCADE;

-- Drop any existing functions
DROP FUNCTION IF EXISTS public.update_updated_at_column() CASCADE;
DROP FUNCTION IF EXISTS public.update_completed_at_column() CASCADE;

-- Note: This will remove ALL existing data in these tables
-- Only run this if you want to start fresh
