-- Migration: Clean document tables (CLEAN VERSION)
-- Date: 2025-06-29
-- Description: Remove all existing document tables and create clean, consistent ones

-- ===============================================
-- CLEANUP: Remove all existing document tables
-- ===============================================

-- Drop all document-related tables (in correct dependency order)
DROP TABLE IF EXISTS public.dev_document_files CASCADE;
DROP TABLE IF EXISTS public.document_files CASCADE;
DROP TABLE IF EXISTS public.dev_stored_documents CASCADE;
DROP TABLE IF EXISTS public.stored_documents CASCADE;
DROP TABLE IF EXISTS public.dev_documents CASCADE;
DROP TABLE IF EXISTS public.documents CASCADE;

-- Drop related functions
DROP FUNCTION IF EXISTS public.update_document_file_stats() CASCADE;
DROP FUNCTION IF EXISTS public.update_dev_document_file_stats() CASCADE;
DROP FUNCTION IF EXISTS public.update_search_content() CASCADE;
DROP FUNCTION IF EXISTS public.update_dev_search_content() CASCADE;

-- ===============================================
-- CREATE CLEAN TABLES
-- ===============================================

-- Create documents table (metadata)
CREATE TABLE public.documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID, -- Optional reference, no foreign key constraint

    -- Document metadata
    title TEXT NOT NULL CHECK (length(title) > 0),
    document_type TEXT NOT NULL CHECK (document_type IN ('worksheet', 'notes', 'textbook', 'slides')),
    detail_level TEXT CHECK (detail_level IN ('minimal', 'medium', 'comprehensive', 'guided')),
    topic TEXT,
    grade_level INTEGER CHECK (grade_level >= 1 AND grade_level <= 12),

    -- Generation metadata
    estimated_duration INTEGER CHECK (estimated_duration > 0),
    total_questions INTEGER CHECK (total_questions >= 0),

    -- Storage metadata
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'generated', 'exporting', 'exported', 'failed', 'deleted', 'archived')),
    file_count INTEGER NOT NULL DEFAULT 0 CHECK (file_count >= 0),
    total_file_size BIGINT NOT NULL DEFAULT 0 CHECK (total_file_size >= 0),

    -- Search and categorization
    tags TEXT[] NOT NULL DEFAULT '{}',
    search_content TEXT NOT NULL DEFAULT '',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- Create document_files table (file references)
CREATE TABLE public.document_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,

    -- File location and format
    file_key TEXT NOT NULL UNIQUE,
    file_format TEXT NOT NULL CHECK (file_format IN ('pdf', 'docx', 'html', 'txt', 'json', 'png', 'jpg', 'svg')),
    version TEXT NOT NULL CHECK (version IN ('student', 'teacher', 'combined')),

    -- File metadata
    file_size BIGINT NOT NULL DEFAULT 0 CHECK (file_size >= 0),
    content_type TEXT NOT NULL DEFAULT '',

    -- R2 storage metadata
    r2_metadata JSONB NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Create dev_documents table (development version)
CREATE TABLE public.dev_documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID, -- Optional reference, no foreign key constraint

    -- Document metadata (same as production)
    title TEXT NOT NULL CHECK (length(title) > 0),
    document_type TEXT NOT NULL CHECK (document_type IN ('worksheet', 'notes', 'textbook', 'slides')),
    detail_level TEXT CHECK (detail_level IN ('minimal', 'medium', 'comprehensive', 'guided')),
    topic TEXT,
    grade_level INTEGER CHECK (grade_level >= 1 AND grade_level <= 12),

    -- Generation metadata
    estimated_duration INTEGER CHECK (estimated_duration > 0),
    total_questions INTEGER CHECK (total_questions >= 0),

    -- Storage metadata
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'generating', 'generated', 'exporting', 'exported', 'failed', 'deleted', 'archived')),
    file_count INTEGER NOT NULL DEFAULT 0 CHECK (file_count >= 0),
    total_file_size BIGINT NOT NULL DEFAULT 0 CHECK (total_file_size >= 0),

    -- Search and categorization
    tags TEXT[] NOT NULL DEFAULT '{}',
    search_content TEXT NOT NULL DEFAULT '',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- Create dev_document_files table (development version)
CREATE TABLE public.dev_document_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES public.dev_documents(id) ON DELETE CASCADE,

    -- File location and format (same as production)
    file_key TEXT NOT NULL UNIQUE,
    file_format TEXT NOT NULL CHECK (file_format IN ('pdf', 'docx', 'html', 'txt', 'json', 'png', 'jpg', 'svg')),
    version TEXT NOT NULL CHECK (version IN ('student', 'teacher', 'combined')),

    -- File metadata
    file_size BIGINT NOT NULL DEFAULT 0 CHECK (file_size >= 0),
    content_type TEXT NOT NULL DEFAULT '',

    -- R2 storage metadata
    r2_metadata JSONB NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- ===============================================
-- INDEXES
-- ===============================================

-- Production table indexes
CREATE INDEX idx_documents_document_type ON public.documents(document_type);
CREATE INDEX idx_documents_status ON public.documents(status);
CREATE INDEX idx_documents_created_at ON public.documents(created_at);
CREATE INDEX idx_documents_tags ON public.documents USING gin(tags);

CREATE INDEX idx_document_files_document_id ON public.document_files(document_id);
CREATE INDEX idx_document_files_file_key ON public.document_files(file_key);

-- Dev table indexes
CREATE INDEX idx_dev_documents_document_type ON public.dev_documents(document_type);
CREATE INDEX idx_dev_documents_status ON public.dev_documents(status);
CREATE INDEX idx_dev_documents_created_at ON public.dev_documents(created_at);

CREATE INDEX idx_dev_document_files_document_id ON public.dev_document_files(document_id);

-- ===============================================
-- TRIGGERS
-- ===============================================

-- Update triggers (assumes update_updated_at_column function exists)
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON public.documents
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_document_files_updated_at
    BEFORE UPDATE ON public.document_files
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_dev_documents_updated_at
    BEFORE UPDATE ON public.dev_documents
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_dev_document_files_updated_at
    BEFORE UPDATE ON public.dev_document_files
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- ===============================================
-- ROW LEVEL SECURITY
-- ===============================================

ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dev_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dev_document_files ENABLE ROW LEVEL SECURITY;

-- Allow all operations for now
CREATE POLICY "Enable all operations" ON public.documents FOR ALL USING (true);
CREATE POLICY "Enable all operations" ON public.document_files FOR ALL USING (true);
CREATE POLICY "Enable all operations" ON public.dev_documents FOR ALL USING (true);
CREATE POLICY "Enable all operations" ON public.dev_document_files FOR ALL USING (true);

-- ===============================================
-- COMMENTS
-- ===============================================

COMMENT ON TABLE public.documents IS 'Document metadata and generation information';
COMMENT ON TABLE public.document_files IS 'File references for documents stored in R2';
COMMENT ON TABLE public.dev_documents IS 'Development version of documents table';
COMMENT ON TABLE public.dev_document_files IS 'Development version of document_files table';

-- Success message
SELECT 'Clean document tables created successfully!
Tables: documents, document_files, dev_documents, dev_document_files
All old inconsistent tables removed.' AS result;
