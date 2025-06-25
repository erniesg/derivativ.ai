-- Migration: Create document storage tables
-- Date: 2025-06-25
-- Description: Create tables for document metadata and file storage

-- Create stored_documents table
CREATE TABLE IF NOT EXISTS public.stored_documents (
    -- Primary key
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    
    -- Foreign key to generation sessions
    session_id UUID REFERENCES public.generation_sessions(session_id),
    
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

-- Create document_files table
CREATE TABLE IF NOT EXISTS public.document_files (
    -- Primary key
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    
    -- Foreign key to stored documents
    document_id UUID NOT NULL REFERENCES public.stored_documents(id) ON DELETE CASCADE,
    
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

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_stored_documents_session_id ON public.stored_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_stored_documents_document_type ON public.stored_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_stored_documents_topic ON public.stored_documents(topic);
CREATE INDEX IF NOT EXISTS idx_stored_documents_grade_level ON public.stored_documents(grade_level);
CREATE INDEX IF NOT EXISTS idx_stored_documents_status ON public.stored_documents(status);
CREATE INDEX IF NOT EXISTS idx_stored_documents_created_at ON public.stored_documents(created_at);
CREATE INDEX IF NOT EXISTS idx_stored_documents_updated_at ON public.stored_documents(updated_at);

-- GIN indexes for array and JSONB columns
CREATE INDEX IF NOT EXISTS idx_stored_documents_tags ON public.stored_documents USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_stored_documents_search_content ON public.stored_documents USING gin(to_tsvector('english', search_content));

-- Indexes for document_files table
CREATE INDEX IF NOT EXISTS idx_document_files_document_id ON public.document_files(document_id);
CREATE INDEX IF NOT EXISTS idx_document_files_file_key ON public.document_files(file_key);
CREATE INDEX IF NOT EXISTS idx_document_files_version ON public.document_files(version);
CREATE INDEX IF NOT EXISTS idx_document_files_format ON public.document_files(file_format);
CREATE INDEX IF NOT EXISTS idx_document_files_created_at ON public.document_files(created_at);

-- GIN index for R2 metadata
CREATE INDEX IF NOT EXISTS idx_document_files_r2_metadata ON public.document_files USING gin(r2_metadata);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_stored_documents_type_status ON public.stored_documents(document_type, status);
CREATE INDEX IF NOT EXISTS idx_stored_documents_topic_grade ON public.stored_documents(topic, grade_level);
CREATE INDEX IF NOT EXISTS idx_stored_documents_status_created ON public.stored_documents(status, created_at);

-- Update triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_stored_documents_updated_at
    BEFORE UPDATE ON public.stored_documents
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_document_files_updated_at
    BEFORE UPDATE ON public.document_files
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Function to update file counts and sizes
CREATE OR REPLACE FUNCTION public.update_document_file_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Update file count and total size on insert
        UPDATE public.stored_documents 
        SET 
            file_count = file_count + 1,
            total_file_size = total_file_size + NEW.file_size,
            updated_at = NOW()
        WHERE id = NEW.document_id;
        
        RETURN NEW;
        
    ELSIF TG_OP = 'UPDATE' THEN
        -- Update total size on file size change
        IF OLD.file_size != NEW.file_size THEN
            UPDATE public.stored_documents 
            SET 
                total_file_size = total_file_size - OLD.file_size + NEW.file_size,
                updated_at = NOW()
            WHERE id = NEW.document_id;
        END IF;
        
        RETURN NEW;
        
    ELSIF TG_OP = 'DELETE' THEN
        -- Update file count and total size on delete
        UPDATE public.stored_documents 
        SET 
            file_count = file_count - 1,
            total_file_size = total_file_size - OLD.file_size,
            updated_at = NOW()
        WHERE id = OLD.document_id;
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_document_file_stats_trigger
    AFTER INSERT OR UPDATE OR DELETE ON public.document_files
    FOR EACH ROW EXECUTE FUNCTION public.update_document_file_stats();

-- Function to automatically update search content
CREATE OR REPLACE FUNCTION public.update_search_content()
RETURNS TRIGGER AS $$
BEGIN
    -- Generate search content from title, document_type, topic, and tags
    NEW.search_content = LOWER(
        COALESCE(NEW.title, '') || ' ' ||
        COALESCE(NEW.document_type, '') || ' ' ||
        COALESCE(NEW.topic, '') || ' ' ||
        COALESCE(array_to_string(NEW.tags, ' '), '')
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_search_content_trigger
    BEFORE INSERT OR UPDATE ON public.stored_documents
    FOR EACH ROW EXECUTE FUNCTION public.update_search_content();

-- Row Level Security (RLS) - enable but allow all for now
ALTER TABLE public.stored_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_files ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (can be refined later)
CREATE POLICY "Enable all operations for authenticated users" ON public.stored_documents
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Enable all operations for authenticated users" ON public.document_files
    FOR ALL USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

-- Create function for document statistics
CREATE OR REPLACE FUNCTION public.get_document_statistics()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_documents', COUNT(*),
        'total_file_size', COALESCE(SUM(total_file_size), 0),
        'documents_by_type', (
            SELECT json_object_agg(document_type, count)
            FROM (
                SELECT document_type, COUNT(*) as count
                FROM public.stored_documents
                WHERE status != 'deleted'
                GROUP BY document_type
            ) type_counts
        ),
        'documents_by_status', (
            SELECT json_object_agg(status, count)
            FROM (
                SELECT status, COUNT(*) as count
                FROM public.stored_documents
                WHERE status != 'deleted'
                GROUP BY status
            ) status_counts
        ),
        'average_file_size', CASE 
            WHEN COUNT(*) > 0 THEN COALESCE(SUM(total_file_size), 0) / COUNT(*)
            ELSE 0
        END,
        'largest_document_size', COALESCE(MAX(total_file_size), 0),
        'generation_success_rate', CASE
            WHEN COUNT(*) > 0 THEN 
                COUNT(CASE WHEN status IN ('exported', 'generated') THEN 1 END)::FLOAT / COUNT(*)
            ELSE 0
        END
    ) INTO result
    FROM public.stored_documents
    WHERE status != 'deleted';
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE public.stored_documents IS 'Document metadata and storage information';
COMMENT ON TABLE public.document_files IS 'File references and R2 storage metadata';

COMMENT ON COLUMN public.stored_documents.id IS 'Unique document identifier';
COMMENT ON COLUMN public.stored_documents.session_id IS 'Reference to generation session';
COMMENT ON COLUMN public.stored_documents.title IS 'Document title';
COMMENT ON COLUMN public.stored_documents.document_type IS 'Type of document (worksheet, notes, textbook, slides)';
COMMENT ON COLUMN public.stored_documents.search_content IS 'Generated searchable content for full-text search';
COMMENT ON COLUMN public.stored_documents.tags IS 'Array of tags for categorization';
COMMENT ON COLUMN public.stored_documents.file_count IS 'Number of associated files (maintained by trigger)';
COMMENT ON COLUMN public.stored_documents.total_file_size IS 'Total size of all files in bytes (maintained by trigger)';

COMMENT ON COLUMN public.document_files.id IS 'Unique file identifier';
COMMENT ON COLUMN public.document_files.document_id IS 'Parent document reference';
COMMENT ON COLUMN public.document_files.file_key IS 'Unique R2 storage key/path';
COMMENT ON COLUMN public.document_files.version IS 'Document version (student, teacher, combined)';
COMMENT ON COLUMN public.document_files.r2_metadata IS 'R2-specific storage metadata and upload information';