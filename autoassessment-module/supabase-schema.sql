CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TYPE user_role AS ENUM ('student', 'teacher', 'superadmin');
CREATE TYPE question_status AS ENUM ('draft', 'reviewed', 'approved', 'deprecated');
CREATE TYPE question_difficulty AS ENUM ('easy', 'medium', 'hard');
CREATE TYPE quiz_status AS ENUM ('active', 'completed', 'abandoned');

CREATE TABLE profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    school VARCHAR(100),
    role user_role DEFAULT 'student' NOT NULL,
    is_active BOOLEAN DEFAULT FALSE NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE topics (
    id SERIAL PRIMARY KEY,
    topic_name TEXT NOT NULL UNIQUE,
    description TEXT
)

CREATE TABLE questions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE NOT NULL,
    ao VARCHAR(3) NOT NULL,
    difficulty question_difficulty NOT NULL,
    stem TEXT NOT NULL,
    options JSONB, 
    correct_answer TEXT,
    rationale TEXT,    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status question_status DEFAULT 'draft' NOT NULL
);

-- Quiz sessions table to track individual quiz attempts
CREATE TABLE quiz_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    status quiz_status DEFAULT 'active' NOT NULL,
    difficulty_distribution JSONB NOT NULL DEFAULT '{"easy": 0.5, "medium": 0.3, "hard": 0.2}',
    questions_selected JSONB NOT NULL DEFAULT '[]',
    total_questions INTEGER DEFAULT 10,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Student topic performance with weighted moving average
CREATE TABLE student_topic_performance (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE NOT NULL,
    current_grade DECIMAL(3,2) DEFAULT 5.00 CHECK (current_grade >= 1.00 AND current_grade <= 9.00),
    wma_grade DECIMAL(3,2) DEFAULT 5.00 CHECK (wma_grade >= 1.00 AND wma_grade <= 9.00),
    total_attempts INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, topic_id)
);

-- Quiz responses to track individual question answers
CREATE TABLE quiz_responses (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    quiz_session_id UUID REFERENCES quiz_sessions(id) ON DELETE CASCADE NOT NULL,
    question_id UUID REFERENCES questions(id) ON DELETE CASCADE NOT NULL,
    user_answer TEXT,
    is_correct BOOLEAN NOT NULL,
    answered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(quiz_session_id, question_id) -- Ensure each question can only be answered once per session
);

-- Quiz results to store aggregated performance per session
CREATE TABLE quiz_results (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    quiz_session_id UUID REFERENCES quiz_sessions(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    topic_scores JSONB NOT NULL DEFAULT '{}', -- JSON object with topic_id: score
    overall_score DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    total_questions INTEGER NOT NULL DEFAULT 0,
    correct_answers INTEGER NOT NULL DEFAULT 0,
    completion_time_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- WMA history to track weighted moving average changes over time
CREATE TABLE wma_history (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE NOT NULL,
    quiz_session_id UUID REFERENCES quiz_sessions(id) ON DELETE CASCADE NOT NULL,
    previous_wma DECIMAL(3,2),
    new_wma DECIMAL(3,2) NOT NULL,
    performance_score DECIMAL(3,2) NOT NULL,
    weight_factor DECIMAL(3,2) DEFAULT 0.3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_topic_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE wma_history ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view their own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile" ON profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can delete their own profile" ON profiles
    FOR DELETE USING (auth.uid() = id);

-- Topics policies (readable by all authenticated users)
CREATE POLICY "Authenticated users can view topics" ON topics
    FOR SELECT USING (auth.uid() IS NOT NULL);

-- Questions policies (students can only view approved questions during quiz)
CREATE POLICY "Students can view approved questions during quiz" ON questions
    FOR SELECT USING (
        auth.uid() IS NOT NULL AND 
        status = 'approved'
    );

-- Quiz sessions policies
CREATE POLICY "Users can view their own quiz sessions" ON quiz_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own quiz sessions" ON quiz_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own quiz sessions" ON quiz_sessions
    FOR UPDATE USING (auth.uid() = user_id);

-- Student topic performance policies
CREATE POLICY "Users can view their own performance" ON student_topic_performance
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own performance" ON student_topic_performance
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own performance" ON student_topic_performance
    FOR UPDATE USING (auth.uid() = user_id);

-- Quiz responses policies
CREATE POLICY "Users can view their own quiz responses" ON quiz_responses
    FOR SELECT USING (
        auth.uid() IN (
            SELECT user_id FROM quiz_sessions WHERE id = quiz_session_id
        )
    );

CREATE POLICY "Users can insert their own quiz responses" ON quiz_responses
    FOR INSERT WITH CHECK (
        auth.uid() IN (
            SELECT user_id FROM quiz_sessions WHERE id = quiz_session_id
        )
    );

-- Quiz results policies
CREATE POLICY "Users can view their own quiz results" ON quiz_results
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own quiz results" ON quiz_results
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- WMA history policies
CREATE POLICY "Users can view their own WMA history" ON wma_history
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own WMA history" ON wma_history
    FOR INSERT WITH CHECK (auth.uid() = user_id);

