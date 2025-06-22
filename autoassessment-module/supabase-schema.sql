CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TYPE user_role AS ENUM ('student', 'teacher', 'superadmin');
CREATE TYPE question_status AS ENUM ('draft', 'reviewed', 'approved', 'deprecated');
CREATE TYPE question_difficulty AS ENUM ('easy', 'medium', 'hard');

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
)

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE questions ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view their own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile" ON profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can delete their own profile" ON profiles
    FOR DELETE USING (auth.uid() = id);

CREATE POLICY "Users can not read questions" ON questions
    FOR SELECT USING (false);

CREATE POLICY "Users can not insert questions" ON questions
    FOR INSERT WITH CHECK (false);

CREATE POLICY "Users can not update questions" ON questions
    FOR UPDATE USING (false);

CREATE POLICY "Users can not delete questions" ON questions
    FOR DELETE USING (false);