#!/usr/bin/env python3
"""
Simplified Supabase Setup Script

This script helps you set up Supabase by:
1. Checking your credentials
2. Testing connection
3. Providing migration SQL to run manually
4. Verifying the setup

Usage:
    python scripts/setup_supabase.py
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_environment():
    """Check if Supabase environment variables are set."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    print("🔍 Checking Supabase Configuration...")

    if not url:
        print("❌ SUPABASE_URL not set")
        print("   Add to .env: SUPABASE_URL=https://your-project.supabase.co")
        return False

    if not key:
        print("❌ SUPABASE_ANON_KEY not set")
        print("   Add to .env: SUPABASE_ANON_KEY=your-anon-key")
        return False

    print(f"✅ SUPABASE_URL: {url}")
    print(f"✅ SUPABASE_ANON_KEY: {key[:20]}...")
    return True


def test_connection():
    """Test basic Supabase connection."""
    try:
        from src.database.supabase_repository import get_supabase_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")

        print("\n🔗 Testing Supabase Connection...")
        client = get_supabase_client(url, key)

        # Try a simple query
        result = client.table("tiers").select("value").limit(1).execute()

        if hasattr(result, "data"):
            print("✅ Connection successful!")
            if result.data:
                print(f"✅ Found data in tiers table: {len(result.data)} rows")
                return True
            else:
                print("⚠️  Connection works but tiers table is empty (migrations needed)")
                return "needs_migration"
        else:
            print("❌ Connection failed - unexpected response format")
            return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        if 'relation "tiers" does not exist' in str(e):
            print("ℹ️  This means migrations haven't been applied yet")
            return "needs_migration"
        return False


def show_migration_instructions():
    """Show how to apply migrations."""
    print("\n📋 Database Setup Instructions:")
    print("=" * 50)

    migrations_dir = Path("supabase/migrations")
    migration_files = sorted(migrations_dir.glob("*.sql"))

    print("\n🗄️ You have these migration files:")
    for i, file in enumerate(migration_files, 1):
        print(f"  {i}. {file.name}")

    print("\n🚀 To apply migrations:")
    print("\nOption A: Supabase Dashboard (Recommended)")
    print("1. Go to your Supabase project dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Copy and run each SQL file in order:")

    for i, file in enumerate(migration_files, 1):
        print(f"\n   Step {i}: Run {file.name}")
        print(f"   File: {file}")

    print("\nOption B: Supabase CLI")
    print("1. Install CLI: npm install -g supabase")
    print("2. Login: supabase login")
    print("3. Link project: supabase link --project-ref YOUR_PROJECT_REF")
    print("4. Apply migrations: supabase db push")

    print("\nOption C: Use our automation script")
    print("   python scripts/run_migrations.py --method cli")


def explain_migrations():
    """Explain what each migration does."""
    print("\n📚 What the Migrations Do:")
    print("=" * 40)

    print("\n🗄️ Migration 1: questions table")
    print("   • Creates main questions storage table")
    print("   • Hybrid storage: fast queries + full data fidelity")
    print("   • Indexes for performance on tier, marks, quality score")
    print("   • Stores Cambridge IGCSE math questions with:")
    print("     - Question text, marks, tier (Core/Extended)")
    print("     - Command words (Calculate, Solve, Find, etc.)")
    print("     - Quality scores (0.0 to 1.0)")
    print("     - Complete question data as JSONB")

    print("\n📊 Migration 2: generation_sessions table")
    print("   • Tracks multi-agent question generation workflows")
    print("   • Stores session metadata (topic, tier, status)")
    print("   • Complete agent results and reasoning steps")
    print("   • Performance metrics and timing data")
    print("   • Status tracking: pending → in_progress → approved/rejected")

    print("\n📝 Migration 3: enum/reference tables")
    print("   • tiers: Core/Extended levels")
    print("   • command_words: Cambridge assessment terms")
    print("   • calculator_policies: Calculator usage rules")
    print("   • generation_statuses: Workflow states")
    print("   • question_origins: Source types (AI, past papers, etc.)")

    print("\n🔒 Security Features:")
    print("   • Row Level Security (RLS) enabled")
    print("   • Authenticated access policies")
    print("   • Automatic timestamps and triggers")


def verify_setup():
    """Verify that setup is working."""
    try:
        from src.database.supabase_repository import QuestionRepository, get_supabase_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")

        print("\n🧪 Verifying Setup...")
        client = get_supabase_client(url, key)
        repo = QuestionRepository(client)

        # Test basic operations
        questions = client.table("generated_questions").select("id").limit(1).execute()
        sessions = client.table("generation_sessions").select("id").limit(1).execute()
        tiers = client.table("tiers").select("*").execute()

        print("✅ All tables accessible:")
        print(f"   • generated_questions table: {len(questions.data)} rows")
        print(f"   • generation_sessions table: {len(sessions.data)} rows")
        print(f"   • tiers table: {len(tiers.data)} rows")

        if tiers.data:
            print(f"   • Available tiers: {[t['value'] for t in tiers.data]}")

        print("\n🎉 Supabase setup is working correctly!")
        return True

    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False


def main():
    """Main setup flow."""
    print("🚀 Derivativ Supabase Setup")
    print("=" * 30)

    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Please set up your Supabase credentials first!")
        print("   1. Create project at https://supabase.com")
        print("   2. Get URL and anon key from Settings → API")
        print("   3. Add to .env file:")
        print("      SUPABASE_URL=https://your-project.supabase.co")
        print("      SUPABASE_ANON_KEY=your-anon-key")
        return False

    # Step 2: Test connection
    connection_status = test_connection()

    if connection_status == "needs_migration":
        explain_migrations()
        show_migration_instructions()
        return True
    elif connection_status is True:
        print("✅ Connection working and migrations appear to be applied!")
        verify_setup()
        return True
    else:
        print("❌ Connection failed. Check your credentials and try again.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
