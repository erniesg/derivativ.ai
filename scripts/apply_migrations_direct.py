#!/usr/bin/env python3
"""
Direct migration application using Supabase Python client.
This bypasses the CLI and applies migrations directly via the API.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def apply_migrations():
    """Apply migrations directly via Supabase client."""
    try:
        from src.database.supabase_repository import get_supabase_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")

        if not url or not key:
            print("❌ Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
            return False

        print(f"🔗 Connecting to Supabase: {url}")
        client = get_supabase_client(url, key)

        migrations_dir = Path("supabase/migrations")
        migration_files = sorted(migrations_dir.glob("*.sql"))

        print(f"📋 Found {len(migration_files)} migration files")

        for migration_file in migration_files:
            print(f"\n🚀 Applying: {migration_file.name}")

            # Read migration content
            sql_content = migration_file.read_text()

            try:
                # Execute SQL via RPC function
                # Note: This uses a raw SQL execution approach
                result = client.rpc("exec_sql", {"sql": sql_content}).execute()
                print(f"✅ Successfully applied: {migration_file.name}")

            except Exception as e:
                error_msg = str(e)
                if "function exec_sql(sql) does not exist" in error_msg:
                    print("⚠️  Direct SQL execution not available via RPC")
                    print(f"📝 Please apply {migration_file.name} manually in the dashboard")
                    print(f"    File location: {migration_file}")
                    continue
                else:
                    print(f"❌ Error applying {migration_file.name}: {e}")
                    return False

        print("\n🎉 Migration process completed!")
        print("\nTo verify, run: python scripts/setup_supabase.py")
        return True

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


if __name__ == "__main__":
    success = apply_migrations()
    sys.exit(0 if success else 1)
