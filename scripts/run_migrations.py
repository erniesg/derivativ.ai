#!/usr/bin/env python3
"""
Automated Supabase Migration Runner

This script automatically applies all database migrations in the correct order.
It can run migrations via:
1. Direct database connection (requires supabase library)
2. Supabase CLI (if installed)
3. Raw SQL execution via psycopg2

Usage:
    python scripts/run_migrations.py
    python scripts/run_migrations.py --dry-run
    python scripts/run_migrations.py --method cli
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)8s] %(message)s")
logger = logging.getLogger(__name__)


class MigrationRunner:
    """Automated migration runner for Supabase database."""

    def __init__(self, migrations_dir: str = "supabase/migrations"):
        self.migrations_dir = Path(migrations_dir)
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not self.migrations_dir.exists():
            raise FileNotFoundError(f"Migrations directory not found: {self.migrations_dir}")

    def get_migration_files(self) -> list[Path]:
        """Get all migration files in numerical order."""
        migration_files = list(self.migrations_dir.glob("*.sql"))
        migration_files.sort()  # Natural sorting by filename

        logger.info(f"Found {len(migration_files)} migration files:")
        for file in migration_files:
            logger.info(f"  - {file.name}")

        return migration_files

    def validate_environment(self) -> bool:
        """Validate that required environment variables are set."""
        if not self.supabase_url:
            logger.error("SUPABASE_URL environment variable not set")
            return False

        if not self.supabase_key:
            logger.error("SUPABASE_ANON_KEY environment variable not set")
            return False

        logger.info(f"Using Supabase URL: {self.supabase_url}")
        return True

    def run_via_supabase_client(self, migration_files: list[Path], dry_run: bool = False) -> bool:
        """Run migrations using the supabase-py client."""
        try:
            from supabase import create_client
        except ImportError:
            logger.error("supabase library not installed. Install with: pip install supabase")
            return False

        if not self.validate_environment():
            return False

        logger.info("Running migrations via Supabase client...")

        client = create_client(self.supabase_url, self.supabase_key)

        for migration_file in migration_files:
            logger.info(f"Applying migration: {migration_file.name}")

            if dry_run:
                logger.info(f"DRY RUN: Would execute {migration_file.name}")
                continue

            try:
                # Read migration content
                sql_content = migration_file.read_text()

                # Execute migration via RPC call
                # Note: This requires a custom RPC function in Supabase for executing raw SQL
                # For now, we'll use the rpc method if available
                result = client.rpc("execute_sql", {"sql": sql_content}).execute()

                if result.data:
                    logger.info(f"✅ Migration {migration_file.name} applied successfully")
                else:
                    logger.error(f"❌ Migration {migration_file.name} failed")
                    return False

            except Exception as e:
                logger.error(f"❌ Error applying migration {migration_file.name}: {e}")
                return False

        logger.info("🎉 All migrations completed successfully!")
        return True

    def run_via_supabase_cli(self, migration_files: list[Path], dry_run: bool = False) -> bool:
        """Run migrations using Supabase CLI."""
        import subprocess

        logger.info("Running migrations via Supabase CLI...")

        # Check if Supabase CLI is installed
        try:
            result = subprocess.run(
                ["supabase", "--version"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.error("Supabase CLI not found. Install with: npm install -g supabase")
                return False
            logger.info(f"Using Supabase CLI: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Supabase CLI not found. Install with: npm install -g supabase")
            return False

        if dry_run:
            logger.info("DRY RUN: Would run migrations via CLI")
            for migration_file in migration_files:
                logger.info(f"DRY RUN: Would apply {migration_file.name}")
            return True

        try:
            # Run migrations using supabase db push
            logger.info("Pushing migrations to Supabase...")
            result = subprocess.run(
                ["supabase", "db", "push"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                logger.info("✅ Migrations applied successfully via CLI")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ CLI migration failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Error running CLI migrations: {e}")
            return False

    def run_via_direct_sql(self, migration_files: list[Path], dry_run: bool = False) -> bool:
        """Run migrations by executing SQL directly via psycopg2."""
        try:
            import importlib.util
            from urllib.parse import urlparse

            # Check if psycopg2 is available without importing it
            if importlib.util.find_spec("psycopg2") is None:
                raise ImportError("psycopg2 not available")
        except ImportError:
            logger.error(
                "psycopg2 library not installed. Install with: pip install psycopg2-binary"
            )
            return False

        if not self.validate_environment():
            return False

        # Parse Supabase URL to connection string
        parsed = urlparse(self.supabase_url)
        connection_string = f"postgresql://postgres:[PASSWORD]@{parsed.hostname}:5432/postgres"

        logger.warning("Direct SQL method requires database password (not anon key)")
        logger.warning("You'll need to use your database password from Supabase project settings")

        if dry_run:
            logger.info("DRY RUN: Would connect to database and execute migrations")
            for migration_file in migration_files:
                logger.info(f"DRY RUN: Would execute {migration_file.name}")
            return True

        # This method requires manual password input for security
        logger.error("Direct SQL method not implemented. Use CLI method instead.")
        return False

    def run_migrations(self, method: str = "client", dry_run: bool = False) -> bool:
        """Run all migrations using the specified method."""
        migration_files = self.get_migration_files()

        if not migration_files:
            logger.warning("No migration files found")
            return True

        if dry_run:
            logger.info("🧪 DRY RUN MODE - No changes will be made")

        if method == "client":
            return self.run_via_supabase_client(migration_files, dry_run)
        elif method == "cli":
            return self.run_via_supabase_cli(migration_files, dry_run)
        elif method == "direct":
            return self.run_via_direct_sql(migration_files, dry_run)
        else:
            logger.error(f"Unknown method: {method}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Supabase database migrations")
    parser.add_argument(
        "--method",
        choices=["client", "cli", "direct"],
        default="cli",
        help="Migration method to use (default: cli)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--migrations-dir",
        default="supabase/migrations",
        help="Directory containing migration files",
    )

    args = parser.parse_args()

    try:
        runner = MigrationRunner(args.migrations_dir)
        success = runner.run_migrations(args.method, args.dry_run)

        if success:
            logger.info("🎉 Migration process completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Migration process failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Migration runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
