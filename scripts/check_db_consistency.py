#!/usr/bin/env python3
"""
Database Consistency Checker
============================

This script checks for table name consistency across the codebase
and validates that all database operations use the centralized table definitions.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database_schema import TableNames, get_table_migration_map

def scan_file_for_table_names(file_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """Scan a file for database table names and return line numbers"""
    table_occurrences = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # All possible table names to check for
        all_table_names = (
            TableNames.get_all_table_names() +
            TableNames.get_legacy_table_names()
        )

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Check for SQL patterns with table names
            for table_name in all_table_names:
                patterns = [
                    rf'\bFROM\s+{re.escape(table_name)}\b',
                    rf'\bINSERT\s+INTO\s+{re.escape(table_name)}\b',
                    rf'\bUPDATE\s+{re.escape(table_name)}\b',
                    rf'\bDELETE\s+FROM\s+{re.escape(table_name)}\b',
                    rf'\bCREATE\s+TABLE.*{re.escape(table_name)}\b',
                    rf'\bREFERENCES\s+{re.escape(table_name)}\b',
                    rf'["\']({re.escape(table_name)})["\']',  # String literals
                ]

                for pattern in patterns:
                    if re.search(pattern, line_lower):
                        if table_name not in table_occurrences:
                            table_occurrences[table_name] = []
                        table_occurrences[table_name].append((line_num, line.strip()))

    except (UnicodeDecodeError, FileNotFoundError):
        pass  # Skip files that can't be read

    return table_occurrences

def check_database_consistency():
    """Check database table name consistency across the codebase"""
    print("🔍 Database Table Consistency Check")
    print("=" * 50)

    # Get project root
    project_root = Path(__file__).parent.parent

    # Directories to scan
    scan_dirs = [
        project_root / "src",
        project_root / "tests",
        project_root / "scripts"
    ]

    # Collect all findings
    all_findings = {}
    files_scanned = 0

    for scan_dir in scan_dirs:
        if scan_dir.exists():
            for py_file in scan_dir.rglob("*.py"):
                files_scanned += 1
                findings = scan_file_for_table_names(str(py_file))
                if findings:
                    relative_path = py_file.relative_to(project_root)
                    all_findings[str(relative_path)] = findings

    print(f"📁 Scanned {files_scanned} Python files")
    print()

    # Analyze findings
    current_tables = set(TableNames.get_all_table_names())
    legacy_tables = set(TableNames.get_legacy_table_names())
    migration_map = get_table_migration_map()

    # Track usage
    current_table_usage = {}
    legacy_table_usage = {}

    for file_path, tables_found in all_findings.items():
        for table_name, occurrences in tables_found.items():
            if table_name in current_tables:
                if table_name not in current_table_usage:
                    current_table_usage[table_name] = []
                current_table_usage[table_name].append((file_path, len(occurrences)))
            elif table_name in legacy_tables:
                if table_name not in legacy_table_usage:
                    legacy_table_usage[table_name] = []
                legacy_table_usage[table_name].append((file_path, len(occurrences)))

    # Report current table usage
    print("✅ Current Table Usage (deriv_ prefixed):")
    if current_table_usage:
        for table_name in sorted(current_table_usage.keys()):
            usage_list = current_table_usage[table_name]
            print(f"   • {table_name}:")
            for file_path, count in usage_list:
                print(f"     - {file_path} ({count} occurrences)")
    else:
        print("   No current table usage found")
    print()

    # Report legacy table usage (needs migration)
    print("⚠️  Legacy Table Usage (needs migration):")
    if legacy_table_usage:
        for table_name in sorted(legacy_table_usage.keys()):
            usage_list = legacy_table_usage[table_name]
            current_equivalent = migration_map.get(table_name, "Unknown")
            print(f"   • {table_name} → should be {current_equivalent}:")
            for file_path, count in usage_list:
                print(f"     - {file_path} ({count} occurrences)")
    else:
        print("   ✅ No legacy table usage found!")
    print()

    # Detailed breakdown for migration
    if legacy_table_usage:
        print("🔧 Migration Required:")
        print("Files that need updating:")
        files_needing_migration = set()
        for table_name, usage_list in legacy_table_usage.items():
            for file_path, count in usage_list:
                files_needing_migration.add(file_path)

        for file_path in sorted(files_needing_migration):
            print(f"   📝 {file_path}")
            # Show specific occurrences
            if file_path in all_findings:
                for table_name, occurrences in all_findings[file_path].items():
                    if table_name in legacy_tables:
                        current_equivalent = migration_map.get(table_name, "Unknown")
                        print(f"      Replace {table_name} → {current_equivalent}")
                        for line_num, line_content in occurrences[:3]:  # Show first 3
                            print(f"         Line {line_num}: {line_content[:60]}...")
        print()

    # Summary
    print("📊 Summary:")
    print(f"   • Files scanned: {files_scanned}")
    print(f"   • Files with database tables: {len(all_findings)}")
    print(f"   • Current tables in use: {len(current_table_usage)}")
    print(f"   • Legacy tables found: {len(legacy_table_usage)}")

    if legacy_table_usage:
        print(f"   ⚠️  Migration needed for {len(set(f for usage_list in legacy_table_usage.values() for f, _ in usage_list))} files")
        return False
    else:
        print("   ✅ All table references are consistent!")
        return True

def show_table_definitions():
    """Show current table definitions"""
    print("\n📋 Current Table Definitions:")
    print("-" * 30)

    current_tables = TableNames.get_all_table_names()
    for table_name in current_tables:
        print(f"   • {table_name}")

    print(f"\n📝 Legacy Tables (deprecated):")
    legacy_tables = TableNames.get_legacy_table_names()
    migration_map = get_table_migration_map()
    for legacy_table in legacy_tables:
        current_table = migration_map.get(legacy_table, "Unknown")
        print(f"   • {legacy_table} → {current_table}")

if __name__ == "__main__":
    print("🚀 Starting Database Consistency Check...\n")

    # Show table definitions first
    show_table_definitions()
    print()

    # Run consistency check
    is_consistent = check_database_consistency()

    if is_consistent:
        print("\n🎉 Database table names are consistent!")
        sys.exit(0)
    else:
        print("\n❌ Database table inconsistencies found - migration needed")
        sys.exit(1)
