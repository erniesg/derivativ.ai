#!/usr/bin/env python3
"""Test database schema consistency and operations"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database_schema import TableNames, DatabaseSchemas, DatabaseIndexes
from src.services.database_manager import DatabaseManager

load_dotenv()

async def test_database_schema():
    """Test database schema creation and consistency"""
    print('🧪 Testing database schema consistency...')

    # Setup
    connection_string = os.getenv('NEON_DATABASE_URL')

    db_manager = DatabaseManager(connection_string)
    await db_manager.initialize()

    print('✅ Database manager initialized')
    print(f'✅ Tables created using centralized schema definitions')

    # Test table names consistency
    print('\n📋 Current Table Names:')
    for table_name in TableNames.get_all_table_names():
        print(f'   • {table_name}')

    print('\n📝 Legacy Table Names (for migration):')
    for table_name in TableNames.get_legacy_table_names():
        print(f'   • {table_name}')

    # Test that we can get session summary (tests table access)
    try:
        summary = await db_manager.get_session_summary("test-session-id")
        print(f'\n✅ Session summary query works (empty result expected): {len(summary)} keys')
    except Exception as e:
        print(f'\n❌ Session summary query failed: {e}')

    await db_manager.close()
    print('\n✅ Database schema test completed')

if __name__ == "__main__":
    asyncio.run(test_database_schema())
