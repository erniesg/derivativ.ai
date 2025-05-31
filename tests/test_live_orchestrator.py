#!/usr/bin/env python3
"""
Test script for ReAct Orchestrator with LIVE agent integration.

This tests the orchestrator with real database and agents to generate actual questions.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services import create_react_orchestrator
from src.services.database_manager import DatabaseManager

load_dotenv()


async def test_live_orchestrator():
    """Test orchestrator with live database integration"""

    print("🧪 Testing ReAct Orchestrator with LIVE Integration")
    print("="*60)

    # Configuration
    manager_config = {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }

    specialist_config = {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }

    # Initialize database with Neon PostgreSQL connection string
    connection_string = os.getenv("NEON_DATABASE_URL")
    if not connection_string:
        print("❌ Error: NEON_DATABASE_URL environment variable is required")
        return

    print(f"📄 Database: Neon PostgreSQL (connected)")

    try:
        db_manager = DatabaseManager(connection_string)
        await db_manager.initialize()
        print("✅ Database connected successfully")

        # Create orchestrator with LIVE integration
        orchestrator = create_react_orchestrator(
            manager_model_config=manager_config,
            specialist_model_config=specialist_config,
            database_manager=db_manager,
            auto_publish=False,  # Set True for Payload publishing
            debug=True
        )

        print(f"\n🔗 Integration Status:")
        print(f"   Real Agents: {'✅ LIVE' if hasattr(orchestrator, 'real_generator_agent') else '❌ Mock'}")
        print(f"   Database: {'✅ Connected' if db_manager else '❌ Not connected'}")
        print(f"   Quality Workflow: {'✅ Available' if hasattr(orchestrator, 'quality_workflow') else '❌ Not available'}")

        # Test live question generation
        print(f"\n🚀 Testing LIVE Question Generation...")

        session = await orchestrator.generate_questions_with_react(
            config_id="algebra_claude4",
            num_questions=1,
            requirements={
                "topics": ["algebra"],
                "difficulty": "foundation",
                "question_types": ["short_answer"],
                "syllabus_focus": "Cambridge IGCSE 0580"
            }
        )

        summary = orchestrator.get_react_session_summary(session)

        print(f"\n📊 LIVE Test Results:")
        print(f"   Session ID: {session.session_id}")
        print(f"   Status: {session.status}")
        print(f"   Questions Generated: {summary['results']['questions_generated']}")
        print(f"   Success Rate: {summary['results']['success_rate']:.1%}")
        print(f"   Duration: {summary.get('duration_seconds', 0):.1f}s")

        if session.errors:
            print(f"\n❌ Errors:")
            for error in session.errors:
                print(f"   • {error}")
        else:
            print(f"\n✅ No errors - LIVE integration successful!")

        await db_manager.close()

    except Exception as e:
        print(f"❌ Live test failed: {e}")
        print("💡 To run live test, ensure:")
        print("   1. NEON_DATABASE_URL environment variable is set")
        print("   2. Database schema is initialized")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY required")
        sys.exit(1)

    try:
        asyncio.run(test_live_orchestrator())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
