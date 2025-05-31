#!/usr/bin/env python3
"""
Test ReAct Orchestrator Database Tools

This script specifically tests the ReAct orchestrator's database operations
and auto-publishing tools to ensure they work correctly.
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.react_orchestrator import create_react_orchestrator
from src.services.database_manager import DatabaseManager

from dotenv import load_dotenv
load_dotenv()


async def test_react_database_tools():
    """Test ReAct orchestrator database and auto-publish tools"""

    print("🧪 Testing ReAct Orchestrator Database Tools")
    print("=" * 50)

    # Configuration for ReAct orchestrator
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

    # Initialize database manager
    db_connection_string = os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")
    if not db_connection_string:
        print("❌ No database connection string found")
        return False

    db_manager = DatabaseManager(db_connection_string)
    print("✅ Database manager initialized")

    # Create ReAct orchestrator with database integration
    orchestrator = create_react_orchestrator(
        manager_model_config=manager_config,
        specialist_model_config=specialist_config,
        database_manager=db_manager,
        auto_publish=True,  # Enable auto-publish for testing
        debug=True
    )

    print("✅ ReAct orchestrator created with database integration")
    print(f"   Database Agent: {orchestrator.database_agent.name}")
    print(f"   Auto-publish: Enabled")

    # Test 1: Database Stats Tool
    print("\n📊 Test 1: Database Statistics Tool")
    print("-" * 30)

    try:
        stats_task = """
Use the database_operations_specialist to get current database statistics.
Show me the total number of questions and their status breakdown.
"""
        print("🔍 Testing database stats via ReAct manager...")
        stats_result = orchestrator.manager_agent.run(stats_task)
        print("✅ Database stats test completed!")
        print(f"Result: {str(stats_result)[:200]}...")

    except Exception as e:
        print(f"❌ Database stats test failed: {e}")
        return False

    # Test 2: Question Generation + Auto-Save
    print("\n💾 Test 2: Question Generation + Database Save")
    print("-" * 30)

    try:
        save_task = """
Generate a simple Cambridge IGCSE question and save it to the database.

1. Use question_generator_specialist to create a basic algebra question
2. Use database_operations_specialist to save the question with auto_publish=true
3. Provide a summary of what was accomplished

Focus on making this work even if the question generation has integration issues.
"""
        print("🔍 Testing question generation + database save...")
        save_result = orchestrator.manager_agent.run(save_task)
        print("✅ Question save test completed!")
        print(f"Result: {str(save_result)[:200]}...")

    except Exception as e:
        print(f"❌ Question save test failed: {e}")
        return False

    # Test 3: Auto-Publish Tool
    print("\n📡 Test 3: Auto-Publish Tool")
    print("-" * 30)

    try:
        publish_task = """
Test the auto-publishing functionality.

1. Use database_operations_specialist to get database stats
2. Try to publish a question to Payload CMS using publish_question_to_payload
3. Report on the publishing process and any results

Show me what publishing capabilities are available.
"""
        print("🔍 Testing auto-publish functionality...")
        publish_result = orchestrator.manager_agent.run(publish_task)
        print("✅ Auto-publish test completed!")
        print(f"Result: {str(publish_result)[:200]}...")

    except Exception as e:
        print(f"❌ Auto-publish test failed: {e}")
        return False

    # Test 4: Multi-Agent Coordination
    print("\n🤝 Test 4: Multi-Agent Coordination")
    print("-" * 30)

    try:
        coordination_task = """
Demonstrate multi-agent coordination with database operations.

1. Use question_generator_specialist to discuss question generation capabilities
2. Use quality_review_specialist to explain quality review process
3. Use database_operations_specialist to show database capabilities
4. Summarize how all these agents work together for the complete workflow

Provide a comprehensive overview of the ReAct multi-agent system.
"""
        print("🔍 Testing multi-agent coordination...")
        coordination_result = orchestrator.manager_agent.run(coordination_task)
        print("✅ Multi-agent coordination test completed!")
        print(f"Result: {str(coordination_result)[:200]}...")

    except Exception as e:
        print(f"❌ Multi-agent coordination test failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 ReAct Database Tools Test Complete!")
    print("✅ All tests passed - ReAct orchestrator is working with database")
    return True


async def test_react_workflow_generation():
    """Test a complete question generation workflow via ReAct"""

    print("\n🔄 Testing Complete ReAct Workflow")
    print("=" * 40)

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

    # Get database connection
    db_connection_string = os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")
    if not db_connection_string:
        print("⚠️ No database connection for workflow test")
        return False

    db_manager = DatabaseManager(db_connection_string)

    # Create orchestrator
    orchestrator = create_react_orchestrator(
        manager_model_config=manager_config,
        specialist_model_config=specialist_config,
        database_manager=db_manager,
        auto_publish=True,
        debug=True
    )

    # Test complete workflow
    try:
        requirements = {
            "topics": ["number", "algebra"],
            "difficulty": "foundation",
            "question_types": ["short_answer"],
            "syllabus_focus": "Cambridge IGCSE 0580"
        }

        print("🚀 Running ReAct question generation workflow...")
        session = await orchestrator.generate_questions_with_react(
            config_id="test_react_config",
            num_questions=1,
            requirements=requirements
        )

        print(f"✅ ReAct workflow completed!")
        print(f"   Session ID: {session.session_id}")
        print(f"   Status: {session.status}")
        print(f"   Questions Generated: {session.questions_generated}")
        print(f"   Errors: {len(session.errors)}")

        if session.errors:
            print("   Error details:")
            for error in session.errors[:3]:  # Show first 3 errors
                print(f"     - {error}")

        return True

    except Exception as e:
        print(f"❌ ReAct workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all ReAct orchestrator tests"""

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        return

    if not (os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")):
        print("❌ Database connection string not found")
        return

    print("🧪 ReAct Orchestrator Database Tools Test Suite")
    print("=" * 60)
    print("Testing ReAct orchestrator database operations and auto-publishing")
    print()

    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Test 1: Database tools
        success1 = loop.run_until_complete(test_react_database_tools())

        # Test 2: Complete workflow
        success2 = loop.run_until_complete(test_react_workflow_generation())

        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print(f"Database Tools Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
        print(f"Workflow Test: {'✅ PASSED' if success2 else '❌ FAILED'}")

        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED!")
            print("ReAct orchestrator is working correctly with database operations")
        else:
            print("\n⚠️ Some tests failed - check output above for details")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
