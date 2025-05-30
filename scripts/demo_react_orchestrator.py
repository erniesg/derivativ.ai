#!/usr/bin/env python3
"""
Demo script for ReAct Multi-Agent Orchestrator.

This script demonstrates the complete ReAct-based question generation pipeline
using the smolagents framework with Manager and Specialist agents.
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services import create_react_orchestrator
from src.services.database_manager import DatabaseManager

from dotenv import load_dotenv
load_dotenv()


async def demo_react_workflow():
    """Demonstrate the complete ReAct workflow"""

    print("🚀 ReAct Multi-Agent Orchestrator Demo")
    print("=" * 50)
    print("This demonstrates the ReAct framework with:")
    print("📋 Manager Agent (CodeAgent) - Advanced reasoning and planning")
    print("🔧 Specialist Agents (ToolCallingAgent) - Focused tool execution")
    print("🔄 ReAct Pattern: Reasoning → Acting → Reasoning → Acting...")
    print()

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

    # Optional: Initialize database manager for full integration
    db_manager = None
    try:
        db_manager = DatabaseManager()
        print("✅ Database manager initialized")
    except Exception as e:
        print(f"⚠️ Database manager not available: {e}")
        print("   (Demo will continue without database integration)")

    # Create ReAct orchestrator
    print("\n🏗️ Creating ReAct Orchestrator...")
    orchestrator = create_react_orchestrator(
        manager_model_config=manager_config,
        specialist_model_config=specialist_config,
        database_manager=db_manager,
        auto_publish=False,  # Set to True to enable Payload CMS publishing
        debug=True
    )

    print("✅ ReAct Orchestrator initialized!")
    print(f"   Manager Agent: {orchestrator.manager_agent.name}")
    print(f"   Specialist Agents: {len(orchestrator.manager_agent.managed_agents)}")
    for i, agent in enumerate(orchestrator.manager_agent.managed_agents):
        # Handle both agent objects and names
        if hasattr(agent, 'name'):
            agent_name = agent.name
            tools_count = len(getattr(agent, 'tools', []))
        else:
            agent_name = str(agent)
            tools_count = "unknown"
        print(f"   - {agent_name} (tools: {tools_count})")

    # Also show our direct references
    print(f"   Generator Agent: {orchestrator.generator_agent.name}")
    print(f"   Reviewer Agent: {orchestrator.reviewer_agent.name}")

    # Demo 1: Simple ReAct demonstration
    print("\n" + "="*50)
    print("🎯 DEMO 1: ReAct Workflow Demonstration")
    print("="*50)

    try:
        demo_result = await orchestrator.demonstrate_react_workflow()

        print(f"Status: {demo_result['status']}")
        if demo_result['status'] == 'success':
            print("✅ ReAct demonstration completed successfully!")
            print(f"\n🧠 Manager Agent Reasoning & Actions:")
            print("-" * 40)
            print(demo_result['manager_result'][:500] + "...")
        else:
            print(f"❌ Demo failed: {demo_result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")

    # Demo 2: Full question generation with ReAct
    print("\n" + "="*50)
    print("🎯 DEMO 2: ReAct Question Generation Pipeline")
    print("="*50)

    try:
        # Define generation requirements
        requirements = {
            "topics": ["algebra", "number"],
            "difficulty": "foundation",
            "question_types": ["short_answer"],
            "syllabus_focus": "Cambridge IGCSE 0580",
            "assessment_objectives": ["AO1", "AO2"]
        }

        print("📋 Generation Requirements:")
        print(json.dumps(requirements, indent=2))

        # Run ReAct generation
        print("\n🚀 Starting ReAct generation session...")
        session = await orchestrator.generate_questions_with_react(
            config_id="foundation_algebra_gpt4o_mini",
            num_questions=1,
            requirements=requirements
        )

        print(f"\n✅ ReAct session completed!")
        print(f"Session ID: {session.session_id}")
        print(f"Status: {session.status}")
        print(f"Duration: {(session.end_time - session.start_time).total_seconds():.1f} seconds")

        # Display session summary
        summary = orchestrator.get_react_session_summary(session)
        print(f"\n📊 ReAct Session Summary:")
        print("-" * 30)
        print(f"Questions Generated: {summary['results']['questions_generated']}")
        print(f"Questions Approved: {summary['results']['questions_approved']}")
        print(f"Success Rate: {summary['results']['success_rate']:.1%}")
        print(f"Approval Rate: {summary['results']['approval_rate']:.1%}")
        print(f"Reasoning Steps: {summary['react_coordination']['reasoning_steps']}")
        print(f"Actions Taken: {summary['react_coordination']['actions_taken']}")

        # Display reasoning trail
        if session.reasoning_steps:
            print(f"\n🧠 ReAct Reasoning Trail:")
            print("-" * 30)
            for i, step in enumerate(session.reasoning_steps, 1):
                print(f"{i}. {step[:100]}...")

        # Display errors if any
        if session.errors:
            print(f"\n❌ Errors Encountered:")
            print("-" * 20)
            for error in session.errors:
                print(f"• {error}")

    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Demo 3: Integration capabilities
    print("\n" + "="*50)
    print("🎯 DEMO 3: Integration Capabilities")
    print("="*50)

    print("🔗 Available Integrations:")
    print(f"   Database Manager: {'✅ Connected' if db_manager else '❌ Not connected'}")
    print(f"   Auto-Publish: {'✅ Enabled' if orchestrator.auto_publish else '❌ Disabled'}")
    print(f"   Quality Workflow: {'✅ Available' if hasattr(orchestrator, 'quality_workflow') else '❌ Not available'}")

    print("\n🏭 Production Features:")
    print("   ✅ ReAct reasoning with audit trail")
    print("   ✅ Multi-agent coordination")
    print("   ✅ Quality control integration")
    print("   ✅ Payload CMS auto-publishing")
    print("   ✅ Database persistence")
    print("   ✅ Error handling and recovery")
    print("   ✅ Session tracking and metrics")

    print("\n" + "="*50)
    print("🎉 ReAct Multi-Agent Orchestrator Demo Complete!")
    print("="*50)
    print("The system is ready for production use with:")
    print("🧠 Advanced reasoning capabilities")
    print("🤝 Multi-agent coordination")
    print("🔄 ReAct pattern implementation")
    print("📊 Complete audit trails")
    print("🚀 Auto-publishing to Payload CMS")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        sys.exit(1)

    # Run the demo
    try:
        asyncio.run(demo_react_workflow())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
