#!/usr/bin/env python3
"""
Test ReAct Multi-Agent Orchestrator functionality.
Tests the complete ReAct-based question generation pipeline using smolagents.
"""

import asyncio
import os
import sys
import json

# Add project root to Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services import ReActMultiAgentOrchestrator, create_react_orchestrator
from smolagents import OpenAIServerModel, LiteLLMModel

from dotenv import load_dotenv
load_dotenv()


async def test_react_orchestrator_demo():
    """Test ReAct orchestrator demonstration workflow"""

    print("🧪 Testing ReAct Orchestrator Demo...")

    # Create models (using OpenAI for reliability)
    manager_model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    specialist_model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create ReAct orchestrator
    orchestrator = ReActMultiAgentOrchestrator(
        manager_model=manager_model,
        specialist_model=specialist_model,
        auto_publish=False,
        debug=True
    )

    try:
        # Run demonstration workflow
        print("\n🚀 Starting ReAct demonstration...")
        demo_result = await orchestrator.demonstrate_react_workflow()

        print(f"✅ Demonstration completed!")
        print(f"   Status: {demo_result['status']}")
        print(f"   Message: {demo_result['message']}")

        if demo_result['status'] == 'success':
            print(f"\n🧠 Manager Agent Result:")
            print(f"   {demo_result['manager_result'][:300]}...")

        return True

    except Exception as e:
        print(f"❌ Error testing ReAct demo: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_react_question_generation():
    """Test full ReAct question generation workflow"""

    print("\n🧪 Testing ReAct Question Generation...")

    # Create orchestrator using factory function
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

    orchestrator = create_react_orchestrator(
        manager_model_config=manager_config,
        specialist_model_config=specialist_config,
        auto_publish=False,
        debug=True
    )

    try:
        # Generate questions with specific requirements
        requirements = {
            "topics": ["algebra", "geometry"],
            "difficulty": "foundation",
            "question_types": ["short_answer", "structured"]
        }

        print("\n🚀 Starting ReAct generation session...")
        session = await orchestrator.generate_questions_with_react(
            config_id="mixed_review_gpt4o_mini",
            num_questions=2,
            requirements=requirements
        )

        print(f"✅ ReAct session completed!")
        print(f"   Session ID: {session.session_id}")
        print(f"   Status: {session.status}")
        print(f"   Questions Generated: {session.questions_generated}")
        print(f"   Questions Approved: {session.questions_approved}")
        print(f"   Questions Published: {session.questions_published}")

        # Display session summary
        summary = orchestrator.get_react_session_summary(session)
        print(f"\n📊 ReAct Session Summary:")
        print(json.dumps(summary, indent=2))

        # Display reasoning steps
        print(f"\n🧠 Reasoning Steps ({len(session.reasoning_steps)}):")
        for i, step in enumerate(session.reasoning_steps):
            print(f"   {i+1}. {step[:100]}...")

        # Display errors if any
        if session.errors:
            print(f"\n❌ Errors ({len(session.errors)}):")
            for error in session.errors:
                print(f"   - {error}")

        return True

    except Exception as e:
        print(f"❌ Error testing ReAct generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_react_tools():
    """Test the individual ReAct tools"""

    print("\n🧪 Testing ReAct Tools...")

    # Import tools for testing
    from src.services.react_orchestrator import (
        generate_igcse_question,
        review_question_quality,
        make_quality_decision,
        get_session_status
    )

    try:
        # Test question generation tool
        print("\n🎯 Testing question generation tool...")
        gen_result = generate_igcse_question(
            config_id="test_config",
            topic="algebra",
            difficulty="foundation",
            question_type="short_answer"
        )
        print(f"   Generation Result: {gen_result['status']} - {gen_result['message']}")
        assert gen_result['status'] == 'success'

        # Test review tool
        print("\n🔍 Testing review tool...")
        review_result = review_question_quality(
            question_id=gen_result['question_id'],
            detailed_analysis=True
        )
        print(f"   Review Result: {review_result['status']} - Score: {review_result['overall_score']}")
        assert review_result['status'] == 'success'

        # Test quality decision tool
        print("\n🚦 Testing quality decision tool...")
        decision_result = make_quality_decision(
            question_id=gen_result['question_id'],
            review_score=review_result['overall_score'],
            auto_publish=True
        )
        print(f"   Decision Result: {decision_result['status']} - {decision_result['decision']}")
        assert decision_result['status'] == 'success'

        # Test session status tool
        print("\n📊 Testing session status tool...")
        status_result = get_session_status("test_session_123")
        print(f"   Status Result: {status_result['status']} - Questions: {status_result['questions_generated']}")
        assert status_result['status'] == 'success'

        print("✅ All ReAct tools working correctly!")
        return True

    except Exception as e:
        print(f"❌ Error testing ReAct tools: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specialist_agents():
    """Test the specialist agent initialization"""

    print("\n🧪 Testing Specialist Agents...")

    try:
        # Create model
        model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Test QuestionGeneratorSpecialistAgent
        from src.services.react_orchestrator import QuestionGeneratorSpecialistAgent
        generator_agent = QuestionGeneratorSpecialistAgent(model)

        print(f"✅ Generator Agent initialized:")
        print(f"   Name: {generator_agent.name}")
        print(f"   Description: {generator_agent.description[:50]}...")
        print(f"   Tools: {len(generator_agent.tools)}")
        print(f"   Max Steps: {generator_agent.max_steps}")

        # Test QualityReviewSpecialistAgent
        from src.services.react_orchestrator import QualityReviewSpecialistAgent
        reviewer_agent = QualityReviewSpecialistAgent(model)

        print(f"✅ Reviewer Agent initialized:")
        print(f"   Name: {reviewer_agent.name}")
        print(f"   Description: {reviewer_agent.description[:50]}...")
        print(f"   Tools: {len(reviewer_agent.tools)}")
        print(f"   Max Steps: {reviewer_agent.max_steps}")

        return True

    except Exception as e:
        print(f"❌ Error testing specialist agents: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_react_manager_coordination():
    """Test Manager Agent coordination of specialist agents"""

    print("\n🧪 Testing Manager Agent Coordination...")

    try:
        # Create simple orchestrator
        manager_model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        specialist_model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        orchestrator = ReActMultiAgentOrchestrator(
            manager_model=manager_model,
            specialist_model=specialist_model,
            debug=True
        )

        # Test simple coordination task
        simple_task = """
Please coordinate with the specialist agents to:
1. Generate 1 test IGCSE question using config 'test_config'
2. Review the quality of that question
3. Make a quality decision
4. Report the results

Show your reasoning at each step.
"""

        print("\n🧠 Testing Manager Agent coordination...")
        result = orchestrator.manager_agent.run(simple_task)

        print(f"✅ Manager coordination completed!")
        print(f"   Result length: {len(result)} characters")
        print(f"   Result preview: {result[:200]}...")

        # Check that manager has access to specialist agents
        print(f"\n🤖 Manager Agent Configuration:")
        print(f"   Managed Agents: {len(orchestrator.manager_agent.managed_agents)}")
        for agent in orchestrator.manager_agent.managed_agents:
            print(f"   - {agent.name}: {len(agent.tools)} tools")

        return True

    except Exception as e:
        print(f"❌ Error testing manager coordination: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all ReAct orchestrator tests"""

    print("🚀 Starting ReAct Multi-Agent Orchestrator Tests\n")

    # Test results tracking
    results = {}

    # Test individual tools (synchronous)
    results["tools"] = test_react_tools()

    # Test specialist agents (synchronous)
    results["specialists"] = test_specialist_agents()

    # Test manager coordination (asynchronous)
    results["manager_coordination"] = await test_react_manager_coordination()

    # Test demonstration workflow (asynchronous)
    results["demo"] = await test_react_orchestrator_demo()

    # Test full generation workflow (asynchronous)
    results["full_generation"] = await test_react_question_generation()

    print(f"\n📊 ReAct Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All ReAct orchestrator tests passed!")
        print("\n💡 The ReAct multi-agent system is ready for:")
        print("   - Advanced reasoning with Manager Agent (CodeAgent)")
        print("   - Specialized tool execution with Specialist Agents (ToolCallingAgent)")
        print("   - Complete question generation → review → quality control workflow")
        print("   - Full audit trail of reasoning steps and actions")
        print("   - Integration with existing quality control and auto-publish systems")
    else:
        print("\n⚠️ Some ReAct tests failed. Check the logs above.")

    return all_passed


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    exit(0 if success else 1)
