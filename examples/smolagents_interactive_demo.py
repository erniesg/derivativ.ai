#!/usr/bin/env python3
"""
Interactive Smolagents Demo - Actually runs the agents!
This shows the real smolagents interface in action.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.smolagents_integration import (  # noqa: E402
    create_derivativ_agent,
)


def check_api_keys():
    """Check what API keys are available."""
    # Check both our LLM keys and HF token for smolagents
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")

    available = [key for key, value in api_keys.items() if value]

    print("🔑 API Key Status:")
    for key, value in api_keys.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}")

    hf_status = "✅" if hf_token else "❌"
    print(f"  {hf_status} HF_TOKEN (for smolagents model reasoning)")

    print("\n📝 Current Setup:")
    if available:
        print(f"  ✅ Our tools use real LLMs: {', '.join(available)}")
    else:
        print("  ❌ Our tools will use mocks (no LLM API keys)")

    if hf_token:
        print("  ✅ Smolagents can use model reasoning")
    else:
        print("  ⚠️  Smolagents will use tools directly (no model reasoning)")
        print("     To enable model reasoning: export HF_TOKEN='your_huggingface_token'")

    return len(available) > 0 or hf_token


def demo_interactive_question_generation():
    """Actually run a smolagents agent interactively!"""
    print("\n🤖 INTERACTIVE SMOLAGENTS DEMO")
    print("=" * 50)

    # Create the agent
    print("Creating smolagents question generator...")
    agent = create_derivativ_agent(agent_type="question_generator")

    print(f"✅ Agent created: {agent.name}")
    print(f"Available tools: {list(agent.tools.keys())}")

    # This is the key part - actually RUN the agent!
    print("\n🚀 Running agent with natural language prompt...")

    prompt = """Generate a Cambridge IGCSE Mathematics question about algebra
    for grade 8 students, worth 4 marks, without calculator allowed.
    Use the command word 'Calculate' and make sure it's engaging."""

    print(f"Prompt: {prompt}")
    print("\n" + "=" * 50)

    try:
        # This will show the actual smolagents interface in action!
        result = agent.run(prompt)

        print("🎉 AGENT RESULT:")
        print(result)

        return result

    except Exception as e:
        print(f"❌ Error running agent: {e}")
        print("This might be due to missing API keys or model access issues.")
        return None


def demo_interactive_quality_workflow():
    """Run the full quality control workflow interactively."""
    print("\n🎯 INTERACTIVE QUALITY CONTROL WORKFLOW")
    print("=" * 50)

    # Create quality control agent
    print("Creating smolagents quality control agent...")
    agent = create_derivativ_agent(agent_type="quality_control")

    print(f"✅ Agent created: {agent.name}")
    print(f"Available tools: {list(agent.tools.keys())}")

    # Run with a complex workflow prompt
    prompt = """I need you to:
    1. Generate a geometry question about triangles for grade 9, worth 5 marks
    2. Review the quality of the question you generated
    3. If the quality score is below 0.8, refine the question to improve it
    4. Give me the final question with quality assessment

    Make sure the question is engaging and follows Cambridge IGCSE standards."""

    print(f"\nPrompt: {prompt}")
    print("\n" + "=" * 50)

    try:
        # This shows the multi-agent workflow in action!
        result = agent.run(prompt)

        print("🎉 WORKFLOW RESULT:")
        print(result)

        return result

    except Exception as e:
        print(f"❌ Error running workflow: {e}")
        print("This might be due to missing API keys or model access issues.")
        return None


def demo_multi_agent_system():
    """Run the full multi-agent system."""
    print("\n🌟 MULTI-AGENT SYSTEM DEMO")
    print("=" * 50)

    # Create multi-agent system with base tools
    print("Creating comprehensive multi-agent system...")
    agent = create_derivativ_agent(agent_type="multi_agent")

    print(f"✅ Agent created: {agent.name}")
    print(f"Available tools: {list(agent.tools.keys())}")

    # Run with a comprehensive prompt
    prompt = """I'm a teacher preparing for an exam. Help me create a high-quality
    mathematics question about probability for grade 8 students. The question should:
    - Be worth 3 marks
    - Not require a calculator
    - Use real-world context (like dice, cards, or sports)
    - Have clear marking criteria

    Please generate the question, review its quality, and if needed, improve it.
    Then provide me with the final question and marking scheme."""

    print(f"\nPrompt: {prompt}")
    print("\n" + "=" * 50)

    try:
        # This shows the full system with web search and other tools!
        result = agent.run(prompt)

        print("🎉 SYSTEM RESULT:")
        print(result)

        return result

    except Exception as e:
        print(f"❌ Error running system: {e}")
        print("This might be due to missing API keys or model access issues.")
        return None


def main():
    """Run the interactive smolagents demonstration."""
    print("🚀 DERIVATIV AI - INTERACTIVE SMOLAGENTS DEMO")
    print("=" * 60)

    # Check API key status
    has_api_keys = check_api_keys()

    if not has_api_keys:
        print("\n🤔 Continue with mock responses? (y/n): ", end="")
        choice = input().lower().strip()
        if choice != "y":
            print("Set your API keys and try again!")
            return

    print("\n" + "=" * 60)

    # Demo 1: Simple question generation
    result1 = demo_interactive_question_generation()

    if result1:
        print("\n" + "=" * 60)
        # Demo 2: Quality workflow
        result2 = demo_interactive_quality_workflow()

        if result2:
            print("\n" + "=" * 60)
            # Demo 3: Full multi-agent system
            result3 = demo_multi_agent_system()

    print("\n✅ INTERACTIVE DEMO COMPLETE!")
    print("=" * 60)
    print("You just saw smolagents in action! 🎉")
    print("\nWhat happened:")
    print("1. Agents received natural language instructions")
    print("2. They used our custom tools (generate_math_question, etc.)")
    print("3. They reasoned about the task and executed workflows")
    print("4. They provided structured responses")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎯 Try running with your own API keys for real LLM responses!")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
