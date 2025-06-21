#!/usr/bin/env python3
"""
Demo script showing Derivativ AI smolagents integration.
This demonstrates both our custom tools and native smolagents CodeAgent usage.
"""

import json
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.smolagents_integration import (  # noqa: E402
    DerivativSmolagents,
    create_derivativ_agent,
    generate_math_question,
    refine_question,
    review_question_quality,
)


def demo_direct_tools():
    """Demo: Use our tools directly (without smolagents agent)."""
    print("\n🔧 DEMO 1: Direct Tool Usage")
    print("=" * 50)

    # Generate a question
    print("1. Generating a math question...")
    question_result = generate_math_question(
        topic="algebra",
        grade_level=8,
        marks=3,
        calculator_policy="not_allowed",
        command_word="Calculate",
    )
    print(f"Question generated: {len(question_result)} characters")

    if question_result.startswith("Error"):
        print(f"Error: {question_result}")
        return None

    # Review the question
    print("\n2. Reviewing question quality...")
    review_result = review_question_quality(question_result)
    print(f"Review completed: {len(review_result)} characters")

    if review_result.startswith("Error"):
        print(f"Error: {review_result}")
        return None

    # Parse to check quality score
    try:
        review_data = json.loads(review_result)
        quality_score = review_data.get("quality_score", 0.5)
        print(f"Quality score: {quality_score}")

        # Refine if needed
        if quality_score < 0.8:
            print("\n3. Refining question (quality < 0.8)...")
            refined_result = refine_question(question_result, review_result)
            print(f"Refinement completed: {len(refined_result)} characters")
            return refined_result
        else:
            print("3. Question quality is good - no refinement needed!")
            return question_result

    except json.JSONDecodeError:
        print("Could not parse review result")
        return question_result


def demo_smolagents_question_generator():
    """Demo: Use smolagents CodeAgent for question generation."""
    print("\n🤖 DEMO 2: Smolagents Question Generator")
    print("=" * 50)

    try:
        # Create a question generator agent
        agent = create_derivativ_agent(agent_type="question_generator")
        print(f"Created agent: {agent.name}")
        print(f"Available tools: {list(agent.tools.keys())}")

        # Note: We can't actually run agent.run() without real API keys
        # But we can show the agent structure
        print("Agent ready for question generation!")
        print("To use: agent.run('Generate a geometry question for grade 9')")

        return agent

    except Exception as e:
        print(f"Error creating smolagents agent: {e}")
        return None


def demo_smolagents_quality_control():
    """Demo: Use smolagents CodeAgent for complete quality workflow."""
    print("\n🎯 DEMO 3: Smolagents Quality Control Agent")
    print("=" * 50)

    try:
        # Create a quality control agent
        agent = create_derivativ_agent(agent_type="quality_control")
        print(f"Created agent: {agent.name}")
        print(f"Available tools: {list(agent.tools.keys())}")

        # Show what this agent can do
        print("\nThis agent can:")
        print("- Generate mathematics questions")
        print("- Review question quality")
        print("- Refine questions based on feedback")
        print("- Coordinate complete quality workflows")

        print("\nExample usage:")
        print("agent.run('Generate an algebra question, review it, and refine if quality < 0.8')")

        return agent

    except Exception as e:
        print(f"Error creating quality control agent: {e}")
        return None


def demo_configuration():
    """Demo: Show configuration and model selection."""
    print("\n⚙️  DEMO 4: Configuration & Model Selection")
    print("=" * 50)

    # Show available models based on API keys
    derivativ = DerivativSmolagents()
    print(f"Selected model: {derivativ.model_id}")

    # Show custom model usage
    custom_derivativ = DerivativSmolagents(model_id="gpt-4o-mini")
    print(f"Custom model: {custom_derivativ.model_id}")

    # Show API key detection
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }

    available_keys = [key for key, value in api_keys.items() if value]
    print(f"\nAvailable API keys: {available_keys or 'None (using mock)'}")


def main():
    """Run all demos."""
    print("🚀 DERIVATIV AI - SMOLAGENTS INTEGRATION DEMO")
    print("=" * 60)

    # Demo 1: Direct tools
    question_result = demo_direct_tools()

    # Demo 2: Smolagents question generator
    question_agent = demo_smolagents_question_generator()

    # Demo 3: Smolagents quality control
    quality_agent = demo_smolagents_quality_control()

    # Demo 4: Configuration
    demo_configuration()

    print("\n✅ DEMO COMPLETE")
    print("=" * 60)
    print("Smolagents integration is working!")
    print("\nNext steps:")
    print("1. Add your API keys to environment variables")
    print("2. Use agent.run() with natural language instructions")
    print("3. Build complex multi-agent workflows")

    return {
        "question_result": question_result,
        "question_agent": question_agent,
        "quality_agent": quality_agent,
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
