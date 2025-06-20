#!/usr/bin/env python3
"""
Interactive script to help set up API keys for live testing.
Guides users through getting API keys and setting up .env file.

Run from project root:
    python -m scripts.setup_api_keys
"""

from pathlib import Path


def create_env_file():
    """Create .env file with user input."""
    env_file = Path(".env")

    if env_file.exists():
        print("📁 .env file already exists!")
        response = input("Do you want to update it? (y/N): ").lower()
        if response != "y":
            return

    print("\n🔑 Setting up API keys for live testing...")
    print("You can get these keys from:")
    print("- OpenAI: https://platform.openai.com/api-keys")
    print("- Anthropic: https://console.anthropic.com/")
    print("- Google: https://makersuite.google.com/app/apikey")
    print("\nPress Enter to skip any key you don't have yet.\n")

    # Collect API keys
    keys = {}

    keys["OPENAI_API_KEY"] = input("OpenAI API Key (sk-...): ").strip()
    keys["ANTHROPIC_API_KEY"] = input("Anthropic API Key (sk-ant-...): ").strip()
    keys["GOOGLE_API_KEY"] = input("Google API Key: ").strip()

    # Optional overrides
    print("\n⚙️  Optional configuration (press Enter for defaults):")

    temp = input("Default temperature (0.7): ").strip()
    if temp:
        keys["LLM_DEFAULT_TEMPERATURE"] = temp

    provider = input("Default provider (openai): ").strip()
    if provider and provider in ["openai", "anthropic", "google"]:
        keys["LLM_PROVIDERS_DEFAULT_PROVIDER"] = provider

    # Write .env file
    env_content = []
    env_content.append("# LLM Provider API Keys")

    for key, value in keys.items():
        if value:
            env_content.append(f"{key}={value}")
        else:
            env_content.append(f"# {key}=your-key-here")

    env_content.extend(
        [
            "",
            "# Development Settings",
            "DEBUG=false",
            "LOG_LEVEL=INFO",
            "DEMO_MODE=false",
            "USE_MOCK_LLM=false",
        ]
    )

    with open(env_file, "w") as f:
        f.write("\n".join(env_content))

    print(f"\n✅ Created {env_file}")

    # Count valid keys
    valid_keys = sum(1 for k, v in keys.items() if v and k.endswith("_API_KEY"))

    if valid_keys > 0:
        print(f"🎯 Added {valid_keys} API key(s)")
        print("\n🚀 Ready to test! Run:")
        print("   python scripts/test_live_apis.py")
    else:
        print("\n⚠️  No API keys added. You can:")
        print("   1. Edit .env file manually")
        print("   2. Run this script again")
        print("   3. Use mock services for development")


def show_pricing_info():
    """Show pricing information for the cheapest models."""
    print("\n💰 Pricing for cheapest models we use:")
    print("=" * 50)

    print("🔵 OpenAI gpt-4o-mini:")
    print("   Input:  $0.000150 / 1K tokens")
    print("   Output: $0.000600 / 1K tokens")
    print("   ~$0.01 for 1000 questions")

    print("\n🟠 Anthropic claude-3-5-haiku:")
    print("   Input:  $0.80 / 1M tokens")
    print("   Output: $4.00 / 1M tokens")
    print("   ~$0.005 for 1000 questions")

    print("\n🔴 Google gemini-1.5-flash:")
    print("   Free tier: 15 requests/minute")
    print("   Paid: $0.075 / 1M input tokens")
    print("   $0.30 / 1M output tokens")

    print("\n🎯 Recommendation: Start with Anthropic (cheapest) or Google (free tier)")


def show_getting_started():
    """Show getting started instructions."""
    print("\n🚀 Getting Started with Live API Testing:")
    print("=" * 50)

    print("1. Get API keys (free credits available):")
    print("   • OpenAI: $5 free credit for new users")
    print("   • Anthropic: $5 free credit for new users")
    print("   • Google: Free tier with rate limits")

    print("\n2. Set up .env file:")
    print("   python scripts/setup_api_keys.py")

    print("\n3. Test your setup:")
    print("   python scripts/test_live_apis.py")

    print("\n4. Use in development:")
    print("   • Agents will automatically use your API keys")
    print("   • Configuration system handles provider switching")
    print("   • Costs are tracked and estimated")

    print("\n5. For development without API keys:")
    print("   • Set USE_MOCK_LLM=true in .env")
    print("   • MockLLMService will generate realistic responses")


def main():
    """Main setup wizard."""
    print("🎯 Derivativ LLM API Setup Wizard")
    print("=" * 40)

    while True:
        print("\nWhat would you like to do?")
        print("1. Set up API keys (.env file)")
        print("2. Show pricing information")
        print("3. Show getting started guide")
        print("4. Exit")

        choice = input("\nChoice (1-4): ").strip()

        if choice == "1":
            create_env_file()
        elif choice == "2":
            show_pricing_info()
        elif choice == "3":
            show_getting_started()
        elif choice == "4":
            print("👋 Happy coding!")
            break
        else:
            print("❌ Invalid choice, please try again.")


if __name__ == "__main__":
    main()
