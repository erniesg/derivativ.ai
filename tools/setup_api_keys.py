#!/usr/bin/env python3
"""
Interactive script to help set up API keys for live testing AND smolagents.
Guides users through getting API keys and setting up .env file.

Run from project root:
    python tools/setup_api_keys.py
"""

import os
from pathlib import Path


def create_env_file():  # noqa: PLR0915
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
    print("- Hugging Face: https://huggingface.co/settings/tokens (for smolagents)")
    print("- Supabase: https://supabase.com/dashboard/project/_/settings/api")
    print("\nPress Enter to skip any key you don't have yet or keep existing.\n")

    # Check existing keys
    existing_keys = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#") and line:
                    key, value = line.split("=", 1)
                    existing_keys[key] = value

    # Collect API keys (showing existing values)
    keys = {}

    current_openai = existing_keys.get("OPENAI_API_KEY", "")
    prompt_openai = f"OpenAI API Key (current: {'***' + current_openai[-4:] if current_openai else 'not set'}): "
    new_openai = input(prompt_openai).strip()
    keys["OPENAI_API_KEY"] = new_openai if new_openai else current_openai

    current_anthropic = existing_keys.get("ANTHROPIC_API_KEY", "")
    prompt_anthropic = f"Anthropic API Key (current: {'***' + current_anthropic[-4:] if current_anthropic else 'not set'}): "
    new_anthropic = input(prompt_anthropic).strip()
    keys["ANTHROPIC_API_KEY"] = new_anthropic if new_anthropic else current_anthropic

    current_google = existing_keys.get("GOOGLE_API_KEY", "")
    prompt_google = f"Google API Key (current: {'***' + current_google[-4:] if current_google else 'not set'}): "
    new_google = input(prompt_google).strip()
    keys["GOOGLE_API_KEY"] = new_google if new_google else current_google

    print("\n🤖 For smolagents model reasoning:")
    current_hf = existing_keys.get("HF_TOKEN", "")
    prompt_hf = (
        f"Hugging Face Token (current: {'***' + current_hf[-4:] if current_hf else 'not set'}): "
    )
    new_hf = input(prompt_hf).strip()
    keys["HF_TOKEN"] = new_hf if new_hf else current_hf

    print("\n🗄️ For database persistence:")
    current_supabase_url = existing_keys.get("SUPABASE_URL", "")
    prompt_supabase_url = f"Supabase URL (current: {'***' + current_supabase_url[-10:] if current_supabase_url else 'not set'}): "
    new_supabase_url = input(prompt_supabase_url).strip()
    keys["SUPABASE_URL"] = new_supabase_url if new_supabase_url else current_supabase_url

    current_supabase_key = existing_keys.get("SUPABASE_ANON_KEY", "")
    prompt_supabase_key = f"Supabase Anon Key (current: {'***' + current_supabase_key[-4:] if current_supabase_key else 'not set'}): "
    new_supabase_key = input(prompt_supabase_key).strip()
    keys["SUPABASE_ANON_KEY"] = new_supabase_key if new_supabase_key else current_supabase_key

    # Optional overrides
    print("\n⚙️  Optional configuration (press Enter for defaults):")

    temp = input("Default temperature (0.7): ").strip()
    if temp:
        keys["LLM_DEFAULT_TEMPERATURE"] = temp

    provider = input("Default provider (openai): ").strip()
    if provider and provider in ["openai", "anthropic", "google"]:
        keys["LLM_PROVIDERS_DEFAULT_PROVIDER"] = provider

    # Preserve other existing settings
    other_settings = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#") and line:
                    key, value = line.split("=", 1)
                    if key not in keys:
                        other_settings[key] = value

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
        ]
    )

    # Add other existing settings
    for key, value in other_settings.items():
        env_content.append(f"{key}={value}")

    # Add defaults if not present
    defaults = {
        "DEBUG": "false",
        "LOG_LEVEL": "INFO",
        "DEMO_MODE": "false",
        "USE_MOCK_LLM": "false",
    }

    for key, default_value in defaults.items():
        if key not in other_settings:
            env_content.append(f"{key}={default_value}")

    with open(env_file, "w") as f:
        f.write("\n".join(env_content))

    print(f"\n✅ Created {env_file}")

    # Count valid keys
    valid_llm_keys = sum(1 for k, v in keys.items() if v and k.endswith("_API_KEY"))
    has_hf_token = bool(keys.get("HF_TOKEN"))
    has_supabase = bool(keys.get("SUPABASE_URL") and keys.get("SUPABASE_ANON_KEY"))

    print("\n🎯 Setup Summary:")
    print(f"   LLM API keys: {valid_llm_keys}")
    print(f"   HF token: {'✅' if has_hf_token else '❌'}")
    print(f"   Supabase: {'✅' if has_supabase else '❌'}")

    if valid_llm_keys > 0 or has_hf_token:
        print("\n🚀 Ready to test! Try:")
        if valid_llm_keys > 0:
            print("   python examples/live_apis.py  # Test LLM APIs")
        if has_hf_token:
            print("   python examples/smolagents_interactive_demo.py  # Interactive smolagents")
        print("   python examples/smolagents_tools_demo.py  # Tools demo")
        if has_supabase:
            print("   python -m pytest tests/integration/  # Run integration tests with Supabase")
            print("   uvicorn src.api.main:app --reload  # Start FastAPI server with database")
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

    print("\n🤖 Hugging Face Inference API:")
    print("   Free tier: Rate limited")
    print("   Paid: Varies by model")
    print("   Used by smolagents for model reasoning")

    print("\n🎯 Recommendation: Start with Anthropic (cheapest) or Google (free tier)")
    print("🎯 For smolagents: Get free HF token for model reasoning")


def show_getting_started():
    """Show getting started instructions."""
    print("\n🚀 Getting Started with Live API Testing:")
    print("=" * 50)

    print("1. Get API keys (free credits available):")
    print("   • OpenAI: $5 free credit for new users")
    print("   • Anthropic: $5 free credit for new users")
    print("   • Google: Free tier with rate limits")

    print("\n2. Set up .env file:")
    print("   python tools/setup_api_keys.py")

    print("\n3. Test your setup:")
    print("   python examples/live_apis.py")
    print("   python examples/smolagents_interactive_demo.py")

    print("\n4. Use in development:")
    print("   • Agents will automatically use your API keys")
    print("   • Configuration system handles provider switching")
    print("   • Costs are tracked and estimated")
    print("   • Smolagents uses HF_TOKEN for model reasoning")

    print("\n5. For development without API keys:")
    print("   • Set USE_MOCK_LLM=true in .env")
    print("   • MockLLMService will generate realistic responses")
    print("   • Smolagents will work in tools-only mode without HF_TOKEN")


def _load_env_vars():
    """Load environment variables from .env file."""
    from pathlib import Path

    env_file = Path(".env")
    env_vars = {}

    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#") and line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
    return env_vars


def _check_api_availability(env_vars):
    """Check which APIs are available."""
    has_hf_token = bool(
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_API_TOKEN")
        or env_vars.get("HF_TOKEN")
        or env_vars.get("HUGGINGFACE_API_TOKEN")
    )
    has_llm_keys = any(
        [
            os.getenv("OPENAI_API_KEY") or env_vars.get("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY") or env_vars.get("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY") or env_vars.get("GOOGLE_API_KEY"),
        ]
    )

    available_apis = []
    if os.getenv("OPENAI_API_KEY") or env_vars.get("OPENAI_API_KEY"):
        available_apis.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY") or env_vars.get("ANTHROPIC_API_KEY"):
        available_apis.append("Anthropic")
    if os.getenv("GOOGLE_API_KEY") or env_vars.get("GOOGLE_API_KEY"):
        available_apis.append("Google")

    return has_hf_token, has_llm_keys, available_apis


def test_pandoc_setup():
    """Test if pandoc is properly installed and configured."""
    print("\n🧪 Testing pandoc setup...")
    print("=" * 40)

    try:
        import subprocess

        # Check if pandoc is installed
        result = subprocess.run(
            ["pandoc", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            print(f"✅ {version}")
        else:
            print("❌ Pandoc not found")
            print("   Install with: brew install pandoc")
            return False

        # Check if pdflatex is available (for PDF generation)
        result = subprocess.run(
            ["pdflatex", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            print("✅ pdflatex available (PDF generation supported)")
        else:
            print("⚠️  pdflatex not found (PDF generation will fail)")
            print("   Install with: brew install --cask mactex")

        print("🚀 Document generation ready!")
        print("   Supported formats: HTML, Markdown, DOCX, LaTeX")
        if result.returncode == 0:
            print("   PDF support: Available")
        else:
            print("   PDF support: Install pdflatex for PDF generation")

        return True

    except FileNotFoundError:
        print("❌ Pandoc not installed")
        print("   Install with: brew install pandoc")
        print("   For PDF support: brew install --cask mactex")
        return False
    except Exception as e:
        print(f"❌ Pandoc test failed: {e}")
        return False


def test_smolagents_setup():
    """Test if smolagents is properly configured."""
    print("\n🧪 Testing smolagents setup...")
    print("=" * 40)

    try:
        # Load environment variables
        env_vars = _load_env_vars()

        # Check if smolagents is installed
        try:
            import smolagents  # noqa: F401
        except ImportError:
            print("❌ smolagents not installed")
            print("   Install with: pip install smolagents")
            return False
        print("✅ smolagents installed")

        # Check API availability
        has_hf_token, has_llm_keys, available_apis = _check_api_availability(env_vars)

        print(f"✅ HF_TOKEN: {'Found' if has_hf_token else 'Missing'}")
        print(f"✅ LLM APIs: {'Found' if has_llm_keys else 'Missing'}")

        if has_llm_keys:
            print(f"   Available: {', '.join(available_apis)}")

        # Test smolagents integration
        from src.agents.smolagents_integration import create_derivativ_agent

        agent = create_derivativ_agent(agent_type="question_generator")
        print("✅ Derivativ smolagents agent created")

        if has_hf_token:
            print("🚀 Ready for full interactive smolagents experience!")
            print("   Run: python examples/smolagents_interactive_demo.py")
        else:
            print("⚠️  Tools-only mode (no model reasoning)")
            print("   Set HF_TOKEN for full interactive experience")
            print("   Run: python examples/smolagents_tools_demo.py")

        if has_llm_keys:
            print("✅ Our tools will use real LLM APIs")
        else:
            print("⚠️  Our tools will use mock responses")

        return True

    except ImportError as e:
        print(f"❌ smolagents not installed: {e}")
        print("   Install with: pip install smolagents")
        return False
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False


def main():
    """Main setup wizard."""
    print("🎯 Derivativ LLM API & Smolagents Setup Wizard")
    print("=" * 50)

    while True:
        print("\nWhat would you like to do?")
        print("1. Set up API keys (.env file)")
        print("2. Test smolagents setup")
        print("3. Test pandoc setup (document generation)")
        print("4. Show pricing information")
        print("5. Show getting started guide")
        print("6. Exit")

        choice = input("\nChoice (1-6): ").strip()

        if choice == "1":
            create_env_file()
        elif choice == "2":
            test_smolagents_setup()
        elif choice == "3":
            test_pandoc_setup()
        elif choice == "4":
            show_pricing_info()
        elif choice == "5":
            show_getting_started()
        elif choice == "6":
            print("👋 Happy coding!")
            break
        else:
            print("❌ Invalid choice, please try again.")


if __name__ == "__main__":
    main()
