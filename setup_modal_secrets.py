#!/usr/bin/env python3
"""
Setup script for Modal secrets configuration.
This script helps you create Modal secrets from your .env file.
"""

import os
import subprocess
import sys


def load_env_file(env_path: str = ".env") -> dict:
    """Load environment variables from .env file."""
    env_vars = {}

    if not os.path.exists(env_path):
        print(f"❌ .env file not found at {env_path}")
        print("Please create a .env file with your API keys first.")
        return env_vars

    with open(env_path) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('"\'')
                env_vars[key] = value

    return env_vars


def check_modal_cli():
    """Check if Modal CLI is installed and authenticated."""
    try:
        result = subprocess.run(['modal', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal CLI found: {result.stdout.strip()}")
        else:
            print("❌ Modal CLI not found. Please install it:")
            print("pip install modal")
            return False
    except FileNotFoundError:
        print("❌ Modal CLI not found. Please install it:")
        print("pip install modal")
        return False

    # Check if authenticated
    try:
        result = subprocess.run(['modal', 'profile', 'current'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal authenticated: {result.stdout.strip()}")
            return True
        else:
            print("❌ Modal not authenticated. Please run:")
            print("modal setup")
            return False
    except Exception as e:
        print(f"❌ Error checking Modal auth: {e}")
        return False


def create_modal_secret(env_vars: dict):
    """Create Modal secret from environment variables."""
    if not env_vars:
        print("❌ No environment variables found")
        return False

    print("\n🔐 Creating Modal secret 'derivativ-secrets'...")

    # Filter out empty values and format for Modal
    filtered_vars = {k: v for k, v in env_vars.items() if v and v.strip()}

    if not filtered_vars:
        print("❌ No valid environment variables found")
        return False

    print(f"📋 Found {len(filtered_vars)} environment variables:")
    for key in filtered_vars.keys():
        # Mask sensitive values
        if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'token', 'password']):
            print(f"  • {key}: ***masked***")
        else:
            print(f"  • {key}: {filtered_vars[key]}")

    # Create the secret using Modal CLI
    try:
        # Prepare the command
        cmd = ['modal', 'secret', 'create', 'derivativ-secrets']

        # Add each environment variable
        for key, value in filtered_vars.items():
            cmd.extend([f'{key}={value}'])

        print(f"\n🚀 Creating secret with {len(filtered_vars)} variables...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Modal secret 'derivativ-secrets' created successfully!")
            return True
        else:
            print("❌ Failed to create Modal secret:")
            print(f"Error: {result.stderr}")

            # Check if secret already exists
            if "already exists" in result.stderr.lower():
                print("\n🔄 Secret already exists. Updating instead...")

                # Try to update the secret
                update_cmd = ['modal', 'secret', 'delete', 'derivativ-secrets']
                subprocess.run(update_cmd, capture_output=True)

                # Recreate
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Modal secret updated successfully!")
                    return True
                else:
                    print(f"❌ Failed to update secret: {result.stderr}")

            return False

    except Exception as e:
        print(f"❌ Error creating Modal secret: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Derivativ AI - Modal Secrets Setup")
    print("=" * 50)

    # Check Modal CLI
    if not check_modal_cli():
        print("\n❌ Please install and authenticate Modal CLI first:")
        print("1. pip install modal")
        print("2. modal setup")
        sys.exit(1)

    # Load environment variables
    print("\n📁 Loading environment variables...")
    env_vars = load_env_file()

    if not env_vars:
        print("\n💡 Create a .env file with your API keys:")
        print("OPENAI_API_KEY=your-openai-key")
        print("ANTHROPIC_API_KEY=your-anthropic-key")
        print("GOOGLE_API_KEY=your-google-key")
        print("SUPABASE_URL=your-supabase-url")
        print("SUPABASE_KEY=your-supabase-key")
        print("DEMO_MODE=true")
        sys.exit(1)

    # Create Modal secret
    success = create_modal_secret(env_vars)

    if success:
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("1. Deploy your app: modal deploy deploy.py")
        print("2. Test deployment: python deploy.py")
        print("3. View in Modal dashboard: https://modal.com/apps")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
