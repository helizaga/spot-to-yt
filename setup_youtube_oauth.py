#!/usr/bin/env python3
"""
YouTube Music OAuth Setup Script

This script helps you set up OAuth authentication for YouTube Music.
It will guide you through the process and generate an oauth.json file.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_requirements():
    """Check if ytmusicapi is installed"""
    try:
        import ytmusicapi  # pylint: disable=import-outside-toplevel

        print(f"✅ ytmusicapi version {ytmusicapi.__version__} is installed")
        return True
    except ImportError:
        print("❌ ytmusicapi is not installed")
        print("Please run: pip3 install -r requirements.txt")
        return False


def setup_oauth():
    """Run the OAuth setup process"""

    # Get credentials from environment or prompt user
    client_id = os.getenv("YTM_CLIENT_ID")
    client_secret = os.getenv("YTM_CLIENT_SECRET")
    oauth_path = os.getenv("YTM_OAUTH_PATH", "oauth.json")

    print("\n🎵 YouTube Music OAuth Setup")
    print("=" * 50)

    if not client_id or not client_secret:
        print("\n⚠️  YouTube Music credentials not found in .env file")
        print("\nYou need to obtain OAuth credentials from Google Cloud Console:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select an existing one")
        print("3. Enable the YouTube Data API v3")
        print("4. Create OAuth 2.0 credentials:")
        print("   - Application type: TVs and Limited Input devices")
        print("5. Copy the Client ID and Client Secret")
        print("\nYou can either:")
        print("a) Add them to your .env file as YTM_CLIENT_ID and YTM_CLIENT_SECRET")
        print("b) Enter them here (they will be used for this session only)")

        use_manual = input("\nEnter credentials now? (y/n): ").strip().lower()
        if use_manual in ["y", "yes"]:
            client_id = input("Enter YouTube Client ID: ").strip()
            client_secret = input("Enter YouTube Client Secret: ").strip()
        else:
            print(
                "\nPlease add the credentials to your .env file and run this script again"
            )
            return False

    if not client_id or not client_secret:
        print("❌ Client ID and Client Secret are required")
        return False

    # Check if oauth.json already exists
    if Path(oauth_path).exists():
        overwrite = (
            input(f"\n⚠️  {oauth_path} already exists. Overwrite? (y/n): ")
            .strip()
            .lower()
        )
        if overwrite not in ["y", "yes"]:
            print("Setup cancelled")
            return False

    print("\n🔐 Setting up OAuth with provided credentials...")
    print(f"   Output file: {oauth_path}")

    # Run the ytmusicapi oauth command
    cmd = [
        "ytmusicapi",
        "oauth",
        "--file",
        oauth_path,
        "--client-id",
        client_id,
        "--client-secret",
        client_secret,
    ]

    try:
        print("\n📱 Starting OAuth flow...")
        print(
            "Note: With TVs and Limited Input devices flow, "
            "you may receive a URL and code to enter."
        )
        print("Follow the instructions to authorize the application.\n")

        result = subprocess.run(cmd, capture_output=False, text=True, check=False)

        if result.returncode == 0:
            if Path(oauth_path).exists():
                print(f"\n✅ OAuth setup successful! Token saved to {oauth_path}")
                print("\nYou can now run the main script:")
                print("  python3 spotify_to_ytmusic_like_optimal.py")
                return True
            print(f"❌ OAuth file {oauth_path} was not created")
            return False
        print(f"❌ OAuth setup failed with exit code {result.returncode}")
        return False

    except FileNotFoundError:
        print("\n❌ ytmusicapi command not found")
        print("\nTry running the command directly:")
        print(f"  ytmusicapi oauth --file {oauth_path} \\ ")
        print("      --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET")
        return False
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        print(f"\n❌ Error during OAuth setup: {e}")
        return False


def main():
    """Main function"""
    print("🎵 YouTube Music OAuth Setup Helper")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Run OAuth setup
    success = setup_oauth()

    if success:
        print("\n✨ Setup complete! You're ready to transfer your Spotify likes.")
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
