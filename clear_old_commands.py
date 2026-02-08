#!/usr/bin/env python3
"""
Simple script to clear old bot commands and set new ones.
Run this script and paste your bot token when prompted.
"""

import requests
import sys

def get_current_commands(token):
    """Get currently registered commands"""
    url = f"https://api.telegram.org/bot{token}/getMyCommands"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['ok']:
            return data['result']
    return []

def delete_all_commands(token):
    """Delete all bot commands"""
    url = f"https://api.telegram.org/bot{token}/deleteMyCommands"
    response = requests.post(url)
    
    if response.status_code == 200:
        data = response.json()
        return data['ok']
    return False

def set_new_commands(token):
    """Set new bot commands"""
    # Define your current bot commands
    commands = [
        {"command": "start", "description": "Start the bot and see welcome message"},
        {"command": "provider", "description": "Switch AI provider (groq/gemini/openrouter)"},
        {"command": "models", "description": "List available models for current provider"},
        {"command": "model", "description": "Switch to a specific model"},
        {"command": "clear", "description": "Clear conversation history"},
        {"command": "help", "description": "Show help message"},
    ]
    
    url = f"https://api.telegram.org/bot{token}/setMyCommands"
    payload = {"commands": commands}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        return data['ok'], commands
    return False, commands

def main():
    print("🤖 Telegram Bot Command Manager")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Show your current bot commands")
    print("  2. Delete ALL old commands (including /skills, /whoami, etc.)")
    print("  3. Set only the new commands for your chat bot")
    print("\n" + "=" * 60)
    
    # Get bot token
    token = input("\n📝 Enter your Telegram Bot Token: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Show current commands
    print("\n📋 Current registered commands:")
    current = get_current_commands(token)
    if current:
        for cmd in current:
            print(f"  /{cmd['command']} - {cmd['description']}")
    else:
        print("  (No commands registered)")
    
    print("\n" + "=" * 60)
    input("\n⚠️  Press ENTER to delete ALL old commands and set new ones...")
    
    # Delete all commands
    print("\n🗑️  Deleting ALL old commands...")
    if delete_all_commands(token):
        print("✅ Successfully deleted all old commands!")
    else:
        print("❌ Failed to delete commands. Check your bot token.")
        sys.exit(1)
    
    # Set new commands
    print("\n✨ Setting new commands...")
    success, commands = set_new_commands(token)
    
    if success:
        print("\n✅ Successfully set new commands:")
        for cmd in commands:
            print(f"  /{cmd['command']} - {cmd['description']}")
        
        print("\n" + "=" * 60)
        print("\n🎉 Done! Your bot commands have been updated.")
        print("\n💡 Tips:")
        print("  • Restart your Telegram app to see changes immediately")
        print("  • Type / in your bot chat to see the new command list")
        print("  • Old commands like /skills and /whoami are now removed")
    else:
        print("\n❌ Failed to set new commands. Check your bot token.")
        sys.exit(1)

if __name__ == "__main__":
    main()
