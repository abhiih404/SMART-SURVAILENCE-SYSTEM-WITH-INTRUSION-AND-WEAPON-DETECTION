"""
setup_telegram.py
──────────────────────────────────────────────────────────────────────────────
Helps you configure the Telegram bot and verify the connection.
Run once before starting the surveillance system.
──────────────────────────────────────────────────────────────────────────────
"""
import asyncio
import sys


async def setup():
    token = input("Paste your Telegram Bot Token: ").strip()
    if not token:
        print("Token cannot be empty.")
        return

    try:
        from telegram import Bot
        bot = Bot(token=token)
        me  = await bot.get_me()
        print(f"\n✅ Bot connected: @{me.username} ({me.first_name})")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("Make sure the token is correct (use @BotFather on Telegram).")
        return

    print("\nTo get your Chat ID:")
    print("  1. Send any message to your bot on Telegram.")
    print("  2. Open: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
    print("  3. Look for 'chat' → 'id' in the JSON response.")
    chat_id = input("\nPaste your Chat ID: ").strip()

    # Send test message
    try:
        from telegram import Bot
        bot = Bot(token=token)
        await bot.send_message(
            chat_id=chat_id,
            text="✅ Smart Surveillance System connected successfully!\nYou will receive alerts here.",
        )
        print("\n✅ Test message sent! Check your Telegram.")
    except Exception as e:
        print(f"\n❌ Could not send test message: {e}")
        return

    # Write to .env
    env_path = ".env"
    lines = []
    try:
        with open(env_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        pass

    def _update(lines, key, value):
        for i, line in enumerate(lines):
            if line.startswith(key + "="):
                lines[i] = f"{key}={value}\n"
                return lines
        lines.append(f"{key}={value}\n")
        return lines

    lines = _update(lines, "TELEGRAM_BOT_TOKEN", token)
    lines = _update(lines, "TELEGRAM_CHAT_ID",   chat_id)

    with open(env_path, "w") as f:
        f.writelines(lines)

    print(f"\n✅ Credentials saved to {env_path}")
    print("You can now run: python main.py")


if __name__ == "__main__":
    asyncio.run(setup())
