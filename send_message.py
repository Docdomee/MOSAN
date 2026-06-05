import sys
import os
import argparse
import time

# Anchor all file operations to the directory of this script (the project root).
# This ensures user_message.txt is always written to the same location that
# agent.py reads from, regardless of the CWD when this script is invoked.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def send_message(message: str) -> None:
    """Writes a message to user_message.txt in the project root."""
    filename = os.path.join(_SCRIPT_DIR, "user_message.txt")
    temp_filename = os.path.join(_SCRIPT_DIR, "user_message.tmp")

    # Write to a temp file first to avoid a race condition where the agent reads
    # a partially-written file. On Linux this is atomic; on Windows it is close enough.
    try:
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(message)
    except OSError as e:
        print(f"Error writing temp message file: {e}")
        return

    try:
        os.replace(temp_filename, filename)
        print(f"Message sent: '{message}'")
        print(f"File written to: {filename}")
        print(f"The agent will process this in the next cycle (if manual_chat_mode is enabled).")
    except OSError as e:
        print(f"Error finalising message file: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a message to the running Agent.")
    parser.add_argument("message", type=str, help="The message content to send.")
    args = parser.parse_args()

    send_message(args.message)
