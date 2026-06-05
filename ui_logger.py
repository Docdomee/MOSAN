# ui_logger.py
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama to work and reset color after each print
init(autoreset=True)

def log(message: str):
    """
    Logs a message to the console with a timestamp and color coding.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Logic for terminal coloring
    color = Fore.WHITE  # Default color
    if "ERROR" in message or "CRITICAL" in message:
        color = Fore.RED
    elif "WARNING" in message:
        color = Fore.YELLOW
    elif "SUCCESS" in message or "completato" in message:
        color = Fore.GREEN
    elif "[APP]" in message or "[State Manager]" in message:
        color = Fore.CYAN
    elif "[AGENT]" in message or "[PARSER]" in message:
        color = Fore.MAGENTA
    elif "--- EXECUTING TOOL" in message or "--- TOOL RESULT" in message:
        color = Fore.BLUE + Style.BRIGHT

    full_message_colored = f"{Style.DIM}[{timestamp}]{Style.RESET_ALL} {color}{message}"

    # Print to the terminal (with colors) — encode safely for Windows consoles
    try:
        print(full_message_colored, flush=True)
    except UnicodeEncodeError:
        safe_message = full_message_colored.encode("ascii", errors="replace").decode("ascii")
        print(safe_message, flush=True)

def init_log():
    """
    Placeholder for compatibility. Does nothing in headless mode.
    """
    log("UI Logger initialized for headless mode.")