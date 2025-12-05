from pathlib import Path

MODE_FILE = Path("mode.txt")
ALLOWED = ["Scare_All", "Scare_Crows", "Scare_Magpies", "Scare_None"]

def get_mode():
    if not MODE_FILE.exists():
        return "Scare_All"
    mode = MODE_FILE.read_text().strip()
    return mode if mode in ALLOWED else "Scare_All"

def set_mode(mode: str):
    if mode not in ALLOWED:
        raise ValueError("Invalid mode")
    MODE_FILE.write_text(mode)
