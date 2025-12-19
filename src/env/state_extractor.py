from pathlib import Path

MODE_FILE = Path("mode.txt")
ALLOWED = ["Scare_All", "Scare_Crows", "Scare_Magpies", "Scare_None"]

STATE_TABLE = [
    ("Crow",   "Scare_All"),
    ("Crow",   "Scare_Crows"),
    ("Crow",   "Scare_Magpies"),
    ("Crow",   "Scare_None"),

    ("Magpie", "Scare_All"),
    ("Magpie", "Scare_Crows"),
    ("Magpie", "Scare_Magpies"),
    ("Magpie", "Scare_None"),

    ("None",   "Scare_All"),
    ("None",   "Scare_Crows"),
    ("None",   "Scare_Magpies"),
    ("None",   "Scare_None"),
]

def get_mode():
    '''
    Reads the mode from the mode.txt file
    '''
    if not MODE_FILE.exists():
        return "Scare_All"
    mode = MODE_FILE.read_text().strip()
    return mode if mode in ALLOWED else "Scare_All"

def get_state(species):
    """
    Returns the state index (0-11) based on detected species and current mode.

    species: "Crow", "Magpie", or "None"
    """

    mode = get_mode()

    for idx, (s, m) in enumerate(STATE_TABLE):
        if s == species and m == mode:
            return idx
    
    
