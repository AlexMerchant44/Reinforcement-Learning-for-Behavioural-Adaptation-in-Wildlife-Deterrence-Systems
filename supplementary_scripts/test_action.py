import sys
from pathlib import Path
import time
import numpy as np

# ----------------------------------------
# Add project root to Python path
# ----------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------
# Import your RL action logic
# ----------------------------------------
# TODO: change rl_actions to your actual filename
from rl_controller import choose_action, LEARNING_STATES, ACTION_OBJECTS, ACTIONS, Q, Epsilon


print("\n========================================")
print("   TESTING choose_action() WITH MOTOR   ")
print("========================================\n")

print("Learning states =", LEARNING_STATES)
print(f"Epsilon = {Epsilon}")
print()


# -----------------------------------------------------
# Test all states 0–11 one time each with real motor actions
# -----------------------------------------------------
print("=== SINGLE PASS THROUGH ALL STATES ===\n")

for state in range(12):
    print(f"State {state}:  (motor will run)")

    idx, action_obj = choose_action(state)

    # Print returned values
    print(f"  → Action index returned: {idx}")
    print(f"  → Action object returned: {action_obj}")
    print()

    # Short wait between tests so motor doesn't get hammered
    time.sleep(5)


# -----------------------------------------------------
# Test ε-greedy randomness for learning states
# -----------------------------------------------------
print("\n=== ε-GREEDY TEST FOR LEARNING STATES ===\n")

for state in LEARNING_STATES:
    print(f"\nState {state}:  (10 trials)")

    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for _ in range(10):
        idx, action_obj = choose_action(state)
        counts[idx] += 1
        time.sleep(0.5)   # allow motor to cool

    print("Action frequencies over 10 trials:")
    print(counts)


# -----------------------------------------------------
# Cleanup at the end
# -----------------------------------------------------
print("\nTest complete. Cleaning up GPIO…")

for a in ACTION_OBJECTS:
    try:
        a.cleanup()
    except:
        pass

print("✓ GPIO cleaned up.\n")
