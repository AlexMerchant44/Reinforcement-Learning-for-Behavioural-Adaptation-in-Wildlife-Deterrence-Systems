import sys
import time
import random
from pathlib import Path

import numpy as np

# ----------------------------------------
# Add project root to Python path
# ----------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------
# Import your RL controller
# ----------------------------------------
import rl_controller


def main():
    print("\n========================================")
    print("     TESTING rl_controller MODULE       ")
    print("========================================\n")

    print(f"Epsilon = {rl_controller.Epsilon}")
    print(f"Alpha   = {rl_controller.Alpha}")
    print("\nInitial Q-table (in memory):")
    print(rl_controller.Q)

    # States you want to test (learning + non-learning)
    test_states = [0, 1, 4, 6, 2, 8]

    print("\n=== RUNNING ACTION + Q-UPDATES ===\n")

    for state_before in test_states:
        print(f"\n--- Testing from state {state_before} ---")
        for trial in range(3):
            print(f" Trial {trial+1}")

            # 1) Choose and execute an action (this runs the motor)
            action_idx, motor = rl_controller.choose_action(state_before)

            # 2) Simulate a next state (for now, just random valid state)
            state_after = random.randrange(rl_controller.NUM_STATES)

            # 3) Update Q-table based on this transition
            reward = rl_controller.update_q(state_before, state_after, action_idx)

            print(f"   state_before = {state_before}")
            print(f"   action_idx   = {action_idx}")
            print(f"   state_after  = {state_after}")
            print(f"   reward       = {reward:.4f}")
            print(f"   Q[{state_before}] after update: {rl_controller.Q[state_before]}")

            time.sleep(0.5)  # small delay so the motor isn't hammered

    # ----------------------------------------
    # Reload Q-table from disk and show it
    # ----------------------------------------
    print("\n========================================")
    print("     Q-TABLE LOADED FROM q_table.npy    ")
    print("========================================\n")

    q_from_disk = np.load(rl_controller.Q_PATH)
    print(q_from_disk)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
