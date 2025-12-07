import os
import random
import numpy as np
from action import MotorAction

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

Epsilon = 0.3
Alpha = 0.1

NUM_STATES = 12     
NUM_ACTIONS = 4      

DATA_DIR = "rl_data"
os.makedirs(DATA_DIR, exist_ok=True)

Q_PATH = os.path.join(DATA_DIR, "q_table.npy")

if os.path.exists(Q_PATH):
    Q = np.load(Q_PATH)
else:
    Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)
    np.save(Q_PATH, Q)


ACTION_PARAMS = [
    (0.0, 0.0),   # a0: no movement
    (0.1, 0.5),   # a1
    (0.1, 1.0),   # a2
    (1.0, 0.5),   # a3
]

motor = MotorAction(gpio_pin=4)

ACTIONS = [
    lambda: motor.run(0.0,   0),     # 0 seconds, 0% (effectively off)
    lambda: motor.run(0.1,  50),     # use 50% duty in code (0.5 for energy model)
    lambda: motor.run(0.1, 100),     # 100%
    lambda: motor.run(1.0,  50),     # 1s at 50%
]

NUM_ACTIONS = len(ACTIONS)
ACTION_OBJECTS = [motor]

LEARNING_STATES = {0, 1, 4, 6} 


def choose_action(state):
    
    if state not in LEARNING_STATES:
        action_idx = 0
        ACTIONS[action_idx]()     
        return action_idx, motor

    # ε-greedy inside learning states
    if random.random() < Epsilon:
        action_idx = random.randrange(NUM_ACTIONS)   # explore
    else:
        action_idx = int(np.argmax(Q[state]))        # exploit

    ACTIONS[action_idx]()                            # run motor
    return action_idx, motor

def get_target_species_from_state(state):
    _, mode = STATE_TABLE[state]

    if mode == "Scare_Crows":
        return {"Crow"}
    if mode == "Scare_Magpies":
        return {"Magpie"}
    if mode == "Scare_All":
        return {"Crow", "Magpie"}
    return set()   # Scare_None


def get_species_from_state(state):
    if state in (0, 1, 2, 3):
        return "Crow"
    if state in (4, 5, 6, 7):
        return "Magpie"
    return None

def compute_reward(state_before, state_after, action_idx):
    """
    Reward:
      +1 if target bird deterred
      -1 if target bird not deterred
      -1 if wrong bird deterred
      +1 if wrong bird not deterred
    Minus an energy cost term based on action_idx.
    """
    species_before = get_species_from_state(state_before)
    species_after  = get_species_from_state(state_after)
    target_species = get_target_species_from_state(state_before)

    reward = 0.0

    # --- Case: a bird was present before ---
    if species_before is not None:

        # Was the bird deterred? (disappeared → state_after has species None)
        deterred = (species_after is None)

        if species_before in target_species:
            # target bird (correct animal)
            if deterred:
                reward += 1.0     # success
            else:
                reward -= 1.0     # failed to deter
        else:
            # wrong species (non-target bird)
            if deterred:
                reward -= 1.0     # punished for scaring wrong bird
            else:
                reward += 1.0     # correct behavior (didn't scare wrong species)

    # --- Energy cost ---
    # E = V * duty_fraction * I_full * duration
    # Here: V=9V, I_full=0.25A
    duration_s, duty_frac = ACTION_PARAMS[action_idx]
    energy_cost = 9.0 * duty_frac * 0.25 * duration_s
    reward -= energy_cost

    return reward

def update_q(state_before, state_after, action_idx):
    """
    Updates the Q-table stored in q_table.npy using:

        Q[s][a] = Q[s][a] + Alpha * (reward - Q[s][a])

    and writes the updated Q-table back to disk.
    """
    global Q

    reward = compute_reward(state_before, state_after, action_idx)

    old_q = Q[state_before, action_idx]
    new_q = old_q + Alpha * (reward - old_q)
    Q[state_before, action_idx] = new_q

    np.save(Q_PATH, Q)

    return reward
