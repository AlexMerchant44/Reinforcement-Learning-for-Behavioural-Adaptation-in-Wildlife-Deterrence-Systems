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

# ---- Load or create Q-table ----
Q_PATH = "q_table.npy"

NUM_STATES = 12      # states 0..11
NUM_ACTIONS = 4

if os.path.exists(Q_PATH):
    Q = np.load(Q_PATH)
else:
    Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)
    np.save(Q_PATH, Q)

# ---- Define actions ----
motor = MotorAction(gpio_pin=4)

ACTIONS = [
    lambda: motor.run(0),                # 0 seconds = motor off
    lambda: motor.run(0.1, 50),          # duration=0.1s, duty=50%
    lambda: motor.run(0.1, 100),         # duration=0.1s, duty=100%
    lambda: motor.run(1, 50),            # duration=1s, duty=50%
]

NUM_ACTIONS = len(ACTIONS)
ACTION_OBJECTS = [motor]
LEARNING_STATES = {0, 1, 4, 6}    # only these use ε-greedy


def choose_action(state):
    """
    Returns (action_idx, action_obj).
    Applies epsilon-greedy ONLY for states 0, 1, 4, 6.
    All other states always do A0.
    """

    # non-learning states → always A0
    if state not in LEARNING_STATES:
        action_idx = 0
        ACTIONS[action_idx]()          # run A0
        return action_idx, motor

    # ε-greedy inside learning states
    if random.random() < Epsilon:
        action_idx = random.randrange(NUM_ACTIONS)   # random action
    else:
        action_idx = int(np.argmax(Q[state]))        # greedy action

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

    actions = [[0,0], [0.1,0.5], [0.1, 1], [1, 0.5]]

    species_before = get_species_from_state(state_before)
    species_after  = get_species_from_state(state_after)

    target_species = get_target_species_from_state(state_before)

    reward = 0

    # --- Case: a bird was present before ---
    if species_before is not None:

        # Was the bird deterred? (disappeared)
        deterred = (species_after is None)

        if species_before in target_species:
            # target bird (correct animal)
            if deterred:
                reward += 1      # success
            else:
                reward -= 1      # failed to deter
        else:
            # wrong species (non-target bird)
            if deterred:
                reward -= 1      # punished for scaring wrong bird
            else:
                reward += 1      # correct behavior (didn't scare wrong species)

    # --- No bird before ---
    else:
        # no reward bonus or penalty; could add noise if desired
        pass

    # --- Energy cost --- #E = V*PWM*I*t
    reward -= 9*actions[action_idx][1]*0.25*actions[action_idx[0]]

    return reward

def update_q(state_before, state_after, action_idx):
    """
    Updates the Q-table stored in q_table.npy using your rule:

        Q[s][a] = Q[s][a] + alpha * (reward - Q[s][a])

    Also writes the updated Q-table back to disk immediately.
    """
    global Q

    # --- 1. Compute reward for this transition ---
    reward = compute_reward(state_before, state_after, action_idx)

    # --- 2. Current Q-value ---
    old_q = Q[state_before, action_idx]

    # --- 3. Apply update rule ---
    new_q = old_q + Alpha * (reward - old_q)
    Q[state_before, action_idx] = new_q

    # --- 4. Save updated Q-table back to q_table.npy ---
    np.save(Q_PATH, Q)
    return reward





