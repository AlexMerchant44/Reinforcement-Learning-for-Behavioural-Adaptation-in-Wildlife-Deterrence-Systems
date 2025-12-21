import os
import random
import numpy as np

# q table dimensions
NUM_STATES = 12
NUM_ACTIONS = 4

DEFAULT_EPSILON = 0.3 # exploration rate
DEFAULT_ALPHA = 0.1 # learning rate

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "runs", "discrete")
os.makedirs(DATA_DIR, exist_ok=True)
Q_PATH = os.path.join(DATA_DIR, "q_table.npy")

ACTION_PARAMS = [
    (0.0, 0.0),  # a0: no movement
    (0.3, 0.5),  # a1
    (0.3, 1.0),  # a2
    (1.0, 0.5),  # a3
]

def _load_or_init_q():
    '''
    Helper function to load or initialise q_table
    Returns q table
    '''
    if os.path.exists(Q_PATH):
        return np.load(Q_PATH)
    Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)
    np.save(Q_PATH, Q)
    return Q

Q = _load_or_init_q()

def choose_action(state, cfg):
    """
    Choose an action index using ε-greedy and return:
      action_idx, duty, duration
    """
    global Q

    # get epsilon from config
    policy_cfg = cfg.get("policy", {})
    epsilon = policy_cfg.get("er", DEFAULT_EPSILON)

    # ε-greedy selection
    if random.random() < epsilon:
        action_idx = random.randrange(NUM_ACTIONS) # explore
    else:
        action_idx = int(np.argmax(Q[state])) # exploit

    duty, duration = ACTION_PARAMS[action_idx]
    return action_idx, duty, duration

def update(state_before, action_idx, duty, duration, cfg, reward):
    return update_q(state_before, action_idx, cfg, reward)

def update_q(
    state_before,
    action_idx,
    cfg,
    reward,
):
    """
    Update Q-table with the existing 1-step rule:
      Q[s,a] <- Qprev[s,a] + alpha * (reward - Qprev[s,a])

    Do not use full Q-learning as actions and rewards are immediate only

    Full Q-Learning update:
      Q[s,a] <- Qprev[s,a] + alpha * (reward + γ*max(Q[s',a'] - Qprev[s,a])

    Where s' is the next state and a' is the next action
    γ is the discount factor

    No planning of future horizon therefore future reward ~ 0
    Therefore γ ~ 0, reducing the equation to the above

    Contextual bandit, not full MDP
    """
    global Q

    # get alpha from config
    policy_cfg = cfg.get("policy", {})
    alpha = policy_cfg.get("lr", DEFAULT_ALPHA)

    old_q = Q[state_before, action_idx]
    new_q = old_q + alpha * (reward - old_q)
    Q[state_before, action_idx] = new_q

    np.save(Q_PATH, Q)

    return reward
