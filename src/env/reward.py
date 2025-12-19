# reward.py

def compute_reward(
    conf_before,
    conf_after,
    duty,
    duration,
    cfg,
    *,
    species_before=None,
    success=False,
):
    """
    Compute reward for one step.

    Parameters

    conf_before : float
        Bird confidence at previous step (0-1)
    conf_after : float
        Bird confidence at current step (0-1)
    duty : float
        Motor duty cycle (0-1)
    duration : float
        Motor duration (seconds)
    cfg : dict
        Config dictionary loaded from YAML
    species : str or None
        'Crow', 'Magpie', or 'None'
    success : bool
        True if bird considered cleared this step

    Returns
    
    reward : float
    """

    # progress reward
    delta_conf = conf_before - conf_after
    r_progress = cfg["reward"]["progress_weight"] * delta_conf

    # power penalty
    power_cost = duty * duration
    r_power = -cfg["reward"]["energy_penalty"] * power_cost

    # success bonus
    r_success = cfg["reward"]["clear_bonus"] if success else 0.0

    # false positive penalty
    r_false = 0.0
    if species_before == "None":
        r_false = -cfg["reward"]["false_positive_weight"] * power_cost

    # total reward
    reward = r_progress + r_power + r_success + r_false
    return reward
