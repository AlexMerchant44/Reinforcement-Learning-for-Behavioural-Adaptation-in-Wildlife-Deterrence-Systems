# reward.py

def compute_reward(
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

    # power penalty
    power_cost = duty * duration
    r_power = -cfg["reward"]["energy_penalty"] * power_cost

    # success bonus
    r_success = cfg["reward"]["clear_bonus"] if success else 0.0

    # false positive penalty (worse if used more energy)
    r_false = 0.0
    if species_before == "None":
        r_false = -cfg["reward"]["false_positive_penalty"] * power_cost

    # false negative penalty, conf_after = 0 if species_after is 'None'
    r_presence = 0.0
    if species_before != "None":
        r_presence = -cfg["reward"]["false_negative_penalty"] * conf_after

    # total reward
    reward = r_power + r_success + r_false + r_presence
    return reward
