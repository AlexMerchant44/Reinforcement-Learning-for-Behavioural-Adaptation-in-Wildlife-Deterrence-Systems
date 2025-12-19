import os
import numpy as np
import torch

NUM_STATES = 12

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
RUN_DIR = os.path.join(PROJECT_ROOT, "data", "runs", "continuous")
os.makedirs(RUN_DIR, exist_ok=True)

PARAMS_PATH = os.path.join(RUN_DIR, "beta_params.npz")   # stores 4 arrays
BASELINE_PATH = os.path.join(RUN_DIR, "baseline.npy")    # stores baseline per state

EPS = 1e-6
MIN_AB = 0.1        # keep alpha/beta positive
MAX_AB = 50.0       # prevent overconfidence exploding

def _cfg_get(cfg, path, default):
    """
    Safe nested get: path like ("policy","lr")
    """
    d = cfg
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _load_or_init_params(cfg):
    """
    Loads alpha/beta for duty and duration. Creates if missing.
    """
    # sensible initial exploration
    ad0 = float(_cfg_get(cfg, ("duration", "alpha_init"), 2.0))
    bd0 = float(_cfg_get(cfg, ("duration", "beta_init"), 2.0))
    at0 = float(_cfg_get(cfg, ("duty", "alpha_init"), 2.0))
    bt0 = float(_cfg_get(cfg, ("duty", "beta_init"), 2.0))

    if os.path.exists(PARAMS_PATH):
        data = np.load(PARAMS_PATH)
        alpha_d = data["alpha_d"].astype(np.float32)
        beta_d  = data["beta_d"].astype(np.float32)
        alpha_t = data["alpha_t"].astype(np.float32)
        beta_t  = data["beta_t"].astype(np.float32)
    else:
        alpha_d = np.full(NUM_STATES, ad0, dtype=np.float32)
        beta_d  = np.full(NUM_STATES, bd0, dtype=np.float32)
        alpha_t = np.full(NUM_STATES, at0, dtype=np.float32)
        beta_t  = np.full(NUM_STATES, bt0, dtype=np.float32)
        np.savez(PARAMS_PATH, alpha_d=alpha_d, beta_d=beta_d, alpha_t=alpha_t, beta_t=beta_t)

    if os.path.exists(BASELINE_PATH):
        baseline = np.load(BASELINE_PATH).astype(np.float32)
        if baseline.shape[0] != NUM_STATES:
            baseline = np.zeros(NUM_STATES, dtype=np.float32)
            np.save(BASELINE_PATH, baseline)
    else:
        baseline = np.zeros(NUM_STATES, dtype=np.float32)
        np.save(BASELINE_PATH, baseline)

    return alpha_d, beta_d, alpha_t, beta_t, baseline


def _save_params(alpha_d, beta_d, alpha_t, beta_t, baseline):
    np.savez(PARAMS_PATH, alpha_d=alpha_d, beta_d=beta_d, alpha_t=alpha_t, beta_t=beta_t)
    np.save(BASELINE_PATH, baseline)


# initialise module-level params
_alpha_d, _beta_d, _alpha_t, _beta_t, _baseline = _load_or_init_params(cfg={})


def _beta_grads(sample_u, alpha, beta):
    """
    Return gradients of log pi(u | alpha,beta) w.r.t alpha and beta.
    u must be in (0,1).
    Using:
      d/dα log π = log u - ψ(α) + ψ(α+β)
      d/dβ log π = log(1-u) - ψ(β) + ψ(α+β)
    """
    # torch for digamma + stability
    u = torch.tensor(float(sample_u), dtype=torch.float32).clamp(EPS, 1.0 - EPS)
    a = torch.tensor(float(alpha), dtype=torch.float32)
    b = torch.tensor(float(beta), dtype=torch.float32)

    dig_a = torch.special.digamma(a)
    dig_b = torch.special.digamma(b)
    dig_ab = torch.special.digamma(a + b)

    grad_a = torch.log(u) - dig_a + dig_ab
    grad_b = torch.log(1.0 - u) - dig_b + dig_ab

    return float(grad_a.item()), float(grad_b.item())


def choose_action(state, cfg):
    """
    Sample continuous action for this state.

    Returns:
      action_idx, duty (>= Dmin), duration (seconds)
    """
    global _alpha_d, _beta_d, _alpha_t, _beta_t

    # ---- bounds from config ----
    Tmin = float(_cfg_get(cfg, ("duration", "Tmin"), 0.2))
    Tmax = float(_cfg_get(cfg, ("duration", "Tmax"), 2.0))

    Dmin = float(_cfg_get(cfg, ("duty", "Dmin"), 0.3))
    Dmax = float(_cfg_get(cfg, ("duty", "Dmax"), 1.0))

    # ---- pull parameters for this state ----
    ad = float(_alpha_d[state]); bd = float(_beta_d[state])
    at = float(_alpha_t[state]); bt = float(_beta_t[state])

    # ---- sample latent variables u in (0,1) ----
    u_d = float(np.random.beta(ad, bd))
    u_t = float(np.random.beta(at, bt))

    # ---- map to physical ranges ----
    duty = Dmin + u_d * (Dmax - Dmin)
    duration = Tmin + u_t * (Tmax - Tmin)

    return -1, duty, duration

def update(state_before, action_idx, duty, duration, cfg, reward):
    return update_policy(state_before, duty, duration, cfg, reward)

def update_policy(
    state_before,
    duty,
    duration,
    cfg,
    reward,
):
    """
    REINFORCE update on Beta parameters per state, using per-state baseline.

    Stores/updates:
      alpha_d[state], beta_d[state]  (duty Beta)
      alpha_t[state], beta_t[state]  (duration Beta)
      baseline[state]
    """
    global _alpha_d, _beta_d, _alpha_t, _beta_t, _baseline

    # learning rates
    lr = float(_cfg_get(cfg, ("policy", "lr"), 0.02))
    blr = float(_cfg_get(cfg, ("policy", "baseline_lr"), 0.05))

    # duration bounds
    Tmin = float(_cfg_get(cfg, ("duration", "Tmin"), 0.2))
    Tmax = float(_cfg_get(cfg, ("duration", "Tmax"), 2.0))
    denomT = max(EPS, (Tmax - Tmin))

    Dmin = float(_cfg_get(cfg, ("duty", "Dmin"), 0.3))
    Dmax = float(_cfg_get(cfg, ("duty", "Dmax"), 1.0))
    denomD = max(EPS, (Dmax - Dmin))

    # ---- invert duration mapping ----
    u_t = (float(duration) - Tmin) / denomT
    u_t = max(EPS, min(1.0 - EPS, u_t))

    # ---- invert duty mapping----
    # first clamp to physical bounds to be safe
    duty_clamped = min(max(float(duty), Dmin), Dmax)

    u_d = (duty_clamped - Dmin) / denomD
    u_d = max(EPS, min(1.0 - EPS, u_d))

    # advantage = reward - baseline(state)
    b = float(_baseline[state_before])
    adv = float(reward - b)

    # baseline update
    _baseline[state_before] = (1.0 - blr) * _baseline[state_before] + blr * reward

    # current params
    ad = float(_alpha_d[state_before]); bd = float(_beta_d[state_before])
    at = float(_alpha_t[state_before]); bt = float(_beta_t[state_before])

    # gradients of log-prob
    grad_ad, grad_bd = _beta_grads(u_d, ad, bd)
    grad_at, grad_bt = _beta_grads(u_t, at, bt)

    # parameter updates
    ad_new = ad + lr * adv * grad_ad
    bd_new = bd + lr * adv * grad_bd
    at_new = at + lr * adv * grad_at
    bt_new = bt + lr * adv * grad_bt

    # clamp to keep valid & stable
    ad_new = float(np.clip(ad_new, MIN_AB, MAX_AB))
    bd_new = float(np.clip(bd_new, MIN_AB, MAX_AB))
    at_new = float(np.clip(at_new, MIN_AB, MAX_AB))
    bt_new = float(np.clip(bt_new, MIN_AB, MAX_AB))

    _alpha_d[state_before] = ad_new
    _beta_d[state_before]  = bd_new
    _alpha_t[state_before] = at_new
    _beta_t[state_before]  = bt_new

    _save_params(_alpha_d, _beta_d, _alpha_t, _beta_t, _baseline)

    return reward
