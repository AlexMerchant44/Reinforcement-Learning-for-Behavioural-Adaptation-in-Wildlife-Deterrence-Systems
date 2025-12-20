from datetime import datetime, time
import time as pytime
import os
import csv
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path

import src.perception.camera as camera
import src.perception.detector as detector

from src.env.state_extractor import get_state, get_mode
from src.env.reward import compute_reward

from src.actuation.action import init_motor, run_motor, cleanup_motor


start_time = time(7, 0)   # 7:00
end_time   = time(17, 0)  # 17:00


def is_now_between(start: time, end: time) -> bool:
    """
    Defines the start and end time for the RL model.
    """
    now = datetime.now().time()
    if start <= end:
        return start <= now <= end
    else:
        return now >= start or now <= end


def load_cfg(path="config/discrete.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def policy_type(cfg):
    return cfg.get("policy", {}).get("type", "discrete").lower()

def get_run_paths(cfg):
    run_dir = os.path.join("data", "runs", policy_type(cfg))
    os.makedirs(run_dir, exist_ok=True)

    episode_dir = os.path.join(run_dir, "Episodes")
    os.makedirs(episode_dir, exist_ok=True)

    history_path = os.path.join(run_dir, "history.csv")
    return episode_dir, history_path


def append_history_row(history_path, row, header):
    write_header = not os.path.exists(history_path)
    with open(history_path, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def save_episode_images(episode_dir, dt, frame_before, frame_after):
    ts = dt.strftime("%Y%m%d_%H%M%S")
    before_path = os.path.join(episode_dir, f"{ts}_before.jpg")
    after_path  = os.path.join(episode_dir, f"{ts}_after.jpg")
    frame_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2RGB)
    frame_after = cv2.cvtColor(frame_after, cv2.COLOR_BGR2RGB)
    cv2.imwrite(before_path, frame_before)
    cv2.imwrite(after_path, frame_after)
    return before_path, after_path


def success_from_species(species_before, species_after):
    """
    Success is defined as deterring the correct species,
    based on the current deterrence mode.
    species_* : "Crow", "Magpie", or "None"
    """

    mode = get_mode()

    # No bird to deter → not a success
    if species_before == "None":
        return False

    # --- mode logic ---
    if mode == "Scare_All":
        return species_after == "None"

    if mode == "Scare_Crows":
        return species_before == "Crow" and species_after != "Crow"

    if mode == "Scare_Magpies":
        return species_before == "Magpie" and species_after != "Magpie"

    # Scare_None
    return False

def load_controller(cfg):
    ptype = policy_type(cfg)
    if ptype == "discrete":
        from src.controllers.discrete import controller as ctrl
        return ctrl
    elif ptype == "continuous":
        from src.controllers.continuous import controller as ctrl
        return ctrl
    else:
        raise ValueError(f"Unknown policy.type: {ptype}")
    
def q_flat_string(ctrl):
    # ctrl.Q is a (12,4) numpy array in your discrete controller
    return " ".join(f"{v:.6f}" for v in ctrl.Q.flatten())

def beta_params_string(ctrl):
    # Preferred: read from in-memory arrays if you exposed them
    # Fallback: load from the saved npz file if present
    if hasattr(ctrl, "_alpha_d"):
        alpha_d = ctrl._alpha_d
        beta_d  = ctrl._beta_d
        alpha_t = ctrl._alpha_t
        beta_t  = ctrl._beta_t
    else:
        # match your continuous controller save path
        here = Path(__file__).resolve().parent
        npz_path = here / "data" / "runs" / "continuous" / "beta_params.npz"
        data = np.load(npz_path)
        alpha_d = data["alpha_d"]
        beta_d  = data["beta_d"]
        alpha_t = data["alpha_t"]
        beta_t  = data["beta_t"]

    joined = np.concatenate([alpha_d.flatten(), beta_d.flatten(), alpha_t.flatten(), beta_t.flatten()])
    return " ".join(f"{v:.6f}" for v in joined)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/discrete.yaml",
        help="Path to config YAML (discrete or continuous)",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    ctrl = load_controller(cfg)

    print(f"{ctrl} controller selected. Setting Up Now")

    episode_dir, history_path = get_run_paths(cfg)

    # Motor init (optional config overrides)
    motor_cfg = cfg.get("motor", {})
    gpio_pin = int(motor_cfg.get("gpio_pin", 4))
    pwm_freq = int(motor_cfg.get("pwm_freq", 8000))
    init_motor(gpio_pin=gpio_pin, frequency=pwm_freq)

    header = [
        "datetime",
        "species_before",
        "species_after",
        "state_before",
        "state_after",
        "action_idx",
        "duty",
        "duration",
        "conf_before",
        "conf_after",
        "reward",
        "success",
    ]

    ptype = policy_type(cfg)
    if ptype == "discrete":
        header.append("q_flattened")
    elif ptype == "continuous":
        header.append("beta_params")

    print("Setup complete. Starting Loop")

    try:
        while True:
            if is_now_between(start_time, end_time):

                dt = datetime.now()

                # ---- BEFORE ----
                frame_before = camera.get_frame()
                species_before, conf_before, frame_before = detector.detect_and_classify(frame_before)
                print(f"Species Detected: {species_before} (conf={conf_before:.2f})")
                state_before = get_state(species_before)

                # ---- CHOOSE ACTION (based on cfg / controller) ----
                action_idx, duty, duration = ctrl.choose_action(state_before, cfg)

                run_motor(float(duty), float(duration))

                # Wait 5s for action and environmental response
                pytime.sleep(5)

                # ---- AFTER ----
                frame_after = camera.get_frame()
                species_after, conf_after, frame_after = detector.detect_and_classify(frame_after)
                state_after = get_state(species_after)

                # ---- SUCCESS ----
                success = success_from_species(species_before, species_after)

                # ---- REWARD (shared) ----
                reward = compute_reward(
                    conf_after=conf_after,
                    duty=float(duty),
                    duration=float(duration),
                    cfg=cfg,
                    species_before=species_before,   # penalise false positives based on BEFORE state
                    success=success,
                )

                # ---- UPDATE (based on cfg / controller) ----
                ctrl.update(
                    state_before,
                    int(action_idx),
                    float(duty),
                    float(duration),
                    cfg,
                    reward,
                )

                # ---- Bird event logging ----
                bird_event = (species_before != "None") or (species_after != "None")
                if bird_event:
                    save_episode_images(episode_dir, dt, frame_before, frame_after)

                row = [
                    dt.isoformat(),
                    species_before,
                    species_after,
                    state_before,
                    state_after,
                    action_idx,
                    float(duty),
                    float(duration),
                    float(conf_before),
                    float(conf_after),
                    float(reward),
                    int(success),
                ]
                ptype = policy_type(cfg)
                if ptype == "discrete":
                    row.append(q_flat_string(ctrl))
                elif ptype == "continuous":
                    row.append(beta_params_string(ctrl))

                append_history_row(history_path, row, header)
                print("Appended to history.csv")

                pytime.sleep(1)

            else:
                print("Outside active hours. Sleeping 60s.")
                pytime.sleep(60)

    finally:
        cleanup_motor()


if __name__ == "__main__":
    main()
