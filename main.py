from datetime import datetime, time
import time as pytime
import camera
import detector
import os
import csv
import cv2
import mode_store
from rl_controller import STATE_TABLE, LEARNING_STATES, choose_action, update_q, Q, Q_PATH

DATA_DIR = os.path.dirname(Q_PATH) or "rl_data"
EPISODE_DIR = os.path.join(DATA_DIR, "Episodes")
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EPISODE_DIR, exist_ok=True)

start_time = time(0, 30)  # 7:00
end_time = time(16, 0)     # 14:00

STATE_LOOKUP = { (s, m): i for i, (s, m) in enumerate(STATE_TABLE) }

def is_now_between(start: time, end: time) -> bool:
    now = datetime.now().time()

    if start <= end:
        return start <= now <= end
    else:
        return now >= start or now <= end
    
def normalise_species(species):
    """
    detector returns: 'Crow', 'Magpie', or None
    STATE_TABLE expects: 'Crow', 'Magpie', 'None'
    """
    return species if species is not None else "None"
    
def get_state(species):
    species = normalise_species(species)
    mode = mode_store.get_mode()
    return STATE_LOOKUP[(species, mode)]
    
def append_history_row(
    dt,
    species_before,
    species_after,
    state_before,
    state_after,
    action_idx,
    reward,
    q_table,
):
    """
    Append one line to history.csv with:
      datetime, species_before, species_after,
      state_before, state_after, action_idx, reward,
      flattened q_table.
    """
    header = [
        "datetime",
        "species_before",
        "species_after",
        "state_before",
        "state_after",
        "action_idx",
        "reward",
        "q_flattened",
    ]
    
    q_flat = " ".join(f"{v:.6f}" for v in q_table.flatten())

    write_header = not os.path.exists(HISTORY_PATH)

    with open(HISTORY_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        writer.writerow([
            dt.isoformat(),
            species_before,
            species_after,
            state_before,
            state_after,
            action_idx,
            f"{reward:.6f}",
            q_flat,
        ])

def save_episode_images(dt, frame_before, frame_after):
    """
    Save before/after frames into Episodes folder with timestamp-based names.
    """
    ts = dt.strftime("%Y%m%d_%H%M%S")
    before_path = os.path.join(EPISODE_DIR, f"{ts}_before.jpg")
    after_path  = os.path.join(EPISODE_DIR, f"{ts}_after.jpg")

    cv2.imwrite(before_path, frame_before)
    cv2.imwrite(after_path, frame_after)

    return before_path, after_path

while is_now_between(start_time, end_time):

    # Use one timestamp per episode
    dt = datetime.now()
    ts = dt.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(EPISODE_DIR, f"{ts}_episode.h264")
    recording_started = False

    # Before frame
    frame_before = camera.get_frame()
    species_before, image = detector.detect_and_classify(frame_before)
    species_before = normalise_species(species_before)

    # Start recording as soon as we see a bird in the before frame
    if species_before != "None":
        camera.start_recording(video_path)
        recording_started = True

    state_before = get_state(species_before)
    action_idx, motor = choose_action(state_before)

    # If no action is taken, tidy up and continue
    if state_before not in LEARNING_STATES:
        # If we did start a recording stop it and delete the temporary clip.
        if recording_started:
            try:
                camera.stop_recording()
            except Exception as e:
                print(f"[Camera] stop_recording (non-learning state) error: {e}")
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[Cleanup] Deleted unused video (non-learning state): {video_path}")

        pytime.sleep(1)
        continue

    # Wait for action
    pytime.sleep(5)

    # After frame
    frame_after = camera.get_frame()
    species_after, image2 = detector.detect_and_classify(frame_after)
    species_after = normalise_species(species_after)
    state_after = get_state(species_after)

    # Stop recording only if we started it
    if recording_started:
        try:
            camera.stop_recording()
        except Exception as e:
            print(f"[Camera] stop_recording error: {e}")


    reward = update_q(state_before, state_after, action_idx)

    # Bird event if a bird was present in either before or after
    bird_event = (species_before != "None") or (species_after != "None")

    if bird_event:
        before_path, after_path = save_episode_images(dt, frame_before, frame_after)

        append_history_row(
            dt=dt,
            species_before=species_before,
            species_after=species_after,
            state_before=state_before,
            state_after=state_after,
            action_idx=action_idx,
            reward=reward,
            q_table=Q,
        )
        print("Appended to history.csv")
        if recording_started:
            print(f"Episode video kept: {video_path}")
    else:
        # No bird in either frame → delete the temporary video if we recorded one
        if recording_started and os.path.exists(video_path):
            os.remove(video_path)
            print(f"[Cleanup] Deleted unused video (no bird): {video_path}")

    pytime.sleep(1)






