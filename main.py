from datetime import datetime, time
import camera
import detector
import mode_store
from rl_controller import STATE_TABLE, LEARNING_STATES, choose_action, update_q

start_time = time(7, 0)  # 7:00
end_time = time(16, 0)     # 14:00

STATE_LOOKUP = { (s, m): i for i, (s, m) in enumerate(STATE_TABLE) }

def is_now_between(start: time, end: time) -> bool:
    now = datetime.now().time()

    if start <= end:
        return start <= now <= end
    else:
        return now >= start or now <= end
    
def get_state(species):
    mode = mode_store.get_mode(species)
    return STATE_LOOKUP[(species, mode)]
    

while is_now_between(start_time, end_time):
    frame = camera.get_frame()
    species, image = detector.detect_and_classify(frame)
    state_before = get_state(species)
    action_idx, action = choose_action(state_before)
    if state_before not in LEARNING_STATES:
        print()
        continue
    time.sleep(5)
    frame1 = camera.get_frame()
    species1, image1 = detector.detect_and_classify(frame1)
    state_after = get_state(species1)
    reward = update_q(state_before, state_after, action_idx)





