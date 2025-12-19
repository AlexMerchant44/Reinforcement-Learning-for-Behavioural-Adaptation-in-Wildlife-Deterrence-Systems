import src.perception.camera as camera
from datetime import datetime
import time as pytime
import src.perception.detector as detector

from main import save_episode_images, load_cfg, get_run_paths  

frame_before = camera.get_frame()
pytime.sleep(5)
frame_after = camera.get_frame()

species_before, conf_before, frame_before = detector.detect_and_classify(frame_before)
species_after, conf_after, frame_after = detector.detect_and_classify(frame_after)
print(f"Species detected before: {species_before} (conf={conf_before:.2f})")
print(f"Species detected after: {species_after} (conf={conf_after:.2f})")

cfg = load_cfg()
run_dir, episode_dir, history_path = get_run_paths(cfg)
dt = datetime.now()
save_episode_images(episode_dir, dt, frame_before, frame_after)
print('Saved Episode Images')





