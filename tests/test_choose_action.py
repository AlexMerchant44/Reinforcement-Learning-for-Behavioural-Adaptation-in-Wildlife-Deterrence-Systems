from main import load_controller, load_cfg

cfg = load_cfg(path="config/continuous.yaml")
ctrl = load_controller(cfg)

action_idx, duty, duration = ctrl.choose_action(2, cfg)
print(f"action_idx={action_idx}, duty cycle = {duty}, duration = {duration}")
