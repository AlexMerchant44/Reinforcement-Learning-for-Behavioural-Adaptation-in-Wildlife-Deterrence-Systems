from main import load_controller, load_cfg, q_flat_string

cfg = load_cfg(path="config/discrete.yaml")
ctrl = load_controller(cfg)

'''
state_before,
int(action_idx),
float(duty),
float(duration),
cfg,
reward
'''

print(f"Params Before: {q_flat_string(ctrl)}")
# ---- UPDATE (based on cfg / controller) ----
ctrl.update(
    1,
    2,
    0.5,
    2,
    cfg,
    -1.3,
)
print(f"Params After: {q_flat_string(ctrl)}")