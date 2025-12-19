from main import load_controller, load_cfg, beta_params_string

cfg = load_cfg(path="config/continuous.yaml")
ctrl = load_controller(cfg)

'''
state_before,
int(action_idx),
float(duty),
float(duration),
cfg,
reward
'''
print(f"Params Before: {beta_params_string(ctrl)}")
# ---- UPDATE (based on cfg / controller) ----
ctrl.update(
    1,
    2,
    0.5,
    2,
    cfg,
    -1.3,
)
print(f"Params After: {beta_params_string(ctrl)}")