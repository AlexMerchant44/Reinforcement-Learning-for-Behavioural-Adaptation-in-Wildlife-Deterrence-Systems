import argparse
from src.actuation.action import init_motor, run_motor, cleanup_motor
from main import load_cfg
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="config/continuous.yaml",
    help="Path to config YAML (discrete or continuous)",
)
args = parser.parse_args()
cfg = load_cfg(args.config)

# Motor init (optional config overrides)
motor_cfg = cfg.get("motor", {})
gpio_pin = int(motor_cfg.get("gpio_pin", 4))
pwm_freq = int(motor_cfg.get("pwm_freq", 8000))
init_motor(gpio_pin=gpio_pin, frequency=pwm_freq)

run_motor(0.6, 3)

cleanup_motor()


