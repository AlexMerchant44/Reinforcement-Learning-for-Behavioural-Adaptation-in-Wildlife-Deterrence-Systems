import time
import RPi.GPIO as GPIO

_PWM = None
_GPIO_PIN = None


def init_motor(gpio_pin: int = 4, frequency: int = 8000):
    """
    Initialize motor PWM.
    Call once at program start.
    """
    global _PWM, _GPIO_PIN

    _GPIO_PIN = gpio_pin

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(gpio_pin, GPIO.OUT)

    _PWM = GPIO.PWM(gpio_pin, frequency)
    _PWM.start(0)


def run_motor(duty: float, duration: float):
    """
    Run motor at given duty cycle [0-1] for duration (seconds).
    """
    global _PWM

    if _PWM is None:
        raise RuntimeError("Motor not initialized. Call init_motor().")

    # --- safety clamps ---
    duty = max(0.0, min(1.0, duty))
    duration = max(0.0, duration)

    duty_percent = duty * 100.0

    if duty_percent > 0:
        _PWM.ChangeDutyCycle(duty_percent)
        time.sleep(duration)

    _PWM.ChangeDutyCycle(0.0)


def stop_motor():
    """
    Immediate motor stop.
    """
    global _PWM
    if _PWM is not None:
        _PWM.ChangeDutyCycle(0.0)


def cleanup_motor():
    """
    Cleanup GPIO on exit.
    """
    global _PWM, _GPIO_PIN
    stop_motor()
    if _GPIO_PIN is not None:
        GPIO.cleanup(_GPIO_PIN)
    _PWM = None
