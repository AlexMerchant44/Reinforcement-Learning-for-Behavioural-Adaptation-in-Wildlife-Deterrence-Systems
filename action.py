"""
Motor control helper for Raspberry Pi.

Usage:
    from action import MotorAction

    motor = MotorAction(gpio_pin=18)  # BCM number
    motor.run(0.5)  # run motor for 0.5 seconds
"""

import time
import RPi.GPIO as GPIO


class MotorAction:
    def __init__(self, gpio_pin: int = 18, active_high: bool = True):
        """
        gpio_pin: BCM pin number used to control motor driver input.
        active_high: True if motor should run when pin is HIGH.
                     Set to False if your driver is active-low.
        """
        self.gpio_pin = gpio_pin
        self.active_high = active_high

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_pin, GPIO.OUT, initial=GPIO.LOW if active_high else GPIO.HIGH)

    def run(self, duration_s: float):
        """
        Run the motor for duration_s seconds.
        Non-blocking beyond time.sleep, so call from your main loop.
        """
        if duration_s <= 0:
            return

        # turn motor ON
        GPIO.output(self.gpio_pin, GPIO.HIGH if self.active_high else GPIO.LOW)
        time.sleep(duration_s)
        # turn motor OFF
        GPIO.output(self.gpio_pin, GPIO.LOW if self.active_high else GPIO.HIGH)

    def stop(self):
        """Force motor OFF."""
        GPIO.output(self.gpio_pin, GPIO.LOW if self.active_high else GPIO.HIGH)

    def cleanup(self):
        """Call once at program exit if this is the only thing using GPIO."""
        GPIO.cleanup(self.gpio_pin)


if __name__ == "__main__":
    # quick manual test: run motor for 0.5s
    m = MotorAction(gpio_pin=18)
    try:
        m.run(0.5)
    finally:
        m.cleanup()
