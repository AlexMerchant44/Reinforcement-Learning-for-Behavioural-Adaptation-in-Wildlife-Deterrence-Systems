# action.py
"""
PWM Motor Control for Raspberry Pi using a MOSFET.
"""

import time
import RPi.GPIO as GPIO


class MotorAction:
    def __init__(self, gpio_pin=4, frequency=1000):
        """
        gpio_pin: BCM pin used for PWM output (Gate of MOSFET)
        frequency: PWM frequency in Hz (1kHz is good for motors)
        """
        self.gpio_pin = gpio_pin
        self.frequency = frequency

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_pin, GPIO.OUT)

        # Create PWM instance but do not start it yet
        self.pwm = GPIO.PWM(self.gpio_pin, self.frequency)
        self.current_speed = 0  # duty cycle %

    def set_speed(self, duty_cycle: float):
        """
        Sets motor speed (0–100%)
        """
        duty_cycle = max(0, min(100, duty_cycle))  # clamp to valid range

        if duty_cycle == 0:
            self.pwm.stop()
        else:
            try:
                self.pwm.start(duty_cycle)
            except RuntimeError:
                # Already started → change duty cycle
                self.pwm.ChangeDutyCycle(duty_cycle)

        self.current_speed = duty_cycle

    def run(self, duration_s: float, duty_cycle: float = 100):
        """
        Run the motor for duration_s seconds at given duty cycle (0–100%)
        """
        if duration_s <= 0:
            return

        self.set_speed(duty_cycle)   # turn motor on
        time.sleep(duration_s)
        self.set_speed(0)            # stop motor

    def stop(self):
        """Emergency stop (motor off instantly)"""
        self.set_speed(0)

    def cleanup(self):
        """Call at program exit to release GPIO"""
        self.stop()
        GPIO.cleanup(self.gpio_pin)
