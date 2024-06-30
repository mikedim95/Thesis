# blink_led.py
import RPi.GPIO as GPIO
import time

LED_PIN = 17  # Change this depending on which GPIO pin your LED is connected to

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)  # Turn LED on
        time.sleep(1)                    # Delay for 1 second
        GPIO.output(LED_PIN, GPIO.LOW)   # Turn LED off
        time.sleep(1)                    # Delay for 1 second
finally:
    GPIO.cleanup()  #
