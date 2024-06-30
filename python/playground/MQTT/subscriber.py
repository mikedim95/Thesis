import time
import RPi.GPIO as GPIO
import asyncio
import mqttools
BROKER_PORT = 1883

LED_PIN = 17  # Change this depending on which GPIO pin your LED is connected to
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)


async def start_client():
    client = mqttools.Client('172.17.0.7', BROKER_PORT, connect_delays=[0.1])
    await client.start()
    print('client started')
    return client


async def client_main():
    """Publish the current time to /ping and wait for the echo client to
    publish it back on /pong, with a one second interval.

    """

    client = await start_client()
    await client.subscribe('/ping')

    while True:

        message = await client.messages.get()
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        try:

            GPIO.output(LED_PIN, GPIO.HIGH)  # Turn LED on
            time.sleep(1)                    # Delay for 1 second
            GPIO.output(LED_PIN, GPIO.LOW)   # Turn LED off
            time.sleep(1)                    # Delay for 1 second
        finally:
            GPIO.cleanup()

        print(f'client: Got {message.message} on {message.topic}.')

        if message is None:
            print('Client connection lost.')
            break


async def main():
    await asyncio.gather(

        client_main()
    )


asyncio.run(main())
