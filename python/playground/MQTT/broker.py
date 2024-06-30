import time
import asyncio
import mqttools
BROKER_PORT = 1883


async def broker_main():
    """The broker, serving both clients, forever.

    """

    broker = mqttools.Broker(('localhost', BROKER_PORT))
    await broker.serve_forever()


async def main():
    await asyncio.gather(
        broker_main(),
        # echo_client_main(),
        # client_main()
    )


asyncio.run(main())
