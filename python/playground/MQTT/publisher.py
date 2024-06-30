import asyncio
import mqttools
BROKER_PORT = 1883


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
    await client.subscribe('/pong')

    while True:
        print()
        message = str('int(time.time())').encode('ascii')
        print(f'client: Publishing {message} on /ping.')
        client.publish(mqttools.Message('/ping', message))

        if message is None:
            print('Client connection lost.')
            break

        await asyncio.sleep(3)


async def main():
    await asyncio.gather(

        client_main()
    )


asyncio.run(main())
