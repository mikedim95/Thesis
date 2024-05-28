import paho.mqtt.client as mqtt

# Define callback function to handle message reception

username = "testUser"
password = "testUser2"
client_id = 'asdasdgeg'


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe('testTopic')


def on_message(client, userdata, message):
    print(f"Received message: {message.payload.decode()}")


# MQTT broker settings
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message
client.connect('i55eb5f8.ala.eu-central-1.emqxsl.com', 8883, 60)
print("listening")
try:
    client.loop_forever()
except KeyboardInterrupt:
    pass
