// utils/mqttSingleton.js

import mqtt from "mqtt";

let mqttClientInstance = null;

/* export function getMqttClient() {
  if (!mqttClientInstance) {
    const mqttServer = "mqtt://test.mosquitto.org"; // Example MQTT server, replace with your own
    mqttClientInstance = mqtt.connect(mqttServer);
  }
  return mqttClientInstance;
} */
const globalTemporal = global as unknown as { client: mqtt.MqttClient };
const mqttInstanse = () => {
  if (globalTemporal.client) {
    return globalTemporal.client;
  }
  const clientId: string = process.env.MQTT_CLIENT || "";

  const clientConnectString: string =
    process.env.MQTT_CLIENT_CONNECT_STRING || "";
  const client = mqtt.connect(clientConnectString, {
    clientId,
  });
  const topic: string = process.env.MQTT_TOPIC!;
  console.log(clientId, topic);
  globalTemporal.client = client;
  globalTemporal.client.on("connect", async () => {
    /* client.subscribe(topic, (err) => {
      if (err) {
        console.error("Error subscribing to topic:", err);
      } else {
        console.log("Subscribed to topic!!!!!!!:", topic);
      }
    });
    client.publish(topic, "message", (err) => {
      if (!err) {
        console.log("Successfully published to topic:", topic);
      } else {
        console.error("Error publishing to topic:", topic, err);
      }
    }); */
    console.error("connected");
  });
  return globalTemporal.client;
};
export default mqttInstanse;
