import mqtt from "mqtt";
// PrismaClient is attached to the `global` object in development to prevent
// exhausting your database connection limit.
//
// Learn more:
// https://pris.ly/d/help/next-js-best-practices
interface Result {
  client?: mqtt.MqttClient;
  firstTime?: boolean;
}

const globalForMQTTClient = global as unknown as { client: mqtt.MqttClient };
const topic: string = process.env.MQTT_TOPIC!;
export function MQTTClientSingleton(): Result {
  if (globalForMQTTClient.client != undefined) {
    const firstTime = false;
    const client = globalForMQTTClient.client; // Save the oldClient
    return { client, firstTime };
  } else {
    try {
      console.log(
        "globalForMQTTClient.client try:",
        globalForMQTTClient.client == undefined,
      );
      const clientConnectString = process.env.MQTT_CLIENT_CONNECT_STRING || ""; // Change this to your MQTT broker address
      const clientId = process.env.MQTT_CLIENT || ""; // Change this to your desired client ID

      const client: mqtt.MqttClient = mqtt.connect(clientConnectString, {
        clientId,
      });

      client.on("connect", () => {
        console.log("MQTT client connected");
      });

      client.on("error", (error) => {
        console.error("MQTT connection error:", error);
      });

      client.on("close", () => {
        console.log("MQTT client connection closed");
      });
      client.subscribe(topic, (err) => {
        if (err) {
          console.error("Error subscribing to topic:", err);
          return {
            success: false,
            error: "Subscription failed",
          };
        } else {
          console.log("Subscribed to topic!!!!!!!:", topic);
          return {
            success: true,
            message: "Subscribed to topic/example",
          };
        }
      });
      globalForMQTTClient.client = client;
      const firstTime = true;
      return { client, firstTime };
    } catch (error) {
      console.error("Error while connecting to MQTT broker:", error);
      return {
        client: undefined,
        firstTime: undefined,
      };
    }
  }
}
export function thereIsNoClientAlready() {
  return globalForMQTTClient.client == undefined;
}
