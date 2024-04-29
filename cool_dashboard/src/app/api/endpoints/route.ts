import { NextResponse } from "next/server";
import mqttInstanse from "../../../utils/server/mqttSingleton";
type ResponseData = {
  message: string;
};

export async function GET(request: any) {
  console.log(request.client);
  const client = mqttInstanse();
  const topic: string = process.env.MQTT_TOPIC!;
  client.subscribe(topic, (err) => {
    if (err) {
      console.error("Error subscribing to topic:", err);
    } else {
      console.log("Subscribed to topic!!!!!!!:", topic);
    }
  });
  /* client.on("connect", function () {
    console.log("Connected to MQTT server");

    // Subscribe to a topic
    client.subscribe("topic/example", function (err: any) {
      if (err) {
        console.error("Subscription failed:", err);
        return NextResponse.json({
          success: false,
          error: "Subscription failed",
        });
      } else {
        console.log("Subscribed to topic/example");
        return NextResponse.json({
          success: true,
          message: "Subscribed to topic/example",
        });
      }
    });
  }); */

  client.on("message", function (topic: any, message: any) {
    console.log("Received message:", message.toString());
    // Process the incoming message as needed
  });

  // Handle errors
}
export async function POST(request: Request) {
  const data = await request.json();
  try {
    return NextResponse.json({ data });
  } catch (err) {
    /*  res.status(500).json({ error: "failed to load data" }); */
  }
}
