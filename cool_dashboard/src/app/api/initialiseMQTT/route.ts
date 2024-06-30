import { NextResponse } from "next/server";
import saveAnomaly from "../../../utils/server/postData";
import { MQTTClientSingleton } from "../../../lib/mqttClientSinlgeton";
import { prisma } from "@/lib/prisma";
type ResponseData = {
  message: string;
};

export async function GET() {
  // This functions instansiates the MQTT Client Singleton
  /*  const { client, firstTime } = MQTTClientSingleton();

  console.log("firstTime:", firstTime);
  if (client && firstTime) {
    console.log("running client.on");
    client.on("message", async function (topic: String, message: any) {
      console.log("Received message2:", JSON.parse(message.toString()));
      const parsedMessage = JSON.parse(message.toString());
      try {
        if (parsedMessage.type === "EdgeGroup") {
          const { groupName, infrastructureType } = parsedMessage;
          const EdgeGroup = await prisma.edgeGroup.create({
            data: {
              groupName,
              infrastructureType,
            },
          });
          console.log(EdgeGroup);
        } else if (parsedMessage.type === "EdgeDevice") {
          const { groupName, edgeName, status, infrastructureType } =
            parsedMessage;
          const EdgeDevice = await prisma.edgeDevice.create({
            data: {
              groupName,
              edgeName,
              status,
              infrastructureType,
            },
          });
          console.log(EdgeDevice);
        } else if (parsedMessage.type === "AnomalyEvent") {
          const { groupName, edgeName, values, isAnomaly } = parsedMessage;
          const AnomalyEvent = await prisma.anomalyEvent.create({
            data: {
              groupName,
              edgeName,
              values,
              isAnomaly,
            },
          });
          console.log(AnomalyEvent);
        } else {
          console.log("Unknown message type:", parsedMessage.type);
        }
      } catch (error) {
        console.error("Error inserting data:", error);
      }
    });
    return NextResponse.json({
      success: true,
      message: "MQTT Client Instansiated",
    });
  } else {
    return NextResponse.json({
      success: true,
      message: "MQTT Client exists already",
    });
  }
 */
  // Handle errors
  return NextResponse.json({
    message: "test purposes only",
  });
}
