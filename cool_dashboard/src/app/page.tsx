import ECommerce from "@/components/Dashboard/E-commerce";
import { Metadata } from "next";
import DefaultLayout from "@/components/Layouts/DefaultLayout";
import { prisma } from "@/lib/prisma";
import { MQTTClientSingleton } from "@/lib/mqttClientSinlgeton";
export const metadata: Metadata = {
  title:
    "Next.js E-commerce Dashboard | TailAdmin - Next.js Dashboard Template",
  description: "This is Next.js Home for TailAdmin Dashboard Template",
};

export default function Home() {
  const { client, firstTime } = MQTTClientSingleton();
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
          /* await prisma.edgeGroup.create({
            data: {
              groupName: parsedMessage.groupName,
              infrastructureType: parsedMessage.infrastructureType,
            },
          }); */
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
          const {
            groupName,
            edgeName,
            anomalyDatapoints,
            datapointsAfterAnomaly,
            datapointsBeforeAnomaly,
          } = parsedMessage;
          const AnomalyEvent = await prisma.anomalyEvent.create({
            data: {
              groupName,
              edgeName,
              anomalyDatapoints,
              datapointsAfterAnomaly,
              datapointsBeforeAnomaly,
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
  }
  return (
    <>
      <DefaultLayout>
        <ECommerce />
      </DefaultLayout>
    </>
  );
}
