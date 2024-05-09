import { NextResponse } from "next/server";
import saveAnomaly from "../../../utils/server/postData";
import { MQTTClientSingleton } from "../../../lib/mqttClientSinlgeton";
type ResponseData = {
  message: string;
};

export async function GET() {
  // This functions instansiates the MQTT Client Singleton
  const { client, firstTime } = MQTTClientSingleton();
  console.log("firstTime:", firstTime);
  if (client && firstTime) {
    console.log("running client.on");
    client.on("message", function (topic: any, message: any) {
      console.log("Received message2:", JSON.parse(message.toString()));
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

  // Handle errors
}
export async function POST(request: Request, res: Response) {
  const data = await request.json();
  try {
    const user2 = await saveAnomaly(data.email);

    return NextResponse.json({ user2 });
  } catch (err) {
    return NextResponse.json({
      success: false,
      message: err,
    });
  }
}
