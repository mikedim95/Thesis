import { NextResponse } from "next/server";
import saveAnomaly from "../../../utils/server/postData";
import { Server } from "socket.io";
type ResponseData = {
  message: string;
};
const io = new Server({
  /* options */
});
io.on("connection", (socket) => {
  // ...
});
export async function GET() {
  // This functions instansiates the MQTT Client Singleton

  return NextResponse.json({
    success: true,
    message: "MQTT Client Instansiated",
  });

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
