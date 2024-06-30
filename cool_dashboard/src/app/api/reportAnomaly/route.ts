import reportAnomaly from "@/utils/server/postData";
import { NextResponse } from "next/server";

export async function POST(request: Request, res: Response) {
  const anomalyData = await request.json();
  console.log(anomalyData);
  try {
    const anomalyEvent = await reportAnomaly(anomalyData);

    return NextResponse.json(anomalyEvent);
  } catch (err) {
    return NextResponse.json({
      success: false,
      message: err,
    });
  }
}
