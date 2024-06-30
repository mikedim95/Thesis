import { prisma } from "@/lib/prisma";

export default async function reportAnomaly(anomalyData: any) {
  console.log("from server action reportAnomaly;", anomalyData);
  const anomalyReport = await prisma.anomalyEvent.create({
    data: {
      edgeName: "agent1",
      groupName: "test2Group",
      values: anomalyData.values,
      anomalyScores: anomalyData.anomalyScores,
      elapsedTime: anomalyData.elapsedTime,
      threshold: anomalyData.threshold,
      indicators: anomalyData.indicators,
    },
  });
  /*   console.log(allUsers); */
  return anomalyReport;
}
