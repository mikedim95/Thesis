"use client";
import React, { useEffect, useState } from "react";
/* import { AnomalyEvent, EdgeDevice, EdgeGroup } from "@/types/edgeEntities"; */
/* import ChartOne from "../Charts/ChartOne";
import ChartThree from "../Charts/ChartThree";
import ChartTwo from "../Charts/ChartTwo";
import ChatCard from "../Chat/ChatCard";
import TableOne from "../Tables/TableOne"; */
import CardDataStats from "../CardDataStats";
/* import MapOne from "../Maps/MapOne"; */
import {
  returnAnomalyEvent,
  returnEdgeDevice,
  returnEdgeGroup,
  returnReportsPopulation,
} from "@/utils/server/_actions";
import { EdgeDevice, EdgeGroup, ErrorResponse } from "@/types/edgeEntities";
const ECommerce: React.FC = () => {
  const [edgeDevices, setEdgeDevices] = useState<
    EdgeDevice | ErrorResponse | null
  >(null);
  const [edgeGroups, setEdgeGroups] = useState<
    EdgeGroup | ErrorResponse | null
  >(null);
  useEffect(() => {
    const fetchData = async () => {
      try {
        /*  const anomalyEvent: any = await returnAnomalyEvent(
          "someEdgeId",
          "someGroupId",
        ); */
        const edgeDevice: EdgeDevice | ErrorResponse = await returnEdgeDevice();
        const edgeGroup: EdgeGroup | ErrorResponse = await returnEdgeGroup();
        console.log(edgeGroup);
        console.log(edgeDevice);

        setEdgeDevices(edgeDevice);
        setEdgeGroups(edgeGroup);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, []);
  const handleClick =
    (edgeId: string, groupId: string) =>
    async (event: React.MouseEvent<HTMLButtonElement>) => {
      // Your code here
      /* const anomalyEvent = await returnAnomalyEvent(edgeId, groupId);
      console.log(anomalyEvent); */
      const edgeDevice = await returnEdgeDevice();
      console.log(edgeDevice);
      const edgeGroup = await returnEdgeGroup();
      console.log(edgeGroup);
    };

  return (
    <div className="flex flex-col gap-6">
      {Array.isArray(edgeDevices) &&
        edgeDevices.map((device, index) => (
          <div
            key={index}
            className="w-full min-w-[100px] transition-all duration-500 ease-in-out md:min-w-[350px] xl:min-w-[450px]"
          >
            <CardDataStats
              edgeName={device.edgeName}
              groupName={device.groupName}
              rate="N/A"
              levelUp={false}
            >
              {device.infrastructureType === "bridge" ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="icon icon-tabler icon-tabler-building-bridge"
                  width="44"
                  height="44"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="#000000"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                  <path d="M6 5l0 14" />
                  <path d="M18 5l0 14" />
                  <path d="M2 15l20 0" />
                  <path d="M3 8a7.5 7.5 0 0 0 3 -2a6.5 6.5 0 0 0 12 0a7.5 7.5 0 0 0 3 2" />
                  <path d="M12 10l0 5" />
                </svg>
              ) : device.infrastructureType === "building" ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="icon icon-tabler icon-tabler-building"
                  width="44"
                  height="44"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="#000000"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                  <path d="M3 21l18 0" />
                  <path d="M9 8l1 0" />
                  <path d="M9 12l1 0" />
                  <path d="M9 16l1 0" />
                  <path d="M14 8l1 0" />
                  <path d="M14 12l1 0" />
                  <path d="M14 16l1 0" />
                  <path d="M5 21v-16a2 2 0 0 1 2 -2h10a2 2 0 0 1 2 2v16" />
                </svg>
              ) : (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="icon icon-tabler icon-tabler-question-mark"
                  width="44"
                  height="44"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="#000000"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                  <path d="M8 8a3.5 3 0 0 1 3.5 -3h1a3.5 3 0 0 1 3.5 3a3 3 0 0 1 -2 3a3 4 0 0 0 -2 4" />
                  <path d="M12 19l0 .01" />
                </svg>
              )}
            </CardDataStats>
          </div>
        ))}
    </div>
  );
};

export default ECommerce;
