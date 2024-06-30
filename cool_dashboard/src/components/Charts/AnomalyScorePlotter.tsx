import { returnAnomalyEvent } from "@/utils/server/_actions";
import { AnomalyEvent } from "@prisma/client";
import { ApexOptions } from "apexcharts";
import React, { useEffect, useState } from "react";
import ReactApexChart from "react-apexcharts";
import { findMax, findMin } from "@/utils/server/_actions";

interface AnomalyPlotterInterface {
  edgeName: string;
  groupName: string;
  index: number;
}

const AnomalyScorePlotter: React.FC<AnomalyPlotterInterface> = ({
  edgeName,
  groupName,
  index,
}) => {
  const [series, setSeries] = useState<
    { name: string; data: { x: string | number; y: number }[] }[]
  >([
    {
      name: "Anomaly Data",
      data: [],
    },
  ]);
  const [thresholdValue, setThresholdValue] = useState(0);
  let max;
  let min;
  let anomalyDatapoints: number[] = [];

  useEffect(() => {
    const fetchData = async () => {
      try {
        const anomalyEvent: AnomalyEvent | null = await returnAnomalyEvent(
          edgeName,
          groupName,
          index,
        );
        console.log("anomalyEventDatapoint", anomalyEvent);
        setThresholdValue(anomalyEvent?.threshold || 0);
        setAnomalyEvent(anomalyEvent);
        if (anomalyEvent) {
          const dataPoints = anomalyEvent.anomalyScores.map((value, idx) => {
            return { x: idx, y: value };
          });

          setSeries([
            {
              name: "Anomaly Data",
              data: dataPoints,
            },
          ]);
        }
      } catch (error) {
        console.error("Error counting reports:", error);
      }
    };

    fetchData();
  }, [edgeName, groupName, index]);

  const [anomalyEvent, setAnomalyEvent] = useState<AnomalyEvent | null>(null);
  const options: ApexOptions = {
    legend: {
      show: false,
      position: "top",
      horizontalAlign: "left",
    },
    colors: ["#80CAEE"],
    chart: {
      fontFamily: "Satoshi, sans-serif",
      height: 335,
      type: "area",
      dropShadow: {
        enabled: true,
        color: "#623CEA14",
        top: 10,
        blur: 4,
        left: 0,
        opacity: 0.1,
      },

      toolbar: {
        show: false,
      },
    },
    annotations: {
      yaxis: [
        {
          y: thresholdValue,
          borderColor: "red",
          label: {
            borderColor: "red",
            style: {
              color: "#fff",
              background: "red",
            },
            text: `Threshold: ${thresholdValue}`,
          },
        },
      ],
    },

    responsive: [
      {
        breakpoint: 1024,
        options: {
          chart: {
            height: 300,
          },
        },
      },
      {
        breakpoint: 1366,
        options: {
          chart: {
            height: 350,
          },
        },
      },
    ],
    stroke: {
      width: [2, 2],
      curve: "smooth",
    },
    // labels: {
    //   show: false,
    //   position: "top",
    // },
    grid: {
      xaxis: {
        lines: {
          show: true,
        },
      },
      yaxis: {
        lines: {
          show: true,
        },
      },
    },
    dataLabels: {
      enabled: false,
    },

    xaxis: {
      type: "numeric",

      tickAmount: 10, // Set the number of ticks on the x-axis
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    yaxis: {
      labels: {
        formatter: (val) => val.toFixed(2), // Reduce decimal places
      },
      title: {
        style: {
          fontSize: "0px",
        },
      },
      min: min,
      max: max,
    },
  };
  return (
    <div className="col-span-12 rounded-sm border border-stroke bg-white px-5 pb-5 pt-7.5 shadow-default dark:border-strokedark dark:bg-boxdark sm:px-7.5 xl:col-span-8">
      <div className="flex flex-wrap items-start justify-between gap-3 sm:flex-nowrap">
        <div className="flex w-full flex-wrap gap-3 sm:gap-5">
          <div className="flex min-w-47.5">
            <span className="mr-2 mt-1 flex h-4 w-full max-w-4 items-center justify-center rounded-full border border-secondary">
              <span className="block h-2.5 w-full max-w-2.5 rounded-full bg-secondary"></span>
            </span>
            <div className="w-full">
              <p className="font-semibold text-secondary">
                Corresponding Anomaly Score
              </p>
            </div>
          </div>
        </div>
        <div className="flex w-full max-w-45 justify-end"></div>
      </div>

      <div>
        <div id="chartOne" className="-ml-5">
          <ReactApexChart
            options={options}
            series={series}
            type="line"
            height={350}
            width={"100%"}
          />
        </div>
      </div>
    </div>
  );
};

export default AnomalyScorePlotter;
