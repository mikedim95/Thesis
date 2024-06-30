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

const DatapointPlotter: React.FC<AnomalyPlotterInterface> = ({
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
  const [indicators, setIndicators] = useState([]);
  let max;
  let min;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const anomalyEvent: AnomalyEvent | null = await returnAnomalyEvent(
          edgeName,
          groupName,
          index,
        );
        setIndicators((anomalyEvent?.indicators as never[]) || []);
        console.log("anomalyEventDatapoint", anomalyEvent);
        setAnomalyEvent(anomalyEvent);
        if (anomalyEvent) {
          const dataPoints = anomalyEvent.values.map((value, idx) => {
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
  const getHighlightAreas = (indicators: number[]) => {
    const areas = [];
    let start = -1;
    for (let i = 0; i < indicators.length; i++) {
      if (indicators[i] === 1 && start === -1) {
        start = i;
      } else if (indicators[i] === 0 && start !== -1) {
        areas.push({ x: start, x2: i - 1, label: `Area ${areas.length + 1}` });
        start = -1;
      }
    }
    if (start !== -1) {
      areas.push({
        x: start,
        x2: indicators.length - 1,
        label: `Area ${areas.length + 1}`,
      });
    }
    return areas;
  };

  const highlightAreas = getHighlightAreas(indicators);

  const [anomalyEvent, setAnomalyEvent] = useState<AnomalyEvent | null>(null);
  const options: ApexOptions = {
    legend: {
      show: false,
      position: "top",
      horizontalAlign: "left",
    },
    colors: ["#3C50E0", "#80CAEE"],
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
    annotations: {
      xaxis: highlightAreas.map((area) => ({
        x: area.x,
        x2: area.x2,
        fillColor: "#FF0000", // Red fill color
        opacity: 0.2,
        label: {
          text: area.label,
          style: {
            background: "#FF0000",
            color: "#777",
          },
        },
      })),
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
            <span className="mr-2 mt-1 flex h-4 w-full max-w-4 items-center justify-center rounded-full border border-primary">
              <span className="block h-2.5 w-full max-w-2.5 rounded-full bg-primary"></span>
            </span>
            <div className="w-full">
              <p className="font-semibold text-primary">
                Sample Batch Datapoints
              </p>
            </div>
          </div>
        </div>
        <div className="flex w-full max-w-45 justify-end"></div>
        <div className="flex w-full max-w-45 justify-end">
          <div className="inline-flex items-center rounded-md bg-whiter p-1.5 dark:bg-meta-4">
            created at: {anomalyEvent?.createdAt?.toDateString()}{" "}
            {anomalyEvent?.createdAt?.toLocaleTimeString()}
          </div>
        </div>
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

export default DatapointPlotter;
