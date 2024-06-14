import { ApexOptions } from "apexcharts";
import React, { useState } from "react";
import ReactApexChart from "react-apexcharts";

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

  xaxis: {
    type: "category",
    categories: [
      "Sep",
      "Oct",
      "Nov",
      "Dec",
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
    ],
    axisBorder: {
      show: false,
    },
    axisTicks: {
      show: false,
    },
  },
  yaxis: {
    title: {
      style: {
        fontSize: "0px",
      },
    },
    min: 72,
    max: 120,
  },
};

interface ChartOneState {
  series: {
    name: string;
    data: number[];
  }[];
}

const ChartOne: React.FC = () => {
  const [state, setState] = useState<ChartOneState>({
    series: [
      {
        name: "Product One",
        data: [
          97.028994, 96.855711, 96.51092, 96.247199, 95.421465, 95.243237,
          95.396408, 95.699644, 96.06739, 96.563938, 97.043515, 97.411748,
          98.397974, 99.667015, 100.65905, 101.71292, 102.67781, 103.75815,
          104.29042, 105.66761, 106.45618, 106.81393, 107.18798, 107.43012,
          107.28255, 107.28397, 107.16303, 106.76015, 106.97184, 106.4589,
          106.11458, 106.01503, 105.70056, 105.15262, 104.68782, 104.14235,
          103.27921, 102.24762, 101.47965, 99.31493, 97.344548, 94.535249,
          92.046894, 89.86669, 88.66062, 88.266112, 88.25612, 89.120806,
          89.55877, 89.82264, 89.619147, 90.544324, 91.83264, 92.507998,
          92.783044, 93.426625, 93.480241, 93.432055, 93.2303, 92.860378,
          92.216228, 91.730487, 90.947678, 89.56443, 88.310806, 87.499044,
          86.561646, 85.898066, 85.363595, 85.173989, 84.371833, 83.901411,
          84.304931, 83.720303, 84.080288, 84.272275, 84.222506, 84.015116,
          84.000464, 83.490034, 83.130167, 83.416285, 83.596158, 84.045088,
          83.731394, 83.304966, 83.166303, 82.719439, 81.820784, 80.791504,
          80.87932, 80.796086, 81.042463, 81.017985, 80.148275, 80.101455,
          79.84784, 79.623646, 79.075056, 78.530902,
        ],
      },

      {
        name: "Product Two",
        data: [30, 25, 36, 30, 45, 35, 64, 52, 59, 36, 39, 51],
      },
    ],
  });

  const handleReset = () => {
    setState((prevState) => ({
      ...prevState,
    }));
  };
  handleReset;

  return (
    <div className="col-span-12 rounded-sm border border-stroke bg-white px-5 pb-5 pt-7.5 shadow-default dark:border-strokedark dark:bg-boxdark sm:px-7.5 xl:col-span-8">
      <div className="flex flex-wrap items-start justify-between gap-3 sm:flex-nowrap">
        <div className="flex w-full flex-wrap gap-3 sm:gap-5">
          <div className="flex min-w-47.5">
            <span className="mr-2 mt-1 flex h-4 w-full max-w-4 items-center justify-center rounded-full border border-primary">
              <span className="block h-2.5 w-full max-w-2.5 rounded-full bg-primary"></span>
            </span>
            <div className="w-full">
              <p className="font-semibold text-primary">Total Revenue</p>
              <p className="text-sm font-medium">12.04.2022 - 12.05.2022</p>
            </div>
          </div>
          <div className="flex min-w-47.5">
            <span className="mr-2 mt-1 flex h-4 w-full max-w-4 items-center justify-center rounded-full border border-secondary">
              <span className="block h-2.5 w-full max-w-2.5 rounded-full bg-secondary"></span>
            </span>
            <div className="w-full">
              <p className="font-semibold text-secondary">Total Sales</p>
              <p className="text-sm font-medium">12.04.2022 - 12.05.2022</p>
            </div>
          </div>
        </div>
        <div className="flex w-full max-w-45 justify-end">
          <div className="inline-flex items-center rounded-md bg-whiter p-1.5 dark:bg-meta-4">
            <button className="rounded bg-white px-3 py-1 text-xs font-medium text-black shadow-card hover:bg-white hover:shadow-card dark:bg-boxdark dark:text-white dark:hover:bg-boxdark">
              Day
            </button>
            <button className="rounded px-3 py-1 text-xs font-medium text-black hover:bg-white hover:shadow-card dark:text-white dark:hover:bg-boxdark">
              Week
            </button>
            <button className="rounded px-3 py-1 text-xs font-medium text-black hover:bg-white hover:shadow-card dark:text-white dark:hover:bg-boxdark">
              Month
            </button>
          </div>
        </div>
      </div>

      <div>
        <div id="chartOne" className="-ml-5">
          <ReactApexChart
            options={options}
            series={state.series}
            type="line"
            height={350}
            width={"100%"}
          />
        </div>
      </div>
    </div>
  );
};

export default ChartOne;
