"use client";
import React, { useState, ReactNode } from "react";
import Sidebar from "@/components/Sidebar";
import { Header, NoAuthHeader } from "@/components/Header";
import { useSession } from "next-auth/react";
export default function DefaultLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { data: session, status } = useSession();
  /* console.log(session); */
  return (
    <>
      {/* <!-- ===== Page Wrapper Start ===== --> */}
      <div className="flex h-screen overflow-hidden">
        {/* <!-- ===== Sidebar Start ===== --> */}
        {status == "authenticated" ? (
          <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
        ) : null}
        {/* {status == "authenticated"? (
          <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
        )} */}

        {/* <!-- ===== Sidebar End ===== --> */}

        {/* <!-- ===== Content Area Start ===== --> */}
        <div className="relative flex flex-1 flex-col overflow-y-auto overflow-x-hidden">
          {/* <!-- ===== Header Start ===== --> */}
          {status == "authenticated" ? (
            <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
          ) : (
            <NoAuthHeader />
          )}

          {/* <!-- ===== Header End ===== --> */}

          {/* <!-- ===== Main Content Start ===== --> */}
          <main>
            <div className="mx-auto max-w-screen-2xl p-4 md:p-6 2xl:p-10">
              {children}
            </div>
          </main>
          {/* <!-- ===== Main Content End ===== --> */}
        </div>
        {/* <!-- ===== Content Area End ===== --> */}
      </div>
      {/* <!-- ===== Page Wrapper End ===== --> */}
    </>
  );
}
