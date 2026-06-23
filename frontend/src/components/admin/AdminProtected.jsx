"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { api } from "@/lib/api";

// Gates admin pages on the separate admin session (session.admin_authenticated).
export default function AdminProtected({ children }) {
  const router = useRouter();
  const [state, setState] = useState("checking"); // checking | ok

  useEffect(() => {
    let active = true;
    api
      .adminCheck()
      .then(() => active && setState("ok"))
      .catch(() => router.replace("/admin/login"));
    return () => {
      active = false;
    };
  }, [router]);

  if (state !== "ok") {
    return (
      <div className="grid min-h-screen min-h-[100svh] place-items-center">
        <Loader2 className="h-8 w-8 animate-spin text-brand-indigo" />
      </div>
    );
  }
  return children;
}
