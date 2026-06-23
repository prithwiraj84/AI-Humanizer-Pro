"use client";

import { MotionConfig } from "framer-motion";
import { AuthProvider } from "@/context/AuthContext";

// reducedMotion="user" makes all Framer Motion animations respect the OS
// "reduce motion" setting (WCAG 2.3.3).
export default function Providers({ children }) {
  return (
    <MotionConfig reducedMotion="user">
      <AuthProvider>{children}</AuthProvider>
    </MotionConfig>
  );
}
