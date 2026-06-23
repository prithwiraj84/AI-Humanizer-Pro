"use client";

import dynamic from "next/dynamic";

// WebGL can't render on the server — load the canvas client-side only.
const SceneCanvas = dynamic(() => import("./SceneCanvas"), { ssr: false });

export default function SceneBackground() {
  return <SceneCanvas />;
}
