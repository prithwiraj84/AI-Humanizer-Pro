"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import {
  Float,
  Icosahedron,
  MeshDistortMaterial,
  Sparkles,
  Stars,
  Torus,
} from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const handler = (e) => setReduced(e.matches);
    mq.addEventListener?.("change", handler);
    return () => mq.removeEventListener?.("change", handler);
  }, []);
  return reduced;
}

// Smoothed, normalized page-scroll progress (0 at top, 1 at bottom).
function useScroll() {
  const value = useRef(0);
  useFrame(() => {
    const max =
      document.documentElement.scrollHeight - window.innerHeight || 1;
    const target = Math.min(1, Math.max(0, window.scrollY / max));
    value.current = THREE.MathUtils.lerp(value.current, target, 0.08);
  });
  return value;
}

function CoreBlob({ scroll }) {
  const mesh = useRef();
  const mat = useRef();
  const colorA = useMemo(() => new THREE.Color("#6366f1"), []);
  const colorB = useMemo(() => new THREE.Color("#22d3ee"), []);
  const tmp = useMemo(() => new THREE.Color(), []);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    const p = scroll.current;
    if (mesh.current) {
      mesh.current.rotation.y = t * 0.18 + p * Math.PI * 2;
      mesh.current.rotation.z = p * 0.8;
      const s = 1 + Math.sin(t * 0.6) * 0.04 + p * 0.25;
      mesh.current.scale.setScalar(s);
    }
    if (mat.current) {
      tmp.copy(colorA).lerp(colorB, (Math.sin(p * Math.PI) + 1) / 2);
      mat.current.color.copy(tmp);
      mat.current.distort = 0.35 + p * 0.25;
    }
  });

  return (
    <Icosahedron ref={mesh} args={[1.35, 12]} position={[0, 0, 0]}>
      <MeshDistortMaterial
        ref={mat}
        color="#6366f1"
        emissive="#3b0a72"
        emissiveIntensity={0.45}
        roughness={0.15}
        metalness={0.65}
        distort={0.4}
        speed={1.6}
      />
    </Icosahedron>
  );
}

function Halo({ scroll }) {
  const ring = useRef();
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (ring.current) {
      ring.current.rotation.x = Math.PI / 2.6 + t * 0.05;
      ring.current.rotation.z = t * 0.12 + scroll.current * 1.5;
    }
  });
  return (
    <Torus ref={ring} args={[2.6, 0.015, 16, 120]}>
      <meshStandardMaterial
        color="#a855f7"
        emissive="#a855f7"
        emissiveIntensity={1.4}
        toneMapped={false}
      />
    </Torus>
  );
}

const SHARDS = [
  { pos: [-3.1, 1.6, -1], c: "#22d3ee", s: 0.32 },
  { pos: [3.2, -1.2, -0.5], c: "#ec4899", s: 0.26 },
  { pos: [2.6, 2.1, -2], c: "#818cf8", s: 0.2 },
  { pos: [-2.7, -1.8, -1.5], c: "#a855f7", s: 0.24 },
  { pos: [0.4, 2.8, -2.5], c: "#34d399", s: 0.18 },
];

function Shards() {
  return SHARDS.map((sh, i) => (
    <Float
      key={i}
      speed={1.4 + i * 0.2}
      rotationIntensity={1.2}
      floatIntensity={1.6}
    >
      <Icosahedron args={[sh.s, 0]} position={sh.pos}>
        <meshStandardMaterial
          color={sh.c}
          emissive={sh.c}
          emissiveIntensity={0.7}
          roughness={0.2}
          metalness={0.6}
          flatShading
        />
      </Icosahedron>
    </Float>
  ));
}

function Rig({ scroll }) {
  useFrame((state) => {
    const p = scroll.current;
    state.camera.position.z = 6 - p * 1.6;
    state.camera.position.x = Math.sin(p * Math.PI) * 1.1;
    state.camera.position.y = p * 0.6;
    state.camera.lookAt(0, 0, 0);
  });
  return null;
}

function Experience() {
  const scroll = useScroll();
  return (
    <>
      <ambientLight intensity={0.35} />
      <pointLight position={[6, 5, 5]} intensity={120} color="#6366f1" />
      <pointLight position={[-6, -3, 3]} intensity={90} color="#22d3ee" />
      <pointLight position={[0, 4, -5]} intensity={70} color="#a855f7" />

      <Rig scroll={scroll} />
      <CoreBlob scroll={scroll} />
      <Halo scroll={scroll} />
      <Shards />

      <Sparkles
        count={70}
        scale={[14, 10, 10]}
        size={2.2}
        speed={0.3}
        color="#a5b4fc"
        opacity={0.7}
      />
      <Stars
        radius={60}
        depth={40}
        count={1800}
        factor={3}
        saturation={0}
        fade
        speed={0.6}
      />
    </>
  );
}

export default function SceneCanvas() {
  const reduced = usePrefersReducedMotion();
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10">
      <Canvas
        dpr={[1, 1.8]}
        frameloop={reduced ? "demand" : "always"}
        camera={{ position: [0, 0, 6], fov: 45 }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      >
        <Experience />
      </Canvas>
      {/* Depth + contrast overlay so foreground text stays readable */}
      <div className="absolute inset-0 bg-gradient-to-b from-ink-950/40 via-ink-950/10 to-ink-950/80" />
    </div>
  );
}
