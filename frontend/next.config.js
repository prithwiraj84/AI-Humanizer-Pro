/** @type {import('next').NextConfig} */

// All browser calls go to /backend/* (same-origin) and are transparently
// proxied to the FastAPI server. This keeps the session cookie same-origin
// (no CORS, no SameSite=None/Secure gymnastics) and avoids any collision
// between Next.js page routes (/login, /dashboard, ...) and the API.
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:5001";

// Single-server deploy: STATIC_EXPORT=true emits static files (frontend/out)
// that FastAPI serves. In that mode there is no proxy — the app calls the API
// same-origin (set NEXT_PUBLIC_API_BASE="").
const isExport = process.env.STATIC_EXPORT === "true";

const nextConfig = {
  reactStrictMode: true,
  ...(isExport ? { output: "export" } : {}),
  async rewrites() {
    if (isExport) return [];
    return [
      { source: "/backend/:path*", destination: `${BACKEND_URL}/:path*` },
    ];
  },
};

module.exports = nextConfig;
