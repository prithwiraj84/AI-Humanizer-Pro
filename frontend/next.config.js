/** @type {import('next').NextConfig} */

// All browser calls go to /backend/* (same-origin) and are transparently
// proxied to the FastAPI server. This keeps the session cookie same-origin
// (no CORS, no SameSite=None/Secure gymnastics) and avoids any collision
// between Next.js page routes (/login, /dashboard, ...) and the API.
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:5001";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: "/backend/:path*", destination: `${BACKEND_URL}/:path*` },
    ];
  },
};

module.exports = nextConfig;
