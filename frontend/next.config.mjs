/** @type {import('next').NextConfig} */
// empty = API on the same origin (single-container deployment); 'self' then covers both https and wss
const api = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const ws = api.replace(/^http/, "ws");
const dev = process.env.NODE_ENV !== "production";

const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline' blob:${dev ? " 'unsafe-eval'" : ""}`, // Next inline bootstrap; blob: for the audio worklet
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "media-src 'self' blob:",
  `connect-src 'self'${api ? ` ${api} ${ws}` : ""}${dev ? " ws://localhost:3000" : ""}`,
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  output: "standalone",
  productionBrowserSourceMaps: false,
  async headers() {
    return [{
      source: "/(.*)",
      headers: [
        { key: "Content-Security-Policy", value: csp },
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "Referrer-Policy", value: "no-referrer" },
        { key: "X-Frame-Options", value: "DENY" },
        { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
        { key: "Permissions-Policy", value: "microphone=(self), camera=(), geolocation=(), payment=()" },
      ],
    }];
  },
};
export default nextConfig;
