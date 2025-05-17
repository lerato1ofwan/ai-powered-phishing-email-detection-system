import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    env: {
      API_ENDPOINT: process.env.API_ENDPOINT,
    }
};

export default nextConfig;
