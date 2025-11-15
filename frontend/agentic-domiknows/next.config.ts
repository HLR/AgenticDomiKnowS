import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  allowedDevOrigins: ['127.0.0.1', 'localhost', 'hlr-demo.egr.msu.edu', '*.hlr-demo.egr.msu.edu'],
  // Increase timeout for API requests (default is 30s, increase to 60s)
  experimental: {
    proxyTimeout: 60000, // 60 seconds in milliseconds
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8001/:path*',
      },
    ];
  },
};

export default nextConfig;

