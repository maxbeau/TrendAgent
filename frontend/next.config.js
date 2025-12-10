/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const fallbackBackend = 'http://127.0.0.1:8000';
    const target = (process.env.BACKEND_API_URL || fallbackBackend).replace(/\/+$/, '');
    return [
      {
        source: '/api/:path*',
        destination: `${target}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
