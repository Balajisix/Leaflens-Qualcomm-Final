import type { NextConfig } from "next";
const { i18n } = require('./next-i18next.config');
const webpack = require('webpack');


const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: true,
  i18n,
  webpack: (config) => {
    config.resolve = {
      ...config.resolve,
      fallback: {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      },
    };
    return config;
  },
  // Enable experimental features if needed
  experimental: {
    serverComponentsExternalPackages: ['@tensorflow/tfjs-tflite'],
  },
};

export default nextConfig;
