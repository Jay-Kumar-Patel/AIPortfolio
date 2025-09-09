// next.config.mjs
/** @type {import('next').NextConfig} */
const repo = 'Portfolio';
const isProd = process.env.NODE_ENV === 'production';

export default {
  output: 'export',
  basePath: isProd ? `/${repo}` : '',
  assetPrefix: isProd ? `/${repo}/` : '',
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_BASE_PATH: isProd ? `/${repo}` : '',
  },
};
