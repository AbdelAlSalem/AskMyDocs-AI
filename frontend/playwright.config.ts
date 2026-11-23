import { defineConfig } from '@playwright/test'

const address = 'http://127.0.0.1:4178'

export default defineConfig({
  testDir: './tests',
  outputDir: '../tmp/frontend-suggestions',
  use: { baseURL: address, channel: process.env.PLAYWRIGHT_CHANNEL },
  webServer: {
    command: 'npm run dev -- --port 4178',
    url: address,
    env: { VITE_API_URL: address },
  },
})
