import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { defineConfig, devices } from '@playwright/test'

const root = resolve(__dirname, '../..')
const frontend = join(root, 'frontend')
const backend = join(root, 'backend/server')
const dataDir = mkdtempSync(join(tmpdir(), 'render2d-product-benchmark-'))
const frontendPort = 5195
const backendPort = 8195

export default defineConfig({
  testDir: __dirname,
  testMatch: 'plan-flow.spec.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 180_000,
  expect: { timeout: 20_000 },
  reporter: [['list']],
  use: {
    ...devices['Desktop Chrome'],
    baseURL: `http://localhost:${frontendPort}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    viewport: { width: 1400, height: 900 },
    serviceWorkers: 'block',
  },
  webServer: [
    {
      command: 'cargo run --bin render2d -- serve',
      cwd: backend,
      url: `http://localhost:${backendPort}/api/health`,
      reuseExistingServer: false,
      timeout: 600_000,
      env: {
        PORT: String(backendPort),
        DATA_DIR: dataDir,
        LOG_DIR: join(dataDir, 'logs'),
        STORAGE_BACKEND: 'files',
        MOCK_LLM: '1',
        OPENROUTER_API_KEY: '',
        RAWTREE_API_KEY: '',
        SUPABASE_URL: '',
        SUPABASE_JWT_SECRET: '',
        SUPABASE_ANON_KEY: '',
        SUPABASE_SERVICE_ROLE_KEY: '',
        BLOB_READ_WRITE_TOKEN: '',
        DAILY_LLM_CAP: '0',
        USAGE_ENFORCE: '0',
        GATE_SHARED_SECRET: '',
      },
    },
    {
      command: `npx vite --port ${frontendPort} --strictPort --host localhost`,
      cwd: frontend,
      url: `http://localhost:${frontendPort}/`,
      reuseExistingServer: false,
      env: { BACKEND_PORT: String(backendPort), GATE_DEV_ORIGIN: '' },
    },
  ],
})
