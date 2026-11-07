import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { expect, test } from '@playwright/test'

const BRIEF = 'A one-storey 1,200 sq ft house with 3 bedrooms and 2 bathrooms'

test('rules-mode generation reaches a visible editor result without browser errors', async ({ page }) => {
  const startedAt = performance.now()
  const browserErrors: string[] = []
  page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
  page.on('console', (message) => {
    if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`)
  })

  await page.goto('/')
  await expect(page.getByTestId('home-ask')).toBeVisible()
  const initialRenderMs = Math.round(performance.now() - startedAt)

  const generationStartedAt = performance.now()
  await page.getByTestId('home-ask').fill(BRIEF)
  await page.getByTestId('home-send').click()
  const editor = page.getByTestId('view-3d')
  const generate = page.getByTestId('intent-generate')
  await expect(editor.or(generate).first()).toBeVisible({ timeout: 60_000 })
  if (await generate.isVisible()) await generate.click()
  await expect(page.getByRole('button', { name: '3D', exact: true })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('code-chip')).toBeVisible({ timeout: 30_000 })
  const generationToEditorMs = Math.round(performance.now() - generationStartedAt)

  const receipt = {
    schema_version: 1,
    mode: 'rules',
    initial_render_ms: initialRenderMs,
    generation_to_editor_ms: generationToEditorMs,
    browser_errors: browserErrors,
  }
  await test.info().attach('browser-benchmark.json', {
    body: JSON.stringify(receipt, null, 2),
    contentType: 'application/json',
  })
  if (process.env.BENCHMARK_RESULT) {
    mkdirSync(dirname(process.env.BENCHMARK_RESULT), { recursive: true })
    writeFileSync(process.env.BENCHMARK_RESULT, `${JSON.stringify(receipt, null, 2)}\n`)
  }

  expect(browserErrors).toEqual([])
})
