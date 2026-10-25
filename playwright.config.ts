import { defineConfig, devices } from "@playwright/test";

// Browser tests run the real Streamlit app with a rule-based stand-in for the AI provider
// (tests/e2e/fake_llm_app.py), so they need no API keys and give the same answers every run.
const PORT = 8599;
const viewport = { width: 1280, height: 900 };

export default defineConfig({
  testDir: "tests/e2e",
  outputDir: "test-results/artifacts",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  // One app server serves every test; each test gets its own browser session.
  workers: 1,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"], viewport } },
    { name: "firefox", use: { ...devices["Desktop Firefox"], viewport } },
    { name: "webkit", use: { ...devices["Desktop Safari"], viewport } },
  ],
  webServer: {
    command:
      "uv run python -m tests.e2e.make_fixtures && " +
      `uv run streamlit run tests/e2e/fake_llm_app.py --server.port ${PORT} --server.headless true ` +
      "--server.fileWatcherType none --browser.gatherUsageStats false",
    url: `http://127.0.0.1:${PORT}/_stcore/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    // Only warnings and errors from the app, so test output stays readable.
    env: { LOG_LEVEL: "WARNING" },
    stdout: "ignore",
    stderr: "pipe",
  },
});
