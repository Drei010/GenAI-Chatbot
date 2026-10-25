import { expect, test, type Locator, type Page, type TestInfo } from "@playwright/test";
import path from "node:path";

// Every test opens a fresh browser session against the app in tests/e2e/fake_llm_app.py, which answers
// from the uploaded PDF with rules instead of a real model. Screenshots of each state are saved to
// test-results/screenshots/<browser>/ for review.

const FIXTURES = path.join(__dirname, "fixtures", "generated");
const NORTHWIND = "northwind-annual-report.pdf";
const HARBOR_TEA = "harbor-tea-report.pdf";
const NOT_A_PDF = "notes.pdf";

const chatBox = (page: Page) => page.getByTestId("stChatInputTextArea");
const userMessages = (page: Page) => page.getByRole("group", { name: "Chat message from user" });
const assistantMessages = (page: Page) => page.getByRole("group", { name: "Chat message from assistant" });
const resetButton = (page: Page) => page.getByRole("button", { name: "Reset chat and document" });

async function snap(page: Page, testInfo: TestInfo, name: string): Promise<void> {
  await page.screenshot({
    path: path.join("test-results", "screenshots", testInfo.project.name, `${name}.png`),
    fullPage: true,
  });
}

async function openApp(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.getByText("Upload a PDF in the sidebar to get started.")).toBeVisible({ timeout: 60_000 });
}

async function upload(page: Page, filename: string): Promise<void> {
  await page.getByTestId("stFileUploaderDropzoneInput").setInputFiles(path.join(FIXTURES, filename));
}

async function uploadAndWait(page: Page, filename: string, pages: number): Promise<void> {
  await upload(page, filename);
  await expect(page.getByText(`${filename} is ready: ${pages} pages read.`)).toBeVisible({ timeout: 60_000 });
  await expect(chatBox(page)).toBeEnabled();
}

/** Ask a question and return the assistant's reply once the app has finished answering. */
async function ask(page: Page, question: string): Promise<Locator> {
  const asked = await userMessages(page).count();
  await chatBox(page).fill(question);
  await chatBox(page).press("Enter");
  await expect(userMessages(page)).toHaveCount(asked + 1);
  await expect(userMessages(page).last()).toHaveText(question);
  // The box is disabled while the app works on a question and comes back when the answer is shown.
  await expect(chatBox(page)).toBeEnabled({ timeout: 60_000 });
  await expect(assistantMessages(page)).toHaveCount(asked + 1);
  return assistantMessages(page).last();
}

async function expectSource(reply: Locator, number: number, filename: string, pageNumber: number) {
  await expect(reply.getByText("Sources", { exact: true })).toBeVisible();
  await expect(reply.getByText(`[${number}] ${filename}, page ${pageNumber}`, { exact: true })).toBeVisible();
}

test("answers a question with its source, then a follow-up", async ({ page }, testInfo) => {
  await openApp(page);
  await expect(chatBox(page)).toBeDisabled();
  await expect(chatBox(page)).toHaveAttribute("placeholder", "Upload a PDF to start asking questions");
  await snap(page, testInfo, "01-before-upload");

  await uploadAndWait(page, NORTHWIND, 5);
  await expect(chatBox(page)).toHaveAttribute("placeholder", "Ask a question about the document");
  await snap(page, testInfo, "02-document-ready");

  const reply = await ask(page, "How many people did Northwind employ?");
  await expect(reply).toContainText("Northwind employed 2,340 people");
  await expectSource(reply, 1, NORTHWIND, 3);
  await expect(reply.getByText("“Northwind employed 2,340 people as of June 30, 2024")).toBeVisible();
  await snap(page, testInfo, "03-answer-with-sources");

  const followUp = await ask(page, "And how many retail stores did it operate?");
  await expect(followUp).toContainText("186 retail stores");
  await expectSource(followUp, 1, NORTHWIND, 3);
  await expect(assistantMessages(page)).toHaveCount(2);
  await snap(page, testInfo, "04-follow-up-answer");
});

test("shows summaries as bullet points and comparisons as tables", async ({ page }, testInfo) => {
  await openApp(page);
  await uploadAndWait(page, NORTHWIND, 5);

  const summary = await ask(page, "Summarize the key facts");
  expect(await summary.locator("li").count()).toBeGreaterThanOrEqual(3);
  await expect(summary.getByText("Sources", { exact: true })).toBeVisible();
  await snap(page, testInfo, "05-summary-bullets");

  const comparison = await ask(page, "Compare revenue by segment in a table");
  const table = comparison.locator("table");
  await expect(table).toBeVisible();
  await expect(table.locator("th")).toHaveText(["Item", "Figures", "Source"]);
  await expect(table.locator("tr", { hasText: "Retail stores" })).toContainText("$ 251.4");
  await expectSource(comparison, 1, NORTHWIND, 2);
  await snap(page, testInfo, "06-comparison-table");
});

test("says so when the document doesn't answer the question", async ({ page }, testInfo) => {
  await openApp(page);
  await uploadAndWait(page, NORTHWIND, 5);

  const reply = await ask(page, "What is the capital of France?");

  await expect(reply).toContainText("I couldn't find an answer to that in the document.");
  await expect(reply.getByText("Sources", { exact: true })).toHaveCount(0);
  await snap(page, testInfo, "07-unsupported-question");
});

test("explains why a file that isn't a real PDF can't be used", async ({ page }, testInfo) => {
  await openApp(page);

  await upload(page, NOT_A_PDF);

  await expect(page.getByText("This file isn't a PDF. Upload a document saved in PDF format.")).toBeVisible({
    timeout: 60_000,
  });
  await expect(chatBox(page)).toBeDisabled();
  await snap(page, testInfo, "08-invalid-pdf");
});

test("reset clears the conversation and the document", async ({ page }, testInfo) => {
  await openApp(page);
  await uploadAndWait(page, NORTHWIND, 5);
  await ask(page, "How many people did Northwind employ?");
  await snap(page, testInfo, "09-before-reset");

  await resetButton(page).click();

  await expect(page.getByText("Upload a PDF in the sidebar to get started.")).toBeVisible();
  await expect(assistantMessages(page)).toHaveCount(0);
  await expect(userMessages(page)).toHaveCount(0);
  await expect(page.getByTestId("stFileChip")).toHaveCount(0);
  await expect(chatBox(page)).toBeDisabled();
  await expect(resetButton(page)).toBeDisabled();
  await snap(page, testInfo, "10-after-reset");
});

test("replacing the document starts a fresh conversation about the new file", async ({ page }, testInfo) => {
  await openApp(page);
  await uploadAndWait(page, NORTHWIND, 5);
  await ask(page, "How many people did Northwind employ?");
  await snap(page, testInfo, "11-before-replacement");

  await uploadAndWait(page, HARBOR_TEA, 2);
  await expect(assistantMessages(page)).toHaveCount(0);
  await expect(page.getByText(`${NORTHWIND} is ready`)).toHaveCount(0);

  const reply = await ask(page, "How many people did Harbor Tea employ?");
  await expect(reply).toContainText("85 people");
  await expectSource(reply, 1, HARBOR_TEA, 1);
  await snap(page, testInfo, "12-after-replacement");
});
