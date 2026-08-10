/**
 * Smoke-check a deployed URL in headless Chrome: load it and wait for the
 * boot-time solve to finish — either from the cloud cache or computed locally.
 * Talks to the real cache. Usage: node scripts/check-live.mjs [url]
 */
import puppeteer from 'puppeteer-core';

const url = process.argv[2] ?? 'https://concept-collection.github.io/turing-surface-cache/';
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu',
    '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 900 });
const problems = [];
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
page.on('requestfailed', (r) => {
  // The cache lookup 404s by design when the selection is not cached.
  if (!r.url().startsWith('https://tempory.net/')) {
    problems.push(`request failed: ${r.url()}`);
  }
});
page.on('console', (m) => {
  if (m.type() === 'error' && !/GL Driver|favicon|tempory\.net/.test(m.text())) {
    problems.push(`console error: ${m.text()}`);
  }
});

try {
  await page.goto(url, { waitUntil: 'load', timeout: 60_000 });
  // Nothing computes without the button, so any selection is safe to idle on.
  await page.waitForSelector('#tend');
  // Terminal statuses only — "checking the cloud cache…" is transient. A
  // miss settles on empty windows asking for the button; nothing computes
  // during this check.
  await page.waitForFunction(
    () => /from the cloud cache|press Compute solution|failed/.test(
      document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  console.log('status:', await page.$eval('#status', (el) => el.textContent));
  const err = await page.$eval('#err', (el) => el.textContent);
  if (err) problems.push(`err: ${err}`);
  const panels = await page.$$eval('.sphere-box canvas', (els) => els.length);
  console.log('sphere canvases:', panels);
  if (panels !== 2) problems.push(`expected 2 sphere canvases, got ${panels}`);
  if (problems.length) {
    console.log('PROBLEMS:');
    for (const p of new Set(problems)) console.log('  ' + p);
    process.exitCode = 1;
  } else {
    console.log('LIVE CHECK: PASS');
  }
} catch (e) {
  console.error(`LIVE CHECK FAIL: ${e.message}`);
  for (const p of new Set(problems)) console.error('  ' + p);
  process.exitCode = 1;
} finally {
  await browser.close();
}
