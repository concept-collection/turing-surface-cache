/** Screenshot the app (dist/) in headless Chrome after the boot-time solve
 *  finishes. Talks to the real cache.
 *  Usage: node scripts/screenshot.mjs out.png [light|dark] [tEnd] */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const out = process.argv[2] ?? 'demo.png';
const scheme = process.argv[3] ?? 'light';
const tEnd = process.argv[4] ?? '100';
const DIST = new URL('../dist/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const server = createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/index.html' : req.url.split('?')[0];
    const data = await readFile(join(DIST, path));
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu',
    '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'],
  // A wait is one CDP call lasting as long as the wait, so the default 180 s
  // cap on a call is what a slow run trips over first.
  protocolTimeout: 900_000,
});
const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 900 });
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
page.on('console', (m) => console.log('  [page]', m.text()));
// The ?tend hook accepts any end time, listed or not, so a short test run
// can be screenshotted too.
await page.goto(`http://127.0.0.1:${port}/index.html?tend=${tEnd}`, { waitUntil: 'load' });
await page.waitForSelector('#tend');
// Terminal statuses only — "checking the cloud cache…" is transient. On a
// miss the page settles on empty windows; press the button so the screenshot
// shows a pattern either way.
const idle = /from the cloud cache|press Compute solution|failed/;
await page.waitForFunction(
  (re) => new RegExp(re).test(document.getElementById('status')?.textContent ?? '') ||
    (document.getElementById('err')?.textContent?.length ?? 0) > 4,
  { timeout: 600_000 },
  idle.source,
);
if (/press Compute solution/.test(await page.$eval('#status', (el) => el.textContent))) {
  await page.click('#solve');
  await page.waitForFunction(
    () => /Not uploaded|Uploaded \d|from the cloud cache|failed/.test(
      document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
}
await new Promise((r) => setTimeout(r, 300));
await page.screenshot({ path: out });
console.log('screenshot:', out);
console.log('status:', await page.$eval('#status', (el) => el.textContent));
const err = await page.$eval('#err', (el) => el.textContent);
if (err) console.log('err:', err);
await browser.close();
server.close();
