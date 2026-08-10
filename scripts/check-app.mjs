/**
 * End-to-end check of the cache flow in headless Chrome (SwiftShader WebGPU),
 * without touching the real cloud cache:
 *
 *   1. cache miss — the page is loaded with the real tempory.net requests
 *      intercepted to 404, the end time is set to 5, and the run computes
 *      locally; the produced .h5 is pulled out of the download link.
 *   2. the .h5 is checked with Python h5py (layout, shapes, spec_json).
 *   3. cache hit — a fresh page, same selection, with the interception now
 *      answering that .h5; the page must show "from the cloud cache".
 *   4. warm start — a fresh page asking for a longer end time, with only the
 *      shorter run's .h5 in the "cache"; the page must resume from it and
 *      compute only the remainder.
 *
 * The pages are opened with the ?tend= test hook so the computed runs stay
 * short (5 and 10 time units instead of the UI's 100+).
 *
 * Usage: node scripts/check-app.mjs
 */
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const DIST = new URL('../dist/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const T_END = '5';
const T_END_LONG = '10';

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
});

const problems = [];
function watch(page, tag) {
  page.on('pageerror', (e) => problems.push(`${tag} pageerror: ${e.message}`));
  page.on('console', (m) => {
    // Cache-lookup 404s log as resource errors by design; ignore them.
    if (
      m.type() === 'error' &&
      !/GL Driver|favicon|tempory\.net|Failed to load resource/.test(m.text())
    ) {
      problems.push(`${tag} console error: ${m.text()}`);
    }
  });
}

/** Intercept tempory.net cache reads; `bytes` null => everything 404s. The
 *  mocked responses need the CORS header the real bucket sends, or the page's
 *  cross-origin fetch is blocked before it sees the status. */
const CORS = { 'access-control-allow-origin': '*' };
async function interceptCache(page, bytes, name) {
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const url = req.url();
    if (!url.startsWith('https://tempory.net/')) return void req.continue();
    if (bytes && name && url.endsWith(`/${name}`)) {
      if (req.method() === 'HEAD') return void req.respond({ status: 200, headers: CORS });
      return void req.respond({
        status: 200,
        headers: CORS,
        contentType: 'application/x-hdf5',
        body: Buffer.from(bytes),
      });
    }
    req.respond({ status: 404, headers: CORS, body: 'not found' });
  });
}

/** Open the app. `tend` selects an end time through the dropdown; a `hash`
 *  instead carries the selection in the URL fragment, exercising the
 *  reload/share restore path. `hookOrder` sets the ?tend test list — its
 *  first entry is the default selection, so a restore test must pass an
 *  order whose default differs from the hash value, or a restore that
 *  silently does nothing would still land on the right selection. */
async function openAndSelect(page, tend, hash = '', hookOrder = `${T_END},${T_END_LONG}`) {
  await page.setViewport({ width: 1100, height: 900 });
  // The ?tend hook replaces the end-time list with short test values.
  await page.goto(
    `http://127.0.0.1:${port}/index.html?tend=${hookOrder}${hash}`,
    { waitUntil: 'load' },
  );
  // Selection changes land long before the WebGPU compile finishes, so the
  // boot-time auto-refresh picks them up.
  await page.waitForSelector('#tend');
  if (tend !== null) await page.select('#tend', tend);
}

const statusOf = (page) => page.$eval('#status', (el) => el.textContent);
const errOf = (page) => page.$eval('#err', (el) => el.textContent);

try {
  // ---- pass 1: miss, compute locally --------------------------------------
  const page1 = await browser.newPage();
  watch(page1, 'miss:');
  await interceptCache(page1, null, '');
  await openAndSelect(page1, T_END);
  // A miss never computes on its own: the page must settle on empty windows
  // asking for the button.
  await page1.waitForFunction(
    () => /press Compute solution|failed/.test(
      document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  console.log('pass 1 idle status:', await statusOf(page1));
  const solveDisabled = await page1.$eval('#solve', (b) => b.disabled);
  if (solveDisabled) problems.push('miss: Compute solution button disabled while idle');
  await page1.click('#solve');
  // "Not uploaded" is the terminal state of a keyless local run — it appears
  // only after the .h5 has been encoded and the download link filled in.
  await page1.waitForFunction(
    () => /Not uploaded|failed/.test(document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  const s1 = await statusOf(page1);
  console.log('pass 1 status:', s1);
  const e1 = await errOf(page1);
  if (e1) problems.push(`miss: err: ${e1}`);
  if (!/computed locally/.test(s1)) problems.push(`miss: unexpected status: ${s1}`);
  if (!/t = 5\b/.test(s1)) problems.push(`miss: did not stop at t = 5: ${s1}`);

  const fileName = await page1.$eval('#download', (a) => a.download);
  const b64 = await page1.$eval('#download', async (a) => {
    const buf = await (await fetch(a.href)).arrayBuffer();
    let out = '';
    const v = new Uint8Array(buf);
    for (let i = 0; i < v.length; i += 0x8000) {
      out += String.fromCharCode(...v.subarray(i, i + 0x8000));
    }
    return btoa(out);
  });
  const bytes = Buffer.from(b64, 'base64');
  console.log(`downloaded ${fileName}: ${bytes.length} bytes`);
  if (!/^[0-9a-f]{64}\.h5$/.test(fileName)) problems.push(`odd file name: ${fileName}`);
  // The selection is mirrored into the URL fragment on every change.
  if (!new URL(page1.url()).hash.includes(`tend=${T_END}`)) {
    problems.push(`miss: selection not in URL: ${page1.url()}`);
  }
  const h5Path = `/tmp/turing-surface-cache-check.h5`;
  await writeFile(h5Path, bytes);
  await page1.close();

  // ---- pass 2: the .h5 itself, via h5py -----------------------------------
  try {
    const out = execFileSync('python3', ['-c', `
import h5py, json, sys
f = h5py.File('${h5Path}', 'r')
spec = json.loads(f.attrs['spec_json'])
assert f.attrs['app'] == 'turing-surface-cache', f.attrs['app']
assert int(f.attrs['format_version']) == 1
assert spec['tEnd'] == 5 and spec['model'] == 'schnakenberg', spec
assert int(f['spec'].attrs['steps']) == round(5 / spec['params']['dt'])
nlm = (spec['lmax'] + 1) * (spec['lmax'] + 2) // 2
for g in ('geometry/Gx', 'geometry/Gy', 'geometry/Gz', 'initial/U', 'initial/V', 'final/U', 'final/V'):
    d = f[g]
    assert d.shape == (2 * nlm,) and d.dtype.kind == 'f', (g, d.shape, d.dtype)
import numpy as np
assert np.isfinite(f['final/U'][:]).all() and np.abs(f['final/U'][:]).max() > 0
print('h5py check ok; species', list(f.attrs['species']), '; adapter:', f.attrs.get('adapter', '?'))
`], { encoding: 'utf8' });
    console.log('pass 2:', out.trim());
  } catch (e) {
    problems.push(`h5py check failed: ${e.stdout ?? ''}${e.stderr ?? e.message}`);
  }

  // ---- pass 3: hit, load from "cache", selection restored from the URL ----
  const page2 = await browser.newPage();
  watch(page2, 'hit:');
  await interceptCache(page2, bytes, fileName);
  // No dropdown interaction: the end time arrives in the fragment, as it
  // would from a shared or reloaded link. The hook default is deliberately
  // the OTHER value, so only a working restore reaches the cached spec.
  await openAndSelect(page2, null, `#tend=${T_END}`, `${T_END_LONG},${T_END}`);
  const restored = await page2.$eval('#tend', (el) => el.value);
  if (restored !== T_END) problems.push(`hit: URL restore failed, tend = ${restored}`);
  // "from the cloud cache" is the hit's terminal status; "checking the cloud
  // cache…" is transient and must not satisfy the wait.
  await page2.waitForFunction(
    () => /from the cloud cache|Not uploaded|failed/.test(
      document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  const s2 = await statusOf(page2);
  console.log('pass 3 status:', s2);
  const e2 = await errOf(page2);
  if (e2) problems.push(`hit: err: ${e2}`);
  if (!/from the.*cloud cache/.test(s2)) problems.push(`hit: expected a cache hit: ${s2}`);
  const note = await page2.$eval('#cachenote', (el) => el.textContent);
  if (!/in the cloud cache/.test(note)) problems.push(`hit: cache note wrong: '${note}'`);
  const panels = await page2.$$eval('.sphere-box canvas', (els) => els.length);
  if (panels !== 2) problems.push(`hit: expected 2 sphere canvases, got ${panels}`);
  await page2.close();

  // ---- pass 4: warm start from the shorter cached run ----------------------
  // Only the t = 5 file is in the "cache"; asking for t = 10 must resume from
  // it and compute just the remainder.
  const page3 = await browser.newPage();
  watch(page3, 'warm:');
  await interceptCache(page3, bytes, fileName);
  await openAndSelect(page3, T_END_LONG);
  await page3.waitForFunction(
    () => /press Compute solution|failed/.test(
      document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  await page3.click('#solve');
  await page3.waitForFunction(
    () => /Not uploaded|failed/.test(document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  const s3 = await statusOf(page3);
  console.log('pass 4 status:', s3);
  const e3 = await errOf(page3);
  if (e3) problems.push(`warm: err: ${e3}`);
  if (!/t = 10\b/.test(s3)) problems.push(`warm: did not stop at t = 10: ${s3}`);
  if (!/resumed from cached t = 5\b/.test(s3)) {
    problems.push(`warm: expected a resume from t = 5: ${s3}`);
  }
  const warmFile = await page3.$eval('#download', (a) => a.download);
  if (warmFile === fileName || !/^[0-9a-f]{64}\.h5$/.test(warmFile)) {
    problems.push(`warm: odd file name: ${warmFile}`);
  }
  await page3.close();

  // ---- pass 5: model in the URL, and a live model switch -------------------
  // Allen–Cahn arrives via the fragment (one species -> one panel); switching
  // to Brusselator recompiles the session and rebuilds the panels (two).
  // Nothing computes: both settle on the idle miss status.
  const page4 = await browser.newPage();
  watch(page4, 'model:');
  await interceptCache(page4, null, '');
  await openAndSelect(page4, null, '#model=allencahn');
  await page4.waitForFunction(
    () => /press Compute solution|failed/.test(
      document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  const modelRestored = await page4.$eval('#model', (el) => el.value);
  if (modelRestored !== 'allencahn') {
    problems.push(`model: URL restore failed, model = ${modelRestored}`);
  }
  const acPanels = await page4.$$eval('.sphere-box canvas', (els) => els.length);
  if (acPanels !== 1) problems.push(`model: allencahn should have 1 panel, got ${acPanels}`);
  await page4.select('#model', 'brusselator');
  // The old idle status is still on screen while the recompile runs, so the
  // wait must demand the new panel count as well.
  await page4.waitForFunction(
    () => (document.querySelectorAll('.sphere-box canvas').length === 2 &&
      /press Compute solution/.test(document.getElementById('status')?.textContent ?? '')) ||
      /failed/.test(document.getElementById('status')?.textContent ?? '') ||
      (document.getElementById('err')?.textContent?.length ?? 0) > 4,
    { timeout: 600_000 },
  );
  const brPanels = await page4.$$eval('.sphere-box canvas', (els) => els.length);
  if (brPanels !== 2) problems.push(`model: brusselator should have 2 panels, got ${brPanels}`);
  const e4 = await errOf(page4);
  if (e4) problems.push(`model: err: ${e4}`);
  console.log(`pass 5: allencahn ${acPanels} panel, brusselator ${brPanels} panels`);
  await page4.close();
} catch (e) {
  problems.push(`fatal: ${e.message}`);
} finally {
  await browser.close();
  server.close();
}

if (problems.length) {
  console.log('PROBLEMS:');
  for (const p of new Set(problems)) console.log('  ' + p);
  process.exitCode = 1;
} else {
  console.log('CHECK-APP: PASS');
}
