/**
 * Pack the command-line bundle as an npm tarball and put it where the page is
 * deployed, so that
 *
 *   npx https://concept-collection.github.io/turing-surface-cache/fill.tgz
 *
 * installs and runs it anywhere node is. Nothing is published to the registry:
 * npm installs a tarball from a URL as happily as from a package name, and
 * this way the command line is always the same commit as the page it fills the
 * cache for.
 *
 * The package.json written here is the whole published manifest. It has no
 * numbl dependency — the compiler is bundled into fill.js exactly as it is
 * into the page — and its only real dependency is h5wasm, whose node build
 * reads its wasm off disk and so cannot be bundled. Dawn is optional, so the
 * install still succeeds on a machine that has no prebuilt binary for it; the
 * CLI says what to do about that if it comes to it.
 *
 * Usage: node scripts/pack-cli.mjs   (after `vite build --config vite.cli.config.ts`)
 */
import { execFileSync } from 'node:child_process';
import { mkdtemp, copyFile, writeFile, readFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const pkg = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));

const manifest = {
  name: 'turing-surface-fill',
  version: pkg.version,
  description: 'Fill the turing-surface-cache shared cache from the command line',
  license: pkg.license,
  type: 'module',
  // The bundle's own floor, not the repo's: node 18 is the oldest with fetch,
  // and the launcher checks for it. Kept in step with cli-launcher.cjs.
  engines: { node: '>=18' },
  // The bin is the launcher, not the bundle: an old node must reach a sentence
  // rather than a syntax error (scripts/cli-launcher.cjs).
  bin: { 'turing-surface-fill': 'launch.cjs' },
  files: ['launch.cjs', 'fill.js'],
  dependencies: { h5wasm: pkg.dependencies.h5wasm },
  optionalDependencies: { webgpu: pkg.optionalDependencies.webgpu },
};

const stage = await mkdtemp(join(tmpdir(), 'turing-fill-'));
try {
  await copyFile(join(root, 'dist-cli/fill.js'), join(stage, 'fill.js'));
  await copyFile(join(root, 'scripts/cli-launcher.cjs'), join(stage, 'launch.cjs'));
  await writeFile(join(stage, 'package.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  const out = execFileSync('npm', ['pack', '--silent', '--pack-destination', stage], {
    cwd: stage,
    encoding: 'utf8',
  }).trim();
  await mkdir(join(root, 'dist'), { recursive: true });
  await copyFile(join(stage, out), join(root, 'dist/fill.tgz'));
  console.log(`dist/fill.tgz  (${manifest.name} ${manifest.version})`);
} finally {
  await rm(stage, { recursive: true, force: true });
}
