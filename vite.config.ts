import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * Which build this is. It ends up in the URL the page hands out for the
 * command line (src/main.ts), because npx keys its install directory on the
 * whole spec string: the same URL keeps running whatever it first installed,
 * however many times the file behind it has changed. A URL carrying the
 * commit is a new spec, and so a fresh install.
 */
function buildId(): string {
  if (process.env.GITHUB_SHA) return process.env.GITHUB_SHA.slice(0, 7);
  try {
    return execFileSync('git', ['rev-parse', '--short', 'HEAD'], {
      cwd: import.meta.dirname,
      encoding: 'utf8',
    }).trim();
  } catch {
    return 'dev';
  }
}

// numbl is a local `file:` dependency, so node_modules/numbl is a symlink to
// the sibling checkout. Its package `exports` map only publishes the runtime
// entry points, not the compiler internals we need (parser + JIT lowering), so
// we reach them through a path alias. (package.json's `imports` field cannot
// express this — Node rejects node_modules targets — and plain Node could not
// resolve numbl's internal `.js`->`.ts` imports anyway, which is why the GPU
// tests run in the browser harness rather than under `node`.)
// Realpath'd through the symlink: dev serves modules under their real ids, so
// aliasing the node_modules path would give the same file two identities (one
// per spelling) and run its side effects twice — the interpreter's builtin
// registry throws on the second.
const numblSrc = realpathSync(resolve(import.meta.dirname, 'node_modules/numbl/src'));

export default defineConfig({
  base: './',
  // The page is the browser build; the command line's bundle sets this true
  // (vite.cli.config.ts). See src/cache/h5file.ts.
  define: { __NODE_BUILD__: 'false', __BUILD_ID__: JSON.stringify(buildId()) },
  resolve: {
    alias: { 'numbl-src': numblSrc },
  },
  server: {
    // the alias resolves outside the project root (through the symlink)
    fs: { allow: [import.meta.dirname, numblSrc] },
  },
  build: {
    target: 'es2022',
  },
});
