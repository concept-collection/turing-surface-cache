/**
 * The command line's bundle: the same source as the page, built for node.
 *
 * Everything under src/ is bundled in, numbl's compiler included, exactly as
 * in the page's build — the published package therefore has no dependency on
 * a numbl checkout. What stays external is what cannot be bundled: h5wasm,
 * whose node build reads its wasm off disk, and Dawn, which is a native
 * addon (and optional, so a machine without it still installs).
 */
import { defineConfig, mergeConfig } from 'vite';
import base from './vite.config.ts';

export default mergeConfig(
  base,
  defineConfig({
    define: { __NODE_BUILD__: 'true' },
    build: {
      ssr: 'src/cli/fill.ts',
      outDir: 'dist-cli',
      target: 'node22',
      emptyOutDir: true,
      minify: false,
      rollupOptions: {
        external: ['h5wasm', 'h5wasm/node', 'webgpu'],
        // The page build's two-HTML input would otherwise merge in here;
        // a string wins over an object in mergeConfig, restoring the one
        // node entry.
        input: 'src/cli/fill.ts',
        output: { entryFileNames: 'fill.js', banner: '#!/usr/bin/env node' },
      },
    },
  }),
);
