/**
 * Desktop WebGPU, via the optional `webgpu` package (prebuilt Google Dawn).
 *
 * Dawn is installed under the globals the transform code expects
 * (navigator.gpu, GPUBufferUsage, …), so everything under src/ runs here
 * unchanged — including requestShtDevice(), which makes the same device
 * request the browser makes. This file is the whole of what the command line
 * has that the page does not.
 *
 * Adapted from turing-surface's scripts/nodeWebGpu.ts.
 */

export const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** Installs Dawn and returns a human-readable runtime description. */
export async function installWebGpu(): Promise<string> {
  // The import specifier is indirect so that a build (and a type-check)
  // does not require the optional package.
  const specifier = 'webgpu';
  let mod: {
    create: (flags: string[]) => GPU;
    globals: Record<string, unknown>;
  };
  try {
    mod = await import(/* @vite-ignore */ specifier);
  } catch (e) {
    // Distinguish "not installed" from "installed but the prebuilt Dawn binary
    // will not load" — the second is what a machine missing a system library
    // looks like, and reporting it as the first sends people in circles.
    const detail = errMsg(e);
    if (/Cannot find (package|module) '?webgpu'?/.test(detail)) {
      throw new Error(
        'desktop WebGPU needs the optional `webgpu` package (prebuilt Google Dawn),\n' +
          'which npm may have skipped silently. `npm ls webgpu` says whether it is\n' +
          'there; `npm install webgpu` installs it.',
      );
    }
    const glibc = /GLIBC_([0-9.]+)/.exec(detail);
    throw new Error(
      `the \`webgpu\` package is installed but did not load:\n  ${detail}\n` +
        (glibc
          ? `Dawn's prebuilt binary wants glibc ${glibc[1]} or newer and this host is older\n` +
            '(`ldd --version` says how old). No flag bridges that — use a container with a\n' +
            'newer base image, or a newer host.'
          : 'That is usually the prebuilt Dawn binary missing a system library.'),
    );
  }
  Object.assign(globalThis, mod.globals);
  // DAWN_FLAGS is ';'-separated because individual Dawn options take
  // comma-separated lists, e.g. 'enable-dawn-features=allow_unsafe_apis,...'
  const dawnFlags = process.env.DAWN_FLAGS?.split(';').filter(Boolean) ?? [];
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu: mod.create(dawnFlags) },
    configurable: true,
    writable: true,
  });
  const { version } = await import(/* @vite-ignore */ `${specifier}/package.json`, {
    with: { type: 'json' },
  }).then(
    (m) => m.default as { version: string },
    () => ({ version: '?' }),
  );
  return `node-webgpu ${version} (Google Dawn)`;
}

/** The hint to print when Dawn loads but finds no adapter. */
export const NO_ADAPTER_HINT =
  '  Dawn reaches the GPU through Vulkan on Linux and Windows, Metal on macOS,\n' +
  "  so a headless box may have no adapter at all. DAWN_FLAGS='backend=vulkan'\n" +
  '  makes it explain itself.';

/**
 * Adapters that are not really GPUs. A run on one of these is perhaps a
 * thousand times slower than on hardware, which is fast enough to look like it
 * is working and slow enough to be worthless — so it is worth saying out loud.
 */
export const isSoftwareAdapter = (name: string): boolean =>
  /swiftshader|llvmpipe|lavapipe|software|microsoft basic|warp/i.test(name);
