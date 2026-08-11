/**
 * Cache files are HDF5, in the layout of turing-surface's reference files
 * (docs/ellipsoid-reference-spec.md there) extended with the cache's own
 * identity at the root: the canonical spec JSON that was hashed into the
 * object name, the app name, and the format version. A cache file is thereby
 * also a valid reference file — turing-surface's "Compare to reference…" mode
 * opens one as-is — and readable from Python with h5py.
 *
 *   /            attrs: app, format_version, spec_json, model, species,
 *                       created_utc, adapter
 *   /backend     attrs: adapter, runtime, precision
 *   /spec        attrs: geometry, lmax, seed, steps, niter, lam3, t_end
 *   /spec/params           attrs: a, b, D1, D2, dt
 *   /spec/geometry_params  attrs: the geometry's params
 *   /grid        attrs: lmax, mmax, nlat, nphi, nlm
 *   /geometry    Gx, Gy, Gz        float32[2*nlm]
 *   /initial     one dataset per species (U, V)   float32[2*nlm]
 *   /final       one dataset per species (U, V)   float32[2*nlm]
 *
 * h5wasm's browser build carries the whole HDF5 library as embedded wasm
 * (~4 MB), so it is imported dynamically and only here: the page pays for it
 * on the first cache hit or upload, never on startup.
 */
import type { ShtConfig } from '../sht/layout.ts';
import { nlmCalc } from '../sht/layout.ts';
import { APP_NAME, FORMAT_VERSION, canonicalJson, stepsFor, type CacheSpec } from './spec.ts';

export interface CacheFileData {
  spec: CacheSpec;
  grid: ShtConfig;
  /** Spectral state names, in order — Schnakenberg's ['U', 'V']. */
  species: string[];
  /** The band-limited surface's own coefficients, [re, im] per (l, m). */
  geometry: { X: Float32Array; Y: Float32Array; Z: Float32Array };
  /** Spectral state at t = 0 (immediately after seeding). */
  initial: Record<string, Float32Array>;
  /** The same, at the spec's end time. */
  final: Record<string, Float32Array>;
  /** Provenance: which GPU computed it. */
  adapter: string;
  /** And what was driving it — a browser, or the command line's Dawn. */
  runtime: string;
}

interface H5Module {
  ready: Promise<unknown>;
  File: new (path: string, mode: string) => H5WFile;
  FS?: unknown;
}

interface H5Attr {
  value: unknown;
}

interface H5Obj {
  attrs: Record<string, H5Attr>;
  get(name: string): unknown;
  create_group(name: string): unknown;
  create_attribute(name: string, data: unknown, shape?: unknown, dtype?: unknown): void;
  create_dataset(args: { name: string; data: unknown; shape?: number[]; dtype?: string }): void;
}

interface H5WFile extends H5Obj {
  close(): void;
}

interface EmFS {
  writeFile(path: string, data: Uint8Array): void;
  readFile(path: string): Uint8Array;
  unlink(path: string): void;
}

let scratchCounter = 0;

/**
 * Where the scratch file that h5wasm reads or writes lives.
 *
 * In the browser it lives in h5wasm's own in-memory filesystem, where any
 * absolute path will do and nothing touches a disk. The node build is
 * compiled with NODERAWFS, which is to say its filesystem *is* the real one:
 * the same path would name a file in the root directory, which fails with a
 * wall of HDF5 diagnostics rather than an error anyone could act on. A real
 * temporary directory is therefore used there, and the command line replaces
 * this default with the platform's own (src/cli/fill.ts).
 */
let scratchDir = __NODE_BUILD__ ? '/tmp/' : '/';

/** Set the directory for those scratch files; node only. */
export function setScratchDir(dir: string): void {
  scratchDir = dir.endsWith('/') ? dir : `${dir}/`;
}

/** Two fills on one machine share that real directory; a page's filesystem is
 *  its own, so there is nothing to distinguish there. */
const scratchTag = __NODE_BUILD__ ? `${process.pid}-` : '';

const scratchPath = (what: string): string =>
  `${scratchDir}turing-surface-cache-${scratchTag}${what}-${scratchCounter++}.h5`;

async function withH5<T>(fn: (h5: H5Module, fs: EmFS) => T | Promise<T>): Promise<T> {
  // Two builds of the same library: the browser one carries the wasm inside
  // the bundle, the node one reads it off disk. __NODE_BUILD__ is a build-time
  // constant (see vite.config.ts and vite.cli.config.ts), so whichever branch
  // this build is not takes no part in it.
  const h5 = (await (__NODE_BUILD__
    ? import('h5wasm/node')
    : import('h5wasm'))) as unknown as H5Module;
  const { FS } = (await h5.ready) as { FS: EmFS };
  return fn(h5, FS);
}

const groupOf = (node: H5Obj, name: string): H5Obj => {
  const g = node.get(name) as H5Obj | null;
  if (!g || typeof g.get !== 'function') throw new Error(`no '${name}/' group`);
  return g;
};

const coeffsOf = (group: H5Obj, groupName: string, name: string, nlm: number): Float32Array => {
  const v = (group.get(name) as { value?: unknown } | null)?.value;
  if (!(v instanceof Float32Array)) {
    throw new Error(`'${groupName}/${name}' is not a float32 dataset`);
  }
  if (v.length !== 2 * nlm) {
    throw new Error(`'${groupName}/${name}' has ${v.length} values, expected 2*nlm = ${2 * nlm}`);
  }
  return v;
};

/** Serialize one solution to HDF5 bytes. */
export function encodeCacheFile(data: CacheFileData): Promise<Uint8Array> {
  return withH5((h5, FS) => {
    const path = scratchPath('encode');
    const file = new h5.File(path, 'w');
    try {
      const { spec, grid } = data;
      file.create_attribute('app', APP_NAME);
      file.create_attribute('format_version', FORMAT_VERSION);
      file.create_attribute('spec_json', canonicalJson(spec));
      file.create_attribute('model', spec.model);
      file.create_attribute('species', data.species);
      file.create_attribute('created_utc', new Date().toISOString());
      file.create_attribute('adapter', data.adapter);

      file.create_group('backend');
      const backend = groupOf(file, 'backend');
      backend.create_attribute('adapter', data.adapter);
      backend.create_attribute('runtime', data.runtime);
      backend.create_attribute('precision', 'fp32');

      file.create_group('spec');
      const specGroup = groupOf(file, 'spec');
      specGroup.create_attribute('geometry', spec.geometry);
      specGroup.create_attribute('lmax', spec.lmax);
      specGroup.create_attribute('seed', spec.seed);
      specGroup.create_attribute('steps', stepsFor(spec));
      specGroup.create_attribute('niter', spec.niter);
      specGroup.create_attribute('lam3', spec.lam3);
      specGroup.create_attribute('t_end', spec.tEnd);
      specGroup.create_group('params');
      const params = groupOf(specGroup, 'params');
      for (const [k, v] of Object.entries(spec.params)) params.create_attribute(k, v);
      specGroup.create_group('geometry_params');
      const gparams = groupOf(specGroup, 'geometry_params');
      for (const [k, v] of Object.entries(spec.geometryParams)) gparams.create_attribute(k, v);

      file.create_group('grid');
      const gridGroup = groupOf(file, 'grid');
      const nlm = nlmCalc(grid.lmax, grid.mmax);
      gridGroup.create_attribute('lmax', grid.lmax);
      gridGroup.create_attribute('mmax', grid.mmax);
      gridGroup.create_attribute('nlat', grid.nlat);
      gridGroup.create_attribute('nphi', grid.nphi);
      gridGroup.create_attribute('nlm', nlm);

      file.create_group('geometry');
      const geom = groupOf(file, 'geometry');
      geom.create_dataset({ name: 'Gx', data: data.geometry.X });
      geom.create_dataset({ name: 'Gy', data: data.geometry.Y });
      geom.create_dataset({ name: 'Gz', data: data.geometry.Z });

      for (const [groupName, states] of [
        ['initial', data.initial],
        ['final', data.final],
      ] as const) {
        file.create_group(groupName);
        const g = groupOf(file, groupName);
        for (const name of data.species) {
          const coeffs = states[name];
          if (!coeffs) throw new Error(`missing ${groupName} state '${name}'`);
          if (coeffs.length !== 2 * nlm) {
            throw new Error(`${groupName}/${name}: ${coeffs.length} values, expected ${2 * nlm}`);
          }
          g.create_dataset({ name, data: coeffs });
        }
      }
    } finally {
      file.close();
    }
    try {
      return FS.readFile(path);
    } finally {
      // Under NODERAWFS this is a real file in a real temporary directory, so
      // it is removed on the way out however this ends.
      FS.unlink(path);
    }
  });
}

export interface DecodedCacheFile {
  /** Parsed from the file's own spec_json — the identity it was stored under. */
  spec: CacheSpec;
  species: string[];
  initial: Record<string, Float32Array>;
  final: Record<string, Float32Array>;
  /** Provenance, when recorded. */
  adapter: string;
  created: string;
}

/**
 * Read cache-file bytes back. `expectSpecJson` is the canonical JSON this
 * client asked the cache for; a mismatch with the file's own means the object
 * store handed back something other than what the key promised (corruption,
 * or a stale format), and is an error rather than a silent wrong answer.
 */
export function decodeCacheFile(
  bytes: Uint8Array,
  expectSpecJson: string,
  expectSpecies: string[],
): Promise<DecodedCacheFile> {
  return withH5((h5, FS) => {
    const path = scratchPath('decode');
    FS.writeFile(path, bytes);
    const file = new h5.File(path, 'r');
    try {
      const attr = (name: string): unknown => file.attrs[name]?.value;
      const app = String(attr('app') ?? '');
      if (app !== APP_NAME) throw new Error(`not a ${APP_NAME} file (app='${app}')`);
      const version = Number(attr('format_version'));
      if (version !== FORMAT_VERSION) throw new Error(`format version ${version}, expected ${FORMAT_VERSION}`);
      const specJson = String(attr('spec_json') ?? '');
      if (specJson !== expectSpecJson) {
        throw new Error('file spec does not match the requested spec');
      }
      const spec = JSON.parse(specJson) as CacheSpec;
      const nlm = nlmCalc(spec.lmax, spec.lmax);
      const initialGroup = groupOf(file, 'initial');
      const finalGroup = groupOf(file, 'final');
      const initial: Record<string, Float32Array> = {};
      const final: Record<string, Float32Array> = {};
      for (const name of expectSpecies) {
        initial[name] = coeffsOf(initialGroup, 'initial', name, nlm);
        final[name] = coeffsOf(finalGroup, 'final', name, nlm);
      }
      return {
        spec,
        species: expectSpecies,
        initial,
        final,
        adapter: String(attr('adapter') ?? ''),
        created: String(attr('created_utc') ?? ''),
      };
    } finally {
      file.close();
      FS.unlink(path);
    }
  });
}
