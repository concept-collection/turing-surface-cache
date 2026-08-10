/**
 * The cache spec: the JSON object that fully identifies one solution, and the
 * mapping from it to an object name in the cloud cache.
 *
 * The name is the SHA-256 of the spec's canonical JSON serialization (keys
 * sorted recursively, numbers as ECMAScript shortest-round-trip strings, which
 * the language specifies exactly). The serialization includes the app name and
 * a format version, so a change to what a spec means changes every hash. The
 * object path also carries the app name, version, and model in the clear —
 * `turing-surface-cache/v1/schnakenberg/<sha256>.h5` — so that future cleanup
 * (lifecycle rules, prefix deletes, per-model sweeps) never has to open a file
 * to know what it belongs to.
 */
import type { Params } from '../mgpu/registry.ts';

export const APP_NAME = 'turing-surface-cache';
/** Bump together with the `v1` path segment below. */
export const FORMAT_VERSION = 1;

export interface CacheSpec {
  app: typeof APP_NAME;
  formatVersion: typeof FORMAT_VERSION;
  model: string;
  /** The model's own parameters, dt included. */
  params: Params;
  geometry: string;
  geometryParams: Params;
  lmax: number;
  niter: number;
  /** Wavelength of the seeded random field. */
  lam3: number;
  seed: number;
  /** Physical end time; steps = tEnd / dt, which must be an integer. */
  tEnd: number;
}

/** Timesteps from t = 0 to the spec's end time. Throws if tEnd is not an
 *  exact multiple of dt — the discrete lists are chosen so it always is. */
export function stepsFor(spec: CacheSpec): number {
  const dt = spec.params.dt;
  if (!(dt > 0)) throw new Error(`bad dt ${dt}`);
  const steps = Math.round(spec.tEnd / dt);
  if (Math.abs(steps * dt - spec.tEnd) > 1e-9 * spec.tEnd) {
    throw new Error(`tEnd ${spec.tEnd} is not a multiple of dt ${dt}`);
  }
  return steps;
}

/** JSON with every object's keys sorted, at every level. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const keys = Object.keys(value as Record<string, unknown>).sort();
  const body = keys
    .map((k) => `${JSON.stringify(k)}:${canonicalJson((value as Record<string, unknown>)[k])}`)
    .join(',');
  return `{${body}}`;
}

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Object name under the bucket's upload prefix (tmpbucket prepends
 * `tmpbucket/`, so the public URL is
 * https://tempory.net/tmpbucket/<this>).
 */
export async function cacheFileName(spec: CacheSpec): Promise<string> {
  const hash = await sha256Hex(canonicalJson(spec));
  return `${APP_NAME}/v${FORMAT_VERSION}/${spec.model}/${hash}.h5`;
}
