/**
 * The cloud side of the cache.
 *
 * Reads are anonymous GETs against the bucket's public base URL: the object
 * name is a deterministic function of the spec (src/cache/spec.ts), so a
 * lookup is one fetch and a 404 is a miss — no index, no API.
 *
 * Writes go through the tmpbucket Worker
 * (https://github.com/scratchrealm/tmpbucket): present an API key and a file name,
 * receive a presigned R2 PUT URL, and upload directly. Only holders of the
 * key can write; everyone can read.
 */
import { cacheFileName, canonicalJson, type CacheSpec } from './spec.ts';

const PUBLIC_BASE = 'https://tempory.net/tmpbucket/';
const WORKER_BASE = 'https://tmpbucket.figurl.workers.dev';
const CONTENT_TYPE = 'application/x-hdf5';

export interface CacheLookup {
  fileName: string;
  url: string;
  specJson: string;
}

export async function lookupFor(spec: CacheSpec): Promise<CacheLookup> {
  const fileName = await cacheFileName(spec);
  return { fileName, url: PUBLIC_BASE + fileName, specJson: canonicalJson(spec) };
}

/** Fetch a cached solution; null on a miss. Throws on network failure or an
 *  unexpected status, which are reported rather than treated as misses. */
export async function fetchCached(lookup: CacheLookup): Promise<Uint8Array | null> {
  const res = await fetch(lookup.url, { cache: 'no-store' });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`cache read: HTTP ${res.status} for ${lookup.url}`);
  return new Uint8Array(await res.arrayBuffer());
}

/** Upload one cache file. Resolves to its public URL. */
export async function uploadCacheFile(
  apiKey: string,
  fileName: string,
  bytes: Uint8Array,
): Promise<string> {
  const res = await fetch(`${WORKER_BASE}/api/upload-url`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ fileName, contentType: CONTENT_TYPE }),
  });
  if (res.status === 401 || res.status === 403) {
    throw new Error('upload not authorized — check the API key');
  }
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const err = (await res.json()) as { message?: string };
      if (err.message) message = err.message;
    } catch {
      // keep the status-only message
    }
    throw new Error(`upload-url request failed: ${message}`);
  }
  const grant = (await res.json()) as {
    uploadUrl: string;
    uploadHeaders?: Record<string, string>;
    downloadUrl: string;
  };
  const put = await fetch(grant.uploadUrl, {
    method: 'PUT',
    headers: grant.uploadHeaders ?? { 'Content-Type': CONTENT_TYPE },
    body: bytes as unknown as BodyInit,
  });
  if (!put.ok) throw new Error(`upload PUT failed: HTTP ${put.status}`);
  return grant.downloadUrl;
}
