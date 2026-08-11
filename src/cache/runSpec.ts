/**
 * Computing one cached solution: the run behind both the page's Compute
 * solution button and the command line's fill walk.
 *
 * What the run does besides stepping to the end time is the part worth having
 * in one place. It starts from the longest cached shorter run of the same
 * spec instead of from t = 0, since the state is Markovian in the spectral
 * coefficients; it captures the state at every listed end time it passes, and
 * encodes and uploads each one while the solver keeps stepping; and it
 * refuses to publish a state that has gone non-finite. All of that is worth
 * exactly one implementation.
 *
 * Everything a caller wants to say about a run in progress — status text,
 * rendering, when to stop — arrives through RunEvents, so nothing here knows
 * whether it is driving a page or a terminal.
 */
import { T_END_CHOICE } from './options.ts';
import { stepsFor, type CacheSpec } from './spec.ts';
import { lookupFor, fetchCached, uploadCacheFile } from './client.ts';
import { encodeCacheFile, decodeCacheFile, type DecodedCacheFile } from './h5file.ts';
import type { SolverSession } from './solver.ts';

/**
 * Steps between syncs: many small submissions queued back to back, one wait.
 * The readbacks and renders that pace a live view happen per chunk, not per
 * submission — that is what lets a run advance at close to the solver's own
 * rate.
 */
const CHUNK_STEPS = 32;

/** What a finished run amounts to, and all a caller needs to describe it. */
export interface RunSummary {
  tEnd: number;
  seconds: number;
  /** The end time this run resumed from, if it resumed from one. */
  warmFrom: number | null;
}

export type RunPhase =
  | { kind: 'warm-search' }
  | { kind: 'seeding' }
  | { kind: 'encoding'; run: RunSummary }
  | { kind: 'uploading'; run: RunSummary; started: number; uploaded: number };

export interface RunProgress {
  /** Simulation time reached, and where the run ends. */
  t: number;
  tEnd: number;
  steps: number;
  totalSteps: number;
  /** Fraction of *this* run's work done: a warm start begins at 0 here. */
  fraction: number;
  /** Steps per second over the run so far. */
  rate: number;
  /** The end time this run resumed from, if it resumed from one. */
  warmFrom: number | null;
  uploadsStarted: number;
  uploadsDone: number;
}

export interface RunEvents {
  onPhase?(phase: RunPhase): void;
  /** The warm start or the seeding is done and the stepping is about to
   *  begin: whatever is on display now belongs to the previous run. */
  onStepping?(): void;
  /** Once per chunk. Callers throttle their own display. */
  onProgress?(p: RunProgress): void;
  /** After each chunk: the caller's chance to draw, and to yield. */
  onTick?(): Promise<void> | void;
  /** The final state is computed and finite, before the file is written. */
  onFinal?(tEnd: number): Promise<void> | void;
  onUploaded?(tEnd: number): void;
  /** The finished file, named as it is in the cache. */
  onFile?(bytes: Uint8Array, fileName: string): void;
  /**
   * Abandon the run at the next safe point, reporting nothing: something else
   * has taken the session over. Distinct from stopRequested, which is a
   * deliberate stop whose partial results still count.
   */
  cancelled?(): boolean;
  /** Stop cleanly at the next chunk boundary, keeping what was uploaded. */
  stopRequested?(): boolean;
}

export type RunOutcome =
  | (RunSummary & {
      kind: 'done';
      fileName: string;
      bytes: Uint8Array;
      /** End times uploaded, in the order they completed. */
      uploaded: number[];
      uploadsStarted: number;
      uploadErrors: string[];
    })
  | { kind: 'stopped'; t: number; uploaded: number[] }
  | { kind: 'diverged'; t: number }
  | { kind: 'abandoned' };

export interface RunOptions {
  solver: SolverSession;
  spec: CacheSpec;
  /** Which GPU computed it; recorded in every file it writes. */
  adapter: string;
  /** What is driving that GPU: 'browser-webgpu', or the command line's Dawn. */
  runtime: string;
  /** Read at every upload point, so a key entered mid-run still contributes. */
  apiKey(): string;
  events?: RunEvents;
}

/**
 * Nothing that is not a number gets uploaded. A combination whose timestep is
 * too large for its reaction blows up rather than failing, and an unattended
 * walk would happily publish the wreckage under a hash someone later trusts.
 */
const stateIsFinite = (state: Record<string, Float32Array>): boolean =>
  Object.values(state).every((a) => a.every(Number.isFinite));

/** Run the solver to the spec's end time, contributing everything it passes. */
export async function runSpec(opts: RunOptions): Promise<RunOutcome> {
  const { solver, spec, adapter, runtime, apiKey } = opts;
  const ev = opts.events ?? {};
  const cancelled = (): boolean => ev.cancelled?.() ?? false;
  const session = solver.live;
  const steps = stepsFor(spec);
  const dt = spec.params.dt;

  // Warm start: the state is Markovian in (U, V), so a cached run of the same
  // spec at a smaller listed end time is an exact prefix of this one. Take the
  // longest one there is and continue from its final state rather than
  // recomputing it.
  let warm: { tEnd: number; decoded: DecodedCacheFile } | null = null;
  const earlier = T_END_CHOICE.values.filter((T) => T < spec.tEnd).sort((a, b) => b - a);
  if (earlier.length) ev.onPhase?.({ kind: 'warm-search' });
  for (const T of earlier) {
    const lookup = await lookupFor({ ...spec, tEnd: T });
    let bytes: Uint8Array | null = null;
    try {
      bytes = await fetchCached(lookup);
    } catch {
      break; // cache unreachable: no point probing further down the ladder
    }
    if (cancelled()) return { kind: 'abandoned' };
    if (!bytes) continue;
    try {
      warm = { tEnd: T, decoded: await decodeCacheFile(bytes, lookup.specJson, solver.model.state) };
      break;
    } catch {
      continue; // an unreadable candidate is skipped, not fatal
    }
  }
  if (cancelled()) return { kind: 'abandoned' };

  let initial: Record<string, Float32Array>;
  if (warm) {
    session.loadState(warm.decoded.final);
    // loadState resets the clock; put it at the cached run's end so the loop
    // below computes only the remainder.
    session.steps = Math.round(warm.tEnd / dt);
    session.t = warm.tEnd;
    // The t = 0 state travels with every file of the chain, so files written
    // from this continuation carry the same initial state as the one resumed.
    initial = warm.decoded.initial;
  } else {
    ev.onPhase?.({ kind: 'seeding' });
    await session.seed(spec.seed);
    if (cancelled()) return { kind: 'abandoned' };
    initial = await session.readState();
    if (cancelled()) return { kind: 'abandoned' };
  }
  const startSteps = session.steps;

  // Snapshot points: every listed end time strictly between the starting point
  // and this run's end. The run passes through each exactly (all are whole
  // multiples of every dt choice).
  const snapshotAt = new Map<number, number>(); // step index -> tEnd value
  for (const T of T_END_CHOICE.values) {
    if (T < spec.tEnd && T > (warm?.tEnd ?? 0)) snapshotAt.set(Math.round(T / dt), T);
  }
  const snapshots: { tEnd: number; state: Record<string, Float32Array> }[] = [];

  // Everything a cache file needs exists before the run starts, so a snapshot
  // is encoded and uploaded the moment it is captured, overlapping the network
  // with the GPU still stepping, rather than queued for the end.
  const geometryCoeffs = {
    X: session.geometry.X,
    Y: session.geometry.Y,
    Z: session.geometry.Z,
  };
  const encode = (t: number, state: Record<string, Float32Array>) =>
    encodeCacheFile({
      spec: { ...spec, tEnd: t },
      grid: session.cfg,
      species: solver.model.state,
      geometry: geometryCoeffs,
      initial,
      final: state,
      adapter,
      runtime,
    });
  const uploadedTimes: number[] = [];
  const uploadErrors: string[] = [];
  let uploadsStarted = 0;
  const pendingUploads: Promise<void>[] = [];
  /** Encode + upload without the stepping loop waiting. A captured snapshot is
   *  a complete solution of its own spec, so this stays valid even if the run
   *  is stopped afterwards. */
  const uploadInBackground = (
    t: number,
    state: Record<string, Float32Array>,
    key: string,
    preEncoded?: Uint8Array,
  ): void => {
    uploadsStarted++;
    pendingUploads.push(
      (async () => {
        const bytes = preEncoded ?? (await encode(t, state));
        const lookup = await lookupFor({ ...spec, tEnd: t });
        await uploadCacheFile(key, lookup.fileName, bytes);
        uploadedTimes.push(t);
        ev.onUploaded?.(t);
      })().catch((e) => {
        uploadErrors.push(`t = ${t}: ${e instanceof Error ? e.message : e}`);
      }),
    );
  };

  ev.onStepping?.();
  const t0 = performance.now();
  while (session.steps < steps) {
    if (cancelled()) return { kind: 'abandoned' };
    if (ev.stopRequested?.()) {
      return { kind: 'stopped', t: session.steps * dt, uploaded: [...uploadedTimes] };
    }
    // One chunk: up to CHUNK_STEPS steps submitted back to back (each
    // submission stays under the dispatch budget), then a single sync and at
    // most one render. Reading back and drawing after every submission is what
    // made the run advance at a fraction of the solver's rate — a readback
    // costs several times the 3-4 steps it fenced. The chunk stops exactly at
    // snapshot points so those states are still captured exactly.
    let target = Math.min(steps, session.steps + CHUNK_STEPS);
    for (const s of snapshotAt.keys()) {
      if (s > session.steps && s < target) target = s;
    }
    while (session.steps < target) {
      session.step(Math.min(solver.stepsPerSubmit, target - session.steps));
    }
    // The sync bounds how far the CPU runs ahead of the GPU, and (being a
    // promise) yields to the event loop, which is what keeps a Stop button
    // clickable.
    await session.sync();
    if (cancelled()) return { kind: 'abandoned' };
    const hit = snapshotAt.get(session.steps);
    if (hit !== undefined) {
      const state = await session.readState();
      if (cancelled()) return { kind: 'abandoned' };
      if (!stateIsFinite(state)) return { kind: 'diverged', t: session.steps * dt };
      // With a key on hand the snapshot goes straight to the cache; without one
      // it is kept, in case a key is entered before the run ends.
      const key = apiKey();
      if (key) uploadInBackground(hit, state, key);
      else snapshots.push({ tEnd: hit, state });
    }
    ev.onProgress?.({
      t: session.steps * dt,
      tEnd: spec.tEnd,
      steps: session.steps,
      totalSteps: steps,
      fraction: (session.steps - startSteps) / (steps - startSteps),
      rate: (session.steps - startSteps) / ((performance.now() - t0) / 1000),
      warmFrom: warm?.tEnd ?? null,
      uploadsStarted,
      uploadsDone: uploadedTimes.length,
    });
    await ev.onTick?.();
  }
  if (cancelled()) return { kind: 'abandoned' };

  const final = await session.readState();
  if (cancelled()) return { kind: 'abandoned' };
  if (!stateIsFinite(final)) return { kind: 'diverged', t: spec.tEnd };
  await ev.onFinal?.(spec.tEnd);
  const summary: RunSummary = {
    tEnd: spec.tEnd,
    seconds: (performance.now() - t0) / 1000,
    warmFrom: warm?.tEnd ?? null,
  };

  ev.onPhase?.({ kind: 'encoding', run: summary });
  const finalBytes = await encode(spec.tEnd, final);
  if (cancelled()) return { kind: 'abandoned' };
  const fileName = (await lookupFor(spec)).fileName.split('/').pop()!;
  ev.onFile?.(finalBytes, fileName);

  // The final solution, plus any snapshots captured before a key was entered.
  const key = apiKey();
  if (key) {
    uploadInBackground(spec.tEnd, final, key, finalBytes);
    for (const snap of snapshots) uploadInBackground(snap.tEnd, snap.state, key);
  }
  if (uploadsStarted > 0) {
    ev.onPhase?.({
      kind: 'uploading',
      run: summary,
      started: uploadsStarted,
      uploaded: uploadedTimes.length,
    });
    await Promise.all(pendingUploads);
    if (cancelled()) return { kind: 'abandoned' };
  }
  return {
    kind: 'done',
    ...summary,
    fileName,
    bytes: finalBytes,
    uploaded: [...uploadedTimes],
    uploadsStarted,
    uploadErrors,
  };
}
