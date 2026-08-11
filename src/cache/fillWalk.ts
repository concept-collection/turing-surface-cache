/**
 * Working through the parameter space on an idle machine: for every target of
 * the walk (src/cache/autoWalk.ts), skip whatever the cloud already has and
 * compute and contribute the rest, until stopped.
 *
 * The loop is short but its rules matter, and they are the same in the page
 * and on the command line. A skip costs one HEAD, and only the longest end
 * time need be asked about, since a run reaching it emits every shorter one on
 * the way. One bad combination — a compile that fails, a solution that blows
 * up — is reported and stepped over rather than ending the walk. And every
 * target goes through the ordinary local run, warm start, background uploads,
 * divergence guard and all.
 */
import type { AutoTarget } from './autoWalk.ts';
import { isCached, lookupFor } from './client.ts';
import { runSpec, type RunEvents, type RunOutcome } from './runSpec.ts';
import type { SolverSession } from './solver.ts';
import type { CacheSpec } from './spec.ts';

export interface FillCounts {
  computed: number;
  skipped: number;
  failed: number;
}

export interface FillEvents extends RunEvents {
  /** A target is being considered; its cache status is not known yet. */
  onTarget?(target: AutoTarget, spec: CacheSpec): void;
  /** Already in the cloud — nothing to do. */
  onCached?(target: AutoTarget, spec: CacheSpec): void;
  /** Not cached: the run is about to start. */
  onComputing?(target: AutoTarget, spec: CacheSpec): void;
  onOutcome?(target: AutoTarget, spec: CacheSpec, outcome: RunOutcome): void;
  onFailure?(target: AutoTarget, spec: CacheSpec, error: unknown): void;
  /** True ends the walk at the next target boundary. */
  walkStopped?(): boolean;
}

export interface FillOptions {
  targets: AutoTarget[];
  solver: SolverSession;
  adapter: string;
  apiKey(): string;
  /**
   * Take the selection to this target and hand back the spec to compute. The
   * page uses this to drive its own dropdowns and URL, so that what is on
   * screen always says what is being computed.
   */
  beforeTarget(target: AutoTarget): CacheSpec | Promise<CacheSpec>;
  events?: FillEvents;
}

export async function fillWalk(opts: FillOptions): Promise<FillCounts> {
  const ev = opts.events ?? {};
  const counts: FillCounts = { computed: 0, skipped: 0, failed: 0 };
  for (const target of opts.targets) {
    if (ev.walkStopped?.()) break;
    const spec = await opts.beforeTarget(target);
    ev.onTarget?.(target, spec);
    try {
      if (await isCached(await lookupFor(spec))) {
        counts.skipped++;
        ev.onCached?.(target, spec);
        continue;
      }
      if (ev.walkStopped?.()) break;
      ev.onComputing?.(target, spec);
      await opts.solver.apply(spec);
      const outcome = await runSpec({
        solver: opts.solver,
        spec,
        adapter: opts.adapter,
        apiKey: opts.apiKey,
        events: ev,
      });
      if (outcome.kind === 'done') counts.computed++;
      else if (outcome.kind === 'diverged') counts.failed++;
      ev.onOutcome?.(target, spec, outcome);
      if (outcome.kind === 'abandoned') break;
    } catch (e) {
      // One bad combination must not end the walk: report it and move on.
      counts.failed++;
      ev.onFailure?.(target, spec, e);
    }
  }
  return counts;
}
