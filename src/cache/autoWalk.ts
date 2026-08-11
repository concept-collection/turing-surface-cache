/**
 * The auto-fill walk: which solutions to compute on an idle machine, and in
 * what order.
 *
 * The whole space is about 8,000 runs — roughly three GPU-weeks — so it is
 * exhaustible in principle, and the question is only what to do first.
 * Demand for it is nothing like uniform: a visitor starts at the defaults and
 * changes one dropdown at a time, so the chance that a combination is ever
 * requested falls off steeply with the number of knobs that differ from the
 * defaults. The walk therefore proceeds by that distance — every one-knob
 * deviation before any two-knob one — which fills the region people actually
 * ask for within a day rather than a month, and still covers everything in
 * the limit.
 *
 * Within a distance the order is random, and that is the whole coordination
 * mechanism between machines: several idle browsers walking the same tiers in
 * different orders, each skipping what it finds already cached, rarely
 * duplicate each other's work and need no coordinator, no queue and no
 * knowledge of one another.
 *
 * The seed and dt are pinned rather than surveyed (see AUTO_SEED / AUTO_DT).
 */
import type { Params } from '../mgpu/registry.ts';
import {
  MODEL_CHOICES,
  DEFAULT_MODEL_KEY,
  GEOMETRY_CHOICES,
  AUTO_DT,
  AUTO_SEED,
  T_END_CHOICE,
  LMAX,
  NITER,
  LAM3,
  type DiscreteChoice,
} from './options.ts';
import { DEFAULT_GEOMETRY_KEY } from '../geom/registry.ts';
import { APP_NAME, FORMAT_VERSION, type CacheSpec } from './spec.ts';

export interface AutoTarget {
  model: string;
  params: Params;
  geometry: string;
  geometryParams: Params;
  /** How many knobs differ from the app's defaults. */
  distance: number;
}

/**
 * Every combination of a choice list, each with the number of entries that
 * differ from their default. A key present in `pinned` takes that value in
 * every combination and never counts toward the distance.
 */
function combos(
  choices: DiscreteChoice[],
  pinned: Params = {},
): { values: Params; distance: number }[] {
  let out = [{ values: { ...pinned }, distance: 0 }];
  for (const c of choices) {
    if (c.key in pinned) continue;
    const next: typeof out = [];
    for (const acc of out) {
      for (const v of c.values) {
        next.push({
          values: { ...acc.values, [c.key]: v },
          distance: acc.distance + (v === c.value ? 0 : 1),
        });
      }
    }
    out = next;
  }
  return out;
}

/** The surfaces to survey, each with its distance from the default shape:
 *  one for being a different geometry, one more per non-default parameter. */
function geometryOptions(): { geometry: string; params: Params; distance: number }[] {
  const out: { geometry: string; params: Params; distance: number }[] = [];
  for (const [key, choices] of Object.entries(GEOMETRY_CHOICES)) {
    for (const c of combos(choices)) {
      // The ellipsoid with all axes 1 *is* the unit sphere, which the sphere
      // geometry already covers. Computing it would fill a second hash with
      // the same problem, so it is left out — 228 runs saved.
      if (key === 'ellipsoid' && c.values.ax === 1 && c.values.ay === 1 && c.values.az === 1) {
        continue;
      }
      out.push({
        geometry: key,
        params: c.values,
        distance: (key === DEFAULT_GEOMETRY_KEY ? 0 : 1) + c.distance,
      });
    }
  }
  return out;
}

/** Every solution the walk will ever compute, unordered. */
export function enumerateTargets(): AutoTarget[] {
  const geometries = geometryOptions();
  const out: AutoTarget[] = [];
  for (const [modelKey, choices] of Object.entries(MODEL_CHOICES)) {
    const modelDistance = modelKey === DEFAULT_MODEL_KEY ? 0 : 1;
    for (const p of combos(choices, { dt: AUTO_DT })) {
      for (const g of geometries) {
        out.push({
          model: modelKey,
          params: p.values,
          geometry: g.geometry,
          geometryParams: g.params,
          distance: modelDistance + p.distance + g.distance,
        });
      }
    }
  }
  return out;
}

/**
 * The solution a target names. The seed is the pinned one, and the end time is
 * the longest listed: a run reaching it passes through every shorter one and
 * contributes those on the way, so one run fills the whole chain. The page
 * sets its dropdowns from this rather than deciding the same thing twice.
 */
export function specForTarget(target: AutoTarget): CacheSpec {
  return {
    app: APP_NAME,
    formatVersion: FORMAT_VERSION,
    model: target.model,
    params: { ...target.params },
    geometry: target.geometry,
    geometryParams: { ...target.geometryParams },
    lmax: LMAX,
    niter: NITER,
    lam3: LAM3,
    seed: AUTO_SEED,
    tEnd: Math.max(...T_END_CHOICE.values),
  };
}

/**
 * The walk order: by distance, randomly within each distance. Shuffling the
 * whole list and then sorting by distance gives exactly that, since Array's
 * sort is stable — the shuffle survives as the within-distance order.
 */
export function autoOrder(rand: () => number = Math.random): AutoTarget[] {
  const all = enumerateTargets();
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [all[i], all[j]] = [all[j], all[i]];
  }
  all.sort((a, b) => a.distance - b.distance);
  return all;
}
