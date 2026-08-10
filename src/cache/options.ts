/**
 * The discrete parameter space.
 *
 * Every knob the app offers is a choice from a short enumerated list, so a run
 * is fully identified by a small tuple of exact values — which is what makes
 * the cloud cache work: the same choices always produce the same cache key
 * (src/cache/spec.ts), with no floating-point formatting ambiguity, because
 * the values *are* these list entries, never something typed or interpolated.
 *
 * The numerical-scheme knobs (lmax, niter, seed wavelength) are fixed in this
 * version rather than offered, but they are recorded in the spec and the cache
 * key all the same, so making them choices later invalidates nothing.
 */
import type { Params } from '../mgpu/registry.ts';

export interface DiscreteChoice {
  key: string;
  label: string;
  /** The allowed values, in display order. */
  values: number[];
  /** Default — must be one of `values`. */
  value: number;
}

/** The model offered first, and what Reset returns to. */
export const DEFAULT_MODEL_KEY = 'schnakenberg';

/** Model parameter choices, by model key. Every dt divides every end time. */
export const MODEL_CHOICES: Record<string, DiscreteChoice[]> = {
  schnakenberg: [
    { key: 'a', label: 'a', values: [0.05, 0.1, 0.15, 0.2], value: 0.1 },
    { key: 'b', label: 'b', values: [0.7, 0.9, 1.1, 1.3], value: 0.9 },
    { key: 'D1', label: 'D₁', values: [1.6e-4, 4e-4, 1e-3], value: 4e-4 },
    { key: 'D2', label: 'D₂', values: [3.2e-3, 8e-3, 2e-2], value: 8e-3 },
    { key: 'dt', label: 'dt', values: [0.02, 0.05, 0.1], value: 0.05 },
  ],
  brusselator: [
    { key: 'A', label: 'A', values: [2, 3, 4], value: 3 },
    { key: 'B', label: 'B', values: [7, 9, 11], value: 9 },
    { key: 'D1', label: 'D₁', values: [1.7e-3, 3.33e-3, 6.7e-3], value: 3.33e-3 },
    { key: 'D2', label: 'D₂', values: [8.3e-3, 1.67e-2, 3.3e-2], value: 1.67e-2 },
    { key: 'dt', label: 'dt', values: [0.01, 0.02, 0.05], value: 0.02 },
  ],
  allencahn: [
    { key: 'eps2', label: 'ε²', values: [5e-4, 1e-3, 2e-3], value: 1e-3 },
    { key: 'dt', label: 'dt', values: [0.01, 0.02, 0.05], value: 0.02 },
  ],
};

/** Geometry parameter choices, by geometry key. The sphere has none. */
export const GEOMETRY_CHOICES: Record<string, DiscreteChoice[]> = {
  sphere: [],
  ellipsoid: [
    { key: 'ax', label: 'a', values: [0.6, 1, 1.5], value: 1.5 },
    { key: 'ay', label: 'b', values: [0.6, 1, 1.5], value: 1 },
    { key: 'az', label: 'c', values: [0.6, 1, 1.5], value: 0.6 },
  ],
  peanut: [
    { key: 'waist', label: 'waist', values: [0.4, 0.6, 0.8], value: 0.6 },
    { key: 'stretch', label: 'stretch', values: [0, 0.6, 1.2], value: 0.6 },
  ],
};

export const SEED_CHOICE: DiscreteChoice = {
  key: 'seed',
  label: 'seed',
  values: [1, 2, 3, 4, 5],
  value: 1,
};

/**
 * End times, in simulation-time units. Every entry is an exact multiple of
 * every dt choice, so a run to any of them lands on a whole number of steps —
 * and a run to a later one passes exactly through the earlier ones. That is
 * where the along-the-way cache snapshots come from, and, in the other
 * direction, why a computation can warm-start from the longest cached
 * shorter run of the same spec.
 */
export const T_END_CHOICE: DiscreteChoice = {
  key: 'tEnd',
  label: 'end time',
  values: [100, 200, 400, 800, 1600],
  value: 100,
};

/** Fixed numerical-scheme settings (recorded in every spec and cache key). */
export const LMAX = 63;
export const NITER = 8;
export const LAM3 = 0.5;

export const defaultChoiceParams = (choices: DiscreteChoice[]): Params =>
  Object.fromEntries(choices.map((c) => [c.key, c.value]));

/** Display formatting: exact and compact ("4e-4", "0.05"). */
export const fmtChoice = (v: number): string =>
  v !== 0 && Math.abs(v) < 0.01 ? v.toExponential() : String(v);
