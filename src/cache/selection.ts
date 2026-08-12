/**
 * The selection — one value chosen from every discrete list — and its URL
 * form.
 *
 * The main page keeps its whole state in the URL fragment, every value
 * written explicitly, so a link keeps meaning the same spec even if a default
 * changes later. The sweep page carries the same fragment plus one extra
 * entry (`sweep=<param>`, which model parameter the knob runs over), and the
 * command line's `sweep <url>` accepts that page's URL as its argument. Three
 * readers of one serialization is the reason it lives here rather than in any
 * of them.
 *
 * Values are only accepted if they are exactly entries of the discrete lists
 * (src/cache/options.ts); anything else keeps the default. That is what makes
 * a fragment safe to hand to the cache: nothing typed or mistyped can name a
 * spec that the dropdowns could not.
 */
import type { Params } from '../mgpu/registry.ts';
import { DEFAULT_GEOMETRY_KEY } from '../geom/registry.ts';
import {
  DEFAULT_MODEL_KEY,
  GEOMETRY_CHOICES,
  LAM3,
  LMAX,
  MODEL_CHOICES,
  NITER,
  SEED_CHOICE,
  T_END_CHOICE,
  defaultChoiceParams,
  fmtChoice,
  type DiscreteChoice,
} from './options.ts';
import { APP_NAME, FORMAT_VERSION, type CacheSpec } from './spec.ts';

export interface Selection {
  model: string;
  params: Params;
  geometry: string;
  geometryParams: Params;
  seed: number;
  tEnd: number;
}

export function defaultSelection(): Selection {
  return {
    model: DEFAULT_MODEL_KEY,
    params: defaultChoiceParams(MODEL_CHOICES[DEFAULT_MODEL_KEY]),
    geometry: DEFAULT_GEOMETRY_KEY,
    geometryParams: defaultChoiceParams(GEOMETRY_CHOICES[DEFAULT_GEOMETRY_KEY]),
    seed: SEED_CHOICE.value,
    tEnd: T_END_CHOICE.value,
  };
}

/** The fragment form: `model=…&a=…&…&geometry=…&…&seed=…&tend=…`. The keys
 *  are the choices' own, except tEnd, which the URL spells `tend`. */
export function selectionToParams(sel: Selection): URLSearchParams {
  const p = new URLSearchParams();
  p.set('model', sel.model);
  for (const c of MODEL_CHOICES[sel.model]) p.set(c.key, fmtChoice(sel.params[c.key]));
  p.set('geometry', sel.geometry);
  for (const c of GEOMETRY_CHOICES[sel.geometry]) {
    p.set(c.key, fmtChoice(sel.geometryParams[c.key]));
  }
  p.set('seed', String(sel.seed));
  p.set('tend', fmtChoice(sel.tEnd));
  return p;
}

/**
 * A fragment string from those parameters. URLSearchParams percent-encodes
 * commas, which turns a sweep's value list into `0.7%2C0.9%2C1.1` — readable
 * to a parser and to nobody else. A fragment is allowed to carry commas
 * literally (RFC 3986 counts them among the sub-delims), and the parser
 * reads an unencoded comma back as the same character, so they are put back.
 */
export const fragmentFor = (p: URLSearchParams): string =>
  p.toString().replace(/%2C/g, ',');

/** Read a selection back from a fragment, defaults standing in for anything
 *  absent or not exactly a listed value. */
export function readSelection(p: URLSearchParams): Selection {
  const sel = defaultSelection();
  const pick = (choice: DiscreteChoice, current: number, name = choice.key): number => {
    const raw = p.get(name);
    if (raw === null) return current;
    const v = Number(raw);
    return choice.values.includes(v) ? v : current;
  };
  const m = p.get('model');
  if (m && MODEL_CHOICES[m]) {
    sel.model = m;
    sel.params = defaultChoiceParams(MODEL_CHOICES[m]);
  }
  const g = p.get('geometry');
  if (g && GEOMETRY_CHOICES[g]) {
    sel.geometry = g;
    sel.geometryParams = defaultChoiceParams(GEOMETRY_CHOICES[g]);
  }
  for (const c of MODEL_CHOICES[sel.model]) sel.params[c.key] = pick(c, sel.params[c.key]);
  for (const c of GEOMETRY_CHOICES[sel.geometry]) {
    sel.geometryParams[c.key] = pick(c, sel.geometryParams[c.key]);
  }
  sel.seed = pick(SEED_CHOICE, sel.seed);
  sel.tEnd = pick(T_END_CHOICE, sel.tEnd, 'tend');
  return sel;
}

/** The one solution a selection names. */
export function specForSelection(sel: Selection): CacheSpec {
  return {
    app: APP_NAME,
    formatVersion: FORMAT_VERSION,
    model: sel.model,
    params: { ...sel.params },
    geometry: sel.geometry,
    geometryParams: { ...sel.geometryParams },
    lmax: LMAX,
    niter: NITER,
    lam3: LAM3,
    seed: sel.seed,
    tEnd: sel.tEnd,
  };
}

// ---------------------------------------------------------------- sweeps
/**
 * A sweep: the same selection, with one model parameter designated as the
 * swept one and a list of values for it. The list defaults to the
 * parameter's own choices but may be an explicit list the user typed, which
 * is the one place the app steps outside its dropdown lists. That is safe
 * for the cache, since a typed value is parsed to a number once and
 * serialized in canonical shortest form ever after (src/cache/spec.ts), so
 * that it names one spec as reliably as a listed value does. It merely names
 * one the main page's dropdowns cannot reach. The selection's own value for
 * the swept parameter is the knob's current position, so a shared sweep link
 * opens at the same place.
 */
export interface SweepSelection {
  sel: Selection;
  /** Which of the model's parameters the knob runs over. */
  key: string;
  /** The values it runs over, in knob order. */
  values: number[];
}

/** The swept parameter's underlying choice (its label and default list). */
export function sweepChoice(sweep: { sel: Selection; key: string }): DiscreteChoice {
  const choice = MODEL_CHOICES[sweep.sel.model].find((c) => c.key === sweep.key);
  if (!choice) {
    throw new Error(`${sweep.sel.model} has no parameter '${sweep.key}'`);
  }
  return choice;
}

/**
 * An explicit value list, as typed: numbers separated by commas or spaces.
 * Anything that is not a finite number is dropped and duplicates collapse,
 * but the order is kept as given, an explicit list being taken at its word.
 */
export function parseValueList(text: string): number[] {
  return [
    ...new Set(
      text
        .split(/[,\s]+/)
        .filter((s) => s.length)
        .map(Number)
        .filter((v) => Number.isFinite(v)),
    ),
  ];
}

/** The sweep page's fragment: the selection plus which parameter sweeps and
 *  the values it runs over, every value written explicitly. */
export function sweepToParams(sweep: SweepSelection): URLSearchParams {
  const p = selectionToParams(sweep.sel);
  // The swept parameter's own entry is the knob position, which for a custom
  // list may be a value selectionToParams could not have written.
  p.set(sweep.key, fmtChoice(sweep.sel.params[sweep.key]));
  p.set('sweep', sweep.key);
  p.set('values', sweep.values.map(fmtChoice).join(','));
  return p;
}

/**
 * Read a sweep from a fragment. Null when the fragment names no swept
 * parameter (or one the model does not have): the page falls back to its
 * default, the command line says the URL is not a sweep link. A missing or
 * empty `values` entry means the parameter's own list.
 */
export function readSweep(p: URLSearchParams): SweepSelection | null {
  const sel = readSelection(p);
  const key = p.get('sweep');
  if (!key || !MODEL_CHOICES[sel.model].some((c) => c.key === key)) return null;
  const sweep: SweepSelection = { sel, key, values: [] };
  const listed = p.get('values');
  const parsed = listed === null ? [] : parseValueList(listed);
  sweep.values = parsed.length ? parsed : [...sweepChoice(sweep).values];
  // The knob position: readSelection validated the swept entry against the
  // dropdown list, which a custom value is deliberately not on, so it is
  // read again against the sweep's own list.
  const raw = p.get(key);
  const v = raw === null ? NaN : Number(raw);
  sel.params[key] = sweep.values.includes(v)
    ? v
    : sweep.values.includes(sel.params[key])
      ? sel.params[key]
      : sweep.values[0];
  return sweep;
}

/** The sweep's solutions, one per value, in knob order. */
export function specsForSweep(
  sweep: SweepSelection,
): { value: number; spec: CacheSpec }[] {
  return sweep.values.map((value) => ({
    value,
    spec: specForSelection({
      ...sweep.sel,
      params: { ...sweep.sel.params, [sweep.key]: value },
    }),
  }));
}
