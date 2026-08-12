/**
 * The parameter sweep page: one model parameter runs over its whole list of
 * values while every other choice stays fixed, and a knob steps the display
 * through the range.
 *
 * The cache is what makes the knob instant. Each value of the swept parameter
 * names one solution (the same specs the main page uses, so the two pages and
 * the walk all share one cache), and on any selection change the page fetches
 * all of them at once — a sweep is three to five files of ~90 KB. Each one is
 * decoded and synthesized to the render grid immediately, so moving the knob
 * afterwards touches no network and no solver: it recolors the mesh from
 * values already in memory. One color scale is computed over the whole sweep
 * and held fixed, so what changes under the knob is the pattern and not the
 * palette.
 *
 * Values nobody has computed show as gaps. Compute missing values runs them
 * here, one after another through the ordinary local run
 * (src/cache/runSpec.ts — warm start, background uploads with a key,
 * divergence guard), and the copyable command at the bottom hands the same
 * sweep to a machine with no browser on it (src/cli/fill.ts `sweep`). Both
 * read the sweep from this page's URL fragment, which carries the whole
 * selection plus which parameter is swept (src/cache/selection.ts).
 */
import { requestShtDevice, describeAdapter } from './sht/sht.ts';
import type { ModelSession } from './mgpu/session.ts';
import { mModels, mModelByKey, type MModel } from './mgpu/registry.ts';
import { formatFailure } from './mgpu/errors.ts';
import { mGeometries, mGeometryByKey } from './geom/registry.ts';
import {
  buildTopology,
  fillPositions,
  fillFieldValues,
  fillColors,
  type SphereMeshTopology,
} from './render/sphereMesh.ts';
import { SphereScene } from './render/SphereScene.ts';
import { Colorbar, floorRange } from './render/colorbar.ts';
import { colormaps } from './render/colormaps.ts';
import {
  MODEL_CHOICES,
  GEOMETRY_CHOICES,
  SEED_CHOICE,
  T_END_CHOICE,
  LMAX,
  NITER,
  defaultChoiceParams,
  fmtChoice,
  type DiscreteChoice,
} from './cache/options.ts';
import { APP_NAME, stepsFor, type CacheSpec } from './cache/spec.ts';
import { lookupFor, fetchCached } from './cache/client.ts';
import { decodeCacheFile } from './cache/h5file.ts';
import { SolverSession } from './cache/solver.ts';
import { type RunEvents, type RunSummary } from './cache/runSpec.ts';
import { fillWalk } from './cache/fillWalk.ts';
import type { AutoTarget } from './cache/autoWalk.ts';
import {
  defaultSelection,
  fragmentFor,
  parseValueList,
  readSelection,
  readSweep,
  selectionToParams,
  specForSelection,
  sweepChoice,
  sweepToParams,
  specsForSweep,
  type Selection,
  type SweepSelection,
} from './cache/selection.ts';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const elModel = $<HTMLSelectElement>('model');
const elSweepOver = $<HTMLSelectElement>('sweepover');
const elParams = $('params');
const elGeometry = $<HTMLSelectElement>('geometry');
const elGeomParams = $('geomparams');
const elSeed = $<HTMLSelectElement>('seed');
const elTend = $<HTMLSelectElement>('tend');
const elCompute = $<HTMLButtonElement>('compute');
const elStop = $<HTMLButtonElement>('stop');
const elReset = $<HTMLButtonElement>('reset');
const elValues = $<HTMLInputElement>('values');
const elValuesNote = $('valuesnote');
const elKnob = $<HTMLInputElement>('knob');
const elTicks = $('ticks');
const elKnobVal = $('knobval');
const elStatus = $('status');
const elPanels = $('panels');
const elResetView = $<HTMLButtonElement>('resetview');
const elStats = $('stats');
const elApiKey = $<HTMLInputElement>('apikey');
const elUploadNote = $('uploadnote');
const elCliBar = $('clibar');
const elCliCmd = $('clicmd');
const elCliCopy = $<HTMLButtonElement>('clicopy');
const elCliCopied = $('clicopied');
const elCliNote = $('clinote');
const elBackLink = $<HTMLAnchorElement>('backlink');
const elErr = $('err');

/** The main page's ?tend test hook, honored here too (src/main.ts). */
{
  const param = new URLSearchParams(location.search).get('tend');
  if (param) {
    const values = param
      .split(',')
      .map(Number)
      .filter((v) => Number.isFinite(v) && v > 0);
    if (values.length) {
      T_END_CHOICE.values = values;
      T_END_CHOICE.value = values[0];
    }
  }
}

/** Shared with the main page: one key entered once covers both. */
const API_KEY_STORAGE = `${APP_NAME}:apiKey`;
const COLORMAP = colormaps.viridis;
const OVERSAMPLE = 2;
const RENDER_EVERY_MS = 250;
const STATUS_EVERY_MS = 200;

// ---------------------------------------------------------------- state
let sel: Selection = defaultSelection();
/** Which of the model's parameters the knob runs over. */
let sweepKey = MODEL_CHOICES[sel.model][0].key;
/** The values it runs over: the parameter's own list until the values box
 *  says otherwise (src/cache/selection.ts). */
let sweepValues: number[] = [...MODEL_CHOICES[sel.model][0].values];

/** One value of the sweep: its solution, and where it stands. `fields` is
 *  the decoded final state synthesized onto the render mesh, one array per
 *  species — everything the knob needs, with the session out of the loop.
 *  'cached' came from the cloud; 'computed' was run here this session (and
 *  is in the cloud too only if a key was present for the uploads). */
interface SweepEntry {
  value: number;
  spec: CacheSpec;
  status:
    | 'loading'
    | 'cached'
    | 'computed'
    | 'missing'
    | 'failed'
    /** No run can reach the end time from this value (a dt that does not
     *  divide it), so there is nothing to fetch or compute. */
    | 'unusable'
    | 'refetch';
  fields: Float32Array[] | null;
}
let entries: SweepEntry[] = [];

let device: GPUDevice | null = null;
let solver: SolverSession | null = null;
let adapterName = '';
function sess(): ModelSession | null {
  return solver?.session ?? null;
}
const curModel = (): MModel => mModelByKey(sel.model)!;
const curChoice = (): DiscreteChoice => sweepChoice({ sel, key: sweepKey });
const curSweep = (): SweepSelection => ({ sel, key: sweepKey, values: sweepValues });

let generation = 0;
let busy = false;
let stopRequested = false;

// view
let topo: SphereMeshTopology | null = null;
let scenes: SphereScene[] = [];
let colorbars: Colorbar[] = [];
let colorbarEls: HTMLElement[] = [];
let colorBufs: Float32Array[] = [];
/** Scratch per-vertex values for the live view while a value is computing. */
let liveBufs: Float32Array[] = [];
/** Smoothed display ranges for that live view (main.ts does the same). */
let liveRanges: { lo: number; hi: number }[] = [];
/** The sweep-wide color range per species, fixed while the knob moves. */
let ranges: { lo: number; hi: number }[] = [];
let resizeObs: ResizeObserver | null = null;

const nextFrame = () => new Promise<number>(requestAnimationFrame);

// ---------------------------------------------------------------- URL state
{
  const hash = location.hash.replace(/^#/, '');
  if (hash) {
    const p = new URLSearchParams(hash);
    const sweep = readSweep(p);
    if (sweep) {
      sel = sweep.sel;
      sweepKey = sweep.key;
      sweepValues = sweep.values;
    } else {
      // A main-page link: same selection, sweeping the first parameter over
      // its own list.
      sel = readSelection(p);
      sweepKey = MODEL_CHOICES[sel.model][0].key;
      sweepValues = [...MODEL_CHOICES[sel.model][0].values];
    }
  }
}

function writeUrlState(): void {
  const p = fragmentFor(sweepToParams(curSweep()));
  history.replaceState(null, '', `${location.pathname}${location.search}#${p}`);
  // Back to the main page on the same selection (the knob's value travels as
  // the swept parameter's value; the search part keeps the ?tend test hook).
  elBackLink.href = `index.html${location.search}#${fragmentFor(selectionToParams(sel))}`;
  updateCliCommand();
}

// ---------------------------------------------------------------- controls
function makeSelect(
  choice: DiscreteChoice,
  get: () => number,
  set: (v: number) => void,
): HTMLLabelElement {
  const label = document.createElement('label');
  label.textContent = `${choice.label} `;
  const select = document.createElement('select');
  for (const v of choice.values) {
    const opt = document.createElement('option');
    opt.value = String(v);
    opt.textContent = fmtChoice(v);
    select.append(opt);
  }
  select.value = String(get());
  select.addEventListener('change', () => {
    set(Number(select.value));
    onSelectionChange();
  });
  label.append(select);
  return label;
}

/** The fixed parameters: every model parameter except the swept one, which
 *  lives on the knob instead. */
function buildParamControls(): void {
  elParams.replaceChildren();
  for (const choice of MODEL_CHOICES[sel.model]) {
    if (choice.key === sweepKey) continue;
    elParams.append(
      makeSelect(choice, () => sel.params[choice.key], (v) => (sel.params[choice.key] = v)),
    );
  }
}

function buildSweepOverControl(): void {
  elSweepOver.replaceChildren();
  for (const choice of MODEL_CHOICES[sel.model]) {
    const opt = document.createElement('option');
    opt.value = choice.key;
    opt.textContent = choice.label;
    elSweepOver.append(opt);
  }
  elSweepOver.value = sweepKey;
}

/**
 * The values box: what the knob runs over, written out. It starts as the
 * parameter's own list, which is what the main page's dropdown offers and
 * what the auto-fill walk surveys, but anything may be typed in its place.
 * This is the one control in the app that is not a choice from a list. A
 * value off the list still names one exact solution and one exact cache
 * entry, since the spec is hashed from the number rather than from the list
 * position, so a sweep over typed values is cached and shared like any
 * other. Of course, the walk only fills the listed combinations, so such a
 * sweep will not already be there.
 */
function showValues(): void {
  elValues.value = sweepValues.map(fmtChoice).join(', ');
  const listed = curChoice().values;
  const custom =
    sweepValues.length !== listed.length || sweepValues.some((v, i) => v !== listed[i]);
  elValuesNote.textContent = custom
    ? `${sweepValues.length} values (the offered list is ${listed.map(fmtChoice).join(', ')})`
    : 'the offered values';
}

/** Read the box back. An empty box means the parameter's own list; values
 *  that are not numbers are dropped by parseValueList and the box is
 *  rewritten with what was understood, so it never disagrees with the knob. */
function applyValues(): void {
  const parsed = parseValueList(elValues.value);
  sweepValues = parsed.length ? parsed : [...curChoice().values];
  if (!sweepValues.includes(sel.params[sweepKey])) sel.params[sweepKey] = sweepValues[0];
  showValues();
  onSelectionChange();
}

function buildGeomParamControls(): void {
  elGeomParams.replaceChildren();
  for (const choice of GEOMETRY_CHOICES[sel.geometry]) {
    elGeomParams.append(
      makeSelect(
        choice,
        () => sel.geometryParams[choice.key],
        (v) => (sel.geometryParams[choice.key] = v),
      ),
    );
  }
}

function buildControls(): void {
  for (const m of mModels) {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = m.label;
    elModel.append(opt);
  }
  elModel.value = sel.model;
  elModel.addEventListener('change', () => {
    sel.model = elModel.value;
    sel.params = defaultChoiceParams(MODEL_CHOICES[sel.model]);
    if (!MODEL_CHOICES[sel.model].some((c) => c.key === sweepKey)) {
      sweepKey = MODEL_CHOICES[sel.model][0].key;
    }
    // Another model's parameter means another quantity: a typed list for the
    // old one would rarely be meaningful for the new one, so the values go
    // back to what this model offers.
    sweepValues = [...curChoice().values];
    buildSweepOverControl();
    buildParamControls();
    showValues();
    onSelectionChange();
  });
  buildSweepOverControl();
  elSweepOver.addEventListener('change', () => {
    // The previously swept parameter keeps the value the knob was on and
    // returns to the fixed row; the newly swept one moves onto the knob,
    // over its own list.
    sweepKey = elSweepOver.value;
    sweepValues = [...curChoice().values];
    if (!sweepValues.includes(sel.params[sweepKey])) sel.params[sweepKey] = sweepValues[0];
    buildParamControls();
    showValues();
    onSelectionChange();
  });
  buildParamControls();
  showValues();
  // Applied on Enter or on leaving the box, not per keystroke: each change
  // refetches the whole sweep.
  elValues.addEventListener('change', () => applyValues());

  for (const g of mGeometries) {
    const opt = document.createElement('option');
    opt.value = g.key;
    opt.textContent = g.label.toLowerCase();
    elGeometry.append(opt);
  }
  elGeometry.value = sel.geometry;
  elGeometry.addEventListener('change', () => {
    sel.geometry = elGeometry.value;
    sel.geometryParams = defaultChoiceParams(GEOMETRY_CHOICES[sel.geometry]);
    buildGeomParamControls();
    onSelectionChange();
  });
  buildGeomParamControls();

  for (const v of SEED_CHOICE.values) {
    const opt = document.createElement('option');
    opt.value = String(v);
    opt.textContent = String(v);
    elSeed.append(opt);
  }
  elSeed.value = String(sel.seed);
  elSeed.addEventListener('change', () => {
    sel.seed = Number(elSeed.value);
    onSelectionChange();
  });

  for (const v of T_END_CHOICE.values) {
    const opt = document.createElement('option');
    opt.value = String(v);
    opt.textContent = String(v);
    elTend.append(opt);
  }
  elTend.value = String(sel.tEnd);
  elTend.addEventListener('change', () => {
    sel.tEnd = Number(elTend.value);
    onSelectionChange();
  });
}

function resetDefaults(): void {
  sel = defaultSelection();
  sweepKey = MODEL_CHOICES[sel.model][0].key;
  sweepValues = [...curChoice().values];
  elModel.value = sel.model;
  buildSweepOverControl();
  buildParamControls();
  showValues();
  elGeometry.value = sel.geometry;
  buildGeomParamControls();
  elSeed.value = String(sel.seed);
  elTend.value = String(sel.tEnd);
  onSelectionChange();
}

/** Selection changes reload the whole sweep; chained so two flows never talk
 *  to the session at once (same discipline as src/main.ts). */
let flowChain: Promise<void> = Promise.resolve();
function onSelectionChange(): void {
  writeUrlState();
  flowChain = flowChain.then(() => reloadSweep()).catch(() => undefined);
}

// ---------------------------------------------------------------- the knob
function knobIndex(): number {
  const i = sweepValues.indexOf(sel.params[sweepKey]);
  return i >= 0 ? i : 0;
}

function rebuildKnob(): void {
  elKnob.min = '0';
  elKnob.max = String(Math.max(0, sweepValues.length - 1));
  elKnob.step = '1';
  elKnob.disabled = busy || sweepValues.length < 2;
  elKnob.value = String(knobIndex());
  elTicks.replaceChildren(
    ...sweepValues.map((v, i) => {
      const b = document.createElement('button');
      b.className = 'tick';
      b.textContent = fmtChoice(v);
      b.addEventListener('click', () => {
        if (!busy) setKnob(i);
      });
      return b;
    }),
  );
  updateTicks();
}

function updateTicks(): void {
  const idx = knobIndex();
  const label = curChoice().label;
  elTicks.querySelectorAll<HTMLButtonElement>('.tick').forEach((b, i) => {
    const e = entries[i];
    b.classList.toggle('cached', e?.status === 'cached' || e?.status === 'computed');
    b.classList.toggle('current', i === idx);
    b.title =
      e?.status === 'cached'
        ? 'in the cloud cache'
        : e?.status === 'computed'
          ? 'computed here'
          : e?.status === 'missing'
            ? 'not computed yet'
            : e?.status === 'unusable'
              ? `no whole number of steps reaches t = ${fmtChoice(sel.tEnd)} at this dt`
              : e?.status === 'failed'
                ? 'unavailable'
                : '';
  });
  elKnobVal.textContent = sweepValues.length
    ? `${label} = ${fmtChoice(sweepValues[idx])}`
    : `no values to sweep ${label} over`;
}

/** Point the knob at value index `i` and show what is there. Pure display:
 *  no network, no solver — that is what the up-front loading bought. */
function setKnob(i: number): void {
  if (!sweepValues.length) return;
  sel.params[sweepKey] = sweepValues[i];
  elKnob.value = String(i);
  writeUrlState();
  showCurrent();
}

elKnob.addEventListener('input', () => {
  if (busy) return;
  setKnob(Number(elKnob.value));
});

// ---------------------------------------------------------------- view
function disposeView(): void {
  for (const s of scenes) s.dispose();
  scenes = [];
  colorbars = [];
  colorbarEls = [];
  topo = null;
  resizeObs?.disconnect();
  resizeObs = null;
  elPanels.replaceChildren();
}

function buildView(surface: Float32Array): void {
  const session = sess();
  if (!session) return;
  const view = session.viewSht;
  const { nphi } = view.cfg;
  const phi = new Float64Array(nphi);
  for (let j = 0; j < nphi; j++) phi[j] = (2 * Math.PI * j) / nphi;
  topo = buildTopology(view.cosTheta, phi);
  const posBuf = new Float32Array(topo.numVertices * 3);
  fillPositions(posBuf, surface, topo, 1);

  const sphereBg = getComputedStyle(document.documentElement)
    .getPropertyValue('--sphere-bg')
    .trim();
  const model = curModel();
  colorBufs = [];
  liveBufs = [];
  liveRanges = [];
  ranges = [];
  for (let k = 0; k < model.species.length; k++) {
    const panel = document.createElement('div');
    panel.className = 'panel';
    const box = document.createElement('div');
    box.className = 'sphere-box';
    const tag = document.createElement('div');
    tag.className = 'species-tag';
    tag.textContent = model.species[k];
    box.append(tag);
    const side = document.createElement('div');
    panel.append(box, side);
    elPanels.append(panel);

    const scene = new SphereScene(
      box,
      topo.numVertices,
      topo.indices,
      Float32Array.from(posBuf),
      sphereBg || undefined,
    );
    scene.fitCamera();
    scenes.push(scene);
    colorbars.push(new Colorbar(side));
    colorbarEls.push(side);
    colorBufs.push(new Float32Array(topo.numVertices * 3));
    liveBufs.push(new Float32Array(topo.numVertices));
    liveRanges.push({ lo: NaN, hi: NaN });
    ranges.push({ lo: NaN, hi: NaN });
  }
  for (let k = 1; k < scenes.length; k++) scenes[0].syncCamerasWith(scenes[k]);

  resizeObs = new ResizeObserver(() => {
    const boxes = elPanels.querySelectorAll<HTMLElement>('.sphere-box');
    boxes.forEach((box, i) => {
      scenes[i]?.resize(box.clientWidth, box.clientHeight);
    });
  });
  elPanels
    .querySelectorAll<HTMLElement>('.sphere-box')
    .forEach((box) => resizeObs!.observe(box));
}

/** Rebuild mesh and panels from the session's surface, keeping the camera. */
async function rebuildViewFromSession(): Promise<void> {
  const session = sess();
  if (!session) return;
  const surface = await session.renderPositions();
  const cam = scenes[0]?.cameraState();
  disposeView();
  buildView(surface);
  if (cam) for (const s of scenes) s.setCameraState(cam);
  grayDisplay();
}

/** The shape with no field on it (NaN renders neutral gray in fillColors). */
function grayDisplay(): void {
  if (!topo) return;
  for (let k = 0; k < scenes.length; k++) {
    liveBufs[k].fill(NaN);
    fillColors(colorBufs[k], liveBufs[k], 0, 1, COLORMAP);
    scenes[k].updateColors(colorBufs[k]);
    colorbarEls[k].style.visibility = 'hidden';
  }
}

/**
 * The sweep-wide color range, per species, over every loaded value. Fixed
 * while the knob moves, so colors mean the same thing at every position;
 * recomputed only when the set of loaded values changes.
 */
function recomputeRanges(): void {
  for (let k = 0; k < ranges.length; k++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const e of entries) {
      const f = e.fields?.[k];
      if (!f) continue;
      for (const v of f) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    ranges[k] = lo <= hi ? floorRange(lo, hi) : { lo: NaN, hi: NaN };
  }
}

/** Show the knob's current value from the in-memory fields. */
function showCurrent(): void {
  updateTicks();
  const entry = entries[knobIndex()];
  if (topo && entry?.fields) {
    for (let k = 0; k < scenes.length; k++) {
      fillColors(colorBufs[k], entry.fields[k], ranges[k].lo, ranges[k].hi, COLORMAP);
      scenes[k].updateColors(colorBufs[k]);
      colorbars[k].update(COLORMAP, ranges[k].lo, ranges[k].hi);
      colorbarEls[k].style.visibility = '';
    }
  } else {
    grayDisplay();
  }
  updateStats();
  if (!busy) updateSweepNote();
}

function updateStats(): void {
  const session = sess();
  if (!session) return;
  const { nlat, nphi } = session.cfg;
  const entry = entries[knobIndex()];
  const showing = entry?.fields
    ? ` · showing <b>${curChoice().label} = ${fmtChoice(entry.value)}</b>` +
      ` at t = <b>${fmtChoice(sel.tEnd)}</b>`
    : '';
  elStats.innerHTML =
    `<b>WebGPU fp32${adapterName ? ` — ${adapterName}` : ''}</b> · ` +
    `grid ${nlat}×${nphi} · lmax ${LMAX} · solve iters ${NITER}${showing}`;
}

// ---------------------------------------------------------------- statuses
function status(html: string): void {
  elStatus.innerHTML = html;
}

const isMissing = (e: SweepEntry): boolean =>
  e.status === 'missing' || e.status === 'failed';

/** The idle status line: how much of the sweep is there, and what to do
 *  about the rest. */
function updateSweepNote(): void {
  if (entries.some((e) => e.status === 'loading')) return;
  const n = entries.length;
  const loaded = entries.filter((e) => e.fields).length;
  const cloud = entries.filter((e) => e.status === 'cached').length;
  const unusable = entries.filter((e) => e.status === 'unusable').length;
  const label = curChoice().label;
  // "in the cloud cache" only when that is where they all came from: a
  // keyless local compute loads a value without contributing it.
  const what = cloud === loaded ? 'in the cloud cache' : 'loaded';
  const aside = unusable
    ? ` ${unusable} of them cannot be solved to t = ${fmtChoice(sel.tEnd)} at all.`
    : '';
  if (loaded === n) {
    const computedHere = n - cloud ? ` (${n - cloud} computed here)` : '';
    status(`all <b>${n} values</b> of ${label} are ${what}${computedHere} — drag the knob.`);
    return;
  }
  const entry = entries[knobIndex()];
  const here =
    entry && !entry.fields && entry.status !== 'unusable'
      ? ` <b>${label} = ${fmtChoice(entry.value)}</b> is one of them.`
      : '';
  const todo = n - loaded - unusable;
  status(
    `<b>${loaded} of ${n}</b> values ${what}; ${todo} not computed yet.${here}${aside} ` +
      (todo
        ? `<b>Compute missing values</b> runs them in your browser, one after another.`
        : ''),
  );
}

function setBusy(next: boolean): void {
  busy = next;
  elStop.hidden = !next;
  elReset.disabled = next;
  elKnob.disabled = next || sweepValues.length < 2;
  elValues.disabled = next;
  document
    .querySelectorAll<HTMLSelectElement>('main .controls select')
    .forEach((s) => (s.disabled = next));
  updateComputeButton();
}

function updateComputeButton(): void {
  elCompute.disabled = busy || !device || !entries.some(isMissing);
}

// ---------------------------------------------------------------- loading
/** Synthesize per-vertex render values from the state the session holds. */
async function fieldsFromSession(): Promise<Float32Array[]> {
  const session = sess();
  if (!session || !topo) throw new Error('no view to synthesize into');
  const out: Float32Array[] = [];
  for (let k = 0; k < curModel().species.length; k++) {
    const field = await session.readSpecies(k);
    const vals = new Float32Array(topo.numVertices);
    fillFieldValues(vals, field, topo);
    out.push(vals);
  }
  return out;
}

/** The same, for a decoded cache file: load its final state first. */
async function fieldsFromState(state: Record<string, Float32Array>): Promise<Float32Array[]> {
  const session = sess();
  if (!session) throw new Error('no solver session');
  session.loadState(state);
  return fieldsFromSession();
}

/**
 * Bring the page in line with the selection: build the sweep's entries, apply
 * the base spec to the solver (recompiling or re-evaluating the surface only
 * when the model or geometry changed), then fetch every value's cache file at
 * once. Fetches run in parallel; the GPU synthesis of whatever arrives is
 * serialized through one chain, since the session is one machine.
 */
async function reloadSweep(): Promise<void> {
  if (!device || !solver || busy) return;
  generation++;
  const gen = generation;
  elErr.textContent = '';
  // A typed dt that does not divide the end time names a run that cannot land
  // on it, which stepsFor refuses. Caught here rather than in the middle of a
  // walk, where it would arrive as a failure per value.
  entries = specsForSweep(curSweep()).map(({ value, spec }) => {
    let usable = true;
    try {
      stepsFor(spec);
    } catch {
      usable = false;
    }
    return {
      value,
      spec,
      status: usable ? ('loading' as const) : ('unusable' as const),
      fields: null,
    };
  });
  const unusable = entries.filter((e) => e.status === 'unusable');
  if (unusable.length) {
    elErr.textContent =
      `${unusable.map((e) => `${curChoice().label} = ${fmtChoice(e.value)}`).join(', ')}: ` +
      `the end time ${fmtChoice(sel.tEnd)} is not a whole number of steps at this dt`;
  }
  rebuildKnob();
  updateComputeButton();
  status('checking the cloud cache…');
  try {
    await solver.apply(specForSelection(sel));
  } catch (e) {
    if (gen === generation) {
      elErr.textContent = formatFailure(e, curModel().source);
      status('failed.');
    }
    return;
  }
  if (gen !== generation) return;
  grayDisplay();
  updateStats();

  let synth: Promise<void> = Promise.resolve();
  await Promise.all(
    entries.map(async (entry) => {
      if (entry.status === 'unusable') return;
      let bytes: Uint8Array | null = null;
      let unreachable = false;
      const lookup = await lookupFor(entry.spec);
      try {
        bytes = await fetchCached(lookup);
      } catch {
        unreachable = true;
      }
      if (gen !== generation) return;
      if (!bytes) {
        entry.status = unreachable ? 'failed' : 'missing';
        if (unreachable) elErr.textContent = 'cloud cache unreachable';
        entrySettled(gen, entry);
        return;
      }
      const data = bytes;
      synth = synth.then(async () => {
        if (gen !== generation) return;
        try {
          const decoded = await decodeCacheFile(data, lookup.specJson, curModel().state);
          if (gen !== generation) return;
          entry.fields = await fieldsFromState(decoded.final);
          entry.status = 'cached';
        } catch (e) {
          entry.status = 'failed';
          elErr.textContent = `${curChoice().label} = ${fmtChoice(entry.value)}: ${
            e instanceof Error ? e.message : e
          }`;
        }
        entrySettled(gen, entry);
      });
      await synth;
    }),
  );
  if (gen !== generation) return;
  updateComputeButton();
  updateSweepNote();
}

/** A value's fate is known (loaded, missing, or broken): fold it into the
 *  common color range and the display as it lands, not at the end. */
function entrySettled(gen: number, entry: SweepEntry): void {
  if (gen !== generation) return;
  if (entry.fields) recomputeRanges();
  showCurrent();
}

// ---------------------------------------------------------------- computing
/** The live view while a value computes (main.ts's draw, with the smoothed
 *  self-scaling range — the sweep-wide scale takes over once it is done). */
async function drawLive(gen: number): Promise<void> {
  const session = sess();
  if (!session || !topo) return;
  for (let k = 0; k < scenes.length; k++) {
    let field: Float32Array;
    try {
      field = await session.readSpecies(k);
    } catch (e) {
      if (gen !== generation) return;
      throw e;
    }
    if (gen !== generation || !topo) return;
    fillFieldValues(liveBufs[k], field, topo);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of liveBufs[k]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const r = liveRanges[k];
    if (!Number.isFinite(r.lo)) {
      r.lo = lo;
      r.hi = hi;
    } else {
      const a = 0.15;
      r.lo += a * (lo - r.lo);
      r.hi += a * (hi - r.hi);
    }
    const shown = floorRange(r.lo, r.hi);
    fillColors(colorBufs[k], liveBufs[k], shown.lo, shown.hi, COLORMAP);
    scenes[k].updateColors(colorBufs[k]);
    colorbars[k].update(COLORMAP, shown.lo, shown.hi);
    colorbarEls[k].style.visibility = '';
  }
}

/**
 * Compute the sweep's uncached values here, in value order, watching each
 * pattern form. Every run is the ordinary local computation (warm start from
 * a shorter cached run, snapshots uploaded in the background when a key is
 * present, divergence guard). The knob follows along so the URL and the
 * readout always say which value is being computed.
 */
async function computeMissing(): Promise<void> {
  if (!device || !solver || busy) return;
  const missing = entries.filter(isMissing);
  if (!missing.length) return;
  setBusy(true);
  stopRequested = false;
  elErr.textContent = '';
  generation++;
  const gen = generation;
  const label = curChoice().label;
  let computing: SweepEntry | null = null;
  let uploads = 0;
  let lastStatus = 0;
  let lastDraw = 0;

  const runLine = (run: RunSummary): string =>
    `<b>${label} = ${fmtChoice(computing?.value ?? NaN)}</b> — computed in ` +
    `${run.seconds.toFixed(1)} s` +
    (run.warmFrom !== null ? ` (resumed from cached t = ${fmtChoice(run.warmFrom)})` : '') +
    '.';

  const runEvents: RunEvents = {
    onPhase(phase) {
      const v = fmtChoice(computing?.value ?? NaN);
      if (phase.kind === 'warm-search') {
        status(`${label} = ${v}: looking for a shorter cached run…`);
      } else if (phase.kind === 'seeding') {
        status(`<b>computing ${label} = ${v}</b>: seeding…`);
      } else if (phase.kind === 'encoding') {
        status(`${runLine(phase.run)} Writing the cache file…`);
      } else {
        status(`${runLine(phase.run)} Uploading (${phase.uploaded}/${phase.started})…`);
      }
    },
    onProgress(p) {
      const now = performance.now();
      if (now - lastStatus < STATUS_EVERY_MS) return;
      lastStatus = now;
      const from = p.warmFrom !== null ? `resumed from cached t = ${fmtChoice(p.warmFrom)} — ` : '';
      const up = p.uploadsStarted
        ? `, uploaded ${p.uploadsDone}/${p.uploadsStarted} snapshots`
        : '';
      status(
        `<b>computing ${label} = ${fmtChoice(computing?.value ?? NaN)}</b> (${from}` +
          `t = ${p.t.toFixed(2)} / ${fmtChoice(p.tEnd)}, ${(100 * p.fraction).toFixed(0)}%, ` +
          `${p.rate.toFixed(0)} steps/s${up})`,
      );
    },
    onStepping() {
      for (const r of liveRanges) {
        r.lo = NaN;
        r.hi = NaN;
      }
    },
    async onTick() {
      // As on the main page: no rendering while hidden, and never a wait on
      // an animation frame there, so a background tab computes at full speed.
      const now = performance.now();
      if (document.hidden || now - lastDraw <= RENDER_EVERY_MS) return;
      lastDraw = now;
      await drawLive(gen);
      if (gen !== generation) return;
      await nextFrame();
    },
    async onFinal() {
      // The session holds the finished state: synthesize it into the sweep
      // while it is there, and the value joins the knob's range.
      if (!computing) return;
      computing.fields = await fieldsFromSession();
      computing.status = 'computed';
      recomputeRanges();
      updateTicks();
    },
    onUploaded: () => void uploads++,
    cancelled: () => gen !== generation,
    stopRequested: () => stopRequested,
  };

  await fillWalk({
    targets: missing.map(
      (e): AutoTarget => ({
        model: sel.model,
        params: { ...e.spec.params },
        geometry: sel.geometry,
        geometryParams: { ...e.spec.geometryParams },
        distance: 0,
      }),
    ),
    solver,
    adapter: adapterName,
    runtime: 'browser-webgpu',
    apiKey: () => elApiKey.value.trim(),
    beforeTarget(target) {
      const entry = entries.find((e) => e.spec.params[sweepKey] === target.params[sweepKey])!;
      computing = entry;
      // The knob follows the walk, so the page always says what is running.
      sel.params[sweepKey] = entry.value;
      elKnob.value = String(knobIndex());
      writeUrlState();
      updateTicks();
      return entry.spec;
    },
    events: {
      ...runEvents,
      onTarget: () => status('checking the cloud cache…'),
      onCached(_target) {
        // Somebody else computed it since the page loaded: fetch it after
        // the walk rather than recomputing it here.
        if (computing) computing.status = 'refetch';
      },
      onOutcome(_target, _spec, outcome) {
        if (outcome.kind === 'diverged' && computing) {
          computing.status = 'failed';
          elErr.textContent =
            `${label} = ${fmtChoice(computing.value)}: the solution went non-finite at ` +
            `t = ${outcome.t.toFixed(2)} — nothing uploaded (unstable at this dt)`;
        }
        updateTicks();
      },
      onFailure(_target, spec, e) {
        if (computing) computing.status = 'failed';
        elErr.textContent = `${label} = ${fmtChoice(spec.params[sweepKey])}: ${formatFailure(
          e,
          curModel().source,
        )}`;
        updateTicks();
      },
      walkStopped: () => stopRequested || gen !== generation,
    },
  });
  if (gen !== generation) return;

  // Values that turned out to be cached meanwhile (or that a stopped run
  // uploaded on the way past) are fetched like any other cache hit.
  for (const entry of entries) {
    if (entry.status !== 'refetch') continue;
    try {
      const lookup = await lookupFor(entry.spec);
      const bytes = await fetchCached(lookup);
      if (gen !== generation) return;
      if (!bytes) {
        entry.status = 'missing';
        continue;
      }
      const decoded = await decodeCacheFile(bytes, lookup.specJson, curModel().state);
      if (gen !== generation) return;
      entry.fields = await fieldsFromState(decoded.final);
      entry.status = 'cached';
    } catch {
      entry.status = 'failed';
    }
  }
  if (gen !== generation) return;

  setBusy(false);
  recomputeRanges();
  showCurrent();
  if (stopRequested) {
    status('stopped.' + (uploads ? ` ${uploads} file${uploads > 1 ? 's' : ''} uploaded.` : ''));
  } else {
    updateSweepNote();
  }
}

// ---------------------------------------------------------------- cloud
function updateUploadNote(): void {
  const hasKey = elApiKey.value.trim().length > 0;
  elUploadNote.textContent = hasKey
    ? 'uploads enabled — locally computed solutions will be contributed'
    : '';
  elCliBar.hidden = !hasKey;
  elCliNote.hidden = !hasKey;
  updateCliCommand();
  elCliCopied.textContent = '';
}

/**
 * The command that fills exactly this sweep on a machine with no browser: the
 * page's own URL is the argument, so there is one serialization of what a
 * sweep is (src/cache/selection.ts) and a colleague can paste the same URL
 * into a browser to see the result. Key masked on screen, real in the
 * clipboard, as on the main page.
 */
function sweepFillCommand(key: string): string {
  const url = new URL(`fill.tgz?v=${__BUILD_ID__}`, location.href).href;
  return `TURING_SURFACE_CACHE_KEY=${key} npx ${url} sweep '${location.href}'`;
}

function updateCliCommand(): void {
  if (!elCliBar.hidden) elCliCmd.textContent = sweepFillCommand('…');
}

elCliCopy.addEventListener('click', () => {
  const key = elApiKey.value.trim();
  if (!key) return;
  navigator.clipboard.writeText(sweepFillCommand(key)).then(
    () => {
      elCliCopied.textContent = 'copied';
      setTimeout(() => (elCliCopied.textContent = ''), 4000);
    },
    () => {
      elCliCmd.textContent = sweepFillCommand(key);
      elCliCopied.textContent = 'clipboard unavailable — the key is now shown above';
    },
  );
});

elApiKey.addEventListener('change', () => {
  const key = elApiKey.value.trim();
  if (key) localStorage.setItem(API_KEY_STORAGE, key);
  else localStorage.removeItem(API_KEY_STORAGE);
  updateUploadNote();
});

// ---------------------------------------------------------------- boot
elCompute.addEventListener('click', () => {
  flowChain = flowChain.then(() => computeMissing()).catch(() => undefined);
});
elStop.addEventListener('click', () => {
  stopRequested = true;
});
elReset.addEventListener('click', () => resetDefaults());
elResetView.addEventListener('click', () => {
  for (const s of scenes) s.resetCamera();
});

async function boot(): Promise<void> {
  buildControls();
  rebuildKnob();
  writeUrlState();
  elApiKey.value = localStorage.getItem(API_KEY_STORAGE) ?? '';
  updateUploadNote();
  try {
    device = await requestShtDevice();
    solver = new SolverSession(device, OVERSAMPLE, {
      onCompiling: (m) => status(`compiling ${m.label}…`),
      onSurface: () => rebuildViewFromSession(),
    });
    adapterName = await describeAdapter(device);
  } catch (e) {
    device = null;
    solver = null;
    elErr.textContent =
      `WebGPU is not available (${e instanceof Error ? e.message : e}). ` +
      `Use a WebGPU-capable browser such as Chrome or Edge.`;
    return;
  }
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      elErr.textContent = `WebGPU device lost: ${info.message}`;
    }
  });
  flowChain = flowChain.then(() => reloadSweep()).catch(() => undefined);
  await flowChain;
}

void boot();
