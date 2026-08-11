/**
 * turing-surface-cache: reaction-diffusion solutions at a chosen end time,
 * from a shared cloud cache when someone has computed them before, and from
 * the local GPU when not.
 *
 * Every control is a choice from a short list (src/cache/options.ts), so the
 * page's whole state is one small spec object. Get solution hashes that spec
 * into a cache object name (src/cache/spec.ts) and fetches it; a 404 means
 * nobody has computed it, so the solver runs here — live, watching the
 * pattern form — and stops at exactly the requested time. A run to T passes
 * exactly through every smaller listed end time, so those states are captured
 * along the way; with an upload API key entered, all of them are contributed
 * back to the cache.
 *
 * The solver is turing-surface's, unchanged: the model and geometry are
 * MATLAB compiled (model) or interpreted (geometry) by numbl, the transforms
 * are WGSL compute shaders. lmax, niter and the seed wavelength are fixed in
 * this app (options.ts) — fewer knobs, same machinery.
 */
import { requestShtDevice, describeAdapter } from './sht/sht.ts';
import type { ModelSession } from './mgpu/session.ts';
import { mModels, mModelByKey, type MModel, type Params } from './mgpu/registry.ts';
import { formatFailure } from './mgpu/errors.ts';
import {
  mGeometryByKey,
  DEFAULT_GEOMETRY_KEY,
  mGeometries,
  type MGeometry,
} from './geom/registry.ts';
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
  DEFAULT_MODEL_KEY,
  GEOMETRY_CHOICES,
  SEED_CHOICE,
  T_END_CHOICE,
  LMAX,
  NITER,
  LAM3,
  AUTO_DT,
  defaultChoiceParams,
  fmtChoice,
  type DiscreteChoice,
} from './cache/options.ts';
import { stepsFor, type CacheSpec, APP_NAME, FORMAT_VERSION } from './cache/spec.ts';
import { lookupFor, fetchCached, headCached, type CacheLookup } from './cache/client.ts';
import { autoOrder, specForTarget, type AutoTarget } from './cache/autoWalk.ts';
import { decodeCacheFile } from './cache/h5file.ts';
import { SolverSession } from './cache/solver.ts';
import { runSpec, type RunEvents, type RunOutcome, type RunSummary } from './cache/runSpec.ts';
import { fillWalk } from './cache/fillWalk.ts';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const elModel = $<HTMLSelectElement>('model');
const elParams = $('params');
const elGeometry = $<HTMLSelectElement>('geometry');
const elGeomParams = $('geomparams');
const elSeed = $<HTMLSelectElement>('seed');
const elTend = $<HTMLSelectElement>('tend');
const elSolve = $<HTMLButtonElement>('solve');
const elStop = $<HTMLButtonElement>('stop');
const elReset = $<HTMLButtonElement>('reset');
const elCacheNote = $('cachenote');
const elStatus = $('status');
const elPanels = $('panels');
const elResetView = $<HTMLButtonElement>('resetview');
const elDownload = $<HTMLAnchorElement>('download');
const elStats = $('stats');
const elApiKey = $<HTMLInputElement>('apikey');
const elUploadNote = $('uploadnote');
const elAutoBar = $('autobar');
const elAuto = $<HTMLButtonElement>('auto');
const elAutoNote = $('autonote');
const elCliBar = $('clibar');
const elCliCmd = $('clicmd');
const elCliCopy = $<HTMLButtonElement>('clicopy');
const elCliCopied = $('clicopied');
const elErr = $('err');

/**
 * Test/debug hook: `?tend=5,10` replaces the end-time list with the given
 * values (still cached under their own honest specs — a test end time hashes
 * to its own object). The headless checks use this to keep their computed
 * runs short; it is not part of the normal UI.
 */
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

const API_KEY_STORAGE = `${APP_NAME}:apiKey`;
const COLORMAP = colormaps.viridis;
/** Render on a 2x finer grid than the solver's; exact interpolation. */
const OVERSAMPLE = 2;
/** How often the live view renders during a computation. */
const RENDER_EVERY_MS = 250;
/** How often the status line is rewritten during a computation. */
const STATUS_EVERY_MS = 200;

// ---------------------------------------------------------------- state
let model: MModel = mModelByKey(DEFAULT_MODEL_KEY)!;
let device: GPUDevice | null = null;
/** The compiled solver and what it has applied (src/cache/solver.ts). */
let solver: SolverSession | null = null;
let adapterName = '';
/** The live session, or null before the GPU is up. */
function sess(): ModelSession | null {
  return solver?.session ?? null;
}

/** The discrete selections, always exactly values from options.ts. */
let params: Params = defaultChoiceParams(MODEL_CHOICES[DEFAULT_MODEL_KEY]);
let geometry: MGeometry = mGeometryByKey(DEFAULT_GEOMETRY_KEY)!;
let geomParams: Params = Object.fromEntries(
  GEOMETRY_CHOICES[DEFAULT_GEOMETRY_KEY].map((c) => [c.key, c.value]),
);
let seed = SEED_CHOICE.value;
let tEnd = T_END_CHOICE.value;

// The URL fragment carries the whole selection, so a reload comes back to it
// and a shared link opens on the same spec (and, through refresh(), the same
// cached solution). Read once at startup; rewritten on every change.
readUrlState();

let topo: SphereMeshTopology | null = null;
let scenes: SphereScene[] = [];
let colorbars: Colorbar[] = [];
/** The colorbar containers, hidden while the windows are empty. */
let colorbarEls: HTMLElement[] = [];
let valueBufs: Float32Array[] = [];
let colorBufs: Float32Array[] = [];
let ranges: { lo: number; hi: number }[] = [];
let resizeObs: ResizeObserver | null = null;
let coords: Float32Array | null = null;
let posBuf: Float32Array | null = null;

let generation = 0;
let busy = false;
/** True while computeLocally is stepping/reading back. Every read shares one
 *  staging buffer (GpuModel#readback), so a new solve must drain the old
 *  loop before issuing reads of its own. */
let pumping = false;
let stopRequested = false;
/** Set while the auto-fill walk owns the page (see autoRun). */
let autoRunning = false;
let autoComputed = 0;
let autoSkipped = 0;
let autoFailed = 0;
/** Simulation time of the state on display (loadState resets session.t). */
let shownT: number | null = null;
let downloadUrl: string | null = null;

const nextFrame = () => new Promise<number>(requestAnimationFrame);

// ---------------------------------------------------------------- spec
function currentSpec(): CacheSpec {
  return {
    app: APP_NAME,
    formatVersion: FORMAT_VERSION,
    model: model.key,
    params: { ...params },
    geometry: geometry.key,
    geometryParams: { ...geomParams },
    lmax: LMAX,
    niter: NITER,
    lam3: LAM3,
    seed,
    tEnd,
  };
}

// ---------------------------------------------------------------- URL state
/**
 * The selection lives in the URL fragment, every value written explicitly
 * (`#a=0.1&b=0.9&…&geometry=ellipsoid&ax=1.5&…&seed=1&tend=100`), so a link
 * keeps meaning the same spec even if a default changes later. The fragment
 * is chosen over the query string to leave `?tend` to the test hook. Values
 * are only accepted if they are exactly entries of the discrete lists;
 * anything else keeps the default.
 */
function readUrlState(): void {
  const hash = location.hash.replace(/^#/, '');
  if (!hash) return;
  const p = new URLSearchParams(hash);
  // `name` is the key as it appears in the URL; it defaults to the choice's
  // own key but is passed explicitly where the two differ (tEnd vs tend).
  const pick = (choice: DiscreteChoice, current: number, name = choice.key): number => {
    const raw = p.get(name);
    if (raw === null) return current;
    const v = Number(raw);
    return choice.values.includes(v) ? v : current;
  };
  const m = p.get('model');
  if (m && mModelByKey(m) && MODEL_CHOICES[m]) {
    model = mModelByKey(m)!;
    params = defaultChoiceParams(MODEL_CHOICES[m]);
  }
  const g = p.get('geometry');
  if (g && mGeometryByKey(g) && GEOMETRY_CHOICES[g]) {
    geometry = mGeometryByKey(g)!;
    geomParams = defaultChoiceParams(GEOMETRY_CHOICES[g]);
  }
  for (const c of MODEL_CHOICES[model.key]) params[c.key] = pick(c, params[c.key]);
  for (const c of GEOMETRY_CHOICES[geometry.key]) geomParams[c.key] = pick(c, geomParams[c.key]);
  seed = pick(SEED_CHOICE, seed);
  tEnd = pick(T_END_CHOICE, tEnd, 'tend');
}

function writeUrlState(): void {
  const p = new URLSearchParams();
  p.set('model', model.key);
  for (const c of MODEL_CHOICES[model.key]) p.set(c.key, fmtChoice(params[c.key]));
  p.set('geometry', geometry.key);
  for (const c of GEOMETRY_CHOICES[geometry.key]) p.set(c.key, fmtChoice(geomParams[c.key]));
  p.set('seed', String(seed));
  p.set('tend', fmtChoice(tEnd));
  history.replaceState(null, '', `${location.pathname}${location.search}#${p.toString()}`);
}

// ---------------------------------------------------------------- controls
/** Every select made by makeSelect, so a reset can push new values into the
 *  ones still on the page. */
const boundSelects: { el: HTMLSelectElement; get: () => number }[] = [];

function syncSelects(): void {
  // Pruned as it goes: auto mode rebuilds the parameter controls once per
  // target, so entries for replaced selects would otherwise pile up.
  for (let i = boundSelects.length - 1; i >= 0; i--) {
    const b = boundSelects[i];
    if (b.el.isConnected) b.el.value = String(b.get());
    else boundSelects.splice(i, 1);
  }
}

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
  boundSelects.push({ el: select, get });
  return label;
}

/** Put every selection back to its default, without refreshing the display. */
function applyDefaults(): void {
  model = mModelByKey(DEFAULT_MODEL_KEY)!;
  params = defaultChoiceParams(MODEL_CHOICES[DEFAULT_MODEL_KEY]);
  geometry = mGeometryByKey(DEFAULT_GEOMETRY_KEY)!;
  elModel.value = model.key;
  buildModelParamControls();
  geomParams = defaultChoiceParams(GEOMETRY_CHOICES[DEFAULT_GEOMETRY_KEY]);
  seed = SEED_CHOICE.value;
  tEnd = T_END_CHOICE.value;
  elGeometry.value = geometry.key;
  buildGeomParamControls();
  elSeed.value = String(seed);
  elTend.value = String(tEnd);
  syncSelects();
  writeUrlState();
}

/** The Reset button: back to the defaults, and show what is there. */
function resetDefaults(): void {
  applyDefaults();
  onSelectionChange();
}

/** Point every control at one walk target (auto mode drives the same
 *  selection the user otherwise would, so the URL and the dropdowns always
 *  say what is being computed). The values come from the target's own spec,
 *  so currentSpec() reproduces exactly what the walk asked for. */
function setSelection(t: AutoTarget): void {
  const spec = specForTarget(t);
  const nextModel = mModelByKey(spec.model)!;
  if (nextModel !== model) {
    model = nextModel;
    elModel.value = model.key;
    buildModelParamControls();
  }
  params = { ...spec.params };
  const nextGeom = mGeometryByKey(spec.geometry)!;
  if (nextGeom !== geometry) {
    geometry = nextGeom;
    elGeometry.value = geometry.key;
    buildGeomParamControls();
  }
  geomParams = { ...spec.geometryParams };
  seed = spec.seed;
  elSeed.value = String(seed);
  tEnd = spec.tEnd;
  elTend.value = String(tEnd);
  syncSelects();
  writeUrlState();
}

function buildModelParamControls(): void {
  elParams.replaceChildren();
  for (const choice of MODEL_CHOICES[model.key]) {
    elParams.append(
      makeSelect(choice, () => params[choice.key], (v) => (params[choice.key] = v)),
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
  elModel.value = model.key;
  elModel.addEventListener('change', () => {
    model = mModelByKey(elModel.value)!;
    params = defaultChoiceParams(MODEL_CHOICES[model.key]);
    buildModelParamControls();
    onSelectionChange();
  });
  buildModelParamControls();
  for (const g of mGeometries) {
    const opt = document.createElement('option');
    opt.value = g.key;
    opt.textContent = g.label.toLowerCase();
    elGeometry.append(opt);
  }
  elGeometry.value = geometry.key;
  elGeometry.addEventListener('change', () => {
    geometry = mGeometryByKey(elGeometry.value)!;
    geomParams = Object.fromEntries(
      GEOMETRY_CHOICES[geometry.key].map((c) => [c.key, c.value]),
    );
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
  elSeed.value = String(seed);
  elSeed.addEventListener('change', () => {
    seed = Number(elSeed.value);
    onSelectionChange();
  });

  for (const v of T_END_CHOICE.values) {
    const opt = document.createElement('option');
    opt.value = String(v);
    opt.textContent = String(v);
    elTend.append(opt);
  }
  elTend.value = String(tEnd);
  elTend.addEventListener('change', () => {
    tEnd = Number(elTend.value);
    onSelectionChange();
  });
}

function buildGeomParamControls(): void {
  elGeomParams.replaceChildren();
  for (const choice of GEOMETRY_CHOICES[geometry.key]) {
    elGeomParams.append(
      makeSelect(choice, () => geomParams[choice.key], (v) => (geomParams[choice.key] = v)),
    );
  }
}

/**
 * A selection change refreshes the display: a cached solution loads and
 * shows immediately, an uncached one shows empty surfaces until the user
 * explicitly presses Compute solution. While a computation is running the
 * change touches nothing — the run keeps going and only the is-it-cached
 * note follows the dropdowns.
 *
 * Refreshes and button presses are chained so two flows never talk to the
 * session at once.
 */
let flowChain: Promise<void> = Promise.resolve();
function onSelectionChange(): void {
  writeUrlState();
  // During a computation the refresh is deferred until the run finishes; the
  // is-it-cached note should follow the dropdowns right away regardless.
  if (busy) void updateCacheNote();
  flowChain = flowChain.then(() => refresh()).catch(() => undefined);
}

// The note carries a token so a slow HEAD for a superseded selection never
// overwrites the note for the current one.
let cacheNoteToken = 0;
async function updateCacheNote(): Promise<void> {
  const token = ++cacheNoteToken;
  elCacheNote.textContent = '';
  let lookup: CacheLookup;
  try {
    lookup = await lookupFor(currentSpec());
  } catch {
    return;
  }
  const present = await headCached(lookup);
  if (token !== cacheNoteToken) return;
  setCacheNote(present);
}

function setCacheNote(present: boolean | null): void {
  if (present === true) {
    elCacheNote.innerHTML = '<b>✓ in the cloud cache</b>';
  } else if (present === false) {
    elCacheNote.textContent = 'not cached yet';
  } else {
    elCacheNote.textContent = '';
  }
}

// ---------------------------------------------------------------- view
function disposeView(): void {
  for (const s of scenes) s.dispose();
  scenes = [];
  colorbars = [];
  colorbarEls = [];
  topo = null;
  coords = null;
  posBuf = null;
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
  coords = surface;
  posBuf = new Float32Array(topo.numVertices * 3);
  fillPositions(posBuf, coords, topo, 1);

  const sphereBg = getComputedStyle(document.documentElement)
    .getPropertyValue('--sphere-bg')
    .trim();
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
    valueBufs[k] = new Float32Array(topo.numVertices);
    colorBufs[k] = new Float32Array(topo.numVertices * 3);
    ranges[k] = { lo: NaN, hi: NaN };
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

async function draw(): Promise<void> {
  const session = sess();
  if (!session || !topo) return;
  const gen = generation;
  for (let k = 0; k < model.species.length; k++) {
    let field: Float32Array;
    try {
      field = await session.readSpecies(k);
    } catch (e) {
      if (gen !== generation) return;
      throw e;
    }
    if (gen !== generation || !topo) return;
    fillFieldValues(valueBufs[k], field, topo);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of valueBufs[k]) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    // Smooth the color range in both directions so the shading evolves gently
    // as the pattern grows (out-of-range values clamp meanwhile).
    const r = ranges[k];
    if (!Number.isFinite(r.lo)) {
      r.lo = lo;
      r.hi = hi;
    } else {
      const a = 0.15;
      r.lo += a * (lo - r.lo);
      r.hi += a * (hi - r.hi);
    }
    const shown = floorRange(r.lo, r.hi);
    fillColors(colorBufs[k], valueBufs[k], shown.lo, shown.hi, COLORMAP);
    scenes[k]?.updateColors(colorBufs[k]);
    colorbars[k]?.update(COLORMAP, shown.lo, shown.hi);
    if (colorbarEls[k]) colorbarEls[k].style.visibility = '';
  }
}

/** Empty windows: the selected surface with no field on it. Shown when the
 *  selection has no cached solution and nothing has been computed yet. */
function clearDisplay(): void {
  shownT = null;
  elDownload.hidden = true;
  if (!topo) return;
  for (let k = 0; k < model.species.length; k++) {
    // NaN renders as neutral gray in fillColors — the shape without a field.
    valueBufs[k].fill(NaN);
    fillColors(colorBufs[k], valueBufs[k], 0, 1, COLORMAP);
    scenes[k]?.updateColors(colorBufs[k]);
    if (colorbarEls[k]) colorbarEls[k].style.visibility = 'hidden';
  }
  updateStats();
}

function resetRanges(): void {
  for (const r of ranges) {
    r.lo = NaN;
    r.hi = NaN;
  }
}

function updateStats(): void {
  const session = sess();
  if (!session) return;
  const { nlat, nphi } = session.cfg;
  const kind = `WebGPU fp32${adapterName ? ` — ${adapterName}` : ''}`;
  const t = shownT !== null ? ` · showing t = <b>${fmtChoice(shownT)}</b>` : '';
  elStats.innerHTML =
    `<b>${kind}</b> · grid ${nlat}×${nphi} · lmax ${LMAX} · ` +
    `solve iters ${NITER}${t}`;
}

// ---------------------------------------------------------------- statuses
function status(html: string): void {
  elStatus.innerHTML = html;
}

function setBusy(next: boolean): void {
  busy = next;
  elSolve.disabled = next;
  elStop.hidden = !next;
}

function offerDownload(bytes: Uint8Array, name: string): void {
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([bytes as BlobPart], { type: 'application/x-hdf5' }));
  elDownload.href = downloadUrl;
  elDownload.download = name;
  elDownload.hidden = false;
}

// ---------------------------------------------------------------- solving
/** Rebuild the mesh and panels from the session's current surface, keeping
 *  the camera. Fresh buffers render black until the first fill, so the bare
 *  surface is shown; the caller's draw or clearDisplay follows right behind. */
async function rebuildViewFromSession(): Promise<void> {
  const session = sess();
  if (!session) return;
  const surface = await session.renderPositions();
  const cam = scenes[0]?.cameraState();
  disposeView();
  buildView(surface);
  if (cam) for (const s of scenes) s.setCameraState(cam);
  clearDisplay();
}

/**
 * Apply a selection to the solver. Which changes are cheap and which pay a
 * recompile is the solver's business (src/cache/solver.ts); the page adds the
 * compiling status and the rebuilt view through the events it installs in
 * boot(), since the panel count follows the model's species (Allen–Cahn has
 * one).
 */
async function applySelection(spec: CacheSpec): Promise<void> {
  if (!solver) throw new Error('no GPU device');
  await solver.apply(spec);
}

/** Decode a fetched cache file and put it on screen. */
async function displayCached(
  bytes: Uint8Array,
  lookup: CacheLookup,
  spec: CacheSpec,
  gen: number,
): Promise<void> {
  const session = sess();
  if (!session) return;
  const decoded = await decodeCacheFile(bytes, lookup.specJson, model.state);
  if (gen !== generation) return;
  session.loadState(decoded.final);
  shownT = spec.tEnd;
  resetRanges();
  await draw();
  updateStats();
  const kb = (bytes.length / 1024).toFixed(0);
  const from = decoded.adapter ? `, computed on ${decoded.adapter}` : '';
  const when = decoded.created ? ` ${decoded.created.slice(0, 10)}` : '';
  status(
    `<b>t = ${fmtChoice(spec.tEnd)}</b> — from the <b>cloud cache</b> ` +
      `(${kb} KB${from}${when}).`,
  );
  offerDownload(bytes, lookup.fileName.split('/').pop()!);
}

/**
 * Bring the display in line with the current selection, without ever
 * starting a computation: a cached solution loads and shows, an uncached one
 * shows empty surfaces and waits for the Compute solution button. Runs on
 * startup and on every selection change; a no-op while a computation is
 * running (the run is not disturbed — only the cache note follows).
 */
async function refresh(): Promise<void> {
  // Before the GPU is up there is nothing to refresh; while a computation
  // runs the note follows the dropdowns and the refresh waits its turn. A
  // missing session is NOT a reason to bail: applySelection rebuilds it,
  // which is also what recovers from a failed compile.
  if (!device || busy) {
    void updateCacheNote();
    return;
  }
  generation++;
  const gen = generation;
  elErr.textContent = '';
  const spec = currentSpec();
  try {
    const lookup = await lookupFor(spec);
    status('checking the cloud cache…');
    let bytes: Uint8Array | null = null;
    let unreachable = false;
    try {
      bytes = await fetchCached(lookup);
    } catch {
      unreachable = true;
    }
    if (gen !== generation) return;
    await applySelection(spec);
    if (gen !== generation) return;
    if (bytes) {
      await displayCached(bytes, lookup, spec, gen);
      setCacheNote(true);
      return;
    }
    clearDisplay();
    setCacheNote(unreachable ? null : false);
    status(
      unreachable
        ? 'cloud cache unreachable — <b>Compute solution</b> runs it in your browser.'
        : `not in the cloud cache — press <b>Compute solution</b> to run it in ` +
          `your browser (up to ${stepsFor(spec).toLocaleString()} steps; a ` +
          `cached shorter run of the same settings is picked up where it left off).`,
    );
  } catch (e) {
    if (gen === generation) {
      elErr.textContent = formatFailure(e, model.source);
      status('failed.');
    }
  }
}

/** The Compute solution button: cache lookup, then either load or compute. */
async function solve(): Promise<void> {
  if (!device || busy) return;
  generation++;
  const gen = generation;
  setBusy(true);
  // A stopped run may still be inside an await; let it see the generation
  // bump and finish before touching the session.
  while (pumping) await nextFrame();
  if (gen !== generation) return;
  stopRequested = false;
  elErr.textContent = '';
  elDownload.hidden = true;
  const spec = currentSpec();
  try {
    const lookup = await lookupFor(spec);
    status('checking the cloud cache…');
    let bytes: Uint8Array | null = null;
    try {
      bytes = await fetchCached(lookup);
    } catch (e) {
      // An unreachable cache degrades to computing locally, and says so.
      status(`cache unreachable (${e instanceof Error ? e.message : e}) — computing locally`);
    }
    if (gen !== generation) return;
    await applySelection(spec);
    if (gen !== generation) return;

    if (bytes) {
      await displayCached(bytes, lookup, spec, gen);
      return;
    }
    await computeLocally(spec, gen);
  } catch (e) {
    if (gen === generation) {
      elErr.textContent = formatFailure(e, model.source);
      status('failed.');
    }
  } finally {
    if (gen === generation) setBusy(false);
    void updateCacheNote();
  }
}

/**
 * How the page tells a run in progress: the status line, the live view, and
 * when to give up. The same events drive the Compute solution button and the
 * auto-fill walk, so the two report a run identically.
 *
 * `gen` is read afresh at every check rather than captured, so the events a
 * walk installs once still speak for whichever target is current.
 */
function runEvents(gen: () => number): RunEvents {
  let lastStatus = 0;
  let lastDraw = 0;
  return {
    onPhase(phase) {
      if (phase.kind === 'warm-search') {
        status('not in the cache — looking for a shorter cached run…');
      } else if (phase.kind === 'seeding') {
        status('not in the cache — <b>computing locally</b>: seeding…');
      } else if (phase.kind === 'encoding') {
        status(`${doneLine(phase.run)} Writing the cache file…`);
      } else {
        status(
          `${doneLine(phase.run)} Uploading to the cache ` +
            `(${phase.uploaded}/${phase.started})…`,
        );
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
        `not in the cache — <b>computing locally</b> (${from}` +
          `t = ${p.t.toFixed(2)} / ${fmtChoice(p.tEnd)}, ${(100 * p.fraction).toFixed(0)}%, ` +
          `${p.rate.toFixed(0)} steps/s${up})`,
      );
    },
    onStepping() {
      shownT = null;
      resetRanges();
    },
    async onTick() {
      // Rendering is skipped entirely while the page is hidden, and the loop
      // never waits on an animation frame there: a backgrounded tab throttles
      // or stops requestAnimationFrame, which would stall an unattended run.
      // The GPU sync inside the run already yields to the event loop, so Stop
      // stays responsive either way.
      const now = performance.now();
      if (document.hidden || now - lastDraw <= RENDER_EVERY_MS) return;
      lastDraw = now;
      await draw();
      if (gen() !== generation) return;
      await nextFrame();
    },
    async onFinal(tEnd) {
      shownT = tEnd;
      await draw();
      updateStats();
    },
    onFile(bytes, name) {
      offerDownload(bytes, name);
    },
    cancelled: () => gen() !== generation,
    stopRequested: () => stopRequested,
  };
}

/** The first sentence of every finished run's status. */
function doneLine(run: RunSummary): string {
  return (
    `<b>t = ${fmtChoice(run.tEnd)}</b> — computed locally in ${run.seconds.toFixed(1)} s` +
    (run.warmFrom !== null ? ` (resumed from cached t = ${fmtChoice(run.warmFrom)})` : '') +
    `.`
  );
}

/** Say how a finished run ended. Returns nothing; the caller counts. */
async function reportOutcome(outcome: RunOutcome): Promise<void> {
  if (outcome.kind === 'abandoned') return;
  if (outcome.kind === 'diverged') {
    elErr.textContent =
      `the solution went non-finite at t = ${outcome.t.toFixed(2)} — nothing uploaded ` +
      `(this combination is unstable at dt = ${fmtChoice(AUTO_DT)})`;
    status('diverged.');
    return;
  }
  if (outcome.kind === 'stopped') {
    shownT = outcome.t;
    await draw();
    updateStats();
    const n = outcome.uploaded.length;
    const up = n ? ` ${n} snapshot${n > 1 ? 's' : ''} already uploaded.` : ' Nothing uploaded.';
    status(`stopped at t = ${outcome.t.toFixed(2)}.${up}`);
    return;
  }
  const line = doneLine(outcome);
  if (outcome.uploadsStarted === 0) {
    status(`${line} Not uploaded (no API key).`);
    return;
  }
  if (outcome.uploadErrors.length) {
    elErr.textContent = `upload: ${outcome.uploadErrors.join('; ')}`;
  }
  const n = outcome.uploaded.length;
  if (n > 0) {
    const times = [...outcome.uploaded].sort((a, b) => a - b).map(fmtChoice).join(', ');
    const failed = outcome.uploadErrors.length
      ? ` (${outcome.uploadErrors.length} failed)`
      : '';
    status(
      `${line} <b>Uploaded ${n} solution${n > 1 ? 's' : ''}</b> ` +
        `to the shared cache (t = ${times})${failed}.`,
    );
  } else {
    status(`${line} Uploads failed.`);
  }
}

/**
 * Run the solver to the spec's end time, watching the pattern form, and
 * capture the state at every smaller listed end time on the way
 * (src/cache/runSpec.ts). Everything the page adds is in runEvents and
 * reportOutcome.
 */
async function computeLocally(spec: CacheSpec, gen: number): Promise<RunOutcome> {
  if (!solver?.session) return { kind: 'abandoned' };
  pumping = true;
  try {
    const outcome = await runSpec({
      solver,
      spec,
      adapter: adapterName,
      apiKey: () => elApiKey.value.trim(),
      events: runEvents(() => gen),
    });
    if (gen === generation) await reportOutcome(outcome);
    return outcome;
  } finally {
    pumping = false;
  }
}

// ---------------------------------------------------------------- auto-fill
function autoNote(target: AutoTarget | null): void {
  if (!autoRunning) {
    elAutoNote.textContent = autoComputed || autoSkipped
      ? `stopped — computed ${autoComputed}, skipped ${autoSkipped} already cached` +
        (autoFailed ? `, ${autoFailed} failed` : '')
      : '';
    return;
  }
  const where = target
    ? `${mModelByKey(target.model)!.label} on ${target.geometry}, ${target.distance} ` +
      `knob${target.distance === 1 ? '' : 's'} from the defaults`
    : '';
  elAutoNote.textContent =
    `auto-filling — computed ${autoComputed}, skipped ${autoSkipped}` +
    (autoFailed ? `, ${autoFailed} failed` : '') + (where ? ` · ${where}` : '');
}

function setAutoUi(on: boolean): void {
  elAuto.textContent = on ? 'Auto-filling…' : 'Auto-fill the cache';
  elAuto.disabled = on;
  elReset.disabled = on;
}

/**
 * Walk the parameter space on this machine, computing and contributing
 * whatever is not cached yet, nearest the defaults first and randomly within
 * a distance (src/cache/autoWalk.ts, src/cache/fillWalk.ts). Runs until
 * stopped.
 *
 * Every target is driven through the same selection the user would set by
 * hand, so the dropdowns and the URL always say what is being computed, and
 * the run itself is the ordinary local computation — including its background
 * uploads, its warm start from a shorter cached run, and its divergence
 * guard.
 */
async function autoRun(): Promise<void> {
  if (!device || !solver || busy || autoRunning) return;
  if (!elApiKey.value.trim()) return;
  autoRunning = true;
  autoComputed = autoSkipped = autoFailed = 0;
  setAutoUi(true);
  setBusy(true);
  elErr.textContent = '';
  // Start from a defined point — which is also the first target, since the
  // defaults are the one combination at distance zero.
  applyDefaults();
  autoNote(null);

  // The generation of the target being computed, read by the run events.
  let walkGen = 0;
  await fillWalk({
    targets: autoOrder(),
    solver,
    adapter: adapterName,
    apiKey: () => elApiKey.value.trim(),
    beforeTarget(target) {
      setSelection(target);
      autoNote(target);
      generation++;
      walkGen = generation;
      stopRequested = false;
      return currentSpec();
    },
    events: {
      ...runEvents(() => walkGen),
      onTarget: () => status('checking the cloud cache…'),
      onCached: (target) => {
        autoSkipped++;
        setCacheNote(true);
        autoNote(target);
      },
      onComputing: () => setCacheNote(false),
      onOutcome: (target, _spec, outcome) => {
        if (outcome.kind === 'done') autoComputed++;
        else if (outcome.kind === 'diverged') autoFailed++;
        autoNote(target);
      },
      onFailure: (target, spec, e) => {
        autoFailed++;
        elErr.textContent =
          `auto (${spec.model}, ${spec.geometry}): ${formatFailure(e, model.source)}`;
        autoNote(target);
      },
      walkStopped: () => !autoRunning,
    },
  });

  autoRunning = false;
  setAutoUi(false);
  setBusy(false);
  autoNote(null);
}

// ---------------------------------------------------------------- boot
elAuto.addEventListener('click', () => {
  flowChain = flowChain.then(() => autoRun()).catch(() => undefined);
});
elSolve.addEventListener('click', () => {
  flowChain = flowChain.then(() => solve()).catch(() => undefined);
});
elStop.addEventListener('click', () => {
  stopRequested = true;
  autoRunning = false;
  setBusy(false);
});
elReset.addEventListener('click', () => resetDefaults());
elResetView.addEventListener('click', () => {
  for (const s of scenes) s.resetCamera();
});
// The view is not drawn while the page is hidden, so it is stale on return.
// Not while a run is reading back: every read shares one staging buffer.
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && sess() && !pumping) void draw();
});
elApiKey.addEventListener('change', () => {
  const key = elApiKey.value.trim();
  if (key) localStorage.setItem(API_KEY_STORAGE, key);
  else localStorage.removeItem(API_KEY_STORAGE);
  updateUploadNote();
});

function updateUploadNote(): void {
  const hasKey = elApiKey.value.trim().length > 0;
  elUploadNote.textContent = hasKey
    ? 'uploads enabled — locally computed solutions will be contributed'
    : '';
  // Auto-fill exists to contribute, so it is offered only to those who can,
  // and so is the command that does the same thing elsewhere.
  elAutoBar.hidden = !hasKey;
  elCliBar.hidden = !hasKey;
  elCliCmd.textContent = fillCommand('…');
  elCliCopied.textContent = '';
  if (!hasKey && autoRunning) autoRunning = false;
}

/**
 * The command that runs this same walk outside a browser. The tarball is
 * deployed beside the page, so the URL is derived from this one and a preview
 * deployment hands out its own command rather than main's.
 *
 * The key travels in the environment rather than in an option because argv is
 * visible to every user on the machine through `ps`, while another process's
 * environment is not. It is masked on screen and real in the clipboard: the
 * displayed command would otherwise put the key in any screenshot of a page
 * that has one, which is what the password field exists to prevent.
 */
function fillCommand(key: string): string {
  const url = new URL('fill.tgz', location.href).href;
  return `TURING_SURFACE_CACHE_KEY=${key} npx ${url}`;
}

elCliCopy.addEventListener('click', () => {
  const key = elApiKey.value.trim();
  if (!key) return;
  navigator.clipboard.writeText(fillCommand(key)).then(
    () => {
      elCliCopied.textContent = 'copied';
      setTimeout(() => (elCliCopied.textContent = ''), 4000);
    },
    () => {
      // No clipboard (an insecure origin, usually). Copying was the intent, so
      // show the whole thing and let it be selected by hand.
      elCliCmd.textContent = fillCommand(key);
      elCliCopied.textContent = 'clipboard unavailable — the key is now shown above';
    },
  );
});

async function boot(): Promise<void> {
  buildControls();
  // Written even before any change, so the address bar is always shareable.
  writeUrlState();
  elApiKey.value = localStorage.getItem(API_KEY_STORAGE) ?? '';
  updateUploadNote();
  void updateCacheNote();
  try {
    device = await requestShtDevice();
    // Before the adapter is even described, so a selection change during boot
    // finds a solver to apply itself to rather than an error.
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

  try {
    await applySelection(currentSpec());
  } catch (e) {
    elErr.textContent = formatFailure(e, model.source);
    status('failed to compile.');
    return;
  }
  // Bring up the default selection if it is cached; otherwise show empty
  // surfaces. Nothing is ever computed without pressing the button.
  flowChain = flowChain.then(() => refresh()).catch(() => undefined);
  await flowChain;
}

void boot();
