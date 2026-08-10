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
import { ModelSession } from './mgpu/session.ts';
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
  defaultChoiceParams,
  fmtChoice,
  type DiscreteChoice,
} from './cache/options.ts';
import { stepsFor, type CacheSpec, APP_NAME, FORMAT_VERSION } from './cache/spec.ts';
import { lookupFor, fetchCached, uploadCacheFile, type CacheLookup } from './cache/client.ts';
import { encodeCacheFile, decodeCacheFile, type DecodedCacheFile } from './cache/h5file.ts';

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
/** Cap on GPU dispatches per submission (watchdog safety; see turing-surface). */
const DISPATCH_BUDGET = 1000;
/** Steps between syncs during a computation: many small submissions queued
 *  back to back, one wait. The readbacks and renders that pace the live view
 *  happen per chunk, not per submission — that is what lets the run advance
 *  at close to the solver's own rate. */
const CHUNK_STEPS = 32;
/** How often the live view renders during a computation. */
const RENDER_EVERY_MS = 250;

// ---------------------------------------------------------------- state
let model: MModel = mModelByKey(DEFAULT_MODEL_KEY)!;
let device: GPUDevice | null = null;
let session: ModelSession | null = null;
let adapterName = '';
/** Steps per GPU submission, sized in boot() so one submission stays under
 *  the dispatch budget however expensive niter has made a step. */
let stepsPerSubmit = 4;

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

/** What the session currently has applied. Params are cheap (uniforms); a
 *  geometry change re-evaluates the surface and rebuilds the mesh; a model
 *  change recompiles the whole session, since the model is compiled into the
 *  GPU step. */
let sessionModelKey = '';
let sessionGeomKey = '';
let sessionGeomParams: Params = {};

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
  for (const b of boundSelects) {
    if (b.el.isConnected) b.el.value = String(b.get());
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

/** Put every selection back to its default and refresh. */
function resetDefaults(): void {
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
  onSelectionChange();
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
  let present: boolean | null = null;
  try {
    const res = await fetch(lookup.url, { method: 'HEAD', cache: 'no-store' });
    present = res.ok ? true : res.status === 404 ? false : null;
  } catch {
    present = null;
  }
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
  if (!session) return;
  const surface = await session.renderPositions();
  const cam = scenes[0]?.cameraState();
  disposeView();
  buildView(surface);
  if (cam) for (const s of scenes) s.setCameraState(cam);
  clearDisplay();
}

/**
 * Compile a full session for the spec's model. The model is the one
 * selection that cannot be swapped into a running session — its step is
 * compiled into the GPU pipelines — so changing it pays a recompile
 * (a second or two on a real GPU). The panel count follows the model's
 * species (Allen–Cahn has one), so the view is rebuilt too.
 */
async function rebuildSession(spec: CacheSpec): Promise<void> {
  if (!device) throw new Error('no GPU device');
  const nextModel = mModelByKey(spec.model)!;
  const geomModel = mGeometryByKey(spec.geometry)!;
  session?.destroy();
  session = null;
  sessionModelKey = '';
  status(`compiling ${nextModel.label}…`);
  session = await ModelSession.create({
    device,
    model: nextModel,
    params: spec.params,
    lmax: spec.lmax,
    oversample: OVERSAMPLE,
    geometry: geomModel,
    geometryParams: spec.geometryParams,
    niter: spec.niter,
    lam3: spec.lam3,
  });
  model = nextModel;
  sessionModelKey = spec.model;
  sessionGeomKey = spec.geometry;
  sessionGeomParams = { ...spec.geometryParams };
  // Never put more dispatches in one submission than the budget allows,
  // however expensive this model's step is.
  const opsPerStep = Math.max(1, session.describe().step.length);
  stepsPerSubmit = Math.max(1, Math.floor(DISPATCH_BUDGET / opsPerStep));
  await rebuildViewFromSession();
  updateStats();
}

/** Apply the current selection to the session: params are a uniform upload;
 *  a geometry change re-evaluates the surface and rebuilds the mesh; a model
 *  change recompiles the session entirely. */
async function applySelection(spec: CacheSpec): Promise<void> {
  if (!session || spec.model !== sessionModelKey) {
    await rebuildSession(spec);
    return;
  }
  session.setParams(spec.params);
  const geomChanged =
    spec.geometry !== sessionGeomKey ||
    JSON.stringify(spec.geometryParams) !== JSON.stringify(sessionGeomParams);
  if (!geomChanged) return;
  const geomModel = mGeometryByKey(spec.geometry)!;
  await session.setGeometry(geomModel, spec.geometryParams);
  sessionGeomKey = spec.geometry;
  sessionGeomParams = { ...spec.geometryParams };
  await rebuildViewFromSession();
}

/** Decode a fetched cache file and put it on screen. */
async function displayCached(
  bytes: Uint8Array,
  lookup: CacheLookup,
  spec: CacheSpec,
  gen: number,
): Promise<void> {
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

/** Run the solver to the spec's end time, watching the pattern form, and
 *  capture the state at every smaller listed end time on the way. */
async function computeLocally(spec: CacheSpec, gen: number): Promise<void> {
  if (!session) return;
  pumping = true;
  try {
    await computeLocallyInner(spec, gen);
  } finally {
    pumping = false;
  }
}

async function computeLocallyInner(spec: CacheSpec, gen: number): Promise<void> {
  if (!session) return;
  const steps = stepsFor(spec);
  const dt = spec.params.dt;

  // Warm start: the state is Markovian in (U, V), so a cached run of the
  // same spec at a smaller listed end time is an exact prefix of this one.
  // Take the longest one there is and continue from its final state rather
  // than recomputing it.
  let warm: { tEnd: number; decoded: DecodedCacheFile } | null = null;
  const earlier = T_END_CHOICE.values.filter((T) => T < spec.tEnd).sort((a, b) => b - a);
  if (earlier.length) status('not in the cache — looking for a shorter cached run…');
  for (const T of earlier) {
    const lookup = await lookupFor({ ...spec, tEnd: T });
    let bytes: Uint8Array | null = null;
    try {
      bytes = await fetchCached(lookup);
    } catch {
      break; // cache unreachable: no point probing further down the ladder
    }
    if (gen !== generation) return;
    if (!bytes) continue;
    try {
      warm = { tEnd: T, decoded: await decodeCacheFile(bytes, lookup.specJson, model.state) };
      break;
    } catch {
      continue; // an unreadable candidate is skipped, not fatal
    }
  }
  if (gen !== generation) return;

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
    status(`not in the cache — <b>computing locally</b>: seeding…`);
    await session.seed(spec.seed);
    if (gen !== generation) return;
    initial = await session.readState();
    if (gen !== generation) return;
  }
  const startSteps = session.steps;

  // Snapshot points: every listed end time strictly between the starting
  // point and this run's end. The run passes through each exactly (all are
  // whole multiples of every dt choice).
  const snapshotAt = new Map<number, number>(); // step index -> tEnd value
  for (const T of T_END_CHOICE.values) {
    if (T < spec.tEnd && T > (warm?.tEnd ?? 0)) snapshotAt.set(Math.round(T / dt), T);
  }
  const snapshots: { tEnd: number; state: Record<string, Float32Array> }[] = [];

  // Everything a cache file needs exists before the run starts, so a snapshot
  // is encoded and uploaded the moment it is captured, overlapping the
  // network with the GPU still stepping, rather than queued for the end.
  const geometryCoeffs = {
    X: session.geometry.X,
    Y: session.geometry.Y,
    Z: session.geometry.Z,
  };
  const encode = (t: number, state: Record<string, Float32Array>) =>
    encodeCacheFile({
      spec: { ...spec, tEnd: t },
      grid: session!.cfg,
      species: model.state,
      geometry: geometryCoeffs,
      initial,
      final: state,
      adapter: adapterName,
    });
  const uploadedTimes: number[] = [];
  const uploadErrors: string[] = [];
  let uploadsStarted = 0;
  const pendingUploads: Promise<void>[] = [];
  /** Encode + upload without the stepping loop waiting. A captured snapshot
   *  is a complete solution of its own spec, so this stays valid even if the
   *  run is stopped afterwards. */
  const uploadInBackground = (
    t: number,
    state: Record<string, Float32Array>,
    apiKey: string,
    preEncoded?: Uint8Array,
  ): void => {
    uploadsStarted++;
    pendingUploads.push(
      (async () => {
        const bytes = preEncoded ?? (await encode(t, state));
        const lookup = await lookupFor({ ...spec, tEnd: t });
        await uploadCacheFile(apiKey, lookup.fileName, bytes);
        uploadedTimes.push(t);
      })().catch((e) => {
        uploadErrors.push(`t = ${fmtChoice(t)}: ${e instanceof Error ? e.message : e}`);
      }),
    );
  };

  shownT = null;
  resetRanges();
  const t0 = performance.now();
  let lastStatus = 0;
  let lastDraw = 0;
  while (session.steps < steps) {
    if (gen !== generation) return;
    if (stopRequested) {
      shownT = session.steps * dt;
      await draw();
      updateStats();
      const up = uploadedTimes.length
        ? ` ${uploadedTimes.length} snapshot${uploadedTimes.length > 1 ? 's' : ''} already uploaded.`
        : ' Nothing uploaded.';
      status(`stopped at t = ${(session.steps * dt).toFixed(2)}.${up}`);
      return;
    }
    // One chunk: up to CHUNK_STEPS steps submitted back to back (each
    // submission stays under the dispatch budget), then a single sync and at
    // most one render. Reading back and drawing after every submission is
    // what made the run advance at a fraction of the solver's rate — a
    // readback costs several times the 3-4 steps it fenced. The chunk stops
    // exactly at snapshot points so those states are still captured exactly.
    let target = Math.min(steps, session.steps + CHUNK_STEPS);
    for (const s of snapshotAt.keys()) {
      if (s > session.steps && s < target) target = s;
    }
    while (session.steps < target) {
      session.step(Math.min(stepsPerSubmit, target - session.steps));
    }
    // The sync bounds how far the CPU runs ahead of the GPU, and (being a
    // promise) yields to the event loop, which is what keeps Stop clickable.
    await session.sync();
    if (gen !== generation) return;
    const hit = snapshotAt.get(session.steps);
    if (hit !== undefined) {
      const state = await session.readState();
      if (gen !== generation) return;
      // With a key on hand the snapshot goes straight to the cache; without
      // one it is kept, in case a key is entered before the run ends.
      const apiKey = elApiKey.value.trim();
      if (apiKey) uploadInBackground(hit, state, apiKey);
      else snapshots.push({ tEnd: hit, state });
    }
    const now = performance.now();
    if (now - lastDraw > RENDER_EVERY_MS || session.steps >= steps) {
      lastDraw = now;
      await draw();
      if (gen !== generation) return;
      await nextFrame();
    }
    if (now - lastStatus > 200) {
      lastStatus = now;
      const t = session.steps * dt;
      const pct = ((100 * (session.steps - startSteps)) / (steps - startSteps)).toFixed(0);
      const rate = (session.steps - startSteps) / ((now - t0) / 1000);
      const from = warm ? `resumed from cached t = ${fmtChoice(warm.tEnd)} — ` : '';
      const up = uploadsStarted
        ? `, uploaded ${uploadedTimes.length}/${uploadsStarted} snapshots`
        : '';
      status(
        `not in the cache — <b>computing locally</b> (${from}` +
          `t = ${t.toFixed(2)} / ${fmtChoice(spec.tEnd)}, ${pct}%, ${rate.toFixed(0)} steps/s${up})`,
      );
    }
  }

  const final = await session.readState();
  if (gen !== generation) return;
  shownT = spec.tEnd;
  await draw();
  updateStats();
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  const doneLine =
    `<b>t = ${fmtChoice(spec.tEnd)}</b> — computed locally in ${secs} s` +
    (warm ? ` (resumed from cached t = ${fmtChoice(warm.tEnd)})` : '') +
    `.`;
  status(`${doneLine} Writing the cache file…`);

  const finalBytes = await encode(spec.tEnd, final);
  if (gen !== generation) return;
  const finalLookup = await lookupFor(spec);
  offerDownload(finalBytes, finalLookup.fileName.split('/').pop()!);

  // The final solution, plus any snapshots captured before a key was entered.
  const apiKey = elApiKey.value.trim();
  if (apiKey) {
    uploadInBackground(spec.tEnd, final, apiKey, finalBytes);
    for (const snap of snapshots) uploadInBackground(snap.tEnd, snap.state, apiKey);
  }
  if (uploadsStarted === 0) {
    status(`${doneLine} Not uploaded (no API key).`);
    return;
  }
  status(`${doneLine} Uploading to the cache (${uploadedTimes.length}/${uploadsStarted})…`);
  await Promise.all(pendingUploads);
  if (gen !== generation) return;

  if (uploadErrors.length) elErr.textContent = `upload: ${uploadErrors.join('; ')}`;
  const n = uploadedTimes.length;
  if (n > 0) {
    const times = [...uploadedTimes].sort((a, b) => a - b).map(fmtChoice).join(', ');
    const failed = uploadErrors.length ? ` (${uploadErrors.length} failed)` : '';
    status(
      `${doneLine} <b>Uploaded ${n} solution${n > 1 ? 's' : ''}</b> ` +
        `to the shared cache (t = ${times})${failed}.`,
    );
  } else {
    status(`${doneLine} Uploads failed.`);
  }
}

// ---------------------------------------------------------------- boot
elSolve.addEventListener('click', () => {
  flowChain = flowChain.then(() => solve()).catch(() => undefined);
});
elStop.addEventListener('click', () => {
  stopRequested = true;
  setBusy(false);
});
elReset.addEventListener('click', () => resetDefaults());
elResetView.addEventListener('click', () => {
  for (const s of scenes) s.resetCamera();
});
elApiKey.addEventListener('change', () => {
  const key = elApiKey.value.trim();
  if (key) localStorage.setItem(API_KEY_STORAGE, key);
  else localStorage.removeItem(API_KEY_STORAGE);
  updateUploadNote();
});

function updateUploadNote(): void {
  elUploadNote.textContent = elApiKey.value.trim()
    ? 'uploads enabled — locally computed solutions will be contributed'
    : '';
}

async function boot(): Promise<void> {
  buildControls();
  // Written even before any change, so the address bar is always shareable.
  writeUrlState();
  elApiKey.value = localStorage.getItem(API_KEY_STORAGE) ?? '';
  updateUploadNote();
  void updateCacheNote();
  try {
    device = await requestShtDevice();
    adapterName = await describeAdapter(device);
  } catch (e) {
    device = null;
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
    await rebuildSession(currentSpec());
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
