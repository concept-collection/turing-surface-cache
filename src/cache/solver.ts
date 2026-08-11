/**
 * A compiled solver session together with the selection it currently has
 * applied.
 *
 * Which changes are cheap and which are not is a property of the solver, not
 * of any front end: parameters are a uniform upload, a geometry change
 * re-evaluates the surface, and a model change recompiles everything, since
 * the model's step is compiled into the GPU pipelines. Both the page and the
 * command line need that distinction — a walk through the parameter space
 * spends its whole time on the cheap side of it — so it lives here rather
 * than in either of them.
 */
import { ModelSession } from '../mgpu/session.ts';
import { mModelByKey, type MModel, type Params } from '../mgpu/registry.ts';
import { mGeometryByKey } from '../geom/registry.ts';
import type { CacheSpec } from './spec.ts';

/** Cap on GPU dispatches per submission (watchdog safety; see turing-surface). */
const DISPATCH_BUDGET = 1000;

export interface SolverEvents {
  /** A model change costs a recompile — a second or two on a real GPU. */
  onCompiling?(model: MModel): void;
  /**
   * The surface has changed (a new session, or a new geometry in the running
   * one), so anything drawing it must be rebuilt. Awaited, so a caller that
   * rebuilds a mesh finishes before the session is used.
   */
  onSurface?(): Promise<void> | void;
}

export class SolverSession {
  session: ModelSession | null = null;
  /** The model the session is compiled for. */
  model: MModel;
  /**
   * Steps per GPU submission, sized on every compile so one submission stays
   * under the dispatch budget however expensive niter has made a step.
   */
  stepsPerSubmit = 4;

  #modelKey = '';
  #geomKey = '';
  #geomParams: Params = {};

  constructor(
    readonly device: GPUDevice,
    /** Render grid fineness; 1 (the default) allocates no display plan. */
    readonly oversample = 1,
    private readonly events: SolverEvents = {},
  ) {
    this.model = mModelByKey('schnakenberg')!;
  }

  /** The session, or a thrown error rather than a silent no-op. */
  get live(): ModelSession {
    if (!this.session) throw new Error('no solver session');
    return this.session;
  }

  /**
   * Bring the session in line with a spec, doing the least work that will do:
   * a uniform upload for parameters, a surface re-evaluation for a geometry
   * change, a full recompile for a model change.
   */
  async apply(spec: CacheSpec): Promise<void> {
    if (!this.session || spec.model !== this.#modelKey) {
      await this.#rebuild(spec);
      return;
    }
    this.session.setParams(spec.params);
    const geomChanged =
      spec.geometry !== this.#geomKey ||
      JSON.stringify(spec.geometryParams) !== JSON.stringify(this.#geomParams);
    if (!geomChanged) return;
    await this.session.setGeometry(mGeometryByKey(spec.geometry)!, spec.geometryParams);
    this.#geomKey = spec.geometry;
    this.#geomParams = { ...spec.geometryParams };
    await this.events.onSurface?.();
  }

  async #rebuild(spec: CacheSpec): Promise<void> {
    const nextModel = mModelByKey(spec.model)!;
    this.session?.destroy();
    this.session = null;
    this.#modelKey = '';
    this.events.onCompiling?.(nextModel);
    this.session = await ModelSession.create({
      device: this.device,
      model: nextModel,
      params: spec.params,
      lmax: spec.lmax,
      oversample: this.oversample,
      geometry: mGeometryByKey(spec.geometry)!,
      geometryParams: spec.geometryParams,
      niter: spec.niter,
      lam3: spec.lam3,
    });
    this.model = nextModel;
    this.#modelKey = spec.model;
    this.#geomKey = spec.geometry;
    this.#geomParams = { ...spec.geometryParams };
    const opsPerStep = Math.max(1, this.session.describe().step.length);
    this.stepsPerSubmit = Math.max(1, Math.floor(DISPATCH_BUDGET / opsPerStep));
    await this.events.onSurface?.();
  }

  destroy(): void {
    this.session?.destroy();
    this.session = null;
    this.#modelKey = '';
  }
}
