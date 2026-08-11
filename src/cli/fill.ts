/**
 * Filling the shared cache from the command line, so that a machine with a GPU
 * and nothing to do can contribute without a browser window open on it.
 *
 * The walk, the runs and the uploads are the page's (src/cache/fillWalk.ts,
 * src/cache/runSpec.ts); what is here is the shell around them — Dawn instead
 * of a browser's WebGPU, a key from the environment instead of localStorage,
 * and lines of text instead of a status bar.
 */
import { requestShtDevice, describeAdapter } from '../sht/sht.ts';
import { mModelByKey } from '../mgpu/registry.ts';
import { formatFailure } from '../mgpu/errors.ts';
import {
  fmtChoice,
  GEOMETRY_CHOICES,
  MODEL_CHOICES,
  T_END_CHOICE,
  type DiscreteChoice,
} from '../cache/options.ts';
import { autoOrder, specForTarget, type AutoTarget } from '../cache/autoWalk.ts';
import { headCached, lookupFor, verifyApiKey } from '../cache/client.ts';
import { SolverSession } from '../cache/solver.ts';
import { fillWalk } from '../cache/fillWalk.ts';
import { stepsFor } from '../cache/spec.ts';
import type { RunSummary } from '../cache/runSpec.ts';
import { installWebGpu, errMsg, isSoftwareAdapter, NO_ADAPTER_HINT } from './webgpu.ts';
import { KEY_ENV, keyPath, maskKey, promptSecret, resolveKey, saveKey } from './key.ts';

const HELP = `turing-surface-cache — fill the shared cache of Turing patterns

Usage
  fill [options]        work through the parameter space, contributing what is
                        missing, until stopped (ctrl-C)
  fill --dry-run [N]    show the first N targets and whether they are cached
  login                 save an upload key for later runs
  --help

Options
  --key <key>       upload key; otherwise $${KEY_ENV}, otherwise the saved key
  --limit <n>       stop after n solutions have been computed
  --model <key>     only targets of one model (schnakenberg, brusselator,
                    allencahn)
  --tend <list>     replace the end-time list, e.g. --tend 5,10 (for testing:
                    a short run hashes to its own honest cache entry)

An upload key is required: the walk exists to contribute. Solutions are read
by everyone and written only by key holders.`;

interface Options {
  command: 'fill' | 'login' | 'help';
  key?: string;
  limit: number;
  model?: string;
  dryRun: number;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = { command: 'fill', limit: Infinity, dryRun: 0 };
  const rest = [...argv];
  if (rest[0] === 'fill' || rest[0] === 'login') opts.command = rest.shift() as 'fill' | 'login';
  /** A count option whose value may be left off (--dry-run, --dry-run 40). */
  const count = (fallback: number): number => {
    const next = rest[0];
    if (next && /^\d+$/.test(next)) return Number(rest.shift());
    return fallback;
  };
  while (rest.length) {
    const arg = rest.shift()!;
    if (arg === '--help' || arg === '-h') opts.command = 'help';
    else if (arg === '--key') opts.key = rest.shift();
    else if (arg === '--limit') opts.limit = Number(rest.shift());
    else if (arg === '--model') opts.model = rest.shift();
    else if (arg === '--dry-run') opts.dryRun = count(20);
    else if (arg === '--tend') setEndTimes(rest.shift());
    else throw new Error(`unknown option ${arg}`);
  }
  if (opts.model && !mModelByKey(opts.model)) throw new Error(`unknown model ${opts.model}`);
  if (!(opts.limit > 0)) throw new Error('--limit wants a positive number');
  return opts;
}

/** The page's ?tend hook, spelled as an option (src/main.ts). */
function setEndTimes(list: string | undefined): void {
  const values = (list ?? '')
    .split(',')
    .map(Number)
    .filter((v) => Number.isFinite(v) && v > 0);
  if (!values.length) throw new Error('--tend wants a comma-separated list of end times');
  T_END_CHOICE.values = values;
  T_END_CHOICE.value = values[0];
}

// ---------------------------------------------------------------- output
const tty = process.stdout.isTTY === true;
/** Written in place on a terminal, and only every so often when piped, so a
 *  log file does not fill with progress. */
const LOG_EVERY_MS = 30_000;
let liveLine = false;

function say(line = ''): void {
  if (liveLine) {
    process.stdout.write('\n');
    liveLine = false;
  }
  process.stdout.write(`${line}\n`);
}

/** One line that keeps being rewritten while a run advances. */
function live(line: string): void {
  if (!tty) return;
  process.stdout.write(`\r${line.padEnd(78).slice(0, 78)}`);
  liveLine = true;
}

const plural = (n: number, word: string): string => `${n} ${word}${n === 1 ? '' : 's'}`;

function duration(seconds: number): string {
  if (!Number.isFinite(seconds)) return '?';
  if (seconds < 90) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  return m < 90 ? `${m}m${String(Math.round(seconds - 60 * m)).padStart(2, '0')}s` : `${(m / 60).toFixed(1)}h`;
}

/** Parameters in the order the app lists them, not the order they were built. */
const paramList = (params: Record<string, number>, choices: DiscreteChoice[]): string =>
  choices
    .filter((c) => c.key in params)
    .map((c) => `${c.key}=${fmtChoice(params[c.key])}`)
    .join(' ');

/** What a target is, in one line. */
function describe(target: AutoTarget): string {
  const geomChoices = GEOMETRY_CHOICES[target.geometry] ?? [];
  const geom = geomChoices.length
    ? `${target.geometry} ${paramList(target.geometryParams, geomChoices)}`
    : target.geometry;
  return (
    `${target.model} ${paramList(target.params, MODEL_CHOICES[target.model])} · ${geom} · ` +
    `${plural(target.distance, 'knob')} from the defaults`
  );
}

const doneLine = (run: RunSummary): string =>
  `computed in ${duration(run.seconds)}` +
  (run.warmFrom !== null ? ` (resumed from cached t = ${fmtChoice(run.warmFrom)})` : '');

// ---------------------------------------------------------------- commands
async function login(): Promise<void> {
  const key = await promptSecret('upload API key: ');
  if (!key) throw new Error('nothing entered');
  let ok: boolean;
  try {
    ok = await verifyApiKey(key);
  } catch (e) {
    say(`could not reach the upload service to check the key (${errMsg(e)}) — saving anyway.`);
    ok = true;
  }
  if (!ok) throw new Error('that key is not allowed to upload — nothing saved');
  say(`key saved to ${await saveKey(key)}`);
}

async function dryRun(opts: Options, targets: AutoTarget[]): Promise<void> {
  const shown = targets.slice(0, opts.dryRun);
  say(`the first ${plural(shown.length, 'target')} of ${targets.length.toLocaleString()}, ` +
    `nearest the defaults first:`);
  say();
  let cached = 0;
  // A handful at a time: a HEAD apiece, and the answers are wanted in order.
  const width = String(shown.length).length;
  for (let i = 0; i < shown.length; i += 8) {
    const batch = shown.slice(i, i + 8);
    const present = await Promise.all(
      batch.map(async (t) => (await headCached(await lookupFor(specForTarget(t)))) === true),
    );
    present.forEach((isThere, k) => {
      if (isThere) cached++;
      say(
        `  [${String(i + k + 1).padStart(width)}] ${isThere ? 'cached ' : 'missing'}  ` +
          describe(batch[k]),
      );
    });
  }
  say();
  say(`${cached} of ${shown.length} already cached; the walk would compute the other ` +
    `${shown.length - cached}.`);
}

async function fill(opts: Options, targets: AutoTarget[], apiKey: string): Promise<void> {
  const runtime = await installWebGpu();
  const device = await requestShtDevice().catch((e: unknown) => {
    throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  });
  const adapter = await describeAdapter(device);
  say(`${runtime} · ${adapter}`);
  say(`uploads enabled (key ${maskKey(apiKey)})`);
  if (isSoftwareAdapter(adapter)) {
    say('');
    say(`WARNING: ${adapter} is a software rasterizer, not a GPU. Runs here are`);
    say('  perhaps a thousand times slower than on hardware — fast enough to look');
    say('  like it is working, slow enough to be worth nothing. Check that the');
    say('  machine has a GPU and its driver, or stop now.');
  }
  say('');
  say(`${targets.length.toLocaleString()} targets, nearest the defaults first; ` +
    `ctrl-C stops after the current run.`);
  say('');

  const solver = new SolverSession(device, 1, {
    onCompiling: (m) => say(`  compiling ${m.label}…`),
  });

  let stopping = false;
  process.on('SIGINT', () => {
    if (stopping) process.exit(130);
    stopping = true;
    say('');
    say('stopping after this run — ctrl-C again to give up on it.');
  });

  let index = 0;
  let computed = 0;
  let uploads = 0;
  let slowNoted = false;
  let lastLog = 0;
  const counts = await fillWalk({
    targets,
    solver,
    adapter,
    apiKey: () => apiKey,
    beforeTarget: (target) => {
      index++;
      return specForTarget(target);
    },
    events: {
      onTarget: (target) => say(`[${index}] ${describe(target)}`),
      onCached: () => say('      already cached'),
      onComputing: (_target, spec) =>
        say(`      computing to t = ${fmtChoice(spec.tEnd)} ` +
          `(${stepsFor(spec).toLocaleString()} steps)`),
      onPhase: (phase) => {
        if (phase.kind === 'warm-search') say('      looking for a shorter cached run…');
        else if (phase.kind === 'seeding') say('      seeding…');
        else if (phase.kind === 'uploading') {
          say(`      ${doneLine(phase.run)} — uploading ${plural(phase.started, 'file')}…`);
        }
      },
      onProgress: (p) => {
        const eta = p.rate > 0 ? (p.totalSteps - p.steps) / p.rate : Infinity;
        const line =
          `      t = ${p.t.toFixed(1)} / ${fmtChoice(p.tEnd)}  ` +
          `${(100 * p.fraction).toFixed(0)}%  ${p.rate.toFixed(0)} steps/s  ` +
          `eta ${duration(eta)}` +
          (p.uploadsStarted ? `  uploaded ${p.uploadsDone}/${p.uploadsStarted}` : '');
        live(line);
        if (!tty && performance.now() - lastLog > LOG_EVERY_MS) {
          lastLog = performance.now();
          say(line);
        }
        // A rate this low means the run is on a software rasterizer, or on a
        // GPU so busy it may as well be. Said once, not every chunk.
        if (!slowNoted && p.rate > 0 && p.rate < 20 && p.steps > 500) {
          slowNoted = true;
          say(`      NOTE: ${p.rate.toFixed(1)} steps/s is far below what a GPU does ` +
            `(${adapter}).`);
        }
      },
      onUploaded: () => uploads++,
      onOutcome: (_target, _spec, outcome) => {
        if (outcome.kind === 'done') {
          computed++;
          const times = [...outcome.uploaded].sort((a, b) => a - b).map(fmtChoice).join(', ');
          say(`      ${doneLine(outcome)} — ` +
            (outcome.uploaded.length
              ? `uploaded ${plural(outcome.uploaded.length, 'solution')} (t = ${times})`
              : 'nothing uploaded'));
          for (const err of outcome.uploadErrors) say(`      upload failed: ${err}`);
        } else if (outcome.kind === 'diverged') {
          say(`      diverged at t = ${outcome.t.toFixed(2)} — discarded, nothing uploaded`);
        } else if (outcome.kind === 'stopped') {
          say(`      stopped at t = ${outcome.t.toFixed(2)}`);
        }
        if (computed >= opts.limit) stopping = true;
      },
      onFailure: (_target, spec, e) => {
        const model = mModelByKey(spec.model);
        say(`      failed: ${formatFailure(e, model?.source ?? '')}`);
      },
      walkStopped: () => stopping,
      stopRequested: () => stopping,
    },
  });

  say('');
  say(`stopped — computed ${counts.computed}, skipped ${counts.skipped} already cached` +
    (counts.failed ? `, ${counts.failed} failed` : '') +
    `; ${plural(uploads, 'file')} uploaded.`);
  solver.destroy();
  device.destroy();
}

// ---------------------------------------------------------------- main
async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.command === 'help') {
    say(HELP);
    return;
  }
  if (opts.command === 'login') {
    await login();
    return;
  }
  const targets = autoOrder().filter((t) => !opts.model || t.model === opts.model);
  say('turing-surface-cache — the shared cache of Turing patterns on curved surfaces');
  if (opts.dryRun) {
    await dryRun(opts, targets);
    return;
  }
  const apiKey = await resolveKey(opts.key);
  if (!apiKey) {
    throw new Error(
      'no upload key. The walk contributes solutions, so it needs one:\n' +
        `  ${KEY_ENV}=… npx <this command>\n` +
        `or save one for later runs with \`login\` (kept in ${keyPath()}).`,
    );
  }
  await fill(opts, targets, apiKey);
}

main().catch((e: unknown) => {
  say('');
  say(errMsg(e));
  process.exitCode = 1;
});
