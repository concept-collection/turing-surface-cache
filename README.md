# turing-surface-cache

Reaction-diffusion systems (Turing patterns) on curved closed surfaces,
evaluated at a chosen end time, with the solutions shared between all visitors
through a cloud cache.

This is a trimmed fork of
[turing-surface](https://github.com/concept-collection/turing-surface), which
solves the same systems live and freely tunable. What this app changes is the
contract: every setting is a choice from a short list, so each combination of
choices names exactly one solution. Solutions already in the shared cache load
by themselves as the choices are browsed; a combination nobody has computed
shows empty surfaces, and nothing runs until the user presses **Compute
solution**, which runs the solver locally (in the browser, on WebGPU),
watching the pattern form and stopping at exactly the requested time. Users
who hold an upload API key contribute their locally-computed solutions back,
so the next visitor who asks for the same combination gets it in a second
rather than a minute.

## The discrete parameter space

Three models ship, all in turing-surface's 6-transform flux form —
Schnakenberg (the default), Brusselator, and Allen–Cahn — on three geometries
(sphere, ellipsoid, peanut). The Algorithm-4 reference variant is deliberately
absent: it solves the same equations as Schnakenberg and would only duplicate
cache entries under different hashes. The choices, defined in
[`src/cache/options.ts`](src/cache/options.ts):

| setting | choices |
|---|---|
| Schnakenberg | a: 0.05/**0.1**/0.15/0.2 · b: 0.7/**0.9**/1.1/1.3 · D₁: 1.6e-4/**4e-4**/1e-3 · D₂: 3.2e-3/**8e-3**/2e-2 · dt: 0.02/**0.05**/0.1 |
| Brusselator | A: 2/**3**/4 · B: 7/**9**/11 · D₁: 1.7e-3/**3.33e-3**/6.7e-3 · D₂: 8.3e-3/**1.67e-2**/3.3e-2 · dt: 0.01/**0.02**/0.05 |
| Allen–Cahn | ε²: 5e-4/**1e-3**/2e-3 · dt: 0.01/**0.02**/0.05 |
| geometry | sphere, **ellipsoid** (axes each 0.6/1/1.5), peanut (waist 0.4/0.6/0.8, stretch 0/0.6/1.2) |
| seed | **1**–5 |
| end time | **100**, 200, 400, 800, 1600 |

The model, unlike every other choice, is compiled into the GPU session, so
switching it pays a recompile of a second or two; everything else swaps into
the running session. Allen–Cahn evolves one species, so it shows one panel
where the others show two.

(Defaults in bold.) The numerical-scheme settings are fixed — lmax 63, 8
solve iterations, seed wavelength λ = 0.5 — but are recorded in every cache
key, so offering them as choices later invalidates nothing.

The whole selection is mirrored into the URL fragment, every value written
explicitly, so reloading returns to the same combination and a shared link
opens on the same spec (and, when cached, the same solution) for whoever
follows it. A "Reset to defaults" button puts every choice back.

Every end time is an exact multiple of every dt, so a run to T = 800 passes
exactly through t = 100, 200 and 400. Those intermediate states are captured
as the run passes them and, for an uploading user, encoded and uploaded in
the background while the run continues: one long run populates four cache
entries, and the earlier ones are already shared before the run finishes. The same structure works in the other
direction: the state is Markovian in the spectral coefficients, so before
computing anything the app looks for the longest cached shorter run of the
same spec and continues from its final state, computing only the remainder.
Asking for T = 1600 when T = 800 is cached costs half the run, and the t = 0
initial state travels inside every file of the chain, so a continuation
writes files identical in kind to a from-scratch run.

## How the cache works

The page's whole state is one small spec object (model, parameters, geometry,
seed, end time, scheme settings, plus the app name and a format version). Its
canonical JSON — keys sorted at every level — is hashed with SHA-256, and the
hash is the object name:

```
https://tempory.net/tmpbucket/turing-surface-cache/v1/schnakenberg/<sha256>.h5
```

A lookup is therefore a single GET with no index or API, and a 404 means a
miss. The path carries the app name, format version and model in the clear so
that future cleanup (lifecycle rules, prefix deletes) never has to open a file
to know what it belongs to; the hash input includes the version string, so a
format change moves every object rather than silently colliding with the old
ones.

## Filling the cache

A cache only pays off once it holds what people ask for, and nobody wants to
sit through the first computation of every combination. **Auto-fill the
cache** — offered only when an upload API key is present — turns an otherwise
idle machine into a contributor: it works through the parameter space,
skipping whatever is already cached and computing and uploading the rest, and
runs until stopped.

Two decisions make that practical. The first is the order. About 8,200
combinations exist (228 model-parameter sets × 36 geometries, with the seed
and dt pinned), which is roughly three GPU-weeks at this repo's ~180 steps/s —
exhaustible in principle, but only if the useful part comes first. Since a
visitor starts at the defaults and changes one dropdown at a time, the chance
that a combination is ever requested falls off steeply with the number of
knobs that differ from the defaults, so the walk proceeds by that distance:
every one-knob deviation before any two-knob one. The model is not counted
among those knobs: somebody who came for Allen–Cahn starts at its defaults
rather than at Schnakenberg's, so the three models are three origins rather
than one origin and two deviations from it, and each is surrounded before any
of them is explored far. One machine overnight covers every one- and two-knob
deviation from every model's defaults, which is most of what anyone will ever
click; the long tail can take as long as it likes.

The second is that within a distance the order is **random**, and that is the
entire coordination mechanism. Several idle browsers walking the same tiers in
different orders, each skipping what it finds already cached, rarely duplicate
each other and need no coordinator, no work queue, and no knowledge of one
another. A skip costs one `HEAD` request, so a machine joining a
well-filled region catches up in seconds.

The seed and dt are pinned rather than surveyed (seed 1, dt 0.05): a seed
picks a draw and means nothing on its own, and dt is a numerical knob rather
than a property of the problem, so surveying either would multiply the work
without adding a solution anyone asked for. Both are the default of every
model, so an auto-filled entry is exactly what a visitor arriving at the
defaults requests. Two smaller points: the walk skips the ellipsoid with all
axes 1, since that *is* the unit sphere and the sphere geometry already
covers it, and any run whose state goes non-finite is reported and discarded
rather than uploaded — an unattended walk must not publish wreckage under a
hash someone later trusts.

Because it is meant to run unattended, the compute loop never waits on an
animation frame and skips rendering entirely while the page is hidden, so a
minimized window or a background tab keeps computing at full speed rather
than being throttled to a crawl.

Uploads go through the [tmpbucket](https://github.com/scratchrealm/tmpbucket)
Worker: the client presents the API key and a file name, receives a presigned
R2 PUT URL, and uploads directly. Only holders of the key can write; everyone
can read. The key is entered in the page and kept in localStorage.

## Filling it from the command line

A browser window is a poor place to leave a long computation, so the same walk
runs outside one:

```
TURING_SURFACE_CACHE_KEY=… npx https://concept-collection.github.io/turing-surface-cache/fill.tgz
```

Nothing is published to the npm registry — npm installs a tarball from a URL
as happily as from a package name, and the tarball is built and deployed
beside the page, so the command line is always the same commit as the app.
The page itself offers this command, ready to copy, once an upload key is
entered; the key is masked in what the page shows and real in what it copies,
so that pasting it onto a fresh machine takes one step while a screenshot of
the page still gives nothing away. The key can also be saved for later runs
(`login` prompts for it and
writes `~/.config/turing-surface-cache/key`), or passed as `--key`, though the
environment is preferable: a key on the command line is visible to every user
on the machine through `ps`, while another process's environment is not.

Node 18 or newer is required — the cache keys are SHA-256 through WebCrypto,
which older node does not have as a global, and node 18 itself has it only
under `node:crypto`, which is worth supporting since that is what several
current distributions ship. An older node than that cannot even parse the
bundle, and would otherwise report a syntax error pointing at a brace, so the
published command is a small ES5 launcher that checks the version first and
says what to do about it.

One consequence of installing from a URL is worth knowing. npx keys its
install directory on the whole spec string it was given, so a URL that never
changes keeps running whatever it first installed, however often the file
behind it has been replaced — and neither `--prefer-online` nor a changed
version in the manifest makes any difference, since nothing remote is
consulted once that directory exists. The command the page offers therefore
carries the build it belongs to (`fill.tgz?v=<commit>`), which makes every
deployment a new spec and so a fresh install. The bare URL above is right the
first time and stale ever after; `--help` says which build is running.

The walk, the runs and the uploads are the page's own — the same modules under
[`src/cache/`](src/cache/), driven by console output instead of a status bar
(see [`src/cli/fill.ts`](src/cli/fill.ts)). What differs is the WebGPU: node
has none, so the command line brings its own, the optional `webgpu` package of
prebuilt [Google Dawn](https://dawn.googlesource.com/dawn) binaries, installed
under the globals the transform code expects. Dawn reaches the GPU through
Vulkan on Linux and Windows and Metal on macOS, so a machine wanting to
contribute needs a GPU and its driver — on a machine without one, Dawn
either finds no adapter at all or falls back to a software rasterizer, which
is roughly a thousand times slower and worth nothing to anybody. The command
names its adapter on startup, reports its rate in steps per second, and says
so plainly when either looks wrong; it does not refuse to run, since the
judgment is the operator's.

A machine can also be too old for the command in a way that has nothing to do
with its GPU. Dawn's prebuilt binary wants glibc 2.34, which a long-lived
Linux workstation may well not have — Rocky and RHEL 8 are on 2.28 — and the
obvious remedy of running the command in a container turns out not to work:
inside one the NVIDIA driver declines to bring up its Vulkan driver
(`vk_icdNegotiateLoaderICDInterfaceVersion` returns
`VK_ERROR_INITIALIZATION_FAILED`), while the same call on the host succeeds.
What does work is to borrow only the userland from a container image and run
node through its loader, on the host, leaving the GPU, `/dev` and `/proc`
exactly as they were; the host's own `/usr/lib64` stays last on the library
path, since the NVIDIA libraries and the Vulkan loader have to match the
running kernel module. The page carries that recipe, folded away beside the
command it belongs to, along with what the other common failures mean —
they are worth writing down where someone will meet them, since none of them
is guessable from the error alone.

Progress is a line per target and a rate that updates in place:

```
[2] schnakenberg a=0.15 b=0.9 D1=4e-4 D2=8e-3 dt=0.05 · sphere · 2 knobs from the defaults
      computing to t = 1600 (32,000 steps)
      t = 812.4 / 1600  51%  184 steps/s  eta 1m11s  uploaded 3/3
      computed in 2m54s — uploaded 5 solutions (t = 100, 200, 400, 800, 1600)
```

When the output is not a terminal the same lines are written periodically
instead of in place, so a `nohup`ed log stays readable. `--dry-run` lists the
first targets and whether each is already cached, which is a cheap way to see
what a machine would take on before committing it; `--limit` and `--model`
narrow the work; and ctrl-C stops after the current run, so nothing in flight
is lost.

## Sweeping one parameter

A second page, [`sweep.html`](sweep.html) (linked from the main one), shows
how the solution changes across one parameter rather than at one point: every
setting is fixed except a single model parameter, which runs over a list of
values, and a knob steps the display through the range. On any selection
change the page fetches every value's cache file at once (a sweep is a handful
of files of about 90 KB) and synthesizes each solution onto the render mesh as
it arrives, so that moving the knob afterwards touches neither the network nor
the solver: it recolors the mesh from values already in memory.
One color scale per species, computed over all loaded values and held fixed,
covers the whole sweep, so what changes under the knob is the pattern rather
than the palette. The URL fragment carries the selection, the swept parameter
(`sweep=b`), its values (`values=0.7,0.9,1.1,1.3`) and the knob's position, so
a shared link opens on the same sweep at the same place; the serialization is
the main page's with two entries added (see
[`src/cache/selection.ts`](src/cache/selection.ts)).

The value list starts as the swept parameter's own choices, which is what the
main page's dropdown offers and what the auto-fill walk surveys, but it is a
text box, and any numbers may be typed in its place. This is the one control
in the app that is not a choice from a list. It is nonetheless safe for the
cache, since a typed number is parsed once and serialized in canonical
shortest form ever after, so that it names one exact spec and one exact
object just as a listed value does. Of course, a value the walk never
surveyed will not already be there, so a sweep over typed values arrives
entirely as gaps and has to be computed.

Values nobody has computed show as gaps on the knob's track. **Compute
missing values** runs them in the browser, one after another, each through the
same local run as the main page, warm start and background snapshot uploads
and divergence guard included. The command line fills a sweep on a machine
with no browser on it:

```
TURING_SURFACE_CACHE_KEY=… npx …/fill.tgz sweep '<sweep page URL>'
```

The argument is the sweep page's own URL: its fragment already says which
parameter sweeps, over which values, and what everything else is fixed to, so
the copied command and the page it came from always name the same solutions,
and pasting the same URL into a browser shows the result. The page offers this
command ready to copy once an upload key is entered, the key masked on screen
and real in the clipboard as before. `--dry-run` lists the sweep's values and
whether each is cached. Note that unlike the auto-fill walk, a sweep honors
the selection's seed and end time exactly; a run to the sweep's end time still
uploads every shorter listed end time it passes, so filling a sweep at
T = 1600 also fills the same sweep at every smaller T.

## The cache file

Cache files are HDF5, written in the browser with
[h5wasm](https://github.com/usnistgov/h5wasm) and readable from Python with
h5py. The layout is turing-surface's reference-file layout (see
`docs/ellipsoid-reference-spec.md` there) extended with the cache's identity
at the root, so a cache file is *also* a valid reference file — it can be
loaded straight into turing-surface's "Compare against uploaded data" mode:

```
/            attrs: app, format_version, spec_json, model, species,
                    created_utc, adapter
├─ backend/  attrs: adapter, runtime, precision
├─ spec/     attrs: geometry, lmax, seed, steps, niter, lam3, t_end
│  ├─ params/           attrs: a, b, D1, D2, dt
│  └─ geometry_params/  attrs: the geometry's params
├─ grid/     attrs: lmax, mmax, nlat, nphi, nlm
├─ geometry/ Gx, Gy, Gz                     float32[2·nlm]
├─ initial/  U, V (spectral state at t = 0) float32[2·nlm]
└─ final/    U, V (at the end time)         float32[2·nlm]
```

`spec_json` is the exact string that was hashed into the object name, and the
reader verifies it matches what was asked for. The initial state is included
so a file fully defines its run; the coefficients are the spherical-harmonic
convention documented in turing-surface (orthonormal + Condon-Shortley,
m-major, [re, im] interleaved). At lmax 63 a file is about 90 KB.

Note that the solver is deterministic given the spec only to fp32 round-off:
different GPUs round differently, so a cached solution and a local recompute
agree closely but not bit-for-bit. The cache stores whichever trusted user
computed a combination first, and the file records which adapter that was.

## Development

```
npm install
npm run dev       # local dev server
npm run build     # type-check + production build to dist/
npm run build:cli # the command-line bundle, packed as dist/fill.tgz
```

`npm run build` runs `build:cli` too, so a deployment carries both. The
command-line bundle is an SSR vite build of
[`src/cli/fill.ts`](src/cli/fill.ts) with everything under `src/` and numbl's
compiler bundled in, exactly as the page's build has them; the only things
left external are h5wasm, whose node build reads its wasm off disk, and Dawn,
which is a native addon. [`scripts/pack-cli.mjs`](scripts/pack-cli.mjs) writes
the published manifest, which therefore depends on neither numbl nor a
checkout of anything.

numbl is a local `file:../../numbl` dependency, exactly as in turing-surface —
a sibling checkout of [numbl](https://github.com/flatironinstitute/numbl) is
required, reached through the `numbl-src` alias in
[`vite.config.ts`](vite.config.ts). See turing-surface's README for the
details; nothing about the arrangement changed here.

Checks:

- `node scripts/check-app.mjs` — end-to-end in headless Chrome (SwiftShader
  WebGPU) with the cloud cache mocked: a miss computes locally and produces
  the .h5 (verified with h5py), a fresh page loads that .h5 as a hit, a
  third page asking for a longer end time warm-starts from it, and the sweep
  page loads that .h5 as one value of a dt sweep, shows the other two as
  gaps, and fills them with Compute missing values. The pages use the
  `?tend=` query hook, which substitutes short test end times for the UI's
  list. This is what CI runs.
- `node scripts/check-live.mjs [url]` — smoke-check a deployed URL against
  the real cache, both pages.
- `node scripts/screenshot.mjs out.png [light|dark] [tEnd] [index|sweep]` —
  screenshot after the boot-time solve.

Deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.

## License

CECILL-2.1 (inherited from SHTNS via shtns-webgpu, whose sources are
vendored under `src/sht/`).
