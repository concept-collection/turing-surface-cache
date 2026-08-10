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

One model ships (Schnakenberg, in turing-surface's 6-transform flux form) on
three geometries (sphere, ellipsoid, peanut). The choices, defined in
[`src/cache/options.ts`](src/cache/options.ts):

| setting | choices |
|---|---|
| a | 0.05, **0.1**, 0.15, 0.2 |
| b | 0.7, **0.9**, 1.1, 1.3 |
| D₁ | 1.6e-4, **4e-4**, 1e-3 |
| D₂ | 3.2e-3, **8e-3**, 2e-2 |
| dt | 0.02, **0.05**, 0.1 |
| geometry | sphere, **ellipsoid** (axes each 0.6/1/1.5), peanut (waist 0.4/0.6/0.8, stretch 0/0.6/1.2) |
| seed | **1**–5 |
| end time | **100**, 200, 400, 800, 1600 |

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

Uploads go through the [tmpbucket](https://github.com/scratchrealm/tmpbucket)
Worker: the client presents the API key and a file name, receives a presigned
R2 PUT URL, and uploads directly. Only holders of the key can write; everyone
can read. The key is entered in the page and kept in localStorage.

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
```

numbl is a local `file:../../numbl` dependency, exactly as in turing-surface —
a sibling checkout of [numbl](https://github.com/flatironinstitute/numbl) is
required, reached through the `numbl-src` alias in
[`vite.config.ts`](vite.config.ts). See turing-surface's README for the
details; nothing about the arrangement changed here.

Checks:

- `node scripts/check-app.mjs` — end-to-end in headless Chrome (SwiftShader
  WebGPU) with the cloud cache mocked: a miss computes locally and produces
  the .h5 (verified with h5py), a fresh page loads that .h5 as a hit, and a
  third page asking for a longer end time warm-starts from it. The pages use
  the `?tend=` query hook, which substitutes short test end times for the
  UI's list. This is what CI runs.
- `node scripts/check-live.mjs [url]` — smoke-check a deployed URL against
  the real cache.
- `node scripts/screenshot.mjs out.png [light|dark] [tEnd]` — screenshot
  after the boot-time solve.

Deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.

## License

CECILL-2.1 (inherited from SHTNS via shtns-webgpu, whose sources are
vendored under `src/sht/`).
