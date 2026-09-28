# gpu-hotspots seat notes (2026-09-24/25)

worktree `wt/hotspots` from `blessed` 8349eabc. brief: `.briefs/gpu-hotspots.md`.

## ranked top passes (whiterun, ultra 1080p) -- source: `docs/research/scouting.md`'s table (itself built from `bench/runs/passes-fork-1` / `passes-native-2` via `passdiff.py`)

`passdiff.py` on those two runs prints its own warning that the two captured
files have their `source=` tags swapped; scouting.md's table already
corrects for that (columns are `fork ms` / `native ms`, not passdiff's raw
`native`/`dxvk` headers). Anyone re-running `passdiff.py bench/runs/passes-fork-1
bench/runs/passes-native-2` directly gets native and dxvk flipped in the
column headers -- read `fork ms`/`native ms` from scouting's table, not the
tool's own labels, or re-derive with corrected filenames.

| rank | pass | what | hash(es) | draws | fork ms | native ms | delta |
|---|---|---|---|---|---|---|---|
| 1 | 10 | main lit pass | `b63b3bf7` (+`c4c12180`,`8a126307`,`18a7b10e`) | 1,288 | 0.659 | 0.640 | +3% |
| 2 | 7 | vol. lighting generate, 91 dispatches | `ab674eb1` | 0 | 0.618 | 0.689 | -10% (fork wins) |
| 3 | 5 | sun cascade, far, 4096² d16 | `08e79d9c` | 2,281 | 0.432 | 0.393 | +10% |
| 4 | 4 | sun cascade, near, 4096² d16 | `08e79d9c` | 273 | 0.353 | 0.384 | -8% (fork wins) |
| 5 | 27 | vol. lighting blur | `cf1e1a21` | 0 | 0.261 | 0.253 | +3% |
| 6 | 9 | sun shadow mask | `b070feb5` | 1 | 0.212 | 0.190 | +12% |
| 7 | 8 | depth prepass | (several) | 1,177 | 0.204 | 0.205 | ~0% |
| 8 | 23 | gbuffer-ish, soft particles | `556a3b73` | 96 | 0.176 | 0.094 | **+88%** |
| 9 | 13 | ssr, half-res 960x540 | `88f49b91` | 1 | 0.175 | 0.211 | -17% (fork wins) |
| 10 | 16 | vanilla sao (skipped under rtao) | `83560015` | 1 | 0.148 | 0.135 | +10% |
| 11 | 61 | taa resolve | `72db6ece` | 1 | 0.143 | 0.135 | +6% |
| 12 | 0,1 | water reflection cube, 2 faces | `79c17b07` | 83+228 | 0.191 | 0.283 | -33% (fork wins) |
| 13 | -- | precipitation occlusion | `489864ca` | 126+26 | 0.028 | 0.017 | +65% (tiny abs.) |

speedup categories per pass (brief's menu: faster dxbc-equivalent / merge
tiny passes / skip unused work / dxvk-side fix):

- 1, 3, 4: already at native speed; not worth touching (dxbc-spirv is not a
  lever here, confirmed by scouting's D5).
- 2, 5: **already built** -- `BLESSED_VOL_COLLAPSE` (fork 0c3cc73d) collapses
  the z-integration chain, `BLESSED_VOL_HALFRATE` amortises the generate.
  fork already wins outright on the generate pass even before those switches.
- 6: sun shadow mask is the rtshadow hook point (`docs/research/point-lights.md`);
  not touched here, out of scope for a vanilla-image-preserving perf pass.
- 8: at parity, skip.
- **9 (rank 8, `556a3b73`): the biggest single-shader delta, and this seat's
  main target. see below.**
- 9: fork already faster (screen-space reflections); a null finding, skip.
- 10: vanilla sao is dead under `blessed` (rtao replaces it); a `perf`-only
  target, and its own shader is a real ~30-tap kernel -- flagged for a future
  seat, not attempted here (time budget).
- 11: small delta on a real shader (taa resolve); same as 10, flagged not built.
- 12: fork already faster (water reflections, plus the halfrate lever already
  built on top); skip.
- 13: proportionally huge but 0.01 ms absolute; already gated by
  `SKYBENCH_PRECIP_OCCLUSION`. skip.

## `556a3b73` (rank 8): root-caused with a non-game harness, no shader fix shipped

The soft-particle/decal pass into the gbuffer (96 draws, alpha blend, 3
render targets + read-only depth test) is the single largest per-shader
dxvk-vs-native gap in the frame (+88% per the frame capture). `docs/research/
scouting.md`'s own D3 called this "an afternoon with the frame dump", and a
prior shader-replace seat had already built exactly the right tool for it
and left a "LEAD PRIORITY" note: `dxvk/blessed-shaders/harness/pass37.cpp`,
a **standalone d3d11 microbenchmark that needs no game and no D: write** --
it replays this pass's real shape (96 draws, 3 rtvs, read-only depth, alpha
blend, 4 dynamic cbuffers/draw, the real vanilla `556a3b73` pixel shader
pulled from `Skyrim - Shaders.bsa`) against whatever `d3d11.dll` sits next
to it, native or our fork.

I built it (`cl /O2 /std:c++17 harness/pass37.cpp`, msvc) and ran it on this
machine, native vs `tools/dxvk-fork/x64`'s d3d11.dll:

- native median 0.1996 ms; dxvk (default profile) 0.2140 ms; dxvk with the
  real SkyrimSE app profile forced (`DXVK_CONFIG="d3d11.cachedDynamicResources
  = a"`, since `pass37.exe` doesn't match the app-profile's `\SkyrimSE\.exe$`
  regex on its own) 0.2294 ms.
- flag sweep (`noclear`, `noblend`, `nodepth`, `onert`, `static`) confirms
  the prior seat's finding: MRT + blend residual on 3 rtvs is real (~10 us,
  matches their own micro-bench almost exactly), not explained by clears,
  depth, or static-vs-dynamic cbuffers.
- **new finding**: `d3d11.cachedDynamicResources`'s `c` (constant buffer)
  flag, the one the SkyrimSE app profile turns on for a cpu-side render
  thread fix (dxvk issue #5885, "skyrim maps ~13k cbuffers a frame"), costs
  this one pass its entire game-profile-vs-default delta: `a` (all flags,
  today's profile) measures 0.229 ms; `vir` (same profile minus `c`) measures
  0.215 ms, identical to leaving the option unset. Reproduced twice each way,
  clean and stable (medians within 0.0004 ms across repeats).

what I did *not* ship: a code change to the app profile or a size-gated
override. `docs/backlog.md` already has a directly relevant dead end here
(the cb-ring vram experiment, `bis-rebar`: -30% fps, "no subset helps, the
cost is per map") from a *different* axis (memory placement, not the cached-
vs-write-combined *mapping mode* this pass profile controls) -- but it is
close enough in shape that flipping the global flag blind, without an
in-game cpu+gpu measurement, risks re-introducing the render-thread stall
the flag exists to fix. That in-game A/B needs no code at all:

```
DXVK_CONFIG=d3d11.cachedDynamicResources = vir
```

launched with the fork, vs the current default (`a`, from the app profile).
Env config overrides the app profile in dxvk's own merge order. If cpu frame
time doesn't regress, `vir` is a straight win on this pass and probably
others like it (many small per-draw dynamic cbuffers); if it does, the two
axes need to be separated by bind-flag combination, not by a single switch.
**owed to the lead: an in-game a/b of this exact env var, watching both cpu
and gpu frame time.**

## a per-buffer fix attempted, built, tested, and disproven -- not shipped

I first tried the obvious code version of the `vir` finding: a new opt-in
option, `d3d11.blessedSmallCbUncachedBelow`, gating `D3D11Buffer::GetMemoryFlags()`
to exclude a dynamic constant buffer from `cachedDynamicResources`'s cached
path when its `ByteWidth` is under a threshold, everything else (the app
profile, every other buffer) untouched. Three small upstream edits
(`src/d3d11/d3d11_options.h` +14, `.cpp` +1, `src/d3d11/d3d11_buffer.cpp`
+11), builds clean, zero warnings.

**I do not trust it and did not merge it.** I built the worktree's dxvk,
pointed `pass37.exe` at the fresh d3d11.dll, and re-ran the same a/b:

- `cachedDynamicResources = a` alone: 0.2284-0.2294 ms (3 runs).
- `cachedDynamicResources = a` + `blessedSmallCbUncachedBelow = 4096`
  (covers every one of pass37's 9 cbuffers, confirmed with a temporary
  `Logger::warn` per buffer -- all 9 fire the "uncaching" branch, sizes
  16/48/144/16/16/3840/3840/3840/3840 match the harness exactly): still
  0.2284-0.2294 ms. **No change at all**, three repeats.
- `cachedDynamicResources = vir` (the global, code-free string): 0.2140-0.2150
  ms, the same recovery as before.

So the *global* flag and a scoped fix that (as far as I can tell) puts the
exact same 9 buffers in the exact same memory type produce different
results. Something about `cachedDynamicResources = a` costs this pass ~14 us
that is **not** explained by these particular buffers' own memory
properties -- some other dynamic resource, an allocator/pool-locality effect
that only shows up when the flag is consistent app-wide, or something I
haven't found. I reverted the three files rather than ship a switch that
looks like a fix and, per my own measurement, is not one. This is a real,
reproducible negative result, not a shrug: **the `vir` finding is solid and
worth an in-game a/b; the size-gated version of it is not, until someone
finds why the two diverge.**

Owed to the lead:
- the in-game a/b of `DXVK_CONFIG=d3d11.cachedDynamicResources = vir` above
  is still the actionable item -- it needs no code, just launching with it
  set and watching both cpu (the render-thread cost dxvk issue #5885 exists
  for) and gpu frame time.
- if that a/b is clean, the next step is finding why the per-buffer version
  didn't reproduce it -- worth knowing before anyone tries a scoped fix
  again, on this pass or another.
- the MRT+blend ~10 us residual (rank 8's other component, alongside the
  cbuffer-caching one) is still unexplained; the prior seat parked it too.
