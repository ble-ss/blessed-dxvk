# cached static sun cascades: seat notes (2026-09-23)

brief: `E:\blessed_skyrim\.briefs\perf-cached-cascades.md`. fork worktree `wt/cached-cascades`, branch `cached-cascades` (off `blessed` `b63e1067`). no plugin code: see "where" below.

## facts from the whiterun dump (`bench/runs/whiterun-dump-1/probe/frame-1920.jsonl`)

- the sun cascades are one `r16_typeless` 4096x4096 texture array with 2 layers (res `0x28013c20270`), d16 dsv, full viewport. layer 0 (near): clear at op 1157, 194 draws. layer 1 (far): clear at op 1645, 1,642 draws. the mask draw reads the whole array at ps srv 4. srv 6 is a different 4096x4096 x8 array (spot/other shadows, sub-viewports), not ours.
- b12 (per-frame) is mapped after the clear and before the first draw of each cascade: the mode is picked at the first draw.
- static and moving casters come in runs: 6 static/moving transitions per cascade (near: D9 S116 D5 S25 D4 S34 D1; far: D50 S765 D85 S157 D38 S542 D5). the build frame's target switches cost 7 render passes per cascade, not one per draw.
- static casters: utility vs with `b2` PerGeometry (c0 ShadowFadeParam, c1-c4 World, c5 EyePos, c6 WaterParams, c7 TreeParams). World's translation is relative to the cascade camera (about -14,000 units along the sun), not to the player camera.
- skinned casters (`aa14ebfd`, `048d5004`, `cda5b1b6`) have no `b2`; bones in `b10`.
- `vs b7` is bound on every cascade draw and is a buffer mapped ~400 times a frame. Utility.hlsl never declares b7, so it is a stale binding: the key hashes only what dxvk's binding masks say each shader reads.
- `TREE_ANIM` (vc + normals) moves vertices by `TreeParams` in the vs: a tree whose TreeParams change never matches its old key, so it is redrawn, never cached.

## facts from the exe (SkyrimSE 1.7.104, read with capstone, `SetFrameCamera` id 108496 rva `0x157f080`)

- the cascade cameras are texel-snapped: at `+0x1545`..`+0x17dc` the main camera's world position (`NiAVObject` world translate, `+0xa0`) is projected on the three light axes, scaled by `1 / cascade+0xcc`, floored (the msvc `cvttss2si` + sign fix-up) and scaled back, then written as the cascade camera's local translate (`+0x6c`). so the projection only changes when the camera crosses a texel-sized step, or when the sun moves (once a second, `fSunShadowUpdateTime`). camera turns do not move it.
- the caster cull planes are built per cascade from the view frustum's split corners (`BuildCascadeCameraCullingPlanes`, id 108499, the call the plugin already thunks): the caster set follows the view direction even while the projection stays. a camera turn can drop a cached caster (the missing-key rule below).

## design (fork only)

- learn: the sun array is the d16 2-4 layer array (>= 1024, square) that a draw with a colour target samples at ps srv 4. largest wins.
- per layer, at the first draw after the engine's clear, from the cascade's projection (vs b12 CameraViewProj + CameraPosAdjust, or all of b12 with `BLESSED_CASCADE_CACHE_PROJ=full`):
  - reuse (cache valid, same projection, nothing pending): copy the cached depth over the layer (`copyImage`), skip every draw whose key is cached (counted per key), draw the rest on top.
  - build (projection stable for `STABLE` frames, cache invalid or a rebuild pending): statics go into the cache image (the context's render target is swapped for it, runs of statics share one pass), moving casters into the layer, then a fullscreen min-composite (LESS_OR_EQUAL, gl_FragDepth from the cache) merges them. the frame is exact.
  - pass: vanilla, while the projection is still moving.
  - verify (every `VERIFY` reuse frames): cached keys are redrawn into a scratch image, everything else into the layer, the scratch is composited in (exact frame) and a compute pass counts texels where scratch != cache.
- static = DrawIndexed, vs reads b2 and not b9/b10, no vs textures/uavs, no hs/ds/gs, only per-vertex slot-0 input, non-dynamic vb and ib, depth write with LESS/LESS_EQUAL, no stencil, one full viewport, no uavs.
- key = vs, ps, layout, rs, dss, ib/vb buffers+offsets+format+stride, draw range, topology, scissor (if on), the bytes of every cbuffer the vs and ps read except b12, the ps's srvs and samplers. identity = the mesh alone (no cbuffers, no ps).
- why exact: depth testing is order-independent, so min(cached statics, this frame's other draws) is what one target with every draw holds, as long as every cached key is drawn again.

## invalidation

- projection changed: cache dropped, pass until stable for `STABLE` frames (default 2), then build.
- missing key (a cached draw not drawn again): its old depth shows for this one frame; rebuild next frame. `TOLERATE=N` keeps reusing for N frames of missing keys (camera turns cull casters out of the cascade; their depth then sits where no visible receiver samples it).
- moved: a cached mesh (identity) drawn with a new key while a key went missing. rebuild next frame, and that mesh is drawn as a moving caster until the sum of its keys holds still for `HOT` frames (default 120).
- promote: a new static key drawn uncached for `PROMOTE` consecutive frames (default 30) triggers a rebuild that absorbs it.
- swap chain resize, a bigger array, or the array not cleared for 300 frames: forget everything.

## cost model

- reuse, per layer: one 4096x4096 d16 copy (32 MB read + 32 MB write; ~0.15 ms at the 3060 ti's 448 GB/s if nothing is compressed) plus the moving and new casters. app thread: one key per static draw (hash of ~150-400 bytes + a map lookup, estimated 50-100 ns). saved: the static draws' dxvk emission (app thread) and cs-thread work, and their raster.
- build, per layer: vanilla + one 4096x4096 fullscreen depth pass (~0.15-0.25 ms) + 6 extra render pass switches + the keys.
- pass: vanilla + one b12 read.

## expected (whiterun, static camera, ultra 1080p)

- vanilla cascades under dxvk: 0.384 + 0.393 ms gpu (passes 4, 5 in the scouting table). reuse: ~0.30 ms of copies + moving casters (~10-20% of the raster) ~0.1 ms, so ~0.35 ms gpu saved; more if the copy runs compressed.
- cpu: ~1,640 draws a frame skipped in dxvk (app + cs thread); the engine's own walk stays (~0.46 us per caster, ~0.75 ms): that is the plugin stage below.

## risks

- a cached caster that moves or vanishes shows its old shadow for one frame (strict mode); then moving meshes go hot.
- camera turns in strict mode: casters culled out of the cascade count as missing, so rebuilds get frequent while turning (each costs ~0.2 ms over vanilla). `TOLERATE` trades that for up to N frames of stale depth where no receiver looks.
- identical meshes (many barrels) share an identity: one moving barrel makes the family hot for 120 frames (correct, less cached).
- `PROJ=vp` assumes the shadow vs reads only CameraViewProj and CameraPosAdjust from b12 (true for Utility.hlsl). `PROJ=full` is the strict fallback if verify ever shows a difference.
- the copy's real cost is unmeasured; if it is near 0.15 ms per layer the next lever is a dirty-rect restore (copy back only last frame's moving-caster rects, bounded from the bones in b10).

## env

- `BLESSED_CASCADE_CACHE=1` on (off by default; ignored when `BLESSED_SKIP_CASCADES=1`).
- `_DEBUG=redrawn` (reuse without the copy: only redrawn casters shadow), `cached` (only cached casters shadow), `build` (rebuild every frame: proves the composite path), `pass` (never cache: prices the hooks).
- `_VERIFY=N` every N-th reuse frame per layer runs verify; results in `cascade_cache.jsonl` (`verify` array) and a log warning on any difference.
- `_STABLE=2`, `_PROMOTE=30`, `_TOLERATE=0`, `_HOT=120`, `_PROJ=vp|full`, `_SLICES=0,1`, `_SRV=4`, `_MIN_SIZE=1024`, `_SKIP_VS=hash,hash` (never cache these vs).
- stats: `<BLESSED_PROBE_DIR>/cascade_cache.jsonl` every 120 presents: per layer frames by mode, cached draws, skipped/redrawn per frame, missing keys, hot meshes, rebuild reasons.

## where: fork, not plugin (the decision)

- only the fork can restore depth, and its key is built from the bytes the gpu actually reads, so "cached" means exactly "would draw the same depth". the plugin would need its own d3d11 copies plus a caster classification that must stay a subset of what the fork cached, or shadows vanish.
- the critical-goal case is gpu-bound (perf ~5.2 ms gpu at 190 fps), so the gpu half comes first.
- stage 2 (plugin, `SKYBENCH_CASCADE_CACHE=1`, not built): in the mode-14 handler `cascade_cull.cpp` already swaps, skip queueing a sun-cascade caster when the fork reports that layer reused and the caster's (vb, ib, World) is cached (a d3d11.dll export the plugin looks up; `BSGeometry` renderer data gives the buffers). the fork then counts plugin-skipped keys as seen. saves the engine's walk (~0.75 ms). optionally widen the cull planes to the cascade's full box on build frames so camera turns stop dropping cached casters.

## dump check of the static rule

- 1,639 of the 1,836 cascade draws in the whiterun dump pass the structural rule (b2 bound, default-usage vb/ib, dynamic b2 readable from the cpu). the other 197 are the skinned draws.

## how the lead tests it

1. build the branch (msvc or mingw), deploy as usual. env: `BLESSED_CASCADE_CACHE=1 BLESSED_PROBE_DIR=<run>/probe`.
2. `cascade_cache.jsonl`: `learned` 1, per layer mostly `reuse`, `cached_draws` near 1,640 (far) and 175 (near), `skipped_per_frame` near those. a `build` about once a second (the sun step) is expected.
3. proof of exactness: add `BLESSED_CASCADE_CACHE_VERIFY=30`. every `verify` entry must read `differ` 0. a `ghost` count means a cached caster went missing (the one stale frame); `missing` means the cache lacks a caster the frame drew with a cached key (a key bug).
4. image: the usual mask comparison against vanilla on a reuse frame.
5. debug views: `BLESSED_CASCADE_CACHE_DEBUG=redrawn` (only moving and new casters shadow), `=cached` (only cached casters shadow), `=build` (rebuild every frame; must look like vanilla).
6. gpu: `BLESSED_GPU_PASSES=1`, cascade passes 4/5 before and after; the restore copy shows up as a transfer event.

## first in-game runs (2026-09-23 evening), the fixes

- `lever-verify-1`: every frame "pass" with no build ever. cause: that run also had `SKYBENCH_CULL_CASCADES=1`, so the engine queues no sun casters and the cascades get only their clears (draws 2,578 a frame vs 5,283 in `bis-cascache`, probe-frames.jsonl). no draw ever reaches the cache, so each cascade closes as "pass". nothing to fix in the fork; the two cannot be combined. without traced shadows that run should also have lost the sun shadows (worth a look at its screenshot).
- `bis-cascache` (cache alone on perf): builds and reuses happen (about 5 builds, 25 pass, 90 reuse per 120 frames per layer), but `cached_draws` 0 and every cascade draw counted as moving. cause: `StaticKey` rejected draws when `state.uav.maxCount` was non-zero. that is the compute stage's uav count, and `CSSetUnorderedAccessViews` only ever grows it (`d3d11_context.cpp`, `std::clamp(StartSlot + NumUAVs, maxCount, ...)`). vanilla's volumetrics bind compute uavs every frame (the whiterun dump: 98 dispatches with `cs_uav`), so from the first frame on every draw was rejected. fix: only graphics uavs bound in `om.uavs[minUav, maxUav)` reject a draw; the shaders' own uav masks were already checked.
- the projection changes about 5 times per 120 frames on a still camera, in bursts (a window can have 66 pass frames and 3 builds), on both layers at once. not explained yet. a shared cause (sun direction, or the camera itself moving slightly) fits better than a per-cascade snap; per-frame TAA jitter does not fit (most frames would never match). the new `cascade_cache_proj.jsonl` answers it on the next run.
- new diagnostics: per layer `draws_per_frame`, `proj_changes`, `not_static_per_frame` (why draws were not cached, by reason: kind, layout, stages, streams, buffers, vs_bindings, uav, depth_state, viewport, skip_vs, cb_read), `proj_read_failed`. `cascade_cache_proj.jsonl`: for the first 64 projection changes per layer and every 16th after, which compared floats changed with old and new values (floats 0-15 CameraViewProj row-major, 16-19 CameraPosAdjust).

## c26-casc-verify-r1: why nothing was cached, and the tolerance (2026-09-23 night)

data (`bench/runs/c26-casc-verify-r1/probe`): verify exact (6 entries, differ 0) but `cached_draws` 0; not-static only "layout" (the skinned draws); `hot_meshes` 173 near / 706 far from the first window to the last.

- **hot meshes: confirmed, a self-sustaining loop.** the first window has `moved` 3 (near) / 13 (far) and 16,387 missing keys (far); every later window has `moved` 0 and `missing_keys` 0, yet `hot_meshes` never drops. a hot mesh stays hot while the keys of its draws change between cascades, and the key held World's translation, which is relative to CameraPosAdjust. CameraPosAdjust changed with the projection (float 16-18 in every proj log entry, ~0.5-1.3 units per change), and the projection changed every 1-21 frames, always inside the 120-frame hot window. so no mesh ever cooled down, and hot meshes never enter the cache. the one-off trigger at load is still unexplained (probably casters fading in after load, ShadowFadeParam in b2 c0); the new `moved` lines in `cascade_cache_proj.jsonl` name the changed b2 floats.
- **fix: the key no longer holds the translation.** b2 floats 7, 11, 15 (World c1.w, c2.w, c3.w in Utility.hlsl's PerGeometry) are zeroed before hashing; each cached draw keeps its world position (translation + CameraPosAdjust, in double) and a later draw matches it within `BLESSED_CASCADE_CACHE_POS_TOL` world units (default 0.05; the engine's float subtraction loses at most ~0.002 at these distances). the hot and promote bookkeeping use key + world position to a quarter unit.
- **the sun drifts, confirmed.** the projection changes 64 times in 444 frames (median 1 frame apart, max 21), and each change moves all of floats 0-11 (the full 3x4, rotation included) plus CameraPosAdjust. over those 444 frames a corner of the cascade box moves ~6 texels in both layers (about 2.4 texels a second), so the sun rotates continuously, not in 1 s steps. exact reuse is only possible between two changes.
- **fix: a bounded tolerance.** `ProjDistance` takes the 8 corners of the cached projection's clip box, back to world space with the cached projection and CameraPosAdjust, forward with this frame's, and returns the largest xy move (texels) and depth move (d16 steps). both maps are affine, so the maximum over the whole cascade volume sits at a corner: this is a bound for every texel of the cache. the cache stays valid while that bound is at most `BLESSED_CASCADE_CACHE_TOL_TEXELS` (default 0.5) and `BLESSED_CASCADE_CACHE_TOL_DEPTH` (default: half the smallest constant DepthBias the cached casters were drawn with, at least 1 step; a d16 DepthBias is in d16 steps). `TOL_TEXELS=0` is the old exact mode.
- **what exactness means now:** cached static depth is the depth of the cached projection. against this frame's vanilla it is off by at most the tolerance: no cached texel is more than 0.5 texel sideways or half the casters' own bias in depth from where vanilla draws it. moving casters are drawn with this frame's projection. the error is largest at the cascade box corners and zero at the rotation centre; at the rebuild the static shadows catch up by at most 0.5 texel.
- **what it saves, from the logged projections** (offline replay of the first 64 changes, corner bound, 6-digit log precision): exact mode rebuilds on all 64 changes in 444 frames; tol 0.5 texel rebuilds 24 times (near) and 11 times (far); tol 1.0 texel and 16 steps rebuilds 15 and 5.
- **verify under a tolerance:** the scratch is drawn with this frame's projection and the cache with the older one, so `differ` is no longer 0. each verify entry now carries `proj_err_texels` / `proj_err_depth` for its frame; `max_diff` should stay near `proj_err_depth`, plus edge texels.
- new stats per layer: `tol_texels`, `tol_depth`, `cached_depth_bias`, `reuse_err_texels_max`, `reuse_err_depth_max`.

## c28: what kept every mesh hot, the skipped restore, the restore's cost (2026-09-24)

- **the moved lines** (`c28-casc-verify-r1..r3/probe/cascade_cache_proj.jsonl`, 288-295 each) all fall in one 30-frame burst right after the first build (r2: frames 17157-17186), then none: after that every static mesh is hot, and hot draws skip the lookup that logs. most lines compare different placements of one mesh (the per-mesh b2 copy is the first placement's), but the clean ones change only register c7: `[28, 4.15335 -> 4.30201], [30, 0.7658 -> 0.7920]`, `[28, 23.0697 -> 23.2184]`, ... c7 is Utility.hlsl's TreeParams; its x is a timer that advances ~0.149 per build-to-now span on every static caster (the whiterun dump shows TreeParams on the position-only vs `138834da` too). every frame every static key changed, so every mesh went hot at load and re-heated at every cascade.
- **fix:** a key now hashes only the part of each cbuffer the shader declares (`dcl_constantbuffer cbN[size]`, read at shader creation into `D3D11CommonShader::BlessedCbvSize`; dynamically indexed buffers still hash whole). a shadow vs that does not animate wind never declares c7, so TreeParams leaves its key; the wind-animated variants declare it and stay moving, which is right. `cascade_cache_proj.jsonl` now lists each static-caster vs once with its declared sizes (`cb_vec4s`), so the next run shows how far each declares b2.
- **no restore when nothing is cached:** a reuse frame with `cached_draws` 0 skips the copy (`restores_skipped` counts them).
- **the restore's own gpu cost, from c28** (gpu busy mean, 3 repeats each): perf-max 4.78 ms; + cache at 0.5 texel 5.12 ms (+0.34); + cache exact 5.06 ms (+0.28). in the tolerant runs nothing was cached, so all of it is overhead: per 120 frames and layer ~105 reuse frames (one full-layer copy each) and ~6 builds (one full-layer composite each), about 1.85 full-layer passes a frame, so **~0.18 ms per 4096x4096 d16 layer restore**, ~0.37 ms a frame for both layers.
- **the ceiling:** vanilla draws both cascades in ~0.78 ms (0.38 near + 0.39 far, scouting table). reuse costs ~0.37 ms of restores plus the moving casters (skinned actors, wind-animated trees, anything hot). if static casters are ~85% of the cascade raster, the best case is ~0.78 - 0.37 - 0.12 = **~0.3 ms a frame**, minus the builds (~0.2 ms extra each, every 10-40 frames at 0.5 texel). with the full-layer restore the cache can never save more than ~0.4 ms.
- **to lift the ceiling:** `BLESSED_CASCADE_CACHE_RESTORE=draw` restores with a fullscreen depth write (compare ALWAYS) instead of a transfer copy, so the layer never leaves the depth-attachment layout; A/B it against the copy. the bigger lever is a dirty-rect restore: copy back only the rectangles last frame's moving casters covered (skinned bounds from the b10 bones, other movers need a per-mesh bound).
- new stats per layer: `restores`, `restores_skipped`, `hot_reheats`.

## the partial restore (BLESSED_CASCADE_CACHE_RESTORE=partial, 2026-09-24)

- **idea:** after a build or a reuse, a layer holds exactly the cache plus what the cascade's other draws (moving, hot, new statics, skinned) wrote. if every one of those draws has a known footprint, the next frame drops the engine's clear of that layer and copies back from the cache only the tiles those footprints touched. this frame's own draws do not need restoring under them: they are depth-tested on top of a layer that already equals the cache there.
- **tiles:** 64x64 per layer (64 texels each at 4096). footprints mark tiles; tile rows become runs, runs with the same columns merge down into rectangles. full restore when the tiles cover more than `_PARTIAL_MAX` (0.4) of the layer, when there are more than `_PARTIAL_RECTS` (128) rectangles, when any layer draw had no bound, after a verify or pass frame, or after any depth write into the layer outside its cascade.
- **bounds** (`src/d3d11/blessed_cascade_bounds.cpp`, `src/dxvk/shaders/blessed_cascade_bounds.comp`): a mesh (index buffer range + streams, keyed by buffer cookies) is measured once on the gpu: object-space aabb of the positions it indexes, its index range, and the bones with a non-zero weight. results are read back 8 presents later; until then the mesh is unknown (full restore).
  - skinned (vs reads b10): union of the aabb under each weighted bone (Arvo), then minus CameraPosAdjust. a skinned vertex is a convex blend of its bones' transforms, so it stays inside.
  - PerGeometry (b2): the aabb under World (camera-relative already), grown first by 1.1 * |TreeParams.z| * sqrt(3) + 0.01 when the vs declares c7 (the most Utility.hlsl's TREE_ANIM moves a vertex).
  - dynamic position streams (actor heads, slot 1): aabb read on the cpu from the mapped vertices over the measured index range, once per mesh per frame.
  - the 8 box corners go through the cascade's view-proj (orthographic, w = 1 checked), plus 2 texels of margin.
- **the dropped clear:** `OnClearDsv` returns true (the context skips its clear) only when the layer's dirty set is valid; the first draw decides the mode, and any cascade that does not restore partially gets the clear emitted right then (or at its end if it has no draws).
- **proof:** with `BLESSED_CASCADE_CACHE_VERIFY=N`, every N-th partial restore compares the whole layer with the cache right after the copies (verify entries `"kind":"restore"`): `differ` must be 0, independent of the projection tolerance. the cache verify (`"kind":"cache"`) is unchanged.
- **cost model:**
  - full restore (c28 data): ~0.18 ms per 4096x4096 d16 layer, ~0.37 ms a frame for both.
  - partial: ~0.18 ms x the covered share + a few us per rectangle. texel density from the whiterun dump's projections: near ~2.1 texels per unit (covers ~2,000 units), far ~0.24 (covers ~17,000). an actor's sun footprint (~50 x 130 units) is ~100 x 270 texels near (~12-20 tiles with margins), ~12 x 31 far (1-2 tiles). a handful of actors plus the wind trees is ~1-3% of a layer: **~0.005-0.01 ms per layer, ~0.01-0.03 ms a frame** instead of 0.37.
  - new ceiling: vanilla cascades ~0.78 ms minus moving casters' raster (~0.1-0.15) minus restores (~0.02) minus builds (vanilla + one composite, every 10-40 frames at 0.5 texel, ~0.02-0.04 ms averaged): **~0.55-0.6 ms a frame** at best, up from ~0.3.
  - cpu (replay thread): one bound per layer draw (hash + lookup + ~40 bone transforms for a skinned draw, ~1 us; ~0.2 us otherwise) plus the head scans (~5 us per head per frame): **~0.2-0.4 ms a frame**. the cs thread saves the ~1,800 skipped draws as before.
  - one-off: each new mesh costs one row in a single per-frame dispatch (up to 512 meshes / 4M indices a frame), 64 bytes of host-visible results (16,384 slots, then new meshes stay unknown: full restore).
- **risks:** a vs that moves vertices beyond World/bones (LOD landscape's `AdjustLodLandscapeVertexPositionMS` if such a draw is ever not cached; any vertex displacement other than TREE_ANIM) under-bounds its footprint: the restore check shows it as `differ` > 0. results slots are never recycled. heads scan on the cpu every frame.

## c37: stale bytes in unread registers (2026-09-24)

- **symptom** (`c37-casc-verify-r1`, `c37-casc-verify-partial-r1`): `cached_draws` 0, `hot_meshes` 174/706, and `hot_reheats` 18,990 (L0) / 75,245 (L1) per 120 frames: every hot mesh's draws change on every cascade, so none ever cools.
- **cause:** the shadow-map vs `138834da` declares `cb2[5]` (`cb_vec4s` in `cascade_cache_proj.jsonl`), because World sits at c1-c4. that range also covers c0 (ShadowFadeParam), which only the shadow-mask permutations read (Utility.hlsl line 223). the engine does not write registers a shader does not read, so c0 holds whatever the cb ring's memory held before: different bytes every frame. the moved lines where only c0 differs (same placement: c1-c4 equal) show it: 62 of 257 in partial-r1, 93 of 286 in r1, e.g. `[[0,0,0.841469],[1,0,0.540305],[2,0,-2.07219e-08],[3,0,-6997.79]]` (another object's World row), `[[0,0,6.70312],[2,0,-0.000130208]]`, `[[0,0,1],[1,0,1],[2,0,1],[3,0,1]]`. (the other moved lines compare two placements of one mesh and say nothing.)
- **fix:** shader creation records every constant register the shader actually reads (source operands `cbN[r]` with an immediate index; a relative index, or r above 63, counts the slot as read whole). a key hashes only the read registers. the declared-range clamp stays as the outer bound. the partial restore's wind margin now reads TreeParams only when the vs reads c7.
- **new log:** `reheat` lines in `cascade_cache_proj.jsonl` (first 48 per layer): the b2 floats of a hot mesh's first draw in the last two cascades and its world positions, to name anything else that keeps a mesh hot. `cb_vec4s` entries now carry the read-register mask (`"0x1e"` = c1-c4).
